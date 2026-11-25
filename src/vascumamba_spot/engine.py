"""Shared training, evaluation, and SSL pretraining engine for VascuMamba-Spot."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from vascumamba_spot.data.augmentations import apply_modality_dropout
from vascumamba_spot.data.cache_dataset import CachedFeatureDataset, collate_variable_sequences
from vascumamba_spot.data.manifests import LOSOFold
from vascumamba_spot.evaluation.metrics import (
    SpottedInterval,
    compute_ece,
    evaluate_recognition,
    evaluate_spotting,
)
from vascumamba_spot.evaluation.postprocess import decode_and_nms_predictions
from vascumamba_spot.losses.class_balanced import compute_effective_class_weights
from vascumamba_spot.losses.composite import VascuMambaCompositeLoss
from vascumamba_spot.losses.distill_ssl import NeuroVascularSSLLoss


def set_deterministic_seed(seed: int) -> None:
    """Set deterministic RNG seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_fold_class_weights(
    fold: LOSOFold,
    emotion_map: dict[str, int],
    beta: float = 0.99,
) -> torch.Tensor:
    """Compute effective-number weights strictly from training fold records (no leakage)."""
    counts = [0] * len(emotion_map)
    for rec in fold.train_records:
        for ann in rec.annotations:
            idx = emotion_map.get(ann.emotion_raw.strip().lower(), -1)
            if idx >= 0:
                counts[idx] += 1
    return compute_effective_class_weights(counts, beta=beta)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: VascuMambaCompositeLoss,
    device: torch.device,
    p_drop_kin: float = 0.15,
    p_drop_phys: float = 0.25,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    """Execute one supervised multi-task training epoch."""
    model.train()
    totals: dict[str, float] = {}
    steps = 0

    for batch in loader:
        x_kin = batch["x_kin"].to(device)
        x_phys = batch["x_phys"].to(device)
        has_rgb = batch["has_rgb"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        x_kin, x_phys, has_rgb = apply_modality_dropout(
            x_kin, x_phys, has_rgb, p_drop_kin=p_drop_kin, p_drop_phys=p_drop_phys, training=True
        )
        dt_frame = torch.tensor([1.0 / max(f, 1.0) for f in batch["fps"]], device=device)

        outputs = model(x_kin, x_phys, has_rgb=has_rgb, dt_frame=dt_frame, valid_mask=valid_mask)
        gpu_batch = {
            "valid_mask": valid_mask,
            "pos_mask": batch["pos_mask"].to(device),
            "apex_target": batch["apex_target"].to(device),
            "boundary_target": batch["boundary_target"].to(device),
            "cls_target": batch["cls_target"].to(device),
        }
        if "biopac_arousal" in batch:
            gpu_batch["biopac_arousal"] = batch["biopac_arousal"].to(device)

        losses = criterion(outputs, gpu_batch)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.item())
        steps += 1

    return {k: v / max(1, steps) for k, v in totals.items()}


@torch.no_grad()
def evaluate_fold(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    emotion_map: dict[str, int],
    postprocess_cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Evaluate spotting, clipped recognition, STR-F1, and ECE on a held-out fold."""
    model.eval()
    pp_cfg = postprocess_cfg or {}
    preds_by_vid: dict[str, list[SpottedInterval]] = {}
    gts_by_vid: dict[str, list[SpottedInterval]] = {}
    y_true_clips: list[int] = []
    y_pred_clips: list[int] = []
    probs_clips: list[np.ndarray] = []

    for batch in loader:
        x_kin = batch["x_kin"].to(device)
        x_phys = batch["x_phys"].to(device)
        has_rgb = batch["has_rgb"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        dt_frame = torch.tensor([1.0 / max(f, 1.0) for f in batch["fps"]], device=device)

        outputs = model(x_kin, x_phys, has_rgb=has_rgb, dt_frame=dt_frame, valid_mask=valid_mask)
        apex_prob = outputs["apex_prob"].cpu().numpy()
        bounds = outputs["boundaries_sec"].cpu().numpy()
        cls_logits = outputs["cls_logits"].cpu().numpy()
        gate_mean = outputs["gate_mean"].cpu().numpy()
        clip_probs = torch.softmax(outputs["clip_cls_logits"], dim=-1).cpu().numpy()

        for i, vid in enumerate(batch["video_ids"]):
            t_len = int(valid_mask[i].sum().item())
            fps = float(batch["fps"][i])
            preds_by_vid[vid] = decode_and_nms_predictions(
                apex_prob=apex_prob[i, :t_len],
                boundaries_sec=bounds[i, :t_len],
                cls_logits=cls_logits[i, :t_len],
                gate_mean=gate_mean[i, :t_len],
                fps=fps,
                **pp_cfg,
            )
            gt_list: list[SpottedInterval] = []
            for on, ap, off, lbl in batch["annotations"][i]:
                dur = (off - on) / max(fps, 1.0)
                etype = "ME" if dur <= 0.50 else "MaE"
                gt_list.append(
                    SpottedInterval(
                        onset=on,
                        apex=ap,
                        offset=off,
                        expr_type=etype,
                        emotion_id=lbl,
                    )
                )
            gts_by_vid[vid] = gt_list

            if len(batch["annotations"][i]) == 1 and batch["annotations"][i][0][3] >= 0:
                y_true_clips.append(batch["annotations"][i][0][3])
                y_pred_clips.append(int(np.argmax(clip_probs[i])))
                probs_clips.append(clip_probs[i])

    spot_metrics = evaluate_spotting(preds_by_vid, gts_by_vid)
    if y_true_clips:
        rec_metrics = evaluate_recognition(np.array(y_true_clips), np.array(y_pred_clips))
        ece = compute_ece(np.stack(probs_clips, axis=0), np.array(y_true_clips))
    else:
        rec_metrics = {"uf1": 0.0, "uar": 0.0, "accuracy": 0.0}
        ece = 0.0
    return {**spot_metrics, **rec_metrics, "ece": ece}


def run_loso_fold(
    fold: LOSOFold,
    cache_path: str | Path,
    model_factory,
    emotion_scheme: str = "5cls",
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 3e-4,
    lambda_nce: float = 0.5,
    lambda_distill: float = 0.5,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Train and evaluate a single subject-disjoint LOSO fold end-to-end."""
    fold.assert_disjoint()
    set_deterministic_seed(seed)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CachedFeatureDataset(
        fold.train_records, cache_path, emotion_scheme=emotion_scheme, training=True
    )
    test_ds = CachedFeatureDataset(
        fold.test_records, cache_path, emotion_scheme=emotion_scheme, training=False
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_variable_sequences
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_variable_sequences
    )

    model = model_factory().to(dev)
    weights = compute_fold_class_weights(fold, train_ds.emotion_map)
    criterion = VascuMambaCompositeLoss(
        lambda_nce=lambda_nce, lambda_distill=lambda_distill, class_weights=weights
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, dev)

    return evaluate_fold(model, test_loader, dev, train_ds.emotion_map)


def pretrain_ssl_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ssl_loss_fn: NeuroVascularSSLLoss,
    device: torch.device,
    mask_ratio: float = 0.30,
    span_len: int = 8,
) -> dict[str, float]:
    """Run one self-supervised masked neuro-vascular reconstruction + contrastive epoch."""
    model.train()
    totals: dict[str, float] = {}
    steps = 0
    for batch in loader:
        x_kin = batch["x_kin"].to(device)
        x_phys = batch["x_phys"].to(device)
        has_rgb = batch["has_rgb"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        bsz, seq_len, _ = x_kin.shape
        mask_kin = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
        mask_phys = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
        n_spans = max(1, int((seq_len * mask_ratio) / max(1, span_len)))
        for b in range(bsz):
            for _ in range(n_spans):
                s1 = int(torch.randint(0, max(1, seq_len - span_len), (1,)).item())
                s2 = int(torch.randint(0, max(1, seq_len - span_len), (1,)).item())
                mask_kin[b, s1 : s1 + span_len] = True
                mask_phys[b, s2 : s2 + span_len] = True

        mask_kin = mask_kin & valid_mask
        mask_phys = mask_phys & valid_mask
        x_kin_masked = x_kin.masked_fill(mask_kin.unsqueeze(-1), 0.0)
        x_phys_masked = x_phys.masked_fill(mask_phys.unsqueeze(-1), 0.0)

        out = model(x_kin_masked, x_phys_masked, has_rgb=has_rgb, valid_mask=valid_mask)
        losses = ssl_loss_fn(
            recon_kin=out["recon_kin"],
            target_kin=x_kin,
            mask_kin=mask_kin,
            recon_phys=out["recon_phys"],
            target_phys=x_phys,
            mask_phys=mask_phys,
            z_kin=out["z_kin_nce"],
            z_phys=out["z_phys_nce"],
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.item())
        steps += 1

    return {k: v / max(1, steps) for k, v in totals.items()}
