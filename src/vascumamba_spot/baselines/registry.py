"""Baseline models covering classical strain spotting, 3D-CNN, Transformer, and Mamba families."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vascumamba_spot.evaluation.metrics import SpottedInterval
from vascumamba_spot.models.vascumamba_spot import VascuMambaConfig, VascuMambaSpot


class OpticalStrainPeakSpotter:
    """Classical non-learning optical strain peak spotter (He et al. style baseline)."""

    def __init__(self, z_thresh: float = 2.5, min_dur_sec: float = 0.04, max_dur_sec: float = 0.50) -> None:
        self.z_thresh = z_thresh
        self.min_dur_sec = min_dur_sec
        self.max_dur_sec = max_dur_sec

    def predict(self, x_kin: np.ndarray, fps: float) -> list[SpottedInterval]:
        """Spot intervals where global AU Frobenius strain exceeds mean + z_thresh * std."""
        # Channels 3, 9, 15, ... in 144-dim X_kin correspond to ROI mean Frobenius strain
        fro_cols = np.arange(3, x_kin.shape[1], 12)
        energy = x_kin[:, fro_cols].mean(axis=1)
        mu, sd = float(np.mean(energy)), max(float(np.std(energy)), 1e-6)
        z = (energy - mu) / sd

        intervals: list[SpottedInterval] = []
        in_peak = False
        start = 0
        for t, val in enumerate(z):
            if val >= self.z_thresh and not in_peak:
                in_peak = True
                start = t
            elif (val < self.z_thresh or t == len(z) - 1) and in_peak:
                end = t if val < self.z_thresh else t
                dur = (end - start) / max(fps, 1.0)
                if self.min_dur_sec <= dur <= 4.0:
                    apex = start + int(np.argmax(z[start : end + 1]))
                    etype = "ME" if dur <= self.max_dur_sec else "MaE"
                    conf = float(min(1.0, z[apex] / (self.z_thresh * 2.0)))
                    intervals.append(
                        SpottedInterval(
                            onset=start,
                            apex=apex,
                            offset=end,
                            expr_type=etype,
                            emotion_id=0,
                            confidence=conf,
                        )
                    )
                in_peak = False
        return intervals


class STSTNetLite(nn.Module):
    """Shallow Triple-Stream CNN adapted for ROI kinematic sequences (clipped recognition)."""

    def __init__(self, d_kin_in: int = 144, num_classes: int = 5) -> None:
        super().__init__()
        self.stream_a = nn.Conv1d(d_kin_in, 32, kernel_size=3, padding=1)
        self.stream_b = nn.Conv1d(d_kin_in, 32, kernel_size=5, padding=2)
        self.stream_c = nn.Conv1d(d_kin_in, 32, kernel_size=7, padding=3)
        self.head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_kin: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        xt = x_kin.transpose(1, 2)
        f = torch.cat(
            [F.relu(self.stream_a(xt)), F.relu(self.stream_b(xt)), F.relu(self.stream_c(xt))],
            dim=1,
        )
        pooled = f.mean(dim=-1)
        logits = self.head(pooled)
        return {"clip_cls_logits": logits}


class MultiScaleTemporalSpotter(nn.Module):
    """Multi-scale 1D-CNN + Dilated Temporal Conv spotting baseline (LSSNet / MTSN family)."""

    def __init__(self, d_kin_in: int = 144, d_model: int = 128, num_classes: int = 5) -> None:
        super().__init__()
        self.stem = nn.Conv1d(d_kin_in, d_model, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=d, dilation=d)
                for d in (1, 2, 4, 8)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.apex_head = nn.Linear(d_model, 1)
        self.boundary_head = nn.Linear(d_model, 2)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x_kin: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        h = F.gelu(self.stem(x_kin.transpose(1, 2)))
        for blk in self.blocks:
            h = h + F.gelu(blk(h))
        ht = self.norm(h.transpose(1, 2))
        apex_logits = self.apex_head(ht).squeeze(-1)
        return {
            "apex_logits": apex_logits,
            "apex_prob": torch.sigmoid(apex_logits),
            "boundaries_sec": F.softplus(self.boundary_head(ht)),
            "cls_logits": self.cls_head(ht),
            "clip_cls_logits": self.cls_head(ht.mean(dim=1)),
            "gate_mean": torch.ones_like(apex_logits),
        }


class OFFApexNetLite(nn.Module):
    """Dual-stream (velocity + strain) apex-guided CNN baseline for clipped recognition."""

    def __init__(self, d_kin_in: int = 144, num_classes: int = 5) -> None:
        super().__init__()
        half = d_kin_in // 2
        self.branch_u = nn.Sequential(nn.Conv1d(half, 32, 3, padding=1), nn.ReLU())
        self.branch_v = nn.Sequential(nn.Conv1d(d_kin_in - half, 32, 3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x_kin: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        half = x_kin.shape[-1] // 2
        fu = self.branch_u(x_kin[..., :half].transpose(1, 2)).mean(dim=-1)
        fv = self.branch_v(x_kin[..., half:].transpose(1, 2)).mean(dim=-1)
        return {"clip_cls_logits": self.fc(torch.cat([fu, fv], dim=-1))}


class FeatRefNetLite(nn.Module):
    """Attention-based Feature Refinement Network baseline."""

    def __init__(self, d_kin_in: int = 144, num_classes: int = 5) -> None:
        super().__init__()
        self.proj = nn.Conv1d(d_kin_in, 96, kernel_size=3, padding=1)
        self.se = nn.Sequential(nn.Linear(96, 24), nn.ReLU(), nn.Linear(24, 96), nn.Sigmoid())
        self.cls = nn.Linear(96, num_classes)

    def forward(self, x_kin: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        h = F.gelu(self.proj(x_kin.transpose(1, 2))).mean(dim=-1)
        refined = h * self.se(h)
        return {"clip_cls_logits": self.cls(refined)}


class DiffMESpotLite(nn.Module):
    """Iterative denoising temporal refinement spotting baseline (DiffME family)."""

    def __init__(self, d_kin_in: int = 144, d_model: int = 128, num_classes: int = 5, steps: int = 2) -> None:
        super().__init__()
        self.steps = steps
        self.stem = nn.Conv1d(d_kin_in, d_model, kernel_size=3, padding=1)
        self.refine = nn.Conv1d(d_model + 3, d_model, kernel_size=3, padding=1)
        self.apex_head = nn.Linear(d_model, 1)
        self.boundary_head = nn.Linear(d_model, 2)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x_kin: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        h = F.gelu(self.stem(x_kin.transpose(1, 2))).transpose(1, 2)
        apex_logits = self.apex_head(h).squeeze(-1)
        bounds = F.softplus(self.boundary_head(h))
        for _ in range(self.steps):
            cond = torch.cat([h, torch.sigmoid(apex_logits).unsqueeze(-1), bounds], dim=-1)
            h = h + F.gelu(self.refine(cond.transpose(1, 2))).transpose(1, 2)
            apex_logits = self.apex_head(h).squeeze(-1)
            bounds = F.softplus(self.boundary_head(h))
        return {
            "apex_logits": apex_logits,
            "apex_prob": torch.sigmoid(apex_logits),
            "boundaries_sec": bounds,
            "cls_logits": self.cls_head(h),
            "clip_cls_logits": self.cls_head(h.mean(dim=1)),
            "gate_mean": torch.ones_like(apex_logits),
        }


class HandcraftedDescriptorBaseline(nn.Module):
    """Handcrafted optical strain descriptor + linear classifier (MDMO / Bi-WOOF family)."""

    def __init__(self, d_kin_in: int = 144, num_classes: int = 5, bi_weighted: bool = False) -> None:
        super().__init__()
        self.bi_weighted = bi_weighted
        self.head = nn.Linear(d_kin_in, num_classes)

    def forward(self, x_kin: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        if self.bi_weighted:
            # Bi-WOOF style weighting emphasizing peak-energy frames
            energy = x_kin.abs().mean(dim=-1, keepdim=True)
            w = torch.softmax(energy, dim=1)
            pooled = torch.sum(x_kin * w, dim=1)
        else:
            pooled = x_kin.mean(dim=1)
        return {"clip_cls_logits": self.head(pooled)}


class SLViTLite(nn.Module):
    """Spatio-Temporal Local Vision Transformer / VideoMAE-ME adapter baseline."""

    def __init__(self, d_kin_in: int = 144, d_model: int = 128, num_classes: int = 5, patch_len: int = 4) -> None:
        super().__init__()
        self.patch_proj = nn.Conv1d(d_kin_in, d_model, kernel_size=patch_len, stride=patch_len)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 4, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=3)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x_kin: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        xt = x_kin.transpose(1, 2)
        if xt.shape[-1] < 4:
            xt = F.pad(xt, (0, 4 - xt.shape[-1]))
        tokens = self.patch_proj(xt).transpose(1, 2)
        h = self.encoder(tokens)
        return {"clip_cls_logits": self.cls(h.mean(dim=1))}


def list_baselines() -> list[str]:
    """Return all 14 registered baseline identifiers."""
    return [
        "optical_strain_peak",
        "mdmo_linear",
        "biwoof_linear",
        "kinematic_only_mamba",
        "vascular_only_mamba",
        "transformer_spot",
        "bigru_spot",
        "lssnet_tcn",
        "mtsn_tcn",
        "metst_spot",
        "diffme_spot_lite",
        "ststnet_lite",
        "off_apexnet_lite",
        "featref_lite",
        "slvit_lite",
        "videomae_me_lite",
        "micromamba_kin",
        "ammsm_kin",
    ]


def build_baseline(name: str, d_kin_in: int = 144, d_phys_in: int = 36, num_classes: int = 5) -> Any:
    """Instantiate a baseline model by name under unified input/output contracts."""
    if name == "optical_strain_peak":
        return OpticalStrainPeakSpotter()
    if name == "mdmo_linear":
        return HandcraftedDescriptorBaseline(d_kin_in=d_kin_in, num_classes=num_classes, bi_weighted=False)
    if name == "biwoof_linear":
        return HandcraftedDescriptorBaseline(d_kin_in=d_kin_in, num_classes=num_classes, bi_weighted=True)
    if name == "kinematic_only_mamba":
        return VascuMambaSpot(
            VascuMambaConfig(
                d_kin_in=d_kin_in,
                d_phys_in=d_phys_in,
                num_classes=num_classes,
                kinematic_only=True,
                use_autonomic_gate=False,
            )
        )
    if name == "vascular_only_mamba":
        return VascuMambaSpot(
            VascuMambaConfig(
                d_kin_in=d_kin_in,
                d_phys_in=d_phys_in,
                num_classes=num_classes,
                vascular_only=True,
                use_autonomic_gate=False,
            )
        )
    if name in {"transformer_spot", "metst_spot"}:
        return VascuMambaSpot(
            VascuMambaConfig(
                d_kin_in=d_kin_in,
                d_phys_in=d_phys_in,
                num_classes=num_classes,
                backbone_type="transformer",
            )
        )
    if name == "bigru_spot":
        return VascuMambaSpot(
            VascuMambaConfig(
                d_kin_in=d_kin_in,
                d_phys_in=d_phys_in,
                num_classes=num_classes,
                backbone_type="bigru",
            )
        )
    if name in {"micromamba_kin", "ammsm_kin"}:
        return VascuMambaSpot(
            VascuMambaConfig(
                d_kin_in=d_kin_in,
                d_phys_in=d_phys_in,
                num_classes=num_classes,
                kinematic_only=True,
                use_mnc_filter=False,
                use_autonomic_gate=False,
            )
        )
    if name in {"lssnet_tcn", "mtsn_tcn"}:
        return MultiScaleTemporalSpotter(d_kin_in=d_kin_in, num_classes=num_classes)
    if name == "ststnet_lite":
        return STSTNetLite(d_kin_in=d_kin_in, num_classes=num_classes)
    if name == "off_apexnet_lite":
        return OFFApexNetLite(d_kin_in=d_kin_in, num_classes=num_classes)
    if name == "featref_lite":
        return FeatRefNetLite(d_kin_in=d_kin_in, num_classes=num_classes)
    if name in {"slvit_lite", "videomae_me_lite"}:
        return SLViTLite(d_kin_in=d_kin_in, num_classes=num_classes)
    if name == "diffme_spot_lite":
        return DiffMESpotLite(d_kin_in=d_kin_in, num_classes=num_classes)
    raise ValueError(f"Unknown baseline: {name}. Available: {list_baselines()}")
