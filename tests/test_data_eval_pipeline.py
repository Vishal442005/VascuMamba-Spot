"""End-to-end tests for LOSO anti-leakage, HDF5 caching, metrics, statistics, and training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vascumamba_spot.data.augmentations import apply_modality_dropout
from vascumamba_spot.data.cache_dataset import write_feature_cache
from vascumamba_spot.data.manifests import (
    ExpressionAnnotation,
    LOSOFold,
    VideoRecord,
    generate_loso_folds,
    load_manifest,
    save_manifest,
)
from vascumamba_spot.engine import run_loso_fold
from vascumamba_spot.evaluation import (
    SpottedInterval,
    compute_ece,
    decode_and_nms_predictions,
    evaluate_recognition,
    evaluate_spotting,
    holm_bonferroni_adjust,
    paired_wilcoxon_test,
)
from vascumamba_spot.models import VascuMambaConfig, VascuMambaSpot


def test_loso_disjoint_and_anti_leakage() -> None:
    """Verify subject-disjoint LOSO generation and leakage error detection."""
    recs = [
        VideoRecord("v1", "ds", "sub1", "rgb", 50.0, 100, "a.mp4", ()),
        VideoRecord("v2", "ds", "sub2", "rgb", 50.0, 100, "b.mp4", ()),
    ]
    folds = generate_loso_folds(recs)
    assert len(folds) == 2

    leaky_fold = LOSOFold(0, "sub1", tuple(recs), (recs[0],))
    with pytest.raises(AssertionError):
        leaky_fold.assert_disjoint()


def test_modality_dropout_never_drops_both() -> None:
    """Verify cross-modal dropout never zeroes both kinematic and physiological inputs."""
    xk = torch.ones(16, 10, 4)
    xp = torch.ones(16, 10, 4)
    has_rgb = torch.ones(16)
    xk_a, xp_a, _ = apply_modality_dropout(xk, xp, has_rgb, p_drop_kin=1.0, p_drop_phys=1.0, training=True)
    both_zero = (xk_a.abs().sum(dim=(1, 2)) == 0) & (xp_a.abs().sum(dim=(1, 2)) == 0)
    assert not torch.any(both_zero)


def test_spotting_recognition_ece_and_stats() -> None:
    """Verify MEGC spotting, STR-F1, UF1/UAR, ECE, and Wilcoxon/Holm-Bonferroni."""
    gt = {
        "vid1": [
            SpottedInterval(10, 15, 22, "ME", emotion_id=1),
            SpottedInterval(60, 80, 110, "MaE", emotion_id=0),
        ]
    }
    pred = {
        "vid1": [
            SpottedInterval(11, 15, 21, "ME", emotion_id=1, confidence=0.9),
            SpottedInterval(62, 80, 108, "MaE", emotion_id=2, confidence=0.8),
        ]
    }
    res = evaluate_spotting(pred, gt, iou_thresh=0.5)
    assert res["f1_me"] == pytest.approx(1.0)
    assert res["str_f1_me"] == pytest.approx(1.0)
    assert res["f1_mae"] == pytest.approx(1.0)

    rec = evaluate_recognition(np.array([0, 1, 2, 0]), np.array([0, 1, 2, 1]))
    assert 0.0 < rec["uf1"] < 1.0
    ece = compute_ece(np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([0, 1]))
    assert 0.0 <= ece <= 0.25

    w = paired_wilcoxon_test([0.8, 0.7, 0.9, 0.85, 0.75], [0.5, 0.4, 0.6, 0.55, 0.50])
    assert w["p_value"] < 0.10
    adj = holm_bonferroni_adjust([0.01, 0.04, 0.20])
    assert adj == pytest.approx([0.03, 0.08, 0.20])


def test_end_to_end_synthetic_loso_and_nms(tmp_path: Path) -> None:
    """Run full manifest -> HDF5 cache -> LOSO train/eval smoke test on synthetic subjects."""
    rng = np.random.default_rng(123)
    cache_path = tmp_path / "features.h5"
    records: list[VideoRecord] = []
    fps = 50.0
    seq_len = 80

    for s_idx in range(3):
        sub = f"sub{s_idx}"
        for v_idx in range(2):
            vid = f"{sub}_v{v_idx}"
            xk = rng.normal(0.0, 0.1, size=(seq_len, 144)).astype(np.float32)
            xp = rng.normal(0.0, 0.1, size=(seq_len, 36)).astype(np.float32)
            xp[:, -1] = 1.0
            # Inject correlated kinematic + vascular signal inside ME window [30, 42]
            xk[30:43, :] += 2.0
            xp[28:45, :28] += 1.5
            write_feature_cache(cache_path, vid, xk, xp, fps=fps, has_rgb=True)

            ann = ExpressionAnnotation(
                onset=30, apex=36, offset=42, expr_type="ME", emotion_raw="happiness", emotion_3cls="positive"
            )
            records.append(VideoRecord(vid, "synth", sub, "rgb", fps, seq_len, "none.mp4", (ann,)))

    manifest_path = tmp_path / "manifest.jsonl"
    save_manifest(records, manifest_path)
    loaded = load_manifest(manifest_path)
    folds = generate_loso_folds(loaded)
    assert len(folds) == 3

    metrics = run_loso_fold(
        fold=folds[0],
        cache_path=cache_path,
        model_factory=lambda: VascuMambaSpot(VascuMambaConfig(d_model=32, num_layers=2, num_classes=5)),
        emotion_scheme="5cls",
        epochs=3,
        batch_size=2,
        seed=42,
        device=torch.device("cpu"),
    )
    assert "f1_me" in metrics and "uf1" in metrics and "ece" in metrics

    # Verify decoder & NMS cleanly parses synthetic peaks
    apex = np.zeros(seq_len, dtype=np.float32)
    apex[36] = 0.9
    bounds = np.full((seq_len, 2), 0.12, dtype=np.float32)
    cls_logits = np.zeros((seq_len, 5), dtype=np.float32)
    gate = np.ones(seq_len, dtype=np.float32)
    decoded = decode_and_nms_predictions(apex, bounds, cls_logits, gate, fps=fps)
    assert len(decoded) == 1 and decoded[0].expr_type == "ME"
