"""Official MEGC spotting (IoU >= 0.5), recognition (UF1, UAR), STR-F1, and ECE metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


@dataclass(frozen=True)
class SpottedInterval:
    """Predicted or ground-truth temporal expression interval."""

    onset: int
    apex: int
    offset: int
    expr_type: str  # "ME" | "MaE"
    emotion_id: int
    confidence: float = 1.0


def temporal_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Compute 1D temporal Intersection-over-Union between closed/half-open intervals."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = max(1, max(a[1], b[1]) - min(a[0], b[0]) + 1)
    return float(inter) / float(union)


def _match_spotting_subset(
    preds_by_vid: dict[str, list[SpottedInterval]],
    gts_by_vid: dict[str, list[SpottedInterval]],
    iou_thresh: float = 0.5,
    require_emotion_match: bool = False,
) -> dict[str, float]:
    tp = 0
    fp = 0
    fn = 0

    all_vids = set(preds_by_vid.keys()) | set(gts_by_vid.keys())
    for vid in all_vids:
        preds = sorted(preds_by_vid.get(vid, []), key=lambda p: p.confidence, reverse=True)
        gts = list(gts_by_vid.get(vid, []))
        matched_gt = set()

        for p in preds:
            best_iou = -1.0
            best_idx = -1
            for idx, g in enumerate(gts):
                if idx in matched_gt:
                    continue
                if require_emotion_match and p.emotion_id != g.emotion_id:
                    continue
                iou = temporal_iou((p.onset, p.offset), (g.onset, g.offset))
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0:
                matched_gt.add(best_idx)
                tp += 1
            else:
                fp += 1
        fn += len(gts) - len(matched_gt)

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2.0 * prec * rec) / max(1e-12, prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def evaluate_spotting(
    preds_by_vid: dict[str, list[SpottedInterval]],
    gts_by_vid: dict[str, list[SpottedInterval]],
    iou_thresh: float = 0.5,
) -> dict[str, float]:
    """Compute MEGC spotting metrics (F1_ME, F1_MaE, F1_all) and end-to-end STR-F1."""
    def filter_type(d: dict[str, list[SpottedInterval]], etype: str) -> dict[str, list[SpottedInterval]]:
        return {k: [x for x in v if x.expr_type == etype] for k, v in d.items()}

    me_res = _match_spotting_subset(filter_type(preds_by_vid, "ME"), filter_type(gts_by_vid, "ME"), iou_thresh)
    mae_res = _match_spotting_subset(filter_type(preds_by_vid, "MaE"), filter_type(gts_by_vid, "MaE"), iou_thresh)
    all_res = _match_spotting_subset(preds_by_vid, gts_by_vid, iou_thresh)
    str_me = _match_spotting_subset(
        filter_type(preds_by_vid, "ME"),
        filter_type(gts_by_vid, "ME"),
        iou_thresh,
        require_emotion_match=True,
    )

    return {
        "precision_me": me_res["precision"],
        "recall_me": me_res["recall"],
        "f1_me": me_res["f1"],
        "precision_mae": mae_res["precision"],
        "recall_mae": mae_res["recall"],
        "f1_mae": mae_res["f1"],
        "precision_all": all_res["precision"],
        "recall_all": all_res["recall"],
        "f1_all": all_res["f1"],
        "str_f1_me": str_me["f1"],
        "fp_me": me_res["fp"],
    }


def evaluate_recognition(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MEGC Unweighted F1 (UF1), Unweighted Average Recall (UAR), and Accuracy."""
    if y_true.size == 0:
        return {"uf1": 0.0, "uar": 0.0, "accuracy": 0.0}
    uf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    uar = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    return {"uf1": uf1, "uar": uar, "accuracy": acc}


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) over predicted confidence bins."""
    if probs.size == 0:
        return 0.0
    confidences = np.max(probs, axis=-1) if probs.ndim == 2 else probs
    preds = np.argmax(probs, axis=-1) if probs.ndim == 2 else (probs >= 0.5).astype(np.int64)
    correct = (preds == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(labels))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        cnt = int(np.sum(in_bin))
        if cnt > 0:
            acc_bin = float(np.mean(correct[in_bin]))
            conf_bin = float(np.mean(confidences[in_bin]))
            ece += (cnt / n) * abs(acc_bin - conf_bin)
    return float(ece)
