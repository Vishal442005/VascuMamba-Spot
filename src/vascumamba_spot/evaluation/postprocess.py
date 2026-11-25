"""Duration-aware Physio-Verified Soft-NMS decoding for untrimmed expression spotting."""

from __future__ import annotations

import numpy as np

from vascumamba_spot.evaluation.metrics import SpottedInterval, temporal_iou


def decode_and_nms_predictions(
    apex_prob: np.ndarray,
    boundaries_sec: np.ndarray,
    cls_logits: np.ndarray,
    gate_mean: np.ndarray,
    fps: float,
    apex_threshold: float = 0.35,
    me_min_sec: float = 0.04,
    me_max_sec: float = 0.50,
    mae_min_sec: float = 0.50,
    mae_max_sec: float = 4.00,
    soft_nms_iou_thresh: float = 0.40,
    soft_nms_sigma: float = 0.50,
    physio_weight_beta: float = 0.50,
    score_cutoff: float = 0.10,
) -> list[SpottedInterval]:
    """Decode frame-wise predictions into duration-partitioned ME/MaE intervals with Soft-NMS."""
    t_len = apex_prob.shape[0]
    candidates: list[SpottedInterval] = []

    for t in range(t_len):
        p_t = float(apex_prob[t])
        if p_t < apex_threshold:
            continue
        # Peak check (keep local maxima or plateau starts)
        left = float(apex_prob[t - 1]) if t > 0 else 0.0
        right = float(apex_prob[t + 1]) if t + 1 < t_len else 0.0
        if p_t < left or p_t < right:
            continue

        d_s_sec = float(boundaries_sec[t, 0])
        d_e_sec = float(boundaries_sec[t, 1])
        dur_sec = d_s_sec + d_e_sec
        if dur_sec < me_min_sec or dur_sec > mae_max_sec:
            continue

        expr_type = "ME" if dur_sec <= me_max_sec else "MaE"
        if expr_type == "MaE" and dur_sec <= mae_min_sec:
            expr_type = "ME"

        onset = max(0, int(round(t - d_s_sec * fps)))
        offset = min(t_len - 1, int(round(t + d_e_sec * fps)))
        if offset <= onset:
            offset = min(t_len - 1, onset + 1)

        emo_id = int(np.argmax(cls_logits[t]))
        g_t = float(np.clip(gate_mean[t], 0.0, 1.0))
        conf = p_t * (physio_weight_beta + (1.0 - physio_weight_beta) * g_t)

        candidates.append(
            SpottedInterval(
                onset=onset,
                apex=t,
                offset=offset,
                expr_type=expr_type,
                emotion_id=emo_id,
                confidence=conf,
            )
        )

    # Class-independent within-type Gaussian Soft-NMS
    kept: list[SpottedInterval] = []
    for target_type in ("ME", "MaE"):
        pool = [c for c in candidates if c.expr_type == target_type]
        pool.sort(key=lambda c: c.confidence, reverse=True)
        while pool:
            best = pool.pop(0)
            if best.confidence < score_cutoff:
                break
            kept.append(best)
            updated_pool: list[SpottedInterval] = []
            for cand in pool:
                iou = temporal_iou((best.onset, best.offset), (cand.onset, cand.offset))
                decay = float(np.exp(-(iou**2) / max(soft_nms_sigma, 1e-6))) if iou > soft_nms_iou_thresh else 1.0
                new_conf = cand.confidence * decay
                if new_conf >= score_cutoff:
                    updated_pool.append(
                        SpottedInterval(
                            onset=cand.onset,
                            apex=cand.apex,
                            offset=cand.offset,
                            expr_type=cand.expr_type,
                            emotion_id=cand.emotion_id,
                            confidence=new_conf,
                        )
                    )
            pool = sorted(updated_pool, key=lambda c: c.confidence, reverse=True)

    return sorted(kept, key=lambda c: c.onset)
