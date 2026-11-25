"""Official MEGC spotting, recognition, calibration, and statistical testing modules."""

from vascumamba_spot.evaluation.metrics import (
    SpottedInterval,
    compute_ece,
    evaluate_recognition,
    evaluate_spotting,
    temporal_iou,
)
from vascumamba_spot.evaluation.postprocess import decode_and_nms_predictions
from vascumamba_spot.evaluation.statistics import (
    bootstrap_ci,
    holm_bonferroni_adjust,
    paired_wilcoxon_test,
)

__all__ = [
    "SpottedInterval",
    "bootstrap_ci",
    "compute_ece",
    "decode_and_nms_predictions",
    "evaluate_recognition",
    "evaluate_spotting",
    "holm_bonferroni_adjust",
    "paired_wilcoxon_test",
    "temporal_iou",
]
