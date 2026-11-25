"""Statistical significance tests (paired Wilcoxon, Holm-Bonferroni, bootstrap CIs)."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon_test(
    ours_scores: np.ndarray | list[float],
    baseline_scores: np.ndarray | list[float],
) -> dict[str, float]:
    """Run two-sided subject-level paired Wilcoxon signed-rank test."""
    a = np.asarray(ours_scores, dtype=np.float64)
    b = np.asarray(baseline_scores, dtype=np.float64)
    diff = a - b
    if np.allclose(diff, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "mean_diff": 0.0}
    res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "mean_diff": float(np.mean(diff)),
    }


def holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    """Adjust p-values for multiple baseline comparisons via Holm-Bonferroni step-down."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def bootstrap_ci(
    scores: np.ndarray | list[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lower_ci, upper_ci) via percentile bootstrap."""
    arr = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = arr.shape[0]
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = np.mean(arr[idx])
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return float(np.mean(arr)), lo, hi
