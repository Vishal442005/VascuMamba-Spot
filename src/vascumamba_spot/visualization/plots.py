"""Matplotlib vector PDF figure builders for VascuMamba-Spot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def set_pub_style() -> None:
    """Configure crisp IEEE/CVPR typography and colorblind-safe aesthetics."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
        }
    )


def plot_neurovascular_timeline(
    time_sec: np.ndarray,
    strain_mag: np.ndarray,
    rppg_bvp: np.ndarray,
    gate_aut: np.ndarray,
    apex_prob: np.ndarray,
    out_path: str | Path,
    biopac_ppg: np.ndarray | None = None,
    blink_intervals_sec: list[tuple[float, float]] | None = None,
    me_intervals_sec: list[tuple[float, float]] | None = None,
) -> Path:
    """Render synchronized kinematic strain, rPPG BVP, BIOPAC GT, gate, and apex probability."""
    set_pub_style()
    n_rows = 4 if biopac_ppg is None else 5
    fig, axes = plt.subplots(n_rows, 1, figsize=(8.0, 1.55 * n_rows), sharex=True, constrained_layout=True)

    def _shade(ax: plt.Axes) -> None:
        if blink_intervals_sec:
            for s, e in blink_intervals_sec:
                ax.axvspan(s, e, color="#D55E00", alpha=0.14, label="Eyeblink (Non-Emotional)")
        if me_intervals_sec:
            for s, e in me_intervals_sec:
                ax.axvspan(s, e, color="#009E73", alpha=0.18, label="True Micro-Expression")

    axes[0].plot(time_sec, strain_mag, color="#0072B2", lw=1.2)
    axes[0].set_ylabel("Optical Strain")
    axes[0].set_title("Synchronized Neuro-Vascular Gating: Eyeblink Suppression vs. Genuine ME")
    _shade(axes[0])

    axes[1].plot(time_sec, rppg_bvp, color="#CC79A7", lw=1.1)
    axes[1].set_ylabel("7-ROI rPPG BVP")
    _shade(axes[1])

    idx = 2
    if biopac_ppg is not None:
        axes[idx].plot(time_sec, biopac_ppg, color="#E69F00", lw=1.1)
        axes[idx].set_ylabel("BIOPAC PPG (GT)")
        _shade(axes[idx])
        idx += 1

    axes[idx].plot(time_sec, gate_aut, color="#009E73", lw=1.3)
    axes[idx].set_ylim(-0.05, 1.05)
    axes[idx].set_ylabel(r"Gate $g_t^{\mathrm{aut}}$")
    _shade(axes[idx])

    axes[idx + 1].plot(time_sec, apex_prob, color="#000000", lw=1.3)
    axes[idx + 1].set_ylim(-0.05, 1.05)
    axes[idx + 1].set_ylabel(r"Apex $\hat{p}_t$")
    axes[idx + 1].set_xlabel("Time (s)")
    _shade(axes[idx + 1])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(), loc="upper right", frameon=False)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_scaling_curves(
    lengths: list[int],
    mamba_ms: list[float],
    transformer_ms: list[float],
    mamba_mem_mb: list[float],
    transformer_mem_mb: list[float],
    out_path: str | Path,
) -> Path:
    """Plot O(T) VascuMamba vs O(T^2) Transformer latency and peak memory curves."""
    set_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.0), constrained_layout=True)

    ax1.plot(lengths, mamba_ms, marker="o", color="#0072B2", lw=1.8, label="VascuMamba-Spot O(T)")
    ax1.plot(lengths, transformer_ms, marker="s", color="#D55E00", lw=1.8, label="MHSA Transformer O(T²)")
    ax1.set_xlabel("Untrimmed Sequence Length T (frames)")
    ax1.set_ylabel("Forward Latency (ms)")
    ax1.set_title("Temporal Latency Scaling")
    ax1.legend(frameon=False)

    ax2.plot(lengths, mamba_mem_mb, marker="o", color="#0072B2", lw=1.8, label="VascuMamba-Spot O(T)")
    ax2.plot(lengths, transformer_mem_mb, marker="s", color="#D55E00", lw=1.8, label="MHSA Transformer O(T²)")
    ax2.set_xlabel("Untrimmed Sequence Length T (frames)")
    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_title("Activation Memory Scaling")
    ax2.legend(frameon=False)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_confusion_and_calibration(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    out_path: str | Path,
    n_bins: int = 10,
) -> Path:
    """Plot row-normalized confusion matrix and confidence reliability diagram (ECE)."""
    set_pub_style()
    y_pred = np.argmax(probs, axis=-1)
    n_cls = len(class_names)
    cm = np.zeros((n_cls, n_cls), dtype=np.float64)
    for yt, yp in zip(y_true, y_pred):
        if 0 <= yt < n_cls and 0 <= yp < n_cls:
            cm[yt, yp] += 1.0
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    cm_norm = cm / row_sums

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    im = ax1.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax1.set_xticks(range(n_cls))
    ax1.set_yticks(range(n_cls))
    ax1.set_xticklabels(class_names, rotation=30, ha="right")
    ax1.set_yticklabels(class_names)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Ground Truth")
    ax1.set_title("Normalized Confusion Matrix")
    for r in range(n_cls):
        for c in range(n_cls):
            val = cm_norm[r, c]
            ax1.text(
                c,
                r,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val > 0.5 else "black",
                fontsize=7.5,
            )
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    confs = np.max(probs, axis=-1)
    correct = (y_pred == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_accs = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if np.any(m):
            bin_centers.append(0.5 * (lo + hi))
            bin_accs.append(float(np.mean(correct[m])))

    ax2.plot([0, 1], [0, 1], ls="--", color="#666666", lw=1.2, label="Perfect Calibration")
    ax2.bar(bin_centers, bin_accs, width=0.08, color="#0072B2", alpha=0.8, label="VascuMamba-Spot")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Empirical Accuracy")
    ax2.set_title("Confidence Reliability Diagram")
    ax2.legend(frameon=False, loc="upper left")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_fp_breakdown(
    categories: list[str],
    kin_only_fps: list[float],
    vascumamba_fps: list[float],
    out_path: str | Path,
) -> Path:
    """Plot false-positive counts across non-emotional artifacts (blinks, head jitter, speech)."""
    set_pub_style()
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.2, 3.0), constrained_layout=True)
    ax.bar(x - width / 2, kin_only_fps, width, color="#D55E00", label="Kinematic-Only")
    ax.bar(x + width / 2, vascumamba_fps, width, color="#009E73", label="VascuMamba-Spot (Ours)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("False Positives per Video")
    ax.set_title("Non-Emotional False-Positive Breakdown")
    ax.legend(frameon=False)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
