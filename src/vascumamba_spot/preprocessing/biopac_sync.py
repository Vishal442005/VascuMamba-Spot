"""CAS(ME)^3 BIOPAC MP160 synchronization and physiological ground-truth correlation audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert, resample
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from vascumamba_spot.preprocessing.rppg_vascular import (
    bandpass_cardiac_signal,
    compute_hr_snr_db,
)


@dataclass(frozen=True)
class BiopacAuditResult:
    """Quantitative verification metrics comparing contactless rPPG against contact BIOPAC."""

    rppg_hr_bpm: float
    biopac_hr_bpm: float
    hr_abs_err_bpm: float
    rppg_snr_db: float
    envelope_pearson_r: float
    me_vs_non_me_auc: float


def load_biopac_signals(biopac_path: str | Path) -> dict[str, np.ndarray]:
    """Load CAS(ME)^3 BIOPAC MP160 channels (PPG/ECG/EDA/RSP) from .mat or .csv."""
    path = Path(biopac_path)
    if path.suffix == ".mat":
        mat = loadmat(str(path))
        out: dict[str, np.ndarray] = {}
        for k, v in mat.items():
            if k.startswith("__") or not isinstance(v, np.ndarray):
                continue
            arr = v.squeeze().astype(np.float32)
            if arr.ndim == 1 and arr.size > 16:
                out[k.lower()] = arr
        return out
    data = np.loadtxt(str(path), delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        return {"ppg": data}
    keys = ["ecg", "ppg", "eda", "rsp"]
    return {keys[i]: data[:, i] for i in range(min(data.shape[1], len(keys)))}


def compute_biopac_arousal_teacher(
    signals: dict[str, np.ndarray],
    num_video_frames: int,
    fps: float,
) -> np.ndarray:
    """Compute normalized [0, 1] sympathetic arousal teacher target from contact BIOPAC signals."""
    components: list[np.ndarray] = []
    if "ppg" in signals:
        ppg = align_biopac_to_video(signals["ppg"], num_video_frames)
        filt = bandpass_cardiac_signal(ppg, fps=fps)
        env = np.abs(hilbert(filt.astype(np.float64))).astype(np.float32)
        env_norm = (env - env.min()) / max(float(env.max() - env.min()), 1e-6)
        components.append(env_norm)
    if "eda" in signals:
        eda = align_biopac_to_video(signals["eda"], num_video_frames)
        phasic = np.maximum(0.0, np.gradient(eda))
        p_max = max(float(np.quantile(phasic, 0.98)), 1e-6)
        components.append(np.clip(phasic / p_max, 0.0, 1.0).astype(np.float32))
    if not components:
        return np.zeros(num_video_frames, dtype=np.float32)
    return np.clip(np.mean(np.stack(components, axis=0), axis=0), 0.0, 1.0).astype(np.float32)


def align_biopac_to_video(
    signal: np.ndarray,
    num_video_frames: int,
) -> np.ndarray:
    """Resample high-rate BIOPAC signal (e.g. 1000 Hz) to match T video frames."""
    if signal.shape[0] == num_video_frames:
        return signal.astype(np.float32)
    return resample(signal.astype(np.float64), num_video_frames).astype(np.float32)


def audit_rppg_against_biopac(
    rppg_bvp: np.ndarray,
    rppg_envelope: np.ndarray,
    biopac_ppg: np.ndarray,
    fps: float,
    me_frame_mask: np.ndarray | None = None,
) -> BiopacAuditResult:
    """Validate contactless facial rPPG against synchronized contact BIOPAC PPG.

    Used in Phase I Go/No-Go auditing on CAS(ME)^3 Part C.
    """
    biopac_aligned = align_biopac_to_video(biopac_ppg, rppg_bvp.shape[0])
    biopac_filt = bandpass_cardiac_signal(biopac_aligned, fps=fps)

    rppg_hr, rppg_snr = compute_hr_snr_db(rppg_bvp, fps=fps)
    bio_hr, _ = compute_hr_snr_db(biopac_filt, fps=fps)

    if np.std(rppg_envelope) < 1e-8 or np.std(biopac_filt) < 1e-8:
        corr = 0.0
    else:
        corr = float(pearsonr(rppg_bvp, biopac_filt).statistic)

    auc = 0.5
    if me_frame_mask is not None and 0 < int(np.sum(me_frame_mask)) < me_frame_mask.size:
        auc = float(roc_auc_score(me_frame_mask.astype(np.int32), rppg_envelope))

    return BiopacAuditResult(
        rppg_hr_bpm=rppg_hr,
        biopac_hr_bpm=bio_hr,
        hr_abs_err_bpm=abs(rppg_hr - bio_hr),
        rppg_snr_db=rppg_snr,
        envelope_pearson_r=corr,
        me_vs_non_me_auc=auc,
    )
