"""Dual-regime 7-ROI facial vascular rPPG & autonomic transient extraction."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch

from vascumamba_spot.data.manifests import ColorMode
from vascumamba_spot.preprocessing.landmarks import VASCULAR_ROI_LANDMARKS


def trimmed_roi_mean(frame: np.ndarray, mask: np.ndarray, trim_frac: float = 0.10) -> np.ndarray:
    """Compute spatially trimmed mean per channel inside mask, rejecting specular/shadow outliers."""
    if frame.ndim == 2:
        vals = frame[mask].astype(np.float32)
        if vals.size == 0:
            return np.zeros(1, dtype=np.float32)
        lo, hi = np.quantile(vals, [trim_frac, 1.0 - trim_frac])
        kept = vals[(vals >= lo) & (vals <= hi)]
        return np.array([kept.mean() if kept.size > 0 else vals.mean()], dtype=np.float32)

    pixels = frame[mask].astype(np.float32)  # (N, 3)
    if pixels.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)
    lum = pixels.mean(axis=1)
    lo, hi = np.quantile(lum, [trim_frac, 1.0 - trim_frac])
    kept = pixels[(lum >= lo) & (lum <= hi)]
    return (kept.mean(axis=0) if kept.shape[0] > 0 else pixels.mean(axis=0)).astype(np.float32)


def pos_chrominance_projection(
    rgb_ts: np.ndarray,
    fps: float,
    window_sec: float = 1.6,
) -> np.ndarray:
    """Apply Plane-Orthogonal-to-Skin (POS) projection to a (T, 3) RGB skin trace.

    Implements temporal mean normalization c_tilde(t) = c(t)/c_bar(t) - 1 followed by
    S1 = G - B, S2 = G + B - 2R, and alpha-tuned combination p(t) = S1 + (std(S1)/std(S2)) S2.
    """
    t_len = rgb_ts.shape[0]
    win = max(4, int(round(fps * window_sec)))
    bvp = np.zeros(t_len, dtype=np.float32)
    if t_len < 4:
        return bvp

    # Overlap-add POS across sliding windows so illumination drift is cancelled locally.
    weights = np.zeros(t_len, dtype=np.float32)
    hann = np.hanning(win).astype(np.float32)
    if hann.sum() <= 0:
        hann = np.ones(win, dtype=np.float32)

    step = max(1, win // 4)
    starts = list(range(0, max(1, t_len - win + 1), step))
    if starts[-1] + win < t_len:
        starts.append(max(0, t_len - win))

    for s in starts:
        e = min(t_len, s + win)
        seg = rgb_ts[s:e]
        mean_rgb = np.maximum(seg.mean(axis=0, keepdims=True), 1e-4)
        cn = (seg / mean_rgb) - 1.0
        s1 = cn[:, 1] - cn[:, 2]
        s2 = cn[:, 1] + cn[:, 2] - 2.0 * cn[:, 0]
        std_s2 = float(np.std(s2))
        alpha = float(np.std(s1)) / max(std_s2, 1e-6)
        p_seg = s1 + alpha * s2
        p_seg = p_seg - p_seg.mean()
        w_seg = hann[: e - s] if (e - s) == win else np.ones(e - s, dtype=np.float32)
        bvp[s:e] += p_seg * w_seg
        weights[s:e] += w_seg

    valid = weights > 1e-6
    bvp[valid] /= weights[valid]
    return bvp


def bandpass_cardiac_signal(
    sig: np.ndarray,
    fps: float,
    low_hz: float = 0.70,
    high_hz: float = 3.50,
    order: int = 3,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter safely clamped below Nyquist frequency."""
    if sig.shape[0] < 16 or fps <= 2.0 * low_hz:
        return sig - sig.mean()
    nyq = 0.5 * fps
    hi = min(high_hz, 0.92 * nyq)
    lo = min(low_hz, 0.5 * hi)
    if lo >= hi:
        return sig - sig.mean()
    b, a = butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    padlen = min(3 * max(len(a), len(b)), sig.shape[0] - 1)
    if padlen < 1:
        return sig - sig.mean()
    return filtfilt(b, a, sig.astype(np.float64), padlen=padlen).astype(np.float32)


def compute_hr_snr_db(
    bvp: np.ndarray,
    fps: float,
    hr_band_hz: tuple[float, float] = (0.75, 3.0),
    peak_halfwidth_hz: float = 0.15,
) -> tuple[float, float]:
    """Estimate dominant heart-rate (BPM) and spectral SNR (dB) from a BVP waveform."""
    if bvp.shape[0] < max(16, int(2 * fps)):
        return 0.0, -30.0
    nperseg = min(bvp.shape[0], max(64, int(round(fps * 8.0))))
    freqs, psd = welch(bvp, fs=fps, nperseg=nperseg, nfft=max(512, nperseg * 4))
    in_band = (freqs >= hr_band_hz[0]) & (freqs <= hr_band_hz[1])
    if not np.any(in_band):
        return 0.0, -30.0
    band_freqs = freqs[in_band]
    band_psd = psd[in_band]
    f_peak = float(band_freqs[int(np.argmax(band_psd))])
    sig_mask = np.abs(freqs - f_peak) <= peak_halfwidth_hz
    noise_mask = in_band & ~sig_mask
    p_sig = float(np.sum(psd[sig_mask]))
    p_noise = float(np.sum(psd[noise_mask])) + 1e-12
    snr_db = 10.0 * np.log10(max(p_sig, 1e-12) / p_noise)
    return f_peak * 60.0, float(snr_db)


def _extract_roi_vascular_features(bvp: np.ndarray, fps: float) -> tuple[np.ndarray, float]:
    """Compute 4 dynamic features + scalar SQI from a single-ROI filtered pulse trace."""
    t_len = bvp.shape[0]
    std = float(np.std(bvp))
    bvp_norm = (bvp - float(np.mean(bvp))) / max(std, 1e-6)

    analytic = hilbert(bvp_norm.astype(np.float64))
    envelope = np.abs(analytic).astype(np.float32)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(phase) * (fps / (2.0 * np.pi))
    inst_freq = np.clip(inst_freq, 0.5, 4.0).astype(np.float32)

    dt = 1.0 / max(fps, 1.0)
    accel = np.gradient(np.gradient(bvp_norm, dt), dt).astype(np.float32)
    accel = np.clip(accel, -50.0, 50.0)

    _, snr_db = compute_hr_snr_db(bvp_norm, fps=fps)
    sqi = float(1.0 / (1.0 + np.exp(-(snr_db - 2.0) / 3.0)))  # smooth [0,1] reliability
    feats = np.stack([bvp_norm, envelope, inst_freq, accel], axis=-1).astype(np.float32)
    if feats.shape[0] != t_len:
        raise RuntimeError("Unexpected length mismatch in vascular feature extraction.")
    return feats, sqi


def extract_vascular_features(
    frames: np.ndarray,
    fps: float,
    vasc_masks: dict[str, np.ndarray],
    color_mode: ColorMode = "rgb",
) -> tuple[np.ndarray, dict[str, float]]:
    """Extract Dual-Regime 7-ROI vascular features X_phys of shape (T, 36).

    Feature layout (D_phys = 36):
      - 7 ROIs * 4 features (normalized BVP, Hilbert envelope, inst freq, 2nd-deriv accel) = 28
      - 7 per-ROI spectral quality indices (SQI in [0, 1]) = 7
      - 1 scalar has_rgb indicator (1.0 for RGB, 0.0 for mono/NIR) = 1
    """
    t_len = frames.shape[0]
    roi_names = list(VASCULAR_ROI_LANDMARKS.keys())
    num_rois = len(roi_names)
    out = np.zeros((t_len, num_rois * 4 + num_rois + 1), dtype=np.float32)
    is_rgb = color_mode == "rgb" and frames.ndim == 4 and frames.shape[-1] == 3
    if t_len < 4:
        out[:, -1] = 1.0 if is_rgb else 0.0
        return out, {"mean_hr_bpm": 0.0, "mean_snr_db": -30.0}

    hrs: list[float] = []
    snrs: list[float] = []
    for r_idx, name in enumerate(roi_names):
        mask = vasc_masks[name]
        if is_rgb:
            rgb_ts = np.stack([trimmed_roi_mean(frames[t], mask) for t in range(t_len)], axis=0)
            raw_pulse = pos_chrominance_projection(rgb_ts, fps=fps)
        else:
            # Monochrome / NIR fallback: normalized intensity fluctuation (no bogus RGB POS)
            gray_frames = frames[..., 0] if frames.ndim == 4 else frames
            mono_ts = np.array(
                [trimmed_roi_mean(gray_frames[t], mask)[0] for t in range(t_len)],
                dtype=np.float32,
            )
            mean_i = max(float(np.mean(mono_ts)), 1e-4)
            raw_pulse = (mono_ts / mean_i) - 1.0

        filtered = bandpass_cardiac_signal(raw_pulse, fps=fps)
        hr_bpm, snr_db = compute_hr_snr_db(filtered, fps=fps)
        hrs.append(hr_bpm)
        snrs.append(snr_db)

        roi_feats, sqi = _extract_roi_vascular_features(filtered, fps=fps)
        out[:, r_idx * 4 : (r_idx + 1) * 4] = roi_feats
        out[:, num_rois * 4 + r_idx] = sqi

    out[:, -1] = 1.0 if is_rgb else 0.0
    diagnostics = {
        "mean_hr_bpm": float(np.mean(hrs)) if hrs else 0.0,
        "mean_snr_db": float(np.mean(snrs)) if snrs else -30.0,
    }
    return out, diagnostics
