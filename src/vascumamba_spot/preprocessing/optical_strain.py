"""Physical-time dense optical flow and 6-channel Cauchy infinitesimal strain tensor extraction."""

from __future__ import annotations

import cv2
import numpy as np

from vascumamba_spot.preprocessing.landmarks import KINEMATIC_AU_LANDMARKS


def to_grayscale_uint8(frames: np.ndarray) -> np.ndarray:
    """Convert (T, H, W, 3) RGB or (T, H, W) grayscale frames into uint8 luminance."""
    if frames.ndim == 4:
        gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames], axis=0)
    else:
        gray = frames
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def compute_farneback_flow(prev_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
    """Compute dense 2D optical flow (H, W, 2) between two uint8 grayscale frames."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    ).astype(np.float32)


def get_flow_estimator(backend: str = "farneback"):
    """Select dense optical flow engine ('farneback', 'tvl1', or 'raft')."""
    if backend == "farneback":
        return compute_farneback_flow
    if backend == "tvl1":
        if not hasattr(cv2, "optflow") or not hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
            raise RuntimeError("OpenCV contrib DualTVL1OpticalFlow is unavailable in this environment.")
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        return lambda p, n: tvl1.calc(p, n, None).astype(np.float32)
    if backend == "raft":
        try:
            import torch
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        except ImportError as exc:
            raise RuntimeError("torchvision is required for RAFT optical flow backend.") from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=False).to(device).eval()
        transforms = weights.transforms()

        @torch.no_grad()
        def _raft_fn(prev_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
            p3 = torch.from_numpy(np.repeat(prev_gray[None, None], 3, axis=1)).to(device)
            n3 = torch.from_numpy(np.repeat(next_gray[None, None], 3, axis=1)).to(device)
            p_t, n_t = transforms(p3, n3)
            flow = model(p_t, n_t)[-1][0].permute(1, 2, 0).cpu().numpy()
            return flow.astype(np.float32)

        return _raft_fn
    raise ValueError(f"Unknown optical flow backend: {backend}")


def compute_cauchy_strain_tensor(
    flow_vel: np.ndarray,
    prev_flow_vel: np.ndarray,
    dt_sec: float,
) -> np.ndarray:
    """Construct 6-channel Cauchy strain rate tensor from velocity field (H, W, 2).

    Channels: [eps_xx, eps_yy, eps_xy, ||eps||_F, du/dt, dv/dt].
    """
    u = flow_vel[..., 0]
    v = flow_vel[..., 1]

    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)

    eps_xx = du_dx
    eps_yy = dv_dy
    eps_xy = 0.5 * (du_dy + dv_dx)
    fro_norm = np.sqrt(eps_xx**2 + eps_yy**2 + 2.0 * (eps_xy**2) + 1e-12)

    du_dt = (u - prev_flow_vel[..., 0]) / max(dt_sec, 1e-4)
    dv_dt = (v - prev_flow_vel[..., 1]) / max(dt_sec, 1e-4)

    return np.stack([eps_xx, eps_yy, eps_xy, fro_norm, du_dt, dv_dt], axis=-1).astype(np.float32)


def extract_au_optical_strain(
    frames: np.ndarray,
    fps: float,
    kin_masks: dict[str, np.ndarray],
    target_delta_tau_sec: float = 0.015,
    flow_fn=compute_farneback_flow,
) -> np.ndarray:
    """Extract physically normalized 12-ROI Cauchy optical strain features X_kin of shape (T, 144).

    Uses integer frame stride k = max(1, round(fps * target_delta_tau_sec)) and normalizes flow
    displacement by exact physical step dt = k / fps so 25/30/100/200 fps videos remain comparable.
    """
    gray = to_grayscale_uint8(frames)
    t_len, h, w = gray.shape
    roi_names = list(KINEMATIC_AU_LANDMARKS.keys())
    num_rois = len(roi_names)
    out = np.zeros((t_len, num_rois * 12), dtype=np.float32)
    if t_len < 2:
        return out

    stride = max(1, int(round(fps * target_delta_tau_sec)))
    dt_phys = float(stride) / max(fps, 1.0)
    prev_vel = np.zeros((h, w, 2), dtype=np.float32)

    for t in range(t_len):
        prev_idx = max(0, t - stride)
        if t == 0:
            vel = np.zeros((h, w, 2), dtype=np.float32)
        else:
            disp = flow_fn(gray[prev_idx], gray[t])
            actual_dt = max(1, t - prev_idx) / max(fps, 1.0)
            vel = disp / actual_dt

        strain = compute_cauchy_strain_tensor(vel, prev_vel, dt_sec=dt_phys)
        prev_vel = vel

        feats: list[np.ndarray] = []
        for name in roi_names:
            mask = kin_masks[name]
            if not np.any(mask):
                feats.append(np.zeros(12, dtype=np.float32))
                continue
            vals = strain[mask]  # (N_pix, 6)
            mean_v = vals.mean(axis=0)
            std_v = vals.std(axis=0)
            feats.append(np.concatenate([mean_v, std_v], axis=0))
        out[t] = np.concatenate(feats, axis=0)

    return out
