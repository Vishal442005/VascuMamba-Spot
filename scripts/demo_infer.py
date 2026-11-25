"""Standalone demo inference script generating spotted JSON + neuro-vascular PDF timeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from vascumamba_spot.evaluation.postprocess import decode_and_nms_predictions
from vascumamba_spot.models import VascuMambaConfig, VascuMambaSpot
from vascumamba_spot.preprocessing.landmarks import FaceLandmarkStabilizer, default_canonical_rois
from vascumamba_spot.preprocessing.optical_strain import extract_au_optical_strain
from vascumamba_spot.preprocessing.rppg_vascular import extract_vascular_features
from vascumamba_spot.visualization.plots import plot_neurovascular_timeline


def render_overlay_video(
    frames_rgb: np.ndarray,
    gate: np.ndarray,
    apex_prob: np.ndarray,
    fps: float,
    out_video: Path,
) -> None:
    """Export annotated MP4 overlay showing autonomic gate and apex spotting confidence."""
    t_len, h, w, _ = frames_rgb.shape
    out_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, float(fps), (w, h))
    for i in range(t_len):
        bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        g_val = float(gate[i])
        p_val = float(apex_prob[i])
        color = (0, 200, 80) if p_val >= 0.35 else (200, 180, 0)
        cv2.putText(
            bgr,
            f"Gate:{g_val:.2f} Apex:{p_val:.2f}",
            (6, max(16, h - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
        writer.write(bgr)
    writer.release()


def make_synthetic_demo_video(
    num_frames: int = 300,
    fps: float = 50.0,
    size: int = 128,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Generate a synthetic RGB facial sequence with an eyeblink artifact and an aroused ME."""
    kin_masks, vasc_masks = default_canonical_rois(size=size)
    t = np.arange(num_frames, dtype=np.float32) / fps
    frames = np.full((num_frames, size, size, 3), 140.0, dtype=np.float32)

    # Cardiac pulse at 72 BPM (1.2 Hz) strongest in Green channel
    pulse = 2.0 * np.sin(2.0 * np.pi * 1.2 * t)
    # Emotional sympathetic pulse modulation during ME window [3.6s, 4.1s]
    arousal_env = 1.0 + 1.5 * np.exp(-0.5 * ((t - 3.85) / 0.25) ** 2)

    for vmask in vasc_masks.values():
        frames[:, vmask, 1] += (pulse * arousal_env)[:, None]
        frames[:, vmask, 0] += (0.3 * pulse * arousal_env)[:, None]
        frames[:, vmask, 2] += (0.5 * pulse * arousal_env)[:, None]

    # Non-emotional eyeblink at [1.2s, 1.38s] (localized motion on eyelids, zero arousal)
    blink_mask = kin_masks["left_eyelid"] | kin_masks["right_eyelid"]
    for i in range(int(1.2 * fps), int(1.38 * fps)):
        shift = np.sin(np.pi * (i - 1.2 * fps) / (0.18 * fps))
        frames[i, blink_mask, :] -= 35.0 * shift

    # True suppressed ME at [3.7s, 4.0s] (zygomaticus / brow micro-strain + arousal)
    me_mask = kin_masks["left_zygomaticus"] | kin_masks["right_zygomaticus"]
    for i in range(int(3.7 * fps), int(4.0 * fps)):
        shift = np.sin(np.pi * (i - 3.7 * fps) / (0.30 * fps))
        frames[i, me_mask, :] += 18.0 * shift

    return np.clip(frames, 0, 255).astype(np.uint8), kin_masks, vasc_masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VascuMamba-Spot demo inference.")
    parser.add_argument("--video", type=Path, default=None, help="Optional video path; uses synthetic demo if omitted.")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--out-video", type=Path, default=None, help="Optional annotated MP4 overlay path.")
    args = parser.parse_args()

    if args.video is None:
        frames, kin_masks, vasc_masks = make_synthetic_demo_video(fps=args.fps)
        blink_wins = [(1.20, 1.38)]
        me_wins = [(3.70, 4.00)]
    else:
        from extract_features import load_video_frames  # type: ignore

        raw = load_video_frames(args.video, is_mono=False)
        stabilizer = FaceLandmarkStabilizer(target_size=256)
        frames, kin_masks, vasc_masks = stabilizer.stabilize_frames(raw)
        blink_wins, me_wins = None, None

    x_kin = extract_au_optical_strain(frames, fps=args.fps, kin_masks=kin_masks)
    x_phys, stats = extract_vascular_features(frames, fps=args.fps, vasc_masks=vasc_masks, color_mode="rgb")

    model = VascuMambaSpot(VascuMambaConfig())
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()

    with torch.no_grad():
        out = model(
            torch.from_numpy(x_kin).unsqueeze(0),
            torch.from_numpy(x_phys).unsqueeze(0),
            has_rgb=torch.tensor([1.0]),
            dt_frame=1.0 / args.fps,
        )
    apex_prob = out["apex_prob"].squeeze(0).numpy()
    bounds = out["boundaries_sec"].squeeze(0).numpy()
    cls_logits = out["cls_logits"].squeeze(0).numpy()
    gate = out["gate_mean"].squeeze(0).numpy()

    spotted = decode_and_nms_predictions(apex_prob, bounds, cls_logits, gate, fps=args.fps)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "predictions.json").write_text(
        json.dumps({"rppg_stats": stats, "spotted": [asdict(s) for s in spotted]}, indent=2),
        encoding="utf-8",
    )

    time_sec = np.arange(len(x_kin), dtype=np.float32) / args.fps
    strain_mag = x_kin[:, 3::12].mean(axis=1)
    rppg_bvp = x_phys[:, 0:28:4].mean(axis=1)
    fig_path = plot_neurovascular_timeline(
        time_sec=time_sec,
        strain_mag=strain_mag,
        rppg_bvp=rppg_bvp,
        gate_aut=gate,
        apex_prob=apex_prob,
        out_path=args.out_dir / "neurovascular_timeline.pdf",
        blink_intervals_sec=blink_wins,
        me_intervals_sec=me_wins,
    )
    if args.out_video is not None:
        render_overlay_video(frames, gate, apex_prob, fps=args.fps, out_video=args.out_video)
    print(f"Demo complete! HR={stats['mean_hr_bpm']:.1f} BPM SNR={stats['mean_snr_db']:.2f} dB -> {fig_path}")


if __name__ == "__main__":
    main()
