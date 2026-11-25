"""CLI script to extract stabilized 12-ROI Cauchy optical strain and 7-ROI vascular features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from vascumamba_spot.data.cache_dataset import write_feature_cache
from vascumamba_spot.data.manifests import load_manifest
from vascumamba_spot.preprocessing.biopac_sync import (
    align_biopac_to_video,
    compute_biopac_arousal_teacher,
    load_biopac_signals,
)
from vascumamba_spot.preprocessing.landmarks import FaceLandmarkStabilizer
from vascumamba_spot.preprocessing.optical_strain import (
    extract_au_optical_strain,
    get_flow_estimator,
)
from vascumamba_spot.preprocessing.rppg_vascular import extract_vascular_features


def load_video_frames(video_path: Path, is_mono: bool) -> np.ndarray:
    """Read video file or ordered image folder into a numpy array."""
    if video_path.is_dir():
        img_paths = sorted(
            p for p in video_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        frames = []
        for p in img_paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE if is_mono else cv2.IMREAD_COLOR)
            if img is None:
                continue
            if not is_mono:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return np.stack(frames, axis=0)

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if is_mono:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract dual-stream features into HDF5 cache.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-cache", required=True, type=Path)
    parser.add_argument("--diagnostics-json", type=Path, default=None)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--flow-backend", default="farneback", choices=["farneback", "tvl1"])
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    stabilizer = FaceLandmarkStabilizer(target_size=args.target_size)
    flow_fn = get_flow_estimator(args.flow_backend)
    diags: dict[str, dict[str, float]] = {}

    for rec in records:
        frames = load_video_frames(Path(rec.video_path), is_mono=(rec.color_mode != "rgb"))
        aligned, kin_masks, vasc_masks = stabilizer.stabilize_frames(frames)
        x_kin = extract_au_optical_strain(aligned, fps=rec.fps, kin_masks=kin_masks, flow_fn=flow_fn)
        x_phys, stats = extract_vascular_features(
            aligned, fps=rec.fps, vasc_masks=vasc_masks, color_mode=rec.color_mode
        )
        extra: dict[str, np.ndarray] = {}
        if rec.biopac_path and Path(rec.biopac_path).exists():
            sigs = load_biopac_signals(rec.biopac_path)
            if "ppg" in sigs:
                extra["biopac_ppg"] = align_biopac_to_video(sigs["ppg"], len(x_kin))
            extra["biopac_arousal"] = compute_biopac_arousal_teacher(sigs, len(x_kin), fps=rec.fps)
        write_feature_cache(
            args.out_cache,
            rec.video_id,
            x_kin=x_kin,
            x_phys=x_phys,
            fps=rec.fps,
            has_rgb=rec.has_rgb,
            extra=extra or None,
        )
        diags[rec.video_id] = stats
        print(f"[{rec.video_id}] T={len(x_kin)} HR={stats['mean_hr_bpm']:.1f} BPM SNR={stats['mean_snr_db']:.2f} dB")

    if args.diagnostics_json:
        args.diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_json.write_text(json.dumps(diags, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
