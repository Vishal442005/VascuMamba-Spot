"""HDF5 feature caching and PyTorch datasets for untrimmed and clipped micro-expression training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from vascumamba_spot.data.augmentations import phase_preserving_temporal_warp
from vascumamba_spot.data.manifests import EMOTION_TO_ID_3CLS, EMOTION_TO_ID_5CLS, VideoRecord


def write_feature_cache(
    cache_path: str | Path,
    video_id: str,
    x_kin: np.ndarray,
    x_phys: np.ndarray,
    fps: float,
    has_rgb: bool,
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    """Persist pre-extracted dual-stream features into an HDF5 archive."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as h5f:
        if video_id in h5f:
            del h5f[video_id]
        grp = h5f.create_group(video_id)
        grp.create_dataset("x_kin", data=x_kin.astype(np.float32), compression="gzip")
        grp.create_dataset("x_phys", data=x_phys.astype(np.float32), compression="gzip")
        grp.attrs["fps"] = float(fps)
        grp.attrs["has_rgb"] = int(has_rgb)
        if extra:
            for k, v in extra.items():
                grp.create_dataset(k, data=v)


def build_frame_targets(
    num_frames: int,
    fps: float,
    intervals: list[tuple[int, int, int, int]],
    apex_sigma_sec: float = 0.06,
) -> dict[str, torch.Tensor]:
    """Build frame-wise dense targets for apex heatmap, boundary offsets (sec), and emotion IDs.

    Args:
        num_frames: Sequence length T.
        fps: Effective feature sampling rate in Hz.
        intervals: List of (onset, apex, offset, emotion_id).
        apex_sigma_sec: Gaussian smoothing sigma around each apex frame in seconds.

    Returns:
        Dictionary with apex_target (T,), boundary_target (T, 2), pos_mask (T,), cls_target (T,).
    """
    apex_target = np.zeros(num_frames, dtype=np.float32)
    boundary_target = np.zeros((num_frames, 2), dtype=np.float32)
    pos_mask = np.zeros(num_frames, dtype=np.bool_)
    cls_target = np.full(num_frames, -100, dtype=np.int64)

    sigma_frames = max(1.0, apex_sigma_sec * fps)
    t_idx = np.arange(num_frames, dtype=np.float32)

    for onset, apex, offset, emo_id in intervals:
        on = max(0, min(num_frames - 1, onset))
        off = max(on + 1, min(num_frames - 1, offset))
        ap = max(on, min(off, apex))

        gauss = np.exp(-0.5 * ((t_idx - ap) / sigma_frames) ** 2)
        apex_target = np.maximum(apex_target, gauss.astype(np.float32))

        for t in range(on, off + 1):
            d_s = max(0.0, (t - on) / fps)
            d_e = max(0.0, (off - t) / fps)
            boundary_target[t, 0] = d_s
            boundary_target[t, 1] = d_e
            pos_mask[t] = True
            if emo_id >= 0:
                cls_target[t] = emo_id

    return {
        "apex_target": torch.from_numpy(apex_target),
        "boundary_target": torch.from_numpy(boundary_target),
        "pos_mask": torch.from_numpy(pos_mask),
        "cls_target": torch.from_numpy(cls_target),
    }


class CachedFeatureDataset(Dataset):
    """PyTorch Dataset reading cached (x_kin, x_phys) sequences for a manifest split."""

    def __init__(
        self,
        records: list[VideoRecord] | tuple[VideoRecord, ...],
        cache_path: str | Path,
        emotion_scheme: str = "5cls",
        training: bool = False,
        temporal_warp_range: tuple[float, float] | None = None,
    ) -> None:
        self.records = list(records)
        self.cache_path = Path(cache_path)
        self.emotion_map = EMOTION_TO_ID_3CLS if emotion_scheme == "3cls" else EMOTION_TO_ID_5CLS
        self.training = training
        self.temporal_warp_range = temporal_warp_range

    def __len__(self) -> int:
        return len(self.records)

    def _map_label(self, raw: str) -> int:
        key = raw.strip().lower()
        return self.emotion_map.get(key, -100)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        with h5py.File(self.cache_path, "r") as h5f:
            grp = h5f[rec.video_id]
            x_kin = torch.from_numpy(grp["x_kin"][:].astype(np.float32))
            x_phys = torch.from_numpy(grp["x_phys"][:].astype(np.float32))
            fps = float(grp.attrs["fps"])
            has_rgb = float(grp.attrs["has_rgb"])
            biopac_arousal = (
                torch.from_numpy(grp["biopac_arousal"][:].astype(np.float32))
                if "biopac_arousal" in grp
                else None
            )

        intervals = [
            (a.onset, a.apex, a.offset, self._map_label(a.emotion_raw))
            for a in rec.annotations
        ]
        if self.training and self.temporal_warp_range is not None:
            x_kin, x_phys, intervals = phase_preserving_temporal_warp(
                x_kin, x_phys, intervals, self.temporal_warp_range
            )

        targets = build_frame_targets(x_kin.shape[0], fps=fps, intervals=intervals)
        item = {
            "video_id": rec.video_id,
            "x_kin": x_kin,
            "x_phys": x_phys,
            "fps": fps,
            "has_rgb": torch.tensor(has_rgb, dtype=torch.float32),
            "annotations": intervals,
            **targets,
        }
        if biopac_arousal is not None and biopac_arousal.shape[0] == x_kin.shape[0]:
            item["biopac_arousal"] = biopac_arousal
        return item


def collate_variable_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length dual-stream sequences along the temporal axis."""
    max_t = max(item["x_kin"].shape[0] for item in batch)
    bsz = len(batch)
    d_kin = batch[0]["x_kin"].shape[1]
    d_phys = batch[0]["x_phys"].shape[1]

    x_kin = torch.zeros(bsz, max_t, d_kin, dtype=torch.float32)
    x_phys = torch.zeros(bsz, max_t, d_phys, dtype=torch.float32)
    valid_mask = torch.zeros(bsz, max_t, dtype=torch.bool)
    apex_target = torch.zeros(bsz, max_t, dtype=torch.float32)
    boundary_target = torch.zeros(bsz, max_t, 2, dtype=torch.float32)
    pos_mask = torch.zeros(bsz, max_t, dtype=torch.bool)
    cls_target = torch.full((bsz, max_t), -100, dtype=torch.int64)
    has_rgb = torch.stack([item["has_rgb"] for item in batch])

    for i, item in enumerate(batch):
        t = item["x_kin"].shape[0]
        x_kin[i, :t] = item["x_kin"]
        x_phys[i, :t] = item["x_phys"]
        valid_mask[i, :t] = True
        apex_target[i, :t] = item["apex_target"]
        boundary_target[i, :t] = item["boundary_target"]
        pos_mask[i, :t] = item["pos_mask"]
        cls_target[i, :t] = item["cls_target"]

    out: dict[str, Any] = {
        "video_ids": [item["video_id"] for item in batch],
        "fps": [item["fps"] for item in batch],
        "annotations": [item["annotations"] for item in batch],
        "x_kin": x_kin,
        "x_phys": x_phys,
        "valid_mask": valid_mask,
        "has_rgb": has_rgb,
        "apex_target": apex_target,
        "boundary_target": boundary_target,
        "pos_mask": pos_mask,
        "cls_target": cls_target,
    }
    if all("biopac_arousal" in item for item in batch):
        bio = torch.zeros(bsz, max_t, dtype=torch.float32)
        for i, item in enumerate(batch):
            t = item["biopac_arousal"].shape[0]
            bio[i, :t] = item["biopac_arousal"]
        out["biopac_arousal"] = bio
    return out
