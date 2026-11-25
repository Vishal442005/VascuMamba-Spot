"""Training-time augmentations for dual-stream kinematic and vascular sequences."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_modality_dropout(
    x_kin: torch.Tensor,
    x_phys: torch.Tensor,
    has_rgb: torch.Tensor,
    p_drop_kin: float = 0.15,
    p_drop_phys: float = 0.25,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independently drop kinematic or physiological modalities without dropping both simultaneously.

    Args:
        x_kin: Kinematic tensor of shape (B, T, D_kin).
        x_phys: Physiological tensor of shape (B, T, D_phys).
        has_rgb: Binary tensor of shape (B,) indicating RGB vs monochrome origin.
        p_drop_kin: Probability of zeroing x_kin for a sample.
        p_drop_phys: Probability of zeroing x_phys (and setting has_rgb=0) for a sample.
        training: Whether augmentation is active.

    Returns:
        Tuple of (x_kin_aug, x_phys_aug, has_rgb_aug).
    """
    if not training or (p_drop_kin <= 0.0 and p_drop_phys <= 0.0):
        return x_kin, x_phys, has_rgb

    batch_size = x_kin.shape[0]
    device = x_kin.device
    drop_kin = torch.rand(batch_size, device=device) < p_drop_kin
    drop_phys = torch.rand(batch_size, device=device) < p_drop_phys

    # Never drop both streams simultaneously for the same sample.
    both_dropped = drop_kin & drop_phys
    drop_kin = drop_kin & ~both_dropped

    kin_mask = (~drop_kin).to(x_kin.dtype).view(batch_size, 1, 1)
    phys_mask = (~drop_phys).to(x_phys.dtype).view(batch_size, 1, 1)
    has_rgb_aug = has_rgb * (~drop_phys).to(has_rgb.dtype)
    return x_kin * kin_mask, x_phys * phys_mask, has_rgb_aug


def phase_preserving_temporal_warp(
    x_kin: torch.Tensor,
    x_phys: torch.Tensor,
    intervals: list[tuple[int, int, int, int]],
    scale_range: tuple[float, float] = (0.85, 1.15),
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, int, int]]]:
    """Uniformly warp sequence length within scale_range while keeping kinematic/BVP alignment.

    Args:
        x_kin: (T, D_kin) tensor.
        x_phys: (T, D_phys) tensor.
        intervals: List of (onset, apex, offset, label) frame tuples.
        scale_range: Multiplicative length scaling bounds.

    Returns:
        Warped (x_kin, x_phys, updated_intervals).
    """
    seq_len = x_kin.shape[0]
    if seq_len < 8:
        return x_kin, x_phys, intervals

    scale = float(torch.empty(1).uniform_(scale_range[0], scale_range[1]).item())
    new_len = max(8, int(round(seq_len * scale)))
    if new_len == seq_len:
        return x_kin, x_phys, intervals

    kin_rs = F.interpolate(
        x_kin.transpose(0, 1).unsqueeze(0), size=new_len, mode="linear", align_corners=False
    ).squeeze(0).transpose(0, 1)
    phys_rs = F.interpolate(
        x_phys.transpose(0, 1).unsqueeze(0), size=new_len, mode="linear", align_corners=False
    ).squeeze(0).transpose(0, 1)

    ratio = (new_len - 1) / max(1, seq_len - 1)
    warped_intervals = [
        (
            int(round(on * ratio)),
            int(round(ap * ratio)),
            int(round(off * ratio)),
            lbl,
        )
        for on, ap, off, lbl in intervals
    ]
    return kin_rs, phys_rs, warped_intervals
