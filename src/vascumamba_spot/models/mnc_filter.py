"""Motion-Orthogonal Adaptive Noise Cancellation (MNC-rPPG) filter."""

from __future__ import annotations

import torch
import torch.nn as nn


class MotionOrthogonalNoiseCancel(nn.Module):
    """Scrubs residual kinematic motion leakage from vascular features before autonomic gating.

    Uses local temporal depthwise convolutions over kinematic strain features as an adaptive
    noise reference and subtracts the predicted motion-correlated interference from x_phys.
    """

    def __init__(self, d_kin: int, d_phys: int, kernel_size: int = 5) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.kin_ref = nn.Sequential(
            nn.Conv1d(d_kin, d_phys, kernel_size=1),
            nn.Conv1d(d_phys, d_phys, kernel_size=kernel_size, padding=pad, groups=d_phys),
            nn.GELU(),
            nn.Conv1d(d_phys, d_phys, kernel_size=1),
        )
        # Learnable bounded cancellation gain initialized small for stability
        self.gain_logit = nn.Parameter(torch.full((d_phys,), -1.5))

    def forward(self, x_phys: torch.Tensor, x_kin: torch.Tensor) -> torch.Tensor:
        """Return motion-debiased vascular tensor of shape (B, T, D_phys)."""
        noise_est = self.kin_ref(x_kin.transpose(1, 2)).transpose(1, 2)
        gain = torch.sigmoid(self.gain_logit).view(1, 1, -1)
        return x_phys - gain * noise_est
