"""1D Temporal Distance-IoU (DIoU) loss for onset/offset boundary regression."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalDIoULoss(nn.Module):
    """1D Distance-IoU loss on (d_s, d_e) boundary offsets over positive expression frames."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean 1D DIoU loss on positive frames.

        Args:
            pred_offsets: Predicted (d_s, d_e) >= 0 of shape (B, T, 2).
            target_offsets: Target (d_s, d_e) >= 0 of shape (B, T, 2).
            pos_mask: Boolean mask (B, T) selecting positive expression frames.
        """
        if not torch.any(pos_mask):
            return pred_offsets.sum() * 0.0

        p = pred_offsets[pos_mask]  # (K, 2)
        g = target_offsets[pos_mask]  # (K, 2)

        # Convert offset pairs relative to frame t into interval [ -d_s, +d_e ]
        p_left, p_right = -p[:, 0], p[:, 1]
        g_left, g_right = -g[:, 0], g[:, 1]

        inter = torch.clamp(torch.minimum(p_right, g_right) - torch.maximum(p_left, g_left), min=0.0)
        area_p = torch.clamp(p_right - p_left, min=self.eps)
        area_g = torch.clamp(g_right - g_left, min=self.eps)
        union = torch.clamp(area_p + area_g - inter, min=self.eps)
        iou = inter / union

        # Enclosing interval c and center distance rho^2
        c_left = torch.minimum(p_left, g_left)
        c_right = torch.maximum(p_right, g_right)
        c_diag_sq = torch.clamp((c_right - c_left) ** 2, min=self.eps)

        p_center = 0.5 * (p_left + p_right)
        g_center = 0.5 * (g_left + g_right)
        rho_sq = (p_center - g_center) ** 2

        diou = iou - (rho_sq / c_diag_sq)
        return torch.mean(1.0 - diou)
