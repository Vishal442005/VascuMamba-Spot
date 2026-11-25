"""Gaussian-smoothed continuous Focal Loss for dense frame-level apex localization."""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalApexLoss(nn.Module):
    """Continuous focal loss adapted from CenterNet for Gaussian-smoothed apex heatmaps."""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        apex_prob: torch.Tensor,
        apex_target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute focal apex loss over valid frames."""
        p = torch.clamp(apex_prob, min=self.eps, max=1.0 - self.eps)
        y = torch.clamp(apex_target, min=0.0, max=1.0)

        pos_mask = y >= 0.99
        neg_mask = ~pos_mask

        pos_loss = -((1.0 - p) ** self.alpha) * torch.log(p) * pos_mask.to(p.dtype)
        neg_weight = (1.0 - y) ** self.beta
        neg_loss = -(p**self.alpha) * torch.log(1.0 - p) * neg_weight * neg_mask.to(p.dtype)

        loss = pos_loss + neg_loss
        if valid_mask is not None:
            m = valid_mask.to(p.dtype)
            loss = loss * m
            num_pos = torch.clamp((pos_mask & valid_mask).sum().to(p.dtype), min=1.0)
            return loss.sum() / num_pos

        num_pos = torch.clamp(pos_mask.sum().to(p.dtype), min=1.0)
        return loss.sum() / num_pos
