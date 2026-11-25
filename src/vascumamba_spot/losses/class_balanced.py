"""Class-balanced focal loss using effective number of samples weighting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_effective_class_weights(
    class_counts: list[int] | torch.Tensor,
    beta: float = 0.99,
) -> torch.Tensor:
    """Compute Cui et al. effective-number weights w_c = (1 - beta) / (1 - beta^{n_c})."""
    counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp(min=1.0)
    eff_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts)
    weights = (1.0 - beta) / torch.clamp(eff_num, min=1e-6)
    return weights / weights.mean()


class ClassBalancedFocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class effective-number weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute class-balanced focal loss for (N, C) or (B, T, C) logits."""
        if logits.ndim == 3:
            logits = logits.reshape(-1, logits.shape[-1])
            targets = targets.reshape(-1)

        valid = targets != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0

        v_logits = logits[valid]
        v_targets = targets[valid]
        ce = F.cross_entropy(v_logits, v_targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce

        if self.class_weights is not None:
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)[v_targets]
            focal = focal * w
        return focal.mean()
