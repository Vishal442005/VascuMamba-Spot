"""Cross-modal InfoNCE aligning kinematic and vascular embeddings at genuine ME apexes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalInfoNCELoss(nn.Module):
    """Symmetric temporal InfoNCE aligning kinematic and vascular representations."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_kin: torch.Tensor,
        z_phys: torch.Tensor,
        pos_mask: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        max_samples: int = 256,
    ) -> torch.Tensor:
        """Align positive expression frame embeddings against same-sequence frames."""
        if valid_mask is not None:
            anchor_mask = pos_mask & valid_mask
            pool_mask = valid_mask
        else:
            anchor_mask = pos_mask
            pool_mask = torch.ones_like(pos_mask, dtype=torch.bool)

        if not torch.any(anchor_mask):
            return z_kin.sum() * 0.0

        zk_pos = F.normalize(z_kin[anchor_mask], dim=-1)
        zp_pos = F.normalize(z_phys[anchor_mask], dim=-1)
        zk_all = F.normalize(z_kin[pool_mask], dim=-1)
        zp_all = F.normalize(z_phys[pool_mask], dim=-1)

        if zk_pos.shape[0] > max_samples:
            perm = torch.randperm(zk_pos.shape[0], device=zk_pos.device)[:max_samples]
            zk_pos = zk_pos[perm]
            zp_pos = zp_pos[perm]
        if zk_all.shape[0] > max_samples:
            perm_all = torch.randperm(zk_all.shape[0], device=zk_all.device)[:max_samples]
            zk_all = zk_all[perm_all]
            zp_all = zp_all[perm_all]

        # Ensure positive pairs are included in the candidate denominator
        cand_p = torch.cat([zp_pos, zp_all], dim=0)
        cand_k = torch.cat([zk_pos, zk_all], dim=0)
        n_pos = zk_pos.shape[0]
        labels = torch.arange(n_pos, device=zk_pos.device)

        logits_k2p = (zk_pos @ cand_p.T) / self.temperature
        logits_p2k = (zp_pos @ cand_k.T) / self.temperature
        return 0.5 * (F.cross_entropy(logits_k2p, labels) + F.cross_entropy(logits_p2k, labels))
