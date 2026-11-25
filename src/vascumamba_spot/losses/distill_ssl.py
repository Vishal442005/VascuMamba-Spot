"""Privileged BIOPAC teacher-student gate distillation and Neuro-Vascular SSL losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrivilegedBiopacDistillLoss(nn.Module):
    """Distill normalized contact BIOPAC arousal teacher scores into contactless rPPG gate."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        gate_pred: torch.Tensor,
        biopac_arousal: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MSE + temporal correlation alignment between gate_pred (B, T) and teacher (B, T)."""
        target = torch.clamp(biopac_arousal, min=0.0, max=1.0)
        diff_sq = (gate_pred - target) ** 2
        if valid_mask is not None:
            m = valid_mask.to(gate_pred.dtype)
            mse = (diff_sq * m).sum() / torch.clamp(m.sum(), min=1.0)
        else:
            mse = diff_sq.mean()
        return mse


class NeuroVascularSSLLoss(nn.Module):
    """Masked reconstruction + cross-modal InfoNCE loss for 4M-frame SSL pretraining."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        recon_kin: torch.Tensor,
        target_kin: torch.Tensor,
        mask_kin: torch.Tensor,
        recon_phys: torch.Tensor,
        target_phys: torch.Tensor,
        mask_phys: torch.Tensor,
        z_kin: torch.Tensor,
        z_phys: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute masked kinematic/vascular reconstruction and global-step contrastive alignment."""
        l_kin = (
            F.smooth_l1_loss(recon_kin[mask_kin], target_kin[mask_kin])
            if torch.any(mask_kin)
            else recon_kin.sum() * 0.0
        )
        l_phys = (
            F.smooth_l1_loss(recon_phys[mask_phys], target_phys[mask_phys])
            if torch.any(mask_phys)
            else recon_phys.sum() * 0.0
        )

        # Clip-level cross-modal contrastive alignment across batch
        zk = F.normalize(z_kin.mean(dim=1), dim=-1)
        zp = F.normalize(z_phys.mean(dim=1), dim=-1)
        if zk.shape[0] > 1:
            logits = (zk @ zp.T) / self.temperature
            labels = torch.arange(zk.shape[0], device=zk.device)
            l_nce = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        else:
            l_nce = zk.sum() * 0.0

        total = l_kin + l_phys + 0.5 * l_nce
        return {"loss": total, "l_recon_kin": l_kin, "l_recon_phys": l_phys, "l_nce_ssl": l_nce}
