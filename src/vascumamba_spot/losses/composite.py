"""Composite multi-task objective combining apex, DIoU, InfoNCE, focal cls, and distillation."""

from __future__ import annotations

import torch
import torch.nn as nn

from vascumamba_spot.losses.class_balanced import ClassBalancedFocalLoss
from vascumamba_spot.losses.diou_1d import TemporalDIoULoss
from vascumamba_spot.losses.distill_ssl import PrivilegedBiopacDistillLoss
from vascumamba_spot.losses.focal_apex import FocalApexLoss
from vascumamba_spot.losses.infonce import CrossModalInfoNCELoss


class VascuMambaCompositeLoss(nn.Module):
    """Four-term (+ optional BIOPAC distillation) training objective."""

    def __init__(
        self,
        lambda_apex: float = 1.0,
        lambda_iou: float = 1.5,
        lambda_nce: float = 0.5,
        lambda_cls: float = 1.0,
        lambda_distill: float = 0.5,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lambda_apex = lambda_apex
        self.lambda_iou = lambda_iou
        self.lambda_nce = lambda_nce
        self.lambda_cls = lambda_cls
        self.lambda_distill = lambda_distill

        self.focal_apex = FocalApexLoss()
        self.diou = TemporalDIoULoss()
        self.infonce = CrossModalInfoNCELoss()
        self.cls_loss = ClassBalancedFocalLoss(gamma=focal_gamma, class_weights=class_weights)
        self.distill = PrivilegedBiopacDistillLoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute total loss and per-term diagnostics dictionary."""
        valid_mask = batch.get("valid_mask")
        pos_mask = batch["pos_mask"]

        l_apex = self.focal_apex(outputs["apex_prob"], batch["apex_target"], valid_mask=valid_mask)
        l_iou = self.diou(outputs["boundaries_sec"], batch["boundary_target"], pos_mask=pos_mask)
        l_nce = (
            self.infonce(
                outputs["z_kin_nce"],
                outputs["z_phys_nce"],
                pos_mask=pos_mask,
                valid_mask=valid_mask,
            )
            if self.lambda_nce > 0.0
            else l_apex * 0.0
        )
        l_cls = self.cls_loss(outputs["cls_logits"], batch["cls_target"])

        total = (
            self.lambda_apex * l_apex
            + self.lambda_iou * l_iou
            + self.lambda_nce * l_nce
            + self.lambda_cls * l_cls
        )

        l_distill = l_apex * 0.0
        if "biopac_arousal" in batch and batch["biopac_arousal"] is not None:
            l_distill = self.distill(
                outputs["gate_mean"], batch["biopac_arousal"], valid_mask=valid_mask
            )
            total = total + self.lambda_distill * l_distill

        return {
            "loss": total,
            "l_apex": l_apex.detach(),
            "l_iou": l_iou.detach(),
            "l_nce": l_nce.detach(),
            "l_cls": l_cls.detach(),
            "l_distill": l_distill.detach(),
        }
