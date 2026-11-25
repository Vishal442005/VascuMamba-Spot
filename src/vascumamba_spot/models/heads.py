"""Multi-task prediction heads and SSL masked reconstruction heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskSpottingHeads(nn.Module):
    """Frame-level apex, physical-second boundary regression, and emotion classification heads."""

    def __init__(self, d_model: int, num_classes: int = 5, dropout: float = 0.15) -> None:
        super().__init__()
        self.shared_norm = nn.LayerNorm(d_model)
        self.apex_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict per-frame apex logits, probabilities, boundary offsets (sec), and class logits."""
        hn = self.shared_norm(h)
        apex_logits = self.apex_head(hn).squeeze(-1)  # (B, T)
        apex_prob = torch.sigmoid(apex_logits)
        boundaries_sec = F.softplus(self.boundary_head(hn))  # (B, T, 2) -> (d_s, d_e) >= 0
        cls_logits = self.cls_head(hn)  # (B, T, C)

        # Attention-pooled clip-level emotion logits for clipped recognition benchmarks
        attn_w = torch.softmax(apex_logits, dim=-1).unsqueeze(-1)  # (B, T, 1)
        clip_feat = torch.sum(hn * attn_w, dim=1)  # (B, D)
        clip_cls_logits = self.cls_head(clip_feat)  # (B, C)

        return {
            "apex_logits": apex_logits,
            "apex_prob": apex_prob,
            "boundaries_sec": boundaries_sec,
            "cls_logits": cls_logits,
            "clip_cls_logits": clip_cls_logits,
        }


class SSLReconstructionHead(nn.Module):
    """Masked neuro-vascular reconstruction head for 4M-frame CAS(ME)^3 SSL pretraining."""

    def __init__(self, d_model: int, d_kin_in: int, d_phys_in: int) -> None:
        super().__init__()
        self.recon_kin = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_kin_in),
        )
        self.recon_phys = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_phys_in),
        )

    def forward(self, h_kin: torch.Tensor, h_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.recon_kin(h_kin), self.recon_phys(h_phys)
