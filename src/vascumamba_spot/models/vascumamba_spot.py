"""End-to-end VascuMamba-Spot architecture unifying stems, MNC filter, Bi-VascuMamba, and heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from vascumamba_spot.models.autonomic_gate import GatingMode
from vascumamba_spot.models.bi_vascu_mamba import (
    BiGRUBackbone,
    TransformerBackbone,
    VascuMambaLayer,
)
from vascumamba_spot.models.heads import MultiTaskSpottingHeads, SSLReconstructionHead
from vascumamba_spot.models.mnc_filter import MotionOrthogonalNoiseCancel

BackboneType = Literal["vascumamba", "transformer", "bigru"]


@dataclass
class VascuMambaConfig:
    """Configuration dataclass for VascuMamba-Spot and its ablation variants."""

    d_kin_in: int = 144
    d_phys_in: int = 36
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    num_layers: int = 4
    dropout: float = 0.15
    gate_floor_mono: float = 0.50
    delta_max: float = 1.0
    ssm_backend: str = "auto"
    backbone_type: BackboneType = "vascumamba"
    bidirectional: bool = True
    use_autonomic_gate: bool = True
    use_mnc_filter: bool = True
    kinematic_only: bool = False
    vascular_only: bool = False
    global_phys_only: bool = False  # ABL-7: simulate global face mean by averaging 7 ROIs
    gating_mode: GatingMode = "multiplicative"
    num_classes: int = 5


class TemporalStem(nn.Module):
    """Causal-friendly 1D convolutional projection stem mapping raw ROI features to d_model."""

    def __init__(self, d_in: int, d_model: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(y)


class VascuMambaSpot(nn.Module):
    """Complete VascuMamba-Spot network for untrimmed ME spotting, recognition, and SSL."""

    def __init__(self, cfg: VascuMambaConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.mnc_filter = (
            MotionOrthogonalNoiseCancel(cfg.d_kin_in, cfg.d_phys_in)
            if cfg.use_mnc_filter
            else None
        )
        self.kin_stem = TemporalStem(cfg.d_kin_in, cfg.d_model, dropout=cfg.dropout)
        self.phys_stem = TemporalStem(cfg.d_phys_in, cfg.d_model, dropout=cfg.dropout)

        if cfg.backbone_type == "vascumamba":
            self.layers = nn.ModuleList(
                [
                    VascuMambaLayer(
                        d_model=cfg.d_model,
                        d_state=cfg.d_state,
                        d_conv=cfg.d_conv,
                        expand=cfg.expand,
                        dropout=cfg.dropout,
                        delta_max=cfg.delta_max,
                        gate_floor_mono=cfg.gate_floor_mono,
                        gating_mode=cfg.gating_mode,
                        bidirectional=cfg.bidirectional,
                        backend=cfg.ssm_backend,
                    )
                    for _ in range(cfg.num_layers)
                ]
            )
            self.alt_backbone = None
        elif cfg.backbone_type == "transformer":
            self.layers = None
            self.alt_backbone = TransformerBackbone(
                d_model=cfg.d_model, num_layers=cfg.num_layers, dropout=cfg.dropout
            )
        elif cfg.backbone_type == "bigru":
            self.layers = None
            self.alt_backbone = BiGRUBackbone(
                d_model=cfg.d_model, num_layers=cfg.num_layers, dropout=cfg.dropout
            )
        else:
            raise ValueError(f"Unknown backbone_type: {cfg.backbone_type}")

        # Projection heads for cross-modal InfoNCE alignment at ME apexes
        self.proj_kin_nce = nn.Linear(cfg.d_model, cfg.d_model)
        self.proj_phys_nce = nn.Linear(cfg.d_model, cfg.d_model)

        self.heads = MultiTaskSpottingHeads(
            d_model=cfg.d_model, num_classes=cfg.num_classes, dropout=cfg.dropout
        )
        self.ssl_head = SSLReconstructionHead(cfg.d_model, cfg.d_kin_in, cfg.d_phys_in)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x_kin: torch.Tensor,
        x_phys: torch.Tensor,
        has_rgb: torch.Tensor | None = None,
        dt_frame: torch.Tensor | float = 0.02,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass through dual stems, MNC filter, Bi-VascuMamba stack, and multi-task heads."""
        bsz, seq_len, _ = x_kin.shape
        if has_rgb is None:
            has_rgb = torch.ones(bsz, device=x_kin.device, dtype=x_kin.dtype)

        if self.cfg.global_phys_only:
            # ABL-7: Collapse 7 vascular ROIs into a single global spatial average
            roi_feats = x_phys[..., :28].view(bsz, seq_len, 7, 4).mean(dim=2, keepdim=True)
            roi_sqi = x_phys[..., 28:35].mean(dim=-1, keepdim=True)
            x_phys = torch.cat(
                [roi_feats.repeat(1, 1, 7, 1).reshape(bsz, seq_len, 28), roi_sqi.repeat(1, 1, 7), x_phys[..., 35:36]],
                dim=-1,
            )

        x_phys_clean = self.mnc_filter(x_phys, x_kin) if self.mnc_filter is not None else x_phys

        if self.cfg.vascular_only:
            # ABL-2: Feed vascular stream directly into main backbone without optical strain
            h_kin = self.phys_stem(x_phys_clean)
            h_phys = h_kin
            force_open = True
        elif self.cfg.kinematic_only:
            # ABL-1: Kinematic-Only control with open gate
            h_kin = self.kin_stem(x_kin)
            h_phys = torch.zeros_like(h_kin)
            force_open = True
        else:
            h_kin = self.kin_stem(x_kin)
            h_phys = self.phys_stem(x_phys_clean)
            force_open = not self.cfg.use_autonomic_gate

        gates: list[torch.Tensor] = []
        if self.layers is not None:
            for layer in self.layers:
                h_kin, h_phys, g_map = layer(
                    h_kin,
                    h_phys,
                    has_rgb=has_rgb,
                    dt_frame=dt_frame,
                    force_open_gate=force_open,
                )
                gates.append(g_map.mean(dim=-1))  # (B, T)
            gate_mean = torch.stack(gates, dim=0).mean(dim=0)  # (B, T)
        else:
            fused_in = h_kin if (self.cfg.kinematic_only or self.cfg.vascular_only) else (h_kin + h_phys)
            pad_mask = ~valid_mask if (valid_mask is not None and self.cfg.backbone_type == "transformer") else None
            if self.cfg.backbone_type == "transformer":
                h_kin = self.alt_backbone(fused_in, src_key_padding_mask=pad_mask)
            else:
                h_kin = self.alt_backbone(fused_in)
            gate_mean = torch.ones(bsz, seq_len, device=x_kin.device, dtype=x_kin.dtype)

        preds = self.heads(h_kin)
        recon_kin, recon_phys = self.ssl_head(h_kin, h_phys)

        return {
            **preds,
            "h_kin": h_kin,
            "h_phys": h_phys,
            "z_kin_nce": self.proj_kin_nce(h_kin),
            "z_phys_nce": self.proj_phys_nce(h_phys),
            "gate_mean": gate_mean,
            "recon_kin": recon_kin,
            "recon_phys": recon_phys,
        }
