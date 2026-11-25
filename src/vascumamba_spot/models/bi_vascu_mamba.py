"""Bidirectional Non-Causal VascuMamba Encoder and matched-width Transformer/BiGRU ablations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vascumamba_spot.models.autonomic_gate import AutonomicGate, GatingMode
from vascumamba_spot.models.ssm_scan import SelectiveSSMCore


class VascuMambaLayer(nn.Module):
    """Single Bidirectional Autonomic-Gated Selective State-Space Layer."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.15,
        delta_max: float = 1.0,
        gate_floor_mono: float = 0.50,
        gating_mode: GatingMode = "multiplicative",
        bidirectional: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        d_inner = d_model * expand

        self.norm_kin = nn.LayerNorm(d_model)
        self.norm_phys = nn.LayerNorm(d_model)

        self.in_proj_kin = nn.Linear(d_model, d_inner)
        self.in_proj_phys = nn.Linear(d_model, d_inner)

        pad = d_conv // 2
        self.conv1d_kin = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, padding=pad, groups=d_inner
        )
        self.conv1d_phys = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, padding=pad, groups=d_inner
        )

        self.gate_fwd = AutonomicGate(
            d_model=d_inner,
            d_state=d_state,
            delta_max=delta_max,
            gate_floor_mono=gate_floor_mono,
            gating_mode=gating_mode,
        )
        self.ssm_fwd = SelectiveSSMCore(d_model=d_inner, d_state=d_state, backend=backend)

        if bidirectional:
            self.gate_bwd = AutonomicGate(
                d_model=d_inner,
                d_state=d_state,
                delta_max=delta_max,
                gate_floor_mono=gate_floor_mono,
                gating_mode=gating_mode,
            )
            self.ssm_bwd = SelectiveSSMCore(d_model=d_inner, d_state=d_state, backend=backend)

        self.out_proj = nn.Linear(d_inner, d_model)
        self.phys_update = nn.Sequential(
            nn.Linear(d_inner, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def _local_conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        t_len = x.shape[1]
        y = conv(x.transpose(1, 2))[..., :t_len].transpose(1, 2)
        return F.silu(y)

    def forward(
        self,
        x_kin: torch.Tensor,
        x_phys: torch.Tensor,
        has_rgb: torch.Tensor,
        dt_frame: torch.Tensor | float = 0.02,
        force_open_gate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning updated (x_kin, x_phys, mean_gate_map)."""
        res_kin = x_kin
        res_phys = x_phys

        xk = self._local_conv(self.in_proj_kin(self.norm_kin(x_kin)), self.conv1d_kin)
        xp = self._local_conv(self.in_proj_phys(self.norm_phys(x_phys)), self.conv1d_phys)

        delta_f, b_f, c_f, g_f = self.gate_fwd(
            xk, xp, has_rgb=has_rgb, dt_frame=dt_frame, force_open_gate=force_open_gate
        )
        h_fwd, _ = self.ssm_fwd(xk, delta_f, b_f, c_f)

        if self.bidirectional:
            xk_rev = torch.flip(xk, dims=[1])
            xp_rev = torch.flip(xp, dims=[1])
            delta_b, b_b, c_b, g_b = self.gate_bwd(
                xk_rev, xp_rev, has_rgb=has_rgb, dt_frame=dt_frame, force_open_gate=force_open_gate
            )
            h_bwd_rev, _ = self.ssm_bwd(xk_rev, delta_b, b_b, c_b)
            h_bwd = torch.flip(h_bwd_rev, dims=[1])
            g_b_aligned = torch.flip(g_b, dims=[1])
            fused = h_fwd * F.silu(h_bwd) + h_fwd + h_bwd
            gate_map = 0.5 * (g_f + g_b_aligned)
        else:
            fused = h_fwd * F.silu(h_fwd)
            gate_map = g_f

        x_kin_out = res_kin + self.drop(self.out_proj(fused))
        x_phys_out = res_phys + self.phys_update(xp)
        return x_kin_out, x_phys_out, gate_map


class TransformerBackbone(nn.Module):
    """Matched-width Multi-Head Self-Attention ablation backbone (ABL-6)."""

    def __init__(self, d_model: int, num_layers: int = 4, nhead: int = 4, dropout: float = 0.15) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class BiGRUBackbone(nn.Module):
    """Matched-width Bidirectional GRU ablation backbone (ABL-6)."""

    def __init__(self, d_model: int, num_layers: int = 4, dropout: float = 0.15) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out
