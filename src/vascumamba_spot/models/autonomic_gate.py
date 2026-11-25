"""Multiplicative Autonomic Gate modulating continuous physical-time SSM step and input projection."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

GatingMode = Literal["multiplicative", "additive", "concat"]


class AutonomicGate(nn.Module):
    """Compute autonomic arousal gate g_t^aut and modulate kinematic (Delta_t, B_t).

    Under 'multiplicative' mode (Proposition 1 compliant):
      g_t = sigmoid(W_g x_phys + b_g)
      Delta_t^kin = dt_frame * g_eff * clamp(softplus(W_delta x_kin + b_delta), max=delta_max)
      B_t^kin = g_eff * (W_B x_kin)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        delta_max: float = 1.0,
        gate_floor_mono: float = 0.50,
        gating_mode: GatingMode = "multiplicative",
        init_null_suppress_eps: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.delta_max = delta_max
        self.gate_floor_mono = gate_floor_mono
        self.gating_mode = gating_mode

        self.gate_proj = nn.Linear(d_model, d_model)
        # Initialize bias moderately negative so resting baseline starts selective (Prop 1)
        bias_init = -math.log((1.0 - init_null_suppress_eps) / init_null_suppress_eps) * 0.25
        nn.init.constant_(self.gate_proj.bias, bias_init)

        self.delta_proj = nn.Linear(d_model, d_model)
        self.p_to_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.gamma_aut = nn.Parameter(torch.ones(d_model))

        self.b_proj = nn.Linear(d_model, d_state, bias=False)
        self.b_gate_proj = nn.Linear(d_model, d_state, bias=True)
        self.c_proj = nn.Linear(d_model, d_state, bias=False)

    def forward(
        self,
        x_kin: torch.Tensor,
        x_phys: torch.Tensor,
        has_rgb: torch.Tensor,
        dt_frame: torch.Tensor | float = 0.02,
        force_open_gate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gated SSM parameters (delta_t, b_t, c_t, g_eff).

        Args:
            x_kin: Kinematic hidden stream (B, T, D).
            x_phys: Vascular hidden stream (B, T, D).
            has_rgb: Binary RGB flag (B,) in {0.0, 1.0}.
            dt_frame: Physical frame step in seconds, scalar or (B,) tensor.
            force_open_gate: If True, fixes g_eff = 1 (ABL-1 Kinematic-Only control).

        Returns:
            delta_t: (B, T, D)
            b_t: (B, T, N)
            c_t: (B, T, N)
            g_eff: (B, T, D) effective autonomic gate in (0, 1).
        """
        bsz, seq_len, _ = x_kin.shape
        if isinstance(dt_frame, (float, int)):
            dt_scale = torch.full((bsz, 1, 1), float(dt_frame), device=x_kin.device, dtype=x_kin.dtype)
        else:
            dt_scale = dt_frame.to(device=x_kin.device, dtype=x_kin.dtype).view(bsz, 1, 1)

        # Normalize dt relative to canonical 50Hz (0.02s) so default delta magnitudes stay well-scaled
        dt_rel = torch.clamp(dt_scale / 0.02, min=0.1, max=4.0)

        if force_open_gate:
            g_eff = torch.ones_like(x_kin)
        else:
            g_raw = torch.sigmoid(self.gate_proj(x_phys))
            rgb_mask = has_rgb.to(device=x_kin.device, dtype=x_kin.dtype).view(bsz, 1, 1)
            g_mono = torch.clamp(g_raw, min=self.gate_floor_mono)
            g_eff = rgb_mask * g_raw + (1.0 - rgb_mask) * g_mono

        raw_delta = self.delta_proj(x_kin)
        b_base = self.b_proj(x_kin)
        c_t = self.c_proj(x_kin)

        if self.gating_mode == "multiplicative":
            delta_base = torch.clamp(F.softplus(raw_delta), max=self.delta_max)
            delta_t = dt_rel * g_eff * delta_base
            g_state = torch.sigmoid(self.b_gate_proj(g_eff)) if not force_open_gate else torch.ones_like(b_base)
            b_t = g_state * b_base
        elif self.gating_mode == "additive":
            phys_mod = self.gamma_aut.view(1, 1, -1) * (g_eff * self.p_to_k_proj(x_phys))
            delta_t = dt_rel * torch.clamp(F.softplus(raw_delta + phys_mod), max=self.delta_max)
            b_t = b_base
        elif self.gating_mode == "concat":
            phys_mod = self.p_to_k_proj(x_phys)
            delta_t = dt_rel * torch.clamp(F.softplus(raw_delta + phys_mod), max=self.delta_max)
            b_t = b_base
        else:
            raise ValueError(f"Unsupported gating_mode: {self.gating_mode}")

        # Ensure delta_t stays strictly positive for ZOH stability
        delta_t = torch.clamp(delta_t, min=1e-5)
        return delta_t, b_t, c_t, g_eff
