"""Tests for VascuMamba-Spot architecture, ablation modes, losses, and baselines."""

from __future__ import annotations

import pytest
import torch

from vascumamba_spot.baselines.registry import build_baseline, list_baselines
from vascumamba_spot.losses import VascuMambaCompositeLoss
from vascumamba_spot.losses.distill_ssl import NeuroVascularSSLLoss, PrivilegedBiopacDistillLoss
from vascumamba_spot.models import VascuMambaConfig, VascuMambaSpot


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"kinematic_only": True, "use_autonomic_gate": False},
        {"vascular_only": True, "use_autonomic_gate": False},
        {"gating_mode": "additive"},
        {"gating_mode": "concat"},
        {"bidirectional": False},
        {"backbone_type": "transformer"},
        {"backbone_type": "bigru"},
        {"use_mnc_filter": False},
    ],
)
def test_vascumamba_forward_and_backward_all_modes(overrides: dict) -> None:
    """Verify forward shapes, parameter efficiency, and gradient flow across all ablation variants."""
    cfg = VascuMambaConfig(d_model=64, num_layers=2, num_classes=5, **overrides)
    model = VascuMambaSpot(cfg)
    assert model.count_parameters() < 2_500_000

    bsz, seq_len = 2, 40
    x_kin = torch.randn(bsz, seq_len, 144)
    x_phys = torch.randn(bsz, seq_len, 36)
    has_rgb = torch.tensor([1.0, 0.0])  # mixed RGB + monochrome batch
    valid_mask = torch.ones(bsz, seq_len, dtype=torch.bool)
    pos_mask = torch.zeros(bsz, seq_len, dtype=torch.bool)
    pos_mask[:, 15:25] = True

    apex_target = torch.zeros(bsz, seq_len)
    apex_target[:, 20] = 1.0
    boundary_target = torch.zeros(bsz, seq_len, 2)
    boundary_target[:, 15:25, 0] = 0.10
    boundary_target[:, 15:25, 1] = 0.12
    cls_target = torch.full((bsz, seq_len), -100, dtype=torch.long)
    cls_target[:, 15:25] = 2

    out = model(x_kin, x_phys, has_rgb=has_rgb, dt_frame=0.02, valid_mask=valid_mask)
    assert out["apex_prob"].shape == (bsz, seq_len)
    assert out["boundaries_sec"].shape == (bsz, seq_len, 2)
    assert out["cls_logits"].shape == (bsz, seq_len, 5)
    assert out["clip_cls_logits"].shape == (bsz, 5)
    assert out["gate_mean"].shape == (bsz, seq_len)

    criterion = VascuMambaCompositeLoss()
    losses = criterion(
        out,
        {
            "valid_mask": valid_mask,
            "pos_mask": pos_mask,
            "apex_target": apex_target,
            "boundary_target": boundary_target,
            "cls_target": cls_target,
            "biopac_arousal": torch.rand(bsz, seq_len),
        },
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])


def test_ssl_and_distillation_losses() -> None:
    """Verify SSL reconstruction + contrastive loss and BIOPAC distillation loss."""
    ssl = NeuroVascularSSLLoss()
    mask = torch.zeros(2, 20, dtype=torch.bool)
    mask[:, 5:10] = True
    loss_dict = ssl(
        recon_kin=torch.randn(2, 20, 144, requires_grad=True),
        target_kin=torch.randn(2, 20, 144),
        mask_kin=mask,
        recon_phys=torch.randn(2, 20, 36, requires_grad=True),
        target_phys=torch.randn(2, 20, 36),
        mask_phys=mask,
        z_kin=torch.randn(2, 20, 64, requires_grad=True),
        z_phys=torch.randn(2, 20, 64, requires_grad=True),
    )
    loss_dict["loss"].backward()
    assert torch.isfinite(loss_dict["loss"])

    distill = PrivilegedBiopacDistillLoss()
    d_loss = distill(torch.rand(2, 20, requires_grad=True), torch.rand(2, 20))
    d_loss.backward()
    assert torch.isfinite(d_loss)


def test_baselines_registry_smoke() -> None:
    """Verify every registered baseline instantiates and runs a forward pass cleanly."""
    x_kin = torch.randn(2, 32, 144)
    x_phys = torch.randn(2, 32, 36)
    for name in list_baselines():
        m = build_baseline(name, num_classes=3)
        if name == "optical_strain_peak":
            intervals = m.predict(x_kin[0].numpy(), fps=50.0)
            assert isinstance(intervals, list)
        else:
            out = m(x_kin=x_kin, x_phys=x_phys)
            assert "clip_cls_logits" in out
