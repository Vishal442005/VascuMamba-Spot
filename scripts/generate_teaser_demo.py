"""Generate Figure 1 teaser comparing Kinematic-Only vs VascuMamba-Spot blink suppression."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vascumamba_spot.data.cache_dataset import build_frame_targets
from vascumamba_spot.engine import set_deterministic_seed
from vascumamba_spot.losses import VascuMambaCompositeLoss
from vascumamba_spot.models import VascuMambaConfig, VascuMambaSpot
from vascumamba_spot.visualization.plots import set_pub_style


def make_synthetic_batch(
    rng: np.random.Generator,
    batch_size: int = 8,
    seq_len: int = 120,
    fps: float = 50.0,
) -> dict[str, torch.Tensor]:
    """Create synthetic sequences with non-emotional blinks and aroused MEs."""
    x_kin = rng.normal(0.0, 0.08, size=(batch_size, seq_len, 144)).astype(np.float32)
    x_phys = rng.normal(0.0, 0.08, size=(batch_size, seq_len, 36)).astype(np.float32)
    x_phys[:, :, -1] = 1.0

    apex_targets = []
    bound_targets = []
    pos_masks = []
    cls_targets = []

    for b in range(batch_size):
        blink_s = int(rng.integers(15, 35))
        blink_e = blink_s + 8
        # Eyeblink: strong kinematic strain, baseline vascular activity
        x_kin[b, blink_s : blink_e + 1, :] += 2.2

        me_s = int(rng.integers(65, 85))
        me_ap = me_s + 6
        me_e = me_s + 12
        # True suppressed ME: strong kinematic strain AND autonomic pulse modulation
        x_kin[b, me_s : me_e + 1, :] += 2.0
        x_phys[b, me_s - 4 : me_e + 5, :28] += 1.6

        targs = build_frame_targets(seq_len, fps=fps, intervals=[(me_s, me_ap, me_e, 1)])
        apex_targets.append(targs["apex_target"])
        bound_targets.append(targs["boundary_target"])
        pos_masks.append(targs["pos_mask"])
        cls_targets.append(targs["cls_target"])

    return {
        "x_kin": torch.from_numpy(x_kin),
        "x_phys": torch.from_numpy(x_phys),
        "has_rgb": torch.ones(batch_size),
        "valid_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "apex_target": torch.stack(apex_targets),
        "boundary_target": torch.stack(bound_targets),
        "pos_mask": torch.stack(pos_masks),
        "cls_target": torch.stack(cls_targets),
    }


def train_quick(model: VascuMambaSpot, batch: dict[str, torch.Tensor], steps: int = 45) -> None:
    crit = VascuMambaCompositeLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    model.train()
    for _ in range(steps):
        out = model(batch["x_kin"], batch["x_phys"], has_rgb=batch["has_rgb"], valid_mask=batch["valid_mask"])
        loss = crit(out, batch)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 1 blink suppression teaser.")
    parser.add_argument("--out-fig", type=Path, default=Path("paper/figures/fig1_teaser.pdf"))
    args = parser.parse_args()

    set_deterministic_seed(42)
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    train_batch = make_synthetic_batch(rng, batch_size=8, seq_len=120)

    kin_only = VascuMambaSpot(VascuMambaConfig(d_model=32, num_layers=2, kinematic_only=True, use_autonomic_gate=False))
    vascu = VascuMambaSpot(VascuMambaConfig(d_model=32, num_layers=2))
    train_quick(kin_only, train_batch)
    train_quick(vascu, train_batch)

    # Deterministic test sequence: blink at [0.50s, 0.66s], true ME at [1.50s, 1.74s]
    seq_len, fps = 120, 50.0
    xk = np.random.default_rng(99).normal(0.0, 0.06, size=(1, seq_len, 144)).astype(np.float32)
    xp = np.random.default_rng(99).normal(0.0, 0.06, size=(1, seq_len, 36)).astype(np.float32)
    xp[:, :, -1] = 1.0
    xk[0, 25:34, :] += 2.2  # blink
    xk[0, 75:88, :] += 2.0  # true ME
    xp[0, 71:92, :28] += 1.6  # autonomic arousal

    kin_only.eval()
    vascu.eval()
    with torch.no_grad():
        out_k = kin_only(torch.from_numpy(xk), torch.from_numpy(xp))
        out_v = vascu(torch.from_numpy(xk), torch.from_numpy(xp))

    t = np.arange(seq_len, dtype=np.float32) / fps
    strain = xk[0].mean(axis=1)
    rppg = xp[0, :, :28].mean(axis=1)
    p_kin = out_k["apex_prob"].squeeze(0).numpy()
    p_vascu = out_v["apex_prob"].squeeze(0).numpy()
    gate = out_v["gate_mean"].squeeze(0).numpy()

    set_pub_style()
    fig, axes = plt.subplots(4, 1, figsize=(7.5, 5.2), sharex=True, constrained_layout=True)
    for ax in axes:
        ax.axvspan(0.50, 0.66, color="#D55E00", alpha=0.14)
        ax.axvspan(1.50, 1.74, color="#009E73", alpha=0.16)

    axes[0].plot(t, strain, color="#0072B2", lw=1.3)
    axes[0].set_ylabel("Optical Strain")
    axes[0].set_title("Neuro-Vascular Gating Eliminates Non-Emotional Eyeblink False Alarms (Synthetic Verification)")
    axes[0].text(0.58, strain.max() * 0.85, "Eyeblink\n(No Arousal)", ha="center", fontsize=7.5, color="#A04000")
    axes[0].text(1.62, strain.max() * 0.85, "True ME\n(Aroused)", ha="center", fontsize=7.5, color="#006B4E")

    axes[1].plot(t, rppg, color="#CC79A7", lw=1.2)
    axes[1].set_ylabel("rPPG Signal")

    axes[2].plot(t, gate, color="#009E73", lw=1.4)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel(r"Gate $g_t^{\mathrm{aut}}$")

    axes[3].plot(t, p_kin, color="#D55E00", ls="--", lw=1.4, label="Kinematic-Only Baseline")
    axes[3].plot(t, p_vascu, color="#000000", lw=1.5, label="VascuMamba-Spot (Ours)")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_ylabel(r"Apex $\hat{p}_t$")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper left", frameon=False)

    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig)
    plt.close(fig)
    print(f"Saved teaser figure -> {args.out_fig} (blink max p_kin={p_kin[25:34].max():.3f}, p_vascu={p_vascu[25:34].max():.3f})")


if __name__ == "__main__":
    main()
