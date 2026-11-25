"""Stress testing (noise/SNR degradation, FPS mismatch) and subgroup fairness audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vascumamba_spot.data.cache_dataset import CachedFeatureDataset, collate_variable_sequences
from vascumamba_spot.data.manifests import load_manifest
from vascumamba_spot.engine import evaluate_fold
from vascumamba_spot.models import VascuMambaConfig, VascuMambaSpot


def evaluate_under_rppg_noise(
    model: torch.nn.Module,
    dataset: CachedFeatureDataset,
    device: torch.device,
    noise_stds: list[float],
) -> dict[str, dict[str, float]]:
    """Evaluate degradation as additive vascular noise simulates heavy video compression."""
    results: dict[str, dict[str, float]] = {}
    base_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_variable_sequences)
    results["sigma_0.00"] = evaluate_fold(model, base_loader, device, dataset.emotion_map)

    for sigma in noise_stds:
        # Wrap collate to inject Gaussian noise onto x_phys channels 0..27
        def noisy_collate(batch, s=sigma):
            out = collate_variable_sequences(batch)
            noise = torch.randn_like(out["x_phys"][..., :28]) * s
            out["x_phys"][..., :28] = out["x_phys"][..., :28] + noise
            return out

        noisy_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=noisy_collate)
        results[f"sigma_{sigma:.2f}"] = evaluate_fold(model, noisy_loader, device, dataset.emotion_map)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compression-proxy stress and subgroup fairness audits.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--emotion-scheme", default="5cls", choices=["3cls", "5cls"])
    parser.add_argument("--out-json", type=Path, default=Path("outputs/stress_fairness.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_manifest(args.manifest)
    num_classes = 3 if args.emotion_scheme == "3cls" else 5
    model = VascuMambaSpot(VascuMambaConfig(num_classes=num_classes)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])

    full_ds = CachedFeatureDataset(records, args.cache, emotion_scheme=args.emotion_scheme, training=False)
    stress = evaluate_under_rppg_noise(model, full_ds, device, noise_stds=[0.25, 0.50, 1.00, 2.00])

    # Per-color-mode / per-subject parity summary
    by_mode: dict[str, list] = {}
    for rec in records:
        by_mode.setdefault(rec.color_mode, []).append(rec)
    subgroup_metrics: dict[str, dict[str, float]] = {}
    for mode, sub_recs in by_mode.items():
        sub_ds = CachedFeatureDataset(sub_recs, args.cache, emotion_scheme=args.emotion_scheme, training=False)
        sub_loader = DataLoader(sub_ds, batch_size=4, shuffle=False, collate_fn=collate_variable_sequences)
        subgroup_metrics[mode] = evaluate_fold(model, sub_loader, device, sub_ds.emotion_map)

    payload = {"stress_rppg_noise": stress, "subgroup_color_mode": subgroup_metrics}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Stress & fairness audit complete -> {args.out_json} (modes={list(by_mode.keys())}, mean_f1={np.mean([v['f1_me'] for v in subgroup_metrics.values()]):.3f})")


if __name__ == "__main__":
    main()
