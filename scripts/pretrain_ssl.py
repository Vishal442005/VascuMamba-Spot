"""Self-supervised 4M-frame neuro-vascular pretraining script (CAS(ME)^3 unlabeled)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vascumamba_spot.data.cache_dataset import CachedFeatureDataset, collate_variable_sequences
from vascumamba_spot.data.manifests import load_manifest
from vascumamba_spot.engine import pretrain_ssl_epoch, set_deterministic_seed
from vascumamba_spot.losses.distill_ssl import NeuroVascularSSLLoss
from vascumamba_spot.models import VascuMambaConfig, VascuMambaSpot


def main() -> None:
    parser = argparse.ArgumentParser(description="SSL pretraining on unlabeled facial videos.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out-ckpt", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_deterministic_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_manifest(args.manifest)
    ds = CachedFeatureDataset(records, args.cache, training=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_variable_sequences)

    model = VascuMambaSpot(VascuMambaConfig()).to(device)
    ssl_loss = NeuroVascularSSLLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    for epoch in range(1, args.epochs + 1):
        stats = pretrain_ssl_epoch(model, loader, optimizer, ssl_loss, device)
        print(f"[SSL Epoch {epoch:03d}] loss={stats['loss']:.4f}")

    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.out_ckpt)
    print(f"Saved SSL checkpoint -> {args.out_ckpt}")


if __name__ == "__main__":
    main()
