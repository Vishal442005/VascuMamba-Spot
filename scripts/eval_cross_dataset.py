"""Track D: Cross-dataset and cross-frame-rate zero-shot generalization evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vascumamba_spot.data.cache_dataset import CachedFeatureDataset, collate_variable_sequences
from vascumamba_spot.data.manifests import load_manifest
from vascumamba_spot.engine import evaluate_fold, set_deterministic_seed, train_one_epoch
from vascumamba_spot.losses import VascuMambaCompositeLoss
from vascumamba_spot.models import VascuMambaConfig, VascuMambaSpot


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset zero-shot transfer evaluation.")
    parser.add_argument("--src-manifest", required=True, type=Path)
    parser.add_argument("--src-cache", required=True, type=Path)
    parser.add_argument("--tgt-manifest", required=True, type=Path)
    parser.add_argument("--tgt-cache", required=True, type=Path)
    parser.add_argument("--emotion-scheme", default="3cls", choices=["3cls", "5cls"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    set_deterministic_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_recs = load_manifest(args.src_manifest)
    tgt_recs = load_manifest(args.tgt_manifest)
    num_classes = 3 if args.emotion_scheme == "3cls" else 5

    train_ds = CachedFeatureDataset(src_recs, args.src_cache, emotion_scheme=args.emotion_scheme, training=True)
    test_ds = CachedFeatureDataset(tgt_recs, args.tgt_cache, emotion_scheme=args.emotion_scheme, training=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_variable_sequences)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_variable_sequences)

    model = VascuMambaSpot(VascuMambaConfig(num_classes=num_classes)).to(device)
    criterion = VascuMambaCompositeLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    for _ in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)

    metrics = evaluate_fold(model, test_loader, device, train_ds.emotion_map)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"cross_dataset_metrics": metrics}, indent=2), encoding="utf-8")
    print(f"Cross-dataset eval complete -> {args.out_json} | F1_ME={metrics['f1_me']:.4f} UF1={metrics['uf1']:.4f}")


if __name__ == "__main__":
    main()
