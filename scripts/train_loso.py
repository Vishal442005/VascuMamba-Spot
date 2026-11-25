"""Subject-disjoint LOSO training and evaluation script for VascuMamba-Spot and baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vascumamba_spot.baselines.registry import build_baseline, list_baselines
from vascumamba_spot.data.manifests import generate_loso_folds, load_manifest
from vascumamba_spot.engine import run_loso_fold
from vascumamba_spot.models import VascuMambaConfig, VascuMambaSpot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full subject-disjoint LOSO evaluation.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--model", default="vascumamba_base", choices=["vascumamba_base", *list_baselines()])
    parser.add_argument("--emotion-scheme", default="5cls", choices=["3cls", "5cls"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--ssl-ckpt", type=Path, default=None)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    folds = generate_loso_folds(records)
    num_classes = 3 if args.emotion_scheme == "3cls" else 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_model() -> torch.nn.Module:
        if args.model == "vascumamba_base":
            m = VascuMambaSpot(VascuMambaConfig(num_classes=num_classes))
        else:
            m = build_baseline(args.model, num_classes=num_classes)
        if args.ssl_ckpt is not None and isinstance(m, VascuMambaSpot):
            ckpt = torch.load(args.ssl_ckpt, map_location="cpu")
            m.load_state_dict(ckpt["model"], strict=False)
        return m

    seed_summaries: list[dict[str, float]] = []
    per_fold_all: dict[int, list[dict[str, float]]] = {}

    for seed in args.seeds:
        fold_metrics: list[dict[str, float]] = []
        for fold in folds:
            res = run_loso_fold(
                fold=fold,
                cache_path=args.cache,
                model_factory=make_model,
                emotion_scheme=args.emotion_scheme,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=seed,
                device=device,
            )
            fold_metrics.append({"subject": fold.test_subject, **res})
        per_fold_all[seed] = fold_metrics

        keys = [k for k in fold_metrics[0].keys() if k != "subject"]
        summary = {k: float(np.mean([f[k] for f in fold_metrics])) for k in keys}
        seed_summaries.append(summary)

    metric_keys = list(seed_summaries[0].keys())
    agg = {
        k: {
            "mean": float(np.mean([s[k] for s in seed_summaries])),
            "std": float(np.std([s[k] for s in seed_summaries])),
        }
        for k in metric_keys
    }
    payload = {
        "model": args.model,
        "emotion_scheme": args.emotion_scheme,
        "num_folds": len(folds),
        "seeds": args.seeds,
        "aggregate": agg,
        "per_fold": per_fold_all,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"LOSO complete -> {args.out_json} | F1_ME={agg['f1_me']['mean']:.4f} | UF1={agg['uf1']['mean']:.4f}")


if __name__ == "__main__":
    main()
