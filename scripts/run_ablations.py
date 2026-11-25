"""Systematic ablation runner covering ABL-1 through ABL-10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vascumamba_spot.data.manifests import generate_loso_folds, load_manifest
from vascumamba_spot.engine import run_loso_fold
from vascumamba_spot.models import VascuMambaConfig, VascuMambaSpot

ABLATION_SPECS: dict[str, dict[str, Any]] = {
    "Full_VascuMamba": {"model": {}, "loss": {}},
    "ABL1_Kinematic_Only": {"model": {"kinematic_only": True, "use_autonomic_gate": False}, "loss": {}},
    "ABL2_Vascular_Only": {"model": {"vascular_only": True, "use_autonomic_gate": False}, "loss": {}},
    "ABL3a_Additive_Gate": {"model": {"gating_mode": "additive"}, "loss": {}},
    "ABL3b_Concat_Fusion": {"model": {"gating_mode": "concat"}, "loss": {}},
    "ABL4_Unidirectional": {"model": {"bidirectional": False}, "loss": {}},
    "ABL5_No_InfoNCE": {"model": {}, "loss": {"lambda_nce": 0.0}},
    "ABL6a_Transformer_MHSA": {"model": {"backbone_type": "transformer"}, "loss": {}},
    "ABL6b_BiGRU": {"model": {"backbone_type": "bigru"}, "loss": {}},
    "ABL7_Global_Face_ROI": {"model": {"global_phys_only": True}, "loss": {}},
    "ABL8_No_MNC_Filter": {"model": {"use_mnc_filter": False}, "loss": {}},
    "ABL10_No_BIOPAC_Distill": {"model": {}, "loss": {"lambda_distill": 0.0}},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run systematic ablations ABL-1..ABL-10 under LOSO.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--emotion-scheme", default="5cls", choices=["3cls", "5cls"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ssl-ckpt", type=Path, default=None, help="Optional SSL checkpoint for ABL-9.")
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    folds = generate_loso_folds(records)
    num_classes = 3 if args.emotion_scheme == "3cls" else 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = dict(ABLATION_SPECS)
    if args.ssl_ckpt is not None:
        specs["ABL9_SSL_Pretrained"] = {"model": {}, "loss": {}, "ssl_ckpt": args.ssl_ckpt}

    results: dict[str, dict] = {}
    for abl_name, spec in specs.items():
        cfg = VascuMambaConfig(num_classes=num_classes, **spec["model"])
        loss_kw = spec["loss"]
        ckpt_path = spec.get("ssl_ckpt")

        def make_model(c=cfg, cp=ckpt_path):
            m = VascuMambaSpot(c)
            if cp is not None:
                m.load_state_dict(torch.load(cp, map_location="cpu")["model"], strict=False)
            return m

        fold_scores = [
            run_loso_fold(
                fold=f,
                cache_path=args.cache,
                model_factory=make_model,
                emotion_scheme=args.emotion_scheme,
                epochs=args.epochs,
                seed=args.seed,
                device=device,
                **loss_kw,
            )
            for f in folds
        ]
        keys = list(fold_scores[0].keys())
        results[abl_name] = {
            "mean": {k: float(np.mean([fs[k] for fs in fold_scores])) for k in keys},
            "folds": fold_scores,
        }
        print(f"[{abl_name}] F1_ME={results[abl_name]['mean']['f1_me']:.4f} UF1={results[abl_name]['mean']['uf1']:.4f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
