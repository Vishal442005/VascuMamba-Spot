"""CLI script to parse raw dataset folders into canonical JSONL manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from vascumamba_spot.data.manifests import generate_loso_folds, save_manifest
from vascumamba_spot.data.parsers import (
    build_composite_megc_3cls,
    parse_casme2,
    parse_casme3_unlabeled,
    parse_mmew,
    parse_samm,
    parse_smic,
    parse_untrimmed_spreadsheet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VascuMamba-Spot dataset manifests.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "casme2",
            "casme_sq",
            "casme3",
            "casme3_unlabeled",
            "samm",
            "samm_lv",
            "smic_hs",
            "smic_vis",
            "smic_nir",
            "mmew",
            "megc",
        ],
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--extra-manifests", nargs="*", type=Path, default=[])
    args = parser.parse_args()

    if args.dataset == "casme2":
        records = parse_casme2(args.data_root)
    elif args.dataset == "casme_sq":
        records = parse_untrimmed_spreadsheet(args.data_root, "casme_sq", "*.xlsx", fps=30.0, color_mode="rgb")
    elif args.dataset == "casme3":
        records = parse_untrimmed_spreadsheet(args.data_root, "casme3", "*.xlsx", fps=30.0, color_mode="rgb")
    elif args.dataset == "casme3_unlabeled":
        records = parse_casme3_unlabeled(args.data_root, fps=30.0)
    elif args.dataset == "samm":
        records = parse_samm(args.data_root)
    elif args.dataset == "samm_lv":
        records = parse_untrimmed_spreadsheet(args.data_root, "samm_lv", "*.xlsx", fps=200.0, color_mode="mono")
    elif args.dataset.startswith("smic_"):
        records = parse_smic(args.data_root, sensor=args.dataset.split("_")[1])
    elif args.dataset == "mmew":
        records = parse_mmew(args.data_root)
    else:
        from vascumamba_spot.data.manifests import load_manifest

        records = build_composite_megc_3cls([load_manifest(p) for p in args.extra_manifests])

    save_manifest(records, args.out_manifest)
    folds = generate_loso_folds(records) if records and args.dataset != "casme3_unlabeled" else []
    print(f"Saved {len(records)} video records across {len(folds)} LOSO folds -> {args.out_manifest}")


if __name__ == "__main__":
    main()
