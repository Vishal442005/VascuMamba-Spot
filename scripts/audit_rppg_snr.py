"""Phase I Verification Gate: audit rPPG HR SNR (>8 dB) and optional BIOPAC correlation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from vascumamba_spot.data.manifests import load_manifest
from vascumamba_spot.preprocessing.biopac_sync import audit_rppg_against_biopac
from vascumamba_spot.preprocessing.rppg_vascular import compute_hr_snr_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase I rPPG SNR & BIOPAC ground-truth audit.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--snr-threshold-db", type=float, default=8.0)
    parser.add_argument("--out-report", type=Path, default=Path("outputs/phase1_rppg_audit.json"))
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    rgb_snrs: list[float] = []
    biopac_corrs: list[float] = []
    biopac_aucs: list[float] = []

    with h5py.File(args.cache, "r") as h5f:
        for rec in records:
            if rec.video_id not in h5f or not rec.has_rgb:
                continue
            grp = h5f[rec.video_id]
            x_phys = grp["x_phys"][:]
            fps = float(grp.attrs["fps"])
            # Average normalized BVP across 7 ROIs (channels 0, 4, 8, ..., 24) and envelope (1, 5, ...)
            bvp = x_phys[:, 0:28:4].mean(axis=1)
            env = x_phys[:, 1:28:4].mean(axis=1)
            _, snr_db = compute_hr_snr_db(bvp, fps=fps)
            rgb_snrs.append(snr_db)

            if "biopac_ppg" in grp:
                me_mask = np.zeros(len(bvp), dtype=bool)
                for a in rec.annotations:
                    if a.expr_type == "ME":
                        me_mask[a.onset : min(len(bvp), a.offset + 1)] = True
                res = audit_rppg_against_biopac(bvp, env, grp["biopac_ppg"][:], fps=fps, me_frame_mask=me_mask)
                biopac_corrs.append(res.envelope_pearson_r)
                biopac_aucs.append(res.me_vs_non_me_auc)

    mean_snr = float(np.mean(rgb_snrs)) if rgb_snrs else -30.0
    report = {
        "num_rgb_videos": len(rgb_snrs),
        "mean_snr_db": mean_snr,
        "passes_snr_gate": mean_snr >= args.snr_threshold_db,
        "mean_biopac_r": float(np.mean(biopac_corrs)) if biopac_corrs else None,
        "mean_me_vs_non_me_auc": float(np.mean(biopac_aucs)) if biopac_aucs else None,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
