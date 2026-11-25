"""Tests for extended dataset parsers, BIOPAC teacher sync, and paper asset generators."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.generate_paper_assets import (
    generate_ablation_tex,
    generate_comparison_tex,
    generate_stress_figure,
)
from vascumamba_spot.data.parsers import parse_mmew, parse_samm, parse_smic
from vascumamba_spot.models import VascuMambaConfig, VascuMambaSpot
from vascumamba_spot.preprocessing.biopac_sync import (
    audit_rppg_against_biopac,
    compute_biopac_arousal_teacher,
    load_biopac_signals,
)
from vascumamba_spot.visualization.plots import plot_confusion_and_calibration, plot_fp_breakdown


def test_extended_parsers_samm_smic_mmew(tmp_path: Path) -> None:
    """Verify SAMM, SMIC (HS/VIS/NIR), and MMEW parsers produce accurate color_mode & fps."""
    # SMIC-NIR mock structure
    smic_clip = tmp_path / "smic" / "NIR" / "s01" / "micro" / "negative" / "clip1"
    smic_clip.mkdir(parents=True)
    for i in range(5):
        (smic_clip / f"img_{i:03d}.png").write_bytes(b"fake")
    smic_recs = parse_smic(tmp_path / "smic", sensor="nir")
    assert len(smic_recs) == 1
    assert smic_recs[0].color_mode == "nir" and smic_recs[0].fps == 25.0 and not smic_recs[0].has_rgb

    # SAMM mock structure
    samm_dir = tmp_path / "samm"
    clip_samm = samm_dir / "006" / "006_1_2"
    clip_samm.mkdir(parents=True)
    for i in range(6):
        (clip_samm / f"006_{i:04d}.jpg").write_bytes(b"fake")
    pd.DataFrame(
        [
            {
                "Subject": 6,
                "Filename": "006_1_2",
                "Onset Frame": 1,
                "Apex Frame": 3,
                "Offset Frame": 5,
                "Estimated Emotion": "Contempt",
                "Action Units": "12+14",
            }
        ]
    ).to_csv(samm_dir / "SAMM_Micro_FACS_Codes_v2.csv", index=False)
    samm_recs = parse_samm(samm_dir)
    assert len(samm_recs) == 1 and samm_recs[0].color_mode == "mono" and not samm_recs[0].has_rgb

    # MMEW mock structure
    mmew_dir = tmp_path / "mmew"
    clip_mmew = mmew_dir / "Micro_Expression" / "happiness" / "ep1"
    clip_mmew.mkdir(parents=True)
    for i in range(4):
        (clip_mmew / f"{i}.jpg").write_bytes(b"fake")
    pd.DataFrame(
        [
            {
                "Subject": "S1",
                "Filename": "ep1",
                "OnsetFrame": 1,
                "ApexFrame": 2,
                "OffsetFrame": 4,
                "Estimated Emotion": "happiness",
            }
        ]
    ).to_csv(mmew_dir / "MMEW_Micro.csv", index=False)
    mmew_recs = parse_mmew(mmew_dir)
    assert len(mmew_recs) == 1 and mmew_recs[0].color_mode == "rgb" and mmew_recs[0].fps == 90.0


def test_biopac_load_teacher_and_audit(tmp_path: Path) -> None:
    """Verify BIOPAC CSV loading, normalized arousal teacher extraction, and rPPG audit."""
    fps = 30.0
    t_vid = 120
    fs_bio = 1000.0
    t_bio = np.arange(int((t_vid / fps) * fs_bio), dtype=np.float32) / fs_bio
    ppg = np.sin(2.0 * np.pi * 1.2 * t_bio)
    eda = np.linspace(1.0, 2.0, t_bio.size, dtype=np.float32)
    arr = np.stack([ppg, ppg, eda, ppg], axis=1)
    csv_path = tmp_path / "biopac.csv"
    np.savetxt(csv_path, arr, delimiter=",")

    sigs = load_biopac_signals(csv_path)
    teacher = compute_biopac_arousal_teacher(sigs, num_video_frames=t_vid, fps=fps)
    assert teacher.shape == (t_vid,)
    assert float(teacher.min()) >= 0.0 and float(teacher.max()) <= 1.0

    t_v = np.arange(t_vid, dtype=np.float32) / fps
    rppg_bvp = np.sin(2.0 * np.pi * 1.2 * t_v)
    env = np.ones_like(rppg_bvp)
    env[40:55] = 2.0
    me_mask = np.zeros(t_vid, dtype=bool)
    me_mask[40:55] = True
    res = audit_rppg_against_biopac(rppg_bvp, env, sigs["ppg"], fps=fps, me_frame_mask=me_mask)
    assert res.hr_abs_err_bpm < 5.0
    assert res.me_vs_non_me_auc > 0.9


def test_paper_asset_generators(tmp_path: Path) -> None:
    """Verify automated LaTeX table and stress PDF figure generation."""
    abl_json = tmp_path / "ablations.json"
    abl_json.write_text(
        json.dumps(
            {
                "Full_VascuMamba": {
                    "mean": {"f1_me": 0.45, "str_f1_me": 0.32, "uf1": 0.78},
                    "folds": [{"f1_me": 0.45}, {"f1_me": 0.46}, {"f1_me": 0.44}, {"f1_me": 0.45}, {"f1_me": 0.47}],
                },
                "ABL1_Kinematic_Only": {
                    "mean": {"f1_me": 0.34, "str_f1_me": 0.22, "uf1": 0.69},
                    "folds": [{"f1_me": 0.33}, {"f1_me": 0.35}, {"f1_me": 0.34}, {"f1_me": 0.33}, {"f1_me": 0.35}],
                },
            }
        ),
        encoding="utf-8",
    )
    out_abl = tmp_path / "ablation.tex"
    generate_ablation_tex(abl_json, out_abl)
    assert r"\toprule" in out_abl.read_text(encoding="utf-8")

    comp_json = tmp_path / "comp.json"
    comp_json.write_text(
        json.dumps(
            {
                "model": "vascumamba_base",
                "aggregate": {
                    "f1_me": {"mean": 0.45, "std": 0.02},
                    "f1_mae": {"mean": 0.61, "std": 0.03},
                    "str_f1_me": {"mean": 0.32, "std": 0.02},
                    "fp_me": {"mean": 3.0, "std": 1.0},
                },
            }
        ),
        encoding="utf-8",
    )
    out_spot = tmp_path / "spot.tex"
    generate_comparison_tex([comp_json], out_spot, mode="spotting")
    assert r"vascumamba\_base" in out_spot.read_text(encoding="utf-8")

    stress_json = tmp_path / "stress.json"
    stress_json.write_text(
        json.dumps(
            {
                "stress_rppg_noise": {
                    "sigma_0.00": {"f1_me": 0.45, "uf1": 0.78},
                    "sigma_0.50": {"f1_me": 0.42, "uf1": 0.75},
                }
            }
        ),
        encoding="utf-8",
    )
    out_fig = tmp_path / "stress.pdf"
    generate_stress_figure(stress_json, out_fig)
    assert out_fig.exists() and out_fig.stat().st_size > 1000

    # Verify ABL-7 global_phys_only mode and confusion/calibration/FP plots
    import torch

    m_abl7 = VascuMambaSpot(VascuMambaConfig(d_model=32, num_layers=1, global_phys_only=True))
    out_abl7 = m_abl7(torch.randn(1, 20, 144), torch.randn(1, 20, 36))
    assert out_abl7["apex_prob"].shape == (1, 20)

    cm_fig = plot_confusion_and_calibration(
        y_true=np.array([0, 1, 2, 0, 1]),
        probs=np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.7, 0.2, 0.1], [0.1, 0.7, 0.2]]),
        class_names=["Neg", "Pos", "Surp"],
        out_path=tmp_path / "cm_cal.pdf",
    )
    fp_fig = plot_fp_breakdown(["Blinks", "Pose", "Speech"], [3.1, 1.4, 2.0], [0.4, 0.5, 0.8], tmp_path / "fp.pdf")
    assert cm_fig.exists() and fp_fig.exists()
