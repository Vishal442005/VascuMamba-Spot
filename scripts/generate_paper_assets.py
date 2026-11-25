"""Automated zero-copy LaTeX table and stress-figure generator from experiment JSON logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vascumamba_spot.evaluation.statistics import holm_bonferroni_adjust, paired_wilcoxon_test
from vascumamba_spot.visualization.plots import set_pub_style


def generate_ablation_tex(ablation_json: Path, out_tex: Path) -> None:
    """Format ABL-1..ABL-8 results with Holm-Bonferroni Wilcoxon significance into LaTeX."""
    data = json.loads(ablation_json.read_text(encoding="utf-8"))
    full_scores = [f["f1_me"] for f in data["Full_VascuMamba"]["folds"]]

    variants = [k for k in data.keys() if k != "Full_VascuMamba"]
    raw_p = [
        paired_wilcoxon_test(full_scores, [f["f1_me"] for f in data[v]["folds"]])["p_value"]
        for v in variants
    ]
    adj_p = dict(zip(variants, holm_bonferroni_adjust(raw_p)))

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Variant & $\text{F1}_{\text{ME}}\uparrow$ & $\text{STR-F1}\uparrow$ & $\text{UF1}\uparrow$ & $p_{\text{adj}}$ \\",
        r"\midrule",
    ]
    for name, vals in data.items():
        m = vals["mean"]
        p_str = "--" if name == "Full_VascuMamba" else f"{adj_p[name]:.3f}"
        star = "" if name == "Full_VascuMamba" else (r"$^*$" if adj_p[name] < 0.05 else "")
        clean_name = name.replace("_", r"\_")
        lines.append(
            f"{clean_name} & {m['f1_me']:.3f}{star} & {m['str_f1_me']:.3f} & {m['uf1']:.3f} & {p_str} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_comparison_tex(json_paths: list[Path], out_tex: Path, mode: str = "spotting") -> None:
    """Generate Track A/C (spotting) or Track B (recognition) comparison LaTeX table."""
    entries = [json.loads(p.read_text(encoding="utf-8")) for p in json_paths]
    if mode == "spotting":
        cols = [("f1_me", r"$\text{F1}_{\text{ME}}\uparrow$"), ("f1_mae", r"$\text{F1}_{\text{MaE}}\uparrow$"), ("str_f1_me", r"$\text{STR-F1}\uparrow$"), ("fp_me", r"$\text{FP}_{\text{ME}}\downarrow$")]
    else:
        cols = [("uf1", r"$\text{UF1}\uparrow$"), ("uar", r"$\text{UAR}\uparrow$"), ("accuracy", r"$\text{Acc}\uparrow$"), ("ece", r"$\text{ECE}\downarrow$")]

    header = "Method & " + " & ".join(lbl for _, lbl in cols) + r" \\"
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", header, r"\midrule"]

    for ent in entries:
        agg = ent["aggregate"]
        row_cells = [ent["model"].replace("_", r"\_")]
        for k, _ in cols:
            m = agg[k]["mean"]
            s = agg[k]["std"]
            row_cells.append(f"{m:.3f} $\\pm$ {s:.3f}")
        lines.append(" & ".join(row_cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_stress_figure(stress_json: Path, out_fig: Path) -> None:
    """Plot F1_ME and UF1 robustness across increasing vascular SNR noise levels."""
    set_pub_style()
    data = json.loads(stress_json.read_text(encoding="utf-8"))["stress_rppg_noise"]
    sigmas = [float(k.split("_")[1]) for k in data.keys()]
    f1s = [v["f1_me"] for v in data.values()]
    uf1s = [v["uf1"] for v in data.values()]

    fig, ax = plt.subplots(figsize=(4.5, 3.0), constrained_layout=True)
    ax.plot(sigmas, f1s, marker="o", color="#0072B2", lw=1.8, label=r"$\mathrm{F1}_{\mathrm{ME}}$")
    ax.plot(sigmas, uf1s, marker="s", color="#009E73", lw=1.8, label=r"$\mathrm{UF1}$")
    ax.set_xlabel(r"Injected rPPG Noise $\sigma$ (Compression Proxy)")
    ax.set_ylabel("Score")
    ax.set_title("Stress Robustness Under Vascular Signal Degradation")
    ax.legend(frameon=False)

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables & figures from JSON logs.")
    parser.add_argument("--ablation-json", type=Path, default=None)
    parser.add_argument("--spotting-jsons", nargs="*", type=Path, default=[])
    parser.add_argument("--recognition-jsons", nargs="*", type=Path, default=[])
    parser.add_argument("--stress-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("paper"))
    args = parser.parse_args()

    if args.ablation_json:
        generate_ablation_tex(args.ablation_json, args.out_dir / "tables" / "ablation_table.tex")
    if args.spotting_jsons:
        generate_comparison_tex(args.spotting_jsons, args.out_dir / "tables" / "spotting_table.tex", mode="spotting")
    if args.recognition_jsons:
        generate_comparison_tex(args.recognition_jsons, args.out_dir / "tables" / "recognition_table.tex", mode="recognition")
    if args.stress_json:
        generate_stress_figure(args.stress_json, args.out_dir / "figures" / "fig5_stress.pdf")
    print(f"Paper assets updated under {args.out_dir}")


if __name__ == "__main__":
    main()
