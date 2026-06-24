#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from shamdose.config import load_config, require_config_key


COHORTS = ["HC", "SZ", "CUD"]

CONDITIONS = [
    "MagVenture_Sham0mm",
    "MagVenture_Sham0mm+electrodes",
    "Magstim_Sham0mm_kernel",
]

COND_LABELS = {
    "MagVenture_Sham0mm": "MV sham",
    "MagVenture_Sham0mm+electrodes": "MV sham + elec",
    "Magstim_Sham0mm_kernel": "Magstim sham",
}

COND_COLORS = {
    "MagVenture_Sham0mm": "#1F77B4",
    "MagVenture_Sham0mm+electrodes": "#D55E00",
    "Magstim_Sham0mm_kernel": "#009E73",
}

PRIMARY_ENDPOINT = "PC_dPEAK_Hz"
SECONDARY_ENDPOINT = "PC_RMS_Hz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create supplementary manuscript figures from cleaned analysis outputs."
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "paths_local.yaml"),
        help="Path to private local config file.",
    )

    parser.add_argument(
        "--figure",
        default="all",
        choices=["all", "kernel", "rms", "csf", "forest", "qctraces"],
        help="Which supplementary figure to make.",
    )

    return parser.parse_args()


def format_p(p: float) -> str:
    if pd.isna(p):
        return "P = NA"
    p = float(p)
    if p < 0.001:
        return "P < 0.001"
    return f"P = {p:.3f}"


def copy_checked(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source figure: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[OK] {src} -> {dst}")


def make_supp_fig01_kernel_validation(supp_figures_dir: Path) -> None:
    """
    Supplementary Figure 1:
    Smith-Peterchev empirical coil-kernel validation.
    """
    src_png = supp_figures_dir / "magstim_empirical_kernel_map.png"
    src_transparent = supp_figures_dir / "magstim_empirical_kernel_map_transparent.png"

    dst_png = supp_figures_dir / "supp_fig01_empirical_coil_kernel_validation.png"
    dst_transparent = supp_figures_dir / "supp_fig01_empirical_coil_kernel_validation_transparent.png"

    copy_checked(src_png, dst_png)
    copy_checked(src_transparent, dst_transparent)

    print("\n[SUPPLEMENTARY FIGURE 1]")
    print("Empirical coil-kernel validation")
    print(f"Standard PNG:      {dst_png}")
    print(f"Transparent PNG:   {dst_transparent}")


def make_supp_fig02_rms_endpoint(tables_dir: Path, supp_figures_dir: Path) -> None:
    """
    Supplementary Figure 2:
    RMS endpoint crossover heatmaps.
    """
    p = tables_dir / "crossover_probability_by_intensity.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing crossover probability table: {p}")

    prob = pd.read_csv(p)
    sub = prob[prob["endpoint"].astype(str) == SECONDARY_ENDPOINT].copy()

    if sub.empty:
        raise ValueError("No PC_RMS_Hz rows found in crossover probability table.")

    intensities = sorted(sub["intensity_MT"].astype(int).unique().tolist())

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.5, 4.0),
        sharey=True,
        constrained_layout=True,
    )

    im = None

    for ax, cohort in zip(axes, COHORTS):
        csub = sub[sub["cohort"].astype(str) == cohort].copy()

        mat = []
        labels = []

        for cond in CONDITIONS:
            r = csub[csub["condition"].astype(str) == cond].copy()
            vals = []

            for intensity in intensities:
                row = r[r["intensity_MT"].astype(int) == intensity]
                if len(row) != 1:
                    raise ValueError(
                        f"{cohort} {cond} {intensity}: expected one RMS probability row, found {len(row)}"
                    )
                vals.append(float(row["pct_crossed"].iloc[0]))

            mat.append(vals)
            labels.append(COND_LABELS[cond])

        mat = np.asarray(mat)

        im = ax.imshow(
            mat,
            aspect="auto",
            vmin=0,
            vmax=100,
            cmap="YlOrRd",
        )

        ax.set_title(cohort, fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(len(intensities)))
        ax.set_xticklabels(intensities, fontsize=8)
        ax.set_xlabel("Intensity (%MT) →")

        if ax is axes[0]:
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels)
        else:
            ax.tick_params(labelleft=False)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "white" if val >= 65 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7.5, color=color)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Subjects with sham ≥ active tDCS (%)")

    out_png = supp_figures_dir / "supp_fig02_rms_endpoint_heatmaps.png"
    out_transparent = supp_figures_dir / "supp_fig02_rms_endpoint_heatmaps_transparent.png"

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_transparent, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print("\n[SUPPLEMENTARY FIGURE 2]")
    print("RMS endpoint heatmaps")
    print(f"Standard PNG:      {out_png}")
    print(f"Transparent PNG:   {out_transparent}")


def make_supp_fig03_csf_anatomy(tables_dir: Path, supp_figures_dir: Path) -> None:
    """
    Supplementary Figure 3:
    CSF thickness versus intensity where sham >= active tDCS.
    """
    crossover_path = tables_dir / "subject_level_crossover.csv"
    spearman_path = tables_dir / "anatomy_spearman_primary.csv"

    if not crossover_path.exists():
        raise FileNotFoundError(f"Missing crossover table: {crossover_path}")
    if not spearman_path.exists():
        raise FileNotFoundError(f"Missing anatomy Spearman table: {spearman_path}")

    crossover = pd.read_csv(crossover_path)
    spearman = pd.read_csv(spearman_path)

    data = crossover[crossover["endpoint"].astype(str) == PRIMARY_ENDPOINT].copy()

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12.8, 10.2),
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )

    rng = np.random.default_rng(123)

    for row_idx, cohort in enumerate(COHORTS):
        for col_idx, cond in enumerate(CONDITIONS):
            ax = axes[row_idx, col_idx]

            sub = data[
                (data["cohort"].astype(str) == cohort)
                & (data["condition"].astype(str) == cond)
            ].copy()

            x = pd.to_numeric(sub["csf_mm"], errors="coerce")
            y = pd.to_numeric(sub["I_cross_MT"], errors="coerce")
            mask = x.notna() & y.notna()

            x = x[mask]
            y = y[mask]
            y_plot = y + rng.uniform(-0.85, 0.85, size=len(y))

            ax.scatter(
                x,
                y_plot,
                s=28,
                color=COND_COLORS[cond],
                alpha=0.72,
                edgecolor="white",
                linewidth=0.4,
            )

            if len(x) >= 3 and x.nunique() > 1:
                coef = np.polyfit(x, y, 1)
                x_line = np.linspace(float(x.min()), float(x.max()), 100)
                y_line = coef[0] * x_line + coef[1]
                ax.plot(x_line, y_line, color="0.25", linewidth=1.3)

            sp = spearman[
                (spearman["cohort"].astype(str) == cohort)
                & (spearman["condition"].astype(str) == cond)
                & (spearman["endpoint"].astype(str) == PRIMARY_ENDPOINT)
                & (spearman["anatomy_variable"].astype(str) == "csf_mm")
            ].copy()

            if not sp.empty:
                s = sp.iloc[0]
                rho = float(s["rho_spearman"])
                p_holm = s.get("p_holm_primary_set", np.nan)
                text = f"ρ = {rho:.2f}\nHolm {format_p(p_holm)}"
                ax.text(
                    0.04,
                    0.96,
                    text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.85", alpha=0.9),
                )

            if row_idx == 0:
                ax.set_title(COND_LABELS[cond], fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{cohort}\nIntensity where\nsham ≥ active tDCS (%MT)")
            else:
                ax.set_ylabel("")

            if row_idx == 2:
                ax.set_xlabel("Local CSF thickness (mm) →")
            else:
                ax.set_xlabel("")

            ax.set_ylim(5, 105)
            ax.set_yticks(np.arange(10, 101, 10))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    out_png = supp_figures_dir / "supp_fig03_csf_anatomy_associations.png"
    out_transparent = supp_figures_dir / "supp_fig03_csf_anatomy_associations_transparent.png"

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_transparent, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print("\n[SUPPLEMENTARY FIGURE 3]")
    print("CSF anatomy associations")
    print(f"Standard PNG:      {out_png}")
    print(f"Transparent PNG:   {out_transparent}")


def make_supp_fig04_adjusted_or_forest(tables_dir: Path, supp_figures_dir: Path) -> None:
    """
    Supplementary Figure 4:
    Adjusted discrete-time survival-model odds-ratio forest plot.
    """
    p = tables_dir / "report_main_adjusted_anatomy_terms.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing report adjusted anatomy table: {p}")

    df = pd.read_csv(p)

    df = df[
        df["term"].isin(["z_gm_depth", "z_csf"])
        & df["condition"].isin(CONDITIONS)
    ].copy()

    if df.empty:
        raise ValueError("No adjusted anatomy rows found for forest plot.")

    term_specs = [
        ("z_gm_depth", "Scalp–GM distance"),
        ("z_csf", "Local CSF thickness"),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 7.2),
        sharey=True,
        constrained_layout=True,
    )

    y_labels = []
    for cohort in COHORTS:
        for cond in CONDITIONS:
            y_labels.append(f"{cohort} | {COND_LABELS[cond]}")

    y_positions = np.arange(len(y_labels))[::-1]

    for ax, (term, title) in zip(axes, term_specs):
        rows = []

        for cohort in COHORTS:
            for cond in CONDITIONS:
                r = df[
                    (df["cohort"].astype(str) == cohort)
                    & (df["condition"].astype(str) == cond)
                    & (df["term"].astype(str) == term)
                ].copy()

                if len(r) != 1:
                    raise ValueError(f"Expected one row for {cohort}, {cond}, {term}; found {len(r)}")

                rows.append(r.iloc[0])

        for ypos, r in zip(y_positions, rows):
            or_val = float(r["odds_ratio"])
            lo = float(r["ci95_low"])
            hi = float(r["ci95_high"])
            cond = str(r["condition"])

            ax.errorbar(
                or_val,
                ypos,
                xerr=[[or_val - lo], [hi - or_val]],
                fmt="o",
                color=COND_COLORS[cond],
                ecolor=COND_COLORS[cond],
                elinewidth=1.4,
                capsize=3,
                markersize=5.5,
            )

            p_holm = r.get("p_holm_primary_anatomy_terms", np.nan)
            if pd.notna(p_holm) and float(p_holm) < 0.05:
                ax.text(
                    min(9.5, hi * 1.08),
                    ypos,
                    "*",
                    va="center",
                    ha="left",
                    fontsize=12,
                    fontweight="bold",
                )

        ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xlim(0.01, 10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Adjusted odds ratio for crossing")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels, fontsize=8.5)
    axes[0].set_ylabel("Cohort | sham condition")

    fig.text(
        0.5,
        -0.02,
        "OR < 1 indicates lower odds of reaching active-tDCS response at a given intensity",
        ha="center",
        va="top",
        fontsize=10,
    )

    out_png = supp_figures_dir / "supp_fig04_adjusted_anatomy_forest.png"
    out_transparent = supp_figures_dir / "supp_fig04_adjusted_anatomy_forest_transparent.png"

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_transparent, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print("\n[SUPPLEMENTARY FIGURE 4]")
    print("Adjusted anatomy OR forest plot")
    print(f"Standard PNG:      {out_png}")
    print(f"Transparent PNG:   {out_transparent}")


def make_supp_fig05_qc_trace_archive(supp_figures_dir: Path) -> None:
    """
    Supplementary Figure 5:
    Standardize representative mean-field QC trace panels if they have been generated.

    The actual traces must be produced by scripts/03_run_meanfield_predictions.py
    using --save-qc-traces for selected subjects.
    """
    qc_root = PROJECT_ROOT / "results" / "figures" / "qc_traces" / "step3"

    if not qc_root.exists():
        print("\n[SUPPLEMENTARY FIGURE 5]")
        print(f"No QC trace directory found: {qc_root}")
        print("Generate representative QC traces first, then rerun with --figure qctraces.")
        return

    pngs = sorted(qc_root.rglob("*.png"))

    if not pngs:
        print("\n[SUPPLEMENTARY FIGURE 5]")
        print(f"No QC trace PNGs found under: {qc_root}")
        print("Generate representative QC traces first, then rerun with --figure qctraces.")
        return

    out_dir = supp_figures_dir / "supp_fig05_meanfield_qc_traces"
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in pngs:
        rel_name = "_".join(src.relative_to(qc_root).parts)
        dst = out_dir / rel_name
        shutil.copy2(src, dst)

    print("\n[SUPPLEMENTARY FIGURE 5]")
    print("Representative mean-field QC traces")
    print(f"Copied {len(pngs)} PNGs to:")
    print(f"     {out_dir}")


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    tables_dir = Path(
        require_config_key(cfg, "outputs", "tables_dir")
    ).expanduser().resolve()

    supp_figures_dir = Path(
        require_config_key(cfg, "outputs", "supplementary_figures_dir")
    ).expanduser().resolve()

    supp_figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n[MAKE SUPPLEMENTARY FIGURES]")
    print(f"Figure: {args.figure}")
    print(f"Tables dir: {tables_dir}")
    print(f"Supplementary figures dir: {supp_figures_dir}")

    if args.figure in {"all", "kernel"}:
        make_supp_fig01_kernel_validation(supp_figures_dir)

    if args.figure in {"all", "rms"}:
        make_supp_fig02_rms_endpoint(tables_dir, supp_figures_dir)

    if args.figure in {"all", "csf"}:
        make_supp_fig03_csf_anatomy(tables_dir, supp_figures_dir)

    if args.figure in {"all", "forest"}:
        make_supp_fig04_adjusted_or_forest(tables_dir, supp_figures_dir)

    if args.figure in {"all", "qctraces"}:
        make_supp_fig05_qc_trace_archive(supp_figures_dir)


if __name__ == "__main__":
    main()
