#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from shamdose.config import load_config, require_config_key
from shamdose.metadata import ALLOWED_COHORTS
import shamdose.plotting as plotting


NO_ELECTRODES_CONDITIONS = [
    "MagVenture_Sham0mm",
    "Magstim_Sham0mm_kernel",
]


def make_no_electrodes_four_panel_figure(
    tables_dir: str | Path,
    out_dir: str | Path,
    cohort: str,
) -> tuple[Path, Path]:
    """
    Make a two-sham-condition version of the main figure.

    Included:
        MagVenture sham
        Magstim sham

    Excluded:
        MagVenture sham + electrodes
    """
    tables_dir = Path(tables_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Temporarily tell plotting helpers to use only the two no-electrode sham conditions.
    plotting.MAIN_CONDITIONS = NO_ELECTRODES_CONDITIONS

    step1 = pd.read_csv(tables_dir / f"step1_efield_{cohort}.csv")
    step3 = pd.read_csv(tables_dir / f"step3_meanfield_{cohort}.csv")
    crossover = pd.read_csv(tables_dir / "subject_level_crossover.csv")
    probability = pd.read_csv(tables_dir / "crossover_probability_by_intensity.csv")
    spearman = pd.read_csv(tables_dir / "anatomy_spearman_primary.csv")
    survival_main = pd.read_csv(tables_dir / "report_main_adjusted_anatomy_terms.csv")

    fig = plt.figure(figsize=(14.2, 10.0))

    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.05, 1.05],
        height_ratios=[1.0, 1.0],
        wspace=0.30,
        hspace=0.34,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])

    inner_d = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=gs[1, 1],
        wspace=0.30,
    )
    ax_d1 = fig.add_subplot(inner_d[0, 0])
    ax_d2 = fig.add_subplot(inner_d[0, 1], sharey=ax_d1)

    # Panel A: GM E-field exposure.
    e_col = "roi_graymatter_mean_|E|_V/m"

    tdcs_efield = plotting.get_tdcs_reference(
        step1,
        value_col=e_col,
        condition_col="condition",
    )

    plotting.plot_subject_curves_with_median(
        ax=ax_a,
        df=step1,
        x_col="intensity (%MT)",
        y_col=e_col,
        subject_col="subject_id",
        condition_col="condition",
        title="",
        y_label="GM exposure (V/m) ↑",
        x_label="Intensity (%MT) →",
        tdcs_ref=tdcs_efield,
    )
    plotting.add_panel_label(ax_a, "A")

    # Panel B: acute PC response.
    tdcs_pc = plotting.get_tdcs_reference(
        step3,
        value_col="PC_dPEAK_Hz",
        condition_col="CONDITION",
    )

    plotting.plot_subject_curves_with_median(
        ax=ax_b,
        df=step3,
        x_col="INTENSITY_MT",
        y_col="PC_dPEAK_Hz",
        subject_col="SUBJECT_ID",
        condition_col="CONDITION",
        title="",
        y_label="Acute PC response (ΔHz) ↑",
        x_label="Intensity (%MT) →",
        tdcs_ref=tdcs_pc,
    )
    plotting.add_panel_label(ax_b, "B")

    # Panel C: heatmap, two conditions only.
    plotting.plot_crossover_heatmap(
        fig=fig,
        ax=ax_c,
        probability=probability,
        cohort=cohort,
    )
    plotting.add_panel_label(ax_c, "C")

    # Panel D: anatomy, two mini-panels only.
    plotting.plot_anatomy_panel(
        ax=ax_d1,
        crossover=crossover,
        spearman=spearman,
        survival_main=survival_main,
        cohort=cohort,
        condition="MagVenture_Sham0mm",
        panel_title="MV sham",
        show_ylabel=True,
    )
    plotting.add_panel_label(ax_d1, "D")

    plotting.plot_anatomy_panel(
        ax=ax_d2,
        crossover=crossover,
        spearman=spearman,
        survival_main=survival_main,
        cohort=cohort,
        condition="Magstim_Sham0mm_kernel",
        panel_title="Magstim sham",
        show_ylabel=False,
    )

    png_path = out_dir / f"fig_main_{cohort}_two_sham_no_electrodes.png"
    transparent_path = out_dir / f"fig_main_{cohort}_two_sham_no_electrodes_transparent.png"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(transparent_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return png_path, transparent_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create two-condition main figures excluding MagVenture sham + electrodes."
        )
    )

    parser.add_argument(
        "--cohort",
        default="ALL",
        choices=ALLOWED_COHORTS + ["ALL"],
        help="Cohort to plot: HC, SZ, CUD, or ALL.",
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "paths_local.yaml"),
        help="Path to private local config file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    tables_dir = Path(
        require_config_key(cfg, "outputs", "tables_dir")
    ).expanduser().resolve()

    main_figures_dir = Path(
        require_config_key(cfg, "outputs", "main_figures_dir")
    ).expanduser().resolve()

    cohorts = ALLOWED_COHORTS if args.cohort == "ALL" else [args.cohort]

    print("\n[MAKE TWO-SHAM MAIN FIGURES]")
    print("Included conditions:")
    print("  MagVenture_Sham0mm")
    print("  Magstim_Sham0mm_kernel")
    print("Excluded condition:")
    print("  MagVenture_Sham0mm+electrodes")
    print(f"Tables dir: {tables_dir}")
    print(f"Main figures dir: {main_figures_dir}")

    for cohort in cohorts:
        png_path, transparent_path = make_no_electrodes_four_panel_figure(
            tables_dir=tables_dir,
            out_dir=main_figures_dir,
            cohort=cohort,
        )

        print(f"\n[OK] Saved {cohort} no-electrodes figure:")
        print(f"     {png_path}")
        print(f"     {transparent_path}")


if __name__ == "__main__":
    main()
