#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from shamdose.config import load_config, require_config_key
from shamdose.magstim_kernel import (
    build_magstim_empirical_kernel,
    plot_kernel_maps,
    print_kernel_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 2c: build empirical active/sham coil-field kernels from "
            "Smith-Peterchev measured active/sham E-field grid."
        )
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "paths_local.yaml"),
        help="Path to private local config file.",
    )

    parser.add_argument(
        "--input-csv",
        default=str(
            PROJECT_ROOT
            / "resources"
            / "external"
            / "smith_peterchev_2018"
            / "processed"
            / "Smith_Peterchev_2018_Efield_DATA.csv"
        ),
        help="Clean 1000-row Smith-Peterchev CSV.",
    )

    parser.add_argument(
        "--active-floor-frac",
        type=float,
        default=0.01,
        help=(
            "Active-field floor as fraction of active peak for regularized ratios. "
            "Default 0.01 = 1% of active peak."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    tables_dir = Path(
        require_config_key(cfg, "outputs", "tables_dir")
    ).expanduser().resolve()

    supp_fig_dir = Path(
        require_config_key(cfg, "outputs", "supplementary_figures_dir")
    ).expanduser().resolve()

    tables_dir.mkdir(parents=True, exist_ok=True)
    supp_fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n[STEP 2c EMPIRICAL COIL KERNELS]")
    print(f"Input CSV: {Path(args.input_csv).expanduser().resolve()}")
    print(f"Active floor fraction: {args.active_floor_frac}")

    kernel_points, summary = build_magstim_empirical_kernel(
        clean_csv=args.input_csv,
        active_floor_frac=args.active_floor_frac,
    )

    print_kernel_summary(summary)

    points_path = tables_dir / "magstim_empirical_kernel_points.csv"
    summary_path = tables_dir / "magstim_empirical_kernel_summary.csv"
    fig_path = supp_fig_dir / "magstim_empirical_kernel_map.png"
    fig_transparent_path = supp_fig_dir / "magstim_empirical_kernel_map_transparent.png"

    kernel_points.to_csv(points_path, index=False)
    summary.to_csv(summary_path, index=False)

    plot_kernel_maps(
        kernel_points=kernel_points,
        out_png=fig_path,
        out_transparent_png=fig_transparent_path,
    )

    print("\n[OK] Saved empirical kernel point table:")
    print(f"     {points_path}")

    print("[OK] Saved empirical kernel summary:")
    print(f"     {summary_path}")

    print("[OK] Saved empirical kernel diagnostic maps:")
    print(f"     {fig_path}")
    print(f"     {fig_transparent_path}")


if __name__ == "__main__":
    main()
