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
from shamdose.metadata import ALLOWED_COHORTS
from shamdose.plotting import make_hc_four_panel_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create main manuscript figures from cleaned analysis tables."
    )

    parser.add_argument(
        "--cohort",
        default="HC",
        choices=ALLOWED_COHORTS,
        help="Cohort to plot. Start with HC; later reuse for SZ and CUD.",
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

    print("\n[MAKE MAIN FIGURES]")
    print(f"Cohort: {args.cohort}")
    print(f"Tables dir: {tables_dir}")
    print(f"Main figures dir: {main_figures_dir}")

    png_path, transparent_path = make_hc_four_panel_figure(
        tables_dir=tables_dir,
        out_dir=main_figures_dir,
        cohort=args.cohort,
    )

    print("\n[OK] Saved main figure:")
    print(f"     {png_path}")
    print(f"     {transparent_path}")


if __name__ == "__main__":
    main()
