#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import pandas as pd

from shamdose.config import load_config, require_config_key
from shamdose.primary_statistics import (
    build_crossover_probability_table,
    build_anatomy_summary,
    run_anatomy_spearman,
    print_primary_statistics_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 5: run primary cohort-specific crossover and anatomy statistics."
        )
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

    crossover_path = tables_dir / "subject_level_crossover.csv"

    if not crossover_path.exists():
        raise FileNotFoundError(
            f"Missing subject-level crossover table: {crossover_path}\n"
            "Run scripts/04_build_subject_level_table.py first."
        )

    print("\n[STEP 5 PRIMARY STATISTICS]")
    print(f"Input crossover table: {crossover_path}")

    crossover = pd.read_csv(crossover_path)

    probability = build_crossover_probability_table(crossover)
    anatomy_summary = build_anatomy_summary(crossover)
    spearman = run_anatomy_spearman(crossover)

    print_primary_statistics_summary(
        probability=probability,
        spearman=spearman,
    )

    out_probability = tables_dir / "crossover_probability_by_intensity.csv"
    out_anatomy_summary = tables_dir / "anatomy_summary_by_cohort.csv"
    out_spearman = tables_dir / "anatomy_spearman_primary.csv"

    probability.to_csv(out_probability, index=False)
    anatomy_summary.to_csv(out_anatomy_summary, index=False)
    spearman.to_csv(out_spearman, index=False)

    print("\n[OK] Saved crossover probability table:")
    print(f"     {out_probability}")

    print("[OK] Saved anatomy summary:")
    print(f"     {out_anatomy_summary}")

    print("[OK] Saved anatomy Spearman table:")
    print(f"     {out_spearman}")


if __name__ == "__main__":
    main()
