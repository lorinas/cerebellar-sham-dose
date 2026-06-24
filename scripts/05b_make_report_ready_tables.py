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
from shamdose.report_tables import (
    clean_survival_coefficients,
    build_main_adjusted_anatomy_table,
    build_key_results_table,
    print_report_table_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create clean report-ready tables from primary statistics outputs."
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

    coef_path = tables_dir / "discrete_time_survival_coefficients.csv"
    crossover_summary_path = tables_dir / "subject_level_crossover_summary.csv"
    spearman_path = tables_dir / "anatomy_spearman_primary.csv"

    for p in [coef_path, crossover_summary_path, spearman_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required table: {p}")

    print("\n[STEP 8b REPORT-READY TABLES]")
    print(f"Tables dir: {tables_dir}")

    coef = pd.read_csv(coef_path)
    crossover_summary = pd.read_csv(crossover_summary_path)
    spearman = pd.read_csv(spearman_path)

    clean_coef = clean_survival_coefficients(coef)
    main_adjusted = build_main_adjusted_anatomy_table(clean_coef)
    key_results = build_key_results_table(
        crossover_summary=crossover_summary,
        spearman=spearman,
        survival_main=main_adjusted,
    )

    print_report_table_summary(
        clean_coef=clean_coef,
        main_adjusted=main_adjusted,
        key_results=key_results,
    )

    out_clean = tables_dir / "report_discrete_time_survival_terms.csv"
    out_main = tables_dir / "report_main_adjusted_anatomy_terms.csv"
    out_key = tables_dir / "report_key_results_primary.csv"

    clean_coef.to_csv(out_clean, index=False)
    main_adjusted.to_csv(out_main, index=False)
    key_results.to_csv(out_key, index=False)

    print("\n[OK] Saved report-ready survival terms:")
    print(f"     {out_clean}")

    print("[OK] Saved main adjusted anatomy table:")
    print(f"     {out_main}")

    print("[OK] Saved key primary results table:")
    print(f"     {out_key}")


if __name__ == "__main__":
    main()
