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
from shamdose.discrete_time_survival import (
    ALL_CONDITIONS,
    run_all_discrete_time_models,
    print_discrete_time_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 8: fit discrete-time logistic survival models of sham-tDCS crossover."
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

    events_path = tables_dir / "discrete_time_crossover_events.csv"

    if not events_path.exists():
        raise FileNotFoundError(
            f"Missing discrete-time event table: {events_path}\n"
            "Run scripts/04_build_subject_level_table.py first."
        )

    print("\n[STEP 8 DISCRETE-TIME SURVIVAL MODELS]")
    print(f"Input event table: {events_path}")

    events = pd.read_csv(events_path)

    endpoints = sorted(events["endpoint"].astype(str).unique().tolist())
    cohorts = ["HC", "SZ", "CUD"]

    print(f"Endpoints: {endpoints}")
    print(f"Conditions: {ALL_CONDITIONS}")
    print(f"Cohorts: {cohorts}")

    coefficients, model_status = run_all_discrete_time_models(
        events=events,
        endpoints=endpoints,
        conditions=ALL_CONDITIONS,
        cohorts=cohorts,
    )

    print_discrete_time_summary(coefficients)

    out_coef = tables_dir / "discrete_time_survival_coefficients.csv"
    out_status = tables_dir / "discrete_time_survival_model_status.csv"

    coefficients.to_csv(out_coef, index=False)
    model_status.to_csv(out_status, index=False)

    print("\n[OK] Saved discrete-time survival coefficients:")
    print(f"     {out_coef}")

    print("[OK] Saved model status table:")
    print(f"     {out_status}")

    failed = model_status[model_status["model_status"] == "FAILED"].copy()
    if not failed.empty:
        print("\n[WARNING] Some models failed. Inspect:")
        print(f"     {out_status}")
        print(failed[["endpoint", "condition", "cohort", "model_type", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
