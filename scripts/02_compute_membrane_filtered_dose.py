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
from shamdose.metadata import ALLOWED_COHORTS
from shamdose.waveforms import (
    compute_waveform_gains,
    print_waveform_gain_summary,
)
from shamdose.step2_effective_dose import (
    compute_effective_dose_for_table,
    summarize_effective_dose,
    print_step2_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 2: compute membrane-filtered effective dose from clean Step 1 E-field tables."
        )
    )

    parser.add_argument(
        "--cohort",
        required=True,
        choices=ALLOWED_COHORTS + ["ALL"],
        help="Cohort to process: HC, SZ, CUD, or ALL.",
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "paths_local.yaml"),
        help="Path to private local config file.",
    )

    return parser.parse_args()


def process_one_cohort(
    cohort: str,
    cfg: dict,
    waveform_gains: pd.DataFrame,
) -> None:
    tables_dir = Path(
        require_config_key(cfg, "outputs", "tables_dir")
    ).expanduser().resolve()

    tau_list_ms = [
        float(x) for x in require_config_key(cfg, "step2", "tau_list_ms")
    ]

    gm_metrics = list(require_config_key(cfg, "step2", "gm_metrics"))
    include_wm_mean = bool(require_config_key(cfg, "step2", "include_wm_mean"))

    step1_path = tables_dir / f"step1_efield_{cohort}.csv"
    step2_path = tables_dir / f"step2_effective_dose_{cohort}.csv"
    summary_path = tables_dir / f"step2_effective_dose_summary_{cohort}.csv"

    if not step1_path.exists():
        raise FileNotFoundError(f"Step 1 table not found: {step1_path}")

    print("\n[STEP 2]")
    print(f"Cohort: {cohort}")
    print(f"Input Step 1 table: {step1_path}")
    print(f"Output Step 2 table: {step2_path}")

    step1 = pd.read_csv(step1_path)

    step2 = compute_effective_dose_for_table(
        step1=step1,
        waveform_gains=waveform_gains,
        tau_list_ms=tau_list_ms,
        gm_metrics=gm_metrics,
        include_wm_mean=include_wm_mean,
    )

    summary = summarize_effective_dose(
        step2=step2,
        tau_list_ms=tau_list_ms,
        cohort=cohort,
    )

    print_step2_summary(step2, summary, cohort)

    step2.to_csv(step2_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n[OK] Saved Step 2 effective-dose table:")
    print(f"     {step2_path}")

    print("[OK] Saved Step 2 summary:")
    print(f"     {summary_path}")


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    tau_list_ms = [
        float(x) for x in require_config_key(cfg, "step2", "tau_list_ms")
    ]

    waveform_cfg = require_config_key(cfg, "waveforms")

    waveform_gains = compute_waveform_gains(
        waveform_config=waveform_cfg,
        tau_list_ms=tau_list_ms,
        project_root=PROJECT_ROOT,
    )

    print_waveform_gain_summary(waveform_gains)

    tables_dir = Path(
        require_config_key(cfg, "outputs", "tables_dir")
    ).expanduser().resolve()

    gains_path = tables_dir / "step2_waveform_gains.csv"
    waveform_gains.to_csv(gains_path, index=False)

    print("\n[OK] Saved waveform gains:")
    print(f"     {gains_path}")

    cohorts = ALLOWED_COHORTS if args.cohort == "ALL" else [args.cohort]

    for cohort in cohorts:
        process_one_cohort(
            cohort=cohort,
            cfg=cfg,
            waveform_gains=waveform_gains,
        )


if __name__ == "__main__":
    main()
