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
from shamdose.metadata import ALLOWED_COHORTS, load_metadata
from shamdose.magstim_kernel_merge import (
    build_step1_magstim_kernel_table,
    print_kernel_step1_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build kernel-updated Step 1 tables by replacing deprecated scalar "
            "Magstim sham rows with empirical-kernel Magstim sham rows."
        )
    )

    parser.add_argument(
        "--cohort",
        default="ALL",
        choices=ALLOWED_COHORTS + ["ALL"],
        help="Cohort to process: HC, SZ, CUD, or ALL.",
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "paths_local.yaml"),
        help="Path to private local config file.",
    )

    parser.add_argument(
        "--metadata",
        default=str(PROJECT_ROOT / "config" / "cohort_metadata.csv"),
        help="Path to cohort metadata CSV.",
    )

    return parser.parse_args()


def process_one_cohort(cohort: str, tables_dir: Path, metadata: pd.DataFrame) -> None:
    step1_path = tables_dir / f"step1_efield_{cohort}.csv"
    kernel_path = tables_dir / f"magstim_kernel_efield_{cohort}.csv"
    out_path = tables_dir / f"step1_efield_{cohort}_magstim_kernel.csv"

    if not step1_path.exists():
        raise FileNotFoundError(f"Missing Step 1 table: {step1_path}")

    if not kernel_path.exists():
        raise FileNotFoundError(f"Missing Magstim kernel rows table: {kernel_path}")

    print("\n[BUILD KERNEL STEP 1 TABLE]")
    print(f"Cohort: {cohort}")
    print(f"Original Step 1: {step1_path}")
    print(f"Kernel rows:     {kernel_path}")
    print(f"Output:          {out_path}")

    step1 = pd.read_csv(step1_path)
    kernel_rows = pd.read_csv(kernel_path)

    out = build_step1_magstim_kernel_table(
        step1=step1,
        kernel_rows=kernel_rows,
        metadata=metadata,
        cohort=cohort,
    )

    print_kernel_step1_summary(out, cohort)

    out.to_csv(out_path, index=False)

    print("\n[OK] Saved kernel-updated Step 1 table:")
    print(f"     {out_path}")


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    metadata = load_metadata(args.metadata)

    tables_dir = Path(
        require_config_key(cfg, "outputs", "tables_dir")
    ).expanduser().resolve()

    cohorts = ALLOWED_COHORTS if args.cohort == "ALL" else [args.cohort]

    for cohort in cohorts:
        process_one_cohort(
            cohort=cohort,
            tables_dir=tables_dir,
            metadata=metadata,
        )


if __name__ == "__main__":
    main()
