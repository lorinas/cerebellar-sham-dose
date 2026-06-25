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
from shamdose.empirical_kernel_merge import (
    build_step1_empirical_kernel_table,
    print_empirical_step1_summary,
    archive_and_promote_step1_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build empirical-kernel Step 1 tables by replacing sham rows with "
            "coil-specific empirical sham/active kernel rows."
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

    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Promote the empirical-kernel Step 1 table to the canonical "
            "step1_efield_<cohort>.csv path. Without this flag, only branch tables are written."
        ),
    )

    return parser.parse_args()


def process_one_cohort(
    cohort: str,
    tables_dir: Path,
    metadata: pd.DataFrame,
    promote: bool,
) -> None:
    step1_path = tables_dir / f"step1_efield_{cohort}.csv"
    magstim_kernel_path = tables_dir / f"magstim_kernel_efield_{cohort}.csv"
    magventure_kernel_path = tables_dir / f"magventure_kernel_efield_{cohort}.csv"
    out_path = tables_dir / f"step1_efield_{cohort}_empirical_kernel.csv"

    if not step1_path.exists():
        raise FileNotFoundError(f"Missing Step 1 table: {step1_path}")

    if not magstim_kernel_path.exists():
        raise FileNotFoundError(
            f"Missing Magstim kernel rows table: {magstim_kernel_path}"
        )

    if not magventure_kernel_path.exists():
        raise FileNotFoundError(
            f"Missing MagVenture kernel rows table: {magventure_kernel_path}\n"
            "Run scripts/02f_extract_magventure_kernel_efield.py for this cohort first."
        )

    print("\n[BUILD EMPIRICAL-KERNEL STEP 1 TABLE]")
    print(f"Cohort: {cohort}")
    print(f"Original Step 1:      {step1_path}")
    print(f"Magstim kernel rows:  {magstim_kernel_path}")
    print(f"MagVenture rows:      {magventure_kernel_path}")
    print(f"Branch output:        {out_path}")
    print(f"Promote to canonical: {promote}")

    step1 = pd.read_csv(step1_path)
    magstim_kernel_rows = pd.read_csv(magstim_kernel_path)
    magventure_kernel_rows = pd.read_csv(magventure_kernel_path)

    out = build_step1_empirical_kernel_table(
        step1=step1,
        magstim_kernel_rows=magstim_kernel_rows,
        magventure_kernel_rows=magventure_kernel_rows,
        metadata=metadata,
        cohort=cohort,
    )

    print_empirical_step1_summary(out, cohort)

    out.to_csv(out_path, index=False)

    print("\n[OK] Saved empirical-kernel Step 1 branch table:")
    print(f"     {out_path}")

    if promote:
        archive_and_promote_step1_table(
            empirical_table_path=out_path,
            canonical_step1_path=step1_path,
            archive_root=tables_dir.parent / "archive",
            cohort=cohort,
        )


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
            promote=args.promote,
        )


if __name__ == "__main__":
    main()
