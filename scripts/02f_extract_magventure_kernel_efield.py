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
from shamdose.magventure_kernel_subjects import (
    build_magventure_kernel_efield_for_cohort,
    print_magventure_kernel_cohort_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract empirical-kernel MagVenture sham E-field rows by applying "
            "the Smith-Peterchev MagVenture sham/active spatial kernel to each "
            "subject's active MagVenture SimNIBS mesh."
        )
    )

    parser.add_argument(
        "--cohort",
        required=True,
        choices=ALLOWED_COHORTS,
        help="Cohort to process: HC, SZ, or CUD.",
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "paths_local.yaml"),
        help="Path to private local config file.",
    )

    parser.add_argument(
        "--metadata",
        default=str(PROJECT_ROOT / "config" / "cohort_metadata.csv"),
        help="Path to private cohort metadata CSV.",
    )

    parser.add_argument(
        "--kernel-points",
        default=str(PROJECT_ROOT / "results" / "tables" / "magstim_empirical_kernel_points.csv"),
        help="Empirical kernel point table from Step 2c.",
    )

    parser.add_argument(
        "--y-sign",
        type=float,
        default=1.0,
        choices=[-1.0, 1.0],
        help="Sign of empirical y-axis relative to SimNIBS coil ydir.",
    )

    parser.add_argument(
        "--max-ratio",
        type=float,
        default=1.0,
        help="Maximum allowed local sham/active ratio after interpolation.",
    )

    parser.add_argument(
        "--limit-subjects",
        type=int,
        default=None,
        help="Optional number of subjects to process for testing.",
    )

    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix for output files, e.g. TEST.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    metadata = load_metadata(args.metadata)

    tables_dir = Path(require_config_key(cfg, "outputs", "tables_dir")).expanduser().resolve()

    headmodels_root = require_config_key(
        cfg,
        "cohorts",
        args.cohort,
        "headmodels_root",
    )

    roi_radius_mm = float(require_config_key(cfg, "step1", "roi_radius_mm"))
    target_distance_below_iz_mm = float(require_config_key(cfg, "step1", "target_distance_below_iz_mm"))

    step1_path = tables_dir / f"step1_efield_{args.cohort}.csv"

    if not step1_path.exists():
        raise FileNotFoundError(f"Missing Step 1 table: {step1_path}")

    print("\n[STEP 2f MAGVENTURE EMPIRICAL-KERNEL SUBJECT EXTRACTION]")
    print(f"Cohort: {args.cohort}")
    print(f"Step 1 table: {step1_path}")
    print(f"Kernel points: {Path(args.kernel_points).expanduser().resolve()}")
    print(f"y_sign: {args.y_sign}")
    print(f"max_ratio: {args.max_ratio}")
    print(f"limit_subjects: {args.limit_subjects}")

    step1 = pd.read_csv(step1_path)

    kernel_rows, qc = build_magventure_kernel_efield_for_cohort(
        step1=step1,
        metadata=metadata,
        cohort=args.cohort,
        headmodels_root=headmodels_root,
        kernel_points_path=args.kernel_points,
        roi_radius_mm=roi_radius_mm,
        target_distance_below_iz_mm=target_distance_below_iz_mm,
        y_sign=args.y_sign,
        max_ratio=args.max_ratio,
        limit_subjects=args.limit_subjects,
    )

    print_magventure_kernel_cohort_summary(
        rows=kernel_rows,
        qc=qc,
        cohort=args.cohort,
    )

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""

    out_rows = tables_dir / f"magventure_kernel_efield_{args.cohort}{suffix}.csv"
    out_qc = tables_dir / f"magventure_kernel_efield_qc_{args.cohort}{suffix}.csv"

    kernel_rows.to_csv(out_rows, index=False)
    qc.to_csv(out_qc, index=False)

    print("\n[OK] Saved corrected MagVenture kernel E-field rows:")
    print(f"     {out_rows}")

    print("[OK] Saved corrected MagVenture kernel QC table:")
    print(f"     {out_qc}")


if __name__ == "__main__":
    main()
