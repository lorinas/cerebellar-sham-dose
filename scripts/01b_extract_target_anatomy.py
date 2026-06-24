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
from shamdose.metadata import ALLOWED_COHORTS, load_metadata
from shamdose.anatomy import (
    extract_target_anatomy_for_cohort,
    print_target_anatomy_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 1b: extract local target anatomy using exact target and ROI "
            "coordinates saved in Step 1 E-field tables."
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
        "--allow-errors",
        action="store_true",
        help="Write ERROR rows instead of stopping at the first failed subject.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    metadata = load_metadata(args.metadata)

    headmodels_root = require_config_key(
        cfg,
        "cohorts",
        args.cohort,
        "headmodels_root",
    )

    tables_dir = Path(
        require_config_key(cfg, "outputs", "tables_dir")
    ).expanduser().resolve()

    step1_table = tables_dir / f"step1_efield_{args.cohort}.csv"
    out_table = tables_dir / f"target_anatomy_{args.cohort}.csv"

    print("\n[STEP 1b TARGET ANATOMY]")
    print(f"Cohort: {args.cohort}")
    print(f"Step 1 table: {step1_table}")
    print(f"Output table: {out_table}")

    anatomy = extract_target_anatomy_for_cohort(
        metadata=metadata,
        cohort=args.cohort,
        headmodels_root=headmodels_root,
        step1_table_path=step1_table,
        allow_errors=args.allow_errors,
    )

    print_target_anatomy_summary(anatomy, args.cohort)

    anatomy.to_csv(out_table, index=False)

    print("\n[OK] Saved target anatomy table:")
    print(f"     {out_table}")


if __name__ == "__main__":
    main()
