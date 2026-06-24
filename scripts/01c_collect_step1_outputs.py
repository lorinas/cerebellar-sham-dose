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
from shamdose.step1_outputs import (
    collect_step1_outputs_for_cohort,
    summarize_collected_step1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect per-subject Step 1 E-field CSVs into one clean cohort-level table. "
            "This does not run SimNIBS."
        )
    )

    parser.add_argument(
        "--cohort",
        required=True,
        choices=ALLOWED_COHORTS,
        help="Cohort to collect: HC, SZ, or CUD.",
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
        "--allow-missing",
        action="store_true",
        help=(
            "Allow missing Step 1 CSVs. Useful while HC is incomplete. "
            "For final runs, do not use this."
        ),
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

    step1_run_subdir = require_config_key(
        cfg,
        "step1",
        "simnibs_run_subdir",
    )

    tables_dir = Path(
        require_config_key(cfg, "outputs", "tables_dir")
    ).expanduser().resolve()

    tables_dir.mkdir(parents=True, exist_ok=True)

    combined, manifest = collect_step1_outputs_for_cohort(
        metadata=metadata,
        cohort=args.cohort,
        headmodels_root=headmodels_root,
        step1_run_subdir=step1_run_subdir,
        require_all_subjects=not args.allow_missing,
    )

    summarize_collected_step1(combined, manifest, args.cohort)

    out_csv = tables_dir / f"step1_efield_{args.cohort}.csv"
    manifest_csv = tables_dir / f"step1_collection_manifest_{args.cohort}.csv"

    combined.to_csv(out_csv, index=False)
    manifest.to_csv(manifest_csv, index=False)

    print("\n[OK] Saved collected Step 1 E-field table:")
    print(f"     {out_csv}")

    print("[OK] Saved collection manifest:")
    print(f"     {manifest_csv}")


if __name__ == "__main__":
    main()
