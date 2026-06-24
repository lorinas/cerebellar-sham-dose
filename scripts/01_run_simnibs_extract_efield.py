#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from shamdose.config import load_config, require_config_key
from shamdose.metadata import ALLOWED_COHORTS, load_metadata
from shamdose.cohorts import (
    build_step1_preflight_table,
    print_step1_preflight_summary,
)
from shamdose.step1_simulation import run_step1_for_subject


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 1 SimNIBS E-field extraction. "
            "Use --dry-run to check subjects; use --run to process READY_TO_RUN subjects."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Check subject selection, m2m folders, and existing Step 1 outputs only.",
    )

    mode.add_argument(
        "--run",
        action="store_true",
        help="Run Step 1 simulations/extraction for subjects marked READY_TO_RUN.",
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
        "--subject",
        action="append",
        default=None,
        help=(
            "Optional subject ID to run/check. Can be repeated. "
            "Example: --subject sub-GC001 --subject sub-GC002"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of READY_TO_RUN subjects to process. Useful for testing.",
    )

    parser.add_argument(
        "--force-rerun-simnibs",
        action="store_true",
        help=(
            "Force rerun of TMS/tDCS SimNIBS simulations inside a READY_TO_RUN subject. "
            "Use rarely."
        ),
    )

    return parser.parse_args()


def select_rows_to_run(
    preflight: pd.DataFrame,
    requested_subjects: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Select only subjects that are safe to run.

    We intentionally run only:
        row_type == EXPECTED_FROM_METADATA
        status == READY_TO_RUN
        needs_step1 == 1

    This avoids overwriting valid existing Step 1 outputs.
    """
    run_rows = preflight[
        (preflight["row_type"] == "EXPECTED_FROM_METADATA")
        & (preflight["status"] == "READY_TO_RUN")
        & (preflight["needs_step1"] == 1)
    ].copy()

    if requested_subjects is not None:
        requested_subjects = list(requested_subjects)
        available = set(run_rows["subject_id"].astype(str))
        missing = sorted(set(requested_subjects) - available)

        if missing:
            raise ValueError(
                "Requested subject(s) are not READY_TO_RUN:\n"
                f"    {missing}\n"
                "Run --dry-run first and check status."
            )

        run_rows = run_rows[run_rows["subject_id"].astype(str).isin(requested_subjects)].copy()

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive.")
        run_rows = run_rows.head(limit).copy()

    return run_rows.reset_index(drop=True)


def main() -> None:
    args = parse_args()

    print("\n[STEP 1]")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Cohort: {args.cohort}")
    print(f"Mode: {'RUN' if args.run else 'DRY RUN'}")
    print(f"Config: {Path(args.config).expanduser().resolve()}")
    print(f"Metadata: {Path(args.metadata).expanduser().resolve()}")

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

    coil_ccd_dir = require_config_key(
        cfg,
        "step1",
        "coil_ccd_dir",
    )

    roi_radius_mm = float(
        require_config_key(cfg, "step1", "roi_radius_mm")
    )

    target_distance_below_iz_mm = float(
        require_config_key(cfg, "step1", "target_distance_below_iz_mm")
    )

    tables_dir = Path(
        require_config_key(cfg, "outputs", "tables_dir")
    ).expanduser().resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)

    preflight = build_step1_preflight_table(
        metadata=metadata,
        cohort=args.cohort,
        headmodels_root=headmodels_root,
        step1_run_subdir=step1_run_subdir,
    )

    print_step1_preflight_summary(preflight, args.cohort)

    preflight_path = tables_dir / f"step1_preflight_{args.cohort}.csv"
    preflight.to_csv(preflight_path, index=False)

    print(f"\n[OK] Saved Step 1 preflight table:")
    print(f"     {preflight_path}")

    if args.dry_run:
        print("\n[DRY RUN COMPLETE]")
        print("No SimNIBS simulations were run.")
        return

    run_rows = select_rows_to_run(
        preflight=preflight,
        requested_subjects=args.subject,
        limit=args.limit,
    )

    if run_rows.empty:
        print("\n[RUN COMPLETE]")
        print("No subjects were READY_TO_RUN.")
        return

    print("\n[SUBJECTS SELECTED FOR STEP 1 RUN]")
    print(run_rows[["subject_id", "status", "m2m_dir"]].to_string(index=False))

    completed: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []

    for _, row in run_rows.iterrows():
        subject_id = str(row["subject_id"])

        try:
            out_csv = run_step1_for_subject(
                subject_id=subject_id,
                subject_dir=str(row["subject_dir"]),
                m2m_dir=str(row["m2m_dir"]),
                out_subdir=step1_run_subdir,
                coil_ccd_dir=coil_ccd_dir,
                run_simnibs_flag=True,
                force_rerun=args.force_rerun_simnibs,
                run_tms=True,
                run_tdcs=True,
                roi_radius_mm=roi_radius_mm,
                inion_below_mm=target_distance_below_iz_mm,
            )

            completed.append(
                {
                    "subject_id": subject_id,
                    "status": "COMPLETED",
                    "step1_csv": out_csv,
                }
            )

        except Exception as exc:
            failed.append(
                {
                    "subject_id": subject_id,
                    "status": "FAILED",
                    "error": repr(exc),
                }
            )
            print(f"\n[ERROR] Step 1 failed for {subject_id}: {exc}", flush=True)
            raise

    run_log = pd.DataFrame(completed + failed)
    run_log_path = tables_dir / f"step1_run_log_{args.cohort}.csv"
    run_log.to_csv(run_log_path, index=False)

    print("\n[RUN COMPLETE]")
    print(f"Completed subjects: {len(completed)}")
    print(f"Failed subjects: {len(failed)}")
    print(f"Run log: {run_log_path}")

    print("\n[NEXT]")
    print(
        "Rerun the dry run to confirm completed subjects now appear as ALREADY_PROCESSED."
    )


if __name__ == "__main__":
    main()
