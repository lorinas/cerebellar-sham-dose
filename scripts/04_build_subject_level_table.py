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
from shamdose.subject_level import (
    load_step3_tables,
    load_anatomy_tables,
    validate_step3_for_subject_level,
    build_subject_info,
    build_crossover_table,
    build_discrete_time_event_table,
    summarize_crossover,
    print_subject_level_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: build subject-level crossover and discrete-time event tables "
            "from Step 3 mean-field outputs."
        )
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    metadata = load_metadata(args.metadata)

    tables_dir = Path(
        require_config_key(cfg, "outputs", "tables_dir")
    ).expanduser().resolve()

    cohorts = ALLOWED_COHORTS

    expected_subject_counts = (
        metadata[metadata["include_primary"] == 1]
        .groupby("cohort")["subject_id"]
        .nunique()
        .to_dict()
    )

    print("\n[STEP 4 SUBJECT-LEVEL TABLE]")
    print(f"Tables dir: {tables_dir}")
    print(f"Cohorts: {cohorts}")
    print(f"Expected subjects: {expected_subject_counts}")

    step3 = load_step3_tables(
        tables_dir=tables_dir,
        cohorts=cohorts,
    )

    anatomy = load_anatomy_tables(
        tables_dir=tables_dir,
        cohorts=cohorts,
    )

    validate_step3_for_subject_level(
        step3=step3,
        expected_subject_counts=expected_subject_counts,
    )

    subject_info = build_subject_info(
        metadata=metadata,
        anatomy=anatomy,
    )

    crossover = build_crossover_table(
        step3=step3,
        subject_info=subject_info,
    )

    event_table = build_discrete_time_event_table(crossover)
    summary = summarize_crossover(crossover)

    print_subject_level_summary(
        crossover=crossover,
        event_table=event_table,
        summary=summary,
    )

    out_crossover = tables_dir / "subject_level_crossover.csv"
    out_events = tables_dir / "discrete_time_crossover_events.csv"
    out_summary = tables_dir / "subject_level_crossover_summary.csv"

    crossover.to_csv(out_crossover, index=False)
    event_table.to_csv(out_events, index=False)
    summary.to_csv(out_summary, index=False)

    print("\n[OK] Saved subject-level crossover table:")
    print(f"     {out_crossover}")

    print("[OK] Saved discrete-time event table:")
    print(f"     {out_events}")

    print("[OK] Saved crossover summary:")
    print(f"     {out_summary}")


if __name__ == "__main__":
    main()
