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
from shamdose.metadata import ALLOWED_COHORTS
from shamdose.step2b_gamma_scan import (
    load_step2_tables,
    scan_gamma_inputs,
    print_gamma_global_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 2b: scan Step 2 effective-dose ranges and compute candidate gamma values."
        )
    )

    parser.add_argument(
        "--cohort",
        default="ALL",
        choices=ALLOWED_COHORTS + ["ALL"],
        help="Cohort to scan: HC, SZ, CUD, or ALL.",
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "paths_local.yaml"),
        help="Path to private local config file.",
    )

    parser.add_argument(
        "--f0-hz",
        type=float,
        default=5.0,
        help="Baseline external drive used in the mean-field model.",
    )

    parser.add_argument(
        "--fmax-hz",
        type=float,
        default=70.0,
        help="Upper drive value used for gamma_max calculation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fmax_hz <= args.f0_hz:
        raise ValueError("--fmax-hz must be greater than --f0-hz")

    cfg = load_config(args.config)

    tables_dir = Path(
        require_config_key(cfg, "outputs", "tables_dir")
    ).expanduser().resolve()

    metrics = list(require_config_key(cfg, "step2", "gm_metrics"))
    tau_list_ms = [
        float(x) for x in require_config_key(cfg, "step2", "tau_list_ms")
    ]

    cohorts = ALLOWED_COHORTS if args.cohort == "ALL" else [args.cohort]

    print("\n[STEP 2b GAMMA INPUT SCAN]")
    print(f"Cohorts: {cohorts}")
    print(f"Tables dir: {tables_dir}")
    print(f"Metrics: {metrics}")
    print(f"Tau values ms: {tau_list_ms}")
    print(f"f0_Hz: {args.f0_hz}")
    print(f"fmax_Hz: {args.fmax_hz}")

    step2_all = load_step2_tables(
        tables_dir=tables_dir,
        cohorts=cohorts,
    )

    by_group, global_summary = scan_gamma_inputs(
        step2_all=step2_all,
        metrics=metrics,
        tau_list_ms=tau_list_ms,
        f0_hz=args.f0_hz,
        fmax_hz=args.fmax_hz,
    )

    if args.cohort == "ALL":
        suffix = "ALL"
    else:
        suffix = args.cohort

    by_group_path = tables_dir / f"step2b_gamma_scan_by_group_{suffix}.csv"
    global_path = tables_dir / f"step2b_gamma_scan_global_{suffix}.csv"

    by_group.to_csv(by_group_path, index=False)
    global_summary.to_csv(global_path, index=False)

    print_gamma_global_summary(global_summary)

    print("\n[OK] Saved by-group gamma scan:")
    print(f"     {by_group_path}")

    print("[OK] Saved global gamma scan:")
    print(f"     {global_path}")


if __name__ == "__main__":
    main()
