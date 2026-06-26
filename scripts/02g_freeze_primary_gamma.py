#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from shamdose.config import load_config, require_config_key


DEFAULT_CONDITIONS_TO_MODEL = [
    "MagVenture_Sham0mm",
    "MagVenture_Sham0mm+electrodes",
    "Magstim_Sham0mm_kernel",
    "tDCS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the primary gamma choice after Step 2b gamma scanning. "
            "This writes results/tables/step2b_primary_gamma_choice.csv and "
            "updates config/paths_local.yaml step3 settings."
        )
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "paths_local.yaml"),
        help="Path to private local config file.",
    )

    parser.add_argument(
        "--analysis-set",
        default="all_sham_tdcs",
        help="Gamma scan analysis set to select.",
    )

    parser.add_argument(
        "--metric",
        default="mean",
        help="Dose metric to select.",
    )

    parser.add_argument(
        "--tau-ms",
        type=float,
        default=0.085,
        help="Membrane-filter tau value to select.",
    )

    parser.add_argument(
        "--gamma-fraction",
        type=float,
        default=0.50,
        help="Fraction of gamma_max to use.",
    )

    parser.add_argument(
        "--no-update-config",
        action="store_true",
        help="Only write the gamma-choice CSV; do not update paths_local.yaml.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)

    tables_dir = Path(
        require_config_key(cfg, "outputs", "tables_dir")
    ).expanduser().resolve()

    global_path = tables_dir / "step2b_gamma_scan_global_ALL.csv"

    if not global_path.exists():
        raise FileNotFoundError(
            f"Missing gamma scan table: {global_path}\n"
            "Run scripts/02b_scan_gamma_inputs.py --cohort ALL first."
        )

    gamma_scan = pd.read_csv(global_path)

    selected = gamma_scan[
        (gamma_scan["analysis_set"].astype(str) == args.analysis_set)
        & (gamma_scan["metric"].astype(str) == args.metric)
        & (pd.to_numeric(gamma_scan["tau_ms"], errors="coerce") == float(args.tau_ms))
    ].copy()

    if len(selected) != 1:
        raise RuntimeError(
            "Expected exactly one primary gamma row, found "
            f"{len(selected)} for analysis_set={args.analysis_set}, "
            f"metric={args.metric}, tau_ms={args.tau_ms}."
        )

    row = selected.iloc[0]

    gamma_max = float(row["gamma_max_Hz_per_Vm"])
    gamma_value = float(args.gamma_fraction) * gamma_max

    gamma_choice = {
        "analysis_set": args.analysis_set,
        "conditions_to_model": DEFAULT_CONDITIONS_TO_MODEL,
        "metric": args.metric,
        "tau_ms": float(row["tau_ms"]),
        "dose_column": str(row["C_column"]),
        "f0_Hz": float(row["f0_Hz"]),
        "fmax_Hz": float(row["fmax_Hz"]),
        "C_max_Vm": float(row["C_max_Vm"]),
        "gamma_max_Hz_per_Vm": gamma_max,
        "gamma_fraction": float(args.gamma_fraction),
        "gamma_Hz_per_Vm": gamma_value,
    }

    out = pd.DataFrame([gamma_choice])
    out_path = tables_dir / "step2b_primary_gamma_choice.csv"
    out.to_csv(out_path, index=False)

    print("\n[PRIMARY GAMMA CHOICE]")
    print(out.T.to_string(header=False))
    print(f"\n[OK] Saved: {out_path}")

    if not args.no_update_config:
        local_cfg = yaml.safe_load(cfg_path.read_text())

        local_cfg["step3"] = {
            "analysis_set": gamma_choice["analysis_set"],
            "conditions_to_model": gamma_choice["conditions_to_model"],
            "primary_metric": gamma_choice["metric"],
            "primary_tau_ms": gamma_choice["tau_ms"],
            "primary_dose_column": gamma_choice["dose_column"],
            "baseline_drive_Hz": gamma_choice["f0_Hz"],
            "calibration_fmax_Hz": gamma_choice["fmax_Hz"],
            "gamma_fraction": gamma_choice["gamma_fraction"],
            "gamma_Hz_per_Vm": gamma_choice["gamma_Hz_per_Vm"],
        }

        cfg_path.write_text(yaml.safe_dump(local_cfg, sort_keys=False))

        print(f"[OK] Updated local Step 3 settings in: {cfg_path}")


if __name__ == "__main__":
    main()
