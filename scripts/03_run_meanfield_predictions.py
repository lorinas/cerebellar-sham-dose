#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from shamdose.config import load_config, require_config_key
from shamdose.metadata import ALLOWED_COHORTS
from shamdose.crbl_model import load_crbl_model
from shamdose.step3_meanfield import (
    DEFAULT_MODELED_CONDITIONS,
    ACTIVE_CONDITIONS,
    run_meanfield_for_cohort,
    run_meanfield_subject_chunk_worker,
    split_subjects,
    summarize_step3_output,
    print_step3_summary,
    validate_step3_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3: run Lorenzi cerebellar mean-field model using Step 2 effective-dose tables."
        )
    )

    parser.add_argument(
        "--cohort",
        required=True,
        choices=ALLOWED_COHORTS + ["ALL"],
        help="Cohort to run: HC, SZ, CUD, or ALL.",
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "paths_local.yaml"),
        help="Path to private local config file.",
    )

    parser.add_argument(
        "--include-active",
        action="store_true",
        help="Include active TMS conditions. Not allowed unless explicitly acknowledged.",
    )

    parser.add_argument(
        "--subject",
        action="append",
        default=None,
        help="Optional subject ID to run. Can be repeated.",
    )

    parser.add_argument(
        "--limit-subjects",
        type=int,
        default=None,
        help="Optional number of subjects to run. Useful for testing.",
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel subject workers. "
            "Use 1 for sequential execution. Start with 2 on a laptop."
        ),
    )

    parser.add_argument(
        "--save-qc-traces",
        action="store_true",
        help="Save selected PC/fmossy time-series QC traces for representative subjects.",
    )

    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix for output filenames, e.g. TEST.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing Step 3 output files.",
    )

    return parser.parse_args()


def build_output_name(base: str, cohort: str, suffix: str, ext: str = ".csv") -> str:
    if suffix:
        return f"{base}_{cohort}_{suffix}{ext}"
    return f"{base}_{cohort}{ext}"


def gamma_column_from_fraction(gamma_fraction: float) -> str:
    """
    Map gamma fraction to the column name in step2b_gamma_scan_global_ALL.csv.
    """
    if abs(gamma_fraction - 0.25) < 1e-9:
        return "gamma_25pct_Hz_per_Vm"
    if abs(gamma_fraction - 0.50) < 1e-9:
        return "gamma_50pct_Hz_per_Vm"
    if abs(gamma_fraction - 0.75) < 1e-9:
        return "gamma_75pct_Hz_per_Vm"
    if abs(gamma_fraction - 1.00) < 1e-9:
        return "gamma_max_Hz_per_Vm"

    raise ValueError(
        f"Unsupported gamma_fraction={gamma_fraction}. "
        "Expected 0.25, 0.50, 0.75, or 1.00."
    )


def load_gamma_from_scan(
    tables_dir: Path,
    analysis_set: str,
    metric: str,
    tau_ms: float,
    gamma_fraction: float,
) -> dict:
    """
    Load gamma from the global Step 2b gamma scan.

    Used especially when --include-active is selected. In that case we should
    automatically switch from all_sham_tdcs gamma to all_conditions gamma,
    because active TMS has much larger C values.
    """
    gamma_scan_path = tables_dir / "step2b_gamma_scan_global_ALL.csv"

    if not gamma_scan_path.exists():
        raise FileNotFoundError(
            f"Missing gamma scan table: {gamma_scan_path}\n"
            "Run scripts/02b_scan_gamma_inputs.py --cohort ALL first."
        )

    scan = pd.read_csv(gamma_scan_path)
    scan["tau_ms"] = pd.to_numeric(scan["tau_ms"], errors="coerce")

    row = scan[
        (scan["analysis_set"].astype(str) == analysis_set)
        & (scan["metric"].astype(str) == metric)
        & (abs(scan["tau_ms"] - float(tau_ms)) < 1e-12)
    ].copy()

    if len(row) != 1:
        raise RuntimeError(
            f"Expected exactly one gamma row for analysis_set={analysis_set}, "
            f"metric={metric}, tau_ms={tau_ms}; found {len(row)}."
        )

    row = row.iloc[0]
    gamma_col = gamma_column_from_fraction(gamma_fraction)

    return {
        "analysis_set": analysis_set,
        "gamma_Hz_per_Vm": float(row[gamma_col]),
        "gamma_max_Hz_per_Vm": float(row["gamma_max_Hz_per_Vm"]),
        "C_max_Vm": float(row["C_max_Vm"]),
        "f0_Hz": float(row["f0_Hz"]),
        "fmax_Hz": float(row["fmax_Hz"]),
        "dose_column": str(row["C_column"]),
        "gamma_source_table": str(gamma_scan_path),
        "gamma_source_column": gamma_col,
    }


def process_one_cohort(
    cohort: str,
    cfg: dict,
    model,
    args: argparse.Namespace,
) -> None:
    tables_dir = Path(require_config_key(cfg, "outputs", "tables_dir")).expanduser().resolve()

    step2_path = tables_dir / f"step2_effective_dose_{cohort}.csv"
    anatomy_path = tables_dir / f"target_anatomy_{cohort}.csv"

    if not step2_path.exists():
        raise FileNotFoundError(f"Missing Step 2 table: {step2_path}")

    if not anatomy_path.exists():
        raise FileNotFoundError(f"Missing anatomy table: {anatomy_path}")

    step3_cfg = require_config_key(cfg, "step3")

    conditions_to_model = list(step3_cfg.get("conditions_to_model", DEFAULT_MODELED_CONDITIONS))
    analysis_set = str(step3_cfg.get("analysis_set", "all_sham_tdcs"))

    if args.include_active:
        # Active TMS has much larger C values than the sham/tDCS set.
        # Therefore, if active is explicitly requested, automatically switch
        # gamma calibration to the all_conditions analysis set.
        for cond in ACTIVE_CONDITIONS:
            if cond not in conditions_to_model:
                conditions_to_model.append(cond)

    dose_column = str(require_config_key(step3_cfg, "primary_dose_column"))
    metric = str(require_config_key(step3_cfg, "primary_metric"))
    tau_ms = float(require_config_key(step3_cfg, "primary_tau_ms"))
    baseline_drive_hz = float(require_config_key(step3_cfg, "baseline_drive_Hz"))
    fmax_hz = float(require_config_key(step3_cfg, "calibration_fmax_Hz"))
    gamma_fraction = float(require_config_key(step3_cfg, "gamma_fraction"))
    gamma_hz_per_vm = float(require_config_key(step3_cfg, "gamma_Hz_per_Vm"))
    gamma_source_table = "config/paths_local.yaml"
    gamma_source_column = "step3.gamma_Hz_per_Vm"
    gamma_max_hz_per_vm = float("nan")
    gamma_cmax_vm = float("nan")

    if args.include_active:
        gamma_info = load_gamma_from_scan(
            tables_dir=tables_dir,
            analysis_set="all_conditions",
            metric=metric,
            tau_ms=tau_ms,
            gamma_fraction=gamma_fraction,
        )
        analysis_set = gamma_info["analysis_set"]
        gamma_hz_per_vm = gamma_info["gamma_Hz_per_Vm"]
        gamma_max_hz_per_vm = gamma_info["gamma_max_Hz_per_Vm"]
        gamma_cmax_vm = gamma_info["C_max_Vm"]
        baseline_drive_hz = gamma_info["f0_Hz"]
        fmax_hz = gamma_info["fmax_Hz"]
        dose_column = gamma_info["dose_column"]
        gamma_source_table = gamma_info["gamma_source_table"]
        gamma_source_column = gamma_info["gamma_source_column"]

    step2 = pd.read_csv(step2_path)
    anatomy = pd.read_csv(anatomy_path)

    if args.subject is not None:
        requested = {str(s) for s in args.subject}
        step2_subjects = set(step2["subject_id"].astype(str))
        missing = sorted(requested - step2_subjects)
        if missing:
            raise ValueError(f"Requested subjects not found in Step 2 table: {missing}")

    qc_trace_dir = PROJECT_ROOT / "results" / "figures" / "qc_traces" / "step3"

    print("\n[STEP 3 RUN]")
    print(f"Cohort: {cohort}")
    print(f"Step 2 table: {step2_path}")
    print(f"Anatomy table: {anatomy_path}")
    print(f"Analysis set: {analysis_set}")
    print(f"Conditions modeled: {conditions_to_model}")
    print(f"Dose column: {dose_column}")
    print(f"Gamma: {gamma_hz_per_vm:.6f} Hz/V/m")
    print(f"Gamma fraction: {gamma_fraction}")
    print(f"Gamma source table: {gamma_source_table}")
    print(f"Gamma source column: {gamma_source_column}")
    if args.include_active:
        print(f"Active-TMS mode: using all_conditions gamma, Cmax={gamma_cmax_vm:.6f} V/m")
    print(f"Baseline drive: {baseline_drive_hz} Hz")
    print(f"Fmax: {fmax_hz} Hz")
    print(f"Save QC traces: {args.save_qc_traces}")

    # Determine the exact subject list that will be modeled.
    step2_for_subjects = step2[step2["condition"].astype(str).isin(conditions_to_model)].copy()

    if args.subject is not None:
        requested = {str(s) for s in args.subject}
        step2_for_subjects = step2_for_subjects[
            step2_for_subjects["subject_id"].astype(str).isin(requested)
        ].copy()

    subject_ids = sorted(step2_for_subjects["subject_id"].astype(str).unique().tolist())

    if args.limit_subjects is not None:
        if args.limit_subjects <= 0:
            raise ValueError("--limit-subjects must be positive.")
        subject_ids = subject_ids[: args.limit_subjects]

    if not subject_ids:
        raise ValueError("No subjects selected for Step 3.")

    print(f"Subjects selected: {len(subject_ids)}")
    print(f"n_workers: {args.n_workers}")

    if args.n_workers < 1:
        raise ValueError("--n-workers must be >= 1")

    if args.save_qc_traces and args.n_workers > 1:
        raise ValueError(
            "--save-qc-traces is only supported with --n-workers 1. "
            "QC plotting is intentionally kept sequential."
        )

    if args.n_workers == 1:
        out = run_meanfield_for_cohort(
            model=model,
            step2=step2,
            anatomy=anatomy,
            cohort=cohort,
            conditions_to_model=conditions_to_model,
            dose_column=dose_column,
            metric=metric,
            tau_ms=tau_ms,
            gamma_hz_per_vm=gamma_hz_per_vm,
            gamma_fraction=gamma_fraction,
            baseline_drive_hz=baseline_drive_hz,
            fmax_hz=fmax_hz,
            analysis_set=analysis_set,
            gamma_source_table=gamma_source_table,
            gamma_source_column=gamma_source_column,
            gamma_max_hz_per_vm=gamma_max_hz_per_vm,
            gamma_cmax_vm=gamma_cmax_vm,
            include_active=args.include_active,
            subject_filter=subject_ids,
            limit_subjects=None,
            save_qc_traces=args.save_qc_traces,
            qc_trace_dir=qc_trace_dir,
        )
    else:
        # Parallel by subject chunks. Each worker loads its own CRBL_MF model.
        chunks = split_subjects(subject_ids, args.n_workers)

        print("\n[PARALLEL STEP 3]")
        for i, chunk in enumerate(chunks, start=1):
            print(f"Worker chunk {i}: {len(chunk)} subjects")

        futures = []

        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            for worker_id, chunk in enumerate(chunks, start=1):
                step2_chunk = step2[
                    step2["subject_id"].astype(str).isin(chunk)
                ].copy()

                futures.append(
                    executor.submit(
                        run_meanfield_subject_chunk_worker,
                        worker_id,
                        str(model.repo_root),
                        step2_chunk,
                        anatomy,
                        cohort,
                        chunk,
                        conditions_to_model,
                        dose_column,
                        metric,
                        tau_ms,
                        gamma_hz_per_vm,
                        gamma_fraction,
                        baseline_drive_hz,
                        fmax_hz,
                        analysis_set,
                        gamma_source_table,
                        gamma_source_column,
                        gamma_max_hz_per_vm,
                        gamma_cmax_vm,
                        args.include_active,
                    )
                )

            outputs = []

            for future in as_completed(futures):
                outputs.append(future.result())

        out = pd.concat(outputs, ignore_index=True)

        # Stable output order.
        out["SUBJECT_ID"] = out["SUBJECT_ID"].astype(str)
        out = out.sort_values(["SUBJECT_ID", "CONDITION", "INTENSITY_MT"]).reset_index(drop=True)

        validate_step3_output(
            out,
            expected_subjects=subject_ids,
            include_active=args.include_active,
        )

    summary = summarize_step3_output(out, cohort)
    print_step3_summary(out, summary, cohort)

    suffix = args.output_suffix.strip()
    out_path = tables_dir / build_output_name("step3_meanfield", cohort, suffix)
    summary_path = tables_dir / build_output_name("step3_meanfield_summary", cohort, suffix)

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {out_path}\n"
            "Use --overwrite or provide --output-suffix."
        )

    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Summary output already exists: {summary_path}\n"
            "Use --overwrite or provide --output-suffix."
        )

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_summary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")

    out.to_csv(tmp_path, index=False)
    summary.to_csv(tmp_summary_path, index=False)

    tmp_path.replace(out_path)
    tmp_summary_path.replace(summary_path)

    print("\n[OK] Saved Step 3 table:")
    print(f"     {out_path}")

    print("[OK] Saved Step 3 summary:")
    print(f"     {summary_path}")

    if args.save_qc_traces:
        print("[OK] QC traces saved under:")
        print(f"     {qc_trace_dir / cohort}")


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    crbl_repo_root = require_config_key(cfg, "crbl_mf", "repo_root")

    print("\n[LOAD CRBL_MF MODEL]")
    print(f"Repo root: {crbl_repo_root}")

    model = load_crbl_model(crbl_repo_root)

    cohorts = ALLOWED_COHORTS if args.cohort == "ALL" else [args.cohort]

    for cohort in cohorts:
        process_one_cohort(
            cohort=cohort,
            cfg=cfg,
            model=model,
            args=args,
        )


if __name__ == "__main__":
    main()
