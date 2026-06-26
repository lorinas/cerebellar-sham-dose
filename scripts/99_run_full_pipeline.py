#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


STAGES = [
    "metadata",
    "collect_step1",
    "target_anatomy",
    "build_empirical_kernel",
    "extract_magstim_kernel",
    "extract_magventure_kernel",
    "merge_empirical_step1",
    "step2_effective_dose",
    "gamma_scan",
    "freeze_gamma",
    "step3_meanfield",
    "step4_subject_level",
    "step5_primary_stats",
    "step8_adjusted_models",
    "report_tables",
    "main_figures",
    "main_figures_no_electrodes",
    "supplementary_figures",
]


DEFAULT_COHORTS = ["HC", "SZ", "CUD"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Cerebellar Sham Dose pipeline after SimNIBS E-field simulations "
            "have already been completed."
        )
    )

    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=DEFAULT_COHORTS,
        choices=DEFAULT_COHORTS,
        help="Cohorts to run.",
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "paths_local.yaml"),
        help="Path to private local config file.",
    )

    parser.add_argument(
        "--metadata",
        default=str(PROJECT_ROOT / "config" / "cohort_metadata.csv"),
        help="Path to private cohort metadata file.",
    )

    parser.add_argument(
        "--n-workers-step3",
        type=int,
        default=2,
        help="Number of workers for Step 3 mean-field predictions.",
    )

    parser.add_argument(
        "--from-stage",
        default=STAGES[0],
        choices=STAGES,
        help="First stage to run.",
    )

    parser.add_argument(
        "--to-stage",
        default=STAGES[-1],
        choices=STAGES,
        help="Last stage to run.",
    )

    parser.add_argument(
        "--allow-missing-step1",
        action="store_true",
        help="Pass --allow-missing to Step 1 collection.",
    )

    parser.add_argument(
        "--skip-magventure-kernel",
        action="store_true",
        help=(
            "Skip MagVenture empirical-kernel extraction. Use only if the required "
            "magventure_kernel_efield_<cohort>.csv files already exist."
        ),
    )

    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip main and supplementary figure generation.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Run without interactive confirmation.",
    )

    return parser.parse_args()


def git_value(args: list[str]) -> str:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return ""


def selected_stages(from_stage: str, to_stage: str, skip_figures: bool) -> list[str]:
    start = STAGES.index(from_stage)
    end = STAGES.index(to_stage)

    if start > end:
        raise ValueError("--from-stage must come before or equal --to-stage")

    stages = STAGES[start : end + 1]

    if skip_figures:
        stages = [
            s for s in stages
            if s not in {"main_figures", "main_figures_no_electrodes", "supplementary_figures"}
        ]

    return stages


def command_plan(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    py = sys.executable
    config = str(Path(args.config).expanduser())
    metadata = str(Path(args.metadata).expanduser())

    plan: list[tuple[str, list[str]]] = []

    stages = selected_stages(args.from_stage, args.to_stage, args.skip_figures)

    if "metadata" in stages:
        plan.append((
            "metadata",
            [py, "scripts/00_check_metadata.py", "--metadata", metadata],
        ))

    if "collect_step1" in stages:
        for cohort in args.cohorts:
            cmd = [
                py,
                "scripts/01c_collect_step1_outputs.py",
                "--cohort",
                cohort,
                "--config",
                config,
                "--metadata",
                metadata,
            ]
            if args.allow_missing_step1:
                cmd.append("--allow-missing")
            plan.append((f"collect_step1_{cohort}", cmd))

    if "target_anatomy" in stages:
        for cohort in args.cohorts:
            plan.append((
                f"target_anatomy_{cohort}",
                [
                    py,
                    "scripts/01b_extract_target_anatomy.py",
                    "--cohort",
                    cohort,
                    "--config",
                    config,
                    "--metadata",
                    metadata,
                ],
            ))

    if "build_empirical_kernel" in stages:
        plan.append((
            "build_empirical_kernel",
            [
                py,
                "scripts/02c_build_magstim_empirical_kernel.py",
                "--config",
                config,
            ],
        ))

    if "extract_magstim_kernel" in stages:
        for cohort in args.cohorts:
            plan.append((
                f"extract_magstim_kernel_{cohort}",
                [
                    py,
                    "scripts/02d_extract_magstim_kernel_efield.py",
                    "--cohort",
                    cohort,
                    "--config",
                    config,
                    "--metadata",
                    metadata,
                ],
            ))

    if "extract_magventure_kernel" in stages and not args.skip_magventure_kernel:
        for cohort in args.cohorts:
            plan.append((
                f"extract_magventure_kernel_{cohort}",
                [
                    py,
                    "scripts/02f_extract_magventure_kernel_efield.py",
                    "--cohort",
                    cohort,
                    "--config",
                    config,
                    "--metadata",
                    metadata,
                ],
            ))

    if "merge_empirical_step1" in stages:
        plan.append((
            "merge_empirical_step1",
            [
                py,
                "scripts/02e_build_step1_empirical_kernel_tables.py",
                "--cohort",
                "ALL",
                "--config",
                config,
                "--metadata",
                metadata,
                "--promote",
            ],
        ))

    if "step2_effective_dose" in stages:
        plan.append((
            "step2_effective_dose",
            [
                py,
                "scripts/02_compute_membrane_filtered_dose.py",
                "--cohort",
                "ALL",
                "--config",
                config,
            ],
        ))

    if "gamma_scan" in stages:
        plan.append((
            "gamma_scan",
            [
                py,
                "scripts/02b_scan_gamma_inputs.py",
                "--cohort",
                "ALL",
                "--config",
                config,
            ],
        ))

    if "freeze_gamma" in stages:
        plan.append((
            "freeze_gamma",
            [
                py,
                "scripts/02g_freeze_primary_gamma.py",
                "--config",
                config,
            ],
        ))

    if "step3_meanfield" in stages:
        for cohort in args.cohorts:
            plan.append((
                f"step3_meanfield_{cohort}",
                [
                    py,
                    "scripts/03_run_meanfield_predictions.py",
                    "--cohort",
                    cohort,
                    "--config",
                    config,
                    "--n-workers",
                    str(args.n_workers_step3),
                    "--overwrite",
                ],
            ))

    if "step4_subject_level" in stages:
        plan.append((
            "step4_subject_level",
            [
                py,
                "scripts/04_build_subject_level_table.py",
                "--config",
                config,
            ],
        ))

    if "step5_primary_stats" in stages:
        plan.append((
            "step5_primary_stats",
            [
                py,
                "scripts/05_run_primary_statistics.py",
                "--config",
                config,
            ],
        ))

    if "step8_adjusted_models" in stages:
        plan.append((
            "step8_adjusted_models",
            [
                py,
                "scripts/08_run_discrete_time_survival_models.py",
                "--config",
                config,
            ],
        ))

    if "report_tables" in stages:
        plan.append((
            "report_tables",
            [
                py,
                "scripts/05b_make_report_ready_tables.py",
                "--config",
                config,
            ],
        ))

    if "main_figures" in stages:
        for cohort in args.cohorts:
            plan.append((
                f"main_figures_{cohort}",
                [
                    py,
                    "scripts/06_make_main_figures.py",
                    "--cohort",
                    cohort,
                    "--config",
                    config,
                ],
            ))

    if "main_figures_no_electrodes" in stages:
        plan.append((
            "main_figures_no_electrodes",
            [
                py,
                "scripts/06b_make_main_figures_no_electrodes.py",
                "--cohort",
                "ALL",
                "--config",
                config,
            ],
        ))

    if "supplementary_figures" in stages:
        plan.append((
            "supplementary_figures",
            [
                py,
                "scripts/07_make_supplementary_figures.py",
                "--figure",
                "all",
                "--config",
                config,
            ],
        ))

    return plan


def stream_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as log:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log.write(line)

        return process.wait()


def main() -> None:
    args = parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "logs" / f"full_pipeline_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    stages = selected_stages(args.from_stage, args.to_stage, args.skip_figures)
    plan = command_plan(args)

    manifest = {
        "run_id": run_id,
        "project_root": str(PROJECT_ROOT),
        "python": sys.executable,
        "cohorts": args.cohorts,
        "from_stage": args.from_stage,
        "to_stage": args.to_stage,
        "stages": stages,
        "dry_run": bool(args.dry_run),
        "git_branch": git_value(["branch", "--show-current"]),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_status_short": git_value(["status", "--short"]),
        "commands": [
            {"name": name, "command": cmd}
            for name, cmd in plan
        ],
    }

    manifest_path = log_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n[POST-SIMNIBS FULL PIPELINE RUNNER]")
    print(f"Run ID: {run_id}")
    print(f"Log dir: {log_dir}")
    print(f"Cohorts: {', '.join(args.cohorts)}")
    print(f"Stages: {args.from_stage} -> {args.to_stage}")
    print(f"Dry run: {args.dry_run}")
    print(f"Manifest: {manifest_path}")

    print("\n[PLANNED COMMANDS]")
    for i, (name, cmd) in enumerate(plan, start=1):
        print(f"{i:02d}. {name}")
        print("    " + " ".join(cmd))

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] No commands were executed.")
        return

    if not args.yes:
        response = input("\nRun these commands? Type YES to continue: ")
        if response != "YES":
            print("Aborted.")
            return

    results = []

    for i, (name, cmd) in enumerate(plan, start=1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(plan)}] {name}")
        print("=" * 80)

        log_path = log_dir / f"{i:02d}_{name}.log"
        rc = stream_command(cmd, log_path)

        results.append(
            {
                "name": name,
                "command": cmd,
                "returncode": rc,
                "log": str(log_path),
            }
        )

        manifest["results"] = results
        manifest_path.write_text(json.dumps(manifest, indent=2))

        if rc != 0:
            print(f"\n[FAILED] {name} returned exit code {rc}")
            print(f"Log: {log_path}")
            sys.exit(rc)

    print("\n[OK] Full post-SimNIBS pipeline completed successfully.")
    print(f"Logs and manifest saved in: {log_dir}")


if __name__ == "__main__":
    main()
