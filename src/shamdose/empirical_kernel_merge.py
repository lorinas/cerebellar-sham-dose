from __future__ import annotations

from pathlib import Path
import shutil
from datetime import datetime

import pandas as pd


DEPRECATED_MAGSTIM_SCALAR_CONDITION = "Magstim_Sham0mm"
MAGSTIM_KERNEL_CONDITION = "Magstim_Sham0mm_kernel"

MAGVENTURE_SHAM_CONDITION = "MagVenture_Sham0mm"
MAGVENTURE_SHAM_ELEC_CONDITION = "MagVenture_Sham0mm+electrodes"

EXPECTED_EMPIRICAL_STEP1_CONDITIONS = [
    "MagVenture_Active0mm",
    "MagVenture_Sham0mm",
    "MagVenture_Sham0mm+electrodes",
    "Magstim_Active0mm",
    "Magstim_Sham0mm_kernel",
    "tDCS",
]

TMS_INTENSITIES = list(range(0, 101, 10))
TDCS_INTENSITIES = [100]


def included_subjects(metadata: pd.DataFrame, cohort: str) -> list[str]:
    cohort = cohort.upper()

    subjects = (
        metadata[
            (metadata["cohort"].astype(str).str.upper() == cohort)
            & (metadata["include_primary"] == 1)
        ]["subject_id"]
        .astype(str)
        .tolist()
    )

    if not subjects:
        raise ValueError(f"No included subjects found for cohort {cohort}")

    return subjects


def validate_kernel_condition_rows(
    rows: pd.DataFrame,
    expected_subjects: list[str],
    expected_conditions: list[str],
    cohort: str,
    label: str,
) -> None:
    rows = rows.copy()
    rows["subject_id"] = rows["subject_id"].astype(str)
    rows["condition"] = rows["condition"].astype(str)

    errors: list[str] = []

    expected_set = set(expected_subjects)
    observed_set = set(rows["subject_id"].unique())

    missing_subjects = sorted(expected_set - observed_set)
    extra_subjects = sorted(observed_set - expected_set)

    if missing_subjects:
        errors.append(f"{label}: missing subjects: {missing_subjects[:20]}")

    if extra_subjects:
        errors.append(f"{label}: unexpected subjects: {extra_subjects[:20]}")

    observed_conditions = sorted(rows["condition"].unique().tolist())

    missing_conditions = [c for c in expected_conditions if c not in observed_conditions]
    extra_conditions = [c for c in observed_conditions if c not in expected_conditions]

    if missing_conditions:
        errors.append(f"{label}: missing expected conditions: {missing_conditions}")

    if extra_conditions:
        errors.append(f"{label}: unexpected conditions: {extra_conditions}")

    for condition in expected_conditions:
        sub = rows[rows["condition"] == condition].copy()
        expected_rows = len(expected_subjects) * 11

        if len(sub) != expected_rows:
            errors.append(
                f"{label}: {condition} expected {expected_rows} rows, observed {len(sub)}"
            )

        intensities = sorted(
            pd.to_numeric(sub["intensity (%MT)"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if intensities != TMS_INTENSITIES:
            errors.append(
                f"{label}: {condition} expected intensities {TMS_INTENSITIES}, observed {intensities}"
            )

    per_subject = rows.groupby(["subject_id", "condition"]).size()
    bad = per_subject[per_subject != 11]

    if len(bad) > 0:
        errors.append(
            f"{label}: subject × condition row counts not equal to 11:\n"
            + bad.head(20).to_string()
        )

    if errors:
        raise ValueError(
            f"Kernel row validation failed for {cohort}:\n"
            + "\n".join(f"- {e}" for e in errors)
        )


def build_step1_empirical_kernel_table(
    step1: pd.DataFrame,
    magstim_kernel_rows: pd.DataFrame,
    magventure_kernel_rows: pd.DataFrame,
    metadata: pd.DataFrame,
    cohort: str,
) -> pd.DataFrame:
    """
    Build empirical-kernel Step 1 table.

    This replaces:
        Magstim_Sham0mm scalar rows
        any existing Magstim_Sham0mm_kernel rows
        MagVenture_Sham0mm rows
        MagVenture_Sham0mm+electrodes rows

    with:
        Magstim_Sham0mm_kernel empirical rows
        MagVenture_Sham0mm empirical rows
        MagVenture_Sham0mm+electrodes empirical rows

    The output keeps 56 rows per subject:
        5 TMS conditions × 11 intensities + 1 tDCS row
    """
    cohort = cohort.upper()

    step1 = step1.copy()
    magstim_kernel_rows = magstim_kernel_rows.copy()
    magventure_kernel_rows = magventure_kernel_rows.copy()

    step1["subject_id"] = step1["subject_id"].astype(str)
    step1["condition"] = step1["condition"].astype(str)

    expected_subjects = included_subjects(metadata, cohort)

    expected_set = set(expected_subjects)
    observed_set = set(step1["subject_id"].unique())

    missing_step1_subjects = sorted(expected_set - observed_set)
    if missing_step1_subjects:
        raise ValueError(
            f"{cohort}: original Step 1 table missing expected subjects: "
            f"{missing_step1_subjects[:20]}"
        )

    validate_kernel_condition_rows(
        rows=magstim_kernel_rows,
        expected_subjects=expected_subjects,
        expected_conditions=[MAGSTIM_KERNEL_CONDITION],
        cohort=cohort,
        label="Magstim kernel rows",
    )

    validate_kernel_condition_rows(
        rows=magventure_kernel_rows,
        expected_subjects=expected_subjects,
        expected_conditions=[MAGVENTURE_SHAM_CONDITION, MAGVENTURE_SHAM_ELEC_CONDITION],
        cohort=cohort,
        label="MagVenture kernel rows",
    )

    remove_conditions = [
        DEPRECATED_MAGSTIM_SCALAR_CONDITION,
        MAGSTIM_KERNEL_CONDITION,
        MAGVENTURE_SHAM_CONDITION,
        MAGVENTURE_SHAM_ELEC_CONDITION,
    ]

    base = step1[~step1["condition"].isin(remove_conditions)].copy()

    out = pd.concat(
        [base, magventure_kernel_rows, magstim_kernel_rows],
        ignore_index=True,
        sort=False,
    )

    validate_empirical_step1_table(
        out,
        cohort=cohort,
        expected_subjects=expected_subjects,
    )

    return out


def validate_empirical_step1_table(
    df: pd.DataFrame,
    cohort: str,
    expected_subjects: list[str],
) -> None:
    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["condition"] = df["condition"].astype(str)

    errors: list[str] = []

    expected_set = set(expected_subjects)
    observed_set = set(df["subject_id"].unique())

    missing_subjects = sorted(expected_set - observed_set)
    extra_subjects = sorted(observed_set - expected_set)

    if missing_subjects:
        errors.append(f"Missing subjects: {missing_subjects[:20]}")

    if extra_subjects:
        errors.append(f"Unexpected subjects: {extra_subjects[:20]}")

    if DEPRECATED_MAGSTIM_SCALAR_CONDITION in set(df["condition"].unique()):
        errors.append(
            f"Deprecated scalar condition still present: {DEPRECATED_MAGSTIM_SCALAR_CONDITION}"
        )

    observed_conditions = sorted(df["condition"].unique().tolist())

    missing_conditions = [
        c for c in EXPECTED_EMPIRICAL_STEP1_CONDITIONS
        if c not in observed_conditions
    ]

    extra_conditions = [
        c for c in observed_conditions
        if c not in EXPECTED_EMPIRICAL_STEP1_CONDITIONS
    ]

    if missing_conditions:
        errors.append(f"Missing expected conditions: {missing_conditions}")

    if extra_conditions:
        errors.append(f"Unexpected conditions: {extra_conditions}")

    per_subject_counts = df.groupby("subject_id").size()
    bad_subjects = per_subject_counts[per_subject_counts != 56]

    if len(bad_subjects) > 0:
        errors.append(
            "Subjects with row count != 56:\n"
            + bad_subjects.head(20).to_string()
        )

    for condition in EXPECTED_EMPIRICAL_STEP1_CONDITIONS:
        sub = df[df["condition"] == condition].copy()

        if condition == "tDCS":
            expected_rows = len(expected_subjects)
            expected_intensities = TDCS_INTENSITIES
        else:
            expected_rows = len(expected_subjects) * 11
            expected_intensities = TMS_INTENSITIES

        if len(sub) != expected_rows:
            errors.append(
                f"{condition}: expected {expected_rows} rows, observed {len(sub)}"
            )

        observed_intensities = sorted(
            pd.to_numeric(sub["intensity (%MT)"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if observed_intensities != expected_intensities:
            errors.append(
                f"{condition}: expected intensities {expected_intensities}, "
                f"observed {observed_intensities}"
            )

    if errors:
        raise ValueError(
            f"Empirical-kernel Step 1 validation failed for {cohort}:\n"
            + "\n".join(f"- {e}" for e in errors)
        )


def print_empirical_step1_summary(df: pd.DataFrame, cohort: str) -> None:
    print("\n[EMPIRICAL-KERNEL STEP 1 SUMMARY]")
    print(f"Cohort: {cohort}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Rows: {len(df)}")

    print("\nRows by condition:")
    print(df["condition"].value_counts().sort_index().to_string())

    print("\nIntensities by condition:")
    intensity_summary = (
        df.groupby("condition")["intensity (%MT)"]
        .apply(lambda x: sorted(set(pd.to_numeric(x, errors='coerce').dropna().astype(int))))
    )
    print(intensity_summary.to_string())


def archive_and_promote_step1_table(
    empirical_table_path: str | Path,
    canonical_step1_path: str | Path,
    archive_root: str | Path,
    cohort: str,
) -> None:
    """
    Promote empirical-kernel Step 1 table to canonical Step 1 path.

    This archives the previous canonical Step 1 table first.
    """
    empirical_table_path = Path(empirical_table_path)
    canonical_step1_path = Path(canonical_step1_path)
    archive_root = Path(archive_root)

    if not empirical_table_path.exists():
        raise FileNotFoundError(f"Missing empirical Step 1 table: {empirical_table_path}")

    if not canonical_step1_path.exists():
        raise FileNotFoundError(f"Missing canonical Step 1 table: {canonical_step1_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = archive_root / f"pre_empirical_kernel_step1_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    archived_path = archive_dir / f"step1_efield_{cohort}_before_empirical_kernel.csv"

    shutil.copy2(canonical_step1_path, archived_path)
    shutil.copy2(empirical_table_path, canonical_step1_path)

    readme = archive_dir / "README.md"
    readme.write_text(
        "# Archived Step 1 table before empirical-kernel promotion\n\n"
        "This table was archived before promoting the empirical-kernel Step 1 table "
        "to the canonical Step 1 path.\n"
    )

    print(f"[OK] Archived previous canonical Step 1 table: {archived_path}")
    print(f"[OK] Promoted empirical Step 1 table to: {canonical_step1_path}")
