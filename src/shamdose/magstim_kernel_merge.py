from __future__ import annotations

from pathlib import Path

import pandas as pd


OLD_MAGSTIM_SCALAR_CONDITION = "Magstim_Sham0mm"
NEW_MAGSTIM_KERNEL_CONDITION = "Magstim_Sham0mm_kernel"

EXPECTED_KERNEL_STEP1_CONDITIONS = [
    "MagVenture_Active0mm",
    "MagVenture_Sham0mm",
    "MagVenture_Sham0mm+electrodes",
    "Magstim_Active0mm",
    "Magstim_Sham0mm_kernel",
    "tDCS",
]

TMS_INTENSITIES = list(range(0, 101, 10))
TDCS_INTENSITIES = [100]


def build_step1_magstim_kernel_table(
    step1: pd.DataFrame,
    kernel_rows: pd.DataFrame,
    metadata: pd.DataFrame,
    cohort: str,
) -> pd.DataFrame:
    """
    Build a kernel-updated Step 1 table.

    This removes the old scalar Magstim sham condition:
        Magstim_Sham0mm

    and appends the empirical-kernel condition:
        Magstim_Sham0mm_kernel

    The output keeps the same overall Step 1 row structure:
        5 TMS conditions × 11 intensities + 1 tDCS row = 56 rows per subject
    """
    cohort = cohort.upper()

    step1 = step1.copy()
    kernel_rows = kernel_rows.copy()
    metadata = metadata.copy()

    step1["subject_id"] = step1["subject_id"].astype(str)
    step1["condition"] = step1["condition"].astype(str)

    kernel_rows["subject_id"] = kernel_rows["subject_id"].astype(str)
    kernel_rows["condition"] = kernel_rows["condition"].astype(str)

    expected_subjects = (
        metadata[
            (metadata["cohort"].astype(str).str.upper() == cohort)
            & (metadata["include_primary"] == 1)
        ]["subject_id"]
        .astype(str)
        .tolist()
    )

    expected_set = set(expected_subjects)

    if not expected_subjects:
        raise ValueError(f"No included subjects found for cohort {cohort}")

    original_subjects = set(step1["subject_id"].unique())
    kernel_subjects = set(kernel_rows["subject_id"].unique())

    missing_kernel_subjects = sorted(expected_set - kernel_subjects)
    extra_kernel_subjects = sorted(kernel_subjects - expected_set)

    if missing_kernel_subjects:
        raise ValueError(
            f"{cohort}: kernel rows missing expected subjects: {missing_kernel_subjects[:20]}"
        )

    if extra_kernel_subjects:
        raise ValueError(
            f"{cohort}: kernel rows contain unexpected subjects: {extra_kernel_subjects[:20]}"
        )

    missing_step1_subjects = sorted(expected_set - original_subjects)
    if missing_step1_subjects:
        raise ValueError(
            f"{cohort}: original Step 1 table missing expected subjects: {missing_step1_subjects[:20]}"
        )

    # Validate kernel condition before merging.
    observed_kernel_conditions = sorted(kernel_rows["condition"].unique().tolist())
    if observed_kernel_conditions != [NEW_MAGSTIM_KERNEL_CONDITION]:
        raise ValueError(
            f"{cohort}: kernel rows should contain only {NEW_MAGSTIM_KERNEL_CONDITION}; "
            f"observed {observed_kernel_conditions}"
        )

    kernel_counts = kernel_rows.groupby("subject_id").size()
    bad_kernel_counts = kernel_counts[kernel_counts != 11]
    if len(bad_kernel_counts) > 0:
        raise ValueError(
            f"{cohort}: kernel subjects with row count != 11:\n"
            f"{bad_kernel_counts.head(20).to_string()}"
        )

    kernel_intensities = sorted(
        pd.to_numeric(kernel_rows["intensity (%MT)"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if kernel_intensities != TMS_INTENSITIES:
        raise ValueError(
            f"{cohort}: unexpected kernel intensities {kernel_intensities}"
        )

    # Remove deprecated old scalar Magstim sham rows.
    old_scalar_rows = step1[step1["condition"] == OLD_MAGSTIM_SCALAR_CONDITION].copy()

    expected_old_scalar_rows = len(expected_subjects) * 11
    if len(old_scalar_rows) != expected_old_scalar_rows:
        raise ValueError(
            f"{cohort}: expected {expected_old_scalar_rows} old scalar Magstim rows, "
            f"found {len(old_scalar_rows)}"
        )

    step1_without_old_magstim = step1[
        step1["condition"] != OLD_MAGSTIM_SCALAR_CONDITION
    ].copy()

    out = pd.concat(
        [step1_without_old_magstim, kernel_rows],
        ignore_index=True,
        sort=False,
    )

    validate_kernel_step1_table(
        out,
        cohort=cohort,
        expected_subjects=expected_subjects,
    )

    return out


def validate_kernel_step1_table(
    df: pd.DataFrame,
    cohort: str,
    expected_subjects: list[str],
) -> None:
    """
    Validate the final kernel-updated Step 1 table.
    """
    errors: list[str] = []

    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["condition"] = df["condition"].astype(str)

    expected_set = set(expected_subjects)
    observed_set = set(df["subject_id"].unique())

    missing_subjects = sorted(expected_set - observed_set)
    extra_subjects = sorted(observed_set - expected_set)

    if missing_subjects:
        errors.append(f"Missing subjects: {missing_subjects[:20]}")

    if extra_subjects:
        errors.append(f"Unexpected subjects: {extra_subjects[:20]}")

    if OLD_MAGSTIM_SCALAR_CONDITION in set(df["condition"].unique()):
        errors.append(
            f"Deprecated condition still present: {OLD_MAGSTIM_SCALAR_CONDITION}"
        )

    observed_conditions = sorted(df["condition"].unique().tolist())
    missing_conditions = [
        c for c in EXPECTED_KERNEL_STEP1_CONDITIONS
        if c not in observed_conditions
    ]
    extra_conditions = [
        c for c in observed_conditions
        if c not in EXPECTED_KERNEL_STEP1_CONDITIONS
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

    for condition in EXPECTED_KERNEL_STEP1_CONDITIONS:
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
            f"Kernel-updated Step 1 validation failed for {cohort}:\n"
            + "\n".join(f"- {e}" for e in errors)
        )


def print_kernel_step1_summary(df: pd.DataFrame, cohort: str) -> None:
    """
    Print concise summary of kernel-updated Step 1 table.
    """
    print("\n[KERNEL-UPDATED STEP 1 SUMMARY]")
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
