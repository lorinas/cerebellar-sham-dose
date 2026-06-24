from __future__ import annotations

from pathlib import Path

import pandas as pd

from shamdose.metadata import ALLOWED_COHORTS


DEFAULT_STEP1_CSV_NAME = "STEP1_E0_inion2cmBelow_roi10mm_allstats_scaled.csv"


MAIN_REQUIRED_CONDITIONS = [
    "MagVenture_Sham0mm",
    "MagVenture_Sham0mm+electrodes",
    "tDCS",
]


def step1_csv_path(
    headmodels_root: str | Path,
    subject_id: str,
    step1_run_subdir: str,
    step1_csv_name: str = DEFAULT_STEP1_CSV_NAME,
) -> Path:
    """
    Return the expected per-subject Step 1 output CSV path.
    """
    return (
        Path(headmodels_root).expanduser().resolve()
        / subject_id
        / step1_run_subdir
        / step1_csv_name
    )


def collect_step1_outputs_for_cohort(
    metadata: pd.DataFrame,
    cohort: str,
    headmodels_root: str | Path,
    step1_run_subdir: str,
    step1_csv_name: str = DEFAULT_STEP1_CSV_NAME,
    require_all_subjects: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect per-subject Step 1 E-field CSVs into one cohort-level table.

    Parameters
    ----------
    metadata:
        Clean metadata dataframe from load_metadata().
    cohort:
        HC, SZ, or CUD.
    headmodels_root:
        Root folder containing sub-*/ folders for this cohort.
    step1_run_subdir:
        Per-subject Step 1 output folder name.
    step1_csv_name:
        Per-subject Step 1 CSV file name.
    require_all_subjects:
        If True, raise an error if any included subject is missing Step 1 output.

    Returns
    -------
    combined:
        Concatenated Step 1 E-field table for included subjects.
    manifest:
        QC table indicating which subjects were found or missing.
    """
    cohort = cohort.upper()

    if cohort not in ALLOWED_COHORTS:
        raise ValueError(f"Unknown cohort: {cohort}. Allowed: {ALLOWED_COHORTS}")

    sub_meta = metadata[
        (metadata["cohort"] == cohort)
        & (metadata["include_primary"] == 1)
    ].copy()

    if sub_meta.empty:
        raise ValueError(f"No included subjects found in metadata for cohort: {cohort}")

    data_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for _, meta_row in sub_meta.iterrows():
        subject_id = str(meta_row["subject_id"])
        csv_path = step1_csv_path(
            headmodels_root=headmodels_root,
            subject_id=subject_id,
            step1_run_subdir=step1_run_subdir,
            step1_csv_name=step1_csv_name,
        )

        found = csv_path.exists()

        manifest_rows.append(
            {
                "subject_id": subject_id,
                "cohort": cohort,
                "step1_csv": str(csv_path),
                "step1_csv_found": int(found),
                "status": "FOUND" if found else "MISSING_STEP1_CSV",
            }
        )

        if not found:
            continue

        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError(f"Step 1 CSV is empty: {csv_path}")

        if "subject_id" not in df.columns:
            raise ValueError(f"Step 1 CSV missing subject_id column: {csv_path}")

        csv_subjects = sorted(df["subject_id"].astype(str).unique().tolist())
        if csv_subjects != [subject_id]:
            raise ValueError(
                f"Step 1 CSV subject mismatch for {subject_id}:\n"
                f"    path: {csv_path}\n"
                f"    subject_id values inside file: {csv_subjects}"
            )

        # Add metadata and source information.
        df.insert(0, "cohort", cohort)
        df.insert(1, "include_primary", int(meta_row["include_primary"]))
        df.insert(2, "sex", meta_row["sex"])
        df.insert(3, "age", meta_row["age"])
        df["source_step1_csv"] = str(csv_path)

        data_frames.append(df)

    manifest = pd.DataFrame(manifest_rows)

    missing = manifest[manifest["step1_csv_found"] == 0]["subject_id"].tolist()
    if missing and require_all_subjects:
        raise FileNotFoundError(
            f"{cohort}: missing Step 1 CSVs for {len(missing)} included subjects:\n"
            f"{missing[:30]}"
        )

    if not data_frames:
        raise RuntimeError(f"No Step 1 CSVs were collected for cohort: {cohort}")

    combined = pd.concat(data_frames, ignore_index=True)

    # If all subjects are required, validate against the full expected cohort.
    # If missing Step 1 CSVs are allowed, validate only against subjects whose
    # Step 1 CSVs were actually found. This is useful while HC is incomplete.
    if require_all_subjects:
        expected_subjects_for_validation = set(sub_meta["subject_id"].astype(str))
    else:
        expected_subjects_for_validation = set(
            manifest.loc[
                manifest["step1_csv_found"] == 1,
                "subject_id",
            ].astype(str)
        )

    validate_collected_step1_table(
        combined=combined,
        cohort=cohort,
        expected_subjects=expected_subjects_for_validation,
    )

    return combined, manifest


def validate_collected_step1_table(
    combined: pd.DataFrame,
    cohort: str,
    expected_subjects: set[str],
) -> None:
    """
    Basic sanity checks for collected Step 1 E-field table.
    """
    errors: list[str] = []

    observed_subjects = set(combined["subject_id"].astype(str).unique())

    missing_subjects = sorted(expected_subjects - observed_subjects)
    extra_subjects = sorted(observed_subjects - expected_subjects)

    if missing_subjects:
        errors.append(f"Missing expected subjects in collected table: {missing_subjects[:30]}")

    if extra_subjects:
        errors.append(f"Unexpected extra subjects in collected table: {extra_subjects[:30]}")

    if "condition" not in combined.columns:
        errors.append("Collected Step 1 table missing condition column.")
    else:
        observed_conditions = set(combined["condition"].astype(str).unique())
        missing_conditions = [
            cond for cond in MAIN_REQUIRED_CONDITIONS
            if cond not in observed_conditions
        ]
        if missing_conditions:
            errors.append(
                f"Missing required main conditions: {missing_conditions}. "
                f"Observed conditions: {sorted(observed_conditions)}"
            )

    if "intensity (%MT)" not in combined.columns:
        errors.append("Collected Step 1 table missing intensity (%MT) column.")

    if errors:
        msg = "\n".join(f"- {err}" for err in errors)
        raise ValueError(f"Collected Step 1 validation failed for {cohort}:\n{msg}")


def summarize_collected_step1(combined: pd.DataFrame, manifest: pd.DataFrame, cohort: str) -> None:
    """
    Print useful QC summary for collected Step 1 outputs.
    """
    print("\n[COLLECTED STEP 1 SUMMARY]")
    print(f"Cohort: {cohort}")
    print(f"Subjects in collected table: {combined['subject_id'].nunique()}")
    print(f"Rows in collected table: {len(combined)}")
    print(f"Missing Step 1 CSVs: {int((manifest['step1_csv_found'] == 0).sum())}")

    if "condition" in combined.columns:
        print("\n[ROWS BY CONDITION]")
        print(combined["condition"].value_counts().sort_index().to_string())

    if "intensity (%MT)" in combined.columns:
        print("\n[INTENSITIES BY CONDITION]")
        intensity_summary = (
            combined.groupby("condition")["intensity (%MT)"]
            .apply(lambda x: sorted(set(x.dropna().astype(int))))
        )
        print(intensity_summary.to_string())
