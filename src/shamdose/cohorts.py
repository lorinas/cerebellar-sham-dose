from __future__ import annotations

from pathlib import Path

import pandas as pd

from shamdose.metadata import ALLOWED_COHORTS, subjects_for_cohort


DEFAULT_STEP1_CSV_NAME = "STEP1_E0_inion2cmBelow_roi10mm_allstats_scaled.csv"

EXPECTED_STEP1_CONDITIONS = [
    "MagVenture_Active0mm",
    "MagVenture_Sham0mm",
    "MagVenture_Sham0mm+electrodes",
    "Magstim_Active0mm",
    "Magstim_Sham0mm",
    "tDCS",
]

EXPECTED_TMS_INTENSITIES = list(range(0, 101, 10))
EXPECTED_TDCS_INTENSITIES = [100]
EXPECTED_ROWS_PER_SUBJECT = 56


def find_m2m_subjects(headmodels_root: str | Path) -> pd.DataFrame:
    """
    Find SimNIBS m2m folders under a cohort headmodels root.

    Expected folder pattern:
        headmodels_root/sub-XXX/m2m_*

    Important:
        This function only reports what exists on disk.
        It does NOT decide cohort membership.
        Cohort membership comes from cohort_metadata.csv.
    """
    root = Path(headmodels_root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Headmodels root not found: {root}")

    rows: list[dict[str, str]] = []

    for m2m_dir in sorted(root.glob("sub-*/m2m_*")):
        if not m2m_dir.is_dir():
            continue

        subject_dir = m2m_dir.parent
        subject_id = subject_dir.name

        rows.append(
            {
                "subject_id": subject_id,
                "subject_dir": str(subject_dir),
                "m2m_dir": str(m2m_dir),
            }
        )

    return pd.DataFrame(rows)


def validate_step1_csv_basic(csv_path: str | Path, subject_id: str) -> tuple[bool, str]:
    """
    Basic validation for an existing per-subject Step 1 CSV.

    This prevents a partial/crashed output file from being treated as complete.

    Checks:
        - file can be read
        - expected row count = 56
        - subject_id column exists and matches requested subject
        - condition column exists
        - all expected conditions are present
        - TMS conditions have intensities 0..100 by 10
        - tDCS has intensity 100
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return False, "MISSING"

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return False, f"READ_ERROR: {exc}"

    if df.empty:
        return False, "EMPTY_CSV"

    if len(df) != EXPECTED_ROWS_PER_SUBJECT:
        return False, f"BAD_ROW_COUNT_N={len(df)}_EXPECTED_{EXPECTED_ROWS_PER_SUBJECT}"

    if "subject_id" not in df.columns:
        return False, "MISSING_subject_id_COLUMN"

    observed_subjects = sorted(df["subject_id"].astype(str).unique().tolist())
    if observed_subjects != [subject_id]:
        return False, f"SUBJECT_ID_MISMATCH_OBSERVED_{observed_subjects}"

    if "condition" not in df.columns:
        return False, "MISSING_condition_COLUMN"

    if "intensity (%MT)" not in df.columns:
        return False, "MISSING_intensity_COLUMN"

    observed_conditions = set(df["condition"].astype(str).unique())
    missing_conditions = [
        cond for cond in EXPECTED_STEP1_CONDITIONS
        if cond not in observed_conditions
    ]

    if missing_conditions:
        return False, f"MISSING_CONDITIONS_{missing_conditions}"

    for cond in EXPECTED_STEP1_CONDITIONS:
        sub = df[df["condition"].astype(str) == cond].copy()

        if cond == "tDCS":
            expected_intensities = EXPECTED_TDCS_INTENSITIES
        else:
            expected_intensities = EXPECTED_TMS_INTENSITIES

        observed_intensities = sorted(
            sub["intensity (%MT)"].dropna().astype(int).unique().tolist()
        )

        if observed_intensities != expected_intensities:
            return (
                False,
                f"BAD_INTENSITIES_{cond}_OBSERVED_{observed_intensities}_EXPECTED_{expected_intensities}",
            )

    return True, "VALID"


def build_step1_preflight_table(
    metadata: pd.DataFrame,
    cohort: str,
    headmodels_root: str | Path,
    step1_run_subdir: str,
    step1_csv_name: str = DEFAULT_STEP1_CSV_NAME,
) -> pd.DataFrame:
    """
    Build a Step 1 preflight table for one cohort.

    This checks:
        - which subjects are expected from cohort_metadata.csv
        - which expected subjects have m2m folders
        - which expected subjects already have valid Step 1 output CSVs
        - which expected subjects need Step 1 simulations
        - which m2m folders exist on disk but are not included in this cohort

    The output is a QC table. It is not a scientific analysis result.
    """
    cohort = cohort.upper()

    if cohort not in ALLOWED_COHORTS:
        raise ValueError(f"Unknown cohort: {cohort}. Allowed: {ALLOWED_COHORTS}")

    root = Path(headmodels_root).expanduser().resolve()

    expected_subjects = subjects_for_cohort(metadata, cohort, primary_only=True)
    expected_set = set(expected_subjects)

    all_metadata = metadata.copy()
    metadata_by_subject = all_metadata.set_index("subject_id", drop=False)

    all_m2m = find_m2m_subjects(root)

    rows: list[dict[str, object]] = []

    # Rows for subjects expected from metadata.
    for subject_id in expected_subjects:
        subject_dir = root / subject_id
        m2m_candidates = sorted(subject_dir.glob("m2m_*")) if subject_dir.exists() else []

        m2m_found = len(m2m_candidates) > 0
        multiple_m2m = len(m2m_candidates) > 1
        m2m_dir = m2m_candidates[0] if m2m_found else None

        step1_dir = subject_dir / step1_run_subdir
        step1_csv = step1_dir / step1_csv_name

        step1_csv_exists = step1_csv.exists()
        step1_csv_valid, step1_validation_message = validate_step1_csv_basic(
            step1_csv,
            subject_id,
        )

        if not m2m_found:
            status = "MISSING_M2M"
            needs_step1 = False
        elif step1_csv_valid:
            status = "ALREADY_PROCESSED"
            needs_step1 = False
        elif step1_csv_exists and not step1_csv_valid:
            status = "EXISTING_STEP1_INVALID"
            needs_step1 = True
        else:
            status = "READY_TO_RUN"
            needs_step1 = True

        if multiple_m2m:
            status = status + "_MULTIPLE_M2M"

        rows.append(
            {
                "row_type": "EXPECTED_FROM_METADATA",
                "requested_cohort": cohort,
                "subject_id": subject_id,
                "metadata_cohort": cohort,
                "include_primary": 1,
                "subject_dir": str(subject_dir),
                "m2m_found": int(m2m_found),
                "n_m2m_dirs": len(m2m_candidates),
                "m2m_dir": str(m2m_dir) if m2m_dir is not None else "",
                "step1_dir": str(step1_dir),
                "step1_csv": str(step1_csv),
                "step1_csv_exists": int(step1_csv_exists),
                "step1_csv_valid": int(step1_csv_valid),
                "step1_validation_message": step1_validation_message,
                "needs_step1": int(needs_step1),
                "status": status,
            }
        )

    # Extra m2m folders found on disk but not expected for this cohort.
    if not all_m2m.empty:
        for _, m2m_row in all_m2m.iterrows():
            subject_id = str(m2m_row["subject_id"])

            if subject_id in expected_set:
                continue

            if subject_id in metadata_by_subject.index:
                meta_row = metadata_by_subject.loc[subject_id]
                metadata_cohort = str(meta_row["cohort"])
                include_primary = int(meta_row["include_primary"])
            else:
                metadata_cohort = ""
                include_primary = ""

            rows.append(
                {
                    "row_type": "EXTRA_M2M_NOT_INCLUDED_FOR_THIS_COHORT",
                    "requested_cohort": cohort,
                    "subject_id": subject_id,
                    "metadata_cohort": metadata_cohort,
                    "include_primary": include_primary,
                    "subject_dir": str(m2m_row["subject_dir"]),
                    "m2m_found": 1,
                    "n_m2m_dirs": "",
                    "m2m_dir": str(m2m_row["m2m_dir"]),
                    "step1_dir": "",
                    "step1_csv": "",
                    "step1_csv_exists": "",
                    "step1_csv_valid": "",
                    "step1_validation_message": "",
                    "needs_step1": 0,
                    "status": "EXTRA_M2M_IGNORED",
                }
            )

    out = pd.DataFrame(rows)

    row_type_order = {
        "EXPECTED_FROM_METADATA": 0,
        "EXTRA_M2M_NOT_INCLUDED_FOR_THIS_COHORT": 1,
    }
    out["_row_type_order"] = out["row_type"].map(row_type_order).fillna(99)
    out = out.sort_values(["_row_type_order", "subject_id"]).drop(columns=["_row_type_order"])
    out = out.reset_index(drop=True)

    return out


def print_step1_preflight_summary(preflight: pd.DataFrame, cohort: str) -> None:
    """
    Print a readable summary of the Step 1 preflight table.
    """
    cohort = cohort.upper()

    expected = preflight[preflight["row_type"] == "EXPECTED_FROM_METADATA"].copy()
    extra = preflight[preflight["row_type"] == "EXTRA_M2M_NOT_INCLUDED_FOR_THIS_COHORT"].copy()

    print("\n[STEP 1 PREFLIGHT SUMMARY]")
    print(f"Cohort requested: {cohort}")
    print(f"Expected included subjects from metadata: {len(expected)}")
    print(f"Extra m2m folders ignored: {len(extra)}")

    if not expected.empty:
        print("\n[EXPECTED SUBJECT STATUS COUNTS]")
        print(expected["status"].value_counts().to_string())

        n_missing = int((expected["status"].str.contains("MISSING_M2M")).sum())
        n_ready = int(expected["needs_step1"].sum())
        n_done = int((expected["status"] == "ALREADY_PROCESSED").sum())
        n_invalid = int((expected["status"].str.contains("EXISTING_STEP1_INVALID")).sum())

        print("\n[PROCESSING SUMMARY]")
        print(f"Missing m2m folders: {n_missing}")
        print(f"Already processed valid Step 1 outputs: {n_done}")
        print(f"Invalid existing Step 1 outputs: {n_invalid}")
        print(f"Subjects ready/needing Step 1: {n_ready}")

        missing_subjects = expected.loc[
            expected["status"].str.contains("MISSING_M2M"),
            "subject_id",
        ].tolist()

        invalid_subjects = expected.loc[
            expected["status"].str.contains("EXISTING_STEP1_INVALID"),
            ["subject_id", "step1_validation_message"],
        ]

        ready_subjects = expected.loc[
            expected["needs_step1"] == 1,
            "subject_id",
        ].tolist()

        if missing_subjects:
            print("\n[FIRST 20 SUBJECTS MISSING M2M]")
            print(missing_subjects[:20])

        if not invalid_subjects.empty:
            print("\n[INVALID EXISTING STEP 1 OUTPUTS]")
            print(invalid_subjects.head(20).to_string(index=False))

        if ready_subjects:
            print("\n[FIRST 20 SUBJECTS READY/NEEDING STEP 1]")
            print(ready_subjects[:20])

    if not extra.empty:
        print("\n[EXTRA M2M FOLDERS IGNORED]")
        cols = ["subject_id", "metadata_cohort", "include_primary"]
        print(extra[cols].head(20).to_string(index=False))
        if len(extra) > 20:
            print(f"... plus {len(extra) - 20} more")
