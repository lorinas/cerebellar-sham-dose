from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_METADATA_COLUMNS = [
    "subject_id",
    "cohort",
    "include_primary",
    "sex",
    "age",
]

ALLOWED_COHORTS = ["HC", "SZ", "CUD"]
ALLOWED_SEX_CODES = [1, 2]


def get_project_root() -> Path:
    """
    Return the root folder of the cerebellar-sham-dose project.

    This file lives at:
        src/shamdose/metadata.py

    Therefore:
        parents[0] = src/shamdose
        parents[1] = src
        parents[2] = project root
    """
    return Path(__file__).resolve().parents[2]


def default_metadata_path() -> Path:
    """
    Default private metadata file.

    This file is ignored by Git because it contains subject-level metadata.
    """
    return get_project_root() / "config" / "cohort_metadata.csv"


def load_metadata(metadata_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load and validate cohort metadata.

    Expected columns:
        subject_id, cohort, include_primary, sex, age

    Missing values such as NA are allowed for sex and age.
    """
    if metadata_path is None:
        metadata_path = default_metadata_path()
    else:
        metadata_path = Path(metadata_path).expanduser().resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(
        metadata_path,
        na_values=["NA", "Na", "na", "N/A", "n/a", ""],
        keep_default_na=True,
    )

    validate_metadata_columns(df)
    df = clean_metadata_values(df)
    validate_metadata_values(df)

    return df


def validate_metadata_columns(df: pd.DataFrame) -> None:
    """
    Check that all required columns exist.
    """
    missing = [col for col in REQUIRED_METADATA_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            "Metadata file is missing required columns:\n"
            f"    {missing}\n"
            "Expected columns:\n"
            f"    {REQUIRED_METADATA_COLUMNS}"
        )


def clean_metadata_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column values.

    - Strip whitespace from subject_id and cohort.
    - Convert include_primary, sex, and age to numeric.
    """
    df = df.copy()

    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["cohort"] = df["cohort"].astype(str).str.strip().str.upper()

    df["include_primary"] = pd.to_numeric(df["include_primary"], errors="coerce")
    df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    return df


def validate_metadata_values(df: pd.DataFrame) -> None:
    """
    Validate cohort labels, include flags, sex codes, age values, and duplicates.
    """
    errors: list[str] = []

    if df["subject_id"].isna().any() or (df["subject_id"].str.len() == 0).any():
        errors.append("Some rows have missing or empty subject_id values.")

    duplicated = df.loc[df["subject_id"].duplicated(), "subject_id"].tolist()
    if duplicated:
        errors.append(f"Duplicate subject_id values found: {duplicated}")

    bad_cohorts = sorted(set(df["cohort"].dropna()) - set(ALLOWED_COHORTS))
    if bad_cohorts:
        errors.append(
            f"Invalid cohort labels found: {bad_cohorts}. "
            f"Allowed labels are: {ALLOWED_COHORTS}"
        )

    if df["include_primary"].isna().any():
        errors.append("include_primary contains missing or non-numeric values.")

    bad_include = sorted(set(df["include_primary"].dropna()) - {0, 1})
    if bad_include:
        errors.append(
            f"include_primary must be 0 or 1. Found: {bad_include}"
        )

    non_missing_sex = df["sex"].dropna()
    bad_sex = sorted(set(non_missing_sex) - set(ALLOWED_SEX_CODES))
    if bad_sex:
        errors.append(
            f"sex must be 1, 2, or NA. Found invalid values: {bad_sex}"
        )

    non_missing_age = df["age"].dropna()
    if (non_missing_age < 0).any() or (non_missing_age > 120).any():
        errors.append("age contains values outside the plausible range 0–120.")

    if errors:
        msg = "\n".join(f"- {err}" for err in errors)
        raise ValueError(f"Metadata validation failed:\n{msg}")


def included_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return subjects marked for primary inclusion.
    """
    return df[df["include_primary"] == 1].copy()


def subjects_for_cohort(df: pd.DataFrame, cohort: str, primary_only: bool = True) -> list[str]:
    """
    Return subject IDs for one cohort.

    Parameters
    ----------
    df:
        Metadata dataframe.
    cohort:
        HC, SZ, or CUD.
    primary_only:
        If True, keep only include_primary == 1.
    """
    cohort = cohort.upper()

    if cohort not in ALLOWED_COHORTS:
        raise ValueError(f"Unknown cohort: {cohort}. Allowed: {ALLOWED_COHORTS}")

    sub = df[df["cohort"] == cohort].copy()

    if primary_only:
        sub = sub[sub["include_primary"] == 1].copy()

    return sub["subject_id"].tolist()


def metadata_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize total and included subjects by cohort.
    """
    rows = []

    for cohort in ALLOWED_COHORTS:
        sub = df[df["cohort"] == cohort].copy()
        inc = sub[sub["include_primary"] == 1].copy()

        rows.append(
            {
                "cohort": cohort,
                "n_total_rows": int(len(sub)),
                "n_include_primary": int(len(inc)),
                "n_missing_age_included": int(inc["age"].isna().sum()),
                "n_missing_sex_included": int(inc["sex"].isna().sum()),
                "age_mean_included": float(inc["age"].mean()) if inc["age"].notna().any() else np.nan,
                "age_min_included": float(inc["age"].min()) if inc["age"].notna().any() else np.nan,
                "age_max_included": float(inc["age"].max()) if inc["age"].notna().any() else np.nan,
                "n_male_included": int((inc["sex"] == 1).sum()),
                "n_female_included": int((inc["sex"] == 2).sum()),
            }
        )

    return pd.DataFrame(rows)


def print_metadata_summary(df: pd.DataFrame) -> None:
    """
    Print a human-readable metadata summary.
    """
    summary = metadata_summary_table(df)

    print("\n[METADATA SUMMARY]")
    print(summary.to_string(index=False))

    included = included_metadata(df)
    print("\n[INCLUDED SUBJECT COUNTS]")
    print(included["cohort"].value_counts().reindex(ALLOWED_COHORTS, fill_value=0).to_string())

    print("\n[FIRST 5 INCLUDED SUBJECTS PER COHORT]")
    for cohort in ALLOWED_COHORTS:
        ids = subjects_for_cohort(df, cohort, primary_only=True)
        print(f"{cohort}: {ids[:5]}")
