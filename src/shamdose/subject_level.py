from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SHAM_CONDITIONS = [
    "MagVenture_Sham0mm",
    "MagVenture_Sham0mm+electrodes",
    "Magstim_Sham0mm_kernel",
]

TDCS_CONDITION = "tDCS"

ENDPOINTS = [
    "PC_dPEAK_Hz",
    "PC_RMS_Hz",
]

CROSSOVER_INTENSITIES = list(range(10, 101, 10))


def load_step3_tables(
    tables_dir: str | Path,
    cohorts: list[str],
) -> pd.DataFrame:
    """
    Load final Step 3 mean-field outputs for all cohorts.
    """
    tables_dir = Path(tables_dir).expanduser().resolve()
    frames = []

    for cohort in cohorts:
        path = tables_dir / f"step3_meanfield_{cohort}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing Step 3 table: {path}")

        df = pd.read_csv(path)
        df["cohort"] = cohort
        df["source_step3_table"] = str(path)

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    return out


def load_anatomy_tables(
    tables_dir: str | Path,
    cohorts: list[str],
) -> pd.DataFrame:
    """
    Load target anatomy tables for all cohorts.
    """
    tables_dir = Path(tables_dir).expanduser().resolve()
    frames = []

    for cohort in cohorts:
        path = tables_dir / f"target_anatomy_{cohort}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing anatomy table: {path}")

        df = pd.read_csv(path)
        df["cohort"] = cohort
        df["source_anatomy_table"] = str(path)

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    return out


def validate_step3_for_subject_level(
    step3: pd.DataFrame,
    expected_subject_counts: dict[str, int],
) -> None:
    """
    Validate that Step 3 tables are suitable for subject-level crossover analysis.
    """
    errors: list[str] = []

    required_cols = [
        "SUBJECT_ID",
        "cohort",
        "CONDITION",
        "INTENSITY_MT",
        *ENDPOINTS,
    ]

    missing_cols = [col for col in required_cols if col not in step3.columns]
    if missing_cols:
        errors.append(f"Missing Step 3 columns: {missing_cols}")

    if missing_cols:
        raise ValueError("\n".join(errors))

    observed_conditions = sorted(step3["CONDITION"].astype(str).unique())

    unexpected_active = [
        cond for cond in observed_conditions
        if "Active" in cond
    ]

    if unexpected_active:
        errors.append(
            f"Active TMS conditions found in final Step 3 table: {unexpected_active}. "
            "Subject-level primary analysis should use sham/tDCS Step 3 outputs."
        )

    expected_conditions = sorted(SHAM_CONDITIONS + [TDCS_CONDITION])
    missing_conditions = [
        cond for cond in expected_conditions
        if cond not in observed_conditions
    ]

    if missing_conditions:
        errors.append(f"Missing expected Step 3 conditions: {missing_conditions}")

    for cohort, expected_n in expected_subject_counts.items():
        sub = step3[step3["cohort"] == cohort].copy()
        observed_n = sub["SUBJECT_ID"].nunique()
        observed_rows = len(sub)
        expected_rows = expected_n * 34

        if observed_n != expected_n:
            errors.append(
                f"{cohort}: subject count mismatch. "
                f"Expected {expected_n}, observed {observed_n}."
            )

        if observed_rows != expected_rows:
            errors.append(
                f"{cohort}: row count mismatch. "
                f"Expected {expected_rows}, observed {observed_rows}."
            )

        per_subject = sub.groupby("SUBJECT_ID").size()
        bad = per_subject[per_subject != 34]

        if len(bad) > 0:
            errors.append(
                f"{cohort}: subjects with row count != 34:\n{bad.head(20).to_string()}"
            )

    for endpoint in ENDPOINTS:
        values = pd.to_numeric(step3[endpoint], errors="coerce")
        if values.isna().any():
            errors.append(f"{endpoint} contains NaN values.")
        if not np.isfinite(values).all():
            errors.append(f"{endpoint} contains non-finite values.")

    if errors:
        raise ValueError("Step 3 validation failed:\n" + "\n".join(f"- {e}" for e in errors))


def build_subject_info(
    metadata: pd.DataFrame,
    anatomy: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one-row-per-subject metadata/anatomy table.
    """
    meta = metadata.copy()
    meta["subject_id"] = meta["subject_id"].astype(str)
    meta["cohort"] = meta["cohort"].astype(str)

    meta = meta[meta["include_primary"] == 1].copy()

    anat = anatomy.copy()
    anat["subject_id"] = anat["subject_id"].astype(str)
    anat["cohort"] = anat["cohort"].astype(str)

    keep_anat_cols = [
        "subject_id",
        "cohort",
        "status",
        "gm_depth_from_scalp_mm",
        "csf_mm",
        "scalp_mm",
        "skull_mm",
        "scalp_snap_distance_mm",
        "depth_consistency_error_mm",
    ]

    available_anat_cols = [col for col in keep_anat_cols if col in anat.columns]
    anat = anat[available_anat_cols].copy()

    info = meta.merge(
        anat,
        on=["subject_id", "cohort"],
        how="left",
        suffixes=("", "_anatomy"),
    )

    return info


def first_crossover_intensity(
    sham_rows: pd.DataFrame,
    endpoint: str,
    tdcs_value: float,
) -> tuple[float, bool]:
    """
    Return first intensity where sham endpoint >= tDCS endpoint.

    Returns:
        I_cross_MT:
            observed crossover intensity, or NaN if not observed by 100% MT.
        crossed_by_100:
            True if observed, False if censored at 100% MT.
    """
    rows = sham_rows.copy()
    rows["INTENSITY_MT"] = pd.to_numeric(rows["INTENSITY_MT"], errors="coerce")
    rows[endpoint] = pd.to_numeric(rows[endpoint], errors="coerce")

    rows = rows[
        rows["INTENSITY_MT"].isin(CROSSOVER_INTENSITIES)
    ].sort_values("INTENSITY_MT")

    crossed = rows[rows[endpoint] >= tdcs_value]

    if crossed.empty:
        return np.nan, False

    return float(crossed["INTENSITY_MT"].iloc[0]), True


def value_at_intensity(
    rows: pd.DataFrame,
    endpoint: str,
    intensity: int,
) -> float:
    sub = rows[pd.to_numeric(rows["INTENSITY_MT"], errors="coerce") == int(intensity)]

    if sub.empty:
        return np.nan

    if len(sub) != 1:
        raise ValueError(f"Expected one row at intensity {intensity}, found {len(sub)}")

    return float(pd.to_numeric(sub[endpoint], errors="coerce").iloc[0])


def build_crossover_table(
    step3: pd.DataFrame,
    subject_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per subject × sham condition × endpoint.
    """
    df = step3.copy()
    df["SUBJECT_ID"] = df["SUBJECT_ID"].astype(str)
    df["CONDITION"] = df["CONDITION"].astype(str)
    df["INTENSITY_MT"] = pd.to_numeric(df["INTENSITY_MT"], errors="coerce")

    info = subject_info.copy()
    info["subject_id"] = info["subject_id"].astype(str)

    rows = []

    for subject_id, sub in df.groupby("SUBJECT_ID", sort=True):
        info_row = info[info["subject_id"] == subject_id]

        if info_row.empty:
            raise ValueError(f"No metadata/anatomy row found for {subject_id}")

        info_row = info_row.iloc[0]

        tdcs = sub[sub["CONDITION"] == TDCS_CONDITION].copy()

        if len(tdcs) != 1:
            raise ValueError(f"{subject_id}: expected exactly one tDCS row, found {len(tdcs)}")

        for endpoint in ENDPOINTS:
            tdcs_value = float(pd.to_numeric(tdcs[endpoint], errors="coerce").iloc[0])

            for condition in SHAM_CONDITIONS:
                sham = sub[sub["CONDITION"] == condition].copy()

                if sham.empty:
                    raise ValueError(f"{subject_id}: missing sham condition {condition}")

                I_cross, crossed_by_100 = first_crossover_intensity(
                    sham_rows=sham,
                    endpoint=endpoint,
                    tdcs_value=tdcs_value,
                )

                sham_100 = value_at_intensity(sham, endpoint, 100)

                if tdcs_value != 0:
                    ratio_100 = sham_100 / tdcs_value
                else:
                    ratio_100 = np.nan

                row = {
                    "subject_id": subject_id,
                    "cohort": info_row["cohort"],
                    "age": info_row.get("age", np.nan),
                    "sex": info_row.get("sex", np.nan),
                    "condition": condition,
                    "endpoint": endpoint,
                    "tdcs_value": tdcs_value,
                    "sham_value_100MT": sham_100,
                    "sham_minus_tdcs_100MT": sham_100 - tdcs_value,
                    "sham_over_tdcs_100MT": ratio_100,
                    "I_cross_MT": I_cross,
                    "crossed_by_100": bool(crossed_by_100),
                    "censored_at_100": bool(not crossed_by_100),
                    "I_cross_display_MT": I_cross if crossed_by_100 else 110.0,
                    "gm_depth_from_scalp_mm": info_row.get("gm_depth_from_scalp_mm", np.nan),
                    "csf_mm": info_row.get("csf_mm", np.nan),
                    "scalp_mm": info_row.get("scalp_mm", np.nan),
                    "skull_mm": info_row.get("skull_mm", np.nan),
                    "scalp_snap_distance_mm": info_row.get("scalp_snap_distance_mm", np.nan),
                    "depth_consistency_error_mm": info_row.get("depth_consistency_error_mm", np.nan),
                }

                for intensity in CROSSOVER_INTENSITIES:
                    val = value_at_intensity(sham, endpoint, intensity)
                    row[f"sham_value_{intensity}MT"] = val
                    row[f"crossed_by_{intensity}MT"] = bool(val >= tdcs_value)

                rows.append(row)

    out = pd.DataFrame(rows)

    validate_crossover_table(out)

    return out


def build_discrete_time_event_table(
    crossover: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build discrete-time event rows for survival/logistic hazard models.

    For each subject × sham condition × endpoint, rows are generated from
    10% MT up to first crossover or censoring at 100% MT.

    event = 1 only at the first crossing intensity.
    """
    rows = []

    for _, r in crossover.iterrows():
        has_event = bool(r["crossed_by_100"])
        event_intensity = r["I_cross_MT"] if has_event else np.nan

        for intensity in CROSSOVER_INTENSITIES:
            if has_event and intensity > event_intensity:
                break

            event = bool(has_event and intensity == event_intensity)

            rows.append(
                {
                    "subject_id": r["subject_id"],
                    "cohort": r["cohort"],
                    "age": r["age"],
                    "sex": r["sex"],
                    "condition": r["condition"],
                    "endpoint": r["endpoint"],
                    "intensity_MT": intensity,
                    "event": int(event),
                    "censored_at_100": bool(r["censored_at_100"]),
                    "tdcs_value": r["tdcs_value"],
                    "sham_value": r[f"sham_value_{intensity}MT"],
                    "gm_depth_from_scalp_mm": r["gm_depth_from_scalp_mm"],
                    "csf_mm": r["csf_mm"],
                }
            )

            if event:
                break

    out = pd.DataFrame(rows)

    return out


def validate_crossover_table(crossover: pd.DataFrame) -> None:
    """
    Validate crossover table.
    """
    errors: list[str] = []

    expected_conditions = set(SHAM_CONDITIONS)
    observed_conditions = set(crossover["condition"].astype(str).unique())

    if observed_conditions != expected_conditions:
        errors.append(
            f"Condition mismatch. Expected {sorted(expected_conditions)}, "
            f"observed {sorted(observed_conditions)}"
        )

    expected_endpoints = set(ENDPOINTS)
    observed_endpoints = set(crossover["endpoint"].astype(str).unique())

    if observed_endpoints != expected_endpoints:
        errors.append(
            f"Endpoint mismatch. Expected {sorted(expected_endpoints)}, "
            f"observed {sorted(observed_endpoints)}"
        )

    per_subject_endpoint = (
        crossover.groupby(["subject_id", "endpoint"]).size()
    )

    bad = per_subject_endpoint[per_subject_endpoint != len(SHAM_CONDITIONS)]
    if len(bad) > 0:
        errors.append(
            "Some subject × endpoint combinations do not have exactly "
            f"{len(SHAM_CONDITIONS)} sham-condition rows:\n{bad.head(20).to_string()}"
        )

    for col in ["tdcs_value", "sham_value_100MT", "gm_depth_from_scalp_mm", "csf_mm"]:
        values = pd.to_numeric(crossover[col], errors="coerce")
        if values.isna().any():
            errors.append(f"{col} contains NaN values.")

    if errors:
        raise ValueError(
            "Crossover table validation failed:\n"
            + "\n".join(f"- {e}" for e in errors)
        )


def summarize_crossover(crossover: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize crossover by cohort, condition, and endpoint.
    """
    rows = []

    for keys, sub in crossover.groupby(["cohort", "condition", "endpoint"], sort=True):
        cohort, condition, endpoint = keys

        observed = sub[sub["crossed_by_100"]].copy()

        row = {
            "cohort": cohort,
            "condition": condition,
            "endpoint": endpoint,
            "n_subjects": int(sub["subject_id"].nunique()),
            "n_crossed_by_100": int(sub["crossed_by_100"].sum()),
            "pct_crossed_by_100": 100.0 * float(sub["crossed_by_100"].mean()),
            "median_I_cross_observed_MT": float(observed["I_cross_MT"].median()) if not observed.empty else np.nan,
            "q25_I_cross_observed_MT": float(observed["I_cross_MT"].quantile(0.25)) if not observed.empty else np.nan,
            "q75_I_cross_observed_MT": float(observed["I_cross_MT"].quantile(0.75)) if not observed.empty else np.nan,
            "median_sham_over_tdcs_100MT": float(sub["sham_over_tdcs_100MT"].median()),
            "q25_sham_over_tdcs_100MT": float(sub["sham_over_tdcs_100MT"].quantile(0.25)),
            "q75_sham_over_tdcs_100MT": float(sub["sham_over_tdcs_100MT"].quantile(0.75)),
        }

        for intensity in [40, 50, 60, 100]:
            col = f"crossed_by_{intensity}MT"
            row[f"pct_crossed_by_{intensity}MT"] = 100.0 * float(sub[col].mean())

        rows.append(row)

    return pd.DataFrame(rows)


def print_subject_level_summary(
    crossover: pd.DataFrame,
    event_table: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """
    Print concise QC summary.
    """
    print("\n[SUBJECT-LEVEL CROSSOVER TABLE]")
    print(f"Rows: {len(crossover)}")
    print(f"Subjects: {crossover['subject_id'].nunique()}")

    print("\nRows by endpoint:")
    print(crossover["endpoint"].value_counts().sort_index().to_string())

    print("\nRows by condition:")
    print(crossover["condition"].value_counts().sort_index().to_string())

    print("\n[DISCRETE-TIME EVENT TABLE]")
    print(f"Rows: {len(event_table)}")
    print(f"Subjects: {event_table['subject_id'].nunique()}")

    print("\n[PRIMARY ENDPOINT SUMMARY: PC_dPEAK_Hz]")
    primary = summary[summary["endpoint"] == "PC_dPEAK_Hz"].copy()
    cols = [
        "cohort",
        "condition",
        "n_subjects",
        "n_crossed_by_100",
        "pct_crossed_by_40MT",
        "pct_crossed_by_50MT",
        "pct_crossed_by_60MT",
        "pct_crossed_by_100MT",
        "median_I_cross_observed_MT",
        "median_sham_over_tdcs_100MT",
    ]
    print(primary[cols].round(3).to_string(index=False))
