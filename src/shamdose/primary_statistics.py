from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MAIN_CONDITIONS = [
    "MagVenture_Sham0mm",
    "MagVenture_Sham0mm+electrodes",
    "Magstim_Sham0mm_kernel",
]

SUPPLEMENTARY_CONDITIONS = []

ALL_CONDITIONS = MAIN_CONDITIONS + SUPPLEMENTARY_CONDITIONS

ANATOMY_VARS = [
    "gm_depth_from_scalp_mm",
    "csf_mm",
]

PRIMARY_ENDPOINT = "PC_dPEAK_Hz"
SECONDARY_ENDPOINT = "PC_RMS_Hz"

INTENSITIES = list(range(10, 101, 10))


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """
    Holm-Bonferroni correction.

    NaN p-values remain NaN.
    """
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)

    valid = p.dropna()

    if valid.empty:
        return out

    order = valid.sort_values().index
    m = len(order)

    adjusted_sorted = []
    running_max = 0.0

    for rank, idx in enumerate(order, start=1):
        adj = (m - rank + 1) * float(p.loc[idx])
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)
        adjusted_sorted.append((idx, running_max))

    for idx, adj in adjusted_sorted:
        out.loc[idx] = adj

    return out


def safe_spearman(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """
    Spearman correlation with safety checks.
    """
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return {
            "n": int(len(x)),
            "rho_spearman": np.nan,
            "p_value": np.nan,
            "status": "TOO_FEW_OBSERVATIONS",
        }

    if x.nunique() < 2:
        return {
            "n": int(len(x)),
            "rho_spearman": np.nan,
            "p_value": np.nan,
            "status": "CONSTANT_X",
        }

    if y.nunique() < 2:
        return {
            "n": int(len(x)),
            "rho_spearman": np.nan,
            "p_value": np.nan,
            "status": "CONSTANT_Y",
        }

    rho, p = spearmanr(x, y)

    return {
        "n": int(len(x)),
        "rho_spearman": float(rho),
        "p_value": float(p),
        "status": "OK",
    }


def add_condition_scope(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["condition_scope"] = np.where(
        out["condition"].isin(MAIN_CONDITIONS),
        "main",
        np.where(out["condition"].isin(SUPPLEMENTARY_CONDITIONS), "supplementary", "other"),
    )
    return out


def build_crossover_probability_table(crossover: pd.DataFrame) -> pd.DataFrame:
    """
    Probability of crossing by each intensity.
    """
    df = add_condition_scope(crossover)

    rows = []

    for keys, sub in df.groupby(["cohort", "condition", "condition_scope", "endpoint"], sort=True):
        cohort, condition, scope, endpoint = keys

        for intensity in INTENSITIES:
            col = f"crossed_by_{intensity}MT"

            if col not in sub.columns:
                raise ValueError(f"Missing column: {col}")

            values = sub[col].astype(bool)

            rows.append(
                {
                    "cohort": cohort,
                    "condition": condition,
                    "condition_scope": scope,
                    "endpoint": endpoint,
                    "intensity_MT": intensity,
                    "n_subjects": int(sub["subject_id"].nunique()),
                    "n_crossed": int(values.sum()),
                    "pct_crossed": 100.0 * float(values.mean()),
                }
            )

    return pd.DataFrame(rows)


def build_anatomy_summary(crossover: pd.DataFrame) -> pd.DataFrame:
    """
    One-row-per-cohort anatomy summary based on unique subjects.
    """
    cols = [
        "subject_id",
        "cohort",
        "age",
        "sex",
        "gm_depth_from_scalp_mm",
        "csf_mm",
        "scalp_mm",
        "skull_mm",
    ]

    available = [c for c in cols if c in crossover.columns]

    subj = (
        crossover[available]
        .drop_duplicates(subset=["subject_id", "cohort"])
        .copy()
    )

    rows = []

    for cohort, sub in subj.groupby("cohort", sort=True):
        row = {
            "cohort": cohort,
            "n_subjects": int(sub["subject_id"].nunique()),
        }

        if "sex" in sub.columns:
            sex = pd.to_numeric(sub["sex"], errors="coerce")
            row["n_male"] = int((sex == 1).sum())
            row["n_female"] = int((sex == 2).sum())
            row["n_missing_sex"] = int(sex.isna().sum())

        for var in ["age", "gm_depth_from_scalp_mm", "csf_mm", "scalp_mm", "skull_mm"]:
            if var not in sub.columns:
                continue

            values = pd.to_numeric(sub[var], errors="coerce")

            row[f"{var}_mean"] = float(values.mean())
            row[f"{var}_sd"] = float(values.std())
            row[f"{var}_median"] = float(values.median())
            row[f"{var}_q25"] = float(values.quantile(0.25))
            row[f"{var}_q75"] = float(values.quantile(0.75))
            row[f"{var}_min"] = float(values.min())
            row[f"{var}_max"] = float(values.max())
            row[f"{var}_n_missing"] = int(values.isna().sum())

        rows.append(row)

    return pd.DataFrame(rows)


def run_anatomy_spearman(crossover: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman anatomy associations with crossover intensity.

    Primary interpretation:
        endpoint == PC_dPEAK_Hz
        condition in MagVenture main conditions
        crossed_by_100 == True

    For censored endpoints, Spearman is calculated only among observed crossers
    and should be treated as descriptive. Censored endpoints are better handled
    by the discrete-time event model.
    """
    df = add_condition_scope(crossover)

    rows = []

    for keys, sub in df.groupby(["cohort", "condition", "condition_scope", "endpoint"], sort=True):
        cohort, condition, scope, endpoint = keys

        n_total = int(sub["subject_id"].nunique())
        n_crossed = int(sub["crossed_by_100"].sum())
        n_censored = int((~sub["crossed_by_100"].astype(bool)).sum())

        observed = sub[sub["crossed_by_100"].astype(bool)].copy()

        for var in ANATOMY_VARS:
            res = safe_spearman(
                observed[var],
                observed["I_cross_MT"],
            )

            is_primary_test = (
                endpoint == PRIMARY_ENDPOINT
                and condition in MAIN_CONDITIONS
            )

            rows.append(
                {
                    "cohort": cohort,
                    "condition": condition,
                    "condition_scope": scope,
                    "endpoint": endpoint,
                    "anatomy_variable": var,
                    "is_primary_test": bool(is_primary_test),
                    "n_total": n_total,
                    "n_crossed_by_100": n_crossed,
                    "n_censored_at_100": n_censored,
                    "analysis_rows": res["n"],
                    "rho_spearman": res["rho_spearman"],
                    "p_value": res["p_value"],
                    "status": res["status"],
                    "analysis_note": (
                        "primary_observed_all_or_nearly_all"
                        if is_primary_test else
                        "descriptive_observed_crossers_only"
                    ),
                }
            )

    out = pd.DataFrame(rows)

    out["p_holm_primary_set"] = np.nan
    primary_mask = out["is_primary_test"] & out["p_value"].notna()

    out.loc[primary_mask, "p_holm_primary_set"] = holm_adjust(
        out.loc[primary_mask, "p_value"]
    )

    out["p_holm_all_finite"] = holm_adjust(out["p_value"])

    return out


def print_primary_statistics_summary(
    probability: pd.DataFrame,
    spearman: pd.DataFrame,
) -> None:
    """
    Print the main results for quick inspection.
    """
    print("\n[CROSSOVER PROBABILITY: PRIMARY ENDPOINT, MAIN CONDITIONS]")
    p = probability[
        (probability["endpoint"] == PRIMARY_ENDPOINT)
        & (probability["condition"].isin(MAIN_CONDITIONS))
        & (probability["intensity_MT"].isin([40, 50, 60, 100]))
    ].copy()

    print(
        p.pivot_table(
            index=["cohort", "condition"],
            columns="intensity_MT",
            values="pct_crossed",
        )
        .round(3)
        .to_string()
    )

    print("\n[ANATOMY SPEARMAN: PRIMARY TESTS]")
    s = spearman[spearman["is_primary_test"]].copy()

    cols = [
        "cohort",
        "condition",
        "anatomy_variable",
        "analysis_rows",
        "rho_spearman",
        "p_value",
        "p_holm_primary_set",
        "status",
    ]

    print(s[cols].round(6).to_string(index=False))
