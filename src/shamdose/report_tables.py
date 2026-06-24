from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INTERPRETABLE_TERMS = [
    "z_gm_depth",
    "z_csf",
    "z_age",
    "sex_female",
]

TERM_LABELS = {
    "z_gm_depth": "Depth to GM",
    "z_csf": "Local CSF thickness",
    "z_age": "Age",
    "sex_female": "Sex: female vs male",
}

TERM_FAMILIES = {
    "z_gm_depth": "anatomy",
    "z_csf": "anatomy",
    "z_age": "demographic",
    "sex_female": "demographic",
}

MAIN_CONDITIONS = [
    "MagVenture_Sham0mm",
    "MagVenture_Sham0mm+electrodes",
    "Magstim_Sham0mm_kernel",
]

SUPPLEMENTARY_CONDITIONS = []

PRIMARY_ENDPOINT = "PC_dPEAK_Hz"
SECONDARY_ENDPOINT = "PC_RMS_Hz"


def p_label(p: float) -> str:
    """
    Format p-values for reporting.
    """
    if pd.isna(p):
        return ""

    p = float(p)

    if p < 0.001:
        return "<0.001"

    return f"{p:.3f}"


def format_or_ci(or_value: float, lo: float, hi: float) -> str:
    """
    Format odds ratio with 95% CI.
    """
    if not np.isfinite(or_value) or not np.isfinite(lo) or not np.isfinite(hi):
        return ""

    return f"{or_value:.2f} [{lo:.2f}, {hi:.2f}]"


def clean_survival_coefficients(coef: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only interpretable covariates from discrete-time survival models.

    Removes:
        Intercept
        intensity dummy terms
        other nuisance terms
    """
    df = coef.copy()

    df = df[df["term"].isin(INTERPRETABLE_TERMS)].copy()

    df["term_label"] = df["term"].map(TERM_LABELS)
    df["term_family"] = df["term"].map(TERM_FAMILIES)

    df["condition_scope"] = np.where(
        df["condition"].isin(MAIN_CONDITIONS),
        "main",
        np.where(df["condition"].isin(SUPPLEMENTARY_CONDITIONS), "supplementary", "other"),
    )

    df["endpoint_scope"] = np.where(
        df["endpoint"] == PRIMARY_ENDPOINT,
        "primary_endpoint",
        np.where(df["endpoint"] == SECONDARY_ENDPOINT, "secondary_endpoint", "other"),
    )

    df["report_scope"] = np.select(
        [
            (df["endpoint"] == PRIMARY_ENDPOINT)
            & (df["condition"].isin(MAIN_CONDITIONS))
            & (df["model_type"] == "age_sex_adjusted")
            & (df["term"].isin(["z_gm_depth", "z_csf"])),

            (df["endpoint"] == PRIMARY_ENDPOINT)
            & (df["condition"].isin(MAIN_CONDITIONS))
            & (df["model_type"] == "age_sex_adjusted")
            & (df["term"].isin(["z_age", "sex_female"])),

            (df["condition"].isin(SUPPLEMENTARY_CONDITIONS))
            | (df["endpoint"] == SECONDARY_ENDPOINT)
            | (df["model_type"] == "anatomy_only"),
        ],
        [
            "main_adjusted_anatomy",
            "supplementary_demographics",
            "supplementary",
        ],
        default="supplementary",
    )

    df["OR_95CI"] = [
        format_or_ci(or_value, lo, hi)
        for or_value, lo, hi in zip(df["odds_ratio"], df["ci95_low"], df["ci95_high"])
    ]

    df["p_label"] = df["p_value"].apply(p_label)

    if "p_holm_primary_anatomy_terms" in df.columns:
        df["p_holm_primary_label"] = df["p_holm_primary_anatomy_terms"].apply(p_label)
    else:
        df["p_holm_primary_label"] = ""

    return df


def build_main_adjusted_anatomy_table(clean_coef: pd.DataFrame) -> pd.DataFrame:
    """
    Main report table for age/sex-adjusted anatomy effects.

    Contains only:
        PC_dPEAK_Hz
        MagVenture sham conditions
        age_sex_adjusted model
        z_gm_depth and z_csf
    """
    main = clean_coef[
        (clean_coef["report_scope"] == "main_adjusted_anatomy")
    ].copy()

    keep_cols = [
        "cohort",
        "condition",
        "term",
        "term_label",
        "n_subjects_model",
        "n_events",
        "odds_ratio",
        "ci95_low",
        "ci95_high",
        "OR_95CI",
        "p_value",
        "p_label",
        "p_holm_primary_anatomy_terms",
        "p_holm_primary_label",
        "model_status",
    ]

    keep_cols = [c for c in keep_cols if c in main.columns]

    main = main[keep_cols].copy()

    order_condition = {
        "MagVenture_Sham0mm": 0,
        "MagVenture_Sham0mm+electrodes": 1,
    }
    order_term = {
        "z_gm_depth": 0,
        "z_csf": 1,
    }

    main["_condition_order"] = main["condition"].map(order_condition)
    main["_term_order"] = main["term"].map(order_term)

    main = (
        main.sort_values(["cohort", "_condition_order", "_term_order"])
        .drop(columns=["_condition_order", "_term_order"])
        .reset_index(drop=True)
    )

    return main


def build_key_results_table(
    crossover_summary: pd.DataFrame,
    spearman: pd.DataFrame,
    survival_main: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact key-results table for manuscript drafting.

    This table does not replace full outputs; it just collects the main values.
    """
    rows = []

    primary_cross = crossover_summary[
        (crossover_summary["endpoint"] == PRIMARY_ENDPOINT)
        & (crossover_summary["condition"].isin(MAIN_CONDITIONS))
    ].copy()

    primary_spear = spearman[
        (spearman["endpoint"] == PRIMARY_ENDPOINT)
        & (spearman["condition"].isin(MAIN_CONDITIONS))
        & (spearman["anatomy_variable"] == "gm_depth_from_scalp_mm")
    ].copy()

    for _, r in primary_cross.iterrows():
        cohort = r["cohort"]
        condition = r["condition"]

        sp = primary_spear[
            (primary_spear["cohort"] == cohort)
            & (primary_spear["condition"] == condition)
        ]

        depth_or = survival_main[
            (survival_main["cohort"] == cohort)
            & (survival_main["condition"] == condition)
            & (survival_main["term"] == "z_gm_depth")
        ]

        row = {
            "cohort": cohort,
            "condition": condition,
            "n_subjects": int(r["n_subjects"]),
            "pct_crossed_by_40MT": float(r["pct_crossed_by_40MT"]),
            "pct_crossed_by_50MT": float(r["pct_crossed_by_50MT"]),
            "pct_crossed_by_60MT": float(r["pct_crossed_by_60MT"]),
            "pct_crossed_by_100MT": float(r["pct_crossed_by_100MT"]),
            "median_I_cross_observed_MT": float(r["median_I_cross_observed_MT"]),
            "median_sham_over_tdcs_100MT": float(r["median_sham_over_tdcs_100MT"]),
        }

        if not sp.empty:
            sp = sp.iloc[0]
            row["depth_spearman_rho"] = float(sp["rho_spearman"])
            row["depth_spearman_p"] = float(sp["p_value"])
            row["depth_spearman_p_holm_primary"] = float(sp["p_holm_primary_set"])

        if not depth_or.empty:
            depth_or = depth_or.iloc[0]
            row["depth_adjusted_OR"] = float(depth_or["odds_ratio"])
            row["depth_adjusted_OR_95CI"] = depth_or["OR_95CI"]
            row["depth_adjusted_p"] = float(depth_or["p_value"])
            row["depth_adjusted_p_holm_primary"] = float(depth_or["p_holm_primary_anatomy_terms"])

        rows.append(row)

    return pd.DataFrame(rows)


def print_report_table_summary(
    clean_coef: pd.DataFrame,
    main_adjusted: pd.DataFrame,
    key_results: pd.DataFrame,
) -> None:
    """
    Print concise report-table summary.
    """
    print("\n[REPORT-READY DISCRETE-TIME TERMS]")
    print(f"Rows: {len(clean_coef)}")

    print("\nRows by report scope:")
    print(clean_coef["report_scope"].value_counts().to_string())

    print("\n[MAIN ADJUSTED ANATOMY TABLE]")
    cols = [
        "cohort",
        "condition",
        "term_label",
        "n_subjects_model",
        "n_events",
        "OR_95CI",
        "p_label",
        "p_holm_primary_label",
    ]
    print(main_adjusted[cols].to_string(index=False))

    print("\n[KEY RESULTS TABLE]")
    key_cols = [
        "cohort",
        "condition",
        "pct_crossed_by_50MT",
        "pct_crossed_by_60MT",
        "median_I_cross_observed_MT",
        "depth_spearman_rho",
        "depth_adjusted_OR_95CI",
        "depth_adjusted_p_holm_primary",
    ]
    existing = [c for c in key_cols if c in key_results.columns]
    print(key_results[existing].round(3).to_string(index=False))
