from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PRIMARY_ENDPOINT = "PC_dPEAK_Hz"

MAIN_CONDITIONS = [
    "MagVenture_Sham0mm",
    "MagVenture_Sham0mm+electrodes",
    "Magstim_Sham0mm_kernel",
]

SUPPLEMENTARY_CONDITIONS = []

ALL_CONDITIONS = MAIN_CONDITIONS + SUPPLEMENTARY_CONDITIONS

ANATOMY_TERMS = [
    "z_gm_depth",
    "z_csf",
]


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """
    Holm-Bonferroni correction.

    NaN values remain NaN.
    """
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)

    valid = p.dropna()

    if valid.empty:
        return out

    order = valid.sort_values().index
    m = len(order)
    running_max = 0.0

    for rank, idx in enumerate(order, start=1):
        adj = (m - rank + 1) * float(p.loc[idx])
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)
        out.loc[idx] = running_max

    return out


def zscore(series: pd.Series) -> pd.Series:
    """
    Standardize a variable using sample SD.

    Returns NaN if the variable has no variance.
    """
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std()

    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)

    return (x - x.mean()) / sd


def prepare_event_data(
    events: pd.DataFrame,
    endpoint: str,
    condition: str,
    cohort: str,
    model_type: str,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    """
    Prepare one model-specific event table.

    model_type:
        anatomy_only
        age_sex_adjusted
    """
    d = events[
        (events["endpoint"] == endpoint)
        & (events["condition"] == condition)
        & (events["cohort"] == cohort)
    ].copy()

    if d.empty:
        raise ValueError(
            f"No event rows for endpoint={endpoint}, condition={condition}, cohort={cohort}"
        )

    d["event"] = pd.to_numeric(d["event"], errors="coerce")
    d["intensity_MT"] = pd.to_numeric(d["intensity_MT"], errors="coerce")
    d["gm_depth_from_scalp_mm"] = pd.to_numeric(d["gm_depth_from_scalp_mm"], errors="coerce")
    d["csf_mm"] = pd.to_numeric(d["csf_mm"], errors="coerce")
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    d["sex"] = pd.to_numeric(d["sex"], errors="coerce")

    d["z_gm_depth"] = zscore(d["gm_depth_from_scalp_mm"])
    d["z_csf"] = zscore(d["csf_mm"])
    d["z_age"] = zscore(d["age"])

    # sex code: 1 = male, 2 = female
    d["sex_female"] = np.where(d["sex"] == 2, 1.0, np.where(d["sex"] == 1, 0.0, np.nan))

    covariates = ["z_gm_depth", "z_csf"]

    if model_type == "age_sex_adjusted":
        covariates.append("z_age")

        # Add sex only if both levels exist after dropping missing rows.
        if d["sex_female"].dropna().nunique() >= 2:
            covariates.append("sex_female")

    elif model_type != "anatomy_only":
        raise ValueError(f"Unknown model_type: {model_type}")

    required = ["subject_id", "event", "intensity_MT", *covariates]

    d_model = d.dropna(subset=required).copy()

    # Categorical intensity is the discrete-time baseline hazard.
    d_model["intensity_MT"] = d_model["intensity_MT"].astype(int)

    info = {
        "endpoint": endpoint,
        "condition": condition,
        "cohort": cohort,
        "model_type": model_type,
        "n_rows_raw": int(len(d)),
        "n_rows_model": int(len(d_model)),
        "n_subjects_raw": int(d["subject_id"].nunique()),
        "n_subjects_model": int(d_model["subject_id"].nunique()),
        "n_events": int(d_model["event"].sum()),
        "n_missing_age_rows": int(d["age"].isna().sum()),
        "n_missing_sex_rows": int(d["sex"].isna().sum()),
        "covariates": ",".join(covariates),
    }

    return d_model, covariates, info


def fit_discrete_time_logit(
    d_model: pd.DataFrame,
    covariates: list[str],
) -> tuple[object, str, str]:
    """
    Fit a discrete-time logistic hazard model.

    Formula:
        event ~ C(intensity_MT) + anatomy/covariates

    Uses cluster-robust SE by subject when possible.
    """
    if d_model.empty:
        raise ValueError("Model data is empty.")

    if d_model["event"].nunique() < 2:
        raise ValueError("Event column has no variation.")

    rhs = "C(intensity_MT) + " + " + ".join(covariates)
    formula = "event ~ " + rhs

    warning_text = ""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        model = smf.glm(
            formula=formula,
            data=d_model,
            family=__import__("statsmodels.api").api.families.Binomial(),
        )

        try:
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": d_model["subject_id"]},
                maxiter=200,
            )
            covariance_type = "cluster_subject"
        except Exception as exc:
            warning_text += f"cluster_fit_failed={repr(exc)}; "
            result = model.fit(maxiter=200)
            covariance_type = "default"

        if caught:
            warning_text += " | ".join(str(w.message) for w in caught)

    return result, formula, covariance_type + ("; " + warning_text if warning_text else "")


def extract_model_coefficients(
    result,
    formula: str,
    model_status: dict[str, object],
) -> pd.DataFrame:
    """
    Extract coefficients, odds ratios and confidence intervals.
    """
    params = result.params
    conf = result.conf_int()

    rows = []

    for term in params.index:
        beta = float(params.loc[term])
        se = float(result.bse.loc[term])
        p = float(result.pvalues.loc[term])
        lo = float(conf.loc[term, 0])
        hi = float(conf.loc[term, 1])

        rows.append(
            {
                **model_status,
                "formula": formula,
                "term": term,
                "beta": beta,
                "se": se,
                "odds_ratio": float(np.exp(beta)),
                "ci95_low": float(np.exp(lo)),
                "ci95_high": float(np.exp(hi)),
                "p_value": p,
            }
        )

    return pd.DataFrame(rows)


def run_all_discrete_time_models(
    events: pd.DataFrame,
    endpoints: list[str],
    conditions: list[str],
    cohorts: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit anatomy-only and age/sex-adjusted discrete-time models.

    Models are run separately within:
        endpoint × condition × cohort

    This preserves the cohort-first analysis design.
    """
    coef_frames = []
    status_rows = []

    for endpoint in endpoints:
        for condition in conditions:
            for cohort in cohorts:
                for model_type in ["anatomy_only", "age_sex_adjusted"]:
                    try:
                        d_model, covariates, info = prepare_event_data(
                            events=events,
                            endpoint=endpoint,
                            condition=condition,
                            cohort=cohort,
                            model_type=model_type,
                        )

                        result, formula, fit_note = fit_discrete_time_logit(
                            d_model=d_model,
                            covariates=covariates,
                        )

                        status = {
                            **info,
                            "model_status": "OK",
                            "fit_note": fit_note,
                            "converged": bool(getattr(result, "converged", True)),
                        }

                        coef = extract_model_coefficients(
                            result=result,
                            formula=formula,
                            model_status=status,
                        )

                        coef_frames.append(coef)
                        status_rows.append(status)

                    except Exception as exc:
                        status = {
                            "endpoint": endpoint,
                            "condition": condition,
                            "cohort": cohort,
                            "model_type": model_type,
                            "model_status": "FAILED",
                            "error": repr(exc),
                        }
                        status_rows.append(status)

    coefficients = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    model_status = pd.DataFrame(status_rows)

    if not coefficients.empty:
        coefficients["is_anatomy_term"] = coefficients["term"].isin(ANATOMY_TERMS)
        coefficients["is_primary_model"] = (
            (coefficients["endpoint"] == PRIMARY_ENDPOINT)
            & (coefficients["condition"].isin(MAIN_CONDITIONS))
            & (coefficients["model_type"] == "age_sex_adjusted")
            & (coefficients["term"].isin(ANATOMY_TERMS))
        )

        coefficients["p_holm_primary_anatomy_terms"] = np.nan
        primary_mask = coefficients["is_primary_model"] & coefficients["p_value"].notna()

        coefficients.loc[primary_mask, "p_holm_primary_anatomy_terms"] = holm_adjust(
            coefficients.loc[primary_mask, "p_value"]
        )

        coefficients["p_holm_all_anatomy_terms"] = np.nan
        anatomy_mask = coefficients["is_anatomy_term"] & coefficients["p_value"].notna()

        coefficients.loc[anatomy_mask, "p_holm_all_anatomy_terms"] = holm_adjust(
            coefficients.loc[anatomy_mask, "p_value"]
        )

    return coefficients, model_status


def print_discrete_time_summary(coefficients: pd.DataFrame) -> None:
    """
    Print primary age/sex-adjusted anatomy odds ratios.
    """
    print("\n[DISCRETE-TIME SURVIVAL MODEL: PRIMARY AGE/SEX-ADJUSTED ANATOMY TERMS]")

    if coefficients.empty:
        print("No model coefficients available.")
        return

    primary = coefficients[coefficients["is_primary_model"]].copy()

    if primary.empty:
        print("No primary coefficients found.")
        return

    cols = [
        "cohort",
        "condition",
        "term",
        "n_subjects_model",
        "n_events",
        "odds_ratio",
        "ci95_low",
        "ci95_high",
        "p_value",
        "p_holm_primary_anatomy_terms",
        "model_status",
    ]

    print(primary[cols].round(6).to_string(index=False))

    print("\nInterpretation:")
    print("  odds_ratio < 1 means the predictor lowers the odds of crossing at a given intensity.")
    print("  For gm_depth, odds_ratio < 1 means deeper GM delays crossover.")
