from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TISSUE_PREFIX = {
    "GM": "roi_graymatter",
    "WM": "roi_whitematter",
}


def tau_tag(tau_ms: float) -> str:
    return f"tau{str(tau_ms).replace('.', 'p')}ms"


def build_input_column(tissue_prefix: str, metric: str) -> str:
    return f"{tissue_prefix}_{metric}_|E|_V/m"


def build_k_column(tau_ms: float) -> str:
    return f"k_tau_{tau_tag(tau_ms)}"


def build_output_column(tissue_short: str, metric: str, tau_ms: float) -> str:
    return f"C_{tissue_short}_ROI_{metric}_V/m_{tau_tag(tau_ms)}"


def coil_family_from_label(coil_label: object) -> str | None:
    """
    Map Step 1 coil labels to waveform families.
    """
    s = str(coil_label).strip().lower()

    if s in {"", "nan", "none", "na"}:
        return None

    if "magventure" in s or "magpro" in s:
        return "MagVenture"

    if "magstim" in s:
        return "Magstim"

    return None


def condition_mode(modality: object, condition: object) -> str:
    """
    Active/Sham/DC mode used for waveform gain assignment.
    """
    modality_s = str(modality).strip().lower()
    condition_s = str(condition).strip().lower()

    if modality_s == "tdcs":
        return "DC"

    if "sham" in condition_s:
        return "Sham"

    return "Active"


def make_gain_lookup(gains: pd.DataFrame) -> dict[tuple[str, str, float], float]:
    """
    Convert waveform gain table to lookup dictionary.
    """
    lookup: dict[tuple[str, str, float], float] = {}

    for _, row in gains.iterrows():
        key = (
            str(row["coil_family"]),
            str(row["mode"]),
            float(row["tau_ms"]),
        )
        lookup[key] = float(row["k_tau"])

    return lookup


def compute_effective_dose_for_table(
    step1: pd.DataFrame,
    waveform_gains: pd.DataFrame,
    tau_list_ms: list[float],
    gm_metrics: list[str],
    include_wm_mean: bool = True,
) -> pd.DataFrame:
    """
    Compute membrane-filtered effective dose from a Step 1 E-field table.

    For TMS:
        C = E0 * k_tau

    For tDCS:
        C = E0
    """
    df = step1.copy()

    required = [
        "subject_id",
        "modality",
        "coil",
        "condition",
        "intensity (%MT)",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Step 1 table missing required columns: {missing}")

    df["coil_family"] = df["coil"].apply(coil_family_from_label)
    df["mode"] = [
        condition_mode(modality, condition)
        for modality, condition in zip(df["modality"], df["condition"])
    ]

    df["intensity (%MT)"] = pd.to_numeric(df["intensity (%MT)"], errors="coerce")

    mask_tms = df["modality"].astype(str).str.lower().eq("tms")
    mask_tdcs = df["modality"].astype(str).str.lower().eq("tdcs")

    unknown_tms = mask_tms & df["coil_family"].isna()
    if unknown_tms.any():
        bad = sorted(df.loc[unknown_tms, "coil"].astype(str).unique().tolist())
        raise ValueError(f"Unknown TMS coil labels: {bad}")

    # Validate E-field columns.
    gm_input_cols = [
        build_input_column(TISSUE_PREFIX["GM"], metric)
        for metric in gm_metrics
    ]

    wm_metrics = ["mean"] if include_wm_mean else []
    wm_input_cols = [
        build_input_column(TISSUE_PREFIX["WM"], metric)
        for metric in wm_metrics
    ]

    for col in gm_input_cols + wm_input_cols:
        if col not in df.columns:
            raise ValueError(f"Missing Step 1 E-field column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    gain_lookup = make_gain_lookup(waveform_gains)

    for tau_ms in tau_list_ms:
        k_col = build_k_column(tau_ms)
        df[k_col] = np.nan

        # DC gain for tDCS.
        df.loc[mask_tdcs, k_col] = 1.0

        # TMS gains by coil family and active/sham mode.
        for coil_family in sorted(df.loc[mask_tms, "coil_family"].dropna().unique()):
            for mode in ["Active", "Sham"]:
                key = (coil_family, mode, float(tau_ms))

                if key not in gain_lookup:
                    raise ValueError(f"Missing waveform gain for {key}")

                row_mask = mask_tms & (df["coil_family"] == coil_family) & (df["mode"] == mode)
                df.loc[row_mask, k_col] = gain_lookup[key]

        if df.loc[mask_tms, k_col].isna().any():
            bad = df.loc[mask_tms & df[k_col].isna(), ["coil", "condition", "mode"]].drop_duplicates()
            raise ValueError(
                f"Some TMS rows did not receive k_tau for tau={tau_ms}:\n"
                f"{bad.to_string(index=False)}"
            )

        # GM effective dose.
        for metric in gm_metrics:
            e_col = build_input_column(TISSUE_PREFIX["GM"], metric)
            c_col = build_output_column("GM", metric, tau_ms)

            df[c_col] = np.nan
            df.loc[mask_tdcs, c_col] = df.loc[mask_tdcs, e_col]
            df.loc[mask_tms, c_col] = df.loc[mask_tms, e_col] * df.loc[mask_tms, k_col]

        # Optional WM mean effective dose.
        if include_wm_mean:
            for metric in wm_metrics:
                e_col = build_input_column(TISSUE_PREFIX["WM"], metric)
                c_col = build_output_column("WM", metric, tau_ms)

                df[c_col] = np.nan
                df.loc[mask_tdcs, c_col] = df.loc[mask_tdcs, e_col]
                df.loc[mask_tms, c_col] = df.loc[mask_tms, e_col] * df.loc[mask_tms, k_col]

    return df


def summarize_effective_dose(
    step2: pd.DataFrame,
    tau_list_ms: list[float],
    cohort: str,
) -> pd.DataFrame:
    """
    Summarize GM mean effective dose at intensity 100 for QC.
    """
    rows = []

    ref = step2[
        pd.to_numeric(step2["intensity (%MT)"], errors="coerce") == 100
    ].copy()

    for tau_ms in tau_list_ms:
        c_col = build_output_column("GM", "mean", tau_ms)

        if c_col not in ref.columns:
            continue

        for condition, sub in ref.groupby("condition"):
            values = pd.to_numeric(sub[c_col], errors="coerce").dropna()

            rows.append(
                {
                    "cohort": cohort,
                    "condition": condition,
                    "tau_ms": tau_ms,
                    "metric": "GM_mean",
                    "n": int(values.shape[0]),
                    "median": float(values.median()) if not values.empty else np.nan,
                    "q25": float(values.quantile(0.25)) if not values.empty else np.nan,
                    "q75": float(values.quantile(0.75)) if not values.empty else np.nan,
                }
            )

    return pd.DataFrame(rows)


def print_step2_summary(step2: pd.DataFrame, summary: pd.DataFrame, cohort: str) -> None:
    """
    Print concise QC summary.
    """
    print("\n[STEP 2 EFFECTIVE DOSE SUMMARY]")
    print(f"Cohort: {cohort}")
    print(f"Subjects: {step2['subject_id'].nunique()}")
    print(f"Rows: {len(step2)}")

    print("\nRows by condition:")
    print(step2["condition"].value_counts().sort_index().to_string())

    if not summary.empty:
        print("\nGM mean effective dose at 100% MT / tDCS row:")
        print(
            summary.sort_values(["tau_ms", "condition"])
            .round(6)
            .to_string(index=False)
        )
