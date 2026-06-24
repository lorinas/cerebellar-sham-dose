from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_SHAM_TDCS_CONDITIONS = [
    "MagVenture_Sham0mm",
    "MagVenture_Sham0mm+electrodes",
    "tDCS",
]

ALL_SHAM_TDCS_CONDITIONS = [
    "MagVenture_Sham0mm",
    "MagVenture_Sham0mm+electrodes",
    "Magstim_Sham0mm_kernel",
    "tDCS",
]


def tau_tag(tau_ms: float) -> str:
    return f"tau{str(tau_ms).replace('.', 'p')}ms"


def c_col(metric: str, tau_ms: float) -> str:
    return f"C_GM_ROI_{metric}_V/m_{tau_tag(tau_ms)}"


def safe_stats(values: pd.Series | np.ndarray) -> dict[str, float]:
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {
            "n_rows": 0,
            "C_max_Vm": np.nan,
            "C_p99_Vm": np.nan,
            "C_p95_Vm": np.nan,
            "C_median_Vm": np.nan,
        }

    return {
        "n_rows": int(x.size),
        "C_max_Vm": float(np.max(x)),
        "C_p99_Vm": float(np.percentile(x, 99)),
        "C_p95_Vm": float(np.percentile(x, 95)),
        "C_median_Vm": float(np.median(x)),
    }


def add_gamma_candidates(
    stats: dict[str, float],
    f0_hz: float,
    fmax_hz: float,
) -> dict[str, float]:
    """
    Add gamma_max and candidate gamma fractions.

    gamma_max = (fmax - f0) / Cmax

    This assumes the Step 3 temporal input shape has peak = 1.
    """
    out = dict(stats)

    cmax = out["C_max_Vm"]

    if np.isfinite(cmax) and cmax > 0:
        gamma_max = (fmax_hz - f0_hz) / cmax
    else:
        gamma_max = np.nan

    out["f0_Hz"] = float(f0_hz)
    out["fmax_Hz"] = float(fmax_hz)
    out["gamma_max_Hz_per_Vm"] = float(gamma_max) if np.isfinite(gamma_max) else np.nan
    out["gamma_25pct_Hz_per_Vm"] = 0.25 * gamma_max if np.isfinite(gamma_max) else np.nan
    out["gamma_50pct_Hz_per_Vm"] = 0.50 * gamma_max if np.isfinite(gamma_max) else np.nan
    out["gamma_75pct_Hz_per_Vm"] = 0.75 * gamma_max if np.isfinite(gamma_max) else np.nan

    return out


def load_step2_tables(
    tables_dir: str | Path,
    cohorts: list[str],
) -> pd.DataFrame:
    """
    Load clean Step 2 cohort-level tables and concatenate them.
    """
    tables_dir = Path(tables_dir).expanduser().resolve()

    frames = []

    for cohort in cohorts:
        path = tables_dir / f"step2_effective_dose_{cohort}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Step 2 table not found: {path}")

        df = pd.read_csv(path)
        df["cohort"] = cohort
        df["source_step2_table"] = str(path)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    return out


def add_analysis_set_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add analysis set labels.

    all_conditions:
        all rows, including active TMS.

    primary_sham_tdcs:
        only the primary MagVenture sham/tDCS comparison set.

    all_sham_tdcs:
        all sham conditions to be modeled in Step 3, plus tDCS.
        This excludes active TMS but includes supplementary Magstim sham.
    """
    df = df.copy()
    df["analysis_set_all_conditions"] = True
    df["analysis_set_primary_sham_tdcs"] = df["condition"].isin(PRIMARY_SHAM_TDCS_CONDITIONS)
    df["analysis_set_all_sham_tdcs"] = df["condition"].isin(ALL_SHAM_TDCS_CONDITIONS)

    return df


def scan_gamma_inputs(
    step2_all: pd.DataFrame,
    metrics: list[str],
    tau_list_ms: list[float],
    f0_hz: float,
    fmax_hz: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan C ranges and gamma candidates.

    Returns:
        by_group:
            grouped by analysis_set, metric, tau, cohort, condition, intensity, etc.

        global_summary:
            global maxima by analysis_set, metric, tau.
    """
    df = add_analysis_set_column(step2_all)

    rows = []

    analysis_sets = {
        "all_conditions": df["analysis_set_all_conditions"],
        "primary_sham_tdcs": df["analysis_set_primary_sham_tdcs"],
        "all_sham_tdcs": df["analysis_set_all_sham_tdcs"],
    }

    group_cols = [
        "analysis_set",
        "metric",
        "tau_ms",
        "cohort",
        "modality",
        "coil_family",
        "mode",
        "condition",
        "distance_mm",
        "intensity (%MT)",
    ]

    for analysis_set, mask in analysis_sets.items():
        df_set = df[mask].copy()

        for metric in metrics:
            for tau_ms in tau_list_ms:
                col = c_col(metric, tau_ms)

                if col not in df_set.columns:
                    raise ValueError(f"Missing Step 2 effective-dose column: {col}")

                df_set[col] = pd.to_numeric(df_set[col], errors="coerce")

                gcols_existing = [
                    c for c in [
                        "cohort",
                        "modality",
                        "coil_family",
                        "mode",
                        "condition",
                        "distance_mm",
                        "intensity (%MT)",
                    ]
                    if c in df_set.columns
                ]

                for key, sub in df_set.groupby(gcols_existing, dropna=False):
                    if not isinstance(key, tuple):
                        key = (key,)

                    key_dict = dict(zip(gcols_existing, key))

                    stats = safe_stats(sub[col])
                    stats = add_gamma_candidates(stats, f0_hz=f0_hz, fmax_hz=fmax_hz)

                    rows.append(
                        {
                            "analysis_set": analysis_set,
                            "metric": metric,
                            "tau_ms": float(tau_ms),
                            "C_column": col,
                            "n_subjects": int(sub["subject_id"].nunique()),
                            **stats,
                            **key_dict,
                        }
                    )

    by_group = pd.DataFrame(rows)

    # Global summary by analysis_set, metric, tau across all rows in that set.
    global_rows = []

    for analysis_set, mask in analysis_sets.items():
        df_set = df[mask].copy()

        for metric in metrics:
            for tau_ms in tau_list_ms:
                col = c_col(metric, tau_ms)

                stats = safe_stats(df_set[col])
                stats = add_gamma_candidates(stats, f0_hz=f0_hz, fmax_hz=fmax_hz)

                global_rows.append(
                    {
                        "analysis_set": analysis_set,
                        "metric": metric,
                        "tau_ms": float(tau_ms),
                        "C_column": col,
                        "n_subjects": int(df_set["subject_id"].nunique()),
                        **stats,
                    }
                )

    global_summary = pd.DataFrame(global_rows)

    sort_cols_group = [
        "analysis_set",
        "metric",
        "tau_ms",
        "cohort",
        "condition",
        "intensity (%MT)",
    ]
    sort_cols_group = [c for c in sort_cols_group if c in by_group.columns]

    by_group = by_group.sort_values(sort_cols_group).reset_index(drop=True)
    global_summary = global_summary.sort_values(
        ["analysis_set", "metric", "tau_ms"]
    ).reset_index(drop=True)

    return by_group, global_summary


def print_gamma_global_summary(global_summary: pd.DataFrame) -> None:
    """
    Print compact global gamma summary.
    """
    print("\n[STEP 2b GLOBAL GAMMA SUMMARY]")
    cols = [
        "analysis_set",
        "metric",
        "tau_ms",
        "n_subjects",
        "n_rows",
        "C_max_Vm",
        "C_p99_Vm",
        "C_p95_Vm",
        "C_median_Vm",
        "gamma_max_Hz_per_Vm",
        "gamma_50pct_Hz_per_Vm",
    ]

    view = global_summary[cols].copy()
    print(view.round(6).to_string(index=False))
