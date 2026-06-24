from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from shamdose.crbl_model import CRBLModel, run_crbl_meanfield, extract_pc_rate_hz


DEFAULT_TMS_INTENSITIES = list(range(0, 101, 10))

DEFAULT_MODELED_CONDITIONS = [
    "MagVenture_Sham0mm",
    "MagVenture_Sham0mm+electrodes",
    "Magstim_Sham0mm_kernel",
    "tDCS",
]

ACTIVE_CONDITIONS = [
    "MagVenture_Active0mm",
    "Magstim_Active0mm",
]

# Population-rate indices in the CRBL_MF output.
# This follows the CRBL_MF state-vector convention used in the original runner:
#   X[:, 0]  = GrC population activity
#   X[:, 1]  = GoC population activity
#   X[:, 9]  = MLI population activity
#   X[:, 10] = PC population activity
POPULATION_RATE_INDICES = {
    "GrC": 0,
    "GoC": 1,
    "MLI": 9,
    "PC": 10,
}

POPULATION_PLOT_ORDER = ["PC", "MLI", "GoC", "GrC", "mossy_drive"]

POPULATION_COLORS = {
    "PC": "#008000",
    "MLI": "#FFD000",      # hard yellow
    "GoC": "#3366CC",
    "GrC": "#CC3333",
    "mossy_drive": "#CC66CC",
}


def safe_name(x: object) -> str:
    """
    Make a string safe for filenames.
    """
    s = str(x)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_")


def make_time_vector(sim_len_s: float, dt_s: float) -> np.ndarray:
    return np.arange(0.0, sim_len_s, dt_s)


def half_sine_bump(t_s: np.ndarray, t0_s: float, dur_s: float, amp: float = 1.0) -> np.ndarray:
    y = np.zeros_like(t_s, dtype=float)
    u = (t_s - t0_s) / dur_s
    mask = (u >= 0.0) & (u <= 1.0)
    y[mask] = np.sin(np.pi * u[mask])
    return amp * y


def stimulus_shape(
    t_s: np.ndarray,
    modality: str,
    stim_on_s: float,
    bump_dur_s: float,
) -> np.ndarray:
    """
    Unit-peak input shape for the mean-field model.

    TMS:
        brief model-scale half-sine perturbation.

    tDCS:
        step onset. This is an acute active-stimulation benchmark,
        not a full-session tDCS model.
    """
    modality = str(modality).lower().strip()

    if modality == "tdcs":
        shape = np.zeros_like(t_s, dtype=float)
        shape[t_s >= stim_on_s] = 1.0
        return shape

    if modality == "tms":
        return half_sine_bump(t_s, stim_on_s, bump_dur_s, amp=1.0)

    raise ValueError(f"Unsupported modality for stimulus_shape: {modality}")


def integrate_auc(y: np.ndarray, x: np.ndarray) -> float:
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(y, x))


def compute_pc_metrics(
    t_s: np.ndarray,
    pc_hz: np.ndarray,
    baseline_win_s: tuple[float, float],
    analysis_win_s: tuple[float, float],
) -> dict[str, float]:
    """
    Compute acute Purkinje-cell response metrics.
    """
    tb0, tb1 = baseline_win_s
    tw0, tw1 = analysis_win_s

    baseline_mask = (t_s >= tb0) & (t_s < tb1)
    window_mask = (t_s >= tw0) & (t_s < tw1)

    if not baseline_mask.any():
        raise RuntimeError("Empty baseline window.")

    if not window_mask.any():
        raise RuntimeError("Empty analysis window.")

    pc_base = pc_hz[baseline_mask]
    pc_win = pc_hz[window_mask]
    t_win = t_s[window_mask]

    base = float(pc_base.mean())
    sd = float(pc_base.std())
    peak = float(pc_win.max())
    trough = float(pc_win.min())

    dev = pc_win - base

    return {
        "PC_baseline_mean_Hz": base,
        "PC_baseline_sd_Hz": sd,
        "PC_peak_Hz": peak,
        "PC_trough_Hz": trough,
        "PC_dPEAK_Hz": peak - base,
        "PC_dTROUGH_Hz": trough - base,
        "PC_RMS_Hz": float(np.sqrt(np.mean(dev ** 2))),
        "PC_AUC_excite_Hz_s": integrate_auc(np.maximum(dev, 0.0), t_win),
        "PC_AUC_inhib_Hz_s": integrate_auc(-np.minimum(dev, 0.0), t_win),
        "PC_AUC_net_Hz_s": integrate_auc(dev, t_win),
    }


def choose_qc_subjects(
    anatomy: pd.DataFrame,
    available_subjects: list[str],
) -> dict[str, str]:
    """
    Pick shallow, median-depth, and deep subjects for QC traces.

    Returns:
        {subject_id: qc_label}
    """
    if anatomy is None or anatomy.empty:
        return {}

    df = anatomy[
        anatomy["subject_id"].astype(str).isin([str(s) for s in available_subjects])
        & (anatomy["status"].astype(str) == "OK")
    ].copy()

    if df.empty or "gm_depth_from_scalp_mm" not in df.columns:
        return {}

    df["gm_depth_from_scalp_mm"] = pd.to_numeric(
        df["gm_depth_from_scalp_mm"], errors="coerce"
    )
    df = df.dropna(subset=["gm_depth_from_scalp_mm"]).copy()

    if df.empty:
        return {}

    shallow_row = df.loc[df["gm_depth_from_scalp_mm"].idxmin()]
    deep_row = df.loc[df["gm_depth_from_scalp_mm"].idxmax()]
    median_depth = float(df["gm_depth_from_scalp_mm"].median())
    median_row = df.iloc[(df["gm_depth_from_scalp_mm"] - median_depth).abs().argsort().iloc[0]]

    selected = {
        str(shallow_row["subject_id"]): "shallow_depth",
        str(median_row["subject_id"]): "median_depth",
        str(deep_row["subject_id"]): "deep_depth",
    }

    return selected


def build_qc_trace_plan(
    anatomy: pd.DataFrame,
    available_subjects: list[str],
) -> dict[tuple[str, str, int], str]:
    """
    Build a selected trace plan.

    Saves traces for:
        - tDCS
        - MagVenture sham at 50 and 100% MT
        - MagVenture sham + electrodes at 50 and 100% MT
        - Magstim sham at 20 and 100% MT

    for shallow, median-depth, and deep subjects.
    """
    selected_subjects = choose_qc_subjects(anatomy, available_subjects)

    trace_conditions = [
        ("tDCS", 100),
        ("MagVenture_Sham0mm", 50),
        ("MagVenture_Sham0mm", 100),
        ("MagVenture_Sham0mm+electrodes", 50),
        ("MagVenture_Sham0mm+electrodes", 100),
        ("Magstim_Sham0mm_kernel", 20),
        ("Magstim_Sham0mm_kernel", 100),
    ]

    plan = {}

    for subject_id, depth_label in selected_subjects.items():
        for condition, intensity in trace_conditions:
            plan[(subject_id, condition, int(intensity))] = depth_label

    return plan


def extract_population_traces(X: np.ndarray, fmossy_hz: np.ndarray) -> dict[str, np.ndarray]:
    """
    Extract population-rate traces from CRBL_MF output.

    The mapping follows the original CRBL_MF runner convention.
    """
    traces = {}

    for name, idx in POPULATION_RATE_INDICES.items():
        if X.shape[1] <= idx:
            raise RuntimeError(
                f"Cannot extract {name}: CRBL_MF output has {X.shape[1]} columns, "
                f"but {name} requires column {idx}."
            )
        traces[name] = np.asarray(X[:, idx], dtype=float)

    traces["mossy_drive"] = np.asarray(fmossy_hz, dtype=float)

    return traces


def save_qc_subject_panels(
    qc_trace_records: list[dict],
    out_dir: str | Path,
    stim_on_s: float,
    baseline_win_s: tuple[float, float],
    analysis_win_s: tuple[float, float],
) -> None:
    """
    Save separate stacked-population QC panels per representative subject.

    Output:
        one PNG for active tDCS
        one PNG for MagVenture sham
        one PNG for MagVenture sham + electrodes
        one PNG for Magstim sham

    Rows:
        PC, MLI, GoC, GrC, mossy drive

    Columns:
        selected intensities for that condition family.

    These PNGs are QC/illustration outputs only.
    They are not used in downstream analyses.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_specs = [
        {
            "panel_name": "active_tDCS",
            "title": "active tDCS",
            "items": [("tDCS", 100)],
        },
        {
            "panel_name": "MagVenture_sham",
            "title": "MagVenture sham",
            "items": [
                ("MagVenture_Sham0mm", 50),
                ("MagVenture_Sham0mm", 100),
            ],
        },
        {
            "panel_name": "MagVenture_sham_plus_electrodes",
            "title": "MagVenture sham + electrodes",
            "items": [
                ("MagVenture_Sham0mm+electrodes", 50),
                ("MagVenture_Sham0mm+electrodes", 100),
            ],
        },
        {
            "panel_name": "Magstim_sham",
            "title": "Magstim sham",
            "items": [
                ("Magstim_Sham0mm_kernel", 20),
                ("Magstim_Sham0mm_kernel", 100),
            ],
        },
    ]

    by_subject: dict[tuple[str, str, str], list[dict]] = {}

    for rec in qc_trace_records:
        key = (
            str(rec["cohort"]),
            str(rec["subject_id"]),
            str(rec["depth_label"]),
        )
        by_subject.setdefault(key, []).append(rec)

    import matplotlib.pyplot as plt

    for (cohort, subject_id, depth_label), records in by_subject.items():
        lookup = {
            (str(r["condition"]), int(r["intensity"])): r
            for r in records
        }

        for spec in panel_specs:
            ordered_records = [
                lookup[item]
                for item in spec["items"]
                if item in lookup
            ]

            if not ordered_records:
                continue

            n_rows = len(POPULATION_PLOT_ORDER)
            n_cols = len(ordered_records)

            fig_width = 4.3 * n_cols
            fig_height = 1.35 * n_rows + 1.15

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(fig_width, fig_height),
                sharex=True,
                constrained_layout=True,
            )

            axes = np.asarray(axes)

            if axes.ndim == 1:
                if n_rows == 1:
                    axes = axes.reshape(1, -1)
                else:
                    axes = axes.reshape(-1, 1)

            for col_idx, rec in enumerate(ordered_records):
                t_s = rec["t_s"]
                t_ms = t_s * 1000.0
                traces = rec["population_traces"]
                condition = str(rec["condition"])
                intensity = int(rec["intensity"])

                if condition == "tDCS":
                    col_title = "2 mA"
                else:
                    col_title = f"{intensity}% MT"

                for row_idx, pop_name in enumerate(POPULATION_PLOT_ORDER):
                    ax = axes[row_idx, col_idx]
                    y = traces[pop_name]
                    color = POPULATION_COLORS.get(pop_name, "black")

                    # Clean white background: no full blue background, no grid.
                    ax.set_facecolor("white")

                    ax.plot(
                        t_ms,
                        y,
                        linewidth=2.0,
                        color=color,
                    )

                    # Stimulation onset.
                    ax.axvline(
                        stim_on_s * 1000.0,
                        linestyle="--",
                        linewidth=0.9,
                        color="black",
                        alpha=0.85,
                    )

                    # Only shade the analysis window.
                    ax.axvspan(
                        analysis_win_s[0] * 1000.0,
                        analysis_win_s[1] * 1000.0,
                        color="0.85",
                        alpha=0.45,
                        linewidth=0,
                    )

                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                    if col_idx == 0:
                        label = "mossy\ndrive" if pop_name == "mossy_drive" else pop_name
                        ax.set_ylabel(f"{label}\nHz", fontsize=9)
                    else:
                        ax.set_ylabel("")

                    if row_idx == 0:
                        ax.set_title(col_title, fontsize=11)

                    if row_idx < n_rows - 1:
                        ax.tick_params(labelbottom=False)

            for ax in axes[-1, :]:
                ax.set_xlabel("Time (ms)", fontsize=9)

            fig.suptitle(
                f"{cohort} | {subject_id} | {depth_label} | {spec['title']}",
                fontsize=13,
                fontweight="bold",
            )

            out_png = (
                out_dir
                / f"{safe_name(cohort)}_{safe_name(subject_id)}_{safe_name(depth_label)}_{safe_name(spec['panel_name'])}_QCpanel.png"
            )

            fig.savefig(out_png, dpi=240, bbox_inches="tight")
            plt.close(fig)


def run_meanfield_subject_chunk_worker(
    worker_id: int,
    repo_root: str,
    step2_chunk: pd.DataFrame,
    anatomy: pd.DataFrame,
    cohort: str,
    subject_ids: list[str],
    conditions_to_model: list[str],
    dose_column: str,
    metric: str,
    tau_ms: float,
    gamma_hz_per_vm: float,
    gamma_fraction: float,
    baseline_drive_hz: float,
    fmax_hz: float,
    analysis_set: str,
    gamma_source_table: str,
    gamma_source_column: str,
    gamma_max_hz_per_vm: float,
    gamma_cmax_vm: float,
    include_active: bool,
) -> pd.DataFrame:
    """
    Worker function for parallel Step 3.

    Each worker loads its own CRBL_MF model and processes a chunk of subjects.
    This avoids trying to pickle the loaded model object across processes.
    """
    from shamdose.crbl_model import load_crbl_model

    print(
        f"[WORKER {worker_id}] Loading CRBL_MF model for {len(subject_ids)} subjects",
        flush=True,
    )

    model = load_crbl_model(repo_root)

    print(
        f"[WORKER {worker_id}] Running subjects: {subject_ids}",
        flush=True,
    )

    out = run_meanfield_for_cohort(
        model=model,
        step2=step2_chunk,
        anatomy=anatomy,
        cohort=cohort,
        conditions_to_model=conditions_to_model,
        dose_column=dose_column,
        metric=metric,
        tau_ms=tau_ms,
        gamma_hz_per_vm=gamma_hz_per_vm,
        gamma_fraction=gamma_fraction,
        baseline_drive_hz=baseline_drive_hz,
        fmax_hz=fmax_hz,
        analysis_set=analysis_set,
        gamma_source_table=gamma_source_table,
        gamma_source_column=gamma_source_column,
        gamma_max_hz_per_vm=gamma_max_hz_per_vm,
        gamma_cmax_vm=gamma_cmax_vm,
        include_active=include_active,
        subject_filter=subject_ids,
        limit_subjects=None,
        save_qc_traces=False,
        qc_trace_dir=None,
    )

    print(
        f"[WORKER {worker_id}] Done: {len(subject_ids)} subjects, {len(out)} rows",
        flush=True,
    )

    return out


def split_subjects(subject_ids: list[str], n_chunks: int) -> list[list[str]]:
    """
    Split subject IDs into approximately equal chunks.
    """
    if n_chunks <= 1:
        return [subject_ids]

    chunks = [[] for _ in range(n_chunks)]

    for i, subject_id in enumerate(subject_ids):
        chunks[i % n_chunks].append(subject_id)

    return [chunk for chunk in chunks if chunk]


def run_meanfield_for_cohort(
    model: CRBLModel,
    step2: pd.DataFrame,
    anatomy: pd.DataFrame,
    cohort: str,
    conditions_to_model: list[str],
    dose_column: str,
    metric: str,
    tau_ms: float,
    gamma_hz_per_vm: float,
    gamma_fraction: float,
    baseline_drive_hz: float,
    fmax_hz: float,
    analysis_set: str = "all_sham_tdcs",
    gamma_source_table: str = "",
    gamma_source_column: str = "",
    gamma_max_hz_per_vm: float = np.nan,
    gamma_cmax_vm: float = np.nan,
    dt_s: float = 1e-4,
    sim_len_s: float = 0.50,
    stim_on_s: float = 0.120,
    baseline_pre_s: float = 0.040,
    analysis_dur_s: float = 0.035,
    bump_dur_s: float = 3.5e-3,
    include_active: bool = False,
    subject_filter: list[str] | None = None,
    limit_subjects: int | None = None,
    save_qc_traces: bool = False,
    qc_trace_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Run the cerebellar mean-field model for one cohort.

    Default use:
        sham conditions + tDCS only.

    Active TMS should only be included explicitly, and only with an appropriate
    gamma calibration or for exploratory diagnostics.
    """
    df = step2.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["condition"] = df["condition"].astype(str)
    df["modality"] = df["modality"].astype(str).str.lower().str.strip()
    df["intensity (%MT)"] = pd.to_numeric(df["intensity (%MT)"], errors="coerce")

    if dose_column not in df.columns:
        raise ValueError(f"Step 2 table missing dose column: {dose_column}")

    conditions = list(conditions_to_model)

    if include_active:
        for cond in ACTIVE_CONDITIONS:
            if cond not in conditions:
                conditions.append(cond)
    else:
        conditions = [c for c in conditions if c not in ACTIVE_CONDITIONS]

    df = df[df["condition"].isin(conditions)].copy()

    if subject_filter is not None:
        keep = {str(s) for s in subject_filter}
        df = df[df["subject_id"].isin(keep)].copy()

    subject_ids = sorted(df["subject_id"].unique().tolist())

    if limit_subjects is not None:
        if limit_subjects <= 0:
            raise ValueError("limit_subjects must be positive.")
        subject_ids = subject_ids[:limit_subjects]
        df = df[df["subject_id"].isin(subject_ids)].copy()

    if df.empty:
        raise ValueError("No Step 2 rows remain after filtering.")

    t_s = make_time_vector(sim_len_s, dt_s)
    baseline_win_s = (stim_on_s - baseline_pre_s, stim_on_s)
    analysis_win_s = (stim_on_s, min(stim_on_s + analysis_dur_s, float(t_s[-1])))

    trace_plan = {}
    qc_trace_records: list[dict] = []

    if save_qc_traces:
        if qc_trace_dir is None:
            raise ValueError("qc_trace_dir must be supplied when save_qc_traces=True")
        trace_plan = build_qc_trace_plan(anatomy, subject_ids)

    rows = []

    for subject_id, subdf in df.groupby("subject_id", sort=True):
        print(f"[STEP3] {cohort} {subject_id}: {len(subdf)} rows", flush=True)

        for _, r in subdf.iterrows():
            condition = str(r["condition"])
            modality = str(r["modality"])
            intensity = int(r["intensity (%MT)"])

            C_val = float(r[dose_column])

            shape = stimulus_shape(
                t_s=t_s,
                modality=modality,
                stim_on_s=stim_on_s,
                bump_dur_s=bump_dur_s,
            )

            fmossy_hz = baseline_drive_hz + gamma_hz_per_vm * C_val * shape

            fmossy_max = float(np.max(fmossy_hz))
            fmossy_min = float(np.min(fmossy_hz))
            fmossy_delta_peak = float(fmossy_max - baseline_drive_hz)

            if fmossy_max > fmax_hz + 1e-9:
                raise RuntimeError(
                    f"{cohort} {subject_id} {condition}@{intensity}: "
                    f"fmossy max {fmossy_max:.3f} Hz exceeds fmax {fmax_hz:.3f} Hz. "
                    "This indicates gamma/calibration mismatch."
                )

            X = run_crbl_meanfield(
                model=model,
                t_s=t_s,
                fmossy_hz=fmossy_hz,
            )
            pc_hz = extract_pc_rate_hz(X)

            metrics = compute_pc_metrics(
                t_s=t_s,
                pc_hz=pc_hz,
                baseline_win_s=baseline_win_s,
                analysis_win_s=analysis_win_s,
            )

            row = {
                "SUBJECT_ID": subject_id,
                "subject_id": subject_id,
                "cohort": cohort,
                "GROUP": cohort,
                "age": r.get("age", np.nan),
                "sex": r.get("sex", np.nan),
                "MODE": "tDCS" if modality == "tdcs" else "TMS",
                "modality": modality,
                "CONDITION": condition,
                "condition": condition,
                "INTENSITY_MT": intensity,
                "PROTOCOL": "tDCS_step" if modality == "tdcs" else "single_pulse_half_sine",
                "MAP": f"linear_gamma_{analysis_set}",
                "analysis_set": analysis_set,
                "METRIC": metric,
                "TAU_ms": float(tau_ms),
                "dose_column": dose_column,
                "SIM_SCOPE": "halfsec_acute",
                "T_STIM_ON_s": float(stim_on_s),
                "T_BASE_START_s": float(baseline_win_s[0]),
                "T_BASE_END_s": float(baseline_win_s[1]),
                "T_WIN_START_s": float(analysis_win_s[0]),
                "T_WIN_END_s": float(analysis_win_s[1]),
                "BUMP_DUR_MS": float(bump_dur_s * 1000.0),
                "dt_s": float(dt_s),
                "sim_len_s": float(sim_len_s),
                "C_val_Vm": float(C_val),
                "baseline_drive_Hz": float(baseline_drive_hz),
                "fmossy_min_Hz": fmossy_min,
                "fmossy_max_Hz": fmossy_max,
                "fmossy_delta_peak_Hz": fmossy_delta_peak,
                "gamma_Hz_per_Vm": float(gamma_hz_per_vm),
                "gamma_fraction": float(gamma_fraction),
                "gamma_max_Hz_per_Vm": float(gamma_max_hz_per_vm),
                "gamma_Cmax_Vm": float(gamma_cmax_vm),
                "gamma_source_table": gamma_source_table,
                "gamma_source_column": gamma_source_column,
                "fmax_Hz": float(fmax_hz),
                "coil_family": r.get("coil_family", ""),
                "mode": r.get("mode", ""),
            }
            row.update(metrics)
            rows.append(row)

            key = (subject_id, condition, intensity)
            if save_qc_traces and key in trace_plan:
                qc_trace_records.append(
                    {
                        "cohort": cohort,
                        "subject_id": subject_id,
                        "depth_label": trace_plan[key],
                        "condition": condition,
                        "intensity": intensity,
                        "t_s": t_s.copy(),
                        "population_traces": {
                            k: v.copy()
                            for k, v in extract_population_traces(X, fmossy_hz).items()
                        },
                    }
                )

    if save_qc_traces and qc_trace_records:
        save_qc_subject_panels(
            qc_trace_records=qc_trace_records,
            out_dir=Path(qc_trace_dir) / cohort,
            stim_on_s=stim_on_s,
            baseline_win_s=baseline_win_s,
            analysis_win_s=analysis_win_s,
        )

    out = pd.DataFrame(rows)

    validate_step3_output(
        out,
        expected_subjects=subject_ids,
        include_active=include_active,
    )

    return out


def validate_step3_output(
    out: pd.DataFrame,
    expected_subjects: list[str],
    include_active: bool,
) -> None:
    """
    Validate row counts and basic finite outputs.
    """
    if out.empty:
        raise ValueError("Step 3 output is empty.")

    expected_rows_per_subject = 56 if include_active else 34

    observed_subjects = sorted(out["SUBJECT_ID"].astype(str).unique().tolist())
    missing = sorted(set(expected_subjects) - set(observed_subjects))
    extra = sorted(set(observed_subjects) - set(expected_subjects))

    if missing:
        raise ValueError(f"Step 3 output missing subjects: {missing[:20]}")

    if extra:
        raise ValueError(f"Step 3 output has unexpected subjects: {extra[:20]}")

    counts = out.groupby("SUBJECT_ID").size()
    bad = counts[counts != expected_rows_per_subject]

    if len(bad) > 0:
        raise ValueError(
            f"Subjects with unexpected Step 3 row counts. "
            f"Expected {expected_rows_per_subject} rows/subject:\n"
            f"{bad.head(20).to_string()}"
        )

    numeric_cols = [
        "C_val_Vm",
        "fmossy_max_Hz",
        "PC_baseline_mean_Hz",
        "PC_dPEAK_Hz",
        "PC_RMS_Hz",
    ]

    for col in numeric_cols:
        if col not in out.columns:
            raise ValueError(f"Missing Step 3 output column: {col}")

        values = pd.to_numeric(out[col], errors="coerce")

        if values.isna().any():
            raise ValueError(f"Column {col} contains NaN values.")

        if not np.all(np.isfinite(values)):
            raise ValueError(f"Column {col} contains non-finite values.")


def summarize_step3_output(out: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """
    Summarize Step 3 outputs for QC.
    """
    rows = []

    for condition, sub in out.groupby("CONDITION"):
        rows.append(
            {
                "cohort": cohort,
                "condition": condition,
                "n_subjects": int(sub["SUBJECT_ID"].nunique()),
                "n_rows": int(len(sub)),
                "C_val_median": float(pd.to_numeric(sub["C_val_Vm"], errors="coerce").median()),
                "fmossy_max_median": float(pd.to_numeric(sub["fmossy_max_Hz"], errors="coerce").median()),
                "PC_dPEAK_median": float(pd.to_numeric(sub["PC_dPEAK_Hz"], errors="coerce").median()),
                "PC_RMS_median": float(pd.to_numeric(sub["PC_RMS_Hz"], errors="coerce").median()),
            }
        )

    return pd.DataFrame(rows)


def print_step3_summary(out: pd.DataFrame, summary: pd.DataFrame, cohort: str) -> None:
    print("\n[STEP 3 MEAN-FIELD SUMMARY]")
    print(f"Cohort: {cohort}")
    print(f"Subjects: {out['SUBJECT_ID'].nunique()}")
    print(f"Rows: {len(out)}")

    print("\nRows by condition:")
    print(out["CONDITION"].value_counts().sort_index().to_string())

    print("\nSummary by condition:")
    print(summary.round(6).to_string(index=False))
