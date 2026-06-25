from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import simnibs

from shamdose.anatomy import parse_xyz_text, find_subject_m2m_dir
from shamdose.magstim_kernel_subjects import (
    INTENSITIES,
    weighted_quantile,
    distribution_stats,
    fill_stat_columns,
    load_kernel_interpolators,
    interpolate_kernel_ratio,
    extract_roi_elements_from_mesh,
    compute_coil_local_basis,
    centroids_to_coil_xy_cm,
    safe_weighted_mean,
    safe_array_min,
    safe_array_max,
)


SOURCE_ACTIVE_CONDITION = "MagVenture_Active0mm"
MAGVENTURE_SHAM_CONDITION = "MagVenture_Sham0mm"
MAGVENTURE_SHAM_ELEC_CONDITION = "MagVenture_Sham0mm+electrodes"
RATIO_COLUMN = "magventure_sham_over_active_regularized"


def find_active_magventure_mesh(
    subject_id: str,
    active_100: pd.Series,
    m2m_dir: str | Path,
) -> tuple[Path, str, str]:
    """
    Prefer the current YAML-derived subject folder over stale Step 1 CSV paths.
    """
    active_mesh_path_from_csv = Path(str(active_100["path"])).expanduser()
    active_mesh_path_from_csv_resolved = active_mesh_path_from_csv.resolve()

    subject_dir = Path(m2m_dir).expanduser().resolve().parent

    candidates = sorted(
        subject_dir.glob("*/TMS_MagVenture_Active_0mm/*_scalar.msh")
    )

    if not candidates:
        candidates = sorted(
            subject_dir.rglob("TMS_MagVenture_Active_0mm/*_scalar.msh")
        )

    if candidates:
        active_mesh_path = max(candidates, key=lambda p: p.stat().st_mtime).resolve()
        active_mesh_path_source = "searched_current_yaml_subject_dir"
    elif active_mesh_path_from_csv_resolved.exists():
        active_mesh_path = active_mesh_path_from_csv_resolved
        active_mesh_path_source = "step1_path_column_fallback"
    else:
        raise FileNotFoundError(
            f"{subject_id}: active MagVenture scalar mesh not found.\n"
            f"  Step 1 CSV path: {active_mesh_path_from_csv_resolved}\n"
            f"  YAML/current subject_dir: {subject_dir}\n"
            "  expected pattern: */TMS_MagVenture_Active_0mm/*_scalar.msh"
        )

    if active_mesh_path != active_mesh_path_from_csv_resolved:
        print(
            f"[INFO] {subject_id}: Step 1 MagVenture mesh path points elsewhere; "
            f"using YAML-root mesh: {active_mesh_path}",
            flush=True,
        )

    return active_mesh_path, active_mesh_path_source, str(active_mesh_path_from_csv)


def estimate_electrode_delta_scale(
    step1_subject_rows: pd.DataFrame,
    active_100: pd.Series,
) -> float:
    """
    Estimate the additive electrode contribution as an active-field-like scale.

    This preserves the previous Smith-Peterchev-based electrode contribution while
    replacing the magnetic MagVenture sham component with the empirical kernel.
    """
    col = "roi_graymatter_mean_|E|_V/m"

    sham100 = step1_subject_rows[
        (step1_subject_rows["condition"].astype(str) == MAGVENTURE_SHAM_CONDITION)
        & (pd.to_numeric(step1_subject_rows["intensity (%MT)"], errors="coerce") == 100)
    ]

    elec100 = step1_subject_rows[
        (step1_subject_rows["condition"].astype(str) == MAGVENTURE_SHAM_ELEC_CONDITION)
        & (pd.to_numeric(step1_subject_rows["intensity (%MT)"], errors="coerce") == 100)
    ]

    if len(sham100) != 1:
        raise ValueError(f"Expected one MagVenture sham 100% row, found {len(sham100)}")

    if len(elec100) != 1:
        raise ValueError(f"Expected one MagVenture sham+electrodes 100% row, found {len(elec100)}")

    active_mean = float(active_100[col])
    sham_mean = float(sham100.iloc[0][col])
    elec_mean = float(elec100.iloc[0][col])

    if active_mean <= 0:
        raise ValueError("Active MagVenture GM mean at 100% is non-positive.")

    delta_scale = (elec_mean - sham_mean) / active_mean

    return float(max(delta_scale, 0.0))


def make_magventure_kernel_rows_for_subject(
    step1_subject_rows: pd.DataFrame,
    m2m_dir: str | Path,
    kernel_linear,
    kernel_nearest,
    roi_radius_mm: float,
    target_distance_below_iz_mm: float,
    y_sign: float = 1.0,
    max_ratio: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Create empirical-kernel MagVenture sham and sham+electrode rows for one subject.

    Output conditions keep the canonical names:
        MagVenture_Sham0mm
        MagVenture_Sham0mm+electrodes
    """
    subject_id = str(step1_subject_rows["subject_id"].iloc[0])

    active_rows = step1_subject_rows[
        step1_subject_rows["condition"].astype(str) == SOURCE_ACTIVE_CONDITION
    ].copy()

    if active_rows.empty:
        raise ValueError(f"{subject_id}: missing {SOURCE_ACTIVE_CONDITION} rows")

    active_100_rows = active_rows[
        pd.to_numeric(active_rows["intensity (%MT)"], errors="coerce") == 100
    ]

    if len(active_100_rows) != 1:
        raise ValueError(f"{subject_id}: expected one active 100% row, found {len(active_100_rows)}")

    active_100 = active_100_rows.iloc[0]

    active_mesh_path, active_mesh_path_source, active_mesh_path_from_csv = find_active_magventure_mesh(
        subject_id=subject_id,
        active_100=active_100,
        m2m_dir=m2m_dir,
    )

    stim_center = parse_xyz_text(active_100["stim_center_subject_xyz_mm"])
    roi_center = parse_xyz_text(active_100["roi_center_gm_subject_xyz_mm"])

    mesh = simnibs.read_msh(str(active_mesh_path))

    basis = compute_coil_local_basis(
        mesh=mesh,
        m2m_dir=m2m_dir,
        stim_center_xyz=stim_center,
        target_distance_below_iz_mm=target_distance_below_iz_mm,
        y_sign=y_sign,
    )

    gm = extract_roi_elements_from_mesh(
        mesh=mesh,
        roi_center_xyz=roi_center,
        roi_radius_mm=roi_radius_mm,
        tissue_tag=simnibs.ElementTags.GM,
    )

    wm = extract_roi_elements_from_mesh(
        mesh=mesh,
        roi_center_xyz=roi_center,
        roi_radius_mm=roi_radius_mm,
        tissue_tag=simnibs.ElementTags.WM,
    )

    gm_xy = centroids_to_coil_xy_cm(gm["centroids"], basis)
    wm_xy = centroids_to_coil_xy_cm(wm["centroids"], basis)

    gm_ratio = interpolate_kernel_ratio(
        gm_xy,
        kernel_linear,
        kernel_nearest,
        min_ratio=0.0,
        max_ratio=max_ratio,
    )

    wm_ratio = interpolate_kernel_ratio(
        wm_xy,
        kernel_linear,
        kernel_nearest,
        min_ratio=0.0,
        max_ratio=max_ratio,
    )

    gm_kernel_E100 = gm["active_E"] * gm_ratio
    wm_kernel_E100 = wm["active_E"] * wm_ratio

    electrode_delta_scale = estimate_electrode_delta_scale(
        step1_subject_rows=step1_subject_rows,
        active_100=active_100,
    )

    gm_kernel_elec_E100 = gm_kernel_E100 + gm["active_E"] * electrode_delta_scale
    wm_kernel_elec_E100 = wm_kernel_E100 + wm["active_E"] * electrode_delta_scale

    rows = []

    for condition, gm_E100, wm_E100 in [
        (MAGVENTURE_SHAM_CONDITION, gm_kernel_E100, wm_kernel_E100),
        (MAGVENTURE_SHAM_ELEC_CONDITION, gm_kernel_elec_E100, wm_kernel_elec_E100),
    ]:
        template_rows = step1_subject_rows[
            step1_subject_rows["condition"].astype(str) == condition
        ].copy()

        if template_rows.empty:
            raise ValueError(f"{subject_id}: missing template rows for {condition}")

        for intensity in INTENSITIES:
            frac = float(intensity) / 100.0

            tr = template_rows[
                pd.to_numeric(template_rows["intensity (%MT)"], errors="coerce") == intensity
            ]

            if len(tr) != 1:
                raise ValueError(f"{subject_id}: {condition} intensity {intensity} row count = {len(tr)}")

            row = tr.iloc[0].copy()

            gm_stats = distribution_stats(gm_E100 * frac, gm["volumes"])
            wm_stats = distribution_stats(wm_E100 * frac, wm["volumes"])

            row = fill_stat_columns(row, "roi_graymatter", gm_stats)
            row = fill_stat_columns(row, "roi_whitematter", wm_stats)

            row["magventure_kernel_method"] = "smith_peterchev_xy_local_ratio"
            row["magventure_kernel_condition_source"] = SOURCE_ACTIVE_CONDITION
            row["magventure_kernel_ratio_col"] = RATIO_COLUMN
            row["magventure_kernel_y_sign"] = float(y_sign)
            row["magventure_kernel_max_ratio"] = float(max_ratio)
            row["magventure_kernel_electrode_delta_scale"] = float(electrode_delta_scale)
            row["magventure_kernel_gm_ratio_mean"] = safe_weighted_mean(gm_ratio, gm["volumes"])
            row["magventure_kernel_gm_ratio_p95"] = weighted_quantile(gm_ratio, gm["volumes"], 0.95)
            row["magventure_kernel_wm_ratio_mean"] = safe_weighted_mean(wm_ratio, wm["volumes"])
            row["magventure_kernel_wm_ratio_p95"] = weighted_quantile(wm_ratio, wm["volumes"], 0.95)
            row["magventure_kernel_stim_recompute_error_mm"] = basis["stim_recompute_error_mm"]
            row["magventure_kernel_orientation_source"] = basis["orientation_source"]
            row["magventure_kernel_cap_csv"] = basis["cap_csv"]
            row["magventure_kernel_active_mesh_source"] = active_mesh_path_source
            row["magventure_kernel_active_mesh_path_from_csv"] = active_mesh_path_from_csv
            row["magventure_kernel_active_mesh_path_used"] = str(active_mesh_path)

            rows.append(row)

    out = pd.DataFrame(rows)

    qc = {
        "subject_id": subject_id,
        "n_gm_elements": int(len(gm["active_E"])),
        "n_wm_elements": int(len(wm["active_E"])),
        "gm_ratio_mean": safe_weighted_mean(gm_ratio, gm["volumes"]),
        "gm_ratio_median": weighted_quantile(gm_ratio, gm["volumes"], 0.50),
        "gm_ratio_p95": weighted_quantile(gm_ratio, gm["volumes"], 0.95),
        "gm_ratio_min": safe_array_min(gm_ratio),
        "gm_ratio_max": safe_array_max(gm_ratio),
        "wm_ratio_mean": safe_weighted_mean(wm_ratio, wm["volumes"]),
        "wm_ratio_median": weighted_quantile(wm_ratio, wm["volumes"], 0.50),
        "wm_ratio_p95": weighted_quantile(wm_ratio, wm["volumes"], 0.95),
        "wm_ratio_min": safe_array_min(wm_ratio),
        "wm_ratio_max": safe_array_max(wm_ratio),
        "electrode_delta_scale": float(electrode_delta_scale),
        "stim_recompute_error_mm": basis["stim_recompute_error_mm"],
        "orientation_source": basis["orientation_source"],
        "active_mesh_path": str(active_mesh_path),
        "active_mesh_path_source": active_mesh_path_source,
        "active_mesh_path_from_csv": active_mesh_path_from_csv,
    }

    return out, qc


def build_magventure_kernel_efield_for_cohort(
    step1: pd.DataFrame,
    metadata: pd.DataFrame,
    cohort: str,
    headmodels_root: str | Path,
    kernel_points_path: str | Path,
    roi_radius_mm: float,
    target_distance_below_iz_mm: float,
    y_sign: float = 1.0,
    max_ratio: float = 1.0,
    limit_subjects: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build empirical-kernel MagVenture sham rows for one cohort.
    """
    cohort = cohort.upper()

    meta = metadata[
        (metadata["cohort"].astype(str).str.upper() == cohort)
        & (metadata["include_primary"] == 1)
    ].copy()

    subject_ids = meta["subject_id"].astype(str).tolist()

    if limit_subjects is not None:
        subject_ids = subject_ids[: int(limit_subjects)]

    linear, nearest, _ = load_kernel_interpolators(
        kernel_points_path,
        ratio_col=RATIO_COLUMN,
    )

    frames = []
    qc_rows = []

    for subject_id in subject_ids:
        print(f"[MAGVENTURE KERNEL] {cohort} {subject_id}", flush=True)

        sub = step1[step1["subject_id"].astype(str) == subject_id].copy()

        if sub.empty:
            raise ValueError(f"{cohort} {subject_id}: no Step 1 rows found")

        m2m_dir = find_subject_m2m_dir(headmodels_root, subject_id)

        rows, qc = make_magventure_kernel_rows_for_subject(
            step1_subject_rows=sub,
            m2m_dir=m2m_dir,
            kernel_linear=linear,
            kernel_nearest=nearest,
            roi_radius_mm=roi_radius_mm,
            target_distance_below_iz_mm=target_distance_below_iz_mm,
            y_sign=y_sign,
            max_ratio=max_ratio,
        )

        frames.append(rows)
        qc["cohort"] = cohort
        qc_rows.append(qc)

    out = pd.concat(frames, ignore_index=True)
    qc = pd.DataFrame(qc_rows)

    return out, qc


def print_magventure_kernel_cohort_summary(rows: pd.DataFrame, qc: pd.DataFrame, cohort: str) -> None:
    """
    Print concise cohort QC summary.
    """
    print("\n[MAGVENTURE KERNEL SUBJECT SUMMARY]")
    print(f"Cohort: {cohort}")
    print(f"Subjects: {rows['subject_id'].nunique()}")
    print(f"Rows: {len(rows)}")

    print("\nRows by condition:")
    print(rows["condition"].value_counts().sort_index().to_string())

    print("\nGM local kernel ratio summary:")
    cols = [
        "gm_ratio_mean",
        "gm_ratio_median",
        "gm_ratio_p95",
        "gm_ratio_min",
        "gm_ratio_max",
        "electrode_delta_scale",
        "stim_recompute_error_mm",
    ]
    print(
        qc[cols]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        .round(6)
        .to_string()
    )
