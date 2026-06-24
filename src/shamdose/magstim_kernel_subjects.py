from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import simnibs
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from shamdose.anatomy import parse_xyz_text, find_subject_m2m_dir
from shamdose.step1_simulation import (
    find_cap_with_labels,
    load_eeg_cap_csv,
    _build_scalp_kdtree,
    _project_to_scalp_nodes,
    _radial_outward_normal,
    geodesic_iz_minus_mm_mesh,
    compute_stim_center_and_ydir_flipped,
    GeodesicStuck,
)


KERNEL_CONDITION = "Magstim_Sham0mm_kernel"
SOURCE_ACTIVE_CONDITION = "Magstim_Active0mm"
SOURCE_SCALAR_CONDITION = "Magstim_Sham0mm"

INTENSITIES = list(range(0, 101, 10))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """
    Weighted quantile for q in [0, 1].
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]

    if values.size == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    return float(np.interp(q, cdf, values))


def safe_array_min(values: np.ndarray) -> float:
    """
    Safe minimum that returns NaN for empty or non-finite arrays.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return float(np.min(values))


def safe_array_max(values: np.ndarray) -> float:
    """
    Safe maximum that returns NaN for empty or non-finite arrays.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return float(np.max(values))


def safe_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted mean that safely returns NaN if there are no positive-volume elements.

    This is useful for WM summaries in small cerebellar ROIs, where a subject may
    have no valid WM elements inside the 10-mm target ROI.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if not mask.any():
        return np.nan

    values = values[mask]
    weights = weights[mask]

    if weights.sum() <= 0:
        return np.nan

    return float(np.average(values, weights=weights))


def distribution_stats(values: np.ndarray, volumes: np.ndarray) -> dict[str, float]:
    """
    Volume-weighted ROI distribution statistics.
    """
    values = np.asarray(values, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    mask = np.isfinite(values) & np.isfinite(volumes) & (volumes > 0)
    values = values[mask]
    volumes = volumes[mask]

    if values.size == 0:
        return {
            "n_elements": 0,
            "volume_mm3": 0.0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }

    total_volume = float(volumes.sum())
    mean = float(np.average(values, weights=volumes))
    var = float(np.average((values - mean) ** 2, weights=volumes))

    return {
        "n_elements": int(values.size),
        "volume_mm3": total_volume,
        "mean": mean,
        "median": weighted_quantile(values, volumes, 0.50),
        "std": float(np.sqrt(var)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p95": weighted_quantile(values, volumes, 0.95),
        "p99": weighted_quantile(values, volumes, 0.99),
    }


def fill_stat_columns(row: pd.Series, prefix: str, stats: dict[str, float]) -> pd.Series:
    """
    Fill Step 1-style ROI statistic columns for one tissue prefix.

    This is deliberately flexible because the exact Step 1 column names come
    from the old extraction script.
    """
    for col in row.index:
        if not str(col).startswith(prefix):
            continue

        lc = str(col).lower()
        row[col] = np.nan

        if "n_element" in lc or lc.endswith("_n"):
            row[col] = stats["n_elements"]
        elif "volume" in lc or "vol" in lc:
            row[col] = stats["volume_mm3"]
        elif "mean" in lc:
            row[col] = stats["mean"]
        elif "median" in lc or "p50" in lc:
            row[col] = stats["median"]
        elif "std" in lc or "sd" in lc:
            row[col] = stats["std"]
        elif "p99" in lc:
            row[col] = stats["p99"]
        elif "p95" in lc:
            row[col] = stats["p95"]
        elif "min" in lc:
            row[col] = stats["min"]
        elif "max" in lc:
            row[col] = stats["max"]

    return row


def get_element_field_values(mesh) -> np.ndarray:
    """
    Extract scalar E-field magnitude from a SimNIBS scalar mesh.

    Prefers element data named magnE. Falls back to the first element-data array
    with one scalar per element.
    """
    n_elements = len(mesh.elm.tag1)

    candidates = []

    for ed in mesh.elmdata:
        name = str(getattr(ed, "field_name", ""))
        value = np.asarray(getattr(ed, "value", None))

        if value is None:
            continue

        value = np.asarray(value)

        if value.shape[0] != n_elements:
            continue

        if value.ndim == 2 and value.shape[1] == 3:
            scalar = np.linalg.norm(value, axis=1)
        else:
            scalar = value.reshape(value.shape[0], -1)[:, 0]

        candidates.append((name, scalar.astype(float)))

    for name, values in candidates:
        if name.lower() in {"magne", "magn_e", "norme", "e_magn", "e_magnitude"}:
            return values

    if candidates:
        return candidates[0][1]

    names = [str(getattr(ed, "field_name", "")) for ed in mesh.elmdata]
    raise RuntimeError(f"Could not find element E-field data. Available elmdata names: {names}")


def tetra_volumes(coords: np.ndarray) -> np.ndarray:
    """
    Compute tetrahedral volumes.

    coords shape:
        n_elements × 4 × 3
    """
    a = coords[:, 0, :]
    b = coords[:, 1, :]
    c = coords[:, 2, :]
    d = coords[:, 3, :]

    return np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6.0


def extract_roi_elements_from_mesh(
    mesh,
    roi_center_xyz: np.ndarray,
    roi_radius_mm: float,
    tissue_tag: int,
) -> dict[str, np.ndarray]:
    """
    Extract tetrahedral ROI elements for a tissue tag.

    Returns:
        centroids
        volumes
        active_E
    """
    nodes = np.asarray(mesh.nodes.node_coord, dtype=float)
    node_list = np.asarray(mesh.elm.node_number_list)
    tags = np.asarray(mesh.elm.tag1)
    elm_type = np.asarray(mesh.elm.elm_type)

    e_values = get_element_field_values(mesh)

    # SimNIBS tetrahedra are usually type 4. Include type 11 as a cautious fallback.
    tetra_mask = np.isin(elm_type, [4, 11]) & (tags == tissue_tag)

    if not tetra_mask.any():
        raise RuntimeError(f"No tetrahedral elements found for tissue tag {tissue_tag}")

    elem_indices = np.where(tetra_mask)[0]
    corner_nodes = node_list[elem_indices, :4].astype(int) - 1

    coords = nodes[corner_nodes]
    centroids = coords.mean(axis=1)
    volumes = tetra_volumes(coords)
    active_E = e_values[elem_indices]

    dist = np.linalg.norm(centroids - roi_center_xyz.reshape(1, 3), axis=1)
    roi_mask = dist <= float(roi_radius_mm)

    return {
        "element_indices": elem_indices[roi_mask],
        "centroids": centroids[roi_mask],
        "volumes": volumes[roi_mask],
        "active_E": active_E[roi_mask],
    }


def load_kernel_interpolators(
    kernel_points_path: str | Path,
    ratio_col: str = "magstim_sham_over_active_regularized",
):
    """
    Build linear and nearest-neighbor interpolators for the empirical kernel.
    """
    kernel_points_path = Path(kernel_points_path).expanduser().resolve()

    if not kernel_points_path.exists():
        raise FileNotFoundError(f"Kernel points table not found: {kernel_points_path}")

    df = pd.read_csv(kernel_points_path)

    required = ["x_cm", "y_cm", ratio_col]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Kernel points table missing columns: {missing}")

    xy = df[["x_cm", "y_cm"]].to_numpy(dtype=float)
    values = pd.to_numeric(df[ratio_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(xy).all(axis=1) & np.isfinite(values)

    xy = xy[mask]
    values = values[mask]

    linear = LinearNDInterpolator(xy, values, fill_value=np.nan)
    nearest = NearestNDInterpolator(xy, values)

    return linear, nearest, df


def interpolate_kernel_ratio(
    xy_cm: np.ndarray,
    linear,
    nearest,
    min_ratio: float = 0.0,
    max_ratio: float = 1.0,
) -> np.ndarray:
    """
    Interpolate empirical local sham/active ratio at local x/y coordinates.

    Linear interpolation is used inside the convex hull; nearest-neighbor fills
    any outside-hull points.
    """
    xy_cm = np.asarray(xy_cm, dtype=float)

    ratios = np.asarray(linear(xy_cm), dtype=float)
    missing = ~np.isfinite(ratios)

    if missing.any():
        ratios[missing] = np.asarray(nearest(xy_cm[missing]), dtype=float)

    ratios = np.clip(ratios, min_ratio, max_ratio)

    return ratios


def compute_coil_local_basis(
    mesh,
    m2m_dir: str | Path,
    stim_center_xyz: np.ndarray,
    target_distance_below_iz_mm: float,
    y_sign: float = 1.0,
) -> dict[str, np.ndarray | float | str]:
    """
    Reconstruct the coil-local basis used in the original Step 1 targeting.

    Local axes:
        x_axis = across figure-8 loops
        y_axis = coil y-dir / handle direction
        z_axis = outward scalp normal

    The empirical Smith-Peterchev grid uses x/y coordinates where x=0 separates
    the two loops. Because the Magstim sham distribution is asymmetric along y,
    y_sign is exposed as a sensitivity parameter.
    """
    head_center_xyz = np.asarray(mesh.nodes.node_coord, dtype=float).mean(axis=0)
    scalp_nodes, scalp_tree = _build_scalp_kdtree(mesh)

    cap_csv = find_cap_with_labels(m2m_dir, required=["Iz", "Oz", "Cz"])
    cap_map = load_eeg_cap_csv(cap_csv)

    try:
        _, oz_surf, stim_center_recomputed = geodesic_iz_minus_mm_mesh(
            cap_map,
            scalp_nodes,
            scalp_tree,
            head_center_xyz,
            dist_mm=float(target_distance_below_iz_mm),
            step_mm=1.0,
        )
        orientation_source = "geodesic_recomputed"
    except GeodesicStuck:
        stim_center_raw, _ = compute_stim_center_and_ydir_flipped(
            cap_map,
            float(target_distance_below_iz_mm),
        )
        stim_center_recomputed = _project_to_scalp_nodes(stim_center_raw, scalp_nodes, scalp_tree)
        oz_surf = _project_to_scalp_nodes(cap_map["Oz"], scalp_nodes, scalp_tree)
        orientation_source = "fallback_projected"

    # Use the actual Step 1 saved target as origin for coordinates.
    origin = np.asarray(stim_center_xyz, dtype=float)

    z_axis = _radial_outward_normal(origin, head_center_xyz)

    ydir_flipped = 2.0 * stim_center_recomputed - oz_surf
    y_vec = ydir_flipped - stim_center_recomputed

    # Project y onto local tangent plane.
    y_vec = y_vec - np.dot(y_vec, z_axis) * z_axis

    if np.linalg.norm(y_vec) == 0:
        raise RuntimeError("Degenerate coil y-axis after projection.")

    y_axis = y_vec / np.linalg.norm(y_vec)
    y_axis = float(y_sign) * y_axis

    x_axis = np.cross(y_axis, z_axis)

    if np.linalg.norm(x_axis) == 0:
        raise RuntimeError("Degenerate coil x-axis.")

    x_axis = x_axis / np.linalg.norm(x_axis)

    # Re-orthogonalize y.
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    stim_recompute_error_mm = float(np.linalg.norm(stim_center_recomputed - origin))

    return {
        "origin": origin,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "stim_recompute_error_mm": stim_recompute_error_mm,
        "orientation_source": orientation_source,
        "cap_csv": str(cap_csv),
    }


def centroids_to_coil_xy_cm(
    centroids_xyz: np.ndarray,
    basis: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Convert subject-space element centroids into coil-local x/y coordinates in cm.
    """
    rel = np.asarray(centroids_xyz, dtype=float) - basis["origin"].reshape(1, 3)

    x_mm = rel @ basis["x_axis"]
    y_mm = rel @ basis["y_axis"]

    return np.column_stack([x_mm / 10.0, y_mm / 10.0])


def make_kernel_rows_for_subject(
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
    Create corrected Magstim sham kernel rows for one subject.
    """
    subject_id = str(step1_subject_rows["subject_id"].iloc[0])

    active_rows = step1_subject_rows[
        step1_subject_rows["condition"].astype(str) == SOURCE_ACTIVE_CONDITION
    ].copy()

    if active_rows.empty:
        raise ValueError(f"{subject_id}: missing {SOURCE_ACTIVE_CONDITION} rows")

    template_rows = active_rows.sort_values("intensity (%MT)").copy()

    # Use 100% active row to locate mesh and coordinates.
    active_100 = active_rows[
        pd.to_numeric(active_rows["intensity (%MT)"], errors="coerce") == 100
    ]

    if len(active_100) != 1:
        raise ValueError(f"{subject_id}: expected one active 100% row, found {len(active_100)}")

    active_100 = active_100.iloc[0]

    # The Step 1 CSV contains an absolute path written at extraction time.
    # For mesh-based re-extraction, prefer the current YAML-derived subject
    # directory, because cohort roots may have been cleaned/reorganized later.
    active_mesh_path_from_csv = Path(str(active_100["path"])).expanduser()
    active_mesh_path_from_csv_resolved = active_mesh_path_from_csv.resolve()

    subject_dir = Path(m2m_dir).expanduser().resolve().parent

    candidates = sorted(
        subject_dir.glob("*/TMS_Magstim_Active_0mm/*_scalar.msh")
    )

    if not candidates:
        candidates = sorted(
            subject_dir.rglob("TMS_Magstim_Active_0mm/*_scalar.msh")
        )

    if candidates:
        active_mesh_path = max(candidates, key=lambda p: p.stat().st_mtime).resolve()
        active_mesh_path_source = "searched_current_yaml_subject_dir"
    elif active_mesh_path_from_csv_resolved.exists():
        active_mesh_path = active_mesh_path_from_csv_resolved
        active_mesh_path_source = "step1_path_column_fallback"
    else:
        raise FileNotFoundError(
            f"{subject_id}: active Magstim scalar mesh not found.\\n"
            f"  Step 1 CSV path: {active_mesh_path_from_csv_resolved}\\n"
            f"  YAML/current subject_dir: {subject_dir}\\n"
            "  expected pattern: */TMS_Magstim_Active_0mm/*_scalar.msh"
        )

    if active_mesh_path != active_mesh_path_from_csv_resolved:
        print(
            f"[INFO] {subject_id}: Step 1 mesh path points elsewhere; "
            f"using YAML-root mesh: {active_mesh_path}",
            flush=True,
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

    rows = []

    for intensity in INTENSITIES:
        frac = float(intensity) / 100.0

        # Use the corresponding active row as template if possible.
        tr = template_rows[
            pd.to_numeric(template_rows["intensity (%MT)"], errors="coerce") == intensity
        ]

        if len(tr) != 1:
            row = active_100.copy()
            row["intensity (%MT)"] = intensity
        else:
            row = tr.iloc[0].copy()

        row["condition"] = KERNEL_CONDITION
        row["file_name"] = active_mesh_path.name
        row["path"] = str(active_mesh_path)
        row["intensity (%MT)"] = intensity

        if "scale_to_active100" in row.index:
            row["scale_to_active100"] = np.nan

        gm_stats = distribution_stats(gm_kernel_E100 * frac, gm["volumes"])
        wm_stats = distribution_stats(wm_kernel_E100 * frac, wm["volumes"])

        row = fill_stat_columns(row, "roi_graymatter", gm_stats)
        row = fill_stat_columns(row, "roi_whitematter", wm_stats)

        row["magstim_kernel_method"] = "smith_peterchev_xy_local_ratio"
        row["magstim_kernel_condition_source"] = SOURCE_ACTIVE_CONDITION
        row["magstim_kernel_y_sign"] = float(y_sign)
        row["magstim_kernel_max_ratio"] = float(max_ratio)
        row["magstim_kernel_gm_ratio_mean"] = safe_weighted_mean(gm_ratio, gm["volumes"])
        row["magstim_kernel_gm_ratio_p95"] = weighted_quantile(gm_ratio, gm["volumes"], 0.95)
        row["magstim_kernel_wm_ratio_mean"] = safe_weighted_mean(wm_ratio, wm["volumes"])
        row["magstim_kernel_wm_ratio_p95"] = weighted_quantile(wm_ratio, wm["volumes"], 0.95)
        row["magstim_kernel_stim_recompute_error_mm"] = basis["stim_recompute_error_mm"]
        row["magstim_kernel_orientation_source"] = basis["orientation_source"]
        row["magstim_kernel_cap_csv"] = basis["cap_csv"]
        row["magstim_kernel_active_mesh_source"] = active_mesh_path_source
        row["magstim_kernel_active_mesh_path_from_csv"] = str(active_mesh_path_from_csv)
        row["magstim_kernel_active_mesh_path_used"] = str(active_mesh_path)

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
        "stim_recompute_error_mm": basis["stim_recompute_error_mm"],
        "orientation_source": basis["orientation_source"],
        "active_mesh_path": str(active_mesh_path),
    }

    return out, qc


def build_magstim_kernel_efield_for_cohort(
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
    Build corrected Magstim sham kernel rows for all included subjects in one cohort.
    """
    cohort = cohort.upper()

    meta = metadata[
        (metadata["cohort"].astype(str).str.upper() == cohort)
        & (metadata["include_primary"] == 1)
    ].copy()

    subject_ids = meta["subject_id"].astype(str).tolist()

    if limit_subjects is not None:
        subject_ids = subject_ids[: int(limit_subjects)]

    linear, nearest, _ = load_kernel_interpolators(kernel_points_path)

    frames = []
    qc_rows = []

    for subject_id in subject_ids:
        print(f"[MAGSTIM KERNEL] {cohort} {subject_id}", flush=True)

        sub = step1[step1["subject_id"].astype(str) == subject_id].copy()

        if sub.empty:
            raise ValueError(f"{cohort} {subject_id}: no Step 1 rows found")

        m2m_dir = find_subject_m2m_dir(headmodels_root, subject_id)

        rows, qc = make_kernel_rows_for_subject(
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


def print_magstim_kernel_cohort_summary(rows: pd.DataFrame, qc: pd.DataFrame, cohort: str) -> None:
    """
    Print concise cohort QC summary.
    """
    print("\n[MAGSTIM KERNEL SUBJECT SUMMARY]")
    print(f"Cohort: {cohort}")
    print(f"Subjects: {rows['subject_id'].nunique()}")
    print(f"Rows: {len(rows)}")

    print("\nRows by condition:")
    print(rows["condition"].value_counts().to_string())

    print("\nGM local kernel ratio summary:")
    cols = [
        "gm_ratio_mean",
        "gm_ratio_median",
        "gm_ratio_p95",
        "gm_ratio_min",
        "gm_ratio_max",
        "stim_recompute_error_mm",
    ]
    print(
        qc[cols]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        .round(6)
        .to_string()
    )
