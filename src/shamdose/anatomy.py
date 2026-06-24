from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import simnibs


# CHARM / SimNIBS tissue labels
LABEL_WM = 1
LABEL_GM = 2
LABEL_CSF = 3
LABEL_BONE1 = 4
LABEL_SCALP = 5
LABEL_BONE2 = 7
LABEL_BONE3 = 8

STEP_MM = 0.25

SCALP_LABELS = {LABEL_SCALP}
SKULL_LABELS = {LABEL_BONE1, LABEL_BONE2, LABEL_BONE3}
CSF_LABELS = {LABEL_CSF}
GM_LABELS = {LABEL_GM}
WM_LABELS = {LABEL_WM}
INHEAD_LABELS = SCALP_LABELS | SKULL_LABELS | CSF_LABELS | GM_LABELS | WM_LABELS


def parse_xyz_text(x: object) -> np.ndarray:
    """
    Parse xyz text stored in Step 1 CSV.

    Expected examples:
        "6.65,-78.78,-49.01"
        "[6.65 -78.78 -49.01]"
    """
    txt = str(x).strip()

    if not txt or txt.lower() in {"nan", "none"}:
        raise ValueError(f"Empty xyz text: {x}")

    for tok in ("np.float64(", "np.float32(", "array("):
        txt = txt.replace(tok, "")

    txt = txt.replace(")", "")
    txt = txt.strip().strip('"').strip("'")
    txt = txt.strip("[]()")

    parts = [p for p in re.split(r"[\s,]+", txt) if p]

    if len(parts) != 3:
        raise ValueError(f"Could not parse xyz triplet from: {x}")

    return np.array([float(p) for p in parts], dtype=float)


def find_subject_m2m_dir(headmodels_root: str | Path, subject_id: str) -> Path:
    """
    Find the m2m directory for one subject.

    Expected:
        headmodels_root/sub-XXX/m2m_*
    """
    subject_dir = Path(headmodels_root).expanduser().resolve() / subject_id
    candidates = sorted([p for p in subject_dir.glob("m2m_*") if p.is_dir()])

    if not candidates:
        raise FileNotFoundError(f"No m2m_* folder found for {subject_id} under {subject_dir}")

    if len(candidates) > 1:
        raise RuntimeError(f"Multiple m2m_* folders found for {subject_id}: {candidates}")

    return candidates[0]


def find_head_mesh(m2m_dir: str | Path) -> Path:
    """
    Find the main SimNIBS head mesh inside an m2m directory.
    """
    m2m_dir = Path(m2m_dir).expanduser().resolve()
    subject_id = m2m_dir.name.replace("m2m_", "")

    preferred = m2m_dir / f"{subject_id}.msh"
    if preferred.exists():
        return preferred

    candidates = sorted(m2m_dir.glob("*.msh"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No .msh head mesh found in {m2m_dir}")


def pick_label_vol(m2m_dir: str | Path) -> Path:
    """
    Find CHARM tissue-label volume.
    """
    m2m_dir = Path(m2m_dir).expanduser().resolve()

    candidates = [
        m2m_dir / "label_prep" / "tissue_labeling_upsampled.nii.gz",
        m2m_dir / "tissue_labeling_upsampled.nii.gz",
        m2m_dir / "tissue_labeling.nii.gz",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"Could not find tissue labeling volume in {m2m_dir}")


def load_nifti_labels(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load tissue-label NIfTI.

    Returns:
        labels, affine, inverse affine
    """
    import nibabel as nib

    img = nib.load(str(path))
    labels = img.get_fdata().astype(np.int16)
    affine = img.affine
    affine_inv = np.linalg.inv(affine)

    return labels, affine, affine_inv


def world_to_vox(affine_inv: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return (affine_inv @ np.array([xyz[0], xyz[1], xyz[2], 1.0]))[:3]


def vox_to_world(affine: np.ndarray, ijk: tuple[int, int, int] | np.ndarray) -> np.ndarray:
    return (affine @ np.array([ijk[0], ijk[1], ijk[2], 1.0]))[:3]


def sample_label(labels: np.ndarray, affine_inv: np.ndarray, xyz: np.ndarray) -> int:
    """
    Sample nearest-neighbor tissue label at world coordinate xyz.
    """
    ijk = world_to_vox(affine_inv, xyz)
    i, j, k = [int(round(v)) for v in ijk]

    nx, ny, nz = labels.shape

    if i < 0 or j < 0 or k < 0 or i >= nx or j >= ny or k >= nz:
        return 0

    return int(labels[i, j, k])


def radial_outward_normal(point_xyz: np.ndarray, head_center_xyz: np.ndarray) -> np.ndarray:
    """
    Approximate outward normal using vector from head center to target point.
    """
    v = point_xyz - head_center_xyz
    norm = float(np.linalg.norm(v))

    if norm == 0:
        raise RuntimeError("Degenerate radial normal: target equals head center.")

    return v / norm


def snap_point_to_nearest_label(
    labels: np.ndarray,
    affine: np.ndarray,
    affine_inv: np.ndarray,
    xyz_world: np.ndarray,
    target_label: int,
    search_mm: float = 8.0,
) -> np.ndarray:
    """
    Snap a world coordinate to the nearest voxel with target_label.

    Used when the Step 1 scalp target lies on the scalp mesh but does not sample
    exactly as SCALP in the voxel label volume.
    """
    vx = float(np.linalg.norm(affine[:3, 0]))
    vy = float(np.linalg.norm(affine[:3, 1]))
    vz = float(np.linalg.norm(affine[:3, 2]))

    rad_i = max(1, int(math.ceil(search_mm / vx)))
    rad_j = max(1, int(math.ceil(search_mm / vy)))
    rad_k = max(1, int(math.ceil(search_mm / vz)))

    i0, j0, k0 = [int(round(v)) for v in world_to_vox(affine_inv, xyz_world)]
    nx, ny, nz = labels.shape

    best_d2 = None
    best_xyz = None

    for i in range(i0 - rad_i, i0 + rad_i + 1):
        if i < 0 or i >= nx:
            continue
        for j in range(j0 - rad_j, j0 + rad_j + 1):
            if j < 0 or j >= ny:
                continue
            for k in range(k0 - rad_k, k0 + rad_k + 1):
                if k < 0 or k >= nz:
                    continue

                if int(labels[i, j, k]) != int(target_label):
                    continue

                xyz = vox_to_world(affine, (i, j, k))
                d2 = float(np.sum((xyz - xyz_world) ** 2))

                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_xyz = xyz

    if best_xyz is None:
        return xyz_world

    return best_xyz


def thickness_along_ray(
    labels: np.ndarray,
    affine: np.ndarray,
    affine_inv: np.ndarray,
    scalp_point_xyz: np.ndarray,
    outward_normal: np.ndarray,
    roi_center_xyz: np.ndarray,
    step_mm: float = STEP_MM,
    max_len_mm: float = 160.0,
) -> dict[str, float]:
    """
    Sample tissue labels along the inward ray beneath the target.

    The ray starts outside the scalp target and marches inward until GM is reached.

    Returns thicknesses from entry into head to first GM:
        scalp_mm
        skull_mm
        csf_mm
        total_to_first_gm_mm

    Also estimates GM thickness along the same ray.
    """
    n_out = outward_normal / (np.linalg.norm(outward_normal) + 1e-12)
    inward = -n_out

    start = scalp_point_xyz + 30.0 * n_out
    n_steps = int(max_len_mm / step_mm)

    ray_labels: list[int] = []
    ray_points: list[np.ndarray] = []

    for t in range(n_steps + 1):
        point = start + (t * step_mm) * inward
        ray_points.append(point)
        ray_labels.append(sample_label(labels, affine_inv, point))

    entry_idx = next((i for i, lab in enumerate(ray_labels) if lab in INHEAD_LABELS), None)
    if entry_idx is None:
        raise RuntimeError("Ray never entered head tissues.")

    gm_idx = next(
        (i for i in range(entry_idx, len(ray_labels)) if ray_labels[i] in GM_LABELS),
        None,
    )
    if gm_idx is None:
        raise RuntimeError("Ray never reached GM.")

    segment_to_gm = ray_labels[entry_idx : gm_idx + 1]
    total_to_first_gm_mm = step_mm * max(0, len(segment_to_gm) - 1)

    scalp_mm = step_mm * sum(1 for lab in segment_to_gm if lab in SCALP_LABELS)
    skull_mm = step_mm * sum(1 for lab in segment_to_gm if lab in SKULL_LABELS)
    csf_mm = step_mm * sum(1 for lab in segment_to_gm if lab in CSF_LABELS)
    other_mm = total_to_first_gm_mm - (scalp_mm + skull_mm + csf_mm)

    dist_to_roi_center_mm = float(np.dot(roi_center_xyz - scalp_point_xyz, inward))
    if dist_to_roi_center_mm < 0:
        dist_to_roi_center_mm = float("nan")

    return {
        "total_to_first_gm_mm": float(total_to_first_gm_mm),
        "scalp_mm": float(scalp_mm),
        "skull_mm": float(skull_mm),
        "csf_mm": float(csf_mm),
        "other_mm": float(other_mm),
        "dist_to_roi_center_mm": float(dist_to_roi_center_mm),
    }


def extract_subject_target_anatomy(
    subject_id: str,
    cohort: str,
    age: float | int | None,
    sex: float | int | None,
    headmodels_root: str | Path,
    step1_subject_rows: pd.DataFrame,
) -> dict[str, object]:
    """
    Extract local target anatomy for one subject.

    The target and ROI centers are read from Step 1 outputs.
    They are not recomputed.
    """
    if step1_subject_rows.empty:
        raise ValueError(f"No Step 1 rows provided for {subject_id}")

    first = step1_subject_rows.iloc[0]

    stim_center = parse_xyz_text(first["stim_center_subject_xyz_mm"])
    roi_center = parse_xyz_text(first["roi_center_gm_subject_xyz_mm"])

    m2m_dir = find_subject_m2m_dir(headmodels_root, subject_id)

    label_vol_from_step1 = str(first.get("label_vol", "")).strip()
    if label_vol_from_step1 and label_vol_from_step1.lower() not in {"nan", "none"}:
        label_vol = Path(label_vol_from_step1)
        if not label_vol.exists():
            label_vol = pick_label_vol(m2m_dir)
    else:
        label_vol = pick_label_vol(m2m_dir)

    head_mesh_path = find_head_mesh(m2m_dir)
    head_mesh = simnibs.read_msh(str(head_mesh_path))
    head_center = np.asarray(head_mesh.nodes.node_coord, dtype=float).mean(axis=0)

    labels, affine, affine_inv = load_nifti_labels(label_vol)

    outward_normal = radial_outward_normal(stim_center, head_center)
    inward = -outward_normal / (np.linalg.norm(outward_normal) + 1e-12)

    scalp_point = stim_center.copy()
    scalp_point_initial = scalp_point.copy()
    scalp_label_at_target = sample_label(labels, affine_inv, scalp_point)

    if scalp_label_at_target != LABEL_SCALP:
        scalp_point = snap_point_to_nearest_label(
            labels,
            affine,
            affine_inv,
            scalp_point,
            LABEL_SCALP,
            search_mm=8.0,
        )

    scalp_snap_distance_mm = float(np.linalg.norm(scalp_point - scalp_point_initial))

    thickness = thickness_along_ray(
        labels=labels,
        affine=affine,
        affine_inv=affine_inv,
        scalp_point_xyz=scalp_point,
        outward_normal=outward_normal,
        roi_center_xyz=roi_center,
        step_mm=STEP_MM,
    )

    gm_depth_from_scalp_mm = float(np.dot(roi_center - scalp_point, inward))

    if np.isfinite(thickness["dist_to_roi_center_mm"]):
        gm_depth_minus_dist_to_roi_mm = (
            gm_depth_from_scalp_mm - thickness["dist_to_roi_center_mm"]
        )
    else:
        gm_depth_minus_dist_to_roi_mm = float("nan")

    depth_consistency_error_mm = (
        gm_depth_from_scalp_mm - float(thickness["total_to_first_gm_mm"])
    )

    return {
        "subject_id": subject_id,
        "cohort": cohort,
        "age": age,
        "sex": sex,
        "status": "OK",
        "error": "",
        "m2m_dir": str(m2m_dir),
        "head_mesh": str(head_mesh_path),
        "label_vol": str(label_vol),
        "coord_source": "step1_efield_table",
        "stim_center_subject_xyz_mm": f"{stim_center[0]:.2f},{stim_center[1]:.2f},{stim_center[2]:.2f}",
        "roi_center_gm_subject_xyz_mm": f"{roi_center[0]:.2f},{roi_center[1]:.2f},{roi_center[2]:.2f}",
        "stim_x": float(stim_center[0]),
        "stim_y": float(stim_center[1]),
        "stim_z": float(stim_center[2]),
        "scalp_x": float(scalp_point[0]),
        "scalp_y": float(scalp_point[1]),
        "scalp_z": float(scalp_point[2]),
        "scalp_snap_distance_mm": float(scalp_snap_distance_mm),
        "roi_x": float(roi_center[0]),
        "roi_y": float(roi_center[1]),
        "roi_z": float(roi_center[2]),
        "normal_x": float(outward_normal[0]),
        "normal_y": float(outward_normal[1]),
        "normal_z": float(outward_normal[2]),
        "gm_depth_from_scalp_mm": float(gm_depth_from_scalp_mm),
        "gm_depth_minus_dist_to_roi_mm": float(gm_depth_minus_dist_to_roi_mm),
        "depth_consistency_error_mm": float(depth_consistency_error_mm),
        **thickness,
    }


def extract_target_anatomy_for_cohort(
    metadata: pd.DataFrame,
    cohort: str,
    headmodels_root: str | Path,
    step1_table_path: str | Path,
    allow_errors: bool = False,
) -> pd.DataFrame:
    """
    Extract target anatomy for all included subjects in one cohort.
    """
    cohort = cohort.upper()

    step1_table_path = Path(step1_table_path).expanduser().resolve()
    if not step1_table_path.exists():
        raise FileNotFoundError(f"Step 1 table not found: {step1_table_path}")

    step1 = pd.read_csv(step1_table_path)

    sub_meta = metadata[
        (metadata["cohort"] == cohort)
        & (metadata["include_primary"] == 1)
    ].copy()

    if sub_meta.empty:
        raise ValueError(f"No included metadata rows found for cohort {cohort}")

    rows: list[dict[str, object]] = []

    for _, meta_row in sub_meta.iterrows():
        subject_id = str(meta_row["subject_id"])
        subject_rows = step1[step1["subject_id"].astype(str) == subject_id].copy()

        try:
            row = extract_subject_target_anatomy(
                subject_id=subject_id,
                cohort=cohort,
                age=meta_row["age"],
                sex=meta_row["sex"],
                headmodels_root=headmodels_root,
                step1_subject_rows=subject_rows,
            )
        except Exception as exc:
            row = {
                "subject_id": subject_id,
                "cohort": cohort,
                "age": meta_row["age"],
                "sex": meta_row["sex"],
                "status": "ERROR",
                "error": repr(exc),
            }

            if not allow_errors:
                raise RuntimeError(f"Anatomy extraction failed for {subject_id}: {exc}") from exc

        rows.append(row)

    return pd.DataFrame(rows)


def print_target_anatomy_summary(df: pd.DataFrame, cohort: str) -> None:
    """
    Print concise QC summary.
    """
    print("\n[TARGET ANATOMY SUMMARY]")
    print(f"Cohort: {cohort}")
    print(f"Rows: {len(df)}")
    print("Status counts:")
    print(df["status"].value_counts().to_string())

    ok = df[df["status"] == "OK"].copy()

    if ok.empty:
        return

    cols = [
        "gm_depth_from_scalp_mm",
        "total_to_first_gm_mm",
        "depth_consistency_error_mm",
        "scalp_snap_distance_mm",
        "scalp_mm",
        "skull_mm",
        "csf_mm",
    ]

    print("\nAnatomy metrics, OK subjects:")
    print(
        ok[cols]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        .round(3)
        .to_string()
    )
