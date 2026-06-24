#!/usr/bin/env python3
"""
CEREBELLUM STEP 1 (UPDATED)

- Stimulation center: 2 cm below inion (Iz) along Iz–Oz midline direction GEODESIC!
- Coil handle direction: flipped 180° (opposite of previous) by reflecting ydir point around centre
- tDCS montage kept exactly as you specified (vermis–chin, 3x3 cm, 2 mA, ydir=Cz, cathode = chin/cheek via MNI->subject)
- ROI:
  - center = first GM point along the inward local scalp normal under the geodesic scalp target
  - radius = 10 mm (1 cm)
- CSV: peak only (no separate max), ROI volume kept as diagnostic

Outputs (per subject)
- sub-XXXX/simnibs_run_2cmBelow_Iz_radius1_geodesic/STEP1_E0_inion2cmBelow_roi10mm_allstats_scaled.csv
"""

import os
import glob
import csv
import math
import time
import numpy as np
import simnibs

from typing import Optional

class GeodesicStuck(RuntimeError):
    """Raised when the scalp geodesic walk stalls (projection does not move)."""
    pass

try:
    import nibabel as nib
except Exception as e:
    nib = None

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

try:
    from simnibs import sim_struct, run_simnibs, mni2subject_coords
except Exception:
    from simnibs import sim_struct, mni2subject_coords
    from simnibs.simulation import run_simnibs


# ============================================================
# USER CONFIG
# ============================================================
HEADMODELS_ROOT = "/Volumes/m-BegueLab/BegueLab/Myriam/HeadModels_Healthy_Control/simnibs_head_models_HC"
OUT_SUBDIR      = "simnibs_run_2cmBelow_Iz_radius1_geodesic"

RUN_SIMNIBS   = True
FORCE_RERUN   = False
RUN_TMS       = True
RUN_TDCS      = True

# ROI definition
ROI_RADIUS_MM = 10.0   # 1 cm sphere

# "2 cm below Iz" = move 20 mm toward neck along midline direction (Iz away from Oz)
INION_BELOW_MM = 20.0

# Intensity scaling (no reruns per intensity)
INTENSITIES_PCT_MT = list(range(0, 101, 10))
TMS_DISTANCES_MM   = [0.0]

# Coil models
COIL_CCD_DIR = "/Users/lorka/Applications/SimNIBS-4.5/simnibs_env/lib/python3.11/site-packages/simnibs/resources/coil_models/Drakaki_BrainStim_2022"
COILS = [
    dict(
        coil_prefix="MagVenture",
        ccd_file="MagVenture_Cool-B65.ccd",
        stim_didt_max_A_per_us=149.8,
        mt_percent_mso=51.7,
        sham_variants=[
            ("Sham+electrodes", 0.10),
            ("Sham",            0.0772),
        ],
    ),
    dict(
        coil_prefix="Magstim",
        ccd_file="MagStim_D70.ccd",
        stim_didt_max_A_per_us=114.7,
        mt_percent_mso=55.1,
        sham_variants=[
            ("Sham", 0.253),
        ],
    ),
]

# tDCS montage (keep exactly as you specified)
TDCS_MONTAGE_NAME = "VermisChin_3x3cm_2mA"
TDCS_CURRENT_mA   = 2.0
TDCS_PAD_DIMS_MM  = [30, 30]   # 3×3 cm
TDCS_SPONGE_MM    = [60, 60]
TDCS_THICKNESS_MM = [4, 2, 4]  # sponge, rubber/plate, sponge
TDCS_AN_YDIR      = "Cz"
TDCS_CA_YDIR      = "Cz"

# Cathode location (chin/cheek) – keep as MNI->subject mapping
# If your old script used a different MNI point, replace this with your exact one.
CHIN_MNI = [0.0, 70.0, -150.0]

# Extra stats
ROI_PCTS = (25, 50, 75, 90, 95, 99)
ROI_THRESHOLDS_VM = (1.0, 2.0, 5.0, 10.0, 20.0)  # set () to disable


# ============================================================
# Filesystem helpers
# ============================================================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def find_subject_m2m_dirs(root: str) -> list[str]:
    patt = os.path.join(root, "sub-*", "m2m_*")
    return sorted([p for p in glob.glob(patt) if os.path.isdir(p)])

def find_head_mesh(m2m_dir: str) -> str:
    subj_id = os.path.basename(m2m_dir).replace("m2m_", "")
    cand = os.path.join(m2m_dir, f"{subj_id}.msh")
    if os.path.exists(cand):
        return cand
    mshs = sorted(glob.glob(os.path.join(m2m_dir, "*.msh")))
    if mshs:
        return mshs[0]
    raise FileNotFoundError(f"Could not find head mesh in {m2m_dir}")

def latest_scalar_mesh(out_dir: str) -> Optional[str]:
    files = glob.glob(os.path.join(out_dir, "**", "*_scalar.msh"), recursive=True)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]

# ============================================================
# FAST scalp geodesic using head mesh scalp nodes (avoid voxel scans)
# ============================================================
def _build_scalp_kdtree(head_mesh: simnibs.mesh_io.Msh):
    scalp = head_mesh.crop_mesh(simnibs.ElementTags.SCALP)
    scalp_nodes = np.asarray(scalp.nodes.node_coord, dtype=float)
    if scalp_nodes.size == 0:
        raise RuntimeError("SCALP crop produced empty node set")
    tree = cKDTree(scalp_nodes) if cKDTree is not None else None
    return scalp_nodes, tree

def _project_to_scalp_nodes(xyz: np.ndarray, scalp_nodes: np.ndarray, tree=None) -> np.ndarray:
    if tree is not None:
        _, idx = tree.query(xyz, k=1)
        return scalp_nodes[int(idx)]

    # Fallback without SciPy: chunked nearest-neighbor to avoid huge temporaries
    best_idx = 0
    best_d2 = float("inf")
    chunk = 50000
    for start in range(0, scalp_nodes.shape[0], chunk):
        block = scalp_nodes[start:start + chunk]
        diff = block - xyz
        d2 = np.einsum("ij,ij->i", diff, diff)
        j = int(np.argmin(d2))
        v = float(d2[j])
        if v < best_d2:
            best_d2 = v
            best_idx = start + j
    return scalp_nodes[best_idx]

def _radial_outward_normal(point_xyz: np.ndarray, head_center_xyz: np.ndarray) -> np.ndarray:
    v = point_xyz - head_center_xyz
    nv = float(np.linalg.norm(v))
    if nv == 0:
        raise RuntimeError("Degenerate radial normal")
    return v / nv

def geodesic_iz_minus_mm_mesh(
    cap_map: dict[str, np.ndarray],
    scalp_nodes: np.ndarray,
    scalp_tree,
    head_center_xyz: np.ndarray,
    dist_mm: float,
    step_mm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk dist_mm along scalp from Iz in inferior midline direction (away from Oz) using mesh scalp nodes."""
    Iz_s = _project_to_scalp_nodes(cap_map["Iz"], scalp_nodes, scalp_tree)
    Oz_s = _project_to_scalp_nodes(cap_map["Oz"], scalp_nodes, scalp_tree)

    cur = Iz_s.copy()
    traveled = 0.0
    it = 0
    max_iter = 500  # hard safety cap
    stagnant = 0

    while traveled < dist_mm - 1e-6:
        it += 1
        if it > max_iter:
            raise GeodesicStuck(f"Exceeded max_iter={max_iter} at traveled={traveled:.2f}mm")

        n_out = _radial_outward_normal(cur, head_center_xyz)
        # tangent direction: away from Oz, projected onto tangent plane
        d = (cur - Oz_s)
        d_tan = d - float(np.dot(d, n_out)) * n_out
        nd = float(np.linalg.norm(d_tan))
        if nd == 0:
            raise GeodesicStuck("Degenerate tangent direction during geodesic walk")
        d_tan /= nd

        base_step = min(step_mm, dist_mm - traveled)

        # Try increasing step sizes in case nearest-node projection keeps returning the same node
        moved = False
        for mult in (1.0, 2.0, 5.0, 10.0):
            cand = cur + (base_step * mult) * d_tan
            nxt = _project_to_scalp_nodes(cand, scalp_nodes, scalp_tree)
            delta = float(np.linalg.norm(nxt - cur))
            if delta > 0.25:  # must move at least one voxel-ish to count
                traveled += delta
                cur = nxt
                moved = True
                stagnant = 0
                break

        if not moved:
            stagnant += 1
            # small nudge along tangent, then reproject (attempt to escape local snapping)
            cand = cur + 0.5 * d_tan
            cur2 = _project_to_scalp_nodes(cand, scalp_nodes, scalp_tree)
            if float(np.linalg.norm(cur2 - cur)) > 0.25:
                traveled += float(np.linalg.norm(cur2 - cur))
                cur = cur2
                stagnant = 0
            if stagnant >= 5:
                raise GeodesicStuck(f"Stalled geodesic: projection did not move (traveled={traveled:.2f}mm)")

    return Iz_s, Oz_s, cur

# ============================================================
# CHARM label volume helpers (for scalp geodesic + ROI GM projection)
# ============================================================
LABEL_WM = 1
LABEL_GM = 2
LABEL_CSF = 3
LABEL_BONE_1 = 4
LABEL_SCALP = 5
LABEL_BONE_2 = 7
LABEL_BONE_3 = 8

def pick_label_vol(m2m_dir: str) -> str:
    cands = [
        os.path.join(m2m_dir, "label_prep", "tissue_labeling_upsampled.nii.gz"),
        os.path.join(m2m_dir, "tissue_labeling_upsampled.nii.gz"),
        os.path.join(m2m_dir, "tissue_labeling.nii.gz"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No tissue_labeling(_upsampled).nii.gz found under {m2m_dir}")

def load_label_vol(m2m_dir: str):
    if nib is None:
        raise RuntimeError("nibabel is required. Install in your SimNIBS env: python -m pip install nibabel")
    lab_path = pick_label_vol(m2m_dir)
    img = nib.load(lab_path)
    lab = img.get_fdata().astype(np.int16)
    A = img.affine
    Ainv = np.linalg.inv(A)
    return lab, A, Ainv, lab_path

def vox_to_world(A: np.ndarray, ijk) -> np.ndarray:
    return (A @ np.array([ijk[0], ijk[1], ijk[2], 1.0]))[:3]

def world_to_vox(Ainv: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return (Ainv @ np.array([xyz[0], xyz[1], xyz[2], 1.0]))[:3]

def _voxel_sizes_mm(A: np.ndarray) -> tuple[float, float, float]:
    vx = float(np.linalg.norm(A[:3, 0]))
    vy = float(np.linalg.norm(A[:3, 1]))
    vz = float(np.linalg.norm(A[:3, 2]))
    return vx, vy, vz

def scalp_air_boundary_points(lab: np.ndarray, A: np.ndarray, Ainv: np.ndarray, center_xyz: np.ndarray, neigh_mm: float) -> list[np.ndarray]:
    """Return world-coords of SCALP voxels that touch AIR (0) in 6-neighborhood within neigh_mm of center_xyz."""
    vx, vy, vz = _voxel_sizes_mm(A)
    rad_i = max(1, int(math.ceil(neigh_mm / vx)))
    rad_j = max(1, int(math.ceil(neigh_mm / vy)))
    rad_k = max(1, int(math.ceil(neigh_mm / vz)))

    i0, j0, k0 = [int(round(x)) for x in world_to_vox(Ainv, center_xyz)]
    nx, ny, nz = lab.shape
    pts: list[np.ndarray] = []

    for i in range(i0 - rad_i, i0 + rad_i + 1):
        if i < 1 or i >= nx - 1:
            continue
        for j in range(j0 - rad_j, j0 + rad_j + 1):
            if j < 1 or j >= ny - 1:
                continue
            for k in range(k0 - rad_k, k0 + rad_k + 1):
                if k < 1 or k >= nz - 1:
                    continue
                if lab[i, j, k] != LABEL_SCALP:
                    continue
                xyz = vox_to_world(A, (i, j, k))
                if float(np.linalg.norm(xyz - center_xyz)) > neigh_mm:
                    continue
                if (
                    lab[i - 1, j, k] == 0
                    or lab[i + 1, j, k] == 0
                    or lab[i, j - 1, k] == 0
                    or lab[i, j + 1, k] == 0
                    or lab[i, j, k - 1] == 0
                    or lab[i, j, k + 1] == 0
                ):
                    pts.append(xyz)

    return pts

def project_to_scalp_boundary(lab: np.ndarray, A: np.ndarray, Ainv: np.ndarray, xyz: np.ndarray, search_mm: float) -> np.ndarray:
    pts = scalp_air_boundary_points(lab, A, Ainv, xyz, search_mm)
    if len(pts) == 0:
        raise RuntimeError(f"No scalp boundary points found within {search_mm} mm. Increase search radius.")
    P = np.asarray(pts, dtype=float)
    idx = int(np.argmin(np.sum((P - xyz) ** 2, axis=1)))
    return P[idx]

def pca_plane_normal(points_xyz: list[np.ndarray], outward_ref_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit plane by SVD/PCA. Return (centroid, outward normal)."""
    P = np.asarray(points_xyz, dtype=float)
    if P.shape[0] < 30:
        raise RuntimeError(f"Too few points for scalp normal fit: {P.shape[0]}")
    mu = P.mean(axis=0)
    X = P - mu
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    n = vt[-1, :]
    n = n / (np.linalg.norm(n) + 1e-12)
    if float(np.dot(n, outward_ref_vec)) < 0:
        n = -n
    return mu, n

def scalp_normal_at_point(lab: np.ndarray, A: np.ndarray, Ainv: np.ndarray, scalp_xyz: np.ndarray, neigh_mm: float) -> np.ndarray:
    shape = np.array(lab.shape[:3], dtype=float)
    center_vox = (shape - 1.0) / 2.0
    center_world = vox_to_world(A, center_vox)
    pts = scalp_air_boundary_points(lab, A, Ainv, scalp_xyz, neigh_mm)
    if len(pts) < 30:
        pts = scalp_air_boundary_points(lab, A, Ainv, scalp_xyz, neigh_mm * 1.8)
    _, n_out = pca_plane_normal(pts, outward_ref_vec=(scalp_xyz - center_world))
    return n_out

def geodesic_iz_minus_mm(lab: np.ndarray, A: np.ndarray, Ainv: np.ndarray, Iz_xyz: np.ndarray, Oz_xyz: np.ndarray, dist_mm: float,
                         step_mm: float = 1.0, neigh_mm_normal: float = 12.0, project_search_mm: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk dist_mm along scalp from Iz in the inferior midline direction (away from Oz), returning (Iz_surf, Oz_surf, target_surf)."""
    Iz_s = project_to_scalp_boundary(lab, A, Ainv, Iz_xyz, 20.0)
    Oz_s = project_to_scalp_boundary(lab, A, Ainv, Oz_xyz, 20.0)

    cur = Iz_s.copy()
    traveled = 0.0
    while traveled < dist_mm - 1e-6:
        n_out = scalp_normal_at_point(lab, A, Ainv, cur, neigh_mm_normal)
        n_out = n_out / (np.linalg.norm(n_out) + 1e-12)

        # tangent direction: away from Oz, projected onto tangent plane
        d = (cur - Oz_s)
        d_tan = d - float(np.dot(d, n_out)) * n_out
        nd = float(np.linalg.norm(d_tan))
        if nd == 0:
            raise RuntimeError("Degenerate tangent direction during geodesic walk")
        d_tan /= nd

        step = min(step_mm, dist_mm - traveled)
        cand = cur + step * d_tan
        nxt = project_to_scalp_boundary(lab, A, Ainv, cand, project_search_mm)

        traveled += float(np.linalg.norm(nxt - cur))
        cur = nxt

    return Iz_s, Oz_s, cur

def first_gm_along_inward_normal(lab: np.ndarray, A: np.ndarray, Ainv: np.ndarray, scalp_xyz: np.ndarray, n_out: np.ndarray, step_mm: float = 0.25,
                                max_len_mm: float = 160.0) -> np.ndarray:
    """March from outside -> inward along -n_out and return the first GM world point encountered."""
    n_out = n_out / (np.linalg.norm(n_out) + 1e-12)
    inward = -n_out
    start = scalp_xyz + 30.0 * n_out
    n_steps = int(max_len_mm / step_mm)

    for t in range(n_steps + 1):
        p = start + (t * step_mm) * inward
        ijk = world_to_vox(Ainv, p)
        i, j, k = [int(round(x)) for x in ijk]
        if i < 0 or j < 0 or k < 0 or i >= lab.shape[0] or j >= lab.shape[1] or k >= lab.shape[2]:
            continue
        if int(lab[i, j, k]) == LABEL_GM:
            return p

    raise RuntimeError("Did not hit GM along inward normal (increase max_len or check geometry)")


# ============================================================
# EEG cap parsing (supports template: Type, X, Y, Z, Label)
# ============================================================
def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

def load_eeg_cap_csv(cap_csv: str) -> dict[str, np.ndarray]:
    """
    Supports:
      - Template: Type, X, Y, Z, Label
      - Legacy:   Label, X, Y, Z
    Returns: {label: np.array([x,y,z])}
    """
    out: dict[str, np.ndarray] = {}
    with open(cap_csv, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            row = [c.strip() for c in row]
            if len(row) >= 5 and _is_float(row[1]) and _is_float(row[2]) and _is_float(row[3]):
                lab = row[4]
                if lab:
                    out[lab] = np.array([float(row[1]), float(row[2]), float(row[3])], dtype=float)
                continue
            if len(row) >= 4 and _is_float(row[1]) and _is_float(row[2]) and _is_float(row[3]):
                lab = row[0]
                if lab:
                    out[lab] = np.array([float(row[1]), float(row[2]), float(row[3])], dtype=float)
    return out

def find_cap_with_labels(m2m_dir: str, required: list[str]) -> str:
    eeg_dir = os.path.join(m2m_dir, "eeg_positions")
    if not os.path.isdir(eeg_dir):
        raise FileNotFoundError(f"Missing eeg_positions folder: {eeg_dir}")

    caps = sorted(glob.glob(os.path.join(eeg_dir, "*.csv")))
    if not caps:
        raise FileNotFoundError(f"No EEG cap CSV found in {eeg_dir}")

    req = set(required)
    candidates: list[tuple[int, str]] = []

    for cap in caps:
        mp = load_eeg_cap_csv(cap)
        if not req.issubset(mp.keys()):
            continue
        score = 0
        fn = os.path.basename(cap).lower()
        if "jurak" in fn:
            score += 100
        if "10-10" in fn or "1010" in fn:
            score += 10
        if "ui" in fn:
            score += 5
        candidates.append((score, cap))

    if not candidates:
        raise KeyError(f"No EEG cap contains required labels: {sorted(req)} in {eeg_dir}")

    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


# ============================================================
# Geometry: compute "2 cm below Iz" and ydir (and flip it)
# ============================================================
def compute_stim_center_and_ydir_flipped(cap_map: dict[str, np.ndarray], below_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      stim_center_xyz: 2 cm below Iz (toward neck), along Iz–Oz direction
      ydir_flipped:    point that flips handle direction 180° relative to using Oz as ydir
    Requires: Iz and Oz
    """
    Iz = cap_map["Iz"]
    Oz = cap_map["Oz"]

    v = Iz - Oz
    nv = float(np.linalg.norm(v))
    if nv == 0.0:
        raise RuntimeError("Iz and Oz are identical; cannot define midline direction")
    u_down = v / nv

    stim_center = Iz + below_mm * u_down

    # Previous "upward" ydir would be Oz. Flip 180° around centre:
    ydir_up = Oz
    ydir_flipped = 2.0 * stim_center - ydir_up

    return stim_center, ydir_flipped


# ============================================================
# ROI center: GM under scalp point (inward shift + snap to GM)
# ============================================================
def snap_to_gm_under_point(head_mesh: simnibs.mesh_io.Msh, scalp_point_xyz: np.ndarray, inward_mm: float) -> np.ndarray:
    nodes = np.asarray(head_mesh.nodes.node_coord, dtype=float)
    head_center = nodes.mean(axis=0)

    v = scalp_point_xyz - head_center
    nv = float(np.linalg.norm(v))
    if nv == 0.0:
        raise RuntimeError("Degenerate head_center/scalp vector")
    outward = v / nv

    p_in = scalp_point_xyz - inward_mm * outward

    gm = head_mesh.crop_mesh(simnibs.ElementTags.GM)
    gm_ctr = np.asarray(gm.elements_baricenters().value, dtype=float)
    if gm_ctr.size == 0:
        raise RuntimeError("GM crop produced empty mesh")

    idx = int(np.argmin(np.sum((gm_ctr - p_in) ** 2, axis=1)))
    return gm_ctr[idx]


# ============================================================
# Field stats (volume-weighted); peak only
# ============================================================
def get_field_name(mesh: simnibs.mesh_io.Msh) -> str:
    if "magnE" in mesh.field:
        return "magnE"
    if "normE" in mesh.field:
        return "normE"
    raise RuntimeError("Mesh has no magnE/normE field")

def weighted_percentiles(values: np.ndarray, weights: np.ndarray, percentiles: tuple[int, ...]) -> dict[int, float]:
    if values.size == 0:
        return {p: float("nan") for p in percentiles}
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return {p: float("nan") for p in percentiles}
    v = v[m]
    w = w[m]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    out = {}
    for p in percentiles:
        out[p] = float(np.interp(p / 100.0, cw, v))
    return out

def roi_distribution(mesh: simnibs.mesh_io.Msh, center_xyz: np.ndarray, radius_mm: float, tag):
    field_name = get_field_name(mesh)
    sub = mesh.crop_mesh(tag)

    E   = np.asarray(sub.field[field_name].value, dtype=float)
    ctr = np.asarray(sub.elements_baricenters().value, dtype=float)
    vol = np.asarray(sub.elements_volumes_and_areas().value, dtype=float)

    d = np.linalg.norm(ctr - center_xyz, axis=1)
    m = d < radius_mm
    if not np.any(m):
        return np.array([]), np.array([]), 0, 0.0

    E_roi = E[m]
    V_roi = vol[m]
    n_el = int(np.sum(m))
    vol_mm3 = float(np.nansum(V_roi))
    return E_roi, V_roi, n_el, vol_mm3

def stats_from_distribution(E_roi: np.ndarray, V_roi: np.ndarray) -> dict[str, float]:
    if E_roi.size == 0 or np.nansum(V_roi) <= 0:
        out = dict(mean=np.nan, std=np.nan, rms=np.nan, min=np.nan, peak=np.nan)
        for p in ROI_PCTS:
            out[f"p{p}"] = np.nan
        for thr in ROI_THRESHOLDS_VM:
            out[f"frac_gt_{thr:g}"] = np.nan
        return out

    mean = float(np.average(E_roi, weights=V_roi))
    var  = float(np.average((E_roi - mean) ** 2, weights=V_roi))
    std  = float(np.sqrt(var))
    rms  = float(np.sqrt(np.average(E_roi ** 2, weights=V_roi)))
    mn   = float(np.nanmin(E_roi))
    peak = float(np.nanmax(E_roi))

    out = dict(mean=mean, std=std, rms=rms, min=mn, peak=peak)

    wp = weighted_percentiles(E_roi, V_roi, ROI_PCTS)
    for p in ROI_PCTS:
        out[f"p{p}"] = wp[p]

    Vtot = float(np.nansum(V_roi))
    for thr in ROI_THRESHOLDS_VM:
        out[f"frac_gt_{thr:g}"] = float(np.nansum(V_roi[E_roi > thr]) / Vtot)

    return out

def fractions_at_scale(E_roi_100: np.ndarray, V_roi: np.ndarray, scale_total: float) -> dict[str, float]:
    out = {}
    if E_roi_100.size == 0 or np.nansum(V_roi) <= 0:
        for thr in ROI_THRESHOLDS_VM:
            out[f"frac_gt_{thr:g}"] = np.nan
        return out

    if scale_total <= 0:
        for thr in ROI_THRESHOLDS_VM:
            out[f"frac_gt_{thr:g}"] = 0.0
        return out

    Vtot = float(np.nansum(V_roi))
    for thr in ROI_THRESHOLDS_VM:
        thr_unscaled = thr / scale_total
        out[f"frac_gt_{thr:g}"] = float(np.nansum(V_roi[E_roi_100 > thr_unscaled]) / Vtot)
    return out

def build_header(prefix: str) -> list[str]:
    cols = [
        f"{prefix}_n_elements_in_roi",
        f"{prefix}_roi_volume_mm3",
        f"{prefix}_mean_|E|_V/m",
        f"{prefix}_std_|E|_V/m",
        f"{prefix}_rms_|E|_V/m",
        f"{prefix}_min_|E|_V/m",
        f"{prefix}_peak_|E|_V/m",
    ]
    for p in ROI_PCTS:
        cols.append(f"{prefix}_p{p}_|E|_V/m")
    for thr in ROI_THRESHOLDS_VM:
        cols.append(f"{prefix}_frac_vol_gt_{thr:g}_V/m")
    return cols

def stats_row_scaled(base: dict, n_el: int, vol_mm3: float,
                     E100: np.ndarray, V: np.ndarray, scale_total: float) -> list:
    if n_el == 0:
        return [0, 0.0] + [float("nan")] * (5 + len(ROI_PCTS) + len(ROI_THRESHOLDS_VM))

    if scale_total <= 0:
        amp = dict(mean=0.0, std=0.0, rms=0.0, min=0.0, peak=0.0, **{f"p{p}": 0.0 for p in ROI_PCTS})
        fr  = {f"frac_gt_{thr:g}": 0.0 for thr in ROI_THRESHOLDS_VM}
    else:
        amp = dict(
            mean=base["mean"] * scale_total,
            std =base["std"]  * scale_total,
            rms =base["rms"]  * scale_total,
            min =base["min"]  * scale_total,
            peak=base["peak"] * scale_total,
            **{f"p{p}": base[f"p{p}"] * scale_total for p in ROI_PCTS}
        )
        fr = fractions_at_scale(E100, V, scale_total)

    row = [n_el, vol_mm3, amp["mean"], amp["std"], amp["rms"], amp["min"], amp["peak"]]
    for p in ROI_PCTS:
        row.append(amp[f"p{p}"])
    for thr in ROI_THRESHOLDS_VM:
        row.append(fr.get(f"frac_gt_{thr:g}", float("nan")))
    return row


# ============================================================
# Clean reusable subject-level runner
# ============================================================
def _format_xyz(xyz: np.ndarray) -> str:
    return f"{xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}"


def run_step1_for_subject(
    subject_id: str,
    subject_dir: str,
    m2m_dir: str,
    out_subdir: str,
    coil_ccd_dir: str | None = None,
    run_simnibs_flag: bool = True,
    force_rerun: bool = False,
    run_tms: bool = True,
    run_tdcs: bool = True,
    roi_radius_mm: float = ROI_RADIUS_MM,
    inion_below_mm: float = INION_BELOW_MM,
) -> str:
    """
    Run Step 1 for one subject and write the per-subject E-field CSV.

    This is the clean version of the old exploratory Step 1 main loop.
    It preserves the scientific setup while avoiding hard-coded cohort starts.

    Returns
    -------
    out_csv:
        Path to the completed per-subject Step 1 CSV.
    """
    coil_ccd_dir = coil_ccd_dir or COIL_CCD_DIR

    subject_dir = os.path.abspath(subject_dir)
    m2m_dir = os.path.abspath(m2m_dir)
    out_root = os.path.join(subject_dir, out_subdir)
    ensure_dir(out_root)

    print(f"[START] {subject_id}", flush=True)
    t0_subj = time.time()

    head_mesh_path = find_head_mesh(m2m_dir)
    print(f"[STEP] {subject_id} read_msh start", flush=True)
    head_mesh = simnibs.read_msh(head_mesh_path)
    print(f"[STEP] {subject_id} read_msh done ({time.time() - t0_subj:.1f}s)", flush=True)

    # Precompute scalp surface nodes + KDTree once per subject for fast geodesic/projection.
    t0 = time.time()
    scalp_nodes, scalp_tree = _build_scalp_kdtree(head_mesh)
    head_center_xyz = np.asarray(head_mesh.nodes.node_coord, dtype=float).mean(axis=0)
    print(
        f"[STEP] {subject_id} scalp_nodes={scalp_nodes.shape[0]} "
        f"kdtree={scalp_tree is not None} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    # Need Iz/Oz for target and Cz for tDCS orientation.
    t0 = time.time()
    cap_csv = find_cap_with_labels(m2m_dir, required=["Iz", "Oz", "Cz"])
    cap_map = load_eeg_cap_csv(cap_csv)
    print(f"[STEP] {subject_id} cap loaded ({time.time() - t0:.1f}s)", flush=True)

    # Load CHARM labels once per subject for GM projection.
    t0 = time.time()
    lab, A, Ainv, lab_path = load_label_vol(m2m_dir)
    print(f"[STEP] {subject_id} label vol loaded ({time.time() - t0:.1f}s)", flush=True)

    # Geodesic scalp target: 20 mm inferior from Iz along midline direction away from Oz.
    t0 = time.time()
    try:
        Iz_surf, Oz_surf, stim_center = geodesic_iz_minus_mm_mesh(
            cap_map,
            scalp_nodes,
            scalp_tree,
            head_center_xyz,
            dist_mm=inion_below_mm,
            step_mm=1.0,
        )
        print(f"[STEP] {subject_id} geodesic done ({time.time() - t0:.1f}s)", flush=True)
    except GeodesicStuck as e:
        print(
            f"[FALLBACK] {subject_id} geodesic stalled ({e}); "
            "using straight-line Iz->Oz shift projected to scalp",
            flush=True,
        )
        stim_center_raw, _ = compute_stim_center_and_ydir_flipped(cap_map, inion_below_mm)
        Iz_surf = _project_to_scalp_nodes(cap_map["Iz"], scalp_nodes, scalp_tree)
        Oz_surf = _project_to_scalp_nodes(cap_map["Oz"], scalp_nodes, scalp_tree)
        stim_center = _project_to_scalp_nodes(stim_center_raw, scalp_nodes, scalp_tree)

    # Coil handle direction: flipped 180 degrees relative to using Oz as ydir.
    ydir_flipped = 2.0 * stim_center - Oz_surf

    # ROI center: first GM point along inward radial normal under the stimulation center.
    n_out = _radial_outward_normal(stim_center, head_center_xyz)
    try:
        roi_center_gm = first_gm_along_inward_normal(
            lab,
            A,
            Ainv,
            stim_center,
            n_out,
            step_mm=0.25,
            max_len_mm=160.0,
        )
    except Exception as e:
        print(
            f"[FALLBACK] {subject_id} GM projection failed ({e}); "
            "using mesh snap_to_gm_under_point(inward_mm=10)",
            flush=True,
        )
        roi_center_gm = snap_to_gm_under_point(head_mesh, stim_center, inward_mm=10.0)

    print(f"[STEP] {subject_id} GM projection done ({time.time() - t0_subj:.1f}s)", flush=True)

    cathode_center = np.asarray(mni2subject_coords(CHIN_MNI, m2m_dir), dtype=float)

    out_csv = os.path.join(
        out_root,
        f"STEP1_E0_inion2cmBelow_roi{int(roi_radius_mm)}mm_allstats_scaled.csv",
    )
    tmp_csv = out_csv + ".tmp"

    # Avoid treating an old temporary file as current work.
    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)

    GM_HDR = build_header("roi_graymatter")
    WM_HDR = build_header("roi_whitematter")

    with open(tmp_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "subject_id",
            "modality",
            "coil",
            "distance_mm",
            "condition",
            "file_name",
            "path",
            "label_vol",
            "roi_radius_mm",
            "stim_center_subject_xyz_mm",
            "roi_center_gm_subject_xyz_mm",
            "intensity (%MT)",
            *GM_HDR,
            *WM_HDR,
            "scale_to_active100",
        ])

        # TMS: active mesh once per coil, then scaled rows for active and sham variants.
        if run_tms:
            for coil in COILS:
                coil_prefix = coil["coil_prefix"]
                coil_ccd = os.path.join(coil_ccd_dir, coil["ccd_file"])
                if not os.path.exists(coil_ccd):
                    raise FileNotFoundError(f"Missing coil CCD: {coil_ccd}")

                didt_active_A_per_us = coil["stim_didt_max_A_per_us"] * (coil["mt_percent_mso"] / 100.0)
                didt_active_A_per_s = didt_active_A_per_us * 1e6

                for dist_mm in TMS_DISTANCES_MM:
                    dist_tag = f"{int(dist_mm)}mm"
                    sim_dir = os.path.join(out_root, f"TMS_{coil_prefix}_Active_{dist_tag}")
                    ensure_dir(sim_dir)

                    scalar_path = latest_scalar_mesh(sim_dir)
                    need_run = run_simnibs_flag and (force_rerun or scalar_path is None)

                    if need_run:
                        print(f"[RUN] {subject_id} | TMS {coil_prefix} Active {dist_tag}", flush=True)
                        S = sim_struct.SESSION()
                        S.fnamehead = head_mesh_path
                        S.pathfem = sim_dir
                        S.open_in_gmsh = False
                        S.eeg_cap = cap_csv

                        tms = S.add_tmslist()
                        tms.fnamecoil = coil_ccd

                        pos = tms.add_position()
                        pos.centre = list(stim_center)
                        pos.pos_ydir = list(ydir_flipped)
                        pos.distance = float(dist_mm)
                        pos.didt = float(didt_active_A_per_s)
                        pos.name = f"active_{dist_tag}"

                        run_simnibs(S)

                    scalar_path = latest_scalar_mesh(sim_dir)
                    if scalar_path is None:
                        raise RuntimeError(f"No *_scalar.msh found in {sim_dir}")

                    mesh = simnibs.read_msh(scalar_path)

                    gmE100, gmV, gmN, gmVol = roi_distribution(
                        mesh,
                        roi_center_gm,
                        roi_radius_mm,
                        simnibs.ElementTags.GM,
                    )
                    wmE100, wmV, wmN, wmVol = roi_distribution(
                        mesh,
                        roi_center_gm,
                        roi_radius_mm,
                        simnibs.ElementTags.WM,
                    )
                    gm100 = stats_from_distribution(gmE100, gmV)
                    wm100 = stats_from_distribution(wmE100, wmV)

                    for inten in INTENSITIES_PCT_MT:
                        frac = float(inten) / 100.0

                        cond_active = f"{coil_prefix}_Active{dist_tag}"
                        scale_total = frac

                        gm_row = stats_row_scaled(gm100, gmN, gmVol, gmE100, gmV, scale_total)
                        wm_row = stats_row_scaled(wm100, wmN, wmVol, wmE100, wmV, scale_total)

                        w.writerow([
                            subject_id,
                            "TMS",
                            coil_prefix,
                            dist_mm,
                            cond_active,
                            os.path.basename(scalar_path),
                            scalar_path,
                            lab_path,
                            roi_radius_mm,
                            _format_xyz(stim_center),
                            _format_xyz(roi_center_gm),
                            inten,
                            *gm_row,
                            *wm_row,
                            scale_total,
                        ])

                        for sham_suffix, sham_scale in coil["sham_variants"]:
                            cond_sham = (
                                f"{coil_prefix}_Sham{dist_tag}+electrodes"
                                if sham_suffix == "Sham+electrodes"
                                else f"{coil_prefix}_Sham{dist_tag}"
                            )

                            scale_total = frac * float(sham_scale)
                            gm_row = stats_row_scaled(gm100, gmN, gmVol, gmE100, gmV, scale_total)
                            wm_row = stats_row_scaled(wm100, wmN, wmVol, wmE100, wmV, scale_total)

                            w.writerow([
                                subject_id,
                                "TMS",
                                coil_prefix,
                                dist_mm,
                                cond_sham,
                                os.path.basename(scalar_path),
                                scalar_path,
                                lab_path,
                                roi_radius_mm,
                                _format_xyz(stim_center),
                                _format_xyz(roi_center_gm),
                                inten,
                                *gm_row,
                                *wm_row,
                                scale_total,
                            ])

        # tDCS: run once, extract unscaled 2 mA field.
        if run_tdcs:
            sim_dir = os.path.join(out_root, f"tDCS_{TDCS_MONTAGE_NAME}")
            ensure_dir(sim_dir)

            scalar_path = latest_scalar_mesh(sim_dir)
            need_run = run_simnibs_flag and (force_rerun or scalar_path is None)

            if need_run:
                print(f"[RUN] {subject_id} | tDCS {TDCS_MONTAGE_NAME}", flush=True)
                S = sim_struct.SESSION()
                S.fnamehead = head_mesh_path
                S.pathfem = sim_dir
                S.fields = "eE"
                S.open_in_gmsh = False
                S.eeg_cap = cap_csv

                td = S.add_tdcslist()
                td.name = f"TDCS_{TDCS_MONTAGE_NAME}"

                an = td.add_electrode()
                an.centre = list(stim_center)
                an.pos_ydir = TDCS_AN_YDIR
                an.shape = "rect"
                an.dimensions = TDCS_PAD_DIMS_MM
                an.thickness = TDCS_THICKNESS_MM
                an.dimensions_sponge = TDCS_SPONGE_MM
                an.channelnr = 1

                ca = td.add_electrode()
                ca.centre = list(cathode_center)
                ca.pos_ydir = TDCS_CA_YDIR
                ca.shape = "rect"
                ca.dimensions = TDCS_PAD_DIMS_MM
                ca.thickness = TDCS_THICKNESS_MM
                ca.dimensions_sponge = TDCS_SPONGE_MM
                ca.channelnr = 2

                current_A = TDCS_CURRENT_mA * 1e-3
                td.currents = [current_A, -current_A]

                run_simnibs(S)

            scalar_path = latest_scalar_mesh(sim_dir)
            if scalar_path is None:
                raise RuntimeError(f"No *_scalar.msh found in {sim_dir}")

            mesh = simnibs.read_msh(scalar_path)

            gmE100, gmV, gmN, gmVol = roi_distribution(
                mesh,
                roi_center_gm,
                roi_radius_mm,
                simnibs.ElementTags.GM,
            )
            wmE100, wmV, wmN, wmVol = roi_distribution(
                mesh,
                roi_center_gm,
                roi_radius_mm,
                simnibs.ElementTags.WM,
            )
            gm100 = stats_from_distribution(gmE100, gmV)
            wm100 = stats_from_distribution(wmE100, wmV)

            gm_row = stats_row_scaled(gm100, gmN, gmVol, gmE100, gmV, 1.0)
            wm_row = stats_row_scaled(wm100, wmN, wmVol, wmE100, wmV, 1.0)

            w.writerow([
                subject_id,
                "tDCS",
                "NA",
                0.0,
                "tDCS",
                os.path.basename(scalar_path),
                scalar_path,
                lab_path,
                roi_radius_mm,
                _format_xyz(stim_center),
                _format_xyz(roi_center_gm),
                100,
                *gm_row,
                *wm_row,
                1.0,
            ])

    # Validate row count before replacing final CSV.
    try:
        import pandas as pd
        tmp_df = pd.read_csv(tmp_csv)
        if len(tmp_df) != 56:
            raise RuntimeError(
                f"Temporary Step 1 CSV has {len(tmp_df)} rows, expected 56: {tmp_csv}"
            )
    except Exception:
        raise

    os.replace(tmp_csv, out_csv)

    print(f"[OK] {subject_id}: wrote {out_csv} (total {time.time() - t0_subj:.1f}s)", flush=True)
    return out_csv

