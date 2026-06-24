from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import numpy as np


# Output column index used by the original CRBL_MF code.
# In the model output X, column 10 corresponds to Purkinje-cell population firing rate.
PC_RATE_INDEX = 10


@dataclass
class CRBLModel:
    repo_root: Path
    coeffp_dir: Path
    TFgrc: Any
    TFgoc: Any
    TFmli: Any
    TFpc: Any
    find_fixed_point_mossy: Any
    Ngrc: int = 28615
    Ngoc: int = 70
    Nmossy: int = 2336
    Nmli: int = 446
    Npc: int = 99
    T_s: float = 3.5e-3
    w: float = 0.0


def load_crbl_model(repo_root: str | Path) -> CRBLModel:
    """
    Load the Lorenzi cerebellar mean-field model from a local CRBL_MF repo.

    This does not modify the CRBL_MF repo. It only imports its code and loads
    the published transfer-function coefficient files.
    """
    repo_root = Path(repo_root).expanduser().resolve()

    if not repo_root.exists():
        raise FileNotFoundError(f"CRBL_MF repo root not found: {repo_root}")

    mf_prediction_dir = repo_root / "MF_prediction"
    if not mf_prediction_dir.exists():
        raise FileNotFoundError(f"MF_prediction folder not found: {mf_prediction_dir}")

    coeffp_candidates = [
        repo_root / "coeffP",
        mf_prediction_dir / "coeffP",
    ]

    coeffp_dir = next((p for p in coeffp_candidates if p.exists()), None)
    if coeffp_dir is None:
        raise FileNotFoundError(
            f"Could not find coeffP directory. Checked: {coeffp_candidates}"
        )

    for path in [repo_root, mf_prediction_dir]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from MF_prediction.load_config_TF import (
        load_transfer_functions,
        load_transfer_functions_goc,
    )
    from MF_prediction.master_equation_CRBL_MF import find_fixed_point_mossy

    file_grc = coeffp_dir / "GrC_fit.npy"
    file_goc = coeffp_dir / "GoC_fit.npy"
    file_mli = coeffp_dir / "MLI_fit.npy"
    file_pc = coeffp_dir / "PC_fit.npy"

    for p in [file_grc, file_goc, file_mli, file_pc]:
        if not p.exists():
            raise FileNotFoundError(f"Missing transfer-function coefficient file: {p}")

    ntwk = "CRBL_CONFIG_20PARALLEL_wN"

    TFgrc = load_transfer_functions("GrC", ntwk, str(file_grc), alpha=2.0)
    TFgoc = load_transfer_functions_goc("GoC", ntwk, str(file_goc), alpha=1.3)
    TFmli = load_transfer_functions("MLI", ntwk, str(file_mli), alpha=5)
    TFpc = load_transfer_functions("PC", ntwk, str(file_pc), alpha=5)

    return CRBLModel(
        repo_root=repo_root,
        coeffp_dir=coeffp_dir,
        TFgrc=TFgrc,
        TFgoc=TFgoc,
        TFmli=TFmli,
        TFpc=TFpc,
        find_fixed_point_mossy=find_fixed_point_mossy,
    )


def run_crbl_meanfield(
    model: CRBLModel,
    t_s: np.ndarray,
    fmossy_hz: np.ndarray,
) -> np.ndarray:
    """
    Run the cerebellar mean-field model for a supplied mossy-fiber drive time series.
    """
    if t_s.ndim != 1:
        raise ValueError("t_s must be one-dimensional.")

    if fmossy_hz.shape != t_s.shape:
        raise ValueError(
            f"fmossy_hz shape {fmossy_hz.shape} does not match t_s shape {t_s.shape}"
        )

    if not np.all(np.isfinite(fmossy_hz)):
        raise ValueError("fmossy_hz contains non-finite values.")

    # Initial-condition vector preserved from the original CRBL_MF runner.
    # The ninth entry is initialized with baseline mossy drive.
    ci_vec = [
        0.5, 5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, float(fmossy_hz[0]), 15,
        38, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5,
    ]

    X = model.find_fixed_point_mossy(
        model.TFgrc,
        model.TFgoc,
        model.TFmli,
        model.TFpc,
        ci_vec,
        t_s,
        model.w,
        fmossy_hz,
        model.Ngrc,
        model.Ngoc,
        model.Nmossy,
        model.Nmli,
        model.Npc,
        model.T_s,
        verbose=False,
    )

    X = np.asarray(X)

    if X.ndim != 2:
        raise RuntimeError(f"CRBL_MF output X should be 2D, got shape {X.shape}")

    if X.shape[1] <= PC_RATE_INDEX:
        raise RuntimeError(
            f"CRBL_MF output has {X.shape[1]} columns; cannot read PC index {PC_RATE_INDEX}"
        )

    if not np.all(np.isfinite(X)):
        raise RuntimeError("CRBL_MF output contains non-finite values.")

    return X


def extract_pc_rate_hz(X: np.ndarray) -> np.ndarray:
    """
    Extract Purkinje-cell population firing rate from CRBL_MF output.
    """
    return np.asarray(X[:, PC_RATE_INDEX], dtype=float)
