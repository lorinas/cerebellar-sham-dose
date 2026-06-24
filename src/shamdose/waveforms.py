from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WaveformGain:
    coil_family: str
    mode: str
    tau_ms: float
    k_tau: float
    dt_us: float
    waveform_column: str
    waveform_path: str


def resolve_path(path: str | Path, project_root: str | Path) -> Path:
    """
    Resolve absolute or project-relative paths.
    """
    path = Path(path).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (Path(project_root).resolve() / path).resolve()


def load_signed_unit_waveform(
    csv_path: str | Path,
    column: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load a digitized signed normalized waveform.

    The waveform is renormalized so max(abs(w)) == 1. This protects us from
    small digitization/rounding differences.

    Returns
    -------
    time_ms:
        Time vector in ms.
    waveform:
        Signed unit-peak waveform.
    dt_s:
        Sampling interval in seconds.
    """
    csv_path = Path(csv_path).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"Waveform file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "time_ms" not in df.columns:
        raise ValueError(f"{csv_path} missing time_ms column.")

    if column not in df.columns:
        raise ValueError(
            f"{csv_path} missing waveform column '{column}'. "
            f"Available columns: {list(df.columns)}"
        )

    time_ms = pd.to_numeric(df["time_ms"], errors="coerce").to_numpy(dtype=float)
    waveform = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(time_ms) & np.isfinite(waveform)
    time_ms = time_ms[mask]
    waveform = waveform[mask]

    if time_ms.size < 10:
        raise ValueError(f"Too few valid waveform samples in {csv_path}")

    order = np.argsort(time_ms)
    time_ms = time_ms[order]
    waveform = waveform[order]

    dt_ms = float(np.median(np.diff(time_ms)))
    if dt_ms <= 0:
        raise ValueError(f"Non-positive waveform dt in {csv_path}")

    max_abs = float(np.max(np.abs(waveform)))
    if max_abs <= 0:
        raise ValueError(f"Waveform column {column} in {csv_path} has zero amplitude.")

    waveform = waveform / max_abs
    dt_s = dt_ms * 1e-3

    if not (np.min(waveform) < 0 and np.max(waveform) > 0):
        raise ValueError(
            f"Waveform {column} in {csv_path} is not biphasic/signed after loading. "
            "Check that the correct column was selected."
        )

    return time_ms, waveform, dt_s


def k_tau_peak_abs_exact_rc(
    waveform: np.ndarray,
    dt_s: float,
    tau_s: float,
) -> float:
    """
    First-order membrane filter gain using exact discrete-time update.

    y[n] = alpha*y[n-1] + (1-alpha)*w[n]

    The gain is the peak absolute filtered response:
        max(abs(y))

    This is important for biphasic signed TMS pulses.
    """
    if tau_s <= 0:
        raise ValueError("tau_s must be positive.")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive.")

    alpha = math.exp(-dt_s / tau_s)
    one_minus = 1.0 - alpha

    y = 0.0
    peak_abs = 0.0

    for sample in waveform:
        y = alpha * y + one_minus * float(sample)
        peak_abs = max(peak_abs, abs(y))

    return float(peak_abs)


def compute_waveform_gains(
    waveform_config: dict,
    tau_list_ms: list[float],
    project_root: str | Path,
) -> pd.DataFrame:
    """
    Compute k_tau for active and sham waveforms for each coil family.
    """
    rows: list[WaveformGain] = []

    for coil_family, cfg in waveform_config.items():
        wf_path = resolve_path(cfg["path"], project_root)
        active_col = cfg["active_column"]
        sham_col = cfg["sham_column"]

        for mode, col in [("Active", active_col), ("Sham", sham_col)]:
            _, waveform, dt_s = load_signed_unit_waveform(wf_path, col)

            for tau_ms in tau_list_ms:
                tau_s = float(tau_ms) * 1e-3
                k_tau = k_tau_peak_abs_exact_rc(
                    waveform=waveform,
                    dt_s=dt_s,
                    tau_s=tau_s,
                )

                rows.append(
                    WaveformGain(
                        coil_family=coil_family,
                        mode=mode,
                        tau_ms=float(tau_ms),
                        k_tau=float(k_tau),
                        dt_us=float(dt_s * 1e6),
                        waveform_column=col,
                        waveform_path=str(wf_path),
                    )
                )

    return pd.DataFrame([row.__dict__ for row in rows])


def print_waveform_gain_summary(gains: pd.DataFrame) -> None:
    """
    Print k_tau table.
    """
    print("\n[WAVEFORM GAINS]")
    view = gains.copy()
    view["tau_ms"] = view["tau_ms"].map(lambda x: f"{x:.3f}")
    view["k_tau"] = view["k_tau"].map(lambda x: f"{x:.6f}")
    view["dt_us"] = view["dt_us"].map(lambda x: f"{x:.2f}")
    print(
        view[
            ["coil_family", "mode", "tau_ms", "k_tau", "dt_us", "waveform_column"]
        ].to_string(index=False)
    )
