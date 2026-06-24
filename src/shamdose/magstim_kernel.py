from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COILS = ["magstim", "magventure"]

CONDITION_PREFIXES = {
    "magstim_active": "magstim_active",
    "magstim_sham": "magstim_sham",
    "magventure_active": "magventure_active",
    "magventure_sham": "magventure_sham",
}


def field_magnitude(df: pd.DataFrame, prefix: str) -> pd.Series:
    """
    Compute E-field vector magnitude from Ex/Ey/Ez columns.
    """
    return np.sqrt(
        pd.to_numeric(df[f"{prefix}_Ex_Vm"], errors="coerce") ** 2
        + pd.to_numeric(df[f"{prefix}_Ey_Vm"], errors="coerce") ** 2
        + pd.to_numeric(df[f"{prefix}_Ez_Vm"], errors="coerce") ** 2
    )


def add_magnitude_and_kernel_columns(
    df: pd.DataFrame,
    active_floor_frac: float = 0.01,
) -> pd.DataFrame:
    """
    Add |E| magnitude and sham/active spatial-kernel columns.

    For each coil:
        active_abs_Vm
        sham_abs_Vm
        active_norm_to_active_peak
        sham_norm_to_active_peak
        sham_over_active_raw
        sham_over_active_regularized

    The regularized ratio divides by max(active_abs, floor), where:
        floor = active_floor_frac × active_peak

    This protects against unstable division by tiny active-field values.
    """
    if active_floor_frac <= 0:
        raise ValueError("active_floor_frac must be positive.")

    out = df.copy()

    out["r_xy_cm"] = np.sqrt(out["x_cm"] ** 2 + out["y_cm"] ** 2)

    for coil in COILS:
        active_prefix = f"{coil}_active"
        sham_prefix = f"{coil}_sham"

        active_abs_col = f"{coil}_active_abs_Vm"
        sham_abs_col = f"{coil}_sham_abs_Vm"

        out[active_abs_col] = field_magnitude(out, active_prefix)
        out[sham_abs_col] = field_magnitude(out, sham_prefix)

        active_peak = float(out[active_abs_col].max())
        active_floor = active_floor_frac * active_peak

        out[f"{coil}_active_norm_to_active_peak"] = out[active_abs_col] / active_peak
        out[f"{coil}_sham_norm_to_active_peak"] = out[sham_abs_col] / active_peak

        # Raw ratio is set to NaN where active field is below the floor.
        out[f"{coil}_sham_over_active_raw"] = np.where(
            out[active_abs_col] >= active_floor,
            out[sham_abs_col] / out[active_abs_col],
            np.nan,
        )

        # Regularized ratio is finite everywhere.
        out[f"{coil}_sham_over_active_regularized"] = (
            out[sham_abs_col] / np.maximum(out[active_abs_col], active_floor)
        )

        out[f"{coil}_active_floor_Vm"] = active_floor

    return out


def peak_location_row(df: pd.DataFrame, value_col: str) -> dict[str, float]:
    """
    Return value and xyz location for the maximum of value_col.
    """
    idx = pd.to_numeric(df[value_col], errors="coerce").idxmax()
    row = df.loc[idx]

    return {
        "value": float(row[value_col]),
        "x_cm": float(row["x_cm"]),
        "y_cm": float(row["y_cm"]),
        "z_cm": float(row["z_cm"]),
        "r_xy_cm": float(row["r_xy_cm"]),
    }


def central_cap_summary(
    df: pd.DataFrame,
    coil: str,
    cap_radius_cm: float,
) -> dict[str, float | int | str]:
    """
    Summarize active/sham ratio in a circular cap around the coil center.
    """
    sub = df[df["r_xy_cm"] <= cap_radius_cm].copy()

    active_col = f"{coil}_active_abs_Vm"
    sham_col = f"{coil}_sham_abs_Vm"
    raw_ratio_col = f"{coil}_sham_over_active_raw"
    reg_ratio_col = f"{coil}_sham_over_active_regularized"

    if sub.empty:
        return {
            "summary_type": "central_cap",
            "coil": coil,
            "cap_radius_cm": cap_radius_cm,
            "n_points": 0,
        }

    active = pd.to_numeric(sub[active_col], errors="coerce")
    sham = pd.to_numeric(sub[sham_col], errors="coerce")
    raw_ratio = pd.to_numeric(sub[raw_ratio_col], errors="coerce")
    reg_ratio = pd.to_numeric(sub[reg_ratio_col], errors="coerce")

    return {
        "summary_type": "central_cap",
        "coil": coil,
        "cap_radius_cm": float(cap_radius_cm),
        "n_points": int(len(sub)),
        "mean_active_abs_Vm": float(active.mean()),
        "mean_sham_abs_Vm": float(sham.mean()),
        "ratio_of_means": float(sham.mean() / active.mean()),
        "median_raw_ratio": float(raw_ratio.median()),
        "mean_raw_ratio": float(raw_ratio.mean()),
        "median_regularized_ratio": float(reg_ratio.median()),
        "mean_regularized_ratio": float(reg_ratio.mean()),
        "max_sham_norm_to_active_peak": float(sub[f"{coil}_sham_norm_to_active_peak"].max()),
    }


def build_kernel_summary(
    kernel_points: pd.DataFrame,
    cap_radii_cm: list[float] | None = None,
) -> pd.DataFrame:
    """
    Build summary table for empirical kernel validation.
    """
    if cap_radii_cm is None:
        cap_radii_cm = [0.25, 0.50, 1.00, 1.50, 2.00]

    rows: list[dict[str, object]] = []

    for coil in COILS:
        active_col = f"{coil}_active_abs_Vm"
        sham_col = f"{coil}_sham_abs_Vm"

        active_peak = peak_location_row(kernel_points, active_col)
        sham_peak = peak_location_row(kernel_points, sham_col)

        rows.append(
            {
                "summary_type": "global_peak",
                "coil": coil,
                "cap_radius_cm": np.nan,
                "n_points": int(len(kernel_points)),
                "active_peak_abs_Vm": active_peak["value"],
                "active_peak_x_cm": active_peak["x_cm"],
                "active_peak_y_cm": active_peak["y_cm"],
                "active_peak_z_cm": active_peak["z_cm"],
                "active_peak_r_xy_cm": active_peak["r_xy_cm"],
                "sham_peak_abs_Vm": sham_peak["value"],
                "sham_peak_x_cm": sham_peak["x_cm"],
                "sham_peak_y_cm": sham_peak["y_cm"],
                "sham_peak_z_cm": sham_peak["z_cm"],
                "sham_peak_r_xy_cm": sham_peak["r_xy_cm"],
                "global_sham_peak_over_active_peak": sham_peak["value"] / active_peak["value"],
            }
        )

        # Nearest central sampled point.
        idx_center = kernel_points["r_xy_cm"].idxmin()
        center = kernel_points.loc[idx_center]

        rows.append(
            {
                "summary_type": "nearest_center",
                "coil": coil,
                "cap_radius_cm": np.nan,
                "n_points": 1,
                "center_x_cm": float(center["x_cm"]),
                "center_y_cm": float(center["y_cm"]),
                "center_z_cm": float(center["z_cm"]),
                "center_r_xy_cm": float(center["r_xy_cm"]),
                "center_active_abs_Vm": float(center[active_col]),
                "center_sham_abs_Vm": float(center[sham_col]),
                "center_sham_over_active": float(center[sham_col] / center[active_col]),
                "center_sham_norm_to_active_peak": float(center[f"{coil}_sham_norm_to_active_peak"]),
            }
        )

        for radius in cap_radii_cm:
            rows.append(central_cap_summary(kernel_points, coil, radius))

    return pd.DataFrame(rows)


def plot_kernel_maps(
    kernel_points: pd.DataFrame,
    out_png: str | Path,
    out_transparent_png: str | Path,
) -> None:
    """
    Diagnostic maps of measured active/sham distributions and local ratios.
    """
    out_png = Path(out_png)
    out_transparent_png = Path(out_transparent_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ("magstim_active_norm_to_active_peak", "Magstim active\n|E| / active peak", 0, 1),
        ("magstim_sham_norm_to_active_peak", "Magstim sham\n|E| / active peak", 0, 0.35),
        ("magstim_sham_over_active_regularized", "Magstim local ratio\nsham / active", 0, 0.60),
        ("magventure_active_norm_to_active_peak", "MagVenture active\n|E| / active peak", 0, 1),
        ("magventure_sham_norm_to_active_peak", "MagVenture sham\n|E| / active peak", 0, 0.15),
        ("magventure_sham_over_active_regularized", "MagVenture local ratio\nsham / active", 0, 0.20),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 7.2),
        constrained_layout=True,
    )

    axes = axes.ravel()

    for ax, (col, title, vmin, vmax) in zip(axes, panels):
        values = pd.to_numeric(kernel_points[col], errors="coerce")

        sc = ax.scatter(
            kernel_points["x_cm"],
            kernel_points["y_cm"],
            c=values,
            s=18,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
        )

        ax.scatter(
            [0],
            [0],
            marker="+",
            s=80,
            color="white",
            linewidth=1.8,
        )
        ax.scatter(
            [0],
            [0],
            marker="+",
            s=80,
            color="black",
            linewidth=0.8,
        )

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Smith–Peterchev empirical E-field maps",
        fontsize=13,
        fontweight="bold",
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_transparent_png, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def build_magstim_empirical_kernel(
    clean_csv: str | Path,
    active_floor_frac: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build empirical kernel point table and summary table.
    """
    clean_csv = Path(clean_csv).expanduser().resolve()

    if not clean_csv.exists():
        raise FileNotFoundError(f"Clean Smith-Peterchev CSV not found: {clean_csv}")

    df = pd.read_csv(clean_csv)

    required_cols = [
        "x_cm",
        "y_cm",
        "z_cm",
        "magstim_active_Ex_Vm",
        "magstim_active_Ey_Vm",
        "magstim_active_Ez_Vm",
        "magstim_sham_Ex_Vm",
        "magstim_sham_Ey_Vm",
        "magstim_sham_Ez_Vm",
        "magventure_active_Ex_Vm",
        "magventure_active_Ey_Vm",
        "magventure_active_Ez_Vm",
        "magventure_sham_Ex_Vm",
        "magventure_sham_Ey_Vm",
        "magventure_sham_Ez_Vm",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Clean CSV missing required columns: {missing}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.shape[0] != 1000:
        raise ValueError(f"Expected 1000 measurement rows, got {df.shape[0]}")

    if df[required_cols].isna().any().any():
        bad = df[required_cols].isna().sum()
        raise ValueError(
            "Clean CSV contains NaNs:\n"
            + bad[bad > 0].to_string()
        )

    kernel_points = add_magnitude_and_kernel_columns(
        df,
        active_floor_frac=active_floor_frac,
    )

    summary = build_kernel_summary(kernel_points)

    return kernel_points, summary


def print_kernel_summary(summary: pd.DataFrame) -> None:
    """
    Print compact kernel summary.
    """
    print("\n[EMPIRICAL KERNEL SUMMARY]")

    global_peak = summary[summary["summary_type"] == "global_peak"].copy()
    cols_global = [
        "coil",
        "active_peak_abs_Vm",
        "sham_peak_abs_Vm",
        "global_sham_peak_over_active_peak",
        "active_peak_x_cm",
        "active_peak_y_cm",
        "sham_peak_x_cm",
        "sham_peak_y_cm",
        "sham_peak_r_xy_cm",
    ]

    print("\nGlobal peak ratios:")
    print(global_peak[cols_global].round(6).to_string(index=False))

    nearest = summary[summary["summary_type"] == "nearest_center"].copy()
    cols_center = [
        "coil",
        "center_r_xy_cm",
        "center_active_abs_Vm",
        "center_sham_abs_Vm",
        "center_sham_over_active",
        "center_sham_norm_to_active_peak",
    ]

    print("\nNearest-center ratios:")
    print(nearest[cols_center].round(6).to_string(index=False))

    caps = summary[summary["summary_type"] == "central_cap"].copy()
    cols_caps = [
        "coil",
        "cap_radius_cm",
        "n_points",
        "ratio_of_means",
        "median_regularized_ratio",
        "max_sham_norm_to_active_peak",
    ]

    print("\nCentral-cap ratios:")
    print(caps[cols_caps].round(6).to_string(index=False))
