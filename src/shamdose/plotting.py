from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


MAIN_CONDITIONS = [
    "MagVenture_Sham0mm",
    "MagVenture_Sham0mm+electrodes",
    "Magstim_Sham0mm_kernel",
]

COND_LABELS = {
    "MagVenture_Sham0mm": "MV sham",
    "MagVenture_Sham0mm+electrodes": "MV sham + elec",
    "Magstim_Sham0mm_kernel": "Magstim sham",
}

COND_COLORS = {
    "MagVenture_Sham0mm": "#1F77B4",
    "MagVenture_Sham0mm+electrodes": "#D55E00",
    "Magstim_Sham0mm_kernel": "#009E73",
}

COND_LINESTYLES = {
    "MagVenture_Sham0mm": "-",
    "MagVenture_Sham0mm+electrodes": "--",
    "Magstim_Sham0mm_kernel": "-.",
}

COND_MARKERS = {
    "MagVenture_Sham0mm": "o",
    "MagVenture_Sham0mm+electrodes": "s",
    "Magstim_Sham0mm_kernel": "^",
}

COND_ZORDER = {
    "MagVenture_Sham0mm": 5,
    "MagVenture_Sham0mm+electrodes": 6,
    "Magstim_Sham0mm_kernel": 4,
}

PRIMARY_ENDPOINT = "PC_dPEAK_Hz"


def format_p(p: float | int | None) -> str:
    if p is None or pd.isna(p):
        return "P = NA"

    p = float(p)

    if p < 0.001:
        return "P < 0.001"

    return f"P = {p:.3f}"


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
    )


def get_tdcs_reference(
    df: pd.DataFrame,
    value_col: str,
    condition_col: str,
) -> tuple[float, float, float]:
    tdcs = df[df[condition_col].astype(str) == "tDCS"].copy()

    if tdcs.empty:
        raise ValueError("No tDCS rows found for reference.")

    values = pd.to_numeric(tdcs[value_col], errors="coerce").dropna().to_numpy(dtype=float)

    if values.size == 0:
        raise ValueError(f"No finite tDCS values found for {value_col}")

    q25, med, q75 = np.percentile(values, [25, 50, 75])

    return float(q25), float(med), float(q75)


def plot_subject_curves_with_median(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    subject_col: str,
    condition_col: str,
    title: str,
    y_label: str,
    x_label: str,
    tdcs_ref: tuple[float, float, float] | None = None,
) -> None:
    """
    Plot faint individual subject curves plus cohort median lines.

    Used for:
        Panel A: GM E-field vs intensity
        Panel B: PC Δpeak vs intensity
    """
    for condition in MAIN_CONDITIONS:
        sub = df[df[condition_col].astype(str) == condition].copy()
        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
        sub = sub.dropna(subset=[x_col, y_col]).copy()
        sub = sub[sub[x_col].between(0, 100)].copy()

        color = COND_COLORS[condition]
        label = COND_LABELS[condition]

        # Individual trajectories.
        for _, g in sub.groupby(subject_col):
            g = g.sort_values(x_col)
            ax.plot(
                g[x_col],
                g[y_col],
                color=color,
                alpha=0.13,
                linewidth=0.8,
                zorder=1,
            )

        # Median trajectory.
        med = (
            sub.groupby(x_col, as_index=False)[y_col]
            .median()
            .sort_values(x_col)
        )

        ax.plot(
            med[x_col],
            med[y_col],
            color=color,
            linewidth=2.9,
            linestyle=COND_LINESTYLES.get(condition, "-"),
            marker=COND_MARKERS.get(condition, "o"),
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            label=label,
            zorder=COND_ZORDER.get(condition, 5),
        )

    if tdcs_ref is not None:
        q25, med, q75 = tdcs_ref
        ax.axhspan(q25, q75, color="0.75", alpha=0.25, zorder=0)
        ax.axhline(
            med,
            color="0.25",
            linestyle="--",
            linewidth=1.6,
            label="active tDCS reference",
            zorder=2,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(-2, 102)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def plot_crossover_heatmap(
    fig,
    ax,
    probability: pd.DataFrame,
    cohort: str,
) -> None:
    """
    Plot crossover probability heatmap for the primary endpoint.
    """
    sub = probability[
        (probability["cohort"].astype(str) == cohort)
        & (probability["endpoint"].astype(str) == PRIMARY_ENDPOINT)
        & (probability["condition"].isin(MAIN_CONDITIONS))
    ].copy()

    if sub.empty:
        raise ValueError(f"No crossover probability rows found for cohort={cohort}")

    intensities = sorted(sub["intensity_MT"].astype(int).unique().tolist())

    mat = []
    row_labels = []

    for condition in MAIN_CONDITIONS:
        csub = sub[sub["condition"] == condition].copy()
        values = []

        for intensity in intensities:
            row = csub[csub["intensity_MT"].astype(int) == intensity]

            if len(row) != 1:
                raise ValueError(
                    f"Expected one probability row for {condition}, {intensity}, found {len(row)}"
                )

            values.append(float(row["pct_crossed"].iloc[0]))

        mat.append(values)
        row_labels.append(COND_LABELS[condition])

    mat = np.asarray(mat, dtype=float)

    im = ax.imshow(
        mat,
        aspect="auto",
        vmin=0,
        vmax=100,
        cmap="YlOrRd",
    )

    ax.set_xlabel("Intensity (%MT) →")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(len(intensities)))
    ax.set_xticklabels(intensities)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Annotate cells.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt_color = "white" if val >= 65 else "black"
            ax.text(
                j,
                i,
                f"{val:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color=txt_color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Subjects with sham ≥ active tDCS (%)")


def plot_anatomy_panel(
    ax,
    crossover: pd.DataFrame,
    spearman: pd.DataFrame,
    survival_main: pd.DataFrame,
    cohort: str,
    condition: str,
    panel_title: str,
    show_ylabel: bool = True,
) -> None:
    """
    Plot GM depth vs crossover intensity for one condition.
    """
    sub = crossover[
        (crossover["cohort"].astype(str) == cohort)
        & (crossover["condition"].astype(str) == condition)
        & (crossover["endpoint"].astype(str) == PRIMARY_ENDPOINT)
    ].copy()

    if sub.empty:
        raise ValueError(f"No crossover rows for {cohort}, {condition}")

    x = pd.to_numeric(sub["gm_depth_from_scalp_mm"], errors="coerce")
    y = pd.to_numeric(sub["I_cross_MT"], errors="coerce")

    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    color = COND_COLORS[condition]

    rng = np.random.default_rng(7)
    y_plot = y + rng.uniform(-0.85, 0.85, size=len(y))

    ax.scatter(
        x,
        y_plot,
        s=36,
        color=color,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # Visual guide only; stats are Spearman and discrete-time model.
    if len(x) >= 3 and x.nunique() > 1:
        coef = np.polyfit(x, y, 1)
        x_line = np.linspace(float(x.min()), float(x.max()), 100)
        y_line = coef[0] * x_line + coef[1]
        ax.plot(
            x_line,
            y_line,
            color="0.2",
            linewidth=1.5,
            linestyle="-",
            alpha=0.8,
            zorder=2,
        )

    sp = spearman[
        (spearman["cohort"].astype(str) == cohort)
        & (spearman["condition"].astype(str) == condition)
        & (spearman["endpoint"].astype(str) == PRIMARY_ENDPOINT)
        & (spearman["anatomy_variable"].astype(str) == "gm_depth_from_scalp_mm")
    ].copy()

    annotation_lines = []

    if not sp.empty:
        sp = sp.iloc[0]
        rho = float(sp["rho_spearman"])
        p = sp.get("p_holm_primary_set", np.nan)
        annotation_lines.append(f"Spearman ρ = {rho:.2f}")
        annotation_lines.append(f"Holm {format_p(p)}")

    ax.text(
        0.04,
        0.96,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.85", alpha=0.9),
    )

    ax.text(
        0.96,
        0.08,
        panel_title,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=8.5,
        fontweight="bold",
        color="0.25",
    )
    ax.set_xlabel("Scalp–GM distance (mm) →")

    if show_ylabel:
        ax.set_ylabel("Intensity where\nsham ≥ active tDCS (%MT)")
    else:
        ax.set_ylabel("")

    ax.set_ylim(5, 105)
    ax.set_yticks(np.arange(10, 101, 10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_hc_four_panel_figure(
    tables_dir: str | Path,
    out_dir: str | Path,
    cohort: str = "HC",
) -> tuple[Path, Path]:
    """
    Make the first four-panel main-figure prototype for one cohort.
    """
    tables_dir = Path(tables_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    step1 = pd.read_csv(tables_dir / f"step1_efield_{cohort}.csv")
    step3 = pd.read_csv(tables_dir / f"step3_meanfield_{cohort}.csv")
    crossover = pd.read_csv(tables_dir / "subject_level_crossover.csv")
    probability = pd.read_csv(tables_dir / "crossover_probability_by_intensity.csv")
    spearman = pd.read_csv(tables_dir / "anatomy_spearman_primary.csv")
    survival_main = pd.read_csv(tables_dir / "report_main_adjusted_anatomy_terms.csv")

    fig = plt.figure(figsize=(16.5, 10))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.00, 1.35],
        height_ratios=[1.0, 1.0],
        wspace=0.30,
        hspace=0.34,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])

    inner_d = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs[1, 1],
        wspace=0.32,
    )
    ax_d1 = fig.add_subplot(inner_d[0, 0])
    ax_d2 = fig.add_subplot(inner_d[0, 1], sharey=ax_d1)
    ax_d3 = fig.add_subplot(inner_d[0, 2], sharey=ax_d1)

    # Panel A: GM E-field.
    e_col = "roi_graymatter_mean_|E|_V/m"
    tdcs_efield = get_tdcs_reference(
        step1,
        value_col=e_col,
        condition_col="condition",
    )

    plot_subject_curves_with_median(
        ax=ax_a,
        df=step1,
        x_col="intensity (%MT)",
        y_col=e_col,
        subject_col="subject_id",
        condition_col="condition",
        title="",
        y_label="GM exposure (V/m) ↑",
        x_label="Intensity (%MT) →",
        tdcs_ref=tdcs_efield,
    )
    add_panel_label(ax_a, "A")

    # Panel B: PC Δpeak.
    tdcs_pc = get_tdcs_reference(
        step3,
        value_col="PC_dPEAK_Hz",
        condition_col="CONDITION",
    )

    plot_subject_curves_with_median(
        ax=ax_b,
        df=step3,
        x_col="INTENSITY_MT",
        y_col="PC_dPEAK_Hz",
        subject_col="SUBJECT_ID",
        condition_col="CONDITION",
        title="",
        y_label="Acute PC response (ΔHz) ↑",
        x_label="Intensity (%MT) →",
        tdcs_ref=tdcs_pc,
    )
    add_panel_label(ax_b, "B")

    # Panel C: heatmap.
    plot_crossover_heatmap(
        fig=fig,
        ax=ax_c,
        probability=probability,
        cohort=cohort,
    )
    add_panel_label(ax_c, "C")

    # Panel D: anatomy split mini-panels.
    plot_anatomy_panel(
        ax=ax_d1,
        crossover=crossover,
        spearman=spearman,
        survival_main=survival_main,
        cohort=cohort,
        condition="MagVenture_Sham0mm",
        panel_title="MV sham",
        show_ylabel=True,
    )
    add_panel_label(ax_d1, "D")

    plot_anatomy_panel(
        ax=ax_d2,
        crossover=crossover,
        spearman=spearman,
        survival_main=survival_main,
        cohort=cohort,
        condition="MagVenture_Sham0mm+electrodes",
        panel_title="MV sham + elec",
        show_ylabel=False,
    )

    plot_anatomy_panel(
        ax=ax_d3,
        crossover=crossover,
        spearman=spearman,
        survival_main=survival_main,
        cohort=cohort,
        condition="Magstim_Sham0mm_kernel",
        panel_title="Magstim sham",
        show_ylabel=False,
    )


    png_path = out_dir / f"fig_main_{cohort}_four_panel_sham_dose.png"
    transparent_path = out_dir / f"fig_main_{cohort}_four_panel_sham_dose_transparent.png"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(transparent_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return png_path, transparent_path
