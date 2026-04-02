from matplotlib import pyplot as plt
import numpy as np
import warnings 
from tail_calibration.types import TailCalibrationTuple, p
from matplotlib.figure import Figure
import pandas as pd

def plot_tail_calibration(
    ePIT_map: dict[p, TailCalibrationTuple],
    plots: list[str] = ['combined', 'severity', 'occurrence']
) -> Figure:
    """Plot tail calibration curves (combined, severity, and/or occurrence) across percentiles.

    Args:
        ePIT_map: Output of tail_calibration(); maps percentile -> TailCalibrationTuple.
        plots: Subset of ['combined', 'severity', 'occurrence'] to include as subplots.
    """
    t = list(ePIT_map.keys())
    u = np.linspace(0, 1, 1000)
    
    valid = ['combined', 'severity', 'occurrence']
    for plot_name in plots:
        if plot_name not in valid:
            raise ValueError(f"Invalid plot '{plot_name}'. Valid options: {valid}")

    n = len(plots)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, plot_name in zip(axes, plots):
        if plot_name == 'occurrence':
            occurrence_ratios = np.array([ePIT_map[p]['occurrence_ratio'] for p in t])
            ax.plot(t, occurrence_ratios, color='black')
            ax.axhline(y=1, color='black', linestyle='--')
            ax.set_xticks(t)
            ax.set_xticklabels([str(v) for v in t])
            ax.set_xlabel('Percentile (t)', size=14)
            ax.set_ylabel('Occurrence Ratio', size=14)

        elif plot_name == 'severity':
            for percentile in t:
                severity_values = [ePIT_map[percentile]['severity_func'](ui) for ui in u]
                ax.plot(u, severity_values, label=str(percentile))
            ax.plot(u, u, color='black', linestyle='--')
            ax.set_xlabel('u', size=14)
            ax.set_ylabel('Severity Ratio', size=14)
            ax.set_xlim(0, 1)
            ax.legend(title='Percentile (t)')

        elif plot_name == 'combined':
            for percentile in t:
                combined_values = [ePIT_map[percentile]['tail_calibration_func'](ui) for ui in u]
                ax.plot(u, combined_values, label=str(percentile))
            ax.plot(u, u, color='black', linestyle='--')
            ax.set_xlabel('u', size=14)
            ax.set_ylabel('Combined Ratio', size=14)
            ax.set_xlim(0, 1)
            ax.legend(title='Percentile (t)')

    plt.tight_layout()

    return fig


def plot_tail_calibration_by_id(
    by_id_map: dict[str, dict[p, TailCalibrationTuple]],
    percentile: p,
    u: np.ndarray = np.linspace(0, 1, 100),
    severity_metric: str = "sup",
) -> Figure:
    """Scatter plot of occurrence ratio vs severity distance, one point per unit ID.

    Args:
        by_id_map: Output of tail_calibration_by_id(); maps unit ID -> ePIT_map.
        percentile: The percentile threshold to plot.
        u: Grid of values in [0, 1] for evaluating the severity function.
        severity_metric: Distance metric for severity: 'sup' (max deviation) or 'l1' (integral).
    """
    ids = []
    occurrence_ratios = []
    severity_distances = []

    for id, ePIT_map in by_id_map.items():
        if percentile not in ePIT_map:
            warnings.warn(f"Percentile {percentile} not found for '{id}', skipping.")
            continue
        occ_ratio = ePIT_map[percentile]['occurrence_ratio']
        severity_values = np.array([ePIT_map[percentile]['severity_func'](ui) for ui in u])
        if severity_metric == "sup":
            distance = np.max(np.abs(severity_values - u))
        elif severity_metric == "l1":
            distance = np.trapezoid(np.abs(severity_values - u), u)
        else:
            raise ValueError(f"severity_metric must be 'sup' or 'l1', got '{severity_metric}'")
        ids.append(id)
        occurrence_ratios.append(occ_ratio)
        severity_distances.append(distance)

    metric_label = "Severity: Sup Distance" if severity_metric == "sup" else "Severity: L1 Distance"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(occurrence_ratios, severity_distances, color='red', s=20)
    for state, occ, dist in zip(ids, occurrence_ratios, severity_distances):
        ax.annotate(state, (occ, dist),
                    textcoords="offset points", xytext=(0, -12),
                    fontsize=6, alpha=0.7, ha='center',
                    fontweight='bold')
    ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Occurrence Ratio', size=16)
    ax.set_ylabel(metric_label, size=16)
    plt.tight_layout()
    return fig


def plot_tail_calibration_by_id_and_horizon(
    by_id_map_h1: dict[str, dict[p, TailCalibrationTuple]],
    by_id_map_h2: dict[str, dict[p, TailCalibrationTuple]],
    percentile: p,
    horizons: tuple[str, str] = ("H1", "H2"),
    u: np.ndarray = np.linspace(0, 1, 100),
    severity_metric: str = "sup",
) -> Figure:
    """Compare tail calibration across two horizons, with arrows connecting matched unit IDs.

    Args:
        by_id_map_h1: Output of tail_calibration_by_id() for the first horizon.
        by_id_map_h2: Output of tail_calibration_by_id() for the second horizon.
        percentile: The percentile threshold to plot.
        horizons: Display labels for the two horizons, shown in the legend.
        u: Grid of values in [0, 1] for evaluating the severity function.
        severity_metric: Distance metric for severity: 'sup' (max deviation) or 'l1' (integral).
    """

    def extract(by_id_map):
        result = {}
        for id, ePIT_map in by_id_map.items():
            if percentile not in ePIT_map:
                continue
            occ = ePIT_map[percentile]['occurrence_ratio']
            severity_values = np.array([ePIT_map[percentile]['severity_func'](ui) for ui in u])
            if severity_metric == "sup":
                dist = np.max(np.abs(severity_values - u))
            elif severity_metric == "l1":
                dist = np.trapezoid(np.abs(severity_values - u), u)
            else:
                raise ValueError(f"severity_metric must be 'sup' or 'l1', got '{severity_metric}'")
            result[id] = (occ, dist)
        return result

    metric_label = "Severity: Sup Distance" if severity_metric == "sup" else "Severity: L1 Distance"

    d1 = extract(by_id_map_h1)
    d2 = extract(by_id_map_h2)

    fig, ax = plt.subplots(figsize=(8, 6))

    for state, (occ, dist) in d1.items():
        ax.scatter(occ, dist, color='red', s=20, zorder=3)
        ax.annotate(state, (occ, dist), textcoords="offset points", xytext=(0, -12),
                    fontsize=6, alpha=0.7, ha='center', fontweight='bold', color='red')

    for state, (occ2, dist2) in d2.items():
        ax.scatter(occ2, dist2, color='blue', s=20, zorder=3)
        if state in d1:
            occ1, dist1 = d1[state]
            ax.annotate("", xy=(occ2, dist2), xytext=(occ1, dist1),
                        arrowprops=dict(arrowstyle="-|>", linestyle="dashed",
                                        color="gray", lw=0.8))

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label=horizons[0]),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label=horizons[1]),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Occurrence Ratio', size=16)
    ax.set_ylabel(metric_label, size=16)
    plt.tight_layout()
    return fig

def plot_tail_miscalibration_by_horizon(
    horizon_maps: dict[int, dict[p, TailCalibrationTuple]],
    percentile: p,
    plots: list[str] = ['combined', 'severity', 'occurrence'],
    u: np.ndarray = np.linspace(0, 1, 100),
) -> Figure:
    """Plot tail calibration curves across forecast horizons for a single percentile.

    Args:
        horizon_maps: Maps horizon integer -> ePIT_map (output of tail_calibration()).
        percentile: The percentile threshold to plot.
        plots: Subset of ['combined', 'severity', 'occurrence'] to include as subplots.
        u: Grid of values in [0, 1] for evaluating calibration functions.
    """

    valid = ['combined', 'severity', 'occurrence']
    for plot_name in plots:
        if plot_name not in valid:
            raise ValueError(f"Invalid plot '{plot_name}'. Valid options: {valid}")

    horizons = list(horizon_maps.keys())
    colors = plt.colormaps['Blues'](np.linspace(0.3, 0.9, len(horizons)))[::-1]

    n = len(plots)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, plot in zip(axes, plots):
        if plot == 'occurrence':
            occ_ratios = [horizon_maps[h][percentile]['occurrence_ratio'] for h in horizons]
            for i, (h, occ) in enumerate(zip(horizons, occ_ratios)):
                ax.plot(h, occ, marker='o', color=colors[i], markersize=8)
            ax.plot(horizons, occ_ratios, color='grey', linewidth=1, zorder=0)
            ax.axhline(y=1, color='black', linestyle='--')
            ax.set_xlabel('Horizon', size=14)
            ax.set_ylabel('Occurrence Ratio', size=14)
            ax.set_xticks(horizons)

        elif plot == 'severity':
            for i, horizon in enumerate(horizons):
                severity_values = [horizon_maps[horizon][percentile]['severity_func'](ui) for ui in u]
                ax.plot(u, severity_values, color=colors[i], label=str(horizon))
            ax.plot(u, u, color='black', linestyle='--')
            ax.set_xlabel('u', size=14)
            ax.set_ylabel('Severity Ratio', size=14)
            ax.set_xlim(0, 1)
            ax.legend(title='Horizon')

        elif plot == 'combined':
            for i, horizon in enumerate(horizons):
                combined_values = [horizon_maps[horizon][percentile]['tail_calibration_func'](ui) for ui in u]
                ax.plot(u, combined_values, color=colors[i], label=str(horizon))
            ax.plot(u, u, color='black', linestyle='--')
            ax.set_xlabel('u', size=14)
            ax.set_ylabel('Combined Ratio', size=14)
            ax.set_xlim(0, 1)
            ax.legend(title='Horizon')

    plt.tight_layout()
    return fig


from scipy import stats
def plot_wis_vs_tail(
    summary_df: pd.DataFrame,
    metric: str = "tail_sup_85",
) -> Figure:
    """Scatter plot of relative WIS vs a tail calibration distance metric, with Pearson correlation.

    Args:
        summary_df: Output of evaluate_models(); must contain 'rwis' and the chosen metric column.
        metric: Column name of the tail calibration metric to plot on the y-axis (e.g. 'tail_sup_85').
    """

    # axis label
    if "sup" in metric:
        ylabel = f"Combined Ratio: Sup Distance ({metric.split('_')[-1]}th Percentile)"
    else:
        ylabel = f"Combined Ratio: L1 Distance ({metric.split('_')[-1]}th Percentile)"

    x = summary_df["rwis"].to_numpy()
    y = summary_df[metric].to_numpy()
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    r, p_val = stats.pearsonr(x, y)  
    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "" # type: ignore
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, color="black", s=30, zorder=3)

    # line of best fit
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, m * x_line + b, color="gray", lw=0.8, linestyle="--", zorder=2)

    ax.text(0.05, 0.95, f"r = {r:.2f}{stars}", transform=ax.transAxes,
            fontsize=9, va="top", ha="left")

    for _, row in summary_df[mask].iterrows():
        ax.annotate(row["model_id"], (row["rwis"], row[metric]),
                    textcoords="offset points", xytext=(0, -8),
                    fontsize=7, ha="center", alpha=0.8)

    ax.axvline(x=1, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Relative WIS", size=14)
    ax.set_ylabel(ylabel, size=14)
    plt.tight_layout()
    return fig