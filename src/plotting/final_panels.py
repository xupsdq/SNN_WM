from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import SubplotSpec
from matplotlib.patches import Circle, Rectangle

from figure_utils_common import (
    COLOR_DYNAMIC,
    COLOR_NOISE,
    COLOR_SAMPLE_ALIGNED,
    COLOR_STATIC,
    PUBLICATION_ANNOTATION_FONT_SIZE,
    get_paper_color_map,
)
from src.plotting.common.publication_panels import (
    plot_distractor_sensory_readout_on_axes,
    plot_input_similarity_summary_on_axes,
    plot_overview_accuracy_panel,
    plot_sample_readout_survival_on_axis,
)
from src.plotting.fig1_baseline_processing import plot_class_recall_summary, plot_confusion_matrix_main
from src.plotting.fig2_silent_memory_effect import plot_representative_activity_panel
from src.plotting.fig3_ux_mechanism import plot_engram_decode_vs_delay
from src.plotting.fig4_external_input_interrogation import plot_ping_selectivity_overlay
from src.plotting.fig5_latent_substrate_dissociation import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    plot_condition_summary_panel,
    plot_confidence_panel,
    plot_error_destination_panel,
    plot_triplet_summary_panel,
)


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.14, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def infer_phase_slices(df_population_rate: pd.DataFrame) -> dict[str, tuple[int, int]]:
    phase_slices: dict[str, tuple[int, int]] = {}
    for phase_name in ["sample", "delay", "probe"]:
        sub = df_population_rate[df_population_rate["phase"] == phase_name].copy()
        if sub.empty:
            continue
        t_steps = sub["t_step"].to_numpy(dtype=int)
        phase_slices[phase_name] = (int(t_steps.min()), int(t_steps.max()) + 1)
    return phase_slices


def infer_layer_neuron_counts(df_raster_points: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for layer_name, sub in df_raster_points.groupby("layer", sort=False):
        counts[str(layer_name)] = int(sub["neuron_index"].to_numpy(dtype=int).max()) + 1
    return counts


def infer_dt_ms(df_population_rate: pd.DataFrame) -> float:
    ordered = df_population_rate.sort_values("t_step", kind="stable").reset_index(drop=True)
    if len(ordered) < 2:
        return 1.0
    time_ms = ordered["time_ms"].to_numpy(dtype=float)
    t_steps = ordered["t_step"].to_numpy(dtype=float)
    delta_t = t_steps[1:] - t_steps[:-1]
    delta_ms = time_ms[1:] - time_ms[:-1]
    mask = delta_t > 0
    if not mask.any():
        return 1.0
    return float((delta_ms[mask] / delta_t[mask]).mean())


def plot_baseline_panels(
    ax_confusion: plt.Axes,
    ax_recall: plt.Axes,
    df_confusion: pd.DataFrame,
    df_recall: pd.DataFrame,
    *,
    n_trials: int,
    num_classes: int,
) -> None:
    plot_confusion_matrix_main(ax_confusion, df_confusion=df_confusion, num_classes=num_classes)
    plot_class_recall_summary(ax_recall, df_recall=df_recall, n_trials=n_trials)


def plot_representative_silent_memory_panel(
    fig: plt.Figure,
    subplot_spec: SubplotSpec,
    df_raster_points: pd.DataFrame,
    df_population_rate: pd.DataFrame,
) -> tuple[list[plt.Axes], plt.Axes]:
    representative_meta = {
        "phase_slices": infer_phase_slices(df_population_rate),
        "layer_neuron_counts": infer_layer_neuron_counts(df_raster_points),
    }
    dt_ms = infer_dt_ms(df_population_rate)
    return plot_representative_activity_panel(
        fig=fig,
        subplot_spec=subplot_spec,
        df_raster_points=df_raster_points,
        df_population_rate=df_population_rate,
        representative_meta=representative_meta,
        dt_ms=dt_ms,
    )


def plot_engram_panel(ax: plt.Axes, df_engram_decode: pd.DataFrame) -> None:
    plot_engram_decode_vs_delay(ax, df_engram_decode, num_classes=10, color_map=get_paper_color_map())


def plot_triplet_summary_only(ax: plt.Axes, df_summary: pd.DataFrame) -> None:
    plot_triplet_summary_panel(ax, df_summary)


def plot_triplet_confidence_only(ax: plt.Axes, df_confidence: pd.DataFrame, df_summary: pd.DataFrame) -> None:
    plot_confidence_panel(ax, df_confidence, df_summary)


def plot_causal_condition_panel(ax: plt.Axes, df_condition: pd.DataFrame) -> None:
    plot_condition_summary_panel(ax, df_condition)


def plot_causal_bias_only_panel(ax: plt.Axes, df_condition: pd.DataFrame) -> None:
    ordered_conditions = [condition for condition in CONDITION_LABELS if condition in set(df_condition["condition"].tolist())]
    sub = df_condition.set_index("condition").reindex(ordered_conditions).reset_index()
    x = np.arange(len(ordered_conditions))
    values = sub["sample_related_bias"].to_numpy(dtype=float)
    bars = ax.bar(
        x,
        values,
        color=[CONDITION_COLORS.get(condition, "#7F7F7F") for condition in ordered_conditions],
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.8,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=PUBLICATION_ANNOTATION_FONT_SIZE,
        )
    upper = max(8.0, float(np.max(values)) * 1.22 + 1.0) if len(values) else 8.0
    ax.set_xticks(x, [CONDITION_LABELS[condition] for condition in ordered_conditions], rotation=20, ha="right")
    ax.set_ylabel("Sample-related bias (%)")
    ax.set_ylim(0.0, upper)
    ax.set_title("Sample-related bias")


def plot_causal_error_destination_panel(
    ax: plt.Axes,
    df_error: pd.DataFrame,
    *,
    legend_outside: bool = False,
) -> None:
    plot_error_destination_panel(ax, df_error)
    if legend_outside:
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.16),
            frameon=False,
            fontsize=8,
            ncol=4,
            columnspacing=0.9,
            handletextpad=0.5,
        )


def plot_ux_shuffle_accuracy_panel(ax: plt.Axes, df_metrics_condition: pd.DataFrame) -> None:
    plot_overview_accuracy_panel(ax, df_metrics_condition)


def plot_external_input_overlay_panel(
    ax: plt.Axes,
    metrics_ping_summary: pd.DataFrame,
    metrics_ping_selectivity: pd.DataFrame,
) -> None:
    plot_ping_selectivity_overlay(ax, metrics_ping_summary=metrics_ping_summary, metrics_ping_selectivity=metrics_ping_selectivity)


def plot_ping_first_fire_accuracy_panel(
    ax: plt.Axes,
    df_condition_summary: pd.DataFrame,
    df_cross_seed_tests: pd.DataFrame,
) -> None:
    order = [
        "A_stsp_on_ping",
        "B_stsp_on_no_ping",
        "C_stsp_off_ping",
        "D_stsp_on_ping_shuffle_ux",
        "E_ping_only_sanity",
    ]
    labels = ["STSP+Ping", "No ping", "Static+Ping", "Ping+shuffle", "Ping only"]
    color_map = {
        "A_stsp_on_ping": COLOR_DYNAMIC,
        "B_stsp_on_no_ping": COLOR_STATIC,
        "C_stsp_off_ping": "#7F7F7F",
        "D_stsp_on_ping_shuffle_ux": COLOR_SAMPLE_ALIGNED,
        "E_ping_only_sanity": COLOR_NOISE,
    }
    summary = df_condition_summary.set_index("condition").reindex(order).reset_index()
    tests = df_cross_seed_tests[df_cross_seed_tests["metric"] == "first_fire_ping_acc"].copy()
    stars_by_condition: dict[str, str] = {}
    for _, row in tests.iterrows():
        condition_b = str(row["condition_b"])
        p_value = float(row["p_one_sided_a_gt_b"])
        stars_by_condition[condition_b] = _p_to_stars(p_value)
    y = 100.0 * summary["first_fire_ping_acc_mean"].to_numpy(dtype=float)
    x = np.arange(len(summary))
    bars = ax.bar(
        x,
        y,
        color=[color_map[str(condition)] for condition in summary["condition"].tolist()],
        edgecolor="#222222",
        linewidth=0.8,
    )
    for bar, value, condition in zip(bars, y, summary["condition"].tolist()):
        label_text = f"{value:.1f}%"
        if condition != "A_stsp_on_ping":
            label_text += stars_by_condition.get(str(condition), "")
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 1.5,
            label_text,
            ha="center",
            va="bottom",
            fontsize=PUBLICATION_ANNOTATION_FONT_SIZE,
        )

    ax.axhline(10.0, color=COLOR_NOISE, linestyle="--", linewidth=1.0)
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylabel("First-fire ping accuracy (%)")
    ax.set_ylim(0.0, max(35.0, float(np.max(y)) + 10.0))
    ax.set_title("Ping first-fire readout")


def plot_distractor_sensory_panel(
    fig: plt.Figure,
    subplot_spec: SubplotSpec,
    metrics_dist: pd.DataFrame,
) -> tuple[plt.Axes, plt.Axes]:
    grid = subplot_spec.subgridspec(1, 2, wspace=0.32)
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(2)]
    plot_distractor_sensory_readout_on_axes(axes, metrics_dist)
    return axes[0], axes[1]


def plot_distractor_accuracy_only_panel(
    ax: plt.Axes,
    metrics_dist: pd.DataFrame,
    *,
    star_pvalues: bool = False,
) -> None:
    row = metrics_dist.iloc[0]
    labels = ["Dynamic", "Static"]
    values = np.array(
        [
            float(row["acc_distractor_dynamic"]),
            float(row["acc_distractor_static"]),
        ],
        dtype=float,
    )
    lower = np.array(
        [
            float(row["acc_distractor_dynamic_ci95_lower"]),
            float(row["acc_distractor_static_ci95_lower"]),
        ],
        dtype=float,
    )
    upper = np.array(
        [
            float(row["acc_distractor_dynamic_ci95_upper"]),
            float(row["acc_distractor_static_ci95_upper"]),
        ],
        dtype=float,
    )
    yerr = np.vstack([values - lower, upper - values])
    bars = ax.bar(labels, values, color=[COLOR_DYNAMIC, COLOR_STATIC], edgecolor="black", alpha=0.9, yerr=yerr, capsize=5)
    for idx, (bar, value) in enumerate(zip(bars, values)):
        label_text = f"{value:.1f}%"
        if idx == 0 and star_pvalues:
            label_text += _p_to_stars(float(row["acc_distractor_gap_p_two_sided_ne0"]))
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 1.2,
            label_text,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0.0, 100.0)
    ax.set_title("Distractor classification")


def plot_sample_survival_panel(
    ax: plt.Axes,
    metrics_survival: pd.DataFrame,
    *,
    star_pvalues: bool = False,
) -> None:
    if not star_pvalues:
        plot_sample_readout_survival_on_axis(ax, metrics_survival)
        return
    row = metrics_survival.iloc[0]
    clean_sample = float(row["clean_sample_bias"])
    distracted_sample = float(row["distracted_sample_bias"])
    distracted_noise = float(row["distracted_noise_bias"])
    ci_clean = (float(row["clean_sample_bias_ci95_lower"]), float(row["clean_sample_bias_ci95_upper"]))
    ci_distracted_sample = (
        float(row["distracted_sample_bias_ci95_lower"]),
        float(row["distracted_sample_bias_ci95_upper"]),
    )
    ci_distracted_noise = (
        float(row["distracted_noise_bias_ci95_lower"]),
        float(row["distracted_noise_bias_ci95_upper"]),
    )
    values = [clean_sample, distracted_sample, distracted_noise]
    yerr = np.array(
        [
            [clean_sample - ci_clean[0], distracted_sample - ci_distracted_sample[0], distracted_noise - ci_distracted_noise[0]],
            [ci_clean[1] - clean_sample, ci_distracted_sample[1] - distracted_sample, ci_distracted_noise[1] - distracted_noise],
        ]
    )
    x = np.arange(3)
    upper_error = yerr[1]
    ax.bar(
        x,
        values,
        color=["#1f77b4", COLOR_DYNAMIC, COLOR_STATIC],
        edgecolor="black",
        alpha=0.9,
        yerr=yerr,
        capsize=6,
    )
    for idx, value in enumerate(values):
        ax.text(
            idx,
            value + upper_error[idx] + 0.012,
            f"{value * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    retention_pct = float(row["sample_bias_retention_pct"])
    ax.text(
        0.05,
        0.96,
        f"{retention_pct:.2f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.96,
        f"Distracted Sample Bias > Noise Bias\n({_format_p_stars(float(row['p_one_sided_sample_gt_noise_distracted']))})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9),
    )
    ax.set_xticks(x, ["Clean\nSample Bias", "Distracted\nSample Bias", "Distracted\nNoise Bias"])
    ax.set_ylabel("Probability in Probe Error Trials")
    ax.set_ylim(0.0, max(values) + 0.15)
    ax.set_title("Sample memory after distractor washout")


def plot_sample_survival_panel_legacy(ax: plt.Axes, metrics_survival: pd.DataFrame) -> None:
    plot_sample_readout_survival_on_axis(ax, metrics_survival)


def plot_input_similarity_panel(
    fig: plt.Figure,
    subplot_spec: SubplotSpec,
    df_summary: pd.DataFrame,
    *,
    shared_legend: bool = False,
    remove_titles: bool = False,
) -> list[plt.Axes]:
    grid = subplot_spec.subgridspec(2, 2, wspace=0.28, hspace=0.36)
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    plot_input_similarity_summary_on_axes(axes, df_summary, source_name="pixel")
    legend_label_map = {"dynamic": "Dynamic", "static_frozen": "Static"}
    handles: list[object] = []
    labels: list[str] = []
    if axes:
        handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        if remove_titles:
            ax.set_title("")
    if shared_legend and handles:
        top_y = max(ax.get_position().y1 for ax in axes)
        fig.legend(
            handles,
            [legend_label_map.get(label, label) for label in labels],
            loc="lower center",
            bbox_to_anchor=(0.5, top_y + 0.02),
            ncol=len(labels),
            frameon=False,
        )
    return axes


def plot_input_similarity_dynamic_only_panel(
    fig: plt.Figure,
    subplot_spec: SubplotSpec,
    df_summary: pd.DataFrame,
) -> list[plt.Axes]:
    metric_df = df_summary.drop_duplicates(subset=["bucket_name", "stsp_mode"]).copy()
    metric_df = metric_df[metric_df["stsp_mode"] == "dynamic"].copy()
    order = [x for x in ["low", "medium", "high"] if x in metric_df["bucket_name"].tolist()]
    metric_specs = [
        ("probe_accuracy", "Probe accuracy (%)"),
        ("distractor_accuracy", "Distractor accuracy (%)"),
        ("sample_bias_retention_pct", "Sample-bias retention (%)"),
        ("sample_bias_minus_noise", "Sample bias - noise bias (pp)"),
    ]
    bucket_colors = {"low": "#4C78A8", "medium": "#F58518", "high": "#54A24B"}
    grid = subplot_spec.subgridspec(2, 2, wspace=0.28, hspace=0.36)
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    sub = metric_df.set_index("bucket_name").reindex(order).reset_index()
    p_values = sub["p_one_sided_gt_chance"].to_numpy(dtype=float) if "p_one_sided_gt_chance" in sub.columns else np.full(len(sub), np.nan)
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        values = sub[metric].to_numpy(dtype=float)
        x = np.arange(len(sub))
        colors = [bucket_colors.get(str(bucket), COLOR_DYNAMIC) for bucket in sub["bucket_name"].tolist()]
        bars = ax.bar(x, values, color=colors, edgecolor="black", alpha=0.9)
        for bar, value, p_value in zip(bars, values, p_values):
            stars = _p_to_stars(p_value) if np.isfinite(p_value) else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + (1.2 if metric != "sample_bias_minus_noise" else 0.6),
                f"{value:.1f}{stars}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_xticks(x, [bucket.capitalize() for bucket in sub["bucket_name"].tolist()])
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title("")
        if metric != "sample_bias_minus_noise":
            ax.set_ylim(0.0, max(100.0, float(np.max(values)) + 12.0))
        else:
            lower = min(0.0, float(np.min(values)) - 1.0)
            upper = max(5.0, float(np.max(values)) + 1.8)
            ax.set_ylim(lower, upper)
    return axes


def draw_example_dms_panel(ax: plt.Axes) -> None:
    _draw_task_example(ax, title="DMS", phases=[("sample", "7", "200ms"), ("delay", None, "500ms"), ("probe", "2", "100ms")])


def draw_example_shuffle_panel(ax: plt.Axes) -> None:
    _draw_task_example(
        ax,
        title="Shuffle experiment",
        phases=[("sample", "7", "200ms"), ("delay", None, "500ms"), ("probe", "2", "100ms")],
        divider_label="shuffle",
    )


def draw_example_ping_panel(ax: plt.Axes) -> None:
    _draw_task_example(
        ax,
        title="Ping Task",
        phases=[
            ("sample", "7", "200ms"),
            ("delay", None, "500ms"),
            ("Ping", "dot", "30ms"),
            ("delay", None, "380ms"),
            ("probe", "2", "100ms"),
        ],
    )


def draw_example_distractor_panel(ax: plt.Axes) -> None:
    _draw_task_example(
        ax,
        title="Distractor Task",
        phases=[
            ("sample", "7", "200ms"),
            ("delay", None, "500ms"),
            ("distractor", "1", "120ms"),
            ("delay", None, "380ms"),
            ("probe", "2", "100ms"),
        ],
    )


def _draw_task_example(
    ax: plt.Axes,
    *,
    title: str,
    phases: Sequence[tuple[str, str | None, str]],
    divider_label: str | None = None,
) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.text(0.5, 0.92, title, ha="center", va="center", fontsize=18, fontweight="semibold")
    n_boxes = len(phases)
    left_margin = 0.06
    right_margin = 0.06
    gap = 0.02
    box_width = (1.0 - left_margin - right_margin - gap * (n_boxes - 1)) / n_boxes
    box_height = 0.42
    y0 = 0.38
    for idx, (phase_name, content, duration_label) in enumerate(phases):
        x0 = left_margin + idx * (box_width + gap)
        ax.add_patch(Rectangle((x0, y0), box_width, box_height, facecolor="black", edgecolor="#888888", linewidth=1.8))
        ax.text(x0 + box_width / 2, y0 + box_height + 0.06, phase_name, ha="center", va="center", fontsize=14, fontweight="semibold")
        ax.text(x0 + box_width / 2, y0 - 0.07, duration_label, ha="center", va="center", fontsize=13, fontweight="semibold")
        if content is not None:
            _draw_box_content(ax, x0 + box_width / 2, y0 + box_height / 2, content)
        if divider_label is not None and phase_name == "delay" and idx == 1:
            divider_x = x0 + box_width + gap / 2
            ax.plot([divider_x, divider_x], [y0 + 0.02, y0 + box_height - 0.02], color="black", linestyle="--", linewidth=3.0)
            ax.text(divider_x, y0 - 0.06, divider_label, ha="center", va="center", fontsize=12, fontweight="semibold")


def _draw_box_content(ax: plt.Axes, x: float, y: float, content: str) -> None:
    if content == "dot":
        ax.add_patch(Circle((x, y), 0.008, facecolor="white", edgecolor="white"))
        return
    ax.text(x, y, content, ha="center", va="center", fontsize=46, fontweight="bold", color="white")


def _p_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _format_p_stars(p_value: float) -> str:
    stars = _p_to_stars(p_value)
    return stars if stars else "n.s."
