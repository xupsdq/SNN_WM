from __future__ import annotations

import numpy as np

from src.plotting.experiments._common import main_for, read_bundle_csv
from src.plotting.common.colors import resolve_plot_color
from src.plotting.common.theme_tokens import GRID_ALPHA_SOFT, LINE_WIDTH_PRIMARY, MARKER_CIRCLE
from src.plotting.experiments._plot_builders import grouped_bar_figure, line_figure


def draw_update_recency_final_g_on_ax(ax, group, *, title: str | None = "Update recency final g") -> None:
    """Draw the update-recency grouped final-g summary on an existing axes."""
    group_col = "group_name" if "group_name" in group.columns else group.columns[0]
    value_col = "mean_final_g" if "mean_final_g" in group.columns else group.select_dtypes("number").columns[-1]
    summary = group.groupby(group_col, sort=True)[value_col].agg(["mean", "count", "std"]).reset_index()
    summary["sem"] = summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))
    labels = summary[group_col].astype(str).tolist()
    x = np.arange(len(labels), dtype=float)
    ax.bar(
        x,
        summary["mean"].to_numpy(dtype=float),
        yerr=summary["sem"].to_numpy(dtype=float),
        color=[resolve_plot_color(label, context=title) for label in labels],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.82,
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Final STSP g")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=GRID_ALPHA_SOFT)


def draw_anchor_prediction_model_comparison_on_ax(ax, prediction, *, title: str | None = "Anchor prediction model comparison") -> None:
    """Draw the anchor-prediction model comparison on an existing axes."""
    draw_update_recency_final_g_on_ax(
        ax,
        prediction.rename(columns={"seq_len": "group_name", "r2_update_plus_recency": "mean_final_g"}),
        title=title,
    )
    ax.set_ylabel("Prediction metric")


def draw_peak_function_spiking_on_ax(ax, probe_summary, *, title: str | None = "Peak function spiking") -> None:
    """Draw the peak-function spiking line plot on an existing axes."""
    x_col = "input_peak_overlap_fraction" if "input_peak_overlap_fraction" in probe_summary.columns else probe_summary.select_dtypes("number").columns[0]
    y_col = "spike_enrichment" if "spike_enrichment" in probe_summary.columns else probe_summary.select_dtypes("number").columns[-1]
    _draw_line_on_ax(ax, probe_summary, x=x_col, y=y_col, hue="probe_group" if "probe_group" in probe_summary.columns else None, title=title, ylabel="Spike enrichment")


def draw_overlap_conditioned_spike_effect_on_ax(ax, paired, *, title: str | None = "Overlap-conditioned spike effect") -> None:
    """Draw the overlap-conditioned spike effect line plot on an existing axes."""
    x_col = "input_peak_overlap_fraction" if "input_peak_overlap_fraction" in paired.columns else paired.select_dtypes("number").columns[0]
    _draw_line_on_ax(
        ax,
        paired,
        x=x_col,
        y="delta_spike_enrichment_intact_vs_flattened",
        hue="probe_group" if "probe_group" in paired.columns else None,
        title=title,
        ylabel="Delta spike enrichment",
    )


def _draw_line_on_ax(ax, df, *, x: str, y: str, hue: str | None, title: str | None, ylabel: str) -> None:
    if hue and hue in df.columns:
        for label, sub in df.groupby(hue, sort=True):
            sub = sub.sort_values(x)
            ax.plot(
                sub[x].to_numpy(),
                sub[y].to_numpy(dtype=float),
                marker=MARKER_CIRCLE,
                linewidth=LINE_WIDTH_PRIMARY,
                color=resolve_plot_color(label, hue, context=title),
                label=str(label),
            )
        ax.legend(frameon=False)
    else:
        plot_df = df.sort_values(x)
        ax.plot(plot_df[x].to_numpy(), plot_df[y].to_numpy(dtype=float), marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=resolve_plot_color(y, context=title))
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=GRID_ALPHA_SOFT)


def plot_bundle(input_dir):
    group = read_bundle_csv(input_dir, "layer1_recency_update_group_summary.csv")
    prediction = read_bundle_csv(input_dir, "layer1_anchor_prediction_summary.csv")
    probe_summary = read_bundle_csv(input_dir, "layer1_peak_function_probe_summary.csv")
    paired = read_bundle_csv(input_dir, "layer1_peak_function_paired_effects.csv")
    return {
        "fig6B_update_recency_final_g": grouped_bar_figure(
            group,
            group="group_name" if "group_name" in group.columns else group.columns[0],
            value="mean_final_g" if "mean_final_g" in group.columns else group.select_dtypes("number").columns[-1],
            title="Update recency final g",
            ylabel="Final STSP g",
        ),
        "fig6C_anchor_prediction_model_comparison": grouped_bar_figure(
            prediction,
            group="seq_len" if "seq_len" in prediction.columns else prediction.columns[0],
            value="r2_update_plus_recency" if "r2_update_plus_recency" in prediction.columns else prediction.select_dtypes("number").columns[-1],
            title="Anchor prediction model comparison",
            ylabel="Prediction metric",
        ),
        "fig6D_peak_function_spiking": line_figure(
            probe_summary,
            x="input_peak_overlap_fraction" if "input_peak_overlap_fraction" in probe_summary.columns else probe_summary.select_dtypes("number").columns[0],
            y="spike_enrichment" if "spike_enrichment" in probe_summary.columns else probe_summary.select_dtypes("number").columns[-1],
            hue="probe_group" if "probe_group" in probe_summary.columns else None,
            title="Peak function spiking",
            ylabel="Spike enrichment",
        ),
        "fig6E_overlap_conditioned_spike_effect": line_figure(
            paired,
            x="input_peak_overlap_fraction" if "input_peak_overlap_fraction" in paired.columns else paired.select_dtypes("number").columns[0],
            y="delta_spike_enrichment_intact_vs_flattened",
            hue="probe_group" if "probe_group" in paired.columns else None,
            title="Overlap-conditioned spike effect",
            ylabel="Delta spike enrichment",
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main_for("chunk_stsp_layer1_overlap_peak_formation", plot_bundle, title="Chunk STSP Layer1 Overlap Peak Formation"))
