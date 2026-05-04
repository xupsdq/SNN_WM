from __future__ import annotations

from src.plotting.experiments._common import main_for, read_bundle_csv
from src.plotting.experiments._plot_builders import grouped_bar_figure, line_figure, scatter_figure


def plot_bundle(input_dir):
    condition = read_bundle_csv(input_dir, "layer2_downstream_condition_summary.csv")
    paired = read_bundle_csv(input_dir, "layer2_downstream_paired_effects.csv")
    probe = read_bundle_csv(input_dir, "layer2_downstream_probe_selection.csv")
    figures = {
        "fig6E_layer2_update_enrichment_by_memory_state": line_figure(
            condition,
            x="l2_memory_condition",
            y="l2_update_enrichment",
            hue="l1_condition" if "l1_condition" in condition.columns else None,
            title="Layer2 update enrichment by memory state",
            ylabel="L2 update enrichment",
        ),
        "low_vs_high_overlap_layer2_update_enrichment": grouped_bar_figure(
            condition,
            group="probe_overlap_group" if "probe_overlap_group" in condition.columns else "l2_memory_condition",
            value="l2_update_enrichment",
            title="Low vs high overlap L2 update enrichment",
            ylabel="L2 update enrichment",
        ),
        "fig6E_layer2_update_difference_by_condition": line_figure(
            condition,
            x="l2_memory_condition",
            y="l2_update_difference",
            hue="l1_condition" if "l1_condition" in condition.columns else None,
            title="Layer2 update difference by condition",
            ylabel="L2 update difference",
        ),
        "l2_update_count_enrichment_by_condition": line_figure(
            condition,
            x="l2_memory_condition",
            y="l2_update_count_enrichment" if "l2_update_count_enrichment" in condition.columns else "l2_update_enrichment",
            hue="l1_condition" if "l1_condition" in condition.columns else None,
            title="L2 update count enrichment by condition",
            ylabel="Count enrichment",
        ),
        "l2_spike_enrichment_by_condition": line_figure(
            condition,
            x="l2_memory_condition",
            y="l2_spike_enrichment" if "l2_spike_enrichment" in condition.columns else "l2_update_enrichment",
            hue="l1_condition" if "l1_condition" in condition.columns else None,
            title="L2 spike enrichment by condition",
            ylabel="Spike enrichment",
        ),
        "probe_selection_overlap_distribution": grouped_bar_figure(
            probe,
            group="probe_overlap_group" if "probe_overlap_group" in probe.columns else "selected",
            value="probe_peak_overlap_fraction" if "probe_peak_overlap_fraction" in probe.columns else "overlap_fraction",
            title="Probe selection overlap distribution",
            ylabel="Overlap fraction",
        ),
    }
    if {"l1_spike_bias", "l2_update_bias"}.issubset(paired.columns):
        figures["fig6E_l1_spike_bias_predicts_l2_update_bias"] = scatter_figure(
            paired,
            x="l1_spike_bias",
            y="l2_update_bias",
            title="L1 spike bias predicts L2 update bias",
            trend=True,
        )
    else:
        figures["fig6E_l1_spike_bias_predicts_l2_update_bias"] = scatter_figure(
            paired,
            x=paired.select_dtypes("number").columns[0],
            y=paired.select_dtypes("number").columns[-1],
            title="L1 spike bias predicts L2 update bias",
            trend=True,
        )
    return figures


if __name__ == "__main__":
    raise SystemExit(main_for("chunk_stsp_layer2_downstream_integration", plot_bundle, title="Chunk STSP Layer2 Downstream Integration"))
