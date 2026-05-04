from __future__ import annotations

from src.plotting.experiments._common import main_for, read_bundle_csv
from src.plotting.experiments._plot_builders import grouped_bar_figure, line_figure


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
