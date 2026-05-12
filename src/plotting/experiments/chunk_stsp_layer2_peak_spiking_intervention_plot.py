from __future__ import annotations

from src.plotting.experiments._common import main_for, read_bundle_csv
from src.plotting.experiments._plot_builders import grouped_bar_figure, line_figure, scatter_figure


def plot_bundle(input_dir):
    trial = read_bundle_csv(input_dir, "layer2_peak_spiking_trial_summary.csv")
    paired = read_bundle_csv(input_dir, "layer2_peak_spiking_paired_effects.csv")
    stage = read_bundle_csv(input_dir, "layer2_peak_spiking_stage_summary.csv")
    overlap_bins = read_bundle_csv(input_dir, "overlap_bin_summary.csv")
    overlap_trials = read_bundle_csv(input_dir, "overlap_trial_effects.csv")

    valid_trial = trial[trial.get("valid", 1) == 1] if "valid" in trial.columns else trial
    return {
        "layer2_peak_spike_fraction_by_condition": grouped_bar_figure(
            valid_trial,
            group="condition",
            value="peak_spike_fraction",
            title="Layer 2 peak spike fraction",
            ylabel="Peak spike fraction",
        ),
        "layer2_spike_enrichment_by_condition": grouped_bar_figure(
            valid_trial,
            group="condition",
            value="spike_enrichment",
            title="Layer 2 spike enrichment",
            ylabel="Peak / nonpeak spike density",
        ),
        "layer2_stage_region_density": grouped_bar_figure(
            stage,
            group="condition",
            value="mean_spike_density",
            title="Layer 2 stage-region spike density",
            ylabel="Mean spike density",
        ),
        "layer2_memory_peak_fraction_effect": line_figure(
            paired,
            x="intervention_stage",
            y="delta_fraction_intact_minus_flatten",
            hue="seq_len" if "seq_len" in paired.columns else None,
            title="Memory peak fraction effect",
            ylabel="Intact - flatten",
        ),
        "layer2_memory_enrichment_effect": line_figure(
            paired,
            x="intervention_stage",
            y="delta_enrichment_intact_minus_flatten",
            hue="seq_len" if "seq_len" in paired.columns else None,
            title="Memory enrichment effect",
            ylabel="Intact - flatten",
        ),
        "overlap_bin_peak_fraction_effect": line_figure(
            overlap_bins,
            x="overlap_bin",
            y="mean_fraction_peakvalley_effect",
            title="Overlap bins: peak fraction effect",
            ylabel="Intact - flatten",
        ),
        "overlap_vs_peak_fraction_effect": scatter_figure(
            overlap_trials,
            x="overlap_fraction" if "overlap_fraction" in overlap_trials.columns else "overlap_enrichment",
            y="fraction_peakvalley_effect",
            title="Overlap vs peak fraction effect",
            ylabel="Intact - flatten",
            trend=True,
        ),
    }


if __name__ == "__main__":
    raise SystemExit(
        main_for(
            "chunk_stsp_layer2_peak_spiking_intervention",
            plot_bundle,
            title="Chunk STSP Layer2 Peak Spiking Intervention",
        )
    )
