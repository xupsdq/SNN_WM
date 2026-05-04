from __future__ import annotations

from src.plotting.experiments._common import main_for
from src.plotting.experiments._plot_builders import bar_figure, grouped_bar_figure, line_figure, scatter_figure, sem
from src.plotting.experiments._common import read_bundle_csv


def plot_bundle(input_dir):
    natural = read_bundle_csv(input_dir, "layer2_peak_boost_natural_contrast.csv")
    selection = read_bundle_csv(input_dir, "layer2_peak_boost_target_selection.csv")
    trial = read_bundle_csv(input_dir, "layer2_peak_boost_trial_summary.csv")
    effects = read_bundle_csv(input_dir, "layer2_peak_boost_effects.csv")
    dose = read_bundle_csv(input_dir, "layer2_peak_boost_dose_response.csv")
    valid_nat = natural[natural.get("valid", 1) == 1] if "valid" in natural.columns else natural
    peak_boost = trial[(trial.get("valid", 1) == 1) & (trial["boost_type"].astype(str) == "peak_boost")] if "boost_type" in trial.columns else trial
    boost_effects = effects[effects["boost_type"].astype(str) == "peak_boost"] if "boost_type" in effects.columns else effects
    figures = {
        "natural_peak_contrast": bar_figure(
            ["peak", "nonpeak"],
            [float(valid_nat["mean_g_peak"].mean()), float(valid_nat["mean_g_nonpeak"].mean())],
            yerr=[sem(valid_nat["mean_g_peak"]), sem(valid_nat["mean_g_nonpeak"])],
            color_keys=["peak_region", "nonpeak_region"],
            title="Natural peak contrast",
            ylabel="mean Layer 2 STSP g",
            rotation=0,
        ),
        "selected_target_overlap_distribution": scatter_figure(
            selection.reset_index(),
            x="index",
            y="overlap_fraction",
            hue="target_overlap_group" if "target_overlap_group" in selection.columns else None,
            title="Selected target overlap",
            xlabel="Selected target",
            ylabel="Overlap fraction",
        ),
        "peak_fraction_dose_response": line_figure(peak_boost.groupby("boost_level", as_index=False)["peak_spike_fraction"].mean(), x="boost_level", y="peak_spike_fraction", title="Peak fraction dose response", ylabel="peak spike fraction"),
        "spike_enrichment_dose_response": line_figure(peak_boost.groupby("boost_level", as_index=False)["spike_enrichment"].mean(), x="boost_level", y="spike_enrichment", title="Spike enrichment dose response", ylabel="spike enrichment"),
        "boost_effect_vs_flatten": line_figure(boost_effects.groupby("boost_level", as_index=False)["delta_fraction_vs_flatten"].mean(), x="boost_level", y="delta_fraction_vs_flatten", title="Boost effect vs flatten", ylabel="delta fraction vs flatten"),
        "dose_response_slope_distribution": bar_figure(
            ["fraction", "enrichment"],
            [float(dose["fraction_slope_per_lambda"].mean()), float(dose["enrichment_slope_per_lambda"].mean())],
            yerr=[sem(dose["fraction_slope_per_lambda"]), sem(dose["enrichment_slope_per_lambda"])],
            color_keys=["peak_boosted", "intact_final"],
            title="Dose response slope",
            ylabel="slope per lambda",
            rotation=0,
        ),
    }
    if "boost_type" in effects.columns:
        figures["optional_nonpeak_or_shuffle_control"] = grouped_bar_figure(effects, group="boost_type", value="delta_fraction_vs_flatten", title="Boost control comparison", ylabel="delta fraction vs flatten")
    return figures


if __name__ == "__main__":
    raise SystemExit(main_for("chunk_stsp_layer2_peak_boost_attraction", plot_bundle, title="Chunk STSP Layer2 Peak Boost Attraction"))
