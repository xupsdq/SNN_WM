from __future__ import annotations

from copy import deepcopy
from typing import Any


CANVAS_MM = (165.0, 152.0)
TWO_COLUMN_SLOTS = {
    "a": [2.0, 2.0, 79.5, 48.0],
    "b": [83.5, 2.0, 79.5, 48.0],
    "c": [2.0, 52.0, 79.5, 48.0],
    "d": [83.5, 52.0, 79.5, 48.0],
    "e": [2.0, 102.0, 79.5, 48.0],
    "f": [83.5, 102.0, 79.5, 48.0],
}
SCHEMATIC_SLOTS = {
    "a": [2.0, 2.0, 161.0, 48.0],
    "b": [2.0, 52.0, 79.5, 48.0],
    "c": [83.5, 52.0, 79.5, 48.0],
    "d": [2.0, 102.0, 79.5, 48.0],
    "e": [83.5, 102.0, 79.5, 48.0],
}
FIG2_SLOTS = {
    "a": [2.0, 2.0, 161.0, 48.0],
    "b": [2.0, 52.0, 52.333, 48.0],
    "c": [56.333, 52.0, 52.334, 48.0],
    "d": [110.667, 52.0, 52.333, 48.0],
}
FIG4_SLOTS = {
    "a": [2.0, 2.0, 79.5, 48.0],
    "b": [83.5, 2.0, 79.5, 48.0],
    "c": [2.0, 52.0, 52.333, 48.0],
    "d": [56.333, 52.0, 52.334, 48.0],
    "e": [110.667, 52.0, 52.333, 48.0],
}
FIG5_SLOTS = {
    "a": [2.0, 2.0, 52.333, 48.0],
    "b": [56.333, 2.0, 52.334, 48.0],
    "c": [110.667, 2.0, 52.333, 48.0],
    "d": [2.0, 52.0, 52.333, 48.0],
    "e": [56.333, 52.0, 52.334, 48.0],
    "f": [110.667, 52.0, 52.333, 48.0],
}
FIG6_SLOTS = {
    "a": [2.0, 2.0, 94.0, 48.0],
    "b": [98.0, 2.0, 65.0, 48.0],
    "c": [2.0, 52.0, 79.5, 48.0],
    "d": [83.5, 52.0, 79.5, 48.0],
    "e": [2.0, 102.0, 79.5, 48.0],
    "f": [83.5, 102.0, 79.5, 48.0],
}


FIGURE_SPECS: dict[str, dict[str, Any]] = {
    "fig1": {
        "chain_role": "inherit",
        "figure_question": "Can activity-silent STSP carry content across a delay?",
        "terminal_inference": (
            "A functional state is inherited after firing subsides and can influence a later input."
        ),
        "forbidden_inferences": [
            "persistent firing is the memory substrate",
            "the state alone reads out a final class",
        ],
        "slots": SCHEMATIC_SLOTS,
        "panels": {
            "a": {
                "claim": "STSP-SNN architecture and feedforward state path",
                "chart": "svg_asset",
                "source": "meta/panel_a_asset_manifest.csv",
                "role": "define the inherited-state substrate",
                "color_roles": ["layer1", "layer2", "layer3", "stsp_support"],
                "legend_owner": "asset",
            },
            "b": {
                "claim": "Accuracy is consistently high across fixed network seeds",
                "chart": "seed_trajectory",
                "source": "data/panel_b_plot_data.csv",
                "x_field": "network_seed",
                "x_label": "Network seed",
                "y_label": "Accuracy (%)",
                "x_limits": [999.5, 1019.5],
                "x_ticks": [1000, 1005, 1010, 1015, 1019],
                "y_limits": [84.0, 96.0],
                "emphasis_band": {
                    "lower": 85.0,
                    "upper": 95.0,
                    "color": "sample_window",
                },
                "band_boundaries": [85.0, 95.0],
                "colors": {"overall_recall": "dynamic"},
                "role": "establish functional premise",
                "legend_owner": "none",
            },
            "c": {
                "claim": "Firing vanishes during the delay",
                "chart": "time_binned_lines",
                "source": "data/panel_c_plot_data.csv",
                "x_field": "time_ms",
                "hue_field": "layer",
                "hue_order": ["layer1", "layer2", "layer3"],
                "hue_labels": {"layer1": "L1", "layer2": "L2", "layer3": "L3"},
                "x_label": "Time (ms)",
                "y_label": "Spike rate (Hz)",
                "y_labelpad": 2.0,
                "y_sci_power": 3,
                "x_limits": [-25.0, 625.0],
                "x_ticks": [0, 200, 400, 600],
                "y_min": 0.0,
                "stimulus_start_field": "stimulus_start_ms",
                "stimulus_end_field": "stimulus_end_ms",
                "stimulus_band_color": "sample_window",
                "colors": {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3"},
                "role": "exclude persistent firing",
                "legend_owner": "panel",
            },
            "d": {
                "claim": "Delay-state u/x retains decodable content",
                "chart": "ordered_lines",
                "source": "data/panel_d_plot_data.csv",
                "x_field": "delay_ms",
                "x_order": [100, 200, 400, 800, 1200],
                "hue_field": "layer",
                "hue_order": ["layer1", "layer2", "layer3"],
                "hue_labels": {"layer1": "L1", "layer2": "L2", "layer3": "L3"},
                "x_label": "Delay (ms)",
                "y_label": "Decode accuracy (%)",
                "y_labelpad": -1.0,
                "y_limits": [0.0, 100.0],
                "references": [{"value": 10.0, "label": "chance"}],
                "colors": {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3"},
                "role": "establish silent content",
                "legend_owner": "panel",
            },
            "e": {
                "claim": "u/x shuffling transfers attribution",
                "chart": "stacked_composition",
                "source": "data/panel_e_plot_data.csv",
                "condition_field": "condition",
                "condition_order": ["dynamic_intact", "ux_trial_shuffle"],
                "condition_labels": {
                    "dynamic_intact": "Dynamic STSP",
                    "ux_trial_shuffle": "u/x shuffle",
                },
                "category_field": "category",
                "category_order": ["Original", "Donor", "Other"],
                "category_labels": {
                    "Original": "Original",
                    "Donor": "Donor",
                    "Other": "Other",
                },
                "x_label": "",
                "y_label": "Composition (%)",
                "y_labelpad": -1.0,
                "y_limits": [0.0, 100.0],
                "y_ticks": [0, 25, 50, 75, 100],
                "annotate_categories": ["Original", "Donor"],
                "annotation_decimals": 1,
                "colors": {
                    "Original": "original_sample_trace",
                    "Donor": "donor_trace",
                    "Other": "other_residual",
                },
                "role": "show functional attribution",
                "legend_owner": "panel",
            },
        },
    },
    "fig2": {
        "chain_role": "transition",
        "canvas_mm": [165.0, 102.0],
        "figure_question": (
            "Does the same current input produce a common state update while "
            "preserving history-dependent consequences?"
        ),
        "terminal_inference": (
            "The same B induces a common update yet leaves a history-dependent "
            "residual concentrated at transition events."
        ),
        "forbidden_inferences": [
            "history universally improves accuracy",
            "K5 belongs to this one-step figure",
            "L1-to-L2 transfer is established in this figure",
        ],
        "slots": FIG2_SLOTS,
        "panels": {
            "a": {
                "claim": "Two inherited histories receive the identical current B",
                "chart": "svg_asset",
                "source": "meta/panel_a_asset_manifest.csv",
                "custom_renderer": "fig2_transition",
                "schematic_layout": {
                    "history_centers": {
                        "A": [42.0, 28.0],
                        "C": [42.0, 12.0],
                    },
                    "history_node_size": [30.0, 8.0],
                    "b_bbox": [108.0, 12.0, 16.0, 16.0],
                },
                "role": "define the exact-B transition",
                "color_roles": ["layer1", "second_item_reference"],
                "legend_owner": "asset",
            },
            "b": {
                "claim": "Aligned history increases rescue and reduces loss",
                "chart": "grouped_bars",
                "source": "data/panel_b_plot_data.csv",
                "x_field": "history_relation",
                "x_order": ["mismatched", "aligned"],
                "x_labels": {"mismatched": "Mismatched", "aligned": "Aligned"},
                "hue_field": "outcome_type",
                "hue_order": ["rescue", "loss"],
                "hue_labels": {
                    "rescue": "Rescue",
                    "loss": "Loss",
                },
                "y_label": "Rate (%)",
                "y_limits": [0.0, 70.0],
                "y_ticks": [0.0, 20.0, 40.0, 60.0],
                "bar_width": 0.32,
                "colors": {"rescue": "dynamic", "loss": "donor_trace"},
                "role": "show bidirectional behavioral rewrite",
                "legend_owner": "panel",
            },
            "c": {
                "claim": "A shared update coexists with a history residual",
                "chart": "threshold_margin_bars",
                "source": "data/panel_c_plot_data.csv",
                "endpoint_order": [
                    "same_B_common_update_cosine",
                    "processing_residual_gamma_energy_fraction",
                ],
                "endpoint_labels": {
                    "same_B_common_update_cosine": "Common\nupdate",
                    "processing_residual_gamma_energy_fraction": "History\nresidual",
                },
                "y_label": "Value − threshold",
                "y_labelpad": -1.0,
                "y_limits": [-0.05, 0.45],
                "y_ticks": [0.0, 0.2, 0.4],
                "bar_width": 0.48,
                "colors": {
                    "same_B_common_update_cosine": "dynamic",
                    "processing_residual_gamma_energy_fraction": "fused_state",
                },
                "role": "separate common processing from history sensitivity",
                "legend_owner": "none",
            },
            "d": {
                "claim": "History-sensitive residual is enriched at events",
                "chart": "ordered_bars",
                "source": "data/panel_d_plot_data.csv",
                "filters": {"record_type": "paired_network_component"},
                "category_field": "condition",
                "category_order": ["matched_random", "changed_events"],
                "category_labels": {
                    "matched_random": "Matched\nrandom",
                    "changed_events": "Changed\nevents",
                },
                "y_label": "Residual magnitude",
                "y_labelpad": -1.0,
                "y_limits": [0.0, 0.05],
                "y_ticks": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
                "bar_width": 0.48,
                "annotate_values": False,
                "colors": {
                    "matched_random": "baseline_control",
                    "changed_events": "fused_state",
                },
                "role": "locate the transition at events",
                "legend_owner": "none",
            },
        },
    },
    "fig3": {
        "chain_role": "implement",
        "figure_question": "How is one history-conditioned transition implemented locally?",
        "terminal_inference": (
            "Retained overlap support biases local advance/recruit dynamics and L2 write-back."
        ),
        "forbidden_inferences": [
            "units or events are independent replicates",
            "local support alone predicts the final class",
        ],
        "slots": TWO_COLUMN_SLOTS,
        "panels": {
            "a": {
                "claim": "Overlap-specific L1 STSP is a causal entry gate",
                "chart": "boxplot",
                "source": "data/panel_a_plot_data.csv",
                "x_field": "endpoint",
                "x_order": [
                    "dynamic_minus_overlap_reset",
                    "nonoverlap_reset_minus_overlap_reset",
                    "random_reset_minus_overlap_reset",
                ],
                "x_labels": {
                    "dynamic_minus_overlap_reset": "Dynamic",
                    "nonoverlap_reset_minus_overlap_reset": "Non-overlap",
                    "random_reset_minus_overlap_reset": "Random",
                },
                "y_label": "Accuracy contrast (%)",
                "references": [{"value": 0.0}],
                "show_fliers": False,
                "colors": {
                    "dynamic_minus_overlap_reset": "dynamic",
                    "nonoverlap_reset_minus_overlap_reset": "non_overlap_control",
                    "random_reset_minus_overlap_reset": "random_control",
                },
                "role": "identify the spatial causal entry",
                "legend_owner": "none",
            },
            "b": {
                "claim": "Support is already structured before the probe",
                "chart": "ordered_bars",
                "source": "data/panel_b_plot_data.csv",
                "filters": {
                    "condition": [
                        "overlap_dominant",
                        "probe_only_dominant",
                        "balanced",
                    ]
                },
                "category_field": "condition",
                "category_order": [
                    "overlap_dominant",
                    "probe_only_dominant",
                    "balanced",
                ],
                "category_labels": {
                    "overlap_dominant": "Overlap",
                    "probe_only_dominant": "Probe-only",
                    "balanced": "Balanced",
                },
                "y_label": "Pre-probe support",
                "y_limits": [0.0, 0.34],
                "y_ticks": [0.0, 0.1, 0.2, 0.3],
                "annotate_values": False,
                "colors": {
                    "overlap_dominant": "sample_probe_overlap",
                    "probe_only_dominant": "probe_only_region",
                    "balanced": "balanced_support",
                },
                "role": "show retained support before input",
                "legend_owner": "none",
            },
            "c": {
                "claim": "Retained support becomes advance and recruitment",
                "chart": "stacked_composition",
                "source": "data/panel_c_plot_data.csv",
                "condition_field": "condition",
                "condition_order": [
                    "overlap_dominant",
                    "probe_only_dominant",
                    "random_matched",
                ],
                "condition_labels": {
                    "overlap_dominant": "Overlap",
                    "probe_only_dominant": "Probe-only",
                    "random_matched": "Random\nmatched",
                },
                "category_field": "endpoint",
                "category_order": ["P_advance", "P_recruit", "P_loss"],
                "category_labels": {
                    "P_advance": "Advance",
                    "P_recruit": "Recruit",
                    "P_loss": "Loss",
                },
                "y_label": "Transition composition (%)",
                "y_limits": [0.0, 100.0],
                "y_ticks": [0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
                "require_sum_100": False,
                "colors": {
                    "P_advance": "transition_advance",
                    "P_recruit": "transition_recruit",
                    "P_loss": "transition_loss",
                },
                "role": "show conversion into local transitions",
                "legend_owner": "panel",
            },
            "d": {
                "claim": "Winner and loser trajectories diverge before events",
                "chart": "ordered_lines",
                "source": "data/panel_d_plot_data.csv",
                "filters": {"record_type": "event_aligned_trace"},
                "x_field": "time_ms",
                "x_order": [
                    -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2,
                    3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                ],
                "numeric_x": True,
                "x_ticks": [-8, -4, 0, 4, 8, 12],
                "hue_field": "condition",
                "hue_order": ["winner_delta_v", "loser_delta_v"],
                "hue_labels": {"winner_delta_v": "Winner", "loser_delta_v": "Loser"},
                "x_label": "Time from winner spike (ms)",
                "y_label": "ΔV (mV)",
                "value_scale": 1000.0,
                "colors": {"winner_delta_v": "winner", "loser_delta_v": "loser"},
                "reference_x": 0.0,
                "reference_y": 0.0,
                "reference_x_style": "-",
                "reference_y_style": "-",
                "show_individual_traces": False,
                "show_markers": False,
                "line_width": 1.5,
                "ci_alpha": 0.12,
                "role": "resolve local competition",
                "legend_owner": "panel",
            },
            "e": {
                "claim": "L2 write-back depends on prior update history",
                "chart": "grouped_bars",
                "source": "data/panel_e_plot_data.csv",
                "filters": {"record_type": "network_probability"},
                "x_field": "condition",
                "x_order": ["dynamic_intact", "static_opportunity"],
                "x_labels": {
                    "dynamic_intact": "Dynamic",
                    "static_opportunity": "Static\nopportunity",
                },
                "hue_field": "history_status",
                "hue_order": ["prior_updated", "not_prior_updated"],
                "hue_labels": {
                    "prior_updated": "Prior updated",
                    "not_prior_updated": "Not prior",
                },
                "y_label": "L2 update (%)",
                "y_limits": [0.0, 30.0],
                "y_ticks": [0.0, 10.0, 20.0, 30.0],
                "colors": {
                    "prior_updated": "prior_updated",
                    "not_prior_updated": "not_prior_updated",
                },
                "role": "show history-dependent L2 rewriting",
                "legend_owner": "panel",
            },
            "f": {
                "claim": "L1 STSP is necessary for early advance or recruitment",
                "chart": "category_points",
                "source": "data/panel_f_plot_data.csv",
                "x_field": "endpoint",
                "x_order": ["dynamic_minus_attenuation", "dynamic_minus_reset"],
                "x_labels": {
                    "dynamic_minus_attenuation": "Attenuate",
                    "dynamic_minus_reset": "Reset",
                },
                "y_label": "Change in P (%)",
                "y_limits": [0.0, 42.0],
                "references": [{"value": 0.0}],
                "color_by_x": True,
                "mean_marker": "D",
                "mean_marker_filled": True,
                "colors": {
                    "dynamic_minus_attenuation": "perturb_attenuate",
                    "dynamic_minus_reset": "perturb_reset",
                },
                "role": "test STSP causal necessity",
                "legend_owner": "none",
            },
        },
    },
    "fig4": {
        "chain_role": "recur",
        "canvas_mm": [165.0, 102.0],
        "figure_question": (
            "Do successive inputs keep rewriting inherited STSP state, and how does "
            "accumulated history alter behavior while the transition machinery persists?"
        ),
        "terminal_inference": (
            "Successive inputs repeatedly rewrite inherited STSP state beyond equal-time "
            "passive evolution; accumulated K5 history lowers relation-balanced rescue and "
            "raises relation-balanced loss while state-transition components, event linkage, "
            "and downstream donor transfer remain detectable."
        ),
        "forbidden_inferences": [
            "K5 rescue has a statistically confirmed aligned-versus-mismatched reversal",
            "K5 loss is equivalent to zero alignment effect",
            "K5 is progressive stage five",
            "the endpoints establish full-sequence organization or long-term memory",
            "every progressive stage has a behavioral or donor-swap replication",
        ],
        "slots": FIG4_SLOTS,
        "comparison_obligations": [
            {
                "panels": ["a"],
                "comparison": "observed input versus equal-time passive at every stage",
                "reader_action": "compare the two stage trajectories without a log or broken axis",
            },
            {
                "panels": ["b"],
                "comparison": "K1 versus K5 within each outcome",
                "reader_action": "track Rescue and Loss separately across history depth",
                "forbidden_comparison": "Rescue and Loss cannot be subtracted because their opportunity denominators differ",
            },
            {
                "panels": ["c", "d", "e"],
                "comparison": "three complementary K5 persistence checks",
                "reader_action": "read state composition, event linkage, and causal downstream access in sequence",
            },
        ],
        "panels": {
            "a": {
                "claim": "Successive inputs repeatedly displace joint STSP state beyond equal-time passive evolution",
                "chart": "ordered_lines",
                "source": "data/panel_a_plot_data.csv",
                "x_field": "stage_k",
                "x_order": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "x_ticks": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "numeric_x": True,
                "x_limits": [1.8, 10.25],
                "hue_field": "condition",
                "hue_order": ["observed", "passive"],
                "hue_labels": {
                    "observed": "Observed",
                    "passive": "Equal-time passive",
                },
                "x_label": "Stage",
                "y_label": "State displacement",
                "y_limits": [0.0, 0.65],
                "y_ticks": [0.0, 0.2, 0.4, 0.6],
                "show_individual_traces": False,
                "show_ci_band": True,
                "line_width": 1.2,
                "marker_size": 3.4,
                "ci_alpha": 0.14,
                "use_persisted_ci": True,
                "markers": {"observed": "o", "passive": "s"},
                "linestyles": {"observed": "-", "passive": "--"},
                "direct_labels": True,
                "direct_label_offsets_pt": {
                    "observed": [-4.0, 4.0],
                    "passive": [-4.0, 5.0],
                },
                "direct_label_horizontal_alignment": "right",
                "plot_bbox_mm": [15.0, 9.0, 63.5, 31.0],
                "colors": {"observed": "dynamic", "passive": "baseline_control"},
                "role": "establish repeated input-driven state rewriting",
                "legend_owner": "none",
            },
            "b": {
                "claim": "Accumulated K5 history lowers rescue and raises loss",
                "chart": "grouped_bars",
                "source": "data/panel_b_plot_data.csv",
                "x_field": "prefix_k",
                "x_order": ["K1", "K5"],
                "x_labels": {"K1": "K1", "K5": "K5"},
                "hue_field": "outcome_type",
                "hue_order": ["rescue", "loss"],
                "hue_labels": {"rescue": "Rescue", "loss": "Loss"},
                "y_label": "Rate (%)",
                "y_limits": [0.0, 70.0],
                "y_ticks": [0.0, 20.0, 40.0, 60.0],
                "bar_width": 0.30,
                "use_persisted_ci": True,
                "plot_bbox_mm": [96.5, 9.0, 63.5, 31.0],
                "colors": {"rescue": "dynamic", "loss": "donor_trace"},
                "role": "show the behavioral cost of accumulated history",
                "legend_owner": "panel",
            },
            "c": {
                "claim": "K5 retains both a common update and a history residual",
                "chart": "threshold_margin_bars",
                "source": "data/panel_c_plot_data.csv",
                "endpoint_order": [
                    "same_B_common_update_cosine",
                    "processing_residual_gamma_energy_fraction",
                ],
                "endpoint_labels": {
                    "same_B_common_update_cosine": "Common\nupdate",
                    "processing_residual_gamma_energy_fraction": "History\nresidual",
                },
                "y_label": "Value − threshold",
                "y_labelpad": -1.0,
                "y_limits": [-0.05, 0.45],
                "y_ticks": [0.0, 0.2, 0.4],
                "bar_width": 0.48,
                "use_persisted_ci": True,
                "plot_bbox_mm": [15.0, 59.0, 36.333, 30.0],
                "colors": {
                    "same_B_common_update_cosine": "dynamic",
                    "processing_residual_gamma_energy_fraction": "fused_state",
                },
                "role": "show that state-transition components persist at K5",
                "legend_owner": "none",
            },
            "d": {
                "claim": "The K5 history residual remains concentrated at changed spike events",
                "chart": "ordered_bars",
                "source": "data/panel_d_plot_data.csv",
                "category_field": "condition",
                "category_order": ["matched_random", "changed_events"],
                "category_labels": {
                    "matched_random": "Matched\nrandom",
                    "changed_events": "Changed\nevents",
                },
                "y_label": "Residual magnitude",
                "y_labelpad": -1.0,
                "y_limits": [0.0, 0.05],
                "y_ticks": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
                "bar_width": 0.48,
                "annotate_values": False,
                "use_persisted_ci": True,
                "plot_bbox_mm": [69.333, 59.0, 36.334, 30.0],
                "colors": {
                    "matched_random": "baseline_control",
                    "changed_events": "fused_state",
                },
                "role": "locate the K5 history residual at real transition events",
                "legend_owner": "none",
            },
            "e": {
                "claim": "Inherited L1 state still enters downstream updating and early output at K5",
                "chart": "category_points",
                "source": "data/panel_e_plot_data.csv",
                "x_field": "endpoint",
                "x_order": [
                    "layer1_only_layer2_update_donor_transfer",
                    "layer1_only_early_class_score_donor_transfer",
                ],
                "x_labels": {
                    "layer1_only_layer2_update_donor_transfer": "L2 update",
                    "layer1_only_early_class_score_donor_transfer": "Early score",
                },
                "y_label": "Donor-transfer index",
                "y_limits": [0.0, 1.0],
                "y_ticks": [0.0, 0.25, 0.5, 0.75, 1.0],
                "color_by_x": True,
                "show_raw_points": True,
                "use_persisted_ci": True,
                "markers": {
                    "layer1_only_layer2_update_donor_transfer": "o",
                    "layer1_only_early_class_score_donor_transfer": "s",
                },
                "plot_bbox_mm": [123.667, 59.0, 36.333, 30.0],
                "colors": {
                    "layer1_only_layer2_update_donor_transfer": "donor_trace",
                    "layer1_only_early_class_score_donor_transfer": "donor_trace",
                },
                "role": "show causal downstream access to inherited state at K5",
                "legend_owner": "none",
            },
        },
    },
    "fig5": {
        "chain_role": "organize",
        "canvas_mm": [165.0, 102.0],
        "figure_question": "What structure emerges after repeated transitions?",
        "terminal_inference": (
            "Repeated transitions form a multi-component, history-specific, serially "
            "organized STSP state whose structural capacity is limited by load and delay."
        ),
        "forbidden_inferences": [
            "structural metrics are functional recall",
            "serial weights prove a recency advantage",
            "the final item completely overwrites earlier constituents",
        ],
        "slots": FIG5_SLOTS,
        "panels": {
            "a": {
                "claim": "Both pair constituents remain represented",
                "chart": "ordered_bars",
                "source": "data/panel_a_plot_data.csv",
                "category_field": "condition",
                "category_order": ["item_a", "item_b"],
                "category_labels": {"item_a": "Item A", "item_b": "Item B"},
                "y_label": "Pair similarity",
                "y_limits": [0.0, 1.02],
                "y_ticks": [0.0, 0.5, 1.0],
                "bar_width": 0.48,
                "annotate_values": False,
                "plot_bbox_mm": [13.0, 10.0, 38.0, 30.0],
                "colors": {
                    "item_a": "first_item_reference",
                    "item_b": "second_item_reference",
                },
                "role": "establish pair component retention",
                "legend_owner": "none",
            },
            "b": {
                "claim": "Experienced pairs exceed shuffled pairs",
                "chart": "boxplot",
                "source": "data/panel_b_plot_data.csv",
                "x_field": "condition",
                "x_order": ["experienced_pair", "shuffled_pair"],
                "x_labels": {
                    "experienced_pair": "Experienced\npair",
                    "shuffled_pair": "Shuffled\npair",
                },
                "y_label": "Pair similarity",
                "y_limits": [0.98, 1.0],
                "y_ticks": [0.98, 0.99, 1.0],
                "plot_bbox_mm": [67.333, 10.0, 38.0, 30.0],
                "colors": {
                    "experienced_pair": "true_pair",
                    "shuffled_pair": "shuffled_pair",
                },
                "role": "establish experienced-pair specificity",
                "legend_owner": "none",
            },
            "c": {
                "claim": "Pair organization exceeds a linear mixture",
                "chart": "ordered_bars",
                "source": "data/panel_c_plot_data.csv",
                "category_field": "condition",
                "category_order": ["experienced_residual", "shuffled_residual"],
                "category_labels": {
                    "experienced_residual": "Experienced\npair",
                    "shuffled_residual": "Shuffled\npair",
                },
                "y_label": "Residual similarity",
                "y_limits": [0.0, 0.65],
                "y_ticks": [0.0, 0.2, 0.4, 0.6],
                "bar_width": 0.48,
                "annotate_values": False,
                "plot_bbox_mm": [121.667, 10.0, 38.0, 30.0],
                "colors": {
                    "experienced_residual": "whole_pair_representation",
                    "shuffled_residual": "shuffled_pair",
                },
                "role": "show beyond-mixture organization",
                "legend_owner": "none",
            },
            "d": {
                "claim": "Effective items expand sublinearly with sequence load",
                "chart": "ordered_lines",
                "source": "data/panel_d_plot_data.csv",
                "x_field": "seq_len",
                "x_order": [3, 5, 7, 10],
                "x_label": "Items (K)",
                "y_label": "Effective items",
                "y_limits": [0.0, 10.5],
                "y_ticks": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                "identity_reference": True,
                "identity_reference_label": "$N_{eff}=K$",
                "show_individual_traces": False,
                "plot_bbox_mm": [13.0, 62.0, 38.0, 28.0],
                "colors": {"single": "dynamic"},
                "role": "show expansion and compression",
                "legend_owner": "none",
            },
            "e": {
                "claim": "The state is organized by serial position",
                "chart": "heatmap",
                "source": "data/panel_e_plot_data.csv",
                "x_field": "item_position",
                "x_order": list(range(1, 11)),
                "y_field": "seq_len",
                "y_order": [3, 5, 7, 10],
                "aggregate": "mean",
                "x_label": "Serial position",
                "y_label": "K",
                "colorbar_label": "Normalized weight",
                "cmap": "item_contribution",
                "vmin": 0.0,
                "unavailable_color": "#FFFFFF",
                "cell_edges": False,
                "plot_bbox_mm": [67.333, 62.0, 38.0, 28.0],
                "colorbar_orientation": "horizontal_top",
                "colorbar_height_mm": 1.4,
                "colorbar_gap_mm": 0.8,
                "colorbar_ticks_position": "top",
                "colorbar_label_position": "top",
                "colorbar_label_pad_pt": 1.0,
                "role": "show serial-position organization",
                "legend_owner": "colorbar",
            },
            "f": {
                "claim": "Structural capacity is bounded by load and delay",
                "chart": "heatmap",
                "source": "data/panel_f_plot_data.csv",
                "x_field": "seq_len",
                "x_order": [3, 5, 7, 10],
                "y_field": "delay_ms",
                "y_order": [800, 400, 200, 100],
                "aggregate": "mean",
                "x_label": "Items (K)",
                "y_label": "Delay (ms)",
                "colorbar_label": "Effective fraction",
                "cmap": "stsp_support",
                "vmin": 0.0,
                "vmax": 1.0,
                "unavailable_color": "#FFFFFF",
                "cell_edges": False,
                "annotate_cells": True,
                "annotation_decimals": 2,
                "plot_bbox_mm": [121.667, 62.0, 38.0, 28.0],
                "colorbar_orientation": "horizontal_top",
                "colorbar_height_mm": 1.4,
                "colorbar_gap_mm": 0.8,
                "colorbar_ticks_position": "top",
                "colorbar_label_position": "top",
                "colorbar_label_pad_pt": 1.0,
                "role": "map the structural boundary",
                "legend_owner": "colorbar",
            },
        },
    },
    "fig6": {
        "chain_role": "access",
        "canvas_mm": [165.0, 152.0],
        "figure_question": "Can the structured state be conditionally accessed by a later cue?",
        "terminal_inference": (
            "Structured state is conditionally accessible, content-selective, constrained by "
            "a load-by-delay operating region, causally supported, and gated by pathway overlap."
        ),
        "forbidden_inferences": [
            "STSP replays memory without a cue",
            "all retained items are perfectly readable",
            "load or delay has a universally monotonic main effect",
        ],
        "slots": FIG6_SLOTS,
        "panels": {
            "a": {
                "claim": "A pair state supports partial-cue recovery of A and B",
                "chart": "partial_cue_split",
                "source": "data/panel_a_plot_data.csv",
                "target_field": "target_item",
                "target_order": ["A", "B"],
                "condition_field": "condition",
                "condition_order": ["S0", "S_A", "S_B", "S_AB"],
                "condition_labels": {
                    "S0": "No memory",
                    "S_A": "Item A",
                    "S_B": "Item B",
                    "S_AB": "Pair",
                },
                "x_field": "keep_prob",
                "x_order": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
                "x_label": "Keep probability",
                "y_label": "Target recovery (%)",
                "y_limits": [0.0, 100.0],
                "y_ticks": [0.0, 50.0, 100.0],
                "legend_ncol": 4,
                "approved_internal_split": True,
                "exception_authority": (
                    "User approved the Target A/B split on 2026-07-31 and explicitly "
                    "required Target A and Target B to read as a close pair while "
                    "allocating more width to Fig.6b on 2026-08-01."
                ),
                "plot_bbox_mm": [2.0, 12.0, 94.0, 28.0],
                "child_plot_bboxes_mm": [
                    [13.0, 12.0, 38.0, 28.0],
                    [55.0, 12.0, 38.0, 28.0],
                ],
                "legend_anchor": [0.5, 1.24],
                "show_right_y_axis": True,
                "colors": {
                    "S0": "baseline_control",
                    "S_A": "first_item_reference",
                    "S_B": "second_item_reference",
                    "S_AB": "true_pair",
                },
                "role": "establish minimal functional access",
                "legend_owner": "panel",
            },
            "b": {
                "claim": "Access generalizes across positions in a ten-item sequence",
                "chart": "ordered_lines",
                "source": "data/panel_b_absolute_access.csv",
                "x_field": "target_position",
                "x_order": list(range(1, 11)),
                "hue_field": "endpoint",
                "hue_order": [
                    "P_target_cue_only",
                    "P_target_single_item_memory",
                    "P_target_sequence_state",
                ],
                "hue_labels": {
                    "P_target_cue_only": "Cue only",
                    "P_target_single_item_memory": "Singleton",
                    "P_target_sequence_state": "Sequence",
                },
                "numeric_x": True,
                "x_ticks": [1, 2, 4, 6, 8, 10],
                "x_limits": [0.5, 10.5],
                "x_label": "Serial position",
                "y_label": "Target readout (%)",
                "y_limits": [0.0, 100.0],
                "y_ticks": [0.0, 50.0, 100.0],
                "show_individual_traces": False,
                "show_ci_band": True,
                "legend_ncol": 3,
                "plot_bbox_mm": [109.0, 12.0, 51.0, 28.0],
                "colors": {
                    "P_target_cue_only": "cue_only",
                    "P_target_single_item_memory": "single_item_memory",
                    "P_target_sequence_state": "sequence_state",
                },
                "role": "generalize access to multiple items",
                "legend_owner": "panel",
            },
            "c": {
                "claim": "Access is selective for matched cue content",
                "chart": "ordered_lines",
                "source": "data/panel_c_position_profiles.csv",
                "x_field": "target_position",
                "x_order": list(range(1, 8)),
                "hue_field": "condition",
                "hue_order": ["matched", "mismatched", "unseen"],
                "hue_labels": {
                    "matched": "Matched",
                    "mismatched": "Mismatched",
                    "unseen": "Unseen",
                },
                "numeric_x": True,
                "x_ticks": [1, 2, 3, 4, 5, 6, 7],
                "x_limits": [0.5, 7.5],
                "x_label": "Serial position",
                "y_label": "Target readout (%)",
                "y_limits": [0.0, 100.0],
                "y_ticks": [0.0, 50.0, 100.0],
                "show_individual_traces": False,
                "show_ci_band": True,
                "legend_ncol": 3,
                "plot_bbox_mm": [13.0, 62.0, 65.5, 28.0],
                "colors": {
                    "matched": "sequence_state",
                    "mismatched": "single_item_memory",
                    "unseen": "cue_only",
                },
                "role": "establish cue-content specificity",
                "legend_owner": "panel",
            },
            "d": {
                "claim": "Functional rescue is bounded by load and delay",
                "chart": "heatmap",
                "source": "data/panel_d_plot_data.csv",
                "x_field": "seq_len",
                "x_order": [3, 5, 7, 10],
                "y_field": "delay_ms",
                "y_order": [800, 400, 200, 100],
                "aggregate": "mean",
                "x_label": "Items (K)",
                "y_label": "Delay (ms)",
                "colorbar_label": "Rescued fraction",
                "cmap": "stsp_support",
                "vmin": 0.0,
                "vmax": 1.0,
                "unavailable_color": "#FFFFFF",
                "cell_edges": False,
                "annotate_cells": True,
                "annotation_decimals": 2,
                "plot_bbox_mm": [94.5, 62.0, 65.5, 28.0],
                "colorbar_orientation": "horizontal_top",
                "colorbar_height_mm": 1.4,
                "colorbar_gap_mm": 0.8,
                "colorbar_ticks_position": "top",
                "colorbar_label_position": "top",
                "colorbar_label_pad_pt": 1.0,
                "role": "map the functional boundary",
                "legend_owner": "colorbar",
            },
            "e": {
                "claim": "High-STSP-overlap sites causally support recruitment",
                "chart": "ordered_bars",
                "source": "data/panel_e_plot_data.csv",
                "category_field": "condition",
                "category_order": ["high_stsp_overlap", "matched_removal"],
                "category_labels": {
                    "high_stsp_overlap": "High\noverlap",
                    "matched_removal": "Matched",
                },
                "y_label": "Recruitment loss (%)",
                "y_limits": [0.0, 4.0],
                "y_ticks": [0.0, 1.0, 2.0, 3.0, 4.0],
                "bar_width": 0.48,
                "annotate_values": False,
                "plot_bbox_mm": [13.0, 110.0, 65.5, 30.0],
                "colors": {
                    "high_stsp_overlap": "high_stsp",
                    "matched_removal": "baseline_control",
                },
                "role": "establish targeted causal contribution",
                "legend_owner": "none",
            },
            "f": {
                "claim": "Retained STSP is expressed only through overlapping input paths",
                "chart": "two_by_two",
                "source": "data/panel_f_plot_data.csv",
                "cell_filter": {"record_type": "network_2x2_cell"},
                "contrast_filter": {"record_type": "paired_network_interaction"},
                "cell_field": "cell_or_interaction",
                "cell_mapping": {
                    "high_overlap_delta": ["overlap", "high"],
                    "low_overlap_delta": ["overlap", "low"],
                    "high_nooverlap_delta": ["no_overlap", "high"],
                    "low_nooverlap_delta": ["no_overlap", "low"],
                },
                "x_field": "_mapped_x",
                "x_order": ["no_overlap", "overlap"],
                "x_labels": {"no_overlap": "No overlap", "overlap": "Overlap"},
                "hue_field": "_mapped_hue",
                "hue_order": ["low", "high"],
                "hue_labels": {"low": "Low STSP", "high": "High STSP"},
                "y_label": "Firing change (%)",
                "y_limits": [0.0, 20.0],
                "y_ticks": [0.0, 5.0, 10.0, 15.0, 20.0],
                "show_raw_points": False,
                "show_contrast_panel": False,
                "plot_bbox_mm": [94.5, 110.0, 65.5, 30.0],
                "colors": {"low": "low_stsp", "high": "high_stsp"},
                "role": "identify the local expression gate",
                "legend_owner": "panel",
            },
        },
    },
}


def _layout_contract(spec: dict[str, Any]) -> dict[str, Any]:
    panels = spec["panels"]
    panel_ids = list(panels)
    rows: dict[float, list[str]] = {}
    geometry: dict[str, dict[str, Any]] = {}
    for panel_id, panel in panels.items():
        x_mm, y_mm, width_mm, height_mm = spec["slots"][panel_id]
        rows.setdefault(y_mm, []).append(panel_id)
        chart = panel["chart"]
        geometry[panel_id] = {
            "chart_family": chart,
            "category_slots": max(1, len(panel.get("x_order", panel.get("category_order", [1])))),
            "natural_aspect": [1.1, 4.0] if width_mm > 100 else [1.1, 2.2],
            "decoration_sides": (
                []
                if chart == "svg_asset"
                else ["bottom"]
                if chart == "estimate_strip"
                else ["left", "right", "bottom"]
                if chart == "heatmap"
                else ["left", "bottom", "top"]
            ),
            "visual_weight": (
                "high"
                if chart in {"svg_asset", "protocol", "heatmap"}
                else "low"
                if chart == "estimate_strip"
                else "medium"
            ),
            "slot_bbox_mm": [x_mm, y_mm, width_mm, height_mm],
            "plot_bbox_mm": panel.get("plot_bbox_mm"),
        }
    alignment_groups = []
    for row_index, (_, members) in enumerate(sorted(rows.items())):
        alignment_groups.append(
            {
                "group_id": f"row_{row_index + 1}",
                "panels": members,
                "target": "slot",
                "edges": ["top", "bottom"],
                "rationale": "Preserve the frozen row-major scientific reading order.",
            }
        )
    if spec.get("figure_id") == "fig1":
        figure_id = "fig1"
        alignment_groups.extend(
            [
                {
                    "group_id": f"{figure_id}_row_2_plot_axes",
                    "panels": ["b", "c"],
                    "target": "plot_area",
                    "edges": ["top", "bottom"],
                    "rationale": "Place the second-row x axes at one physical height.",
                    "comparison_basis": "Shared quantitative decoration margins.",
                },
                {
                    "group_id": f"{figure_id}_row_3_plot_axes",
                    "panels": ["d", "e"],
                    "target": "plot_area",
                    "edges": ["top", "bottom"],
                    "rationale": "Place the third-row x axes at one physical height.",
                    "comparison_basis": "Shared quantitative decoration margins.",
                },
                {
                    "group_id": f"{figure_id}_left_column_plot_axes",
                    "panels": ["b", "d"],
                    "target": "plot_area",
                    "edges": ["left", "right"],
                    "rationale": "Align the left-column y axes and plot widths.",
                    "comparison_basis": "Identical 79.5 mm slots and 11/3 mm side margins.",
                },
                {
                    "group_id": f"{figure_id}_right_column_plot_axes",
                    "panels": ["c", "e"],
                    "target": "plot_area",
                    "edges": ["left", "right"],
                    "rationale": "Align the right-column y axes and plot widths.",
                    "comparison_basis": "Identical 79.5 mm slots and 11/3 mm side margins.",
                },
            ]
        )
    elif spec.get("figure_id") == "fig2":
        alignment_groups.append(
            {
                "group_id": "fig2_quantitative_row_plot_axes",
                "panels": ["b", "c", "d"],
                "target": "plot_area",
                "edges": ["top", "bottom"],
                "rationale": (
                    "Place all three quantitative panels on one shared physical baseline."
                ),
                "comparison_basis": (
                    "Equal-height 48 mm slots with identical 11/3/8/10 mm "
                    "quantitative decoration margins."
                ),
            }
        )
    elif spec.get("figure_id") == "fig4":
        alignment_groups.extend(
            [
                {
                    "group_id": "fig4_row_1_plot_axes",
                    "panels": ["a", "b"],
                    "target": "plot_area",
                    "edges": ["top", "bottom"],
                    "rationale": "Align the recurrence premise with its behavioral consequence.",
                    "comparison_basis": "Explicit 31 mm plot heights within the shared first row.",
                },
                {
                    "group_id": "fig4_row_2_plot_axes",
                    "panels": ["c", "d", "e"],
                    "target": "plot_area",
                    "edges": ["top", "bottom"],
                    "rationale": "Keep the three K5 persistence checks on one physical baseline.",
                    "comparison_basis": "Explicit 30 mm plot heights within equal-width second-row slots.",
                },
            ]
        )
    elif spec.get("figure_id") == "fig5":
        alignment_groups.extend(
            [
                {
                    "group_id": "fig5_row_1_plot_axes",
                    "panels": ["a", "b", "c"],
                    "target": "plot_area",
                    "edges": ["top", "bottom"],
                    "rationale": "Align the three pair-organization comparisons on one evidence row.",
                    "comparison_basis": "All three panels use 30 mm-high categorical data regions.",
                },
                {
                    "group_id": "fig5_row_2_plot_axes",
                    "panels": ["d", "e", "f"],
                    "target": "plot_area",
                    "edges": ["top", "bottom"],
                    "rationale": "Place the multi-item trajectory and both heatmaps on one structural row.",
                    "comparison_basis": "All three panels use aligned 28 mm-high data regions; e/f colorbars occupy the top decoration band.",
                },
            ]
        )
    elif spec.get("figure_id") == "fig6":
        alignment_groups.extend(
            [
                {
                    "group_id": "fig6_row_1_plot_axes",
                    "panels": ["a", "b"],
                    "target": "plot_area",
                    "edges": ["top", "bottom"],
                    "rationale": "Pair and multi-item access form the opening functional row.",
                    "comparison_basis": "Target A/B form one close pair of 38 x 28 mm axes separated by 4 mm; panel b receives a wider 51 x 28 mm data region.",
                },
                {
                    "group_id": "fig6_row_2_plot_axes",
                    "panels": ["c", "d"],
                    "target": "plot_area",
                    "edges": ["top", "bottom"],
                    "rationale": "Content specificity and the global operating boundary form the constraint row.",
                    "comparison_basis": "Both use aligned 28 mm-high data regions while d places its colorbar in the top decoration band.",
                },
                {
                    "group_id": "fig6_row_3_plot_axes",
                    "panels": ["e", "f"],
                    "target": "plot_area",
                    "edges": ["top", "bottom"],
                    "rationale": "Targeted contribution and local overlap gating form the causal row.",
                    "comparison_basis": "Both use 30 mm-high quantitative regions in equal slots.",
                },
            ]
        )
    canvas_width, canvas_height = [
        float(value) for value in spec.get("canvas_mm", CANVAS_MM)
    ]
    comparison_groups = []
    for index, obligation in enumerate(
        spec.get("reader_contract", {}).get("comparison_obligations", [])
    ):
        members = list(obligation.get("panels", []))
        if len(members) < 2:
            continue
        comparison_groups.append(
            {
                "group_id": f"{spec.get('figure_id', 'figure')}_comparison_{index + 1}",
                "panels": members,
                "comparison_basis": obligation.get("comparison_basis")
                or obligation.get("comparison", ""),
                "reader_task": obligation.get("reader_task")
                or obligation.get("reader_action", ""),
            }
        )
    return {
        "version": "practical_layout_v1",
        "status": "frozen",
        "semantic_units": [
            {"unit_id": f"unit_{panel_id}", "panels": [panel_id], "role": panels[panel_id]["role"]}
            for panel_id in panel_ids
        ],
        "comparison_groups": comparison_groups,
        "alignment_groups": alignment_groups,
        "panel_geometry": geometry,
        "bar_width_policy": {
            "mode": "within_panel_only",
            "scope": "No cross-panel physical bar-width constraint is needed.",
            "tradeoff": "Scientific comparison and readable plot area take precedence.",
        },
        "topology": {
            "reading_direction": "row_major",
            "unit_sequence": [f"unit_{panel_id}" for panel_id in panel_ids],
            "rationale": "Follow the frozen panel sequence and argument DAG.",
            "released_alignment_edges": (
                [
                    {
                        "between_rows": [["a", "b"], ["c", "d", "e"]],
                        "edges": ["left", "right"],
                        "rationale": (
                            "The first row groups premise and behavioral consequence, whereas "
                            "the second row groups three K5 persistence checks; their columns "
                            "are not semantically corresponding."
                        ),
                    }
                ]
                if spec.get("figure_id") == "fig4"
                else [
                    {
                        "between_rows": [["a", "b"], ["c", "d"], ["e", "f"]],
                        "edges": ["left", "right"],
                        "rationale": (
                            "Fig.6 row 1 uses a grouped 94:65 mm split because panel a "
                            "contains two closely related targets while panel b needs greater "
                            "serial-position width; rows 2 and 3 each "
                            "contain two equal panel columns and do not correspond vertically "
                            "to the first-row thirds."
                        ),
                    }
                ]
                if spec.get("figure_id") == "fig6"
                else []
            ),
        },
        "hard_constraints": [
            f"{canvas_width:g} mm x {canvas_height:g} mm canvas",
            "frozen slots and panel order",
            (
                "equal row heights; Fig.6 row 1 uses the user-approved 94:65 mm grouped "
                "a:b split, while rows 2 and 3 have two equal panels"
                if spec.get("figure_id") == "fig6"
                else "equal row heights and equal widths within every row"
            ),
            "9 pt text and 12 pt lowercase panel labels",
            (
                "no clipping, decorative grid, or dual y axis; Fig.6a alone has the "
                "user-approved Target A/B internal split"
                if spec.get("figure_id") == "fig6"
                else "no clipping, decorative grid, or dual y axis"
            ),
        ],
        "soft_targets": [
            "balanced optical weight within each row",
            "centered frameless legends owned by one panel",
        ],
        "qa": {
            "final_size_render": True,
            "collision_check": True,
            "clipping_check": True,
            "alignment_measurement": True,
            "grayscale_check": True,
        },
    }


def get_figure_spec(figure_id: str) -> dict[str, Any]:
    if figure_id not in FIGURE_SPECS:
        raise KeyError(f"unknown final-six figure: {figure_id}")
    spec = deepcopy(FIGURE_SPECS[figure_id])
    spec["figure_id"] = figure_id
    spec["canvas_mm"] = list(spec.get("canvas_mm", CANVAS_MM))
    comparison_obligations = list(spec.pop("comparison_obligations", []))
    task_graph_edges = list(
        spec.pop(
            "task_graph_edges",
            [
                [left, right]
                for left, right in zip(
                    list(spec["panels"])[:-1], list(spec["panels"])[1:]
                )
            ],
        )
    )
    spec["reader_contract"] = {
        "figure_question": spec.pop("figure_question"),
        "terminal_inference": spec.pop("terminal_inference"),
        "forbidden_inferences": spec.pop("forbidden_inferences"),
        "semantic_units": {
            panel_id: panel["role"] for panel_id, panel in spec["panels"].items()
        },
        "task_graph": {
            "nodes": list(spec["panels"]),
            "edges": task_graph_edges,
        },
        "comparison_obligations": comparison_obligations,
        "topology_invariants": ["frozen slots", "row-major panel order"],
        "topology_freedoms": ["optical padding inside each slot"],
    }
    spec["layout_contract"] = _layout_contract(spec)
    return spec


__all__ = ["CANVAS_MM", "FIGURE_SPECS", "get_figure_spec"]
