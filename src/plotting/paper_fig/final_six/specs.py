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
FIG3_SLOTS = {
    **deepcopy(TWO_COLUMN_SLOTS),
    "g": [2.0, 152.0, 161.0, 48.0],
}
FIG4_SLOTS = {
    "a": [2.0, 2.0, 79.5, 48.0],
    "b": [83.5, 2.0, 79.5, 48.0],
    "c": [2.0, 52.0, 79.5, 48.0],
    "d": [83.5, 52.0, 79.5, 48.0],
}
FIG5_SLOTS = deepcopy(TWO_COLUMN_SLOTS)
FIG6_SLOTS = deepcopy(TWO_COLUMN_SLOTS)


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
                "claim": "Alternative preceding inputs are followed by the same B in paired runs before state and behavioral responses are compared",
                "chart": "schematic",
                "source": "data/panel_a_input_stimuli.csv",
                "custom_renderer": "fig2_paired_dms",
                "schematic_layout": {
                    "content_bounds": [0.0, 0.0, 152.0, 40.0],
                    "history_rows": {
                        "A": {
                            "image_bbox": [2.0, 24.0, 10.0, 10.0],
                            "label_xy": [7.0, 36.5],
                            "delay_bbox": [18.0, 23.0, 18.0, 12.0],
                            "center_y": 29.0,
                        },
                        "C": {
                            "image_bbox": [2.0, 6.0, 10.0, 10.0],
                            "label_xy": [7.0, 18.5],
                            "delay_bbox": [18.0, 5.0, 18.0, 12.0],
                            "center_y": 11.0,
                        },
                    },
                    "shared_b": {
                        "image_bbox": [50.0, 14.0, 12.0, 12.0],
                        "label_xy": [56.0, 29.5],
                    },
                    "comparison_bbox": [76.0, 3.0, 73.0, 34.0],
                    "state_icon_bbox": [88.0, 16.0, 12.0, 12.0],
                    "behavior_icon_bbox": [125.0, 16.0, 12.0, 12.0],
                },
                "role": "define the exact-input counterfactual and the paired response comparison",
                "color_roles": ["dynamic", "fused_state", "neutral_text"],
                "legend_owner": "none",
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
                    "processing_residual_gamma_norm_ratio",
                ],
                "endpoint_labels": {
                    "same_B_common_update_cosine": "Common\nupdate",
                    "processing_residual_gamma_norm_ratio": "History\nresidual",
                },
                "y_label": "Value − threshold",
                "y_labelpad": -1.0,
                "y_limits": [-0.05, 0.45],
                "y_ticks": [0.0, 0.2, 0.4],
                "bar_width": 0.48,
                "colors": {
                    "same_B_common_update_cosine": "dynamic",
                    "processing_residual_gamma_norm_ratio": "fused_state",
                },
                "role": "separate common processing from history sensitivity",
                "legend_owner": "none",
            },
            "d": {
                "claim": "History-sensitive residual is larger at transition events",
                "chart": "category_points",
                "source": "data/panel_d_plot_data.csv",
                "x_field": "condition",
                "x_order": ["matched_random", "changed_events"],
                "x_labels": {
                    "matched_random": "Matched\nrandom",
                    "changed_events": "Changed\nevents",
                },
                "y_label": "Residual magnitude",
                "y_limits": [0.0, 0.055],
                "y_ticks": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
                "show_raw_points": True,
                "color_by_x": True,
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
        "canvas_mm": [165.0, 202.0],
        "chain_role": "implement",
        "figure_question": "How is one history-conditioned transition implemented locally?",
        "terminal_inference": (
            "Retained overlap support biases local advance/recruit dynamics and L2 write-back."
        ),
        "forbidden_inferences": [
            "units or events are independent replicates",
            "local support alone predicts the final class",
        ],
        "slots": FIG3_SLOTS,
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
                "claim": "Retained support changes advance, recruitment, and loss probabilities",
                "chart": "ordered_lines",
                "source": "data/panel_c_plot_data.csv",
                "x_field": "condition",
                "x_order": [
                    "overlap_dominant",
                    "probe_only_dominant",
                    "random_matched",
                ],
                "x_labels": {
                    "overlap_dominant": "Overlap",
                    "probe_only_dominant": "Probe-only",
                    "random_matched": "Random\nmatched",
                },
                "hue_field": "endpoint",
                "hue_order": ["P_advance", "P_recruit", "P_loss"],
                "hue_labels": {
                    "P_advance": "Advance",
                    "P_recruit": "Recruit",
                    "P_loss": "Loss",
                },
                "y_label": "Transition probability (%)",
                "y_limits": [0.0, 30.0],
                "y_ticks": [0.0, 10.0, 20.0, 30.0],
                "show_individual_traces": False,
                "show_ci_band": True,
                "show_markers": True,
                "colors": {
                    "P_advance": "transition_advance",
                    "P_recruit": "transition_recruit",
                    "P_loss": "transition_loss",
                },
                "role": "show group-level local transition probabilities",
                "legend_owner": "panel",
            },
            "d": {
                "claim": "Layer-1 STSP contributes to early advance or recruitment",
                "chart": "category_points",
                "source": "data/panel_d_plot_data.csv",
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
                "role": "test the STSP contribution to early processing",
                "legend_owner": "none",
            },
            "e": {
                "claim": "Current processing writes a history-dependent Layer-2 state",
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
                "claim": "Layer-1-only u/x exchange redirects the Layer-2 successor",
                "chart": "category_points",
                "source": "data/panel_f_plot_data.csv",
                "x_field": "endpoint",
                "x_order": ["layer1_only_layer2_update_donor_transfer"],
                "x_labels": {
                    "layer1_only_layer2_update_donor_transfer": "L2 successor",
                },
                "y_label": "Donor-transfer index",
                "y_limits": [0.0, 1.0],
                "y_ticks": [0.0, 0.25, 0.5, 0.75, 1.0],
                "references": [{"value": 0.0}],
                "color_by_x": True,
                "mean_marker": "o",
                "mean_marker_filled": True,
                "colors": {
                    "layer1_only_layer2_update_donor_transfer": "donor_trace",
                },
                "role": "close bounded downstream successor formation",
                "legend_owner": "none",
            },
            "g": {
                "claim": (
                    "Successive inputs selectively read, decay, and rewrite "
                    "the inherited STSP state"
                ),
                "chart": "svg_asset",
                "source": "meta/panel_g_asset_manifest.csv",
                "asset_embedding": "inline",
                "asset_viewbox_override": "0 0 1560 420",
                "asset_top_padding_mm": 3.0,
                "role": (
                    "synthesize the local mechanism as a reusable "
                    "state-transition unit"
                ),
                "color_roles": [
                    "mechanism_teal",
                    "dynamic",
                    "baseline_control",
                    "fused_state",
                ],
                "legend_owner": "none",
            },
        },
    },
    "fig4": {
        "chain_role": "recur",
        "canvas_mm": [165.0, 102.0],
        "figure_question": (
            "Does a post-B Layer-2 successor condition an identical C and seed the "
            "next successor, while state updating recurs across longer sequences?"
        ),
        "terminal_inference": (
            "Under the tested intervention, the post-B Layer-2 STSP successor is sufficient "
            "to redirect early Layer-2 processing of an identical C and the post-C Layer-3 "
            "successor; input-associated updating remains detectable across later stages "
            "while behavioral interference accumulates."
        ),
        "forbidden_inferences": [
            "C5 establishes necessity, complete mediation, or uniqueness",
            "every progressive boundary repeats the complete C5 intervention",
            "the persisted passive scalar provides complete passive T0 arrays",
            "Fig.4 causally generates either Fig.5 morphology or Fig.6 function",
        ],
        "slots": FIG4_SLOTS,
        "comparison_obligations": [
            {
                "panels": ["a", "b"],
                "comparison": "C5 processing consequence versus next-successor consequence",
                "reader_action": "read the same K1/K5 intervention across the two endpoints",
            },
            {
                "panels": ["c", "d"],
                "comparison": "recurrence breadth and accumulated behavioral consequence",
                "reader_action": "keep the persisted passive boundary separate from relation-balanced outcomes",
            },
        ],
        "panels": {
            "a": {
                "claim": "The transplanted successor redirects early Layer-2 processing of C",
                "chart": "category_points",
                "source": "data/panel_a_plot_data.csv",
                "x_field": "condition",
                "x_order": ["K1", "K5"],
                "x_labels": {"K1": "K1", "K5": "K5"},
                "y_label": "Donor-transfer index",
                "y_limits": [0.0, 0.40],
                "y_ticks": [0.0, 0.1, 0.2, 0.3, 0.4],
                "references": [{"value": 0.0}],
                "show_raw_points": True,
                "color_by_x": True,
                "use_persisted_ci": True,
                "plot_bbox_mm": [13.0, 12.0, 65.5, 28.0],
                "colors": {"K1": "donor_trace", "K5": "fused_state"},
                "role": "establish the next-input processing consequence",
                "legend_owner": "none",
            },
            "b": {
                "claim": "The transplanted successor redirects the post-C Layer-3 successor",
                "chart": "category_points",
                "source": "data/panel_b_plot_data.csv",
                "x_field": "condition",
                "x_order": ["K1", "K5"],
                "x_labels": {"K1": "K1", "K5": "K5"},
                "y_label": "Donor-transfer index",
                "y_limits": [0.0, 0.35],
                "y_ticks": [0.0, 0.1, 0.2, 0.3],
                "references": [{"value": 0.0}],
                "show_raw_points": True,
                "color_by_x": True,
                "use_persisted_ci": True,
                "plot_bbox_mm": [94.5, 12.0, 65.5, 28.0],
                "colors": {"K1": "donor_trace", "K5": "fused_state"},
                "role": "establish the next-successor consequence",
                "legend_owner": "none",
            },
            "c": {
                "claim": "Input-associated state displacement recurs across stages 2–10",
                "chart": "ordered_lines",
                "source": "data/panel_c_plot_data.csv",
                "x_field": "stage_k",
                "x_order": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "x_ticks": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "numeric_x": True,
                "x_limits": [1.8, 10.25],
                "hue_field": "condition",
                "hue_order": ["observed", "passive"],
                "hue_labels": {
                    "observed": "Observed",
                    "passive": "Persisted passive",
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
                "plot_bbox_mm": [13.0, 62.0, 65.5, 28.0],
                "colors": {"observed": "dynamic", "passive": "baseline_control"},
                "role": "generalize recurrence breadth with the passive-lineage boundary",
                "legend_owner": "none",
            },
            "d": {
                "claim": "Accumulated history lowers rescue and raises loss",
                "chart": "grouped_bars",
                "source": "data/panel_d_plot_data.csv",
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
                "plot_bbox_mm": [94.5, 62.0, 65.5, 28.0],
                "colors": {"rescue": "dynamic", "loss": "donor_trace"},
                "role": "show accumulated behavioral interference without mixing denominators",
                "legend_owner": "panel",
            },
        },
    },
    "fig5": {
        "chain_role": "parallel_outcome_morphology",
        "canvas_mm": [165.0, 152.0],
        "figure_question": "What terminal morphology is retained after a sequence?",
        "terminal_inference": (
            "Across the corresponding pair and multi-item protocols, retained STSP "
            "has bounded multi-component, history-specific, load- and delay-dependent "
            "morphology."
        ),
        "forbidden_inferences": [
            "structural metrics are functional recall",
            "N_eff is a capacity or accessible-item count",
            "latest-item weights establish method-independent primacy or recency",
            "Fig.6 accesses the morphology defined here",
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
                "plot_bbox_mm": [13.0, 12.0, 65.5, 28.0],
                "colors": {
                    "item_a": "first_item_reference",
                    "item_b": "second_item_reference",
                },
                "role": "establish pair component retention",
                "legend_owner": "none",
            },
            "b": {
                "claim": "Experienced pairs exceed one-constituent-held shuffles",
                "chart": "boxplot",
                "source": "data/panel_b_plot_data.csv",
                "x_field": "condition",
                "x_order": ["experienced_pair", "shuffled_pair"],
                "x_labels": {
                    "experienced_pair": "Experienced\npair",
                    "shuffled_pair": "Held-item\nshuffle",
                },
                "y_label": "Pair similarity",
                "y_limits": [0.98, 1.0],
                "y_ticks": [0.98, 0.99, 1.0],
                "plot_bbox_mm": [94.5, 12.0, 65.5, 28.0],
                "colors": {
                    "experienced_pair": "true_pair",
                    "shuffled_pair": "shuffled_pair",
                },
                "role": "establish experienced-pair specificity",
                "legend_owner": "none",
            },
            "c": {
                "claim": "Effective component number increases with sequence load",
                "chart": "ordered_lines",
                "source": "data/panel_c_plot_data.csv",
                "x_field": "seq_len",
                "x_order": [3, 5, 7, 10],
                "x_label": "Items (K)",
                "y_label": "Effective components",
                "y_limits": [0.0, 10.5],
                "y_ticks": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                "identity_reference": True,
                "identity_reference_label": "$N_{eff}=K$",
                "show_individual_traces": False,
                "plot_bbox_mm": [13.0, 62.0, 65.5, 28.0],
                "colors": {"single": "dynamic"},
                "role": "establish multi-component terminal state without a capacity claim",
                "legend_owner": "none",
            },
            "d": {
                "claim": "The latest item does not dominate the terminal state",
                "chart": "ordered_lines",
                "source": "data/panel_d_plot_data.csv",
                "x_field": "seq_len",
                "x_order": [3, 5, 7, 10],
                "x_label": "Items (K)",
                "y_label": "Latest-item weight",
                "y_limits": [0.0, 0.60],
                "y_ticks": [0.0, 0.2, 0.4, 0.6],
                "references": [{"value": 0.5, "label": "latest-only"}],
                "show_individual_traces": False,
                "plot_bbox_mm": [94.5, 62.0, 65.5, 28.0],
                "colors": {"single": "fused_state"},
                "role": "exclude latest-item-only collapse without a recency-direction claim",
                "legend_owner": "none",
            },
            "e": {
                "claim": "Layer-1 g spatial footprint varies with load and delay",
                "chart": "heatmap",
                "source": "data/panel_e_plot_data.csv",
                "x_field": "seq_len",
                "x_order": [3, 5, 7, 10],
                "y_field": "delay_ms",
                "y_order": [800, 400, 200, 100],
                "aggregate": "mean",
                "x_label": "Items (K)",
                "y_label": "Delay (ms)",
                "colorbar_label": "Effective area",
                "cmap": "stsp_support",
                "vmin": 0.0,
                "vmax": 0.60,
                "unavailable_color": "#FFFFFF",
                "cell_edges": False,
                "annotate_cells": True,
                "annotation_decimals": 2,
                "plot_bbox_mm": [13.0, 112.0, 65.5, 28.0],
                "colorbar_orientation": "horizontal_top",
                "colorbar_height_mm": 1.4,
                "colorbar_gap_mm": 0.8,
                "colorbar_ticks_position": "top",
                "colorbar_label_position": "top",
                "colorbar_label_pad_pt": 1.0,
                "role": "map coefficient-free spatial footprint",
                "legend_owner": "colorbar",
            },
            "f": {
                "claim": "Morphology remains history-specific across load and delay",
                "chart": "heatmap",
                "source": "data/panel_f_plot_data.csv",
                "x_field": "seq_len",
                "x_order": [3, 5, 7, 10],
                "y_field": "delay_ms",
                "y_order": [800, 400, 200, 100],
                "aggregate": "mean",
                "x_label": "Items (K)",
                "y_label": "Delay (ms)",
                "colorbar_label": "Matched − deranged",
                "cmap": "stsp_support",
                "vmin": 0.0,
                "vmax": 0.45,
                "unavailable_color": "#FFFFFF",
                "cell_edges": False,
                "annotate_cells": True,
                "annotation_decimals": 2,
                "plot_bbox_mm": [94.5, 112.0, 65.5, 28.0],
                "colorbar_orientation": "horizontal_top",
                "colorbar_height_mm": 1.4,
                "colorbar_gap_mm": 0.8,
                "colorbar_ticks_position": "top",
                "colorbar_label_position": "top",
                "colorbar_label_pad_pt": 1.0,
                "role": "map coefficient-free morphology specificity",
                "legend_owner": "colorbar",
            },
        },
    },
    "fig6": {
        "chain_role": "parallel_outcome_function",
        "canvas_mm": [165.0, 152.0],
        "figure_question": (
            "Under which later-input or cue conditions does retained STSP alter readout "
            "or recruitment?"
        ),
        "terminal_inference": (
            "In independent later-input and cue protocols, retained STSP conditionally "
            "alters readout and recruitment as a function of content, load, delay, and "
            "local pathway overlap."
        ),
        "forbidden_inferences": [
            "STSP replays memory without a cue",
            "all retained items are perfectly readable",
            "load or delay has a universally monotonic main effect",
            "Fig.6 accesses the morphology defined by Fig.5",
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
                    "User approved the Target A/B internal split on 2026-07-31; the "
                    "revised figure retains that local exception inside an otherwise "
                    "equal two-column grid."
                ),
                "plot_bbox_mm": [2.0, 12.0, 79.5, 28.0],
                "child_plot_bboxes_mm": [
                    [13.0, 12.0, 31.0, 28.0],
                    [47.0, 12.0, 31.0, 28.0],
                ],
                "legend_anchor": [0.5, 1.08],
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
                "plot_bbox_mm": [94.5, 12.0, 65.5, 28.0],
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
                "hue_order": ["matched", "same_label_novel", "unseen"],
                "hue_labels": {
                    "matched": "Matched",
                    "same_label_novel": "Same-label\nnovel",
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
                    "same_label_novel": "single_item_memory",
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
                "claim": "High-STSP-overlap sites contribute under exact matching",
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
                "plot_bbox_mm": [13.0, 112.0, 65.5, 28.0],
                "colors": {
                    "high_stsp_overlap": "high_stsp",
                    "matched_removal": "baseline_control",
                },
                "role": "establish targeted causal contribution",
                "legend_owner": "none",
            },
            "f": {
                "claim": "STSP changes early firing only along overlapping input pathways",
                "chart": "two_by_two",
                "source": "data/panel_f_plot_data.csv",
                "x_field": "overlap_group",
                "x_order": ["no_overlap", "overlap"],
                "x_labels": {
                    "no_overlap": "No overlap",
                    "overlap": "Overlap",
                },
                "hue_field": "stsp_group",
                "hue_order": ["low", "high"],
                "hue_labels": {"low": "Low STSP", "high": "High STSP"},
                "y_label": "Firing change (pp)",
                "y_limits": [0.0, 25.0],
                "y_ticks": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
                "show_raw_points": True,
                "show_contrast_panel": False,
                "references": [{"value": 0.0}],
                "colors": {"low": "low_stsp", "high": "high_stsp"},
                "plot_bbox_mm": [94.5, 112.0, 65.5, 28.0],
                "role": "show the structural zero and overlap-conditioned expression",
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

    def plot_bbox(
        panel: dict[str, Any],
        slot: list[float],
    ) -> list[float]:
        explicit = panel.get("plot_bbox_mm")
        if explicit is not None:
            return [float(value) for value in explicit]
        x_mm, y_mm, width_mm, height_mm = [float(value) for value in slot]
        chart = str(panel["chart"])
        if chart in {"svg_asset", "protocol", "schematic"}:
            left, right, top, bottom = 5.0, 4.0, 5.0, 3.0
        elif chart in {"forest", "estimate_strip"}:
            left, right, top, bottom = 27.0, 3.0, 7.0, 9.0
        elif chart == "heatmap":
            left, right, top, bottom = 11.0, 11.0, 7.0, 9.0
        else:
            left, right, top, bottom = 11.0, 3.0, 8.0, 10.0
        return [
            x_mm + left,
            y_mm + top,
            width_mm - left - right,
            height_mm - top - bottom,
        ]

    for panel_id, panel in panels.items():
        slot = spec["slots"][panel_id]
        x_mm, y_mm, width_mm, height_mm = slot
        rows.setdefault(y_mm, []).append(panel_id)
        chart = panel["chart"]
        geometry[panel_id] = {
            "chart_family": chart,
            "category_slots": max(1, len(panel.get("x_order", panel.get("category_order", [1])))),
            "natural_aspect": [1.1, 4.0] if width_mm > 100 else [1.1, 2.2],
            "decoration_sides": (
                []
                if chart in {"svg_asset", "protocol", "schematic"}
                else ["bottom"]
                if chart == "estimate_strip"
                else ["left", "right", "bottom"]
                if chart == "heatmap"
                else ["left", "bottom", "top"]
            ),
            "visual_weight": (
                "high"
                if chart in {"svg_asset", "protocol", "schematic", "heatmap"}
                else "low"
                if chart == "estimate_strip"
                else "medium"
            ),
            "slot_bbox_mm": [x_mm, y_mm, width_mm, height_mm],
            "plot_bbox_mm": plot_bbox(panel, slot),
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
                    "rationale": "Align the two C5 causal consequences.",
                    "comparison_basis": "Both panels show K1/K5 donor-transfer endpoints in equal slots.",
                },
                {
                    "group_id": "fig4_row_2_plot_axes",
                    "panels": ["c", "d"],
                    "target": "plot_area",
                    "edges": ["top", "bottom"],
                    "rationale": "Align recurrence breadth with accumulated behavioral consequences.",
                    "comparison_basis": "Both occupy 28 mm quantitative regions in equal slots.",
                },
            ]
        )
    elif spec.get("figure_id") == "fig5":
        alignment_groups.extend(
            [
                {
                    "group_id": "fig5_row_1_plot_axes",
                    "panels": ["a", "b"],
                    "target": "plot_area",
                    "edges": ["top", "bottom"],
                    "rationale": "Align constituent retention and pair specificity.",
                    "comparison_basis": "Both panels use 28 mm-high pair-state data regions.",
                },
                {
                    "group_id": "fig5_row_2_plot_axes",
                    "panels": ["c", "d"],
                    "target": "plot_area",
                    "edges": ["top", "bottom"],
                    "rationale": "Align the two bounded multi-component summaries.",
                    "comparison_basis": "Both panels use the same K axis and 28 mm-high data regions.",
                },
                {
                    "group_id": "fig5_row_3_plot_axes",
                    "panels": ["e", "f"],
                    "target": "plot_area",
                    "edges": ["top", "bottom"],
                    "rationale": "Align the two coefficient-free K-by-delay maps.",
                    "comparison_basis": "Both heatmaps use equal slots and independent top colorbars.",
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
                    "comparison_basis": "Both panels occupy equal slots; panel a alone contains the approved A/B internal split.",
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
        "grid_policy": {
            "equal_row_heights": True,
            "equal_width_within_row": True,
            "panel_atomicity": True,
        },
        "approved_exceptions": (
            [
                {
                    "approved_exception": True,
                    "scope": "fig6.panel_a.target_pair",
                    "panels": ["a"],
                    "released_constraint": "panel_atomicity",
                    "rationale": (
                        "The user-approved Target A/B split is retained inside panel a; "
                        "all outer panel slots remain equal within each row."
                    ),
                }
            ]
            if spec.get("figure_id") == "fig6"
            else []
        ),
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
            "released_alignment_edges": [],
        },
        "hard_constraints": [
            f"{canvas_width:g} mm x {canvas_height:g} mm canvas",
            "frozen slots and panel order",
            "equal row heights and equal widths within every row",
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
