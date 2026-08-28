from __future__ import annotations


FIGURE_ID = "fig4_overlap_reentry"
FIG4_DESIGN_VERSION = "overlap_gated_reentry_causal_decision_dynamics"
NUM_CLASSES = 10
CORE_CONDITIONS = (
    "full_dynamic",
    "full_static",
    "sample_keep_overlap_only_dynamic",
    "sample_keep_nonoverlap_only_dynamic",
    "sample_random_matched_dynamic",
)
D_L1_STSP_CONDITIONS = (
    "full_static",
    "full_dynamic_intact",
    "l1_overlap_reset",
    "l1_nonoverlap_reset",
    "l1_random_matched_reset",
)
CONDITION_LABELS = {
    "full_dynamic": "Dynamic",
    "full_static": "Static",
    "sample_keep_overlap_only_dynamic": "Overlap support",
    "sample_keep_nonoverlap_only_dynamic": "Non-overlap support",
    "sample_random_matched_dynamic": "Random matched",
    "full_dynamic_intact": "Dynamic",
    "l1_overlap_reset": "L1 overlap reset",
    "l1_nonoverlap_reset": "L1 non-overlap reset",
    "l1_random_matched_reset": "L1 random reset",
}
D_L1_STSP_CONDITION_LABELS = {
    "full_static": "Static baseline",
    "full_dynamic_intact": "Dynamic",
    "l1_overlap_reset": "L1 overlap reset",
    "l1_nonoverlap_reset": "L1 non-overlap reset",
    "l1_random_matched_reset": "L1 random reset",
}
SAMPLE_SIDE_MASKS = {
    "full_dynamic": "full_sample",
    "full_static": "full_sample",
    "sample_keep_overlap_only_dynamic": "sample_nonoverlap_mask",
    "sample_keep_nonoverlap_only_dynamic": "sample_overlap_mask",
    "sample_random_matched_dynamic": "random_matched_keep_support",
}
FIG4_MAIN_PANELS = {
    "A": "DMS sample-delay-probe / overlap-gated re-entry schematic",
    "B": "sample-probe similarity dependence of prior-history effect",
    "C": "highest-similarity-bin overlap dependence of accuracy drop",
    "D": "pre-probe layer1 STSP overlap reset accuracy drop",
    "E": "time-resolved L3 and decision-spike displacement",
    "F": "L3 accumulator replay / decision trajectory deflection",
}
FIG4_SUMMARY_PANELS = {
    "A": "DMS / overlap-gated re-entry schematic",
    "B": "similarity dependence",
    "C": "highest-similarity overlap accuracy drop",
    "D": "L1 STSP overlap reset",
    "E": "L3 / decision-spike displacement",
    "F": "L3 accumulator replay / decision trajectory deflection",
}
FIG4_LEGACY_METHODS = {
    "B": "similarity_bias_experiment-compatible snapshot readout",
    "C": "legacy overlap localization and iso-similarity controls",
    "D": "legacy overlap_causal_input_perturbation-compatible encoded-spike sample-side perturbation",
    "E": "probe_l3_trace / s2p DPI",
    "F": "l3_accumulator_mechanism-compatible L3 region deletion/replacement replay",
}
FIG4_SUPPLEMENT_PLAN = {
    "S7": "overlap transition and similarity-dissociation controls",
    "S8": "decision-dynamics and trajectory-deflection controls",
}
FIG4_MAIN_REQUIRED_OUTPUTS = [
    "data/metrics/panel_b_similarity_entry_metrics.csv",
    "data/metrics/panel_b_similarity_bin_summary.csv",
    "data/metrics/panel_b_similarity_accuracy_drop_summary.csv",
    "data/metrics/panel_c_high_similarity_overlap_accuracy_drop.csv",
    "data/metrics/panel_c_high_similarity_overlap_accuracy_drop_summary.csv",
    "data/metrics/panel_c_high_similarity_overlap_accuracy_drop_contrast.csv",
    "data/raw/panel_d_l1_stsp_overlap_perturbation_trial_readout.csv",
    "data/metrics/panel_d_l1_stsp_overlap_perturbation_summary.csv",
    "data/metrics/panel_d_l1_stsp_overlap_perturbation_contrast.csv",
    "data/metrics/panel_d_l1_stsp_overlap_perturbation_audit.csv",
    "data/metrics/panel_e_time_resolved_l3_displacement.csv",
    "data/metrics/panel_e_decision_spike_displacement.csv",
    "data/metrics/panel_f_l3_accumulator_region_replay_metrics.csv",
    "data/metrics/panel_f_l3_accumulator_summary.csv",
]
FIG4_S7_OUTPUTS = [
    "data/metrics/supp_s7_similarity_bin_full_trend.csv",
    "data/metrics/supp_s7_overlap_matching_diagnostics.csv",
    "data/metrics/supp_s7_iso_similarity_overlap_contrast.csv",
    "data/metrics/supp_s7_iso_similarity_permutation_null.csv",
    "data/metrics/supp_s7_overlap_regression_controls.csv",
    "data/metrics/supp_s7_random_nonoverlap_perturbation_controls.csv",
]
FIG4_S8_OUTPUTS = [
    "data/metrics/supp_s8_time_resolved_l3_displacement.csv",
    "data/metrics/supp_s8_decision_spike_displacement.csv",
    "data/metrics/supp_s8_l3_accumulator_replay_metrics.csv",
    "data/metrics/supp_s8_l3_accumulator_summary.csv",
    "data/metrics/supp_s8_decision_deflection_metrics.csv",
    "data/metrics/supp_s8_decision_deflection_summary.csv",
]
FIG4_COMPATIBILITY_OUTPUTS = [
    "data/metrics/panel_c_overlap_localization_metrics.csv",
    "data/metrics/panel_c_overlap_matched_comparison.csv",
    "data/metrics/panel_d_overlap_perturbation_metrics.csv",
    "data/metrics/panel_d_overlap_perturbation_summary.csv",
    "data/metrics/panel_d_overlap_perturbation_contrast.csv",
    "data/metrics/panel_d_overlap_accuracy_pair_table.csv",
    "data/metrics/panel_d_iso_similarity_matched_pairs.csv",
    "data/metrics/panel_d_overlap_accuracy_permutation_null.csv",
    "data/metrics/panel_d_overlap_accuracy_contrast_by_network.csv",
    "data/metrics/panel_d_matching_balance_diagnostics.csv",
    "data/metrics/supp_overlap_preserving_perturbation_metrics.csv",
    "data/metrics/supp_overlap_preserving_perturbation_summary.csv",
    "data/metrics/supp_decision_deflection_metrics.csv",
]
PERTURBATION_CONDITION_MAP = {
    "overlap": "sample_keep_overlap_only_dynamic",
    "nonoverlap": "sample_keep_nonoverlap_only_dynamic",
    "random": "sample_random_matched_dynamic",
    "dynamic": "full_dynamic",
    "static": "full_static",
}


__all__ = [name for name in globals() if name.isupper()]
