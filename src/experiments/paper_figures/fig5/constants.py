from __future__ import annotations


FIGURE_ID = "fig5_local_support_competition"

FIG5_DESIGN_VERSION = "local_support_competition_l1_stsp_perturbation"

PRIMARY_LAYER = "layer1"

MAX_WINNERS_PER_TRIAL = 3

PRIMARY_PRE_WINDOW_MS = (-8.0, -1.0)

LATE_PRE_WINDOW_MS = (-4.0, -1.0)

UNIT_GROUPS = ("overlap_dominant", "probe_only_dominant", "balanced", "random_matched")

MAIN_CONDITIONS = (
    "dynamic_intact",
    "attenuate_l1_stsp",
    "reset_l1_stsp",
)

L1_STSP_PERTURBATION_CONDITIONS = (
    "attenuate_l1_stsp",
    "reset_l1_stsp",
)

LEGACY_REGION_PERTURBATION_CONDITIONS = (
    "dynamic_intact",
    "attenuate_overlap_high_support",
    "reset_overlap_high_support",
)

REFERENCE_CONDITIONS = (
    "static_frozen",
)

SUPP_CONDITIONS = (
    "sham_perturbation",
)

REMOVED_FROM_MAIN_CONDITIONS = (
    "flatten_overlap_high_support",
    "flatten_nonoverlap_high_support",
    "flatten_random_high_support_matched",
)

PERTURBATION_MAIN_CONDITIONS = {
    "dynamic": "dynamic_intact",
    "static": "static_frozen",
    "attenuate": "attenuate_l1_stsp",
    "reset": "reset_l1_stsp",
    "sham": "sham_perturbation",
}

MAIN_PANEL_DESCRIPTIONS = {
    "A": "pre-probe overlap-aligned STSP support",
    "B": "dynamic-vs-static early spike transition",
    "C": "winner-loser event-aligned voltage and inhibition",
    "D": "Layer1 STSP perturbation transition composition",
}

MAIN_CLAIM = (
    "Overlap-aligned STSP support biases early recruitment and local competition; "
    "Layer1 STSP attenuation/reset alters dynamic Layer1 transition composition."
)

SUPPLEMENT_PLAN = {
    "S9": "local firing-transition and event-chain controls",
    "S10": "support-perturbation causal controls",
}

FIG5_MAIN_REQUIRED_OUTPUTS = [
    "data/metrics/panel_a_preprobe_support_metrics.csv",
    "data/metrics/panel_b_early_firing_transition_metrics.csv",
    "data/metrics/panel_b_transition_summary_by_group.csv",
    "data/metrics/panel_c_winner_loser_event_metrics.csv",
    "data/metrics/panel_c_event_trace_summary.csv",
    "data/metrics/panel_d_l1_stsp_perturbation_unit_transitions.csv",
    "data/metrics/panel_d_l1_stsp_perturbation_transition_summary.csv",
    "data/metrics/panel_d_l1_stsp_perturbation_audit.csv",
    "data/metrics/panel_d_l1_stsp_perturbation_contrast.csv",
]

FIG5_S9_OUTPUTS = [
    "data/metrics/supp_early_window_robustness.csv",
    "data/metrics/supp_s9_transition_composition_by_group.csv",
    "data/metrics/supp_s9_event_trace_summary.csv",
    "data/metrics/supp_event_chain_fraction_metrics.csv",
    "data/metrics/supp_event_chain_null_baselines.csv",
    "data/metrics/supp_s9_event_chain_null_summary.csv",
    "data/metrics/supp_s9_neighborhood_radius_robustness.csv",
    "data/metrics/supp_s9_event_selection_audit.csv",
]

FIG5_S10_OUTPUTS = [
    "data/metrics/supp_s10_perturbation_ux_audit.csv",
    "data/metrics/supp_s10_perturbation_transition_contrast.csv",
    "data/metrics/supp_s10_same_winner_disruption.csv",
    "data/metrics/supp_s10_dynamic_like_recovery_after_perturbation.csv",
    "data/metrics/supp_s10_support_perturbation_controls.csv",
    "data/metrics/supp_s10_perturbation_matching_diagnostics.csv",
]

FIG5_BACKWARD_COMPATIBLE_OUTPUTS = [
    "data/metrics/panel_a_preprobe_support_metrics.csv",
    "data/metrics/panel_b_early_firing_transition_metrics.csv",
    "data/metrics/panel_b_transition_summary_by_group.csv",
    "data/metrics/panel_c_winner_loser_event_metrics.csv",
    "data/metrics/panel_c_event_trace_summary.csv",
    "data/raw/panel_c_event_aligned_traces.npz",
    "data/metrics/panel_d_perturbation_unit_transitions.csv",
    "data/metrics/panel_d_perturbation_transition_summary_by_group.csv",
    "data/metrics/panel_d_perturbation_transition_contrast.csv",
    "data/metrics/supp_perturbation_ux_audit.csv",
    "data/metrics/supp_support_perturbation_controls.csv",
    "data/metrics/supp_perturbation_matching_diagnostics.csv",
]

NULL_TYPES = (
    "event_time_shuffle",
    "winner_loser_pairing_shuffle",
    "neighborhood_shuffle",
    "dynamic_static_label_shuffle",
    "trial_shuffle",
)

TRIAL_COLUMNS = [
    "network_seed",
    "trial_id",
    "sample_image_id",
    "sample_label",
    "probe_image_id",
    "probe_label",
    "sample_foreground_area",
    "probe_foreground_area",
    "sample_entry_area",
    "probe_entry_area",
    "overlap_area",
    "probe_only_area",
    "overlap_quantile",
    "selected_trial_group",
    "input_energy_sample",
    "input_energy_probe",
    "pixel_similarity",
    "dice_overlap",
    "overlap_mask_mode",
    "class_pair",
    "trial_seed",
]

UNIT_GROUP_COLUMNS = [
    "network_seed",
    "trial_id",
    "layer",
    "unit_id",
    "row",
    "col",
    "unit_group",
    "overlap_drive_score",
    "probe_only_drive_score",
    "support_value",
    "is_overlap_dominant",
    "is_probe_only_dominant",
    "is_random_matched",
]

PERTURBATION_UNIT_COLUMNS = [
    "network_seed",
    "trial_id",
    "condition",
    "unit_id",
    "unit_group",
    "original_support",
    "perturbed_support",
    "support_delta",
    "row",
    "col",
    "matched_to_condition",
    "matching_error_support",
    "matching_error_spike_count",
    "intervention_timing",
    "probe_input_changed",
]

PERTURBATION_UX_AUDIT_COLUMNS = [
    "network_seed",
    "trial_id",
    "condition",
    "unit_id",
    "row",
    "col",
    "u_before_mean",
    "x_before_mean",
    "g_before_mean",
    "u_after_mean",
    "x_after_mean",
    "g_after_mean",
    "u_delta_mean",
    "x_delta_mean",
    "g_delta_mean",
]

PANEL_A_COLUMNS = ["network_seed", "trial_id", "unit_group", "layer", "state_variable", "mean_support", "total_support", "support_area", "support_enrichment", "overlap_minus_probe_only_support", "n_units"]

PANEL_B_UNIT_COLUMNS = ["network_seed", "trial_id", "unit_id", "unit_group", "early_window_ms", "transition_type", "first_spike_dynamic", "first_spike_static", "delta_first_spike_latency", "early_spike_count_dynamic", "early_spike_count_static", "delta_early_spike_count"]

PANEL_B_SUMMARY_COLUMNS = ["network_seed", "trial_id", "unit_group", "early_window_ms", "P_advance", "P_recruit", "P_loss", "P_unchanged", "P_advance_plus_recruit", "mean_delta_early_spike_count", "mean_delta_first_spike_latency", "n_units"]

PANEL_C_EVENT_COLUMNS = ["network_seed", "trial_id", "event_id", "winner_unit_idx", "loser_unit_idx", "winner_group", "loser_group", "winner_first_spike_dynamic", "winner_first_spike_static", "loser_first_spike_dynamic", "loser_first_spike_static", "winner_pre_spike_delta_v_mean", "winner_pre_spike_boost", "winner_spikes_earlier", "loser_post_winner_delta_v_mean", "loser_post_winner_inh_rise", "loser_post_winner_suppressed", "winner_loser_latency_gap", "neighborhood_radius", "local_distance"]

PANEL_C_TRACE_COLUMNS = ["network_seed", "time_ms", "trace_type", "mean_value", "sem_value", "n_events"]

PANEL_D_UNIT_TRANSITION_COLUMNS = ["network_seed", "trial_id", "condition", "unit_id", "unit_group", "row", "col", "first_spike_static", "first_spike_same", "first_spike_condition", "transition_vs_static", "transition_vs_same", "same_winner", "condition_winner", "same_winner_preserved", "same_winner_delayed", "same_winner_lost", "same_winner_reverted_to_static", "same_winner_lost_or_delayed", "delta_latency_vs_static", "delta_latency_vs_same", "early_spike_count_static", "early_spike_count_same", "early_spike_count_condition", "delta_early_spike_count_vs_static", "delta_early_spike_count_vs_same"]

PANEL_D_TRANSITION_SUMMARY_COLUMNS = ["network_seed", "trial_id", "condition", "unit_group", "P_advance", "P_recruit", "P_loss", "P_unchanged", "P_advance_plus_recruit", "P_same_winner_preserved", "P_same_winner_delayed", "P_same_winner_lost", "P_same_winner_reverted_to_static", "P_same_winner_lost_or_delayed", "mean_delta_latency_vs_static", "mean_delta_latency_vs_same", "mean_delta_early_spike_count_vs_static", "mean_delta_early_spike_count_vs_same", "n_units", "n_same_winner_units"]

PANEL_D_TRANSITION_CONTRAST_COLUMNS = ["network_seed", "trial_id", "unit_group", "attenuate_delta_P_advance_plus_recruit", "reset_delta_P_advance_plus_recruit", "attenuate_delta_P_loss", "reset_delta_P_loss", "attenuate_delta_P_same_winner_lost_or_delayed", "reset_delta_P_same_winner_lost_or_delayed", "reset_minus_attenuate_delta_P_advance_plus_recruit", "attenuate_delta_latency_vs_same", "reset_delta_latency_vs_same", "n_units", "n_trials"]

PANEL_D_L1_STSP_UNIT_COLUMNS = ["network_seed", "trial_id", "condition", "condition_label", "unit_id", "unit_group", "layer_or_map", "row", "col", "included_in_main", "first_spike_static", "first_spike_condition", "transition_vs_static", "early_spike_count_static", "early_spike_count_condition", "delta_early_spike_count_vs_static", "perturbation_mode", "perturbed_layer", "perturbed_variables"]

PANEL_D_L1_STSP_SUMMARY_COLUMNS = ["network_seed", "condition", "condition_label", "P_advance", "P_recruit", "P_loss", "P_unchanged", "P_advance_plus_recruit", "transition_mass", "n_units", "n_trials", "included_unit_groups", "perturbation_mode", "perturbed_layer", "perturbed_variables"]

PANEL_D_L1_STSP_AUDIT_COLUMNS = ["network_seed", "trial_id", "condition", "perturbation_mode", "perturbed_layer", "perturbed_variables", "n_l1_stsp_sites", "l1_u_before_mean", "l1_u_after_mean", "l1_u_delta_mean", "l1_x_before_mean", "l1_x_after_mean", "l1_x_delta_mean", "l1_u_before_std", "l1_u_after_std", "l1_x_before_std", "l1_x_after_std", "layer1_perturbed", "layer2_perturbed", "layer3_perturbed", "restore_ok", "perturbation_ok"]

PANEL_D_L1_STSP_CONTRAST_COLUMNS = ["network_seed", "dynamic_transition_mass", "attenuate_transition_mass", "reset_transition_mass", "dynamic_minus_attenuate_transition_mass", "dynamic_minus_reset_transition_mass", "attenuate_minus_reset_transition_mass", "dynamic_P_advance", "attenuate_P_advance", "reset_P_advance", "dynamic_P_recruit", "attenuate_P_recruit", "reset_P_recruit", "dynamic_P_loss", "attenuate_P_loss", "reset_P_loss"]

PANEL_D_NODE_COLUMNS = ["network_seed", "trial_id", "condition", "perturbed_unit_group", "n_perturbed_units", "mean_pre_perturb_support", "mean_post_perturb_support", "P_advance", "P_recruit", "P_advance_plus_recruit", "delta_early_spike_count", "delta_first_spike_latency", "winner_pre_spike_delta_v_mean", "winner_pre_spike_boost", "loser_post_winner_inh_rise", "loser_post_winner_delta_v_mean", "loser_post_winner_suppressed", "spike_pattern_displacement", "dynamic_like_spike_similarity", "decision_deflection_score", "dynamic_like_readout_recovery"]

PANEL_D_TRIAL_COLUMNS = ["network_seed", "trial_id", "condition", "prediction", "probe_prediction", "probe_correct", "pred_matches_dynamic", "pred_matches_static", "first_fire_time_ms", "first_fire_time", "spike_count", "early_spike_count", "total_spike_count", "dynamic_like_spike_similarity", "dynamic_like_readout_recovery", "decision_deflection_score"]

PANEL_E_COLUMNS = ["network_seed", "node", "metric", "dynamic_intact_value", "overlap_perturbed_value", "random_perturbed_value", "nonoverlap_perturbed_value", "static_value", "overlap_disruption", "random_disruption", "nonoverlap_disruption", "normalized_overlap_disruption"]

PANEL_D_EFFECT_SUMMARY_COLUMNS = ["network_seed", "metric", "dynamic_value", "static_value", "attenuate_value", "reset_value", "sham_value", "attenuate_disruption_vs_dynamic", "reset_disruption_vs_dynamic", "sham_disruption_vs_dynamic", "attenuate_recovery_toward_static", "reset_recovery_toward_static", "reset_minus_attenuate_disruption", "n_trials", "metric_direction", "notes"]

SUPP_EVENT_AUDIT_COLUMNS = ["network_seed", "trial_id", "event_id", "selection_step", "included", "exclusion_reason", "winner_group", "loser_group", "neighborhood_radius", "drive_score_winner", "drive_score_loser"]

SUPP_NULL_COLUMNS = ["network_seed", "null_type", "metric", "observed_value", "null_mean", "null_p95", "observed_minus_null", "empirical_p", "n_null"]

SUPP_S9_TRANSITION_COMPOSITION_COLUMNS = ["network_seed", "unit_group", "P_advance", "P_recruit", "P_loss", "P_unchanged", "P_advance_plus_recruit", "n_units", "n_trials"]

SUPP_S9_EVENT_CHAIN_NULL_COLUMNS = ["network_seed", "null_type", "observed_full_chain_fraction", "null_full_chain_fraction_mean", "observed_minus_null", "p_value_or_percentile", "n_events", "notes"]

SUPP_S10_SAME_WINNER_DISRUPTION_COLUMNS = ["network_seed", "unit_group", "condition", "P_same_winner_preserved", "P_same_winner_lost", "P_same_winner_delayed", "P_same_winner_reverted_to_static", "P_same_winner_lost_or_delayed", "n_dynamic_winners"]

SUPP_S10_DYNAMIC_RECOVERY_COLUMNS = ["network_seed", "condition", "dynamic_like_spike_similarity_mean", "dynamic_like_readout_recovery_mean", "decision_deflection_score_mean", "spike_count_mean", "first_fire_time_ms_mean", "n_trials"]


__all__ = ['FIGURE_ID', 'FIG5_DESIGN_VERSION', 'PRIMARY_LAYER', 'MAX_WINNERS_PER_TRIAL', 'PRIMARY_PRE_WINDOW_MS', 'LATE_PRE_WINDOW_MS', 'UNIT_GROUPS', 'MAIN_CONDITIONS', 'L1_STSP_PERTURBATION_CONDITIONS', 'LEGACY_REGION_PERTURBATION_CONDITIONS', 'REFERENCE_CONDITIONS', 'SUPP_CONDITIONS', 'REMOVED_FROM_MAIN_CONDITIONS', 'PERTURBATION_MAIN_CONDITIONS', 'MAIN_PANEL_DESCRIPTIONS', 'MAIN_CLAIM', 'SUPPLEMENT_PLAN', 'FIG5_MAIN_REQUIRED_OUTPUTS', 'FIG5_S9_OUTPUTS', 'FIG5_S10_OUTPUTS', 'FIG5_BACKWARD_COMPATIBLE_OUTPUTS', 'NULL_TYPES', 'TRIAL_COLUMNS', 'UNIT_GROUP_COLUMNS', 'PERTURBATION_UNIT_COLUMNS', 'PERTURBATION_UX_AUDIT_COLUMNS', 'PANEL_A_COLUMNS', 'PANEL_B_UNIT_COLUMNS', 'PANEL_B_SUMMARY_COLUMNS', 'PANEL_C_EVENT_COLUMNS', 'PANEL_C_TRACE_COLUMNS', 'PANEL_D_UNIT_TRANSITION_COLUMNS', 'PANEL_D_TRANSITION_SUMMARY_COLUMNS', 'PANEL_D_TRANSITION_CONTRAST_COLUMNS', 'PANEL_D_L1_STSP_UNIT_COLUMNS', 'PANEL_D_L1_STSP_SUMMARY_COLUMNS', 'PANEL_D_L1_STSP_AUDIT_COLUMNS', 'PANEL_D_L1_STSP_CONTRAST_COLUMNS', 'PANEL_D_NODE_COLUMNS', 'PANEL_D_TRIAL_COLUMNS', 'PANEL_E_COLUMNS', 'PANEL_D_EFFECT_SUMMARY_COLUMNS', 'SUPP_EVENT_AUDIT_COLUMNS', 'SUPP_NULL_COLUMNS', 'SUPP_S9_TRANSITION_COMPOSITION_COLUMNS', 'SUPP_S9_EVENT_CHAIN_NULL_COLUMNS', 'SUPP_S10_SAME_WINNER_DISRUPTION_COLUMNS', 'SUPP_S10_DYNAMIC_RECOVERY_COLUMNS']
