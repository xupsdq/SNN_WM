from __future__ import annotations


FIGURE_ID = "fig6_peak_amplified_reentry"

FIG6_DESIGN_VERSION = "multiitem_stsp_field_spike_recruitment"

PRIMARY_LAYER = "layer1"

STATE_VARIABLE = "g"

MAIN_PANELS = {
    "A": "high-STSP-overlap ablation against matched removal",
    "B": "region-gated ping readout bias",
    "C": "global ping STSP score predicts Layer 1 spike recruitment",
    "D": "real-probe entry-gated STSP score predicts Layer 1 spike deflection",
    "E": "probe overlap gates high-STSP Layer 1 recruitment",
    "F": "mechanism schematic metadata only",
}

MAIN_CLAIM = "Multi-item STSP fields bias Layer 1 recruitment only where later input enters the high-gain field."

MECHANISM_BOUNDARY = {
    "score": "rho_stsp_gain_ratio",
    "summary": "rho(q) = G_final(q) / (G_baseline(q) + eps); local and entry-gated scores average rho over each Layer 1 receptive field",
    "primary_endpoint": "Layer 1 spatial spike recruitment / spike deflection",
    "forbidden_claims": [
        "score predicts final label",
        "STSP alone determines firing",
        "high STSP automatically fires without entry",
        "connection weights define the score",
        "inhibition is part of the score",
    ],
}

SUPPLEMENT_PLAN = {
    "S7": "active overlap-gated controls and default robustness extensions",
}

MAIN_REQUIRED_OUTPUTS = [
    "data/metrics/panel_a_high_stsp_overlap_ablation.csv",
    "data/metrics/panel_a_high_stsp_overlap_ablation_summary.csv",
    "data/metrics/panel_b_region_ping_readout_bias.csv",
    "data/metrics/panel_c_global_ping_score_spike_prediction.csv",
    "data/metrics/panel_d_real_probe_score_spike_deflection.csv",
    "data/metrics/panel_e_overlap_gated_stsp_recruitment.csv",
    "data/metrics/panel_e_overlap_gated_stsp_interaction.csv",
    "data/raw/panel_f_global_mechanism_metadata.json",
]

OPTIONAL_MAIN_OUTPUTS = [
    "data/metrics/panel_f_high_stsp_overlap_ablation.csv",
    "data/metrics/panel_f_high_stsp_overlap_ablation_summary.csv",
]

SUPPLEMENTARY_OUTPUTS = [
    "data/metrics/supp_s11a_score_input_ping_audit.csv",
    "data/metrics/supp_s11b_global_ping_count_endpoint.csv",
    "data/metrics/supp_s11c_real_probe_window_robustness.csv",
    "data/metrics/supp_s11d_overlap_interaction_window_robustness.csv",
    "data/metrics/supp_s11e_overlap_site_availability.csv",
    "data/metrics/supp_s11f_high_stsp_ablation_paired_difference.csv",
]

OPTIONAL_SUPPLEMENTARY_OUTPUTS = [
    "data/metrics/supp_s11g_score_shuffle_null.csv",
    "data/metrics/supp_s11h_threshold_sensitivity.csv",
]

PERTURBATION_UNIT_SET_ORDER = ("route_peak", "route_nonpeak", "nonroute_peak", "random_matched")

PERTURBATION_UNIT_SET_LABELS = {
    "route_peak": "Route peak",
    "route_nonpeak": "Route non-peak",
    "nonroute_peak": "Non-route peak",
    "random_matched": "Random",
}

UPDATE_GROUPS = ("single_old", "multi_old", "single_recent", "multi_recent")

MODEL_NAMES = (
    "baseline_only",
    "update_only",
    "recency_only",
    "overlap_only",
    "update_plus_recency",
    "update_times_recency",
)

DOWNSTREAM_METRICS = (
    "early_recruitment_gain",
    "P_advance",
    "P_recruit",
    "spike_advance",
    "response_pattern_displacement",
    "decision_deflection_score",
    "partial_cue_completion_gain",
)

_FIG6_DELEGATES = {
    "_ablation_condition_metrics": "helpers_1",
    "_alternative_peak_definitions": "helpers_2",
    "_artifact_manifest": "output_contract",
    "_as_float_or_nan": "helpers_2",
    "_blur3": "helpers_2",
    "_bool_col": "helpers_2",
    "_bool_value": "helpers_2",
    "_centered_cosine": "helpers_2",
    "_claim_strength": "helpers_2",
    "_class_readout_vector_from_trace": "helpers_2",
    "_common_failure_reasons": "peak_perturbation",
    "_cv_r2": "helpers_2",
    "_df_all_proxy": "helpers_2",
    "_df_all_true": "helpers_2",
    "_dice": "helpers_2",
    "_early_spike_count": "helpers_2",
    "_encode_sequence_cached": "helpers_1",
    "_ensure_probe_trials": "helpers_1",
    "_entropy_from_logits": "peak_perturbation",
    "_entry_mask_to_input_tensor": "helpers_1",
    "_entry_score_audit_row": "helpers_1",
    "_failure_count": "peak_perturbation",
    "_fire_delta": "helpers_2",
    "_fired_site_score_percentile_mean": "helpers_1",
    "_fired_site_score_percentiles": "helpers_1",
    "_first_nonzero_step": "helpers_2",
    "_fit_ols": "helpers_2",
    "_flush_score_audits": "helpers_1",
    "_foreground_mask": "helpers_2",
    "_gain_ratio_audit_row": "helpers_1",
    "_global_support_controls": "helpers_2",
    "_group_mask": "helpers_2",
    "_high_overlap_mask": "helpers_2",
    "_high_rho_site_mask": "helpers_1",
    "_high_score_basin_hit_rate": "helpers_1",
    "_image_array": "helpers_1",
    "_images_for_ids": "helpers_1",
    "_insufficient_count": "peak_perturbation",
    "_is_proxy_mode": "helpers_2",
    "_jaccard": "helpers_2",
    "_js_divergence": "peak_perturbation",
    "_label_evidence": "helpers_2",
    "_layer1_input_shape": "helpers_1",
    "_leave_one_out_support_map": "helpers_1",
    "_leave_one_out_support_maps_batch": "helpers_1",
    "_leave_one_out_timing_controls": "helpers_2",
    "_main_proxy_mode": "helpers_2",
    "_make_score_region_ping_masks": "helpers_1",
    "_matched_lookup": "helpers_2",
    "_matched_nonpeak_mask": "helpers_2",
    "_matched_peak_comparison": "helpers_2",
    "_matched_probe_removal_mask": "helpers_1",
    "_matched_random_controls": "helpers_2",
    "_matched_random_unit_mask": "peak_perturbation",
    "_matched_raw_overlap_groups": "helpers_2",
    "_mean_bool": "helpers_2",
    "_mean_col": "helpers_2",
    "_mean_latency_ms": "helpers_1",
    "_missing_route_peak_unit_sets": "peak_perturbation",
    "_model_formula": "helpers_2",
    "_nan_subtract": "helpers_2",
    "_normal_two_sided_p": "helpers_2",
    "_normalize": "helpers_2",
    "_num": "helpers_2",
    "_output_distribution_row": "peak_perturbation",
    "_overlap_gated_group_metrics": "helpers_1",
    "_overlap_gated_interaction_row": "helpers_1",
    "_overlap_gated_single_group_row": "helpers_1",
    "_overlay_payload": "helpers_1",
    "_paired_unit_set_difference": "peak_perturbation",
    "_pairwise_image_sims": "helpers_2",
    "_panel_d_matched_contrast": "supplement",
    "_panel_d_summary": "supplement",
    "_panel_e_breakdown": "supplement",
    "_panel_e_summary": "supplement",
    "_peak_perturbation_claim_upgrade_allowed": "helpers_2",
    "_peak_perturbation_status": "helpers_2",
    "_peak_source_old_vs_recent": "helpers_2",
    "_perturbation_target": "helpers_2",
    "_perturbation_unit_sets": "helpers_2",
    "_plain_cosine": "helpers_2",
    "_prepare_entry_rollout_state": "helpers_1",
    "_probe_entry_mask": "helpers_1",
    "_random_window_overlap_controls": "helpers_2",
    "_real_downstream_metric_definitions": "helpers_2",
    "_real_reentry_control_s0_static": "helpers_2",
    "_real_rollout_scientific_use_audit": "supplement",
    "_recent_overlap_window_robustness": "helpers_2",
    "_record_entry_score_audit": "helpers_1",
    "_record_gain_ratio_audit": "helpers_1",
    "_regression_long_table": "supplement",
    "_regression_rows": "helpers_2",
    "_remove_probe_sites_from_spikes": "helpers_1",
    "_removed_probe_energy": "helpers_1",
    "_reset_layer1_stsp_units_to_s0": "peak_perturbation",
    "_resize_array": "helpers_2",
    "_restore_boundary_state": "helpers_1",
    "_route_peak_downstream_contrast": "peak_perturbation",
    "_route_peak_downstream_summary": "peak_perturbation",
    "_route_peak_failure_reason": "peak_perturbation",
    "_route_peak_perturbation_audit": "peak_perturbation",
    "_route_peak_reentry_contrast": "peak_perturbation",
    "_route_peak_reentry_summary": "peak_perturbation",
    "_route_peak_scientific_use_audit": "peak_perturbation",
    "_route_peak_success": "peak_perturbation",
    "_run_masked_ping_layer1_capture": "helpers_1",
    "_run_real_probe_conditions_batch": "helpers_1",
    "_run_real_probe_from_condition": "helpers_1",
    "_run_real_probe_layer1_capture_batch": "helpers_1",
    "_run_real_probe_layer1_capture": "helpers_1",
    "_run_real_probe_with_route_peak_reset": "peak_perturbation",
    "_s11_alternative_peak_definitions": "supplement",
    "_s11_leave_one_out_source_details": "supplement",
    "_s11_peak_update_group_enrichment": "supplement",
    "_s11_recent_overlap_window_robustness": "supplement",
    "_s11_update_recency_model_comparison": "supplement",
    "_s11_visual_energy_classpair_controls": "supplement",
    "_s12_global_support_controls": "supplement",
    "_s12_peak_weighted_regression_controls": "supplement",
    "_safe_div": "helpers_2",
    "_save_global_debug_figure": "debug_figures",
    "_save_panel_c_example": "helpers_2",
    "_save_panel_d_example": "helpers_2",
    "_score_quantile_indices": "helpers_1",
    "_sem": "helpers_2",
    "_sequence_index": "helpers_2",
    "_sequence_labels_from_meta": "helpers_1",
    "_sequence_support_maps": "helpers_1",
    "_sequence_support_maps_batch": "helpers_1",
    "_serial_age_bin": "helpers_1",
    "_serial_position_for_label": "helpers_1",
    "_shuffle_fired_percentile_baseline": "helpers_1",
    "_shuffle_peak_enrichment": "helpers_2",
    "_shuffled_basin_hit_rate": "helpers_1",
    "_sigmoid": "helpers_2",
    "_softmax_np": "peak_perturbation",
    "_spearman": "helpers_2",
    "_spike_timing_metrics": "helpers_2",
    "_standardize_panel_d_metrics": "supplement",
    "_standardize_panel_e_metrics": "supplement",
    "_standardized_coef": "helpers_2",
    "_step_network_once": "helpers_1",
    "_step_network_once_capture_layer1": "helpers_1",
    "_step_network_once_with_l3": "helpers_1",
    "_summary_regression_rows": "supplement",
    "_summary_route_peak_panel": "output_contract",
    "_summary_route_peak_perturbation": "output_contract",
    "_support_from_net": "helpers_1",
    "_to_tensor": "helpers_1",
    "_top_mask": "helpers_2",
    "_trial_condition_audit": "helpers_2",
    "_unit_set_valid": "peak_perturbation",
    "_visual_energy_controls": "helpers_2",
    "_write_config_files": "output_contract",
    "_write_standardized_panel_d_outputs": "supplement",
    "_write_standardized_panel_e_outputs": "supplement",
    "_write_standardized_peak_perturbation_outputs": "peak_perturbation",
    "_write_summary": "output_contract",
    "build_later_probe_peak_overlap_trials": "real_reentry_rollout",
    "build_probe_candidate_trials": "real_reentry_rollout",
    "build_sequence_trials": "sequence_bank",
    "collapse_layer1_spikes_spatial": "helpers_1",
    "compute_basin_enrichment": "helpers_1",
    "compute_entry_gated_stsp_score_map": "helpers_1",
    "compute_field_ping_readout": "field_ping_readout",
    "compute_fig6_downstream_exploratory": "fig6_downstream_exploratory",
    "compute_gain_ratio_map": "helpers_1",
    "compute_global_ping_score_spike_prediction": "global_ping_score_spike_prediction",
    "compute_high_stsp_overlap_ablation": "high_stsp_overlap_ablation",
    "compute_legacy_supplement_outputs": "supplement",
    "compute_overlap_threshold_sensitivity_extension": "supplement",
    "compute_score_shuffle_null_extension": "supplement",
    "compute_overlap_gated_stsp_recruitment": "overlap_gated_stsp_recruitment",
    "compute_peak_input_overlap_origin": "peak_input_overlap_origin",
    "compute_peak_source_attribution": "peak_source_attribution",
    "compute_peak_update_history": "peak_update_history",
    "compute_peak_weighted_downstream_metrics": "real_downstream_metrics",
    "compute_peak_weighted_overlap_definitions": "peak_weighted_overlap",
    "compute_peak_weighted_reentry_metrics": "real_reentry_rollout",
    "compute_ping_score_spike_prediction": "ping_score_spike_prediction",
    "compute_probe_overlap_map": "helpers_1",
    "compute_real_peak_overlap_downstream_metrics": "real_downstream_metrics",
    "compute_real_peak_weighted_reentry_metrics": "real_reentry_rollout",
    "compute_real_probe_score_spike_deflection": "real_probe_score_spike_deflection",
    "compute_route_peak_perturbation_outputs": "peak_perturbation",
    "compute_score_basin_sparsification": "score_basin_sparsification",
    "compute_score_quantile_metrics": "helpers_1",
    "compute_spike_deflection_metrics": "helpers_1",
    "compute_supp_update_recency_support_model": "update_recency_model",
    "compute_supplement_outputs": "supplement",
    "define_final_peaks_and_update_groups": "peak_enrichment",
    "fit_update_recency_support_models": "update_recency_model",
    "run_leave_one_item_out_support_bank": "sequence_bank",
    "run_probe_candidate_reentry_rollouts": "real_reentry_rollout",
    "run_real_probe_reentry_rollouts": "real_reentry_rollout",
    "run_sequence_bank": "sequence_bank",
    "save_debug_figures": "debug_figures",
    "shuffle_score_control": "helpers_1",
    "write_fig6_supplement_aliases": "supplement",
    "write_global_mechanism_metadata": "supplement"
}

PANEL_B_REGION_PING_COLUMNS = ["network_seed", "sequence_id", "entry_condition", "old_mass", "middle_mass", "recent_mass", "other_mass", "silent_rate", "n_trials", "ping_active_sites", "total_ping_current"]

PANEL_C_GLOBAL_PING_SCORE_COLUMNS = ["network_seed", "sequence_id", "early_window_ms", "score_quantile_bin", "mean_score", "n_sites", "fired_site_count", "spike_probability", "mean_early_spike_count", "mean_first_spike_latency_ms", "fired_site_score_percentile_mean"]

PANEL_C_PING_SCORE_COLUMNS = PANEL_C_GLOBAL_PING_SCORE_COLUMNS

PANEL_D_REAL_PROBE_SCORE_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_label", "early_window_ms", "score_quantile_bin", "mean_score", "n_sites", "dynamic_spike_probability", "baseline_spike_probability", "delta_spike_probability", "mean_delta_spike_count", "recruit_probability", "advance_probability", "valid_site_count", "probe_active_area", "prior_updated_overlap_area"]

PANEL_E_OVERLAP_GATED_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_label", "early_window_ms", "stsp_group", "overlap_group", "n_sites", "mean_local_stsp_score", "mean_probe_overlap", "dynamic_spike_probability", "baseline_spike_probability", "delta_spike_probability", "mean_delta_spike_count", "recruit_probability", "stsp_group_quantile", "overlap_threshold"]

PANEL_E_OVERLAP_GATED_INTERACTION_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_label", "early_window_ms", "stsp_group_quantile", "overlap_threshold", "stsp_effect_with_overlap", "stsp_effect_without_overlap", "interaction_delta", "high_overlap_delta", "low_overlap_delta", "high_nooverlap_delta", "low_nooverlap_delta", "n_sites_high_overlap", "n_sites_low_overlap", "n_sites_high_nooverlap", "n_sites_low_nooverlap"]

PANEL_F_HIGH_STSP_ABLATION_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_label", "condition", "early_window_ms", "removed_active_area", "removed_input_energy", "dynamic_spike_probability", "baseline_spike_probability", "delta_spike_probability", "mean_delta_spike_count"]

PANEL_F_HIGH_STSP_ABLATION_SUMMARY_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_label", "early_window_ms", "loss_condition", "loss_delta_spike_probability", "loss_mean_delta_spike_count", "removed_active_area", "removed_input_energy"]

PANEL_E_BASIN_COLUMNS = ["network_seed", "sequence_id", "entry_type", "entry_condition", "basin_radius", "top_score_quantile", "n_fired_sites", "fired_site_score_percentile_mean", "fired_site_score_percentile_sem", "high_score_basin_hit_rate", "shuffled_hit_rate", "enrichment_over_shuffle"]

FIG6_GAIN_RATIO_AUDIT_COLUMNS = ["network_seed", "sequence_id", "raw_ratio_min", "raw_ratio_max", "raw_ratio_q01", "raw_ratio_q99", "clipped_ratio_min", "clipped_ratio_max", "nonfinite_raw_count", "baseline_floor_count", "clip_quantile_low", "clip_quantile_high", "score_use_log_gain"]

FIG6_ENTRY_SCORE_AUDIT_COLUMNS = ["network_seed", "sequence_id", "entry_type", "entry_condition", "valid_site_count", "score_shape", "entry_area", "rf_empty_excluded_count", "score_finite_count", "layer1_spike_shape", "spike_score_shape_aligned", "channel_policy"]

SEQUENCE_TRIAL_COLUMNS = ["network_seed", "sequence_id", "seq_len", "stage_k", "item_image_id", "item_label", "ordered_item_ids", "ordered_item_labels", "sequence_seed", "mean_pairwise_image_similarity", "max_pairwise_image_similarity", "min_pairwise_image_similarity"]

PROBE_TRIAL_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_image_id", "probe_label", "probe_source", "entry_mask_mode", "raw_overlap", "peak_weighted_overlap", "peak_overlap_fraction", "nonpeak_overlap_fraction", "visual_similarity", "input_energy", "class_pair", "candidate_seed", "peak_support_sum", "nonpeak_support_sum"]

MATCHED_GROUP_COLUMNS = ["network_seed", "matched_group_id", "high_peak_candidate_id", "low_peak_candidate_id", "raw_overlap_difference", "visual_similarity_difference", "input_energy_difference", "peak_weighted_overlap_difference", "class_pair_matched", "notes"]

STATE_BANK_MANIFEST_COLUMNS = ["network_seed", "sequence_id", "seq_len", "state_condition", "stage_k", "layer", "state_variable", "shape", "storage_file", "storage_key", "captured_after", "sample_ms", "delay_ms"]

PANEL_A_UNIT_COLUMNS = ["network_seed", "sequence_id", "seq_len", "layer", "state_variable", "unit_id", "update_count", "last_update_position", "time_since_last_update", "recency_group", "multiplicity_group", "update_history_group", "is_peak", "final_support", "baseline_support", "delta_support"]

PANEL_A_SUMMARY_COLUMNS = ["network_seed", "update_history_group", "P_peak", "mean_final_support", "mean_delta_support", "peak_enrichment", "n_units"]

PANEL_B_METRIC_COLUMNS = ["network_seed", "sequence_id", "layer", "state_variable", "target", "model_name", "r2", "cv_r2", "auc_if_binary", "delta_r2_vs_overlap_only", "delta_r2_vs_update_only", "delta_r2_vs_recency_only", "n_units"]

PANEL_B_COEF_COLUMNS = ["network_seed", "model_name", "coefficient_name", "coefficient_value", "standardized_coefficient", "p_value", "notes"]

PANEL_C_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_label", "raw_overlap", "peak_weighted_overlap", "peak_overlap_fraction", "nonpeak_overlap_fraction", "visual_similarity", "input_energy", "peak_support_sum", "nonpeak_support_sum"]

PANEL_D_METRIC_COLUMNS = ["network_seed", "sequence_id", "probe_id", "matched_group_id", "raw_overlap", "peak_weighted_overlap", "peak_overlap_group", "visual_similarity", "input_energy", "reentry_strength", "DPI_L3", "dynamic_like_recovery", "decision_deflection_score"]

PANEL_E_METRIC_COLUMNS = ["network_seed", "sequence_id", "probe_id", "matched_group_id", "raw_overlap", "peak_weighted_overlap", "peak_overlap_group", "visual_similarity", "input_energy", "early_recruitment_gain", "P_advance", "P_recruit", "spike_advance", "response_pattern_displacement", "decision_deflection_score", "partial_cue_completion_gain"]

PANEL_A_SOURCE_COLUMNS = ["network_seed", "sequence_id", "seq_len", "removed_position", "removed_label", "removed_image_id", "peak_loss", "nonpeak_loss", "prior_updated_loss", "peak_loss_fraction", "nonpeak_loss_fraction", "peak_vs_nonpeak_loss_ratio", "support_loss_total", "leave_one_out_mode", "proxy_mode"]

PANEL_A_SOURCE_SUMMARY_COLUMNS = ["network_seed", "seq_len", "removed_position", "relative_position_from_end", "mean_peak_loss_fraction", "sem_peak_loss_fraction", "mean_peak_vs_nonpeak_loss_ratio", "n_sequences"]

PANEL_B_UPDATE_HISTORY_COLUMNS = ["network_seed", "sequence_id", "seq_len", "unit_id", "is_peak", "is_nonpeak_control", "update_count", "last_update_position", "time_since_last_update", "recent_w2", "recent_w3", "recent_w4", "recent_w5", "is_multi_update", "is_multi_recent_w2", "is_multi_recent_w3", "is_multi_recent_w4", "is_multi_recent_w5", "final_support", "delta_support"]

PANEL_B_UPDATE_HISTORY_SUMMARY_COLUMNS = ["network_seed", "group", "mean_update_count", "P_update_ge_2", "P_update_ge_3", "mean_time_since_last_update", "P_recent_w2", "P_recent_w3", "P_recent_w4", "P_recent_w5", "P_multi_recent_w2", "P_multi_recent_w3", "P_multi_recent_w4", "P_multi_recent_w5", "n_units"]

PANEL_C_ORIGIN_COLUMNS = ["network_seed", "sequence_id", "seq_len", "overlap_window", "window_start_position", "window_end_position", "n_items_in_window", "overlap_type", "n_overlap_pixels", "n_peak_pixels", "dice_peak_overlap", "jaccard_peak_overlap", "peak_coverage", "overlap_precision", "cosine_delta_support_overlap_count", "spearman_delta_support_overlap_count", "fallback_used"]

PANEL_C_ORIGIN_SUMMARY_COLUMNS = ["network_seed", "overlap_window", "mean_dice", "sem_dice", "mean_peak_coverage", "mean_cosine", "n_sequences"]

PANEL_D_TRIAL_DEFINITION_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_image_id", "probe_label", "probe_source", "entry_mask_mode", "raw_overlap", "peak_weighted_overlap", "peak_overlap_fraction", "nonpeak_overlap_fraction", "visual_similarity", "input_energy", "peak_support_sum", "nonpeak_support_sum", "class_pair", "candidate_seed"]

PANEL_D_REAL_METRIC_COLUMNS = ["network_seed", "sequence_id", "probe_id", "matched_group_id", "raw_overlap", "peak_weighted_overlap", "peak_overlap_group", "visual_similarity", "input_energy", "prediction_Sfinal", "prediction_S0", "correct_Sfinal", "correct_S0", "first_fire_time_Sfinal", "first_fire_time_S0", "first_fire_time_delta", "l3_trace_delta_norm", "reentry_strength_real", "dynamic_like_recovery_real", "decision_deflection_score_real", "proxy_mode"]

PANEL_E_REAL_METRIC_COLUMNS = ["network_seed", "sequence_id", "probe_id", "matched_group_id", "raw_overlap", "peak_weighted_overlap", "peak_overlap_group", "visual_similarity", "input_energy", "early_recruitment_gain_real", "P_advance_real", "P_recruit_real", "spike_advance_real", "response_pattern_displacement_real", "decision_deflection_score_real", "partial_cue_completion_gain_real", "proxy_mode"]

PERTURBATION_COLUMNS = ["network_seed", "sequence_id", "probe_id", "condition", "n_perturbed_units", "raw_overlap", "peak_weighted_overlap", "reentry_strength", "DPI_L3", "early_recruitment_gain", "decision_deflection_score", "completion_gain"]

PANEL_D_ROUTE_PEAK_TRIAL_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_label", "seq_len", "perturbation_unit_set", "perturbation_condition", "perturbation_mode", "state_condition", "raw_overlap", "peak_weighted_overlap", "route_unit_count", "peak_unit_count", "route_peak_unit_count", "route_nonpeak_unit_count", "nonroute_peak_unit_count", "random_unit_count", "insufficient_units", "reentry_strength_intact", "reentry_strength_perturbed", "reentry_strength_s0", "reentry_loss", "normalized_reentry_loss", "prediction_intact", "prediction_perturbed", "prediction_s0", "first_fire_time_intact", "first_fire_time_perturbed", "first_fire_time_s0", "restore_ok", "perturbation_ok", "denominator_choice", "reset_variables", "probe_input_unchanged", "failure_reason"]

PANEL_D_ROUTE_PEAK_SUMMARY_COLUMNS = ["network_seed", "perturbation_unit_set", "condition_label", "mean_reentry_loss", "sem_reentry_loss", "mean_normalized_reentry_loss", "sem_normalized_reentry_loss", "n_trials", "n_valid_trials", "n_skipped_missing_boundary", "n_skipped_insufficient_units", "n_perturbation_failed", "insufficient_fraction", "denominator_choice"]

PANEL_D_ROUTE_PEAK_CONTRAST_COLUMNS = ["network_seed", "contrast", "metric", "route_peak_minus_control", "route_peak_minus_route_nonpeak", "route_peak_minus_nonroute_peak", "route_peak_minus_random", "route_peak_effect_size", "n_valid_pairs"]

PANEL_E_ROUTE_PEAK_TRIAL_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_label", "perturbation_unit_set", "perturbation_condition", "response_displacement_intact", "response_displacement_perturbed", "response_displacement_s0", "response_displacement_loss", "decision_deflection_intact", "decision_deflection_perturbed", "decision_deflection_s0", "decision_deflection_loss", "prediction_intact", "prediction_perturbed", "prediction_s0", "output_switch", "output_distribution_JS", "perturbation_ok", "insufficient_units", "failure_reason"]

PANEL_E_ROUTE_PEAK_SUMMARY_COLUMNS = ["network_seed", "perturbation_unit_set", "condition_label", "P_output_switch", "sem_output_switch", "mean_response_displacement_loss", "sem_response_displacement_loss", "mean_decision_deflection_loss", "sem_decision_deflection_loss", "n_trials", "n_valid_trials"]

PANEL_E_ROUTE_PEAK_CONTRAST_COLUMNS = ["network_seed", "metric", "contrast", "route_peak_minus_route_nonpeak", "route_peak_minus_nonroute_peak", "route_peak_minus_random", "n_valid_pairs"]

PANEL_E_ROUTE_PEAK_OUTPUT_DISTRIBUTION_COLUMNS = ["network_seed", "sequence_id", "probe_id", "perturbation_unit_set", "output_distribution_JS", "intact_entropy", "condition_entropy"]

ROUTE_PEAK_UNIT_SET_COLUMNS = ["network_seed", "sequence_id", "probe_id", "perturbation_unit_set", "unit_id", "notes"]


__all__ = ['FIGURE_ID', 'FIG6_DESIGN_VERSION', 'PRIMARY_LAYER', 'STATE_VARIABLE', 'MAIN_PANELS', 'MAIN_CLAIM', 'MECHANISM_BOUNDARY', 'SUPPLEMENT_PLAN', 'MAIN_REQUIRED_OUTPUTS', 'OPTIONAL_MAIN_OUTPUTS', 'SUPPLEMENTARY_OUTPUTS', 'OPTIONAL_SUPPLEMENTARY_OUTPUTS', 'PERTURBATION_UNIT_SET_ORDER', 'PERTURBATION_UNIT_SET_LABELS', 'UPDATE_GROUPS', 'MODEL_NAMES', 'DOWNSTREAM_METRICS', '_FIG6_DELEGATES', 'PANEL_B_REGION_PING_COLUMNS', 'PANEL_C_GLOBAL_PING_SCORE_COLUMNS', 'PANEL_C_PING_SCORE_COLUMNS', 'PANEL_D_REAL_PROBE_SCORE_COLUMNS', 'PANEL_E_OVERLAP_GATED_COLUMNS', 'PANEL_E_OVERLAP_GATED_INTERACTION_COLUMNS', 'PANEL_F_HIGH_STSP_ABLATION_COLUMNS', 'PANEL_F_HIGH_STSP_ABLATION_SUMMARY_COLUMNS', 'PANEL_E_BASIN_COLUMNS', 'FIG6_GAIN_RATIO_AUDIT_COLUMNS', 'FIG6_ENTRY_SCORE_AUDIT_COLUMNS', 'SEQUENCE_TRIAL_COLUMNS', 'PROBE_TRIAL_COLUMNS', 'MATCHED_GROUP_COLUMNS', 'STATE_BANK_MANIFEST_COLUMNS', 'PANEL_A_UNIT_COLUMNS', 'PANEL_A_SUMMARY_COLUMNS', 'PANEL_B_METRIC_COLUMNS', 'PANEL_B_COEF_COLUMNS', 'PANEL_C_COLUMNS', 'PANEL_D_METRIC_COLUMNS', 'PANEL_E_METRIC_COLUMNS', 'PANEL_A_SOURCE_COLUMNS', 'PANEL_A_SOURCE_SUMMARY_COLUMNS', 'PANEL_B_UPDATE_HISTORY_COLUMNS', 'PANEL_B_UPDATE_HISTORY_SUMMARY_COLUMNS', 'PANEL_C_ORIGIN_COLUMNS', 'PANEL_C_ORIGIN_SUMMARY_COLUMNS', 'PANEL_D_TRIAL_DEFINITION_COLUMNS', 'PANEL_D_REAL_METRIC_COLUMNS', 'PANEL_E_REAL_METRIC_COLUMNS', 'PERTURBATION_COLUMNS', 'PANEL_D_ROUTE_PEAK_TRIAL_COLUMNS', 'PANEL_D_ROUTE_PEAK_SUMMARY_COLUMNS', 'PANEL_D_ROUTE_PEAK_CONTRAST_COLUMNS', 'PANEL_E_ROUTE_PEAK_TRIAL_COLUMNS', 'PANEL_E_ROUTE_PEAK_SUMMARY_COLUMNS', 'PANEL_E_ROUTE_PEAK_CONTRAST_COLUMNS', 'PANEL_E_ROUTE_PEAK_OUTPUT_DISTRIBUTION_COLUMNS', 'ROUTE_PEAK_UNIT_SET_COLUMNS']
