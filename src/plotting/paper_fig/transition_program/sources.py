from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


EXPECTED_NETWORKS = tuple(range(1000, 1020))
CONFIRMATORY_NETWORKS = tuple(range(1001, 1020))


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    mode: str
    relative_root: str
    filename: str
    required_columns: tuple[str, ...]
    seeds: tuple[int, ...] = ()
    scientific_unit: str = "network"
    post_hoc: bool = False


def _p0(key: str, filename: str, columns: Sequence[str]) -> DatasetSpec:
    return DatasetSpec(
        key=key,
        mode="single",
        relative_root="new_results_reanalysis/metrics",
        filename=filename,
        required_columns=tuple(columns),
        scientific_unit="network",
        post_hoc=True,
    )


def _seed(
    key: str,
    root: str,
    filename: str,
    columns: Sequence[str],
    *,
    seeds: Sequence[int] = EXPECTED_NETWORKS,
    unit: str = "network",
) -> DatasetSpec:
    return DatasetSpec(
        key=key,
        mode="seed_metric",
        relative_root=root,
        filename=filename,
        required_columns=tuple(columns),
        seeds=tuple(int(seed) for seed in seeds),
        scientific_unit=unit,
    )


def _fixed_seed(
    key: str,
    filename: str,
    columns: Sequence[str],
    *,
    raw: bool = False,
    unit: str = "network",
) -> DatasetSpec:
    return DatasetSpec(
        key=key,
        mode="fixed_seed_raw" if raw else "fixed_seed_metric",
        relative_root="fig2_fixed_b_mechanism_confirmatory",
        filename=filename,
        required_columns=tuple(columns),
        seeds=EXPECTED_NETWORKS,
        scientific_unit=unit,
    )


def _fixed_aggregate(
    key: str,
    filename: str,
    columns: Sequence[str],
) -> DatasetSpec:
    return DatasetSpec(
        key=key,
        mode="fixed_aggregate",
        relative_root="fig2_fixed_b_mechanism_confirmatory",
        filename=filename,
        required_columns=tuple(columns),
        seeds=EXPECTED_NETWORKS,
        scientific_unit="network",
    )


def _bridge_seed(
    key: str,
    filename: str,
    columns: Sequence[str],
    *,
    unit: str = "cell",
) -> DatasetSpec:
    return DatasetSpec(
        key=key,
        mode="bridge_seed_metric",
        relative_root="history_rewrite_bridge/bridge",
        filename=filename,
        required_columns=tuple(columns),
        seeds=CONFIRMATORY_NETWORKS,
        scientific_unit=unit,
    )


def _bridge_single(
    key: str,
    relative_root: str,
    filename: str,
    columns: Sequence[str],
) -> DatasetSpec:
    return DatasetSpec(
        key=key,
        mode="single",
        relative_root=relative_root,
        filename=filename,
        required_columns=tuple(columns),
        seeds=CONFIRMATORY_NETWORKS,
        scientific_unit="network",
    )


FIG1_ROOT = "fig1_functional_stsp_substrate/fig1_functional_stsp_substrate"
OVERLAP_ROOT = "fig4_overlap_reentry"
COMPETITION_ROOT = "fig5_local_support_competition"
MULTI_ROOT = "fig6_peak_amplified_reentry"
PAIR_ROOT = "fig2_pair_fused_stsp_state/fig2_pair_fused_stsp_state"
PROGRESSIVE_ROOT = "fig3_multiitem_peak_landscape"


DATASET_SPECS: dict[str, DatasetSpec] = {
    spec.key: spec
    for spec in (
        _p0(
            "p0.phase_firing",
            "fig1_phase_firing_network_metrics.csv",
            ("network_seed", "layer", "phase", "mean_spike_rate_hz"),
        ),
        _p0(
            "p0.delay_silence",
            "fig1_delay_silence_network_metrics.csv",
            (
                "network_seed",
                "layer",
                "delay_mean_hz",
                "stimulus_reference_hz",
                "delay_to_stimulus_ratio",
            ),
        ),
        _p0(
            "p0.delay_trend",
            "fig1_delay_trend_network_metrics.csv",
            (
                "network_seed",
                "log2_delay_slope",
                "short_delay_mean",
                "long_delay_mean",
            ),
        ),
        _p0(
            "p0.event_chain",
            "fig3_event_chain_network_metrics.csv",
            (
                "network_seed",
                "null_type",
                "observed_value",
                "null_mean",
                "observed_minus_null",
            ),
        ),
        _p0(
            "p0.event_selection",
            "fig3_event_selection_network_audit.csv",
            ("network_seed",),
        ),
        _p0(
            "p0.same_trial_path",
            "fig3_same_trial_path_network_metrics.csv",
            (
                "network_seed",
                "standardized_l1_to_l2_beta",
                "incremental_r2",
                "raw_within_condition_correlation",
            ),
        ),
        _p0(
            "p0.writeback",
            "fig3_writeback_network_metrics.csv",
            (
                "network_seed",
                "dynamic_minus_static_prior_fraction",
                "conditional_difference_in_differences",
            ),
        ),
        _p0(
            "p0.progressive_stage",
            "fig4_layer2_progressive_stage_metrics.csv",
            (
                "network_seed",
                "state_variable",
                "stage_k",
                "state_displacement",
                "natural_decay_displacement",
                "observed_minus_natural_decay",
            ),
        ),
        _p0(
            "p0.progressive_network",
            "fig4_layer2_progressive_network_metrics.csv",
            (
                "network_seed",
                "state_variable",
                "early_mean_k2_k5",
                "late_mean_k7_k10",
                "early_minus_late",
                "terminal_observed_minus_decay",
            ),
        ),
        _p0(
            "p0.terminal_equivalence",
            "fig4_layer2_terminal_equivalence.csv",
            (
                "network_seed",
                "max_abs_stage_final_difference",
                "exact_equal",
            ),
        ),
        _p0(
            "p0.pair_network",
            "fig6_layer2_pair_network_metrics.csv",
            (
                "network_seed",
                "fusion_dual_score",
                "min_component_similarity",
                "true_minus_shuffled",
                "unconstrained_cv_r2",
                "residual_norm_ratio",
                "linear_mixture_gain",
                "residual_pair_specificity",
            ),
        ),
        _p0(
            "p0.multi_network",
            "fig6_layer2_multi_network_metrics.csv",
            (
                "network_seed",
                "seq_len",
                "n_eff",
                "normalized_n_eff",
                "similarity_entropy",
                "recency_bias",
            ),
        ),
        _p0(
            "p0.multi_item_weights",
            "fig6_layer2_multi_item_weights.csv",
            (
                "network_seed",
                "sequence_id",
                "seq_len",
                "item_position",
                "item_weight",
                "constituent_similarity",
                "is_latest",
            ),
        ),
        _p0(
            "p0.multi_sequence",
            "fig6_layer2_multi_sequence_metrics.csv",
            ("network_seed", "sequence_id", "seq_len"),
        ),
        _seed(
            "fig1.baseline",
            FIG1_ROOT,
            "panel_b_baseline_metrics_by_network.csv",
            (
                "network_seed",
                "overall_recall",
                "error_rate",
                "silent_rate",
                "n_trials",
            ),
        ),
        _seed(
            "fig1.delay_decode",
            FIG1_ROOT,
            "panel_c_delay_decode_metrics.csv",
            (
                "network_seed",
                "layer",
                "delay_ms",
                "feature_type",
                "acc",
                "macro_f1",
                "chance",
            ),
        ),
        _seed(
            "fig1.condition",
            FIG1_ROOT,
            "panel_d_condition_metrics.csv",
            (
                "network_seed",
                "condition",
                "acc_probe",
                "sample_attribution_rate",
                "donor_attribution_rate",
                "silent_rate",
            ),
        ),
        _seed(
            "fig1.attribution",
            FIG1_ROOT,
            "panel_e_attribution_metrics.csv",
            (
                "network_seed",
                "condition",
                "original_sample_attribution",
                "donor_sample_attribution",
                "donor_shift_gain_vs_dynamic",
            ),
        ),
        _seed(
            "fig1.class_recall",
            FIG1_ROOT,
            "supp_class_recall_by_digit.csv",
            ("network_seed", "digit", "class_recall", "n_trials"),
        ),
        _seed(
            "fig1.confusion",
            FIG1_ROOT,
            "supp_confusion_matrix_long.csv",
            ("network_seed", "true_label", "pred_label", "count"),
        ),
        _seed(
            "fig1.delay_curve",
            FIG1_ROOT,
            "supp_delay_decode_curve.csv",
            (
                "network_seed",
                "layer",
                "delay_ms",
                "feature_type",
                "acc",
                "macro_f1",
                "chance",
            ),
        ),
        _seed(
            "fig1.delay_contrast",
            FIG1_ROOT,
            "supp_dms_delay_sweep_contrast.csv",
            (
                "network_seed",
                "delay_ms",
                "acc_dynamic",
                "acc_static",
                "stsp_interference",
                "sample_bias_dynamic",
                "sample_bias_static",
                "sample_bias_excess_dynamic_minus_static",
            ),
        ),
        _seed(
            "fig1.delay_metrics",
            FIG1_ROOT,
            "supp_dms_delay_sweep_metrics.csv",
            (
                "network_seed",
                "delay_ms",
                "condition",
                "acc_probe",
                "sample_attribution_rate",
                "silent_rate",
            ),
        ),
        _seed(
            "fig1.phase_rates",
            FIG1_ROOT,
            "supp_phase_firing_rates.csv",
            (
                "network_seed",
                "trial_id",
                "layer",
                "phase",
                "spike_rate_hz",
            ),
            unit="trial",
        ),
        _seed(
            "fig1.substrate",
            FIG1_ROOT,
            "supp_substrate_shuffle_metrics.csv",
            (
                "network_seed",
                "condition",
                "substrate",
                "acc_probe",
                "sample_attribution_rate",
                "donor_attribution_rate",
            ),
        ),
        _fixed_aggregate(
            "fixed.scalars",
            "fixed_b_confirmatory_network_scalars.csv",
            (
                "network_seed",
                "family",
                "endpoint",
                "prefix_k",
                "value",
                "role",
            ),
        ),
        _fixed_seed(
            "fixed.decomp_cell",
            "fixed_b_decomposition_cell_metrics.csv",
            (
                "network_seed",
                "prefix_k",
                "history_family_id",
                "same_B_common_update_cosine",
                "total_contrast_norm",
                "local_replay_contrast_norm",
                "processing_residual_gamma_norm",
                "total_contrast_fraction",
                "local_replay_fraction",
                "processing_residual_gamma_energy_fraction",
                "decomposition_relative_error",
                "valid",
            ),
            unit="cell",
        ),
        _fixed_seed(
            "fixed.decomp_summary",
            "fixed_b_decomposition_summary.csv",
            (
                "network_seed",
                "prefix_k",
                "mean_same_B_common_update_cosine",
                "mean_processing_residual_gamma_energy_fraction",
                "mean_total_contrast_fraction",
                "mean_local_replay_fraction",
                "max_decomposition_relative_error",
            ),
        ),
        _fixed_seed(
            "fixed.event_cell",
            "fixed_b_event_gamma_cell_metrics.csv",
            (
                "network_seed",
                "prefix_k",
                "history_family_id",
                "changed_event_coordinate_fraction",
                "event_gamma_enrichment",
                "event_gamma_enrichment_ratio",
                "changed_coordinate_gamma_energy_fraction",
                "valid",
            ),
            unit="cell",
        ),
        _fixed_seed(
            "fixed.event_summary",
            "fixed_b_event_gamma_summary.csv",
            (
                "network_seed",
                "prefix_k",
                "valid_coverage",
                "mean_event_gamma_enrichment",
                "mean_changed_event_coordinate_fraction",
                "mean_changed_coordinate_gamma_energy_fraction",
                "fraction_positive_event_gamma_enrichment",
            ),
        ),
        _fixed_seed(
            "fixed.swap_cell",
            "fixed_b_swap_cell_metrics.csv",
            (
                "network_seed",
                "prefix_k",
                "swap_scope",
                "endpoint",
                "donor_transfer_index",
                "effect_alignment_cosine",
                "valid",
            ),
            unit="cell",
        ),
        _fixed_seed(
            "fixed.swap_summary",
            "fixed_b_swap_summary.csv",
            (
                "network_seed",
                "prefix_k",
                "swap_scope",
                "endpoint",
                "valid_coverage",
                "mean_donor_transfer_index",
                "fraction_positive",
                "mean_effect_alignment_cosine",
            ),
        ),
        _fixed_seed(
            "fixed.gates",
            "fixed_b_engineering_gates.csv",
            ("network_seed", "gate", "passed", "observed", "threshold_or_expected"),
        ),
        _fixed_seed(
            "fixed.trajectory",
            "fixed_b_state_trajectory_rows.csv",
            (
                "network_seed",
                "prefix_k",
                "history_condition",
                "track",
                "branch",
                "checkpoint",
                "elapsed_steps",
                "layer2_ux_displacement_norm",
                "layer2_ux_passive_corrected_norm",
                "target_margin",
                "class_score_vector",
            ),
            raw=True,
            unit="trajectory row",
        ),
        _seed(
            "overlap.entry",
            OVERLAP_ROOT,
            "panel_b_similarity_entry_metrics.csv",
            (
                "network_seed",
                "pair_id",
                "class_pair",
                "pixel_similarity",
                "dice_overlap",
                "acc_drop",
                "drop_event",
            ),
            unit="pair",
        ),
        _seed(
            "overlap.matched",
            OVERLAP_ROOT,
            "panel_c_overlap_matched_comparison.csv",
            (
                "network_seed",
                "matched_group_id",
                "overlap_group",
                "pixel_similarity",
                "dice_overlap",
                "class_pair",
                "DPI_L3",
                "acc_drop",
                "decision_deflection",
            ),
            unit="matched pair",
        ),
        _seed(
            "overlap.perturb_contrast",
            OVERLAP_ROOT,
            "panel_d_l1_stsp_overlap_perturbation_contrast.csv",
            (
                "network_seed",
                "dynamic_minus_overlap_reset",
                "nonoverlap_reset_minus_overlap_reset",
                "random_reset_minus_overlap_reset",
                "n_valid_pairs",
            ),
        ),
        _seed(
            "overlap.perturb_summary",
            OVERLAP_ROOT,
            "panel_d_l1_stsp_overlap_perturbation_summary.csv",
            (
                "network_seed",
                "condition",
                "condition_label",
                "mean_accuracy_drop_vs_static",
                "mean_probe_accuracy",
                "n_valid_pairs",
            ),
        ),
        _seed(
            "overlap.l3_time",
            OVERLAP_ROOT,
            "panel_e_time_resolved_l3_displacement.csv",
            (
                "network_seed",
                "pair_id",
                "condition",
                "time_ms",
                "DPI_L3_t",
                "overlap_bin",
                "similarity_bin",
            ),
            unit="pair-time",
        ),
        _seed(
            "overlap.decision",
            OVERLAP_ROOT,
            "supp_s8_decision_deflection_summary.csv",
            (
                "network_seed",
                "condition",
                "mean_dynamic_like_recovery",
                "mean_decision_deflection_score",
                "n_pairs",
            ),
        ),
        _seed(
            "overlap.class_pair",
            OVERLAP_ROOT,
            "supp_class_pair_breakdown.csv",
            (
                "network_seed",
                "class_pair",
                "n_pairs",
                "mean_acc_drop",
                "mean_DPI_L3",
                "mean_decision_deflection",
            ),
        ),
        _seed(
            "overlap.alt_defs",
            OVERLAP_ROOT,
            "supp_alternative_overlap_definitions.csv",
            (
                "network_seed",
                "overlap_definition",
                "overlap_value",
                "dynamic_effect_metric",
                "metric_value",
            ),
            unit="pair",
        ),
        _seed(
            "overlap.two_by_two",
            OVERLAP_ROOT,
            "supp_overlap_similarity_2x2.csv",
            (
                "network_seed",
                "similarity_group",
                "overlap_group",
                "metric",
                "value",
            ),
        ),
        _seed(
            "overlap.regression",
            OVERLAP_ROOT,
            "supp_overlap_accuracy_regression.csv",
            (
                "network_seed",
                "metric",
                "beta_overlap",
                "beta_similarity",
                "r2",
                "n_pairs",
            ),
        ),
        _seed(
            "overlap.random_controls",
            OVERLAP_ROOT,
            "supp_random_mask_perturbation_controls.csv",
            (
                "network_seed",
                "condition",
                "DPI_L3",
                "dynamic_like_recovery",
                "decision_deflection_score",
            ),
            unit="random mask",
        ),
        _seed(
            "competition.support",
            COMPETITION_ROOT,
            "panel_a_preprobe_support_metrics.csv",
            (
                "network_seed",
                "trial_id",
                "unit_group",
                "layer",
                "state_variable",
                "mean_support",
                "support_enrichment",
                "n_units",
            ),
            unit="trial",
        ),
        _seed(
            "competition.transitions",
            COMPETITION_ROOT,
            "panel_b_transition_summary_by_group.csv",
            (
                "network_seed",
                "trial_id",
                "unit_group",
                "early_window_ms",
                "P_advance",
                "P_recruit",
                "P_loss",
                "P_unchanged",
                "P_advance_plus_recruit",
                "n_units",
            ),
            unit="trial",
        ),
        _seed(
            "competition.event_trace",
            COMPETITION_ROOT,
            "panel_c_event_trace_summary.csv",
            (
                "network_seed",
                "time_ms",
                "trace_type",
                "mean_value",
                "sem_value",
                "n_events",
            ),
        ),
        _seed(
            "competition.winner",
            COMPETITION_ROOT,
            "panel_c_winner_loser_network_summary.csv",
            (
                "network_seed",
                "n_events_eligible",
                "winner_full_pre_delta_v_mean",
                "loser_full_pre_delta_v_mean",
                "winner_minus_loser_full_pre_delta_v_mean",
            ),
        ),
        _seed(
            "competition.perturb_contrast",
            COMPETITION_ROOT,
            "panel_d_l1_stsp_perturbation_contrast.csv",
            (
                "network_seed",
                "dynamic_transition_mass",
                "attenuate_transition_mass",
                "reset_transition_mass",
                "dynamic_minus_attenuate_transition_mass",
                "dynamic_minus_reset_transition_mass",
            ),
        ),
        _seed(
            "competition.perturb_effect",
            COMPETITION_ROOT,
            "panel_d_perturbation_effect_summary.csv",
            (
                "network_seed",
                "metric",
                "dynamic_value",
                "static_value",
                "attenuate_value",
                "reset_value",
                "attenuate_disruption_vs_dynamic",
                "reset_disruption_vs_dynamic",
            ),
        ),
        _seed(
            "competition.writeback",
            COMPETITION_ROOT,
            "supp_postprobe_l2_writeback_by_network.csv",
            (
                "network_seed",
                "condition",
                "layer",
                "unit_group",
                "metric",
                "value",
                "n_trials",
            ),
        ),
        _seed(
            "competition.window",
            COMPETITION_ROOT,
            "supp_early_window_robustness.csv",
            (
                "network_seed",
                "early_window_ms",
                "unit_group",
                "P_advance",
                "P_recruit",
                "P_loss",
                "P_unchanged",
                "P_advance_plus_recruit",
            ),
        ),
        _seed(
            "competition.nulls",
            COMPETITION_ROOT,
            "supp_event_chain_null_baselines.csv",
            (
                "network_seed",
                "null_type",
                "metric",
                "observed_value",
                "null_mean",
                "observed_minus_null",
            ),
        ),
        _seed(
            "competition.radius",
            COMPETITION_ROOT,
            "supp_neighborhood_radius_robustness.csv",
            (
                "network_seed",
                "neighborhood_radius",
                "winner_pre_spike_delta_v_mean",
                "loser_post_winner_inh_rise",
                "loser_post_winner_suppressed",
            ),
        ),
        _seed(
            "competition.same_winner",
            COMPETITION_ROOT,
            "supp_s10_same_winner_disruption.csv",
            (
                "network_seed",
                "unit_group",
                "condition",
                "P_same_winner_preserved",
                "P_same_winner_lost_or_delayed",
            ),
        ),
        _bridge_seed(
            "bridge.cell",
            "bridge_cell_metrics.csv",
            (
                "network_seed",
                "prefix_k",
                "history_family_id",
                "history_condition",
                "b_anchor_id",
                "B_label",
                "c_anchor_id",
                "C_label",
                "layer1_written_ux_norm",
                "layer1_to_layer2_update_donor_transfer",
                "layer1_to_early_class_score_donor_transfer",
                "post_B_C_target_early_score",
                "post_passive_C_target_early_score",
                "layer1_swap_C_target_early_score",
            ),
        ),
        _bridge_single(
            "bridge.network",
            "history_rewrite_bridge/bridge/aggregate/data/metrics",
            "bridge_cohort_network_scalars.csv",
            (
                "network_seed",
                "prefix_k",
                "n_cells",
                "n_history_families",
                "n_b_anchors",
                "layer1_to_layer2_update_donor_transfer",
                "layer1_to_early_class_score_donor_transfer",
                "mean_layer1_written_ux_norm",
            ),
        ),
        _bridge_single(
            "bridge.inference",
            "history_rewrite_bridge/boundary_analysis/data/metrics",
            "boundary_transition_inference.csv",
            (
                "endpoint",
                "prefix_k",
                "n_networks",
                "mean",
                "ci95_low",
                "ci95_high",
                "fraction_above_zero",
                "holm_adjusted_p",
            ),
        ),
        _bridge_single(
            "bridge.boundary",
            "history_rewrite_bridge/boundary_analysis/data/metrics",
            "boundary_displacement_network_scalars.csv",
            (
                "network_seed",
                "prefix_k",
                "stage_k",
                "state_variable",
                "endpoint",
                "value",
                "n_sequences",
            ),
        ),
        _seed(
            "multi.region",
            MULTI_ROOT,
            "panel_b_region_ping_readout_bias.csv",
            (
                "network_seed",
                "sequence_id",
                "entry_condition",
                "old_mass",
                "middle_mass",
                "recent_mass",
                "other_mass",
                "silent_rate",
            ),
            unit="sequence",
        ),
        _seed(
            "multi.global_ping",
            MULTI_ROOT,
            "panel_c_global_ping_score_spike_prediction.csv",
            (
                "network_seed",
                "sequence_id",
                "early_window_ms",
                "score_quantile_bin",
                "mean_score",
                "n_sites",
                "spike_probability",
                "mean_early_spike_count",
            ),
            unit="sequence-quantile",
        ),
        _seed(
            "multi.real_probe",
            MULTI_ROOT,
            "panel_d_real_probe_score_spike_deflection.csv",
            (
                "network_seed",
                "sequence_id",
                "probe_id",
                "early_window_ms",
                "score_quantile_bin",
                "delta_spike_probability",
                "mean_delta_spike_count",
                "valid_site_count",
            ),
            unit="probe-quantile",
        ),
        _seed(
            "multi.interaction",
            MULTI_ROOT,
            "panel_e_overlap_gated_stsp_interaction.csv",
            (
                "network_seed",
                "sequence_id",
                "probe_id",
                "early_window_ms",
                "stsp_group_quantile",
                "overlap_threshold",
                "interaction_delta",
                "high_overlap_delta",
                "low_overlap_delta",
            ),
            unit="probe",
        ),
        _seed(
            "multi.ablation",
            MULTI_ROOT,
            "panel_f_high_stsp_overlap_ablation_summary.csv",
            (
                "network_seed",
                "sequence_id",
                "probe_id",
                "early_window_ms",
                "loss_condition",
                "loss_delta_spike_probability",
                "loss_mean_delta_spike_count",
            ),
            unit="probe",
        ),
        _seed(
            "multi.shuffle",
            MULTI_ROOT,
            "supp_s11g_score_shuffle_null.csv",
            (
                "network_seed",
                "endpoint",
                "metric",
                "condition",
                "observed_value",
                "null_value",
                "value",
            ),
        ),
        _seed(
            "multi.window_probe",
            MULTI_ROOT,
            "supp_s11c_real_probe_window_robustness.csv",
            (
                "network_seed",
                "metric",
                "condition",
                "early_window_ms",
                "value",
                "q1_mean",
                "q5_mean",
            ),
        ),
        _seed(
            "multi.window_interaction",
            MULTI_ROOT,
            "supp_s11d_overlap_interaction_window_robustness.csv",
            (
                "network_seed",
                "metric",
                "condition",
                "early_window_ms",
                "value",
                "fraction_positive",
            ),
        ),
        _seed(
            "multi.availability",
            MULTI_ROOT,
            "supp_s11e_overlap_site_availability.csv",
            (
                "network_seed",
                "metric",
                "condition",
                "stsp_group",
                "overlap_group",
                "value",
                "median_sites",
                "nonzero_fraction",
            ),
        ),
        _seed(
            "multi.ablation_pair",
            MULTI_ROOT,
            "supp_s11f_high_stsp_ablation_paired_difference.csv",
            (
                "network_seed",
                "metric",
                "condition",
                "value",
                "high_stsp_overlap",
                "matched_removal",
            ),
            unit="probe",
        ),
        _seed(
            "multi.threshold",
            MULTI_ROOT,
            "supp_s11h_threshold_sensitivity.csv",
            (
                "network_seed",
                "metric",
                "condition",
                "stsp_group_quantile",
                "overlap_threshold",
                "early_window_ms",
                "value",
            ),
            unit="probe",
        ),
        _seed(
            "pair.dual",
            PAIR_ROOT,
            "panel_b_dual_retention_metrics.csv",
            (
                "network_seed",
                "pair_id",
                "layer",
                "state_variable",
                "sim_to_A",
                "sim_to_B",
                "fusion_dual_score",
                "min_component_similarity",
            ),
            unit="pair",
        ),
        _seed(
            "pair.specificity",
            PAIR_ROOT,
            "panel_c_pair_specificity_metrics.csv",
            (
                "network_seed",
                "pair_id",
                "layer",
                "state_variable",
                "true_pair_score",
                "shuffled_pair_score",
                "true_minus_shuffled",
            ),
            unit="pair",
        ),
        _seed(
            "pair.mixture",
            PAIR_ROOT,
            "panel_d_linear_mixture_fit_metrics.csv",
            (
                "network_seed",
                "pair_id",
                "layer",
                "state_variable",
                "model_name",
                "cv_r2",
                "residual_norm_ratio",
                "linear_mixture_gain",
            ),
            unit="pair",
        ),
        _seed(
            "pair.interaction",
            PAIR_ROOT,
            "panel_d_crossfit_interaction_network_metrics.csv",
            (
                "network_seed",
                "layer",
                "state_variable",
                "delta_r2_interaction_beyond_bounded_saturation",
                "relative_mse_reduction_beyond_bounded_saturation",
            ),
        ),
        _seed(
            "pair.residual",
            PAIR_ROOT,
            "panel_d_linear_residual_pair_specificity_metrics.csv",
            (
                "network_seed",
                "pair_id",
                "layer",
                "state_variable",
                "residual_pair_specificity",
                "beyond_linear_pair_index",
            ),
            unit="pair",
        ),
        _seed(
            "pair.neutral_ping",
            PAIR_ROOT,
            "panel_e_neutral_ping_metrics.csv",
            (
                "network_seed",
                "state_condition",
                "P_A",
                "P_B",
                "P_pair",
                "P_silent",
                "pair_access_gain_SAB_vs_S0",
                "dual_access_balance",
            ),
        ),
        _seed(
            "pair.partial_cue",
            PAIR_ROOT,
            "panel_f_partial_cue_metrics.csv",
            (
                "network_seed",
                "state_condition",
                "target_item",
                "keep_prob",
                "P_target",
                "target_recovery_gain_vs_S0",
                "target_recovery_gain_vs_relevant_single",
                "target_recovery_gain_vs_irrelevant_single",
            ),
        ),
        _seed(
            "pair.delay_contrast",
            PAIR_ROOT,
            "supp_completion_delay_sweep_contrast.csv",
            (
                "network_seed",
                "delay2_ms",
                "keep_prob",
                "completion_gain_SAB_minus_SB",
                "completion_gain_SAB_minus_S0",
            ),
        ),
        _seed(
            "pair.delay_metrics",
            PAIR_ROOT,
            "supp_completion_delay_sweep_metrics.csv",
            (
                "network_seed",
                "delay2_ms",
                "state_condition",
                "keep_prob",
                "target_recovery_rate",
                "silent_rate",
            ),
        ),
        _seed(
            "pair.delay_layer",
            PAIR_ROOT,
            "supp_delay_layer_fused_state_metrics.csv",
            (
                "network_seed",
                "pair_id",
                "layer",
                "delay2_ms",
                "state_variable",
                "metric",
                "value",
            ),
            unit="pair",
        ),
        _seed(
            "pair.model_comparison",
            PAIR_ROOT,
            "supp_linear_mixture_model_comparison.csv",
            (
                "network_seed",
                "pair_id",
                "layer",
                "state_variable",
                "model_name",
                "cv_r2",
                "residual_norm_ratio",
            ),
            unit="pair",
        ),
        _seed(
            "pair.null",
            PAIR_ROOT,
            "supp_crossfit_null_network_metrics.csv",
            (
                "network_seed",
                "layer",
                "state_variable",
                "null_model",
                "endpoint",
                "delta_r2",
                "observed_reference_delta_r2",
            ),
        ),
        _seed(
            "pair.ping_sweep",
            PAIR_ROOT,
            "supp_ping_sweep_metrics.csv",
            (
                "network_seed",
                "sweep_type",
                "ping_amp",
                "ping_ms",
                "state_condition",
                "pair_member_readout_rate",
                "silent_rate",
            ),
        ),
        _seed(
            "prog.weights",
            PROGRESSIVE_ROOT,
            "panel_b_prefix_item_weights.csv",
            (
                "network_seed",
                "sequence_id",
                "delay_ms",
                "seq_len",
                "stage_k",
                "item_position",
                "item_weight",
                "is_latest",
            ),
            unit="sequence-item",
        ),
        _seed(
            "prog.update",
            PROGRESSIVE_ROOT,
            "panel_b_progressive_update_metrics.csv",
            (
                "network_seed",
                "sequence_id",
                "delay_ms",
                "seq_len",
                "stage_k",
                "layer",
                "state_variable",
                "state_displacement",
                "natural_decay_displacement",
                "observed_minus_natural_decay",
            ),
            unit="sequence-stage",
        ),
        _seed(
            "prog.serial",
            PROGRESSIVE_ROOT,
            "panel_b_morphology_serial_profile.csv",
            (
                "network_seed",
                "sequence_id",
                "seq_len",
                "delay_ms",
                "layer",
                "state_variable",
                "serial_position",
                "beta",
                "is_latest",
                "N_eff",
            ),
            unit="sequence-item",
        ),
        _seed(
            "prog.cue",
            PROGRESSIVE_ROOT,
            "panel_c_cue_specificity_memory_gain.csv",
            (
                "network_seed",
                "sequence_id",
                "seq_len",
                "delay_ms",
                "target_position",
                "cue_type",
                "keep_prob",
                "target_memory_gain",
                "seen_item_memory_gain",
                "silent_delta",
            ),
            unit="sequence-cue",
        ),
        _seed(
            "prog.access",
            PROGRESSIVE_ROOT,
            "panel_c_neutral_ping_access_summary.csv",
            (
                "network_seed",
                "sequence_id",
                "seq_len",
                "delay_ms",
                "state_condition",
                "P_seen_item",
                "P_silent",
                "latest_item_mass",
                "earlier_item_mass",
            ),
            unit="sequence",
        ),
        _seed(
            "prog.boundary",
            PROGRESSIVE_ROOT,
            "panel_d_functional_boundary_metrics.csv",
            (
                "network_seed",
                "sequence_id",
                "seq_len",
                "delay_ms",
                "keep_prob",
                "accessible_item_count",
                "rescued_count",
                "rescued_fraction",
                "functional_retention_index",
            ),
            unit="sequence",
        ),
        _seed(
            "prog.coupling",
            PROGRESSIVE_ROOT,
            "panel_e_morphology_function_coupling.csv",
            (
                "network_seed",
                "sequence_id",
                "seq_len",
                "delay_ms",
                "state_variable",
                "serial_position",
                "is_latest",
                "N_eff",
                "morphology_support_beta",
                "functional_gain_norm",
                "functional_gain",
                "keep_prob",
            ),
            unit="sequence-item",
        ),
        _seed(
            "prog.boundary_summary",
            PROGRESSIVE_ROOT,
            "panel_f_boundary_summary.csv",
            (
                "network_seed",
                "seq_len",
                "delay_ms",
                "N_eff",
                "accessible_item_count",
                "rescued_fraction",
                "support_gain_corr",
                "ping_latest_item_mass",
            ),
        ),
        _seed(
            "prog.order",
            PROGRESSIVE_ROOT,
            "panel_f_order_specificity_control.csv",
            (
                "network_seed",
                "sequence_id",
                "seq_len",
                "delay_ms",
                "condition",
                "order_specificity_index",
                "serial_support_corr",
            ),
            unit="sequence",
        ),
        _seed(
            "prog.peak_summary",
            PROGRESSIVE_ROOT,
            "supp_network_peak_valley_summary.csv",
            (
                "network_seed",
                "seq_len",
                "mean_peak_valley_delta",
                "fraction_structured_sequences",
                "mean_support_gini",
                "n_sequences",
            ),
        ),
        _seed(
            "prog.peak_contrast",
            PROGRESSIVE_ROOT,
            "supp_peak_valley_contrast.csv",
            (
                "network_seed",
                "sequence_id",
                "seq_len",
                "delay_ms",
                "peak_mean_support",
                "valley_mean_support",
                "random_mean_support",
                "peak_valley_delta",
            ),
            unit="sequence",
        ),
    )
}


@dataclass
class SourceStore:
    repo_root: Path
    paper_root: Path
    cache: dict[str, pd.DataFrame] = field(default_factory=dict)
    source_records: dict[str, dict[str, object]] = field(default_factory=dict)

    def read(self, key: str) -> pd.DataFrame:
        if key in self.cache:
            return self.cache[key]
        if key not in DATASET_SPECS:
            raise KeyError(f"Unknown dataset key: {key}")
        spec = DATASET_SPECS[key]
        frames: list[pd.DataFrame] = []
        paths = self._resolve_paths(spec)
        if not paths:
            raise FileNotFoundError(f"{key}: no source files resolved")
        for seed, path in paths:
            frame = self._read_one(spec, seed, path)
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        self._validate_network_cohort(spec, combined)
        self.cache[key] = combined
        return combined

    def validate(self, keys: Iterable[str]) -> None:
        for key in sorted(set(keys)):
            self.read(key)

    def verify_unchanged(self) -> list[str]:
        changed: list[str] = []
        for record in self.source_records.values():
            path = self.repo_root / str(record["relative_path"])
            if not path.exists():
                changed.append(f"missing after build: {path}")
                continue
            current = _sha256_file(path)
            if current != record["sha256"]:
                changed.append(f"hash changed: {path}")
        return changed

    def _resolve_paths(
        self,
        spec: DatasetSpec,
    ) -> list[tuple[int | None, Path]]:
        root = self.paper_root / spec.relative_root
        if spec.mode == "single":
            return [(None, root / spec.filename)]
        if spec.mode == "seed_metric":
            return [
                (
                    seed,
                    root / f"seed_{seed}" / "data" / "metrics" / spec.filename,
                )
                for seed in spec.seeds
            ]
        if spec.mode == "fixed_aggregate":
            return [(None, root / "aggregate" / spec.filename)]
        if spec.mode == "fixed_seed_metric":
            return [
                (
                    seed,
                    root / f"seed_{seed}" / "data" / "metrics" / spec.filename,
                )
                for seed in spec.seeds
            ]
        if spec.mode == "fixed_seed_raw":
            return [
                (
                    seed,
                    root / f"seed_{seed}" / "data" / "raw" / spec.filename,
                )
                for seed in spec.seeds
            ]
        if spec.mode == "bridge_seed_metric":
            return [
                (
                    seed,
                    root / f"seed_{seed}" / "data" / "metrics" / spec.filename,
                )
                for seed in spec.seeds
            ]
        raise ValueError(f"{spec.key}: unsupported mode {spec.mode!r}")

    def _read_one(
        self,
        spec: DatasetSpec,
        seed: int | None,
        path: Path,
    ) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"{spec.key}: missing parent {path}")
        frame = pd.read_csv(path)
        if seed is not None and "network_seed" not in frame.columns:
            frame.insert(0, "network_seed", int(seed))
        missing = [
            column
            for column in spec.required_columns
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"{spec.key}: {path} is missing required columns {missing}"
            )
        if seed is not None:
            observed = set(
                pd.to_numeric(frame["network_seed"], errors="raise")
                .astype(int)
                .unique()
            )
            if observed != {int(seed)}:
                raise ValueError(
                    f"{spec.key}: {path} expected network_seed={seed}, "
                    f"observed={sorted(observed)}"
                )
        resolved = path.resolve()
        relative = _display_path(resolved, self.repo_root)
        self.source_records[relative] = {
            "dataset_key": spec.key,
            "relative_path": relative,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "required_columns": ";".join(spec.required_columns),
            "scientific_unit": spec.scientific_unit,
            "post_hoc": bool(spec.post_hoc),
            "size_bytes": int(resolved.stat().st_size),
            "mtime_ns": int(resolved.stat().st_mtime_ns),
            "sha256": _sha256_file(resolved),
        }
        return frame

    @staticmethod
    def _validate_network_cohort(
        spec: DatasetSpec,
        frame: pd.DataFrame,
    ) -> None:
        if "network_seed" not in frame.columns or not spec.seeds:
            return
        observed = tuple(
            sorted(
                pd.to_numeric(frame["network_seed"], errors="raise")
                .astype(int)
                .unique()
            )
        )
        if observed != tuple(spec.seeds):
            raise ValueError(
                f"{spec.key}: expected cohort {tuple(spec.seeds)}, "
                f"observed {observed}"
            )


def all_contract_dataset_keys(contracts: Iterable[object]) -> set[str]:
    keys: set[str] = set()
    for contract in contracts:
        for panel in contract.panels:
            keys.update(panel.datasets)
    return keys


def validate_dataset_registry(keys: Iterable[str]) -> None:
    missing = sorted(set(keys) - set(DATASET_SPECS))
    if missing:
        raise ValueError(f"Panel contracts reference unknown datasets: {missing}")


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
