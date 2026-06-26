from __future__ import annotations

SCHEMA_NAME = "fig5_runtime_artifacts"
SCHEMA_VERSION = 1

TASK_ALL = "all"
TASK_TRIAL_SAMPLING = "trial_sampling"
TASK_SUPPORT_BANK = "preprobe_support_bank"
TASK_PREPROBE_SUPPORT = "preprobe_support"
TASK_EARLY_FIRING = "early_firing"
TASK_LOCAL_EVENTS = "local_events"
TASK_SUPPORT_PERTURBATION = "support_perturbation"
TASK_SUPPLEMENT = "supplement"
TASK_PROBE_STSP_UPDATE_BANK = "probe_stsp_update_bank"
TASK_POSTPROBE_STSP_UPDATE = "postprobe_stsp_update"

TASK_IDS = (
    TASK_ALL,
    TASK_TRIAL_SAMPLING,
    TASK_SUPPORT_BANK,
    TASK_PREPROBE_SUPPORT,
    TASK_EARLY_FIRING,
    TASK_LOCAL_EVENTS,
    TASK_SUPPORT_PERTURBATION,
    TASK_SUPPLEMENT,
    TASK_PROBE_STSP_UPDATE_BANK,
    TASK_POSTPROBE_STSP_UPDATE,
)

REUSE_MODES = ("off", "auto", "require", "force")

TABLE_MANIFEST_COLUMNS = (
    "name",
    "filename",
    "rows",
    "columns",
    "sha256",
    "table_digest",
)

ARRAY_MANIFEST_COLUMNS = (
    "name",
    "storage_file",
    "storage_key",
    "shape",
    "dtype",
    "sha256",
)

TRIAL_SAMPLING_FILES = {
    "trials": "local_competition_trials.csv",
    "trial_condition_audit": "supp_trial_condition_audit.csv",
}

SUPPORT_BANK_FILES = {
    "unit_groups": "unit_group_definitions.csv",
    "perturbation_sets": "perturbation_unit_sets.csv",
    "perturbation_ux_audit": "supp_perturbation_ux_audit.csv",
    "l1_stsp_perturbation_audit": "panel_d_l1_stsp_perturbation_audit.csv",
    "rollout_manifest": "rollout_manifest.csv",
    "trace_manifest": "layer1_probe_trace_manifest.csv",
}

PROBE_STSP_UPDATE_TABLE_FILES = {
    "trials": "local_competition_trials.csv",
    "unit_groups": "unit_group_definitions.csv",
    "condition_manifest": "condition_manifest.csv",
    "snapshot_manifest": "snapshot_manifest.csv",
}

POSTPROBE_L2_WRITEBACK_SCHEMA_VERSION = 1

PROBE_STSP_CONDITION_MANIFEST_COLUMNS = (
    "condition",
    "condition_label",
    "stsp_mode",
    "perturbation_mode",
    "perturbed_layer",
    "perturbed_variables",
    "branch_role",
)

SNAPSHOT_MANIFEST_COLUMNS = (
    "snapshot_id",
    "network_seed",
    "trial_id",
    "trial_chunk_id",
    "condition",
    "layer",
    "storage_file",
    "storage_key",
    "variable_set",
    "shape",
    "dtype",
    "sha256",
    "n_units",
    "parent_trial_hash",
    "parent_support_bank_digest",
)

POSTPROBE_L2_SUMMARY_COLUMNS = (
    "network_seed",
    "condition",
    "memory_control_condition",
    "layer",
    "unit_group",
    "n_trials",
    "n_l2_total_elements",
    "n_l2_prior_updated",
    "n_l2_prior_retained_memory",
    "n_l2_probe_update_dynamic",
    "n_l2_probe_update_static_opportunity",
    "n_l2_probe_update_static_actual",
    "n_l2_reupdate_dynamic",
    "n_l2_reupdate_static_opportunity",
    "n_memory_enabled_l2_update",
    "n_memory_enabled_l2_reupdate",
    "frac_prior_among_probe_updates_dynamic",
    "frac_prior_among_probe_updates_static",
    "dynamic_minus_static_frac_prior",
    "frac_reupdate_dynamic_among_prior",
    "frac_reupdate_static_among_prior",
    "frac_memory_enabled_reupdate_among_prior",
    "prior_update_base_rate",
)

POSTPROBE_L2_BY_TRIAL_COLUMNS = (
    "network_seed",
    "trial_id",
    "condition",
    "layer",
    "unit_group",
    "memory_control_condition",
    "n_l2_total_elements",
    "n_l2_prior_updated",
    "n_l2_prior_retained_memory",
    "n_l2_probe_update_dynamic",
    "n_l2_probe_update_static_opportunity",
    "n_l2_probe_update_static_actual",
    "n_l2_reupdate_dynamic",
    "n_l2_reupdate_static_opportunity",
    "n_memory_enabled_l2_update",
    "n_memory_enabled_l2_reupdate",
    "frac_prior_among_probe_updates_dynamic",
    "frac_prior_among_probe_updates_static",
    "dynamic_minus_static_frac_prior",
    "frac_reupdate_dynamic_among_prior",
    "frac_reupdate_static_among_prior",
    "frac_memory_enabled_reupdate_among_prior",
    "prior_update_base_rate",
)

POSTPROBE_L2_BY_NETWORK_COLUMNS = (
    "network_seed",
    "condition",
    "layer",
    "unit_group",
    "metric",
    "value",
    "n_trials",
    "n_l2_total_elements",
)

POSTPROBE_L2_HISTORY_BY_TRIAL_COLUMNS = (
    "network_seed",
    "trial_id",
    "layer",
    "memory_control_condition",
    "n_l2_total_elements",
    "n_l2_prior_updated",
    "n_l2_not_prior_updated",
    "n_l2_probe_update_dynamic",
    "n_l2_probe_update_static_opportunity",
    "n_l2_dynamic_prior_update",
    "n_l2_dynamic_nonprior_update",
    "n_l2_static_prior_opportunity",
    "n_l2_static_nonprior_opportunity",
    "dynamic_prior_fraction_among_updates",
    "dynamic_nonprior_fraction_among_updates",
    "static_prior_fraction_among_opportunities",
    "static_nonprior_fraction_among_opportunities",
    "dynamic_minus_static_prior_fraction",
    "dynamic_minus_static_nonprior_fraction",
    "p_dynamic_update_given_prior",
    "p_dynamic_update_given_nonprior",
    "p_static_opportunity_given_prior",
    "p_static_opportunity_given_nonprior",
    "dynamic_conditional_prior_minus_nonprior",
    "static_conditional_prior_minus_nonprior",
    "conditional_difference_in_differences",
)

POSTPROBE_L2_HISTORY_COMPOSITION_COLUMNS = (
    "network_seed",
    "condition",
    "condition_label",
    "condition_order",
    "memory_control_condition",
    "source_condition",
    "layer",
    "history_status",
    "history_label",
    "history_order",
    "n_trials",
    "n_l2_total_elements",
    "n_l2_history_sites",
    "n_l2_updated_sites",
    "n_l2_total_updated_sites",
    "fraction_among_updates",
    "update_probability_given_history",
    "dynamic_minus_static_prior_fraction",
    "dynamic_conditional_prior_minus_nonprior",
    "static_conditional_prior_minus_nonprior",
    "conditional_difference_in_differences",
    "denominator_definition",
)

POSTPROBE_L2_MEMORY_OVERLAP_COLUMNS = (
    "network_seed",
    "trial_id",
    "condition",
    "memory_control_condition",
    "layer",
    "unit_group",
    "n_l2_total_elements",
    "n_l2_prior_updated",
    "n_l2_reupdate_dynamic",
    "n_l2_reupdate_static_opportunity",
    "n_memory_enabled_l2_reupdate",
    "frac_prior_among_probe_updates_dynamic",
    "frac_prior_among_probe_updates_static",
    "dynamic_minus_static_frac_prior",
)

POSTPROBE_L1_FIRING_BRIDGE_COLUMNS = (
    "network_seed",
    "trial_id",
    "condition",
    "layer",
    "unit_group",
    "memory_control_condition",
    "n_total_l1_units",
    "n_prior_fired",
    "n_probe_fire_memory",
    "n_probe_fire_nomemory",
    "n_memory_enabled_fire",
    "n_memory_suppressed_fire",
    "n_changed_fire",
    "n_changed_prior_fired",
    "n_early_fired",
    "frac_changed",
    "frac_prior_among_changed",
    "frac_changed_among_prior",
    "prior_fire_base_rate",
    "enrichment_vs_prior_base_rate",
)

POSTPROBE_MAGNITUDE_QC_COLUMNS = (
    "network_seed",
    "trial_id",
    "condition",
    "layer",
    "unit_group",
    "n_total_stsp_elements",
    "mean_u_pre",
    "mean_x_pre",
    "mean_G_pre",
    "mean_u_post",
    "mean_x_post",
    "mean_G_post",
    "mean_delta_G",
    "mean_abs_delta_G",
    "early_fire_fraction",
)


def normalize_reuse_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in REUSE_MODES:
        choices = ", ".join(REUSE_MODES)
        raise ValueError(f"Unsupported reuse-artifacts mode: {value!r}. Expected one of: {choices}")
    return mode


__all__ = [
    "ARRAY_MANIFEST_COLUMNS",
    "REUSE_MODES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SNAPSHOT_MANIFEST_COLUMNS",
    "POSTPROBE_L1_FIRING_BRIDGE_COLUMNS",
    "POSTPROBE_L2_BY_NETWORK_COLUMNS",
    "POSTPROBE_L2_BY_TRIAL_COLUMNS",
    "POSTPROBE_L2_HISTORY_BY_TRIAL_COLUMNS",
    "POSTPROBE_L2_HISTORY_COMPOSITION_COLUMNS",
    "POSTPROBE_L2_MEMORY_OVERLAP_COLUMNS",
    "POSTPROBE_L2_WRITEBACK_SCHEMA_VERSION",
    "POSTPROBE_MAGNITUDE_QC_COLUMNS",
    "POSTPROBE_L2_SUMMARY_COLUMNS",
    "PROBE_STSP_CONDITION_MANIFEST_COLUMNS",
    "PROBE_STSP_UPDATE_TABLE_FILES",
    "SUPPORT_BANK_FILES",
    "TABLE_MANIFEST_COLUMNS",
    "TASK_ALL",
    "TASK_EARLY_FIRING",
    "TASK_IDS",
    "TASK_LOCAL_EVENTS",
    "TASK_POSTPROBE_STSP_UPDATE",
    "TASK_PREPROBE_SUPPORT",
    "TASK_PROBE_STSP_UPDATE_BANK",
    "TASK_SUPPORT_BANK",
    "TASK_SUPPORT_PERTURBATION",
    "TASK_SUPPLEMENT",
    "TASK_TRIAL_SAMPLING",
    "TRIAL_SAMPLING_FILES",
    "normalize_reuse_mode",
]
