from __future__ import annotations

SCHEMA_VERSION = 1
SCHEMA_NAME = "fig2_runtime_artifacts"

TASK_ALL = "all"
TASK_PAIR_TRIAL_SPECS = "pair_trial_specs"
TASK_STATE_BANK = "state_bank"
TASK_CROSSFIT_SPLIT_SPECS = "crossfit_split_specs"
TASK_CROSSFIT_INTERACTION = "crossfit_interaction"
TASK_CROSSFIT_NULL_SPECS = "crossfit_null_specs"
TASK_CROSSFIT_NULL_CALIBRATION = "crossfit_null_calibration"
TASK_MORPHOLOGY = "morphology"
TASK_LINEAR_MIXTURE = "linear_mixture"
TASK_NEUTRAL_PING = "neutral_ping"
TASK_PARTIAL_CUE_MASK_SPECS = "partial_cue_mask_specs"
TASK_PARTIAL_CUE = "partial_cue"
TASK_PING_SWEEP = "ping_sweep"
TASK_COMPLETION_DELAY_BOUNDARY_BANK = "completion_delay_boundary_bank"
TASK_COMPLETION_DELAY_MASK_SPECS = "completion_delay_mask_specs"
TASK_COMPLETION_DELAY_SWEEP = "completion_delay_sweep"
TASK_FIXED_B_SPECS = "fixed_b_specs"
TASK_FIXED_B_INPUT_BANK = "fixed_b_input_bank"
TASK_FIXED_B_HISTORY_BANK = "fixed_b_history_bank"
TASK_FIXED_B_PROTOCOL = "fixed_b_protocol"
TASK_FIXED_B_REPLAY_BANK = "fixed_b_replay_bank"
TASK_FIXED_B_ROLLOUT_BANK = "fixed_b_rollout_bank"
TASK_FIXED_B_SWAP_BANK = "fixed_b_swap_bank"
TASK_FIXED_B_ANALYSIS = "fixed_b_analysis"
TASK_FIXED_B_COHORT_AGGREGATE = "fixed_b_cohort_aggregate"

TASK_SUPPLEMENT = "supplement"

TASK_IDS = (
    TASK_ALL,
    TASK_PAIR_TRIAL_SPECS,
    TASK_STATE_BANK,
    TASK_CROSSFIT_SPLIT_SPECS,
    TASK_CROSSFIT_INTERACTION,
    TASK_CROSSFIT_NULL_SPECS,
    TASK_CROSSFIT_NULL_CALIBRATION,
    TASK_MORPHOLOGY,
    TASK_LINEAR_MIXTURE,
    TASK_NEUTRAL_PING,
    TASK_PARTIAL_CUE_MASK_SPECS,
    TASK_PARTIAL_CUE,
    TASK_PING_SWEEP,
    TASK_COMPLETION_DELAY_BOUNDARY_BANK,
    TASK_COMPLETION_DELAY_MASK_SPECS,
    TASK_COMPLETION_DELAY_SWEEP,
    TASK_SUPPLEMENT,
    TASK_FIXED_B_SPECS,
    TASK_FIXED_B_INPUT_BANK,
    TASK_FIXED_B_HISTORY_BANK,
    TASK_FIXED_B_PROTOCOL,
    TASK_FIXED_B_REPLAY_BANK,
    TASK_FIXED_B_ROLLOUT_BANK,
    TASK_FIXED_B_SWAP_BANK,
    TASK_FIXED_B_ANALYSIS,
    TASK_FIXED_B_COHORT_AGGREGATE,

)

REUSE_MODES = ("off", "auto", "require", "force")

PAIR_SPEC_FILES = {
    "pair_trials": "pair_trials.csv",
    "candidate_pool": "pair_candidate_pool.csv",
}
TABLE_MANIFEST_COLUMNS = (
    "name",
    "filename",
    "rows",
    "columns",
    "sha256",
    "table_digest",
)
PAIR_SPEC_MANIFEST_COLUMNS = TABLE_MANIFEST_COLUMNS

CROSSFIT_SPLIT_FILE = "crossfit_split_specs.csv"
CROSSFIT_SPLIT_COLUMNS = (
    "network_seed",
    "pair_id",
    "A_image_id",
    "B_image_id",
    "component_id",
    "component_size",
    "fold",
)
CROSSFIT_NULL_SPEC_FILE = "crossfit_null_specs.csv"
CROSSFIT_NULL_SPEC_COLUMNS = (
    "network_seed",
    "null_model",
    "replicate",
    "random_seed",
    "feature_count",
    "feature_selection_seed",
    "noise_scale_ratio",
    "permutation_rule",
    "endpoint",
)
CROSSFIT_TASK_CONTRACTS = {
    TASK_CROSSFIT_SPLIT_SPECS: {
        "required_parent_artifacts": (TASK_PAIR_TRIAL_SPECS,),
        "output_files": (f"data/intermediates/{TASK_CROSSFIT_SPLIT_SPECS}/{CROSSFIT_SPLIT_FILE}",),
        "panel_consumers": (),
    },
    TASK_CROSSFIT_INTERACTION: {
        "required_parent_artifacts": (
            TASK_PAIR_TRIAL_SPECS,
            TASK_STATE_BANK,
            TASK_CROSSFIT_SPLIT_SPECS,
        ),
        "output_files": (
            "data/metrics/panel_d_crossfit_interaction_network_metrics.csv",
            "data/metrics/panel_d_crossfit_interaction_fold_metrics.csv",
            "data/metrics/panel_d_crossfit_interaction_pair_metrics.csv",
            "data/metrics/supp_crossfit_interaction_coefficients.csv",
            "data/metrics/panel_d_crossfit_interaction_analysis_spec.json",
        ),
        "panel_consumers": ("fig2:D", "supp_fig_s2:B"),
    },
    TASK_CROSSFIT_NULL_SPECS: {
        "required_parent_artifacts": (TASK_PAIR_TRIAL_SPECS, TASK_CROSSFIT_SPLIT_SPECS),
        "output_files": (f"data/intermediates/{TASK_CROSSFIT_NULL_SPECS}/{CROSSFIT_NULL_SPEC_FILE}",),
        "panel_consumers": (),
    },
    TASK_CROSSFIT_NULL_CALIBRATION: {
        "required_parent_artifacts": (
            TASK_PAIR_TRIAL_SPECS,
            TASK_STATE_BANK,
            TASK_CROSSFIT_SPLIT_SPECS,
            TASK_CROSSFIT_NULL_SPECS,
        ),
        "output_files": (
            "data/metrics/supp_crossfit_null_network_metrics.csv",
            "data/metrics/supp_crossfit_null_analysis_spec.json",
        ),
        "panel_consumers": ("supp_fig_s2:C",),
    },
}

FIXED_B_TASK_IDS = (
    TASK_FIXED_B_SPECS,
    TASK_FIXED_B_INPUT_BANK,
    TASK_FIXED_B_HISTORY_BANK,
    TASK_FIXED_B_PROTOCOL,
    TASK_FIXED_B_REPLAY_BANK,
    TASK_FIXED_B_ROLLOUT_BANK,
    TASK_FIXED_B_SWAP_BANK,
    TASK_FIXED_B_ANALYSIS,
    TASK_FIXED_B_COHORT_AGGREGATE,
)

FIXED_B_TASK_CONTRACTS = {
    TASK_FIXED_B_SPECS: {
        "required_parent_artifacts": (),
        "output_files": (
            f"data/intermediates/{TASK_FIXED_B_SPECS}/history_specs.csv",
            f"data/intermediates/{TASK_FIXED_B_SPECS}/b_anchor_specs.csv",
            f"data/intermediates/{TASK_FIXED_B_SPECS}/cell_specs.csv",
            f"data/intermediates/{TASK_FIXED_B_SPECS}/fold_specs.csv",
        ),
        "panel_consumers": (),
    },
    TASK_FIXED_B_INPUT_BANK: {
        "required_parent_artifacts": (TASK_FIXED_B_SPECS,),
        "output_files": (f"data/intermediates/{TASK_FIXED_B_INPUT_BANK}/arrays.npz",),
        "panel_consumers": (),
    },
    TASK_FIXED_B_HISTORY_BANK: {
        "required_parent_artifacts": (TASK_FIXED_B_SPECS, TASK_FIXED_B_INPUT_BANK),
        "output_files": (
            f"data/intermediates/{TASK_FIXED_B_HISTORY_BANK}/arrays.npz",
            f"data/intermediates/{TASK_FIXED_B_HISTORY_BANK}/restoration_audit.csv",
        ),
        "panel_consumers": (),
    },
    TASK_FIXED_B_PROTOCOL: {
        "required_parent_artifacts": (
            TASK_FIXED_B_SPECS,
            TASK_FIXED_B_INPUT_BANK,
            TASK_FIXED_B_HISTORY_BANK,
        ),
        "output_files": (
            "frozen_protocol/cache_key.json",
            "frozen_protocol/manifest.csv",
            "frozen_protocol/protocol.json",
        ),
        "panel_consumers": (),
    },
    TASK_FIXED_B_REPLAY_BANK: {
        "required_parent_artifacts": (
            TASK_FIXED_B_SPECS,
            TASK_FIXED_B_INPUT_BANK,
            TASK_FIXED_B_HISTORY_BANK,
        ),
        "output_files": (f"data/intermediates/{TASK_FIXED_B_REPLAY_BANK}/arrays.npz",),
        "panel_consumers": (),
    },
    TASK_FIXED_B_ROLLOUT_BANK: {
        "required_parent_artifacts": (
            TASK_FIXED_B_SPECS,
            TASK_FIXED_B_INPUT_BANK,
            TASK_FIXED_B_HISTORY_BANK,
            TASK_FIXED_B_REPLAY_BANK,
        ),
        "output_files": (
            f"data/intermediates/{TASK_FIXED_B_ROLLOUT_BANK}/arrays.npz",
            f"data/intermediates/{TASK_FIXED_B_ROLLOUT_BANK}/rollout_rows.csv",
        ),
        "panel_consumers": (),
    },
    TASK_FIXED_B_SWAP_BANK: {
        "required_parent_artifacts": (
            TASK_FIXED_B_SPECS,
            TASK_FIXED_B_INPUT_BANK,
            TASK_FIXED_B_HISTORY_BANK,
        ),
        "output_files": (
            f"data/intermediates/{TASK_FIXED_B_SWAP_BANK}/arrays.npz",
            f"data/intermediates/{TASK_FIXED_B_SWAP_BANK}/swap_rows.csv",
        ),
        "panel_consumers": (),
    },
    TASK_FIXED_B_ANALYSIS: {
        "required_parent_artifacts": (
            TASK_FIXED_B_SPECS,
            TASK_FIXED_B_INPUT_BANK,
            TASK_FIXED_B_HISTORY_BANK,
            TASK_FIXED_B_REPLAY_BANK,
            TASK_FIXED_B_ROLLOUT_BANK,
            TASK_FIXED_B_SWAP_BANK,
        ),
        "output_files": (
            "data/metrics/fixed_b_primary_crossfit_metrics.csv",
            "data/metrics/fixed_b_donor_transfer_metrics.csv",
            "data/metrics/fixed_b_prediction_checklist.csv",
            "data/metrics/fixed_b_single_seed_decision.json",
        ),
        "panel_consumers": (),
    },
    TASK_FIXED_B_COHORT_AGGREGATE: {
        "required_parent_artifacts": (),
        "output_files": (
            "aggregate/fixed_b_confirmatory_network_scalars.csv",
            "aggregate/fixed_b_confirmatory_inference.csv",
            "aggregate/fixed_b_confirmatory_verdict.json",
        ),
        "panel_consumers": (),
    },
}

BOUNDARY_STATE_KEYS = ("v_mem", "g_e", "res", "inh_trace", "u", "x")
STATE_BANK_ARRAY_VARIABLES = ("u", "x", "g")
STATE_BANK_MANIFEST_COLUMNS = (
    "network_seed",
    "state_condition",
    "layer",
    "state_variable",
    "shape",
    "storage_file",
    "storage_key",
    "sha256",
)
BOUNDARY_MANIFEST_COLUMNS = (
    "network_seed",
    "state_condition",
    "layer",
    "state_key",
    "shape",
    "path",
    "sha256",
)
COMPLETION_BOUNDARY_MANIFEST_COLUMNS = (
    "network_seed",
    "delay2_ms",
    "state_condition",
    "layer",
    "state_key",
    "shape",
    "path",
    "sha256",
)
COMPLETION_CONDITIONS = ("S0", "S_B", "S_AB")

PANEL_E_RAW_COLUMNS = (
    "network_seed",
    "pair_id",
    "state_condition",
    "ping_repeat",
    "ping_seed",
    "A_label",
    "B_label",
    "prediction",
    "pred_is_A",
    "pred_is_B",
    "pred_is_pair_member",
    "pred_is_other",
    "silent",
    "first_fire_time_ms",
    "ping_spike_count",
    "ping_energy",
    "readout_margin_A",
    "readout_margin_B",
)
SUPP_PING_SWEEP_RAW_COLUMNS = (
    "network_seed",
    "pair_id",
    "state_condition",
    "sweep_type",
    "ping_amp",
    "ping_ms",
    "ping_repeat",
    "A_label",
    "B_label",
    "prediction",
    "pred_is_A",
    "pred_is_B",
    "pred_is_pair_member",
    "pred_is_other",
    "silent",
    "first_fire_time_ms",
)
SUPP_COMPLETION_DELAY_RAW_COLUMNS = (
    "network_seed",
    "pair_id",
    "delay2_ms",
    "state_condition",
    "target_item",
    "target_label",
    "A_label",
    "B_label",
    "keep_prob",
    "repeat_id",
    "prediction",
    "correct_target",
    "pred_is_A",
    "pred_is_B",
    "pred_is_other",
    "silent",
    "first_fire_time_ms",
    "weak_probe_scale",
    "weak_spike_count",
)
PANEL_F_RAW_COLUMNS = (
    "network_seed",
    "pair_id",
    "state_condition",
    "target_item",
    "target_label",
    "other_pair_label",
    "keep_prob",
    "repeat_id",
    "mask_id",
    "prediction",
    "pred_is_target",
    "pred_is_A",
    "pred_is_B",
    "pred_is_pair_member",
    "pred_is_other_pair_member",
    "pred_is_other_class",
    "silent",
    "first_fire_time_ms",
    "mask_space",
    "weak_probe_scale",
    "weak_probe_noise",
    "weak_probe_metric_mode",
    "realized_keep_fraction",
    "cue_fraction_actual",
    "weak_spike_fraction",
    "same_mask_used_across_states",
    "cue_pixel_count",
    "target_foreground_count",
    "cue_energy",
    "encoded_spike_count",
)

WEAK_PROBE_MASK_COLUMNS = (
    "network_seed",
    "mask_id",
    "pair_id",
    "target_item",
    "target_label",
    "keep_prob",
    "repeat_id",
    "mask_seed",
    "mask_space",
    "same_mask_used_across_states",
    "weak_probe_scale",
    "weak_probe_noise",
    "realized_keep_fraction",
    "full_spike_count",
    "weak_spike_count",
    "weak_spike_fraction",
    "cue_pixel_count",
    "target_foreground_count",
    "cue_fraction_actual",
    "cue_energy",
    "encoded_spike_count",
)
COMPLETION_DELAY_MASK_COLUMNS = (
    "network_seed",
    "mask_id",
    "pair_id",
    "delay2_ms",
    "target_item",
    "target_label",
    "A_label",
    "B_label",
    "keep_prob",
    "repeat_id",
    "mask_seed",
    "mask_space",
    "same_mask_used_across_states",
    "weak_probe_scale",
    "weak_probe_noise",
    "realized_keep_fraction",
    "full_spike_count",
    "weak_spike_count",
    "weak_spike_fraction",
    "cue_pixel_count",
    "target_foreground_count",
    "cue_fraction_actual",
    "cue_energy",
    "encoded_spike_count",
)


def normalize_reuse_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in REUSE_MODES:
        choices = ", ".join(REUSE_MODES)
        raise ValueError(f"Unsupported reuse-artifacts mode: {value!r}. Expected one of: {choices}")
    return mode


__all__ = [
    "BOUNDARY_MANIFEST_COLUMNS",
    "BOUNDARY_STATE_KEYS",
    "COMPLETION_BOUNDARY_MANIFEST_COLUMNS",
    "COMPLETION_CONDITIONS",
    "COMPLETION_DELAY_MASK_COLUMNS",
    "CROSSFIT_SPLIT_COLUMNS",
    "CROSSFIT_SPLIT_FILE",
    "CROSSFIT_NULL_SPEC_COLUMNS",
    "CROSSFIT_NULL_SPEC_FILE",
    "CROSSFIT_TASK_CONTRACTS",
    "FIXED_B_TASK_CONTRACTS",
    "FIXED_B_TASK_IDS",
    "PAIR_SPEC_FILES",
    "PAIR_SPEC_MANIFEST_COLUMNS",
    "PANEL_E_RAW_COLUMNS",
    "PANEL_F_RAW_COLUMNS",
    "REUSE_MODES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "STATE_BANK_ARRAY_VARIABLES",
    "STATE_BANK_MANIFEST_COLUMNS",
    "SUPP_COMPLETION_DELAY_RAW_COLUMNS",
    "SUPP_PING_SWEEP_RAW_COLUMNS",
    "TABLE_MANIFEST_COLUMNS",
    "TASK_ALL",
    "TASK_COMPLETION_DELAY_BOUNDARY_BANK",
    "TASK_COMPLETION_DELAY_MASK_SPECS",
    "TASK_COMPLETION_DELAY_SWEEP",
    "TASK_CROSSFIT_INTERACTION",
    "TASK_CROSSFIT_NULL_CALIBRATION",
    "TASK_CROSSFIT_NULL_SPECS",
    "TASK_CROSSFIT_SPLIT_SPECS",
    "TASK_FIXED_B_ANALYSIS",
    "TASK_FIXED_B_COHORT_AGGREGATE",
    "TASK_FIXED_B_HISTORY_BANK",
    "TASK_FIXED_B_INPUT_BANK",
    "TASK_FIXED_B_REPLAY_BANK",
    "TASK_FIXED_B_PROTOCOL",
    "TASK_FIXED_B_ROLLOUT_BANK",
    "TASK_FIXED_B_SPECS",
    "TASK_FIXED_B_SWAP_BANK",
    "TASK_IDS",
    "TASK_LINEAR_MIXTURE",
    "TASK_MORPHOLOGY",
    "TASK_NEUTRAL_PING",
    "TASK_PAIR_TRIAL_SPECS",
    "TASK_PARTIAL_CUE",
    "TASK_PARTIAL_CUE_MASK_SPECS",
    "TASK_PING_SWEEP",
    "TASK_STATE_BANK",
    "TASK_SUPPLEMENT",
    "WEAK_PROBE_MASK_COLUMNS",
    "normalize_reuse_mode",
]
