from __future__ import annotations

SCHEMA_VERSION = 1
SCHEMA_NAME = "fig2_runtime_artifacts"

TASK_ALL = "all"
TASK_PAIR_TRIAL_SPECS = "pair_trial_specs"
TASK_STATE_BANK = "state_bank"
TASK_MORPHOLOGY = "morphology"
TASK_LINEAR_MIXTURE = "linear_mixture"
TASK_NEUTRAL_PING = "neutral_ping"
TASK_PARTIAL_CUE_MASK_SPECS = "partial_cue_mask_specs"
TASK_PARTIAL_CUE = "partial_cue"
TASK_PING_SWEEP = "ping_sweep"
TASK_COMPLETION_DELAY_BOUNDARY_BANK = "completion_delay_boundary_bank"
TASK_COMPLETION_DELAY_MASK_SPECS = "completion_delay_mask_specs"
TASK_COMPLETION_DELAY_SWEEP = "completion_delay_sweep"
TASK_SUPPLEMENT = "supplement"

TASK_IDS = (
    TASK_ALL,
    TASK_PAIR_TRIAL_SPECS,
    TASK_STATE_BANK,
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
    "PAIR_SPEC_FILES",
    "PAIR_SPEC_MANIFEST_COLUMNS",
    "REUSE_MODES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "STATE_BANK_ARRAY_VARIABLES",
    "STATE_BANK_MANIFEST_COLUMNS",
    "TABLE_MANIFEST_COLUMNS",
    "TASK_ALL",
    "TASK_COMPLETION_DELAY_BOUNDARY_BANK",
    "TASK_COMPLETION_DELAY_MASK_SPECS",
    "TASK_COMPLETION_DELAY_SWEEP",
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
