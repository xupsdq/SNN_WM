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

TASK_IDS = (
    TASK_ALL,
    TASK_TRIAL_SAMPLING,
    TASK_SUPPORT_BANK,
    TASK_PREPROBE_SUPPORT,
    TASK_EARLY_FIRING,
    TASK_LOCAL_EVENTS,
    TASK_SUPPORT_PERTURBATION,
    TASK_SUPPLEMENT,
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
    "SUPPORT_BANK_FILES",
    "TABLE_MANIFEST_COLUMNS",
    "TASK_ALL",
    "TASK_EARLY_FIRING",
    "TASK_IDS",
    "TASK_LOCAL_EVENTS",
    "TASK_PREPROBE_SUPPORT",
    "TASK_SUPPORT_BANK",
    "TASK_SUPPORT_PERTURBATION",
    "TASK_SUPPLEMENT",
    "TASK_TRIAL_SAMPLING",
    "TRIAL_SAMPLING_FILES",
    "normalize_reuse_mode",
]
