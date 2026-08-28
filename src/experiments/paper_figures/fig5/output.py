from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.paper_figures.common.bundle_io import (
    json_safe,
    prepare_seed_dirs,
    relative_to_root,
    resolve_seed_dir,
    save_csv_with_registry,
    write_artifact_manifest,
    write_json_file,
    write_run_log,
)
from src.experiments.paper_figures.fig5.constants import (
    FIG5_BACKWARD_COMPATIBLE_OUTPUTS,
    FIG5_DESIGN_VERSION,
    FIG5_MAIN_REQUIRED_OUTPUTS,
    FIG5_S10_OUTPUTS,
    FIG5_S9_OUTPUTS,
    FIGURE_ID,
    LATE_PRE_WINDOW_MS,
    MAIN_CLAIM,
    MAIN_CONDITIONS,
    MAIN_PANEL_DESCRIPTIONS,
    MAX_WINNERS_PER_TRIAL,
    NULL_TYPES,
    PRIMARY_LAYER,
    PRIMARY_PRE_WINDOW_MS,
    REMOVED_FROM_MAIN_CONDITIONS,
    SUPP_CONDITIONS,
    SUPPLEMENT_PLAN,
    UNIT_GROUPS,
)
from src.experiments.paper_figures.fig5.subexperiments.helpers import _csv_nonempty
from src.experiments.paper_figures.fig5.types import ExperimentContext


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_config_files(ctx: ExperimentContext) -> None:
    payload = json_safe(asdict(ctx.cfg))
    write_json_file(payload, ctx.config_dir / "run_config.json")
    write_json_file(payload, ctx.seed_dir / "run_config.json")
    write_json_file(
        {
            "fig5_design_version": FIG5_DESIGN_VERSION,
            "main_panels": MAIN_PANEL_DESCRIPTIONS,
            "main_claim": MAIN_CLAIM,
            "supplement_plan": SUPPLEMENT_PLAN,
            "main_required_outputs": FIG5_MAIN_REQUIRED_OUTPUTS,
            "supplementary_outputs": {
                "S9": FIG5_S9_OUTPUTS,
                "S10": FIG5_S10_OUTPUTS,
            },
            "backward_compatible_outputs": FIG5_BACKWARD_COMPATIBLE_OUTPUTS,
            "overlap_mask_mode": str(ctx.cfg.overlap_mask_mode),
        },
        ctx.config_dir / "figure_requirements.json",
    )
    write_json_file(
        {
            "main_conditions": list(MAIN_CONDITIONS),
            "reference_condition": "static_frozen",
            "supplementary_controls": list(SUPP_CONDITIONS),
            "deprecated_conditions": list(REMOVED_FROM_MAIN_CONDITIONS),
            "panel_d": (
                "Layer1-only STSP attenuation/reset at the pre-probe boundary; probe input unchanged; "
                "tests whether Layer1 STSP state supports dynamic Layer1 transition composition."
            ),
            "perturbation_semantics": {
                "attenuate_l1_stsp": "Layer1 only: u_pre = U0 + alpha*(u_pre-U0); x_pre = 1 + alpha*(x_pre-1)",
                "reset_l1_stsp": "Layer1 only: u_pre = U0; x_pre = 1.0",
                "attenuate_overlap_high_support": "attenuate u_pre toward baseline for overlap high-support units; x_pre unchanged unless existing implementation differs",
                "reset_overlap_high_support": "reset u_pre to baseline and x_pre to 1.0 for overlap high-support units",
                "sham_perturbation": "matched procedural control without intended support reduction",
            },
            "static_frozen": "Probe uses model stsp_mode=static_frozen as the transition reference when a checkpoint is available.",
            "entry_mask_definition": "DoG encoded-spike spatial support"
            if str(ctx.cfg.overlap_mask_mode) == "encoded_spike"
            else "legacy thresholded image foreground",
            "overlap_mask_mode": str(ctx.cfg.overlap_mask_mode),
            "allow_proxy": False,
        },
        ctx.config_dir / "condition_spec.json",
    )
    write_json_file(
        {
            "primary_intervention": "attenuate_or_reset_layer1_stsp",
            "main_conditions": list(MAIN_CONDITIONS),
            "reference_condition": "static_frozen",
            "attenuate_definition": "u_pre = U_baseline + attenuation_factor * (u_pre - U_baseline); x_pre = 1 + attenuation_factor * (x_pre - 1)",
            "reset_definition": "u_pre = U_baseline; x_pre = 1.0",
            "attenuation_factor": float(ctx.cfg.perturbation_attenuation_factor),
            "perturbed_layer": PRIMARY_LAYER,
            "perturbed_variables": ["u_pre", "x_pre"],
            "probe_input_changed": False,
            "intervention_timing": "pre_probe_boundary",
            "boundary_policy": "restore_preprobe_boundary",
            "neutral_reset_restore_policy": False,
            "main_metric": "transition_composition",
            "old_global_all_layer_perturbation_demoted_to_legacy": True,
            "old_overlap_high_support_perturbation_demoted_to_supplement": True,
        },
        ctx.config_dir / "support_perturbation_spec.json",
    )
    write_json_file(
        {
            "local_kernel_radius": int(ctx.cfg.local_kernel_radius),
            "event_align_pre_steps": int(ctx.cfg.event_align_pre_steps),
            "event_align_post_steps": int(ctx.cfg.event_align_post_steps),
            "max_winners_per_trial": int(MAX_WINNERS_PER_TRIAL),
            "require_complete_alignment_window": True,
            "primary_pre_window_ms": list(PRIMARY_PRE_WINDOW_MS),
            "descriptive_late_pre_window_ms": list(LATE_PRE_WINDOW_MS),
            "aggregation": "event_to_trial_to_network",
        },
        ctx.config_dir / "event_selection_spec.json",
    )
    write_json_file({"null_types": list(NULL_TYPES), "n_null": int(ctx.cfg.n_null)}, ctx.config_dir / "null_baseline_spec.json")


def write_summary(ctx: ExperimentContext) -> dict[str, Any]:
    required_main: list[Path] = []
    if ctx.cfg.run_preprobe_support:
        required_main.append(ctx.metrics_dir / "panel_a_preprobe_support_metrics.csv")
    if ctx.cfg.run_early_firing:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_b_early_firing_transition_metrics.csv",
                ctx.metrics_dir / "panel_b_transition_summary_by_group.csv",
            ]
        )
    if ctx.cfg.run_local_events:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_c_winner_loser_event_metrics.csv",
                ctx.metrics_dir / "panel_c_event_trace_summary.csv",
            ]
        )
    if ctx.cfg.run_support_perturbation:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_d_l1_stsp_perturbation_unit_transitions.csv",
                ctx.metrics_dir / "panel_d_l1_stsp_perturbation_transition_summary.csv",
                ctx.metrics_dir / "panel_d_l1_stsp_perturbation_audit.csv",
                ctx.metrics_dir / "panel_d_l1_stsp_perturbation_contrast.csv",
            ]
        )
    required_supp: list[Path] = []
    if ctx.cfg.run_supplement:
        required_supp.extend(ctx.seed_dir / output for output in FIG5_S9_OUTPUTS + FIG5_S10_OUTPUTS)
    support_downstream_available = bool(
        ctx.availability.get(
            "support_perturbation_downstream_available",
            _csv_nonempty(ctx.metrics_dir / "panel_d_support_perturbation_trial_metrics.csv")
            and _csv_nonempty(ctx.metrics_dir / "panel_d_support_perturbation_node_metrics.csv"),
        )
    )
    if (
        ctx.cfg.run_support_perturbation
        and not support_downstream_available
        and not ctx.availability.get("support_perturbation_downstream_missing_reason")
    ):
        ctx.availability["support_perturbation_downstream_missing_reason"] = "panel_d_support_perturbation_metrics_missing_or_empty"
    perturbation_effect_available = bool(
        ctx.availability.get(
            "perturbation_effect_summary_available",
            _csv_nonempty(ctx.metrics_dir / "panel_d_perturbation_effect_summary.csv"),
        )
    )
    proxy_mode = False
    main_available = all(path.exists() for path in required_main)
    included_fig5d_groups = ["overlap_dominant", "probe_only_dominant", "random_matched"] + (
        ["balanced"] if bool(ctx.cfg.fig5d_include_balanced) else []
    )
    summary = {
        "figure": FIGURE_ID,
        "network_seed": int(ctx.cfg.network_seed),
        "run_mode": "single_network",
        "fig5_design_version": FIG5_DESIGN_VERSION,
        "main_claim": MAIN_CLAIM,
        "smoke": bool(ctx.cfg.smoke),
        "completed_modules": ctx.completed_modules,
        "output_files": ctx.output_files,
        "n_trials": int(ctx.n_trials),
        "n_events": int(ctx.n_events),
        "main_panels": MAIN_PANEL_DESCRIPTIONS,
        "supplement_plan": SUPPLEMENT_PLAN,
        "fig5e_removed_from_main": True,
        "old_flatten_conditions_removed": True,
        "old_flatten_nonoverlap_random_removed_from_main": True,
        "conditions": list(MAIN_CONDITIONS),
        "unit_groups": list(UNIT_GROUPS),
        "overlap_mask_mode": str(ctx.cfg.overlap_mask_mode),
        "mask_definition": {
            "overlap_mask_mode": str(ctx.cfg.overlap_mask_mode),
            "foreground_threshold": float(ctx.cfg.foreground_threshold),
            "sample_steps": int(ctx.cfg.sample_steps),
            "probe_steps": int(ctx.cfg.probe_steps),
        },
        "main_fig5d_conditions": list(MAIN_CONDITIONS),
        "reference_condition": "static_frozen",
        "perturbation_conditions": list(MAIN_CONDITIONS + SUPP_CONDITIONS),
        "current_perturbation_conditions": list(MAIN_CONDITIONS[1:] + SUPP_CONDITIONS),
        "deprecated_flatten_conditions": list(REMOVED_FROM_MAIN_CONDITIONS),
        "main_fig5d_metric": "transition_composition",
        "attenuation_definition": "u_pre = U0 + factor*(u_pre-U0); x_pre = 1 + factor*(x_pre-1)",
        "reset_definition": "u_pre = U0; x_pre = 1.0",
        "fig5d_l1_stsp_perturbation": {
            "enabled": True,
            "main_metric": "transition_composition",
            "included_unit_groups": included_fig5d_groups,
            "conditions": list(MAIN_CONDITIONS),
            "static_reference": "static_frozen",
            "perturbed_layer": PRIMARY_LAYER,
            "perturbed_variables": ["u_pre", "x_pre"],
            "attenuation_factor": float(ctx.cfg.perturbation_attenuation_factor),
            "boundary_policy": "restore_preprobe_boundary",
            "neutral_reset_restore_policy": False,
            "proxy_mode": False,
            "allow_proxy": False,
            "old_global_all_layer_perturbation_demoted_to_legacy": True,
            "old_overlap_high_support_perturbation_demoted_to_supplement": True,
        },
        "legacy_fig5d_outputs": [
            "panel_d_global_stsp_perturbation_transition_summary.csv",
            "panel_d_global_stsp_perturbation_audit.csv",
            "panel_d_perturbation_transition_summary_by_group.csv",
            "panel_d_support_perturbation_node_metrics.csv",
            "panel_d_perturbation_effect_summary.csv",
        ],
        "support_perturbation_downstream_available": support_downstream_available,
        "support_perturbation_downstream_missing_reason": ctx.availability.get("support_perturbation_downstream_missing_reason"),
        "perturbation_effect_summary_available": perturbation_effect_available,
        "perturbation_effect_summary_missing_reason": ctx.availability.get("perturbation_effect_summary_missing_reason"),
        "supplement_alias_missing_reasons": ctx.availability.get("supplement_alias_missing_reasons", {}),
        "proxy_mode": bool(proxy_mode),
        "allow_proxy": False,
        "final_scientific_use": bool(ctx.net is not None and ctx.encoder is not None and main_available),
        "event_selection": {"local_kernel_radius": int(ctx.cfg.local_kernel_radius), "n_events": int(ctx.n_events)},
        "warnings": ctx.warnings,
        "main_claim_supported_fields_available": bool(main_available),
        "missing_for_main_figure": [relative_to_root(path, ctx.seed_dir) for path in required_main if not path.exists()],
        "missing_for_supplementary": [relative_to_root(path, ctx.seed_dir) for path in required_supp if not path.exists()],
    }
    write_json_file(summary, ctx.seed_dir / "summary.json")
    ctx.output_files["summary"] = "summary.json"
    write_artifact_manifest(ctx, experiment_id=FIGURE_ID, title="Fig.5 local support competition")
    return summary


def write_run_log_file(ctx: ExperimentContext) -> None:
    write_run_log(ctx, now_text=utc_now())


def save_csv(ctx: ExperimentContext, df: pd.DataFrame, path: Path) -> None:
    save_csv_with_registry(ctx, df, path)


def prepare_dirs(seed_dir: Path) -> dict[str, Path]:
    return prepare_seed_dirs(seed_dir, include_root_layout=True)


def seed_output_dir(output_root: Path, network_seed: int) -> Path:
    return resolve_seed_dir(output_root, network_seed)


def rel(path: Path, root: Path) -> str:
    return relative_to_root(path, root)
