from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from src.experiments.paper_figures.common.bundle_io import (
    relative_to_root,
    write_artifact_manifest,
    write_json_file,
    write_run_log as write_bundle_run_log,
)
from src.experiments.paper_figures.fig1.constants import (
    DMS_DELAY_SWEEP_CONDITIONS,
    FIGURE_ID,
    MAIN_CONDITIONS,
    SUPP_CONDITIONS,
)
from src.experiments.paper_figures.fig1.types import ExperimentContext


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_config_files(ctx: ExperimentContext) -> None:
    payload = asdict(ctx.cfg)
    payload["strict_all_three_distinct_donor"] = True
    payload["shuffle_semantics"] = {
        "shuffle_compat_mode": bool(ctx.cfg.shuffle_compat_mode),
        "pure_substrate_only": bool(ctx.cfg.pure_substrate_only),
        "donor_plan": "constrained_all_three_label_distinct",
        "strict_all_three_distinct_donor": True,
        "substrate_definition": "legacy_shuffle",
        "dms_delay_ms": int(ctx.cfg.dms_delay_ms),
        "dms_num_trials": int(ctx.cfg.dms_num_trials),
        "dms_batch_size": int(ctx.cfg.dms_batch_size),
    }
    payload["delay_sweep_semantics"] = {
        "dms_delay_sweep_ms": [int(v) for v in ctx.cfg.dms_delay_sweep_ms],
        "delay_sweep_conditions": list(DMS_DELAY_SWEEP_CONDITIONS),
        "delay_sweep_design": (
            "sample and probe have different labels; sample-induced STSP state is captured after each delay "
            "and the same probe is evaluated under dynamic_intact and static_frozen conditions."
        ),
        "primary_metric": "static-minus-dynamic probe accuracy contrast vs delay",
        "contrast": "acc_static - acc_dynamic",
    }
    write_json_file(payload, ctx.config_dir / "run_config.json")
    write_json_file(payload, ctx.seed_dir / "run_config.json")
    write_json_file(
        {
            "main_panels": ["A", "B", "C", "D", "E"],
            "required_main_conditions": list(MAIN_CONDITIONS),
            "supplementary_outputs": [
                "supp_class_recall_by_digit.csv",
                "supp_confusion_matrix_long.csv",
                "supp_delay_decode_curve.csv",
                "supp_dms_delay_sweep_metrics.csv",
                "supp_dms_delay_sweep_contrast.csv",
                "supp_dms_delay_sweep_trial_readout.csv",
                "supp_substrate_shuffle_metrics.csv",
                "supp_phase_firing_rates.csv",
                "supp_trial_condition_audit.csv",
                "supp_dms_shuffle_donor_constraint_audit.csv",
            ],
        },
        ctx.config_dir / "figure_requirements.json",
    )
    write_json_file(
        {
            "conditions": {
                "dynamic_intact": "Boundary state restored; dynamic STSP during probe.",
                "ux_trial_shuffle": "Legacy-compatible pure-substrate trial shuffle of u_pre/x_pre. Full network state is reset before probe and only shuffled u/x substrate is restored.",
                "spike_state_shuffle": "Legacy-compatible pure-substrate trial shuffle of g_e/res/lateral_inh.inh_trace.",
                "membrane_state_shuffle": "Legacy-compatible pure-substrate trial shuffle of v_mem only.",
                "static_frozen": "Boundary state restored; probe uses stsp_mode=static_frozen.",
            },
            "donor_plan": "constrained_all_three_label_distinct",
            "strict_all_three_distinct_donor": True,
            "substrate_definition": "legacy_shuffle",
            "pure_substrate_only": bool(ctx.cfg.pure_substrate_only),
            "boundary_once_design": "Each DMS batch runs sample+delay once, captures the boundary, then runs one probe per condition.",
            "static_frozen_approximation": "The low-level API exposes static_frozen rather than exact freeze-current-u/x-gain; this approximation is recorded in outputs.",
        },
        ctx.config_dir / "condition_spec.json",
    )


def write_summary(ctx: ExperimentContext) -> dict[str, Any]:
    required_main = [
        ctx.metrics_dir / "panel_b_baseline_metrics_by_network.csv",
        ctx.metrics_dir / "panel_c_delay_decode_metrics.csv",
        ctx.metrics_dir / "panel_d_condition_metrics.csv",
        ctx.metrics_dir / "panel_e_attribution_metrics.csv",
    ]
    required_supp = [
        ctx.metrics_dir / "supp_class_recall_by_digit.csv",
        ctx.metrics_dir / "supp_confusion_matrix_long.csv",
        ctx.metrics_dir / "supp_delay_decode_curve.csv",
        ctx.metrics_dir / "supp_substrate_shuffle_metrics.csv",
        ctx.metrics_dir / "supp_phase_firing_rates.csv",
        ctx.metrics_dir / "supp_trial_condition_audit.csv",
        ctx.metrics_dir / "supp_dms_shuffle_donor_constraint_audit.csv",
    ]
    if ctx.cfg.run_dms_delay_sweep:
        required_supp.extend(
            [
                ctx.metrics_dir / "supp_dms_delay_sweep_metrics.csv",
                ctx.metrics_dir / "supp_dms_delay_sweep_contrast.csv",
                ctx.raw_dir / "supp_dms_delay_sweep_trial_readout.csv",
            ]
        )
    compat_paths = [
        ctx.metrics_dir / "compat_metrics_condition_summary.csv",
        ctx.metrics_dir / "compat_metrics_error_bias.csv",
        ctx.metrics_dir / "compat_metrics_collapse_summary.csv",
        ctx.metrics_dir / "compat_metrics_bootstrap_tests.csv",
    ]
    summary = {
        "figure": FIGURE_ID,
        "network_seed": int(ctx.cfg.network_seed),
        "run_mode": "single_network",
        "smoke": bool(ctx.cfg.smoke),
        "shuffle_compat_mode": bool(ctx.cfg.shuffle_compat_mode),
        "pure_substrate_only": bool(ctx.cfg.pure_substrate_only),
        "donor_plan": "constrained_all_three_label_distinct",
        "strict_all_three_distinct_donor": True,
        "substrate_definition": "legacy_shuffle",
        "dms_delay_ms": int(ctx.cfg.dms_delay_ms),
        "dms_delay_sweep_ms": [int(v) for v in ctx.cfg.dms_delay_sweep_ms],
        "dms_num_trials": int(ctx.cfg.dms_num_trials),
        "dms_batch_size": int(ctx.cfg.dms_batch_size),
        "delay_sweep_completed": bool(ctx.completed_modules.get("dms_delay_sweep", False)),
        "validation_passed": bool(ctx.completed_modules.get("dms_shuffle", False)) if ctx.cfg.run_dms_shuffle else None,
        "compatibility_metrics_available": all(path.exists() for path in compat_paths),
        "completed_modules": ctx.completed_modules,
        "output_files": ctx.output_files,
        "n_trials": ctx.n_trials,
        "conditions": {"main": list(MAIN_CONDITIONS), "supplementary": list(SUPP_CONDITIONS)},
        "warnings": ctx.warnings,
        "main_claim_supported_fields_available": all(path.exists() for path in required_main),
        "missing_for_main_figure": [relative_to_root(path, ctx.seed_dir) for path in required_main if not path.exists()],
        "missing_for_supplementary": [relative_to_root(path, ctx.seed_dir) for path in required_supp if not path.exists()],
    }
    summary.update(
        ctx.donor_constraint_summary
        or {
            "donor_constraint_audit_available": False,
            "strict_all_three_distinct_donor": True,
            "n_donor_sample_conflict": 0,
            "n_donor_probe_conflict": 0,
            "n_sample_probe_conflict": 0,
            "n_all_three_distinct_fail": 0,
            "n_self_swap": 0,
            "used_relaxed_rule": 0,
            "donor_constraint_status": "not_run" if not ctx.cfg.run_dms_shuffle else "failed",
        }
    )
    write_json_file(summary, ctx.seed_dir / "summary.json")
    ctx.output_files["summary"] = "summary.json"
    write_artifact_manifest(ctx, experiment_id=FIGURE_ID, title="Fig.1 functional STSP substrate")
    return summary


def write_run_log(ctx: ExperimentContext) -> None:
    write_bundle_run_log(ctx, now_text=utc_now())


__all__ = ["utc_now", "write_config_files", "write_run_log", "write_summary"]
