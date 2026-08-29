from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.config.units import ms
from src.experiments.paper_figures.common.bundle_io import (
    prepare_seed_dirs,
    relative_to_root,
    resolve_seed_dir,
    save_csv_with_registry,
    write_artifact_manifest,
    write_json_file,
    write_run_log,
)
from src.experiments.paper_figures.fig2.constants import (
    FIGURE_ID,
    MIXTURE_MODELS,
    RESIDUAL_TEMPLATE_DEFINITION,
    SINGLE_NETWORK_MODE,
    STATE_CONDITIONS,
)
from src.experiments.paper_figures.fig2.types import ExperimentContext


def write_config_files(ctx: ExperimentContext) -> None:
    cfg = ctx.cfg
    _write_json(_json_safe(asdict(cfg)), ctx.config_dir / "run_config.json")
    _write_json(_json_safe(asdict(cfg)), ctx.seed_dir / "run_config.json")
    _write_json(
        {
            "main_panels": ["A", "B", "C", "D", "E", "F"],
            "state_bank_required": True,
            "primary_layer": cfg.primary_layer,
            "primary_state_variable": cfg.primary_state_variable,
            "supplementary_outputs": [
                "supp_layerwise_morphology_metrics.csv",
                "supp_linear_mixture_model_comparison.csv",
                "supp_delay_layer_fused_state_metrics.csv",
                "supp_pair_sampling_audit.csv",
                "supp_additive_null_metrics.csv",
                "supp_completion_target_B_metrics.csv",
                "supp_ping_sweep_metrics.csv",
                "supp_completion_delay_sweep_metrics.csv",
                "supp_completion_delay_sweep_contrast.csv",
                "supp_trial_condition_audit.csv",
            ],
            "supplementary_figures": {
                "S3": {
                    "title": "Morphological controls for pair-specific fused STSP states.",
                    "outputs": ["supp_layerwise_morphology_metrics.csv", "supp_linear_mixture_model_comparison.csv"],
                },
                "S4": {
                    "title": "Functional robustness of fused-state access.",
                    "outputs": [
                        "supp_ping_sweep_metrics.csv",
                        "supp_completion_delay_sweep_metrics.csv",
                        "supp_completion_delay_sweep_contrast.csv",
                    ],
                },
            },
        },
        ctx.config_dir / "figure_requirements.json",
    )
    _write_json(
        {
            "state_conditions": list(STATE_CONDITIONS),
            "primary_representation": {"layer": cfg.primary_layer, "state_variable": cfg.primary_state_variable},
            "similarity": "centered_cosine",
            "pair_composite": "0.5*(S_A + S_B)",
            "neutral_ping": "real network rollout from restored S0/S_A/S_B/S_AB boundary states using class-uninformative ping_drive",
            "partial_cue": "real weak-probe rollout from restored S0/S_A/S_B/S_AB boundary states using Fig.4-compatible encoded-spike dropout by default",
            "functional_readout_source": "decode_prediction_and_fire_time_from_layer3",
            "weak_probe_mask_space": str(cfg.weak_probe_mask_space),
            "weak_probe_use_same_mask_across_states": bool(cfg.weak_probe_use_same_mask_across_states),
            "weak_probe_scale": float(cfg.weak_probe_scale),
            "weak_probe_noise": float(cfg.weak_probe_noise),
            "weak_probe_metric_mode": str(cfg.weak_probe_metric_mode),
            "fig4_weak_probe_compat_enabled": bool(cfg.weak_probe_mask_space == "encoded_spikes" and cfg.weak_probe_metric_mode == "fig4_compat"),
        },
        ctx.config_dir / "condition_spec.json",
    )
    _write_json(
        {
            "restore_mode": str(cfg.functional_restore_mode),
            "restore_convention": "Functional readout restores condition-specific STSP u/x states and resets non-STSP fast activity before ping/probe readout.",
            "ping_mode": str(cfg.ping_mode),
            "ping_amp": float(cfg.ping_amp),
            "ping_repeats": int(cfg.ping_repeats),
            "ping_noise": float(cfg.ping_noise),
            "run_ping_sweep": bool(cfg.run_ping_sweep),
            "ping_amp_sweep": list(cfg.ping_amp_sweep),
            "ping_ms_sweep": list(cfg.ping_ms_sweep),
            "weak_probe_keep_probs": list(cfg.weak_probe_keep_probs),
            "weak_probe_repeats": int(cfg.weak_probe_repeats),
            "weak_probe_mask_space": str(cfg.weak_probe_mask_space),
            "weak_probe_use_same_mask_across_states": bool(cfg.weak_probe_use_same_mask_across_states),
            "weak_probe_scale": float(cfg.weak_probe_scale),
            "weak_probe_noise": float(cfg.weak_probe_noise),
            "weak_probe_metric_mode": str(cfg.weak_probe_metric_mode),
            "fig4_weak_probe_compat_enabled": bool(cfg.weak_probe_mask_space == "encoded_spikes" and cfg.weak_probe_metric_mode == "fig4_compat"),
            "run_completion_delay_sweep": bool(cfg.run_completion_delay_sweep),
            "completion_delay_sweep_ms": list(cfg.completion_delay_sweep_ms),
            "completion_delay_keep_prob": float(cfg.completion_delay_keep_prob),
            "completion_delay_repeats": int(cfg.completion_delay_repeats),
            "foreground_threshold": float(cfg.foreground_threshold),
            "functional_restore_mode": str(cfg.functional_restore_mode),
            "decoder_name": "decode_prediction_and_fire_time_from_layer3",
            "proxy_used_for_main": False,
            "save_functional_traces": bool(cfg.save_functional_traces),
        },
        ctx.config_dir / "functional_readout_spec.json",
    )
    _write_json(
        {
            "models": list(MIXTURE_MODELS),
            "baseline_subtraction": "x_A=S_A-S0; x_B=S_B-S0; y_AB=S_AB-S0",
            "cv": {"folds": int(cfg.linear_mixture_cv_folds), "unit": "feature_dimensions"},
            "residual_template_definition": RESIDUAL_TEMPLATE_DEFINITION,
        },
        ctx.config_dir / "linear_mixture_spec.json",
    )


def write_summary(ctx: ExperimentContext) -> dict[str, Any]:
    required_main = [
        ctx.metrics_dir / "panel_b_dual_retention_metrics.csv",
        ctx.metrics_dir / "panel_c_pair_specificity_metrics.csv",
        ctx.metrics_dir / "panel_d_pair_level_organization_metrics.csv",
        ctx.metrics_dir / "panel_d_linear_mixture_fit_metrics.csv",
        ctx.metrics_dir / "panel_d_linear_residual_pair_specificity_metrics.csv",
        ctx.metrics_dir / "panel_e_neutral_ping_metrics.csv",
        ctx.metrics_dir / "panel_f_partial_cue_metrics.csv",
        ctx.metrics_dir / "panel_f_partial_cue_auc_metrics.csv",
        ctx.metrics_dir / "compat_fig4_weak_probe_summary.csv",
    ]
    required_supp = [
        ctx.metrics_dir / "supp_pair_sampling_audit.csv",
        ctx.metrics_dir / "supp_trial_condition_audit.csv",
    ]
    if ctx.cfg.run_morphology and ctx.cfg.run_linear_mixture:
        required_supp.append(ctx.metrics_dir / "supp_layerwise_morphology_metrics.csv")
    if ctx.cfg.run_linear_mixture:
        required_supp.extend(
            [
                ctx.metrics_dir / "supp_additive_null_metrics.csv",
                ctx.metrics_dir / "supp_linear_mixture_model_comparison.csv",
            ]
        )
    if ctx.cfg.run_partial_cue:
        required_supp.append(ctx.metrics_dir / "supp_completion_target_B_metrics.csv")
    if ctx.cfg.run_supplement:
        required_supp.append(ctx.metrics_dir / "supp_delay_layer_fused_state_metrics.csv")
    if ctx.cfg.run_ping_sweep:
        required_supp.extend(
            [
                ctx.raw_dir / "supp_ping_sweep_trial_readout.csv",
                ctx.metrics_dir / "supp_ping_sweep_metrics.csv",
            ]
        )
    if ctx.cfg.run_completion_delay_sweep:
        required_supp.extend(
            [
                ctx.raw_dir / "supp_completion_delay_sweep_trial_readout.csv",
                ctx.metrics_dir / "supp_completion_delay_sweep_metrics.csv",
                ctx.metrics_dir / "supp_completion_delay_sweep_contrast.csv",
            ]
        )
    summary = {
        "figure": FIGURE_ID,
        "network_seed": int(ctx.cfg.network_seed),
        "run_mode": SINGLE_NETWORK_MODE,
        "smoke": bool(ctx.cfg.smoke),
        "completed_modules": ctx.completed_modules,
        "output_files": ctx.output_files,
        "n_pairs": int(ctx.n_pairs),
        "state_conditions": list(STATE_CONDITIONS),
        "linear_mixture_models": list(MIXTURE_MODELS),
        "fig2_supplement_plan": {
            "S3": "Morphological controls: WPRI across layers, residual pair-specificity across layers, linear mixture model comparison.",
            "S4": "Functional robustness: ping amplitude sweep, ping duration sweep, completion gain across post-pair retention delays.",
        },
        "ping_sweep_completed": bool(ctx.completed_modules.get("ping_sweep", False)),
        "completion_delay_sweep_completed": bool(ctx.completed_modules.get("completion_delay_sweep", False)),
        "warnings": ctx.warnings,
        "residual_template_definition": RESIDUAL_TEMPLATE_DEFINITION,
        "functional_readout_mode": "real_network_rollout",
        "neutral_ping_proxy_used_for_main": False,
        "partial_cue_proxy_used_for_main": False,
        "weak_probe_mask_space": str(ctx.cfg.weak_probe_mask_space),
        "weak_probe_use_same_mask_across_states": bool(ctx.cfg.weak_probe_use_same_mask_across_states),
        "weak_probe_scale": float(ctx.cfg.weak_probe_scale),
        "weak_probe_noise": float(ctx.cfg.weak_probe_noise),
        "weak_probe_metric_mode": str(ctx.cfg.weak_probe_metric_mode),
        "fig4_weak_probe_compat_enabled": bool(ctx.cfg.weak_probe_mask_space == "encoded_spikes" and ctx.cfg.weak_probe_metric_mode == "fig4_compat"),
        "proxy_diagnostics_available": bool((ctx.metrics_dir / "supp_functional_proxy_diagnostics.csv").exists()),
        "proxy_used_for_main": False,
        "functional_readout_note": "Panel F uses Fig.4-compatible encoded-spike dropout weak probes by default, extended to S0/S_A/S_B/S_AB and bidirectional A/B target recovery.",
        "main_claim_supported_fields_available": all(path.exists() for path in required_main),
        "missing_for_main_figure": [_rel(path, ctx.seed_dir) for path in required_main if not path.exists()],
        "missing_for_supplementary": [_rel(path, ctx.seed_dir) for path in required_supp if not path.exists()],
    }
    _write_json(summary, ctx.seed_dir / "summary.json")
    ctx.output_files["summary"] = "summary.json"
    write_artifact_manifest(ctx, experiment_id=FIGURE_ID, title="Fig.2 pair-fused STSP state")
    return summary


def write_run_log_file(ctx: ExperimentContext) -> None:
    write_run_log(ctx, now_text=utc_now())


def _save_csv(ctx: ExperimentContext, df: pd.DataFrame, path: Path) -> None:
    save_csv_with_registry(ctx, df, path)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    write_json_file(payload, path, json_safe_fn=_json_safe)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def prepare_dirs(seed_dir: Path) -> dict[str, Path]:
    return prepare_seed_dirs(seed_dir, include_root_layout=True)


def seed_output_dir(output_root: Path, network_seed: int) -> Path:
    return resolve_seed_dir(output_root, network_seed)


def _rel(path: Path, root: Path) -> str:
    return relative_to_root(path, root)


def ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ms_to_steps",
    "prepare_dirs",
    "seed_output_dir",
    "utc_now",
    "write_config_files",
    "write_run_log_file",
    "write_summary",
]
