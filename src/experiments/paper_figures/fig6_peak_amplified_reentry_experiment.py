from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - import error is reported by runtime validation.
    torch = None  # type: ignore[assignment]

from src.config.units import ms
try:
    from src.experiments.common.dataset import build_class_index, encode_images
    from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
    from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
    from src.experiments.common.model_io import load_model_and_encoder
    from src.experiments.common.monitored_dms import snapshot_boundary_state
    from src.experiments.common.ping_common import prepare_network_state
    from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
    from src.experiments.common.runtime import resolve_device, seed_everything
    _COMMON_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only in minimal Python envs.
    _COMMON_IMPORT_ERROR = exc

    def build_class_index(dataset, num_classes: int) -> dict[int, list[int]]:
        out = {label: [] for label in range(int(num_classes))}
        for idx in range(len(dataset)):
            label = int(dataset[idx][1])
            if label in out:
                out[label].append(idx)
        return out

    def encode_images(*args, **kwargs):
        raise RuntimeError(f"Shared spike encoder unavailable: {_COMMON_IMPORT_ERROR}")

    def decode_prediction_and_fire_time_from_layer3(*args, **kwargs):
        raise RuntimeError(f"Shared layer3 decoder unavailable: {_COMMON_IMPORT_ERROR}")

    def load_mnist_skeleton_dataset(*args, **kwargs):
        raise RuntimeError(f"Shared MNIST loader unavailable: {_COMMON_IMPORT_ERROR}")

    def load_model_and_encoder(*args, **kwargs):
        raise RuntimeError(f"Shared model loader unavailable: {_COMMON_IMPORT_ERROR}")

    def snapshot_boundary_state(*args, **kwargs):
        return {}

    def prepare_network_state(*args, **kwargs) -> None:
        return None

    def build_run_info(**kwargs):
        return dict(kwargs)

    def finalize_run_info(meta_dir: Path, run_info: Mapping[str, Any], *, status: str):
        payload = dict(run_info)
        payload["status"] = status
        write_run_info(meta_dir, payload)

    def write_run_info(meta_dir: Path, run_info: Mapping[str, Any]):
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "run_info.json").write_text(json.dumps(_json_safe(run_info), indent=2, sort_keys=True), encoding="utf-8")

    def resolve_device(device: str):
        return "cpu"

    def seed_everything(seed: int) -> None:
        np.random.seed(int(seed))
from src.experiments.paper_figures.common.bundle_io import (
    prepare_seed_dirs,
    relative_to_root,
    resolve_seed_dir,
    save_csv_with_registry,
    write_json_file,
    write_run_log,
)
from src.experiments.paper_figures.common.progress import ProgressTracker, planned_phases
from src.experiments.paper_figures.fig6.types import ExperimentContext, Fig6Config, PeakAmplifiedReentryBank
from src.plotting.common.io import apply_publication_style, save_figure_all_formats

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _progress(iterable, *, total=None, desc: str = "", enabled: bool = True):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _config_from_args(args)
    run(cfg)
    return 0


def run(cfg: Fig6Config) -> dict[str, Any]:
    cfg = _effective_config(cfg)
    if torch is None:
        raise RuntimeError("PyTorch is required for Fig.6 real-network runs.")
    if _COMMON_IMPORT_ERROR is not None:
        raise RuntimeError(f"Shared Fig.6 experiment imports failed: {_COMMON_IMPORT_ERROR}") from _COMMON_IMPORT_ERROR
    seed_everything(int(cfg.network_seed))
    seed_dir = _resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = _prepare_dirs(seed_dir)
    warnings_list: list[str] = []
    device = resolve_device(cfg.device)
    dataset = _load_dataset_required(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, 10)
    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Fig.6 model checkpoint not found: {model_path}")
    try:
        net, encoder = load_model_and_encoder(
            cfg.model_path,
            device=device,
            dt=cfg.dt,
            max_duration_ms=max(cfg.sample_ms, cfg.probe_ms, 100),
        )
    except Exception as exc:
        raise RuntimeError(f"Fig.6 model load failed for {cfg.model_path}: {exc}") from exc

    ctx = ExperimentContext(
        cfg=cfg,
        seed_dir=seed_dir,
        config_dir=dirs["config"],
        trial_specs_dir=dirs["trial_specs"],
        raw_dir=dirs["raw"],
        metrics_dir=dirs["metrics"],
        debug_dir=dirs["debug"],
        meta_dir=dirs["meta"],
        device=device,
        dataset=dataset,
        class_index=class_index,
        net=net,
        encoder=encoder,
        warnings=warnings_list,
        output_files={},
        completed_modules={},
        run_log=[f"{_now()} start {FIGURE_ID} seed={cfg.network_seed} smoke={cfg.smoke}"],
    )
    run_info = build_run_info(
        experiment_name=FIGURE_ID,
        output_dir=seed_dir,
        entry_script="src.experiments.paper_figures.fig6_peak_amplified_reentry_experiment",
        seed=cfg.network_seed,
        dataset=f"MNIST:{cfg.split}",
        command=" ".join(sys.argv),
        model_path=cfg.model_path,
        status="running",
    )
    write_run_info(ctx.meta_dir, run_info)
    try:
        needs_sequence_trials = any(
            (
                cfg.run_sequence_bank,
                cfg.run_peak_source_attribution,
                cfg.run_peak_update_history,
                cfg.run_peak_input_overlap_origin,
                cfg.run_real_reentry_rollout,
                cfg.run_real_downstream_metrics,
                cfg.run_peak_enrichment,
                cfg.run_update_recency_model,
                cfg.run_peak_weighted_overlap,
                cfg.run_reentry_prediction,
                cfg.run_downstream_prediction,
                cfg.run_peak_perturbation,
                cfg.run_supplement,
                cfg.run_legacy_supplement,
                cfg.run_score_shuffle_null,
                cfg.run_overlap_threshold_sensitivity,
                cfg.run_field_ping_readout,
                cfg.run_global_ping_score_spike_prediction,
                cfg.run_ping_score_spike_prediction,
                cfg.run_real_probe_score_spike_deflection,
                cfg.run_overlap_gated_stsp_recruitment,
                cfg.run_high_stsp_overlap_ablation,
                cfg.run_score_basin_sparsification,
                cfg.run_fig6_downstream_exploratory,
            )
        )
        needs_bank = needs_sequence_trials
        needs_later_probe_trials = any(
            (
                cfg.run_real_reentry_rollout,
                cfg.run_real_downstream_metrics,
                cfg.run_reentry_prediction,
                cfg.run_downstream_prediction,
                cfg.run_peak_perturbation,
                cfg.run_supplement,
                cfg.run_score_shuffle_null,
                cfg.run_overlap_threshold_sensitivity,
                cfg.run_real_probe_score_spike_deflection,
                cfg.run_overlap_gated_stsp_recruitment,
                cfg.run_high_stsp_overlap_ablation,
                cfg.run_score_basin_sparsification,
                cfg.run_fig6_downstream_exploratory,
            )
        )
        needs_real_rollout = any(
            (
                cfg.run_real_reentry_rollout,
                cfg.run_real_downstream_metrics,
                cfg.run_peak_weighted_overlap,
                cfg.run_reentry_prediction,
                cfg.run_downstream_prediction,
                cfg.run_legacy_supplement,
                cfg.run_peak_perturbation,
            )
        )
        sequence_trials: pd.DataFrame | None = None
        bank: PeakAmplifiedReentryBank | None = None
        progress = ProgressTracker(
            ctx,
            planned_phases(
                (
                    ("config", True),
                    ("sequence_trials", needs_sequence_trials),
                    ("sequence_bank", needs_bank),
                    ("peak_source_attribution", cfg.run_peak_source_attribution or cfg.run_legacy_supplement),
                    ("peak_update_history", cfg.run_peak_update_history or cfg.run_legacy_supplement),
                    ("peak_input_overlap_origin", cfg.run_peak_input_overlap_origin or cfg.run_legacy_supplement),
                    ("peak_enrichment", cfg.run_peak_enrichment),
                    ("update_recency_model", cfg.run_update_recency_model),
                    ("peak_weighted_overlap", cfg.run_peak_weighted_overlap),
                    ("later_probe_trials", needs_later_probe_trials),
                    ("real_reentry_rollout", needs_real_rollout),
                    ("real_reentry_metrics", cfg.run_real_reentry_rollout or cfg.run_legacy_supplement or cfg.run_peak_weighted_overlap),
                    ("real_downstream_metrics", cfg.run_real_downstream_metrics or cfg.run_legacy_supplement),
                    ("reentry_prediction", cfg.run_reentry_prediction),
                    ("downstream_prediction", cfg.run_downstream_prediction),
                    ("field_ping_readout", cfg.run_field_ping_readout),
                    ("global_ping_score_spike_prediction", cfg.run_global_ping_score_spike_prediction or cfg.run_ping_score_spike_prediction),
                    ("real_probe_score_spike_deflection", cfg.run_real_probe_score_spike_deflection),
                    ("overlap_gated_stsp_recruitment", cfg.run_overlap_gated_stsp_recruitment),
                    ("high_stsp_overlap_ablation", cfg.run_high_stsp_overlap_ablation),
                    ("score_basin_sparsification", cfg.run_score_basin_sparsification),
                    ("fig6_downstream_exploratory", cfg.run_fig6_downstream_exploratory),
                    ("peak_perturbation", cfg.run_peak_perturbation),
                    ("supplement", cfg.run_supplement),
                    ("score_shuffle_null", cfg.run_score_shuffle_null),
                    ("overlap_threshold_sensitivity", cfg.run_overlap_threshold_sensitivity),
                    ("legacy_supplement", cfg.run_legacy_supplement),
                    ("aliases", True),
                    ("mechanism_metadata", True),
                    ("debug_figures", cfg.save_debug_figures),
                    ("summary", True),
                )
            ),
            fig_id="fig6",
        )
        with progress.phase("config"):
            _write_config_files(ctx)
        if needs_sequence_trials:
            with progress.phase("sequence_trials"):
                sequence_trials = build_sequence_trials(ctx)
        if needs_bank and sequence_trials is not None:
            with progress.phase("sequence_bank"):
                bank = run_sequence_bank(ctx, sequence_trials)
        if bank is not None and (cfg.run_peak_source_attribution or cfg.run_legacy_supplement):
            with progress.phase("peak_source_attribution"):
                loo_bank = run_leave_one_item_out_support_bank(ctx, bank)
                compute_peak_source_attribution(ctx, bank, loo_bank)
        if bank is not None and (cfg.run_peak_update_history or cfg.run_legacy_supplement):
            with progress.phase("peak_update_history"):
                compute_peak_update_history(ctx, bank)
        if bank is not None and (cfg.run_peak_input_overlap_origin or cfg.run_legacy_supplement):
            with progress.phase("peak_input_overlap_origin"):
                compute_peak_input_overlap_origin(ctx, bank)
        if bank is not None and cfg.run_peak_enrichment:
            with progress.phase("peak_enrichment"):
                define_final_peaks_and_update_groups(ctx, bank)
        if bank is not None and cfg.run_update_recency_model:
            with progress.phase("update_recency_model"):
                if not (ctx.metrics_dir / "supp_legacy_panel_a_multi_recent_peak_enrichment.csv").exists():
                    define_final_peaks_and_update_groups(ctx, bank)
                fit_update_recency_support_models(ctx, bank)
        if bank is not None and cfg.run_peak_weighted_overlap:
            with progress.phase("peak_weighted_overlap"):
                compute_peak_weighted_overlap_definitions(ctx, bank)
        if bank is not None and needs_later_probe_trials:
            with progress.phase("later_probe_trials"):
                build_later_probe_peak_overlap_trials(ctx, bank)
        if bank is not None and needs_real_rollout:
            with progress.phase("real_reentry_rollout"):
                if bank.probe_trials.empty:
                    build_later_probe_peak_overlap_trials(ctx, bank)
                run_real_probe_reentry_rollouts(ctx, bank)
        if bank is not None and ((cfg.run_real_reentry_rollout or cfg.run_legacy_supplement) or (cfg.run_peak_weighted_overlap and not bank.reentry_metrics.empty)):
            with progress.phase("real_reentry_metrics"):
                if cfg.run_real_reentry_rollout or cfg.run_legacy_supplement:
                    compute_real_peak_weighted_reentry_metrics(ctx, bank)
                if cfg.run_peak_weighted_overlap and not bank.reentry_metrics.empty:
                    compute_real_peak_weighted_reentry_metrics(ctx, bank)
        if bank is not None and (cfg.run_real_downstream_metrics or cfg.run_legacy_supplement):
            with progress.phase("real_downstream_metrics"):
                compute_real_peak_overlap_downstream_metrics(ctx, bank)
        if bank is not None and cfg.run_reentry_prediction:
            with progress.phase("reentry_prediction"):
                compute_peak_weighted_reentry_metrics(ctx, bank)
        if bank is not None and cfg.run_downstream_prediction:
            with progress.phase("downstream_prediction"):
                compute_peak_weighted_downstream_metrics(ctx, bank)
        if bank is not None and cfg.run_field_ping_readout:
            with progress.phase("field_ping_readout"):
                compute_field_ping_readout(ctx, bank)
        if bank is not None and (cfg.run_global_ping_score_spike_prediction or cfg.run_ping_score_spike_prediction):
            with progress.phase("global_ping_score_spike_prediction"):
                compute_global_ping_score_spike_prediction(ctx, bank)
        if bank is not None and cfg.run_real_probe_score_spike_deflection:
            with progress.phase("real_probe_score_spike_deflection"):
                compute_real_probe_score_spike_deflection(ctx, bank)
        if bank is not None and cfg.run_overlap_gated_stsp_recruitment:
            with progress.phase("overlap_gated_stsp_recruitment"):
                compute_overlap_gated_stsp_recruitment(ctx, bank)
        if bank is not None and cfg.run_high_stsp_overlap_ablation:
            with progress.phase("high_stsp_overlap_ablation"):
                compute_high_stsp_overlap_ablation(ctx, bank)
        if bank is not None and cfg.run_score_basin_sparsification:
            with progress.phase("score_basin_sparsification"):
                compute_score_basin_sparsification(ctx, bank)
        if bank is not None and cfg.run_fig6_downstream_exploratory:
            with progress.phase("fig6_downstream_exploratory"):
                compute_fig6_downstream_exploratory(ctx, bank)
        if bank is not None and cfg.run_peak_perturbation:
            with progress.phase("peak_perturbation"):
                compute_route_peak_perturbation_outputs(ctx, bank)
        if bank is not None and cfg.run_supplement:
            with progress.phase("supplement"):
                compute_supplement_outputs(ctx, bank)
        if bank is not None and cfg.run_score_shuffle_null:
            with progress.phase("score_shuffle_null"):
                compute_score_shuffle_null_extension(ctx, bank)
        if bank is not None and cfg.run_overlap_threshold_sensitivity:
            with progress.phase("overlap_threshold_sensitivity"):
                compute_overlap_threshold_sensitivity_extension(ctx, bank)
        if bank is not None and cfg.run_legacy_supplement:
            with progress.phase("legacy_supplement"):
                compute_legacy_supplement_outputs(ctx, bank)
        with progress.phase("aliases"):
            write_fig6_supplement_aliases(ctx)
        if bank is not None:
            with progress.phase("mechanism_metadata"):
                write_global_mechanism_metadata(ctx)
        if cfg.save_debug_figures:
            with progress.phase("debug_figures"):
                save_debug_figures(ctx)
        with progress.phase("summary"):
            _flush_score_audits(ctx)
            summary = _write_summary(ctx)
            _write_run_log(ctx)
        finalize_run_info(ctx.meta_dir, run_info, status="success")
        return summary
    except Exception:
        finalize_run_info(ctx.meta_dir, run_info, status="failed")
        raise


def _effective_config(cfg: Fig6Config) -> Fig6Config:
    if not cfg.force_main_outputs:
        return cfg
    return replace(
        cfg,
        run_sequence_bank=True,
        run_field_ping_readout=True,
        run_global_ping_score_spike_prediction=True,
        run_ping_score_spike_prediction=False,
        run_real_probe_score_spike_deflection=True,
        run_overlap_gated_stsp_recruitment=True,
        run_high_stsp_overlap_ablation=True,
        run_score_basin_sparsification=False,
        run_peak_source_attribution=False,
        run_peak_update_history=False,
        run_peak_input_overlap_origin=False,
        run_real_reentry_rollout=False,
        run_real_downstream_metrics=False,
        run_peak_perturbation=False,
        run_legacy_supplement=False,
        run_score_shuffle_null=bool(cfg.run_score_shuffle_null or cfg.run_supplement),
        run_overlap_threshold_sensitivity=bool(cfg.run_overlap_threshold_sensitivity or cfg.run_supplement),
        real_rollout_required_for_main=False,
    )


def _delegate_to_fig6_submodule(module_name: str, function_name: str):
    def _delegated(*args, **kwargs):
        module = __import__(
            f"src.experiments.paper_figures.fig6.subexperiments.{module_name}",
            fromlist=[function_name],
        )
        return getattr(module, function_name)(*args, **kwargs)

    _delegated.__name__ = function_name
    _delegated.__qualname__ = function_name
    return _delegated


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

globals().update({name: _delegate_to_fig6_submodule(module_name, name) for name, module_name in _FIG6_DELEGATES.items()})


def _write_run_log(ctx: ExperimentContext) -> None:
    write_run_log(ctx, now_text=_now())


def _load_dataset_required(dataset_root: str, split: str):
    return load_mnist_skeleton_dataset(dataset_root, split)


def _save_csv(ctx: ExperimentContext, df: pd.DataFrame, path: Path) -> None:
    save_csv_with_registry(ctx, df, path)


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _copy_csv_if_exists(src: Path, dst: Path, ctx: ExperimentContext) -> bool:
    df = _read_csv_if_exists(src)
    if df is None:
        return False
    _save_csv(ctx, df, dst)
    return True


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    write_json_file(payload, path, json_safe_fn=_json_safe)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _prepare_dirs(seed_dir: Path) -> dict[str, Path]:
    return prepare_seed_dirs(seed_dir, include_root_layout=True)


def _resolve_seed_dir(output_root: Path, network_seed: int) -> Path:
    return resolve_seed_dir(output_root, network_seed)


def _rel(path: Path, root: Path) -> str:
    return relative_to_root(path, root)


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Fig.6 peak-amplified overlap re-entry experiment.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-sequence-bank", action="store_true")
    parser.add_argument("--run-peak-source-attribution", action="store_true")
    parser.add_argument("--run-peak-update-history", action="store_true")
    parser.add_argument("--run-peak-input-overlap-origin", action="store_true")
    parser.add_argument("--run-real-reentry-rollout", action="store_true")
    parser.add_argument("--run-real-downstream-metrics", action="store_true")
    parser.add_argument("--run-peak-enrichment", action="store_true")
    parser.add_argument("--run-update-recency-model", action="store_true")
    parser.add_argument("--run-peak-weighted-overlap", action="store_true")
    parser.add_argument("--run-reentry-prediction", action="store_true")
    parser.add_argument("--run-downstream-prediction", action="store_true")
    parser.add_argument("--run-peak-perturbation", action="store_true")
    parser.add_argument("--run-field-ping-readout", action="store_true")
    parser.add_argument("--run-global-ping-score-spike-prediction", action="store_true")
    parser.add_argument("--run-ping-score-spike-prediction", action="store_true")
    parser.add_argument("--run-real-probe-score-spike-deflection", action="store_true")
    parser.add_argument("--run-overlap-gated-stsp-recruitment", action="store_true")
    parser.add_argument("--run-high-stsp-overlap-ablation", action="store_true")
    parser.add_argument("--run-score-basin-sparsification", action="store_true")
    parser.add_argument("--run-fig6-downstream-exploratory", action="store_true")
    parser.add_argument("--run-supplement", action="store_true")
    parser.add_argument("--run-legacy-supplement", action="store_true")
    parser.add_argument("--run-score-shuffle-null", action="store_true")
    parser.add_argument("--run-overlap-threshold-sensitivity", action="store_true")
    parser.add_argument("--force-main-outputs", dest="force_main_outputs", action="store_true", default=True)
    parser.add_argument("--no-force-main-outputs", dest="force_main_outputs", action="store_false")
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sequence-lengths", default="10")
    parser.add_argument("--primary-sequence-length", type=int, default=7)
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay-ms", type=int, default=200)
    parser.add_argument("--ping-ms", type=int, default=30)
    parser.add_argument("--ping-amp", type=float, default=1.0)
    parser.add_argument("--global-ping-ms", type=int, default=30)
    parser.add_argument("--global-ping-amp", type=float, default=1.0)
    parser.add_argument("--probe-ms", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--num-probe-candidates-per-sequence", type=int, default=8)
    parser.add_argument("--peak-q", type=float, default=0.20)
    parser.add_argument("--recent-window", type=int, default=2)
    parser.add_argument("--multi-update-threshold", type=int, default=2)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--n-matched-groups", type=int, default=100)
    parser.add_argument("--foreground-threshold", type=float, default=0.1)
    parser.add_argument("--functional-restore-mode", choices=["full_boundary", "stsp_only"], default="full_boundary")
    parser.add_argument("--recent-overlap-windows", default="2,3,4,5")
    parser.add_argument("--score-eps", type=float, default=1e-6)
    parser.add_argument("--score-early-windows-ms", default="5,10,15,20")
    parser.add_argument("--primary-score-early-window-ms", type=int, default=10)
    parser.add_argument("--score-n-bins", type=int, default=5)
    parser.add_argument("--basin-radius", type=int, default=2)
    parser.add_argument("--basin-top-q", type=float, default=0.20)
    parser.add_argument("--stsp-group-quantile", type=float, default=0.20)
    parser.add_argument("--overlap-threshold", type=float, default=0.05)
    parser.add_argument("--gain-ratio-clip-quantiles", default="0.01,0.99")
    parser.add_argument("--real-probe-entry-mode", default="foreground", choices=["foreground", "encoded_spike"])
    parser.add_argument("--score-use-log-gain", action="store_true")
    parser.add_argument("--leave-one-out-mode", default="blank_same_timing", choices=["blank_same_timing"])
    parser.add_argument("--real-rollout-required-for-main", dest="real_rollout_required_for_main", action="store_true", default=True)
    parser.add_argument("--allow-proxy-main", dest="real_rollout_required_for_main", action="store_false")
    parser.add_argument("--save-full-traces", action="store_true")
    parser.add_argument("--no-save-l3-trace", action="store_true")
    parser.add_argument("--save-spike-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-encode-cache", action="store_true")
    parser.add_argument("--enable-probe-batch", action="store_true")
    parser.add_argument("--enable-high-stsp-ablation-batch", action="store_true")
    parser.add_argument("--enable-sequence-bank-batch", action="store_true")
    parser.add_argument("--enable-leave-one-out-batch", action="store_true")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> Fig6Config:
    smoke = bool(args.smoke)
    run_all = bool(args.run_all)
    seq_lengths = tuple(int(v) for v in str(args.sequence_lengths).split(",") if str(v).strip())
    recent_windows = tuple(int(v) for v in str(args.recent_overlap_windows).split(",") if str(v).strip())
    score_windows = tuple(int(v) for v in str(args.score_early_windows_ms).split(",") if str(v).strip())
    clip_quantiles = tuple(float(v) for v in str(args.gain_ratio_clip_quantiles).split(",") if str(v).strip())
    return Fig6Config(
        model_path=str(args.model_path),
        dataset_root=str(args.dataset_root),
        output_root=str(args.output_root),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        sequence_lengths=seq_lengths,
        primary_sequence_length=int(args.primary_sequence_length),
        sample_ms=int(args.sample_ms),
        delay_ms=int(args.delay_ms),
        ping_ms=int(args.ping_ms),
        ping_amp=float(args.ping_amp),
        global_ping_ms=int(args.global_ping_ms),
        global_ping_amp=float(args.global_ping_amp),
        probe_ms=int(args.probe_ms),
        batch_size=min(int(args.batch_size), 2) if smoke else int(args.batch_size),
        num_sequences=4 if smoke else int(args.num_sequences),
        num_probe_candidates_per_sequence=2 if smoke else int(args.num_probe_candidates_per_sequence),
        peak_q=float(args.peak_q),
        recent_window=int(args.recent_window),
        multi_update_threshold=int(args.multi_update_threshold),
        n_null=8 if smoke else int(args.n_null),
        n_matched_groups=4 if smoke else int(args.n_matched_groups),
        foreground_threshold=float(args.foreground_threshold),
        functional_restore_mode=str(args.functional_restore_mode),
        save_full_traces=bool(args.save_full_traces),
        save_l3_trace=not bool(args.no_save_l3_trace),
        save_spike_cache=bool(args.save_spike_cache),
        run_sequence_bank=run_all or bool(args.run_sequence_bank),
        run_peak_source_attribution=run_all or bool(args.run_peak_source_attribution),
        run_peak_update_history=run_all or bool(args.run_peak_update_history),
        run_peak_input_overlap_origin=run_all or bool(args.run_peak_input_overlap_origin),
        run_real_reentry_rollout=run_all or bool(args.run_real_reentry_rollout),
        run_real_downstream_metrics=run_all or bool(args.run_real_downstream_metrics),
        run_peak_enrichment=run_all or bool(args.run_peak_enrichment),
        run_update_recency_model=run_all or bool(args.run_update_recency_model),
        run_peak_weighted_overlap=run_all or bool(args.run_peak_weighted_overlap),
        run_reentry_prediction=run_all or bool(args.run_reentry_prediction),
        run_downstream_prediction=run_all or bool(args.run_downstream_prediction),
        run_peak_perturbation=run_all or bool(args.run_peak_perturbation),
        run_field_ping_readout=run_all or bool(args.run_field_ping_readout),
        run_global_ping_score_spike_prediction=run_all or bool(args.run_global_ping_score_spike_prediction) or bool(args.run_ping_score_spike_prediction),
        run_ping_score_spike_prediction=False,
        run_real_probe_score_spike_deflection=run_all or bool(args.run_real_probe_score_spike_deflection),
        run_overlap_gated_stsp_recruitment=run_all or bool(args.run_overlap_gated_stsp_recruitment),
        run_high_stsp_overlap_ablation=bool(args.run_high_stsp_overlap_ablation),
        run_score_basin_sparsification=bool(args.run_score_basin_sparsification),
        run_fig6_downstream_exploratory=bool(args.run_fig6_downstream_exploratory),
        run_supplement=run_all or bool(args.run_supplement),
        run_legacy_supplement=bool(args.run_legacy_supplement),
        run_score_shuffle_null=run_all or bool(args.run_supplement) or bool(args.run_score_shuffle_null),
        run_overlap_threshold_sensitivity=run_all or bool(args.run_supplement) or bool(args.run_overlap_threshold_sensitivity),
        force_main_outputs=bool(args.force_main_outputs),
        score_eps=float(args.score_eps),
        score_early_windows_ms=score_windows,
        primary_score_early_window_ms=int(args.primary_score_early_window_ms),
        score_n_bins=int(args.score_n_bins),
        basin_radius=int(args.basin_radius),
        basin_top_q=float(args.basin_top_q),
        stsp_group_quantile=float(args.stsp_group_quantile),
        overlap_threshold=float(args.overlap_threshold),
        gain_ratio_clip_quantiles=clip_quantiles if len(clip_quantiles) == 2 else (0.01, 0.99),
        real_probe_entry_mode=str(args.real_probe_entry_mode),
        score_use_log_gain=bool(args.score_use_log_gain),
        recent_overlap_windows=recent_windows,
        leave_one_out_mode=str(args.leave_one_out_mode),
        real_rollout_required_for_main=bool(args.real_rollout_required_for_main),
        save_debug_figures=bool(args.save_debug_figures),
        show_progress=not bool(args.no_progress),
        use_encode_cache=not bool(args.no_encode_cache),
        enable_probe_batch=bool(args.enable_probe_batch),
        enable_high_stsp_ablation_batch=bool(args.enable_high_stsp_ablation_batch),
        enable_sequence_bank_batch=bool(args.enable_sequence_bank_batch),
        enable_leave_one_out_batch=bool(args.enable_leave_one_out_batch),
        smoke=smoke,
    )


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
PROBE_TRIAL_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_image_id", "probe_label", "probe_source", "raw_overlap", "peak_weighted_overlap", "peak_overlap_fraction", "nonpeak_overlap_fraction", "visual_similarity", "input_energy", "class_pair", "candidate_seed", "peak_support_sum", "nonpeak_support_sum"]
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
PANEL_D_TRIAL_DEFINITION_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_image_id", "probe_label", "probe_source", "raw_overlap", "peak_weighted_overlap", "peak_overlap_fraction", "nonpeak_overlap_fraction", "visual_similarity", "input_energy", "peak_support_sum", "nonpeak_support_sum", "class_pair", "candidate_seed"]
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


if __name__ == "__main__":
    raise SystemExit(main())
