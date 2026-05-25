from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.core.network import SDNN_Network
from src.data.encoding import DoGSpikeEncoder
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import reset_all_state_restore_selected_stsp_in_place, snapshot_boundary_state
from src.experiments.common.ping_common import LAYER_KEYS, prepare_network_state, snapshot_ux_state
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.paper_figures.common.bundle_io import (
    prepare_seed_dirs,
    relative_to_root,
    resolve_seed_dir,
    save_csv_with_registry,
    write_artifact_manifest,
    write_json_file,
    write_run_log,
)
from src.experiments.paper_figures.common.progress import ProgressTracker, planned_phases
from src.experiments.paper_figures.fig2.types import ExperimentContext, Fig2Config, FunctionalReadout, PairEpisodeStateBank
from src.plotting.common.io import apply_publication_style, save_figure_all_formats

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _progress(iterable, *, total=None, desc: str = "", enabled: bool = True):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


FIGURE_ID = "fig2_pair_fused_stsp_state"
NUM_CLASSES = 10
STATE_CONDITIONS = ("S0", "S_A", "S_B", "S_AB")
STATE_VARIABLES = ("g", "u", "x", "ux_concat")
MIXTURE_MODELS = ("A_only", "B_only", "mean_AB", "sum_AB", "unconstrained_AB", "convex_AB")
SINGLE_NETWORK_MODE = "single_network"
RESIDUAL_TEMPLATE_DEFINITION = "residual_true=y_AB-yhat_unconstrained; true_template=y_AB-0.5*(x_A+x_B); shuffled_template=y_AB-0.5*(x_A+x_B_j), j!=i"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _config_from_args(args)
    run(cfg)
    return 0


def run(cfg: Fig2Config) -> dict[str, Any]:
    seed_everything(int(cfg.network_seed))
    seed_dir = _resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = _prepare_dirs(seed_dir)
    device = resolve_device(cfg.device)
    dataset = load_mnist_skeleton_dataset(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, NUM_CLASSES)
    max_duration = max(cfg.sample_ms, cfg.second_item_ms, cfg.weak_probe_ms, 100)
    warnings: list[str] = []
    if Path(cfg.model_path).exists():
        net, encoder = load_model_and_encoder(cfg.model_path, device=device, dt=cfg.dt, max_duration_ms=max_duration)
    elif cfg.smoke:
        seed_everything(int(cfg.network_seed))
        net = SDNN_Network(device=str(device)).to(device)
        net.eval()
        encoder = DoGSpikeEncoder(dt=cfg.dt, max_duration=max_duration * ms, device=str(device))
        warnings.append(
            "Model checkpoint missing; smoke mode used an untrained repo SDNN_Network instance. "
            "Functional E/F outputs are still real network rollouts, but are not manuscript evidence."
        )
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {cfg.model_path}")
    ctx = ExperimentContext(
        cfg=cfg,
        seed_dir=seed_dir,
        config_dir=dirs["config"],
        trial_specs_dir=dirs["trial_specs"],
        raw_dir=dirs["raw"],
        metrics_dir=dirs["metrics"],
        debug_dir=dirs["debug"],
        device=device,
        dataset=dataset,
        class_index=class_index,
        net=net,
        encoder=encoder,
        warnings=warnings,
        output_files={},
        completed_modules={},
        run_log=[f"{_now()} start {FIGURE_ID} seed={cfg.network_seed} smoke={cfg.smoke}"],
    )
    run_info = build_run_info(
        experiment_name=FIGURE_ID,
        output_dir=seed_dir,
        entry_script="src.experiments.paper_figures.fig2_pair_fused_stsp_state_experiment",
        seed=cfg.network_seed,
        dataset=f"MNIST:{cfg.split}",
        command=" ".join(sys.argv),
        model_path=cfg.model_path,
        status="running",
    )
    write_run_info(seed_dir / "meta", run_info)

    try:
        bank: PairEpisodeStateBank | None = None
        needs_bank = any(
            (
                cfg.run_state_bank,
                cfg.run_morphology,
                cfg.run_linear_mixture,
                cfg.run_neutral_ping,
                cfg.run_partial_cue,
                cfg.run_supplement,
                cfg.run_ping_sweep,
            )
        )
        progress = ProgressTracker(
            ctx,
            planned_phases(
                (
                    ("config", True),
                    ("trial_specs", True),
                    ("state_bank", needs_bank),
                    ("morphology", cfg.run_morphology),
                    ("linear_mixture", cfg.run_linear_mixture),
                    ("neutral_ping", cfg.run_neutral_ping),
                    ("ping_sweep", cfg.run_ping_sweep),
                    ("partial_cue", cfg.run_partial_cue),
                    ("completion_delay_sweep", cfg.run_completion_delay_sweep),
                    ("proxy_functional_debug", cfg.save_proxy_functional_debug),
                    ("supplement", cfg.run_supplement),
                    ("debug_figures", cfg.save_debug_figures),
                    ("summary", True),
                )
            ),
            fig_id="fig2",
        )
        with progress.phase("config"):
            _write_config_files(ctx)
        with progress.phase("trial_specs"):
            pair_trials = build_pair_trial_specs(ctx)
        if needs_bank:
            with progress.phase("state_bank"):
                bank = run_pair_episode_state_bank(ctx, pair_trials)
        if bank is not None and cfg.run_morphology:
            with progress.phase("morphology"):
                compute_dual_retention_metrics(ctx, bank)
                compute_pair_specificity_metrics(ctx, bank)
                compute_pair_level_organization_metrics(ctx, bank)
        if bank is not None and cfg.run_linear_mixture:
            with progress.phase("linear_mixture"):
                compute_linear_mixture_metrics(ctx, bank)
                compute_linear_residual_pair_specificity(ctx, bank)
        if bank is not None and cfg.run_neutral_ping:
            with progress.phase("neutral_ping"):
                run_neutral_ping_real_rollout_from_state_bank(ctx, bank)
        if bank is not None and cfg.run_ping_sweep:
            with progress.phase("ping_sweep"):
                run_neutral_ping_parameter_sweep(ctx, bank)
        if bank is not None and cfg.run_partial_cue:
            with progress.phase("partial_cue"):
                run_partial_cue_real_rollout_from_state_bank(ctx, bank)
        if cfg.run_completion_delay_sweep:
            with progress.phase("completion_delay_sweep"):
                run_completion_delay_sweep_from_pair_trials(ctx, pair_trials)
        if bank is not None and cfg.save_proxy_functional_debug:
            with progress.phase("proxy_functional_debug"):
                write_functional_proxy_diagnostics(ctx, bank)
        if bank is not None and cfg.run_supplement:
            with progress.phase("supplement"):
                compute_supplementary_metrics(ctx, bank)
        if cfg.save_debug_figures:
            with progress.phase("debug_figures"):
                save_debug_figures(ctx)
        with progress.phase("summary"):
            summary = _write_summary(ctx)
            _write_run_log(ctx)
        finalize_run_info(seed_dir / "meta", run_info, status="success")
        return summary
    except Exception:
        finalize_run_info(seed_dir / "meta", run_info, status="failed")
        raise


def build_pair_trial_specs(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.trial_specs import build_pair_trial_specs as _impl

    return _impl(*args, **kwargs)


def run_pair_episode_state_bank(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.state_bank import run_pair_episode_state_bank as _impl

    return _impl(*args, **kwargs)


def compute_dual_retention_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.morphology import compute_dual_retention_metrics as _impl

    return _impl(*args, **kwargs)


def compute_pair_specificity_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.morphology import compute_pair_specificity_metrics as _impl

    return _impl(*args, **kwargs)


def compute_pair_level_organization_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.morphology import compute_pair_level_organization_metrics as _impl

    return _impl(*args, **kwargs)


def compute_linear_mixture_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.linear_mixture import compute_linear_mixture_metrics as _impl

    return _impl(*args, **kwargs)


def compute_linear_residual_pair_specificity(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.linear_mixture import compute_linear_residual_pair_specificity as _impl

    return _impl(*args, **kwargs)


def run_neutral_ping_real_rollout_from_state_bank(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.neutral_ping import run_neutral_ping_real_rollout_from_state_bank as _impl

    return _impl(*args, **kwargs)


def run_neutral_ping_parameter_sweep(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.ping_sweep import run_neutral_ping_parameter_sweep as _impl

    return _impl(*args, **kwargs)


def run_partial_cue_real_rollout_from_state_bank(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.partial_cue import run_partial_cue_real_rollout_from_state_bank as _impl

    return _impl(*args, **kwargs)


def run_completion_delay_sweep_from_pair_trials(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.completion_delay_sweep import run_completion_delay_sweep_from_pair_trials as _impl

    return _impl(*args, **kwargs)


def run_neutral_ping_accessibility_proxy(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.supplement import run_neutral_ping_accessibility_proxy as _impl

    return _impl(*args, **kwargs)


def run_partial_cue_accessibility_proxy(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.supplement import run_partial_cue_accessibility_proxy as _impl

    return _impl(*args, **kwargs)


def _make_weak_probe_spikes_encoded_dropout(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _make_weak_probe_spikes_encoded_dropout as _impl

    return _impl(*args, **kwargs)


def _make_weak_probe_spikes_image_foreground(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _make_weak_probe_spikes_image_foreground as _impl

    return _impl(*args, **kwargs)


def _weak_probe_mask_row(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _weak_probe_mask_row as _impl

    return _impl(*args, **kwargs)


def compute_supplementary_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.supplement import compute_supplementary_metrics as _impl

    return _impl(*args, **kwargs)


def _write_layerwise_morphology_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.morphology import _write_layerwise_morphology_metrics as _impl

    return _impl(*args, **kwargs)


def _capture_pair_batch(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _capture_pair_batch as _impl

    return _impl(*args, **kwargs)


def _step_network_once(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _step_network_once as _impl

    return _impl(*args, **kwargs)


def _fit_mixture_models(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _fit_mixture_models as _impl

    return _impl(*args, **kwargs)


def _linear_model_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _linear_model_metrics as _impl

    return _impl(*args, **kwargs)


def _fixed_model_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _fixed_model_metrics as _impl

    return _impl(*args, **kwargs)


def _fit_single(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _fit_single as _impl

    return _impl(*args, **kwargs)


def _fit_single_coeffs(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _fit_single_coeffs as _impl

    return _impl(*args, **kwargs)


def _fit_two(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _fit_two as _impl

    return _impl(*args, **kwargs)


def _fit_two_coeffs(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _fit_two_coeffs as _impl

    return _impl(*args, **kwargs)


def _convex_weight(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _convex_weight as _impl

    return _impl(*args, **kwargs)


def _convex_prediction(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _convex_prediction as _impl

    return _impl(*args, **kwargs)


def _cv_r2(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _cv_r2 as _impl

    return _impl(*args, **kwargs)


def _predict_model(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _predict_model as _impl

    return _impl(*args, **kwargs)


def _r2(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _r2 as _impl

    return _impl(*args, **kwargs)


def _access_scores(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _access_scores as _impl

    return _impl(*args, **kwargs)


def _prediction_from_scores(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _prediction_from_scores as _impl

    return _impl(*args, **kwargs)


def _partial_cue_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _partial_cue_metrics as _impl

    return _impl(*args, **kwargs)


def _ping_sweep_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _ping_sweep_metrics as _impl

    return _impl(*args, **kwargs)


def _completion_delay_sweep_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _completion_delay_sweep_metrics as _impl

    return _impl(*args, **kwargs)


def _completion_delay_sweep_contrast(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _completion_delay_sweep_contrast as _impl

    return _impl(*args, **kwargs)


def _stable_sweep_seed(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _stable_sweep_seed as _impl

    return _impl(*args, **kwargs)


def _partial_cue_auc_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _partial_cue_auc_metrics as _impl

    return _impl(*args, **kwargs)


def _partial_cue_pair_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _partial_cue_pair_metrics as _impl

    return _impl(*args, **kwargs)


def _normalized_auc(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _normalized_auc as _impl

    return _impl(*args, **kwargs)


def _p50_from_curve(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _p50_from_curve as _impl

    return _impl(*args, **kwargs)


def _nan_diff(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _nan_diff as _impl

    return _impl(*args, **kwargs)


def _mode_value(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _mode_value as _impl

    return _impl(*args, **kwargs)


def _compat_fig4_weak_probe_outputs(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _compat_fig4_weak_probe_outputs as _impl

    return _impl(*args, **kwargs)


def _cue_gain(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _cue_gain as _impl

    return _impl(*args, **kwargs)


def run_ping_readout_from_boundary(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import run_ping_readout_from_boundary as _impl

    return _impl(*args, **kwargs)


def run_probe_readout_from_boundary(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import run_probe_readout_from_boundary as _impl

    return _impl(*args, **kwargs)


def restore_condition_state_for_functional_readout(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import restore_condition_state_for_functional_readout as _impl

    return _impl(*args, **kwargs)


def boundary_state_to_restore_ux_by_layer(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import boundary_state_to_restore_ux_by_layer as _impl

    return _impl(*args, **kwargs)


def slice_boundary_state(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import slice_boundary_state as _impl

    return _impl(*args, **kwargs)


def concat_condition_boundaries(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import concat_condition_boundaries as _impl

    return _impl(*args, **kwargs)


def _forward_three_layers_with_optional_trace(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _forward_three_layers_with_optional_trace as _impl

    return _impl(*args, **kwargs)


def _layer_input_shapes_from_boundary(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _layer_input_shapes_from_boundary as _impl

    return _impl(*args, **kwargs)


def _layer_input_shapes_for_batch(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _layer_input_shapes_for_batch as _impl

    return _impl(*args, **kwargs)


def _pack_trace(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _pack_trace as _impl

    return _impl(*args, **kwargs)


def _readout_margin_for_class(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _readout_margin_for_class as _impl

    return _impl(*args, **kwargs)


def _readout_margin_value(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _readout_margin_value as _impl

    return _impl(*args, **kwargs)


def _ping_spike_count(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _ping_spike_count as _impl

    return _impl(*args, **kwargs)


def _ping_energy(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _ping_energy as _impl

    return _impl(*args, **kwargs)


def _neutral_ping_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _neutral_ping_metrics as _impl

    return _impl(*args, **kwargs)


def write_functional_proxy_diagnostics(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.supplement import write_functional_proxy_diagnostics as _impl

    return _impl(*args, **kwargs)


def _metric_lookup(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _metric_lookup as _impl

    return _impl(*args, **kwargs)


def _linear_metric_lookup(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _linear_metric_lookup as _impl

    return _impl(*args, **kwargs)


def save_debug_figures(ctx: ExperimentContext) -> None:
    apply_publication_style()
    jobs = [
        ("panel_b_dual_retention_metrics.csv", "fusion_dual_score", "fig2_debug_dual_retention"),
        ("panel_c_pair_specificity_metrics.csv", "true_minus_shuffled", "fig2_debug_pair_specificity"),
        ("panel_d_pair_level_organization_metrics.csv", "WPRI", "fig2_debug_wpri"),
        ("panel_d_linear_residual_pair_specificity_metrics.csv", "residual_pair_specificity", "fig2_debug_linear_residual"),
        ("panel_e_neutral_ping_metrics.csv", "P_pair", "fig2_debug_real_neutral_ping"),
        ("panel_f_partial_cue_metrics.csv", "P_target", "fig2_debug_real_partial_cue"),
    ]
    for filename, column, stem in jobs:
        path = ctx.metrics_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        part = df[(df.get("layer", "") == "layer3") & (df.get("state_variable", "") == "g")] if "layer" in df.columns else df
        if column not in part.columns or part.empty:
            continue
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=150)
        ax.hist(pd.to_numeric(part[column], errors="coerce").dropna(), bins=20, color="#4C78A8", alpha=0.8)
        ax.set_title(stem)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        save_figure_all_formats(fig, ctx.debug_dir / stem)
        plt.close(fig)
    ping_path = ctx.metrics_dir / "supp_ping_sweep_metrics.csv"
    if ping_path.exists():
        import matplotlib.pyplot as plt

        ping_df = pd.read_csv(ping_path)
        for sweep_type, x_col, stem in (
            ("amplitude", "ping_amp", "supp_ping_amp_sweep_pair_member_readout"),
            ("duration", "ping_ms", "supp_ping_ms_sweep_pair_member_readout"),
        ):
            part = ping_df[ping_df["sweep_type"].astype(str).eq(sweep_type)] if "sweep_type" in ping_df.columns else pd.DataFrame()
            if part.empty or not {x_col, "state_condition", "pair_member_readout_rate"}.issubset(part.columns):
                continue
            fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=150)
            for condition, cond_part in part.groupby("state_condition", sort=True):
                ordered = cond_part.sort_values(x_col)
                ax.plot(ordered[x_col], ordered["pair_member_readout_rate"], marker="o", label=str(condition))
            ax.set_xlabel(x_col)
            ax.set_ylabel("pair_member_readout_rate")
            ax.legend(frameon=False, fontsize=7)
            save_figure_all_formats(fig, ctx.debug_dir / stem)
            plt.close(fig)
    completion_path = ctx.metrics_dir / "supp_completion_delay_sweep_contrast.csv"
    if completion_path.exists():
        import matplotlib.pyplot as plt

        comp_df = pd.read_csv(completion_path)
        if not comp_df.empty and {"delay2_ms", "completion_gain_SAB_minus_SB"}.issubset(comp_df.columns):
            fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=150)
            ordered = comp_df.sort_values("delay2_ms")
            ax.plot(ordered["delay2_ms"], ordered["completion_gain_SAB_minus_SB"], marker="o")
            ax.axhline(0.0, color="0.5", linewidth=0.8)
            ax.set_xlabel("delay2_ms")
            ax.set_ylabel("completion_gain_SAB_minus_SB")
            save_figure_all_formats(fig, ctx.debug_dir / "supp_completion_delay_gain")
            plt.close(fig)
    ctx.completed_modules["debug_figures"] = True


def _pair_sampling_audit(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _pair_sampling_audit as _impl

    return _impl(*args, **kwargs)


def _trial_condition_audit(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _trial_condition_audit as _impl

    return _impl(*args, **kwargs)


def _image_similarity_and_overlap(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _image_similarity_and_overlap as _impl

    return _impl(*args, **kwargs)


def _selection_bin(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _selection_bin as _impl

    return _impl(*args, **kwargs)


def _row_centered_cosine(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _row_centered_cosine as _impl

    return _impl(*args, **kwargs)


def _centered_cosine(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _centered_cosine as _impl

    return _impl(*args, **kwargs)


def _slice_boundary_state(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _slice_boundary_state as _impl

    return _impl(*args, **kwargs)


def _concat_boundary_states(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _concat_boundary_states as _impl

    return _impl(*args, **kwargs)


def _images_for_ids(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _images_for_ids as _impl

    return _impl(*args, **kwargs)


def _encode_cached(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _encode_cached as _impl

    return _impl(*args, **kwargs)


def _iter_batches(*args, **kwargs):
    from src.experiments.paper_figures.fig2.subexperiments.helpers import _iter_batches as _impl

    return _impl(*args, **kwargs)


def _write_config_files(ctx: ExperimentContext) -> None:
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


def _write_summary(ctx: ExperimentContext) -> dict[str, Any]:
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


def _write_run_log(ctx: ExperimentContext) -> None:
    write_run_log(ctx, now_text=_now())


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


def _prepare_dirs(seed_dir: Path) -> dict[str, Path]:
    return prepare_seed_dirs(seed_dir, include_root_layout=True)


def _resolve_seed_dir(output_root: Path, network_seed: int) -> Path:
    return resolve_seed_dir(output_root, network_seed)


def _rel(path: Path, root: Path) -> str:
    return relative_to_root(path, root)


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


def _maybe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _maybe_int(value: Any) -> int | float:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return float("nan")
    return int(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_from_args(args: argparse.Namespace) -> Fig2Config:
    smoke = bool(args.smoke)
    run_all = bool(args.run_all)
    keep_probs = tuple(float(v) for v in str(args.weak_probe_keep_probs).split(",") if str(v).strip())
    ping_amp_sweep = tuple(float(v) for v in str(args.ping_amp_sweep).split(",") if str(v).strip())
    ping_ms_sweep = tuple(int(v) for v in str(args.ping_ms_sweep).split(",") if str(v).strip())
    completion_delay_sweep_ms = tuple(int(v) for v in str(args.completion_delay_sweep_ms).split(",") if str(v).strip())
    completion_delay_repeats = int(args.completion_delay_repeats)
    if smoke:
        keep_probs = (0.2, 0.7)
        ping_amp_sweep = (1.0,)
        ping_ms_sweep = ping_ms_sweep[:2]
        completion_delay_sweep_ms = completion_delay_sweep_ms[:2]
        completion_delay_repeats = 1
    delay_grid = tuple(int(v) for v in str(args.delay_layer_grid).split(",") if str(v).strip())
    return Fig2Config(
        model_path=str(args.model_path),
        dataset_root=str(args.dataset_root),
        output_root=str(args.output_root),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        sample_ms=int(args.sample_ms),
        delay1_ms=int(args.delay1_ms),
        second_item_ms=int(args.second_item_ms),
        delay2_ms=int(args.delay2_ms),
        ping_ms=int(args.ping_ms),
        ping_amp=float(args.ping_amp),
        ping_repeats=1 if smoke else int(args.ping_repeats),
        ping_mode=str(args.ping_mode),
        ping_noise=float(args.ping_noise),
        ping_amp_sweep=ping_amp_sweep,
        ping_ms_sweep=ping_ms_sweep,
        weak_probe_ms=int(args.weak_probe_ms),
        weak_probe_keep_probs=keep_probs,
        weak_probe_repeats=1 if smoke else int(args.weak_probe_repeats),
        weak_probe_mask_space=str(args.weak_probe_mask_space),
        weak_probe_use_same_mask_across_states=bool(args.weak_probe_use_same_mask_across_states),
        weak_probe_scale=float(args.weak_probe_scale),
        weak_probe_noise=float(args.weak_probe_noise),
        weak_probe_metric_mode=str(args.weak_probe_metric_mode),
        foreground_threshold=float(args.foreground_threshold),
        functional_restore_mode=str(args.functional_restore_mode),
        num_pairs=min(int(args.num_pairs), 20) if smoke else int(args.num_pairs),
        batch_size=min(int(args.batch_size), 4) if smoke else int(args.batch_size),
        n_shuffle=4 if smoke else int(args.n_shuffle),
        delay_layer_grid=delay_grid[:2] if smoke else delay_grid,
        linear_mixture_cv_folds=2 if smoke else int(args.linear_mixture_cv_folds),
        run_state_bank=run_all or bool(args.run_state_bank),
        run_morphology=run_all or bool(args.run_morphology),
        run_linear_mixture=run_all or bool(args.run_linear_mixture),
        run_neutral_ping=run_all or bool(args.run_neutral_ping),
        run_partial_cue=run_all or bool(args.run_partial_cue),
        run_supplement=run_all or bool(args.run_supplement),
        run_ping_sweep=run_all or bool(args.run_ping_sweep),
        run_completion_delay_sweep=run_all or bool(args.run_completion_delay_sweep),
        completion_delay_sweep_ms=completion_delay_sweep_ms,
        completion_delay_keep_prob=float(args.completion_delay_keep_prob),
        completion_delay_repeats=completion_delay_repeats,
        save_debug_figures=bool(args.save_debug_figures),
        save_spike_cache=bool(args.save_spike_cache),
        save_all_layer_state_bank=bool(args.save_all_layer_state_bank),
        save_functional_traces=bool(args.save_functional_traces),
        save_proxy_functional_debug=bool(args.save_proxy_functional_debug),
        show_progress=not bool(args.no_progress),
        use_encode_cache=not bool(args.no_encode_cache),
        enable_partial_cue_batch=bool(args.enable_partial_cue_batch),
        functional_readout_batch_size=max(1, int(args.functional_readout_batch_size)),
        smoke=smoke,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Fig.2 pair-fused STSP state experiment.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-state-bank", action="store_true")
    parser.add_argument("--run-morphology", action="store_true")
    parser.add_argument("--run-linear-mixture", action="store_true")
    parser.add_argument("--run-neutral-ping", action="store_true")
    parser.add_argument("--run-partial-cue", action="store_true")
    parser.add_argument("--run-supplement", action="store_true")
    parser.add_argument("--run-ping-sweep", action="store_true")
    parser.add_argument("--run-completion-delay-sweep", action="store_true")
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--save-spike-cache", action="store_true")
    parser.add_argument("--save-all-layer-state-bank", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-encode-cache", action="store_true")
    parser.add_argument("--enable-partial-cue-batch", action="store_true")
    parser.add_argument("--functional-readout-batch-size", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay1-ms", type=int, default=200)
    parser.add_argument("--second-item-ms", type=int, default=200)
    parser.add_argument("--delay2-ms", type=int, default=400)
    parser.add_argument("--ping-ms", type=int, default=30)
    parser.add_argument("--ping-amp", type=float, default=1.0)
    parser.add_argument("--ping-repeats", type=int, default=1)
    parser.add_argument("--ping-mode", default="constant_drive", choices=["constant_drive", "bernoulli_drive"])
    parser.add_argument("--ping-noise", type=float, default=0.0)
    parser.add_argument("--ping-amp-sweep", default="0.5,1.0,1.5")
    parser.add_argument("--ping-ms-sweep", default="10,30,60")
    parser.add_argument("--weak-probe-ms", type=int, default=100)
    parser.add_argument("--weak-probe-keep-probs", default="0.05,0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    parser.add_argument("--weak-probe-repeats", type=int, default=20)
    parser.add_argument("--weak-probe-mask-space", default="encoded_spikes", choices=["encoded_spikes", "image_foreground"])
    parser.add_argument("--weak-probe-use-same-mask-across-states", dest="weak_probe_use_same_mask_across_states", action="store_true", default=True)
    parser.add_argument("--weak-probe-independent-masks-across-states", dest="weak_probe_use_same_mask_across_states", action="store_false")
    parser.add_argument("--weak-probe-scale", type=float, default=0.35)
    parser.add_argument("--weak-probe-noise", type=float, default=0.0)
    parser.add_argument("--weak-probe-metric-mode", default="fig4_compat", choices=["fig4_compat", "legacy"])
    parser.add_argument("--foreground-threshold", type=float, default=0.1)
    parser.add_argument("--functional-restore-mode", choices=["full_boundary", "stsp_only"], default="full_boundary")
    parser.add_argument("--num-pairs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-shuffle", type=int, default=50)
    parser.add_argument("--delay-layer-grid", default="200,400,800")
    parser.add_argument("--linear-mixture-cv-folds", type=int, default=5)
    parser.add_argument("--completion-delay-sweep-ms", default="100,200,300,400,800,1200")
    parser.add_argument("--completion-delay-keep-prob", type=float, default=0.2)
    parser.add_argument("--completion-delay-repeats", type=int, default=5)
    parser.add_argument("--save-functional-traces", action="store_true")
    parser.add_argument("--save-proxy-functional-debug", action="store_true")
    return parser.parse_args(argv)


PANEL_E_RAW_COLUMNS = [
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
]
SUPP_PING_SWEEP_RAW_COLUMNS = [
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
]
WEAK_PROBE_MASK_COLUMNS = [
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
]
SUPP_COMPLETION_DELAY_RAW_COLUMNS = [
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
]
PANEL_F_RAW_COLUMNS = [
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
]


if __name__ == "__main__":
    raise SystemExit(main())
