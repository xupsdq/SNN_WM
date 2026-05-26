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
from src.experiments.paper_figures.fig3.types import ExperimentContext, Fig3Config, MultiItemSequenceLandscapeBank
from src.plotting.common.io import apply_publication_style, save_figure_all_formats

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _progress(iterable, *, total=None, desc: str = "", enabled: bool = True):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


FIGURE_ID = "fig3_multiitem_peak_landscape"
NUM_CLASSES = 10
PRIMARY_LAYER = "layer1"
PRIMARY_STATE_VARIABLE = "g"
STATE_VARIABLES = ("g", "u", "x")
CUE_CONDITIONS = ("peak", "valley", "random")
MEMORY_CONDITIONS = ("cue_only", "single_item_memory", "sequence_state")
SINGLE_NETWORK_MODE = "single_network"
FIG3_DESIGN_VERSION = "multiitem_peak_landscape_structured_readable_stsp"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _config_from_args(args)
    run(cfg)
    return 0


def run(cfg: Fig3Config) -> dict[str, Any]:
    seed_everything(int(cfg.network_seed))
    seed_dir = _resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = _prepare_dirs(seed_dir)
    device = resolve_device(cfg.device)
    dataset = load_mnist_skeleton_dataset(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, NUM_CLASSES)
    warnings_list: list[str] = []
    if Path(cfg.model_path).exists():
        net, encoder = load_model_and_encoder(cfg.model_path, device=device, dt=cfg.dt, max_duration_ms=max(cfg.sample_ms, cfg.weak_probe_ms, 100))
    elif cfg.smoke:
        seed_everything(int(cfg.network_seed))
        net = SDNN_Network(device=str(device)).to(device)
        net.eval()
        encoder = DoGSpikeEncoder(dt=cfg.dt, max_duration=max(cfg.sample_ms, cfg.weak_probe_ms, 100) * ms, device=str(device))
        warnings_list.append(
            "Model checkpoint missing; smoke mode used an untrained repo SDNN_Network instance. "
            "Functional D/F outputs are real rollouts but are not manuscript evidence."
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
        warnings=warnings_list,
        output_files={},
        completed_modules={},
        run_log=[f"{_now()} start {FIGURE_ID} seed={cfg.network_seed} smoke={cfg.smoke}"],
    )
    run_info = build_run_info(
        experiment_name=FIGURE_ID,
        output_dir=seed_dir,
        entry_script="src.experiments.paper_figures.fig3_multiitem_peak_landscape_experiment",
        seed=cfg.network_seed,
        dataset=f"MNIST:{cfg.split}",
        command=" ".join(sys.argv),
        model_path=cfg.model_path,
        status="running",
    )
    write_run_info(seed_dir / "meta", run_info)
    try:
        bank: MultiItemSequenceLandscapeBank | None = None
        needs_bank = any(
            (
                cfg.run_state_bank,
                cfg.run_progressive_update,
                cfg.run_peak_valley_landscape,
                cfg.run_neutral_ping,
                cfg.run_weak_probe,
                cfg.run_region_ping,
                cfg.run_region_ping_amp_sweep,
                cfg.run_peak_cue_main,
                cfg.run_structural_weak_cue_supplement,
                cfg.run_population_morphology_supplement,
                cfg.run_supplement,
            )
        )
        progress = ProgressTracker(
            ctx,
            planned_phases(
                (
                    ("config", True),
                    ("trial_specs", True),
                    ("state_bank", needs_bank),
                    ("progressive_update", cfg.run_progressive_update),
                    ("peak_valley_landscape", cfg.run_peak_valley_landscape or cfg.run_population_morphology_supplement or cfg.run_supplement),
                    ("neutral_ping", cfg.run_neutral_ping),
                    ("weak_probe", cfg.run_weak_probe),
                    ("region_ping", cfg.run_region_ping),
                    ("peak_cue_main", cfg.run_peak_cue_main),
                    ("structural_weak_cue_supplement", cfg.run_structural_weak_cue_supplement),
                    ("supplement", cfg.run_supplement),
                    ("debug_figures", cfg.save_debug_figures),
                    ("summary", True),
                )
            ),
            fig_id="fig3",
        )
        with progress.phase("config"):
            _write_config_files(ctx)
        with progress.phase("trial_specs"):
            seq_trials, singleton_trials, partial_trials = build_sequence_trial_specs(ctx)
        if needs_bank:
            with progress.phase("state_bank"):
                bank = run_multiitem_sequence_state_bank(ctx, seq_trials)
        if bank is not None and cfg.run_progressive_update:
            with progress.phase("progressive_update"):
                compute_progressive_update_metrics(ctx, bank)
        if bank is not None and (cfg.run_peak_valley_landscape or cfg.run_population_morphology_supplement or cfg.run_supplement):
            with progress.phase("peak_valley_landscape"):
                compute_final_support_landscape(ctx, bank)
        if bank is not None and cfg.run_neutral_ping:
            with progress.phase("neutral_ping"):
                run_neutral_ping_readout_distribution(ctx, bank)
        if bank is not None and cfg.run_weak_probe:
            with progress.phase("weak_probe"):
                run_sequence_weak_probe_real_rollout_from_state_bank(ctx, bank)
        if bank is not None and cfg.run_region_ping:
            with progress.phase("region_ping"):
                run_region_gated_ping_readout(ctx, bank)
        if bank is not None and cfg.run_peak_cue_main:
            with progress.phase("peak_cue_main"):
                run_peak_cue_main_from_state_bank(ctx, bank)
        if bank is not None and cfg.run_structural_weak_cue_supplement:
            with progress.phase("structural_weak_cue_supplement"):
                ensure_structural_weak_cue_outputs(ctx, bank)
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


def build_sequence_trial_specs(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.trial_specs import build_sequence_trial_specs as _impl

    return _impl(*args, **kwargs)


def run_multiitem_sequence_state_bank(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.state_bank import run_multiitem_sequence_state_bank as _impl

    return _impl(*args, **kwargs)


def compute_progressive_update_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.progressive_update import compute_progressive_update_metrics as _impl

    return _impl(*args, **kwargs)


def compute_final_support_landscape(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.peak_valley_landscape import compute_final_support_landscape as _impl

    return _impl(*args, **kwargs)


def run_neutral_ping_from_final_state(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.neutral_ping import run_neutral_ping_from_final_state as _impl

    return _impl(*args, **kwargs)


def run_neutral_ping_readout_distribution(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.neutral_ping import run_neutral_ping_readout_distribution as _impl

    return _impl(*args, **kwargs)


def _serial_bin_for_ping_row(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.neutral_ping import _serial_bin_for_ping_row as _impl

    return _impl(*args, **kwargs)


def run_region_gated_ping_readout(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.region_ping import run_region_gated_ping_readout as _impl

    return _impl(*args, **kwargs)


def _region_gated_ping_trial_rows(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.region_ping import _region_gated_ping_trial_rows as _impl

    return _impl(*args, **kwargs)


def _make_region_ping_masks(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.region_ping import _make_region_ping_masks as _impl

    return _impl(*args, **kwargs)


def _run_masked_ping_from_boundary(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.region_ping import _run_masked_ping_from_boundary as _impl

    return _impl(*args, **kwargs)


def run_sequence_weak_probe_real_rollout_from_state_bank(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.weak_probe import run_sequence_weak_probe_real_rollout_from_state_bank as _impl

    return _impl(*args, **kwargs)


def _make_weak_probe_spikes_encoded_dropout(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.weak_probe import _make_weak_probe_spikes_encoded_dropout as _impl

    return _impl(*args, **kwargs)


def slice_boundary_state(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import slice_boundary_state as _impl

    return _impl(*args, **kwargs)


def concat_sequence_condition_boundaries(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import concat_sequence_condition_boundaries as _impl

    return _impl(*args, **kwargs)


def concat_named_boundaries(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import concat_named_boundaries as _impl

    return _impl(*args, **kwargs)


def _weak_probe_memory_specs_for_target(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _weak_probe_memory_specs_for_target as _impl

    return _impl(*args, **kwargs)


def run_probe_readout_from_boundary(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import run_probe_readout_from_boundary as _impl

    return _impl(*args, **kwargs)


def _fig3f_memory_states(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _fig3f_memory_states as _impl

    return _impl(*args, **kwargs)


def _memory_condition_label(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _memory_condition_label as _impl

    return _impl(*args, **kwargs)


def _weak_probe_target_sources(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _weak_probe_target_sources as _impl

    return _impl(*args, **kwargs)


def run_structural_weak_cue_classification_supplement(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import run_structural_weak_cue_classification_supplement as _impl

    return _impl(*args, **kwargs)


def ensure_structural_weak_cue_outputs(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import ensure_structural_weak_cue_outputs as _impl

    return _impl(*args, **kwargs)


def run_peak_cue_main_from_state_bank(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.peak_cue_main import run_peak_cue_main_from_state_bank as _impl

    return _impl(*args, **kwargs)


def _filter_peak_cue_main_raw(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.peak_cue_main import _filter_peak_cue_main_raw as _impl

    return _impl(*args, **kwargs)


def _peak_cue_accuracy(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.peak_cue_main import _peak_cue_accuracy as _impl

    return _impl(*args, **kwargs)


def _peak_cue_memory_gain(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.peak_cue_main import _peak_cue_memory_gain as _impl

    return _impl(*args, **kwargs)


def _peak_cue_matching_diagnostics(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.peak_cue_main import _peak_cue_matching_diagnostics as _impl

    return _impl(*args, **kwargs)


def run_structural_weak_cue_classification(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import run_structural_weak_cue_classification as _impl

    return _impl(*args, **kwargs)


def run_peak_aligned_completion(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.peak_aligned_completion import run_peak_aligned_completion as _impl

    return _impl(*args, **kwargs)


def compute_supplementary_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.supplement import compute_supplementary_metrics as _impl

    return _impl(*args, **kwargs)


def compute_anchor_dynamics_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.supplement import compute_anchor_dynamics_metrics as _impl

    return _impl(*args, **kwargs)


def compute_weak_probe_target_source_control(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.supplement import compute_weak_probe_target_source_control as _impl

    return _impl(*args, **kwargs)


def compute_peak_cue_serial_position_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.peak_cue_main import compute_peak_cue_serial_position_metrics as _impl

    return _impl(*args, **kwargs)


def _target_position_bin(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.peak_cue_main import _target_position_bin as _impl

    return _impl(*args, **kwargs)


def _capture_sequence(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _capture_sequence as _impl

    return _impl(*args, **kwargs)


def _capture_singleton_refs(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _capture_singleton_refs as _impl

    return _impl(*args, **kwargs)


def _capture_singleton_refs_and_boundaries(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _capture_singleton_refs_and_boundaries as _impl

    return _impl(*args, **kwargs)


def _snapshot_arrays(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _snapshot_arrays as _impl

    return _impl(*args, **kwargs)


def _landscape_for_sequence(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _landscape_for_sequence as _impl

    return _impl(*args, **kwargs)


def _save_example_landscape(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _save_example_landscape as _impl

    return _impl(*args, **kwargs)


def _example_landscape_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _example_landscape_summary as _impl

    return _impl(*args, **kwargs)


def boundary_state_to_restore_ux_by_layer(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import boundary_state_to_restore_ux_by_layer as _impl

    return _impl(*args, **kwargs)


def _layer_input_shapes_from_boundary(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _layer_input_shapes_from_boundary as _impl

    return _impl(*args, **kwargs)


def _layer_input_shapes_for_batch(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _layer_input_shapes_for_batch as _impl

    return _impl(*args, **kwargs)


def restore_condition_state_for_functional_readout(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import restore_condition_state_for_functional_readout as _impl

    return _impl(*args, **kwargs)


def _run_ping_from_boundary(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _run_ping_from_boundary as _impl

    return _impl(*args, **kwargs)


def _run_weak_cue_spikes_from_boundary(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _run_weak_cue_spikes_from_boundary as _impl

    return _impl(*args, **kwargs)


def _run_weak_cue_from_boundary(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _run_weak_cue_from_boundary as _impl

    return _impl(*args, **kwargs)


def _run_weak_cue_multi_boundary_batch(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _run_weak_cue_multi_boundary_batch as _impl

    return _impl(*args, **kwargs)


def _step_network_once(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _step_network_once as _impl

    return _impl(*args, **kwargs)


def _restore_boundary_state(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _restore_boundary_state as _impl

    return _impl(*args, **kwargs)


def _region_ping_serial_bins(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _region_ping_serial_bins as _impl

    return _impl(*args, **kwargs)


def _region_ping_position_distribution(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _region_ping_position_distribution as _impl

    return _impl(*args, **kwargs)


def _region_ping_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _region_ping_summary as _impl

    return _impl(*args, **kwargs)


def _region_ping_contrast(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _region_ping_contrast as _impl

    return _impl(*args, **kwargs)


def _region_ping_current_matching(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _region_ping_current_matching as _impl

    return _impl(*args, **kwargs)


def _region_ping_current_matching_status(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _region_ping_current_matching_status as _impl

    return _impl(*args, **kwargs)


def _region_ping_amp_sweep_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _region_ping_amp_sweep_summary as _impl

    return _impl(*args, **kwargs)


def _region_ping_amp_sweep_latency(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _region_ping_amp_sweep_latency as _impl

    return _impl(*args, **kwargs)


def _serial_distribution(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _serial_distribution as _impl

    return _impl(*args, **kwargs)


def _js_divergence(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _js_divergence as _impl

    return _impl(*args, **kwargs)


def _kl_divergence(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _kl_divergence as _impl

    return _impl(*args, **kwargs)


def _tv_distance(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _tv_distance as _impl

    return _impl(*args, **kwargs)


def compute_fig3e_weak_probe_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.weak_probe import compute_fig3e_weak_probe_metrics as _impl

    return _impl(*args, **kwargs)


def compute_fig3e_weak_probe_auc_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.weak_probe import compute_fig3e_weak_probe_auc_metrics as _impl

    return _impl(*args, **kwargs)


def compute_fig3e_weak_probe_memory_gain(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.weak_probe import compute_fig3e_weak_probe_memory_gain as _impl

    return _impl(*args, **kwargs)


def compute_fig3e_weak_probe_position_stratified_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.weak_probe import compute_fig3e_weak_probe_position_stratified_metrics as _impl

    return _impl(*args, **kwargs)


def compute_fig3f_weak_probe_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.weak_probe import compute_fig3f_weak_probe_metrics as _impl

    return _impl(*args, **kwargs)


def compute_fig3f_weak_probe_auc_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.weak_probe import compute_fig3f_weak_probe_auc_metrics as _impl

    return _impl(*args, **kwargs)


def compute_fig3f_weak_probe_memory_gain(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.weak_probe import compute_fig3f_weak_probe_memory_gain as _impl

    return _impl(*args, **kwargs)


def _normalized_auc(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _normalized_auc as _impl

    return _impl(*args, **kwargs)


def _p50_from_curve(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _p50_from_curve as _impl

    return _impl(*args, **kwargs)


def _nan_diff(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _nan_diff as _impl

    return _impl(*args, **kwargs)


def _mode_value(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _mode_value as _impl

    return _impl(*args, **kwargs)


def _fig3f_cue_gain(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.peak_cue_main import _fig3f_cue_gain as _impl

    return _impl(*args, **kwargs)


def _completion_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.peak_aligned_completion import _completion_metrics as _impl

    return _impl(*args, **kwargs)


def _main_sequence_meta(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _main_sequence_meta as _impl

    return _impl(*args, **kwargs)


def _weak_cue_target_sources(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _weak_cue_target_sources as _impl

    return _impl(*args, **kwargs)


def _sample_weak_cue_target(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _sample_weak_cue_target as _impl

    return _impl(*args, **kwargs)


def _support_map_for_structural_cue(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _support_map_for_structural_cue as _impl

    return _impl(*args, **kwargs)


def build_ranked_foreground_masks(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import build_ranked_foreground_masks as _impl

    return _impl(*args, **kwargs)


def _mask_from_flat_indices(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _mask_from_flat_indices as _impl

    return _impl(*args, **kwargs)


def _selected_quantile_mean(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _selected_quantile_mean as _impl

    return _impl(*args, **kwargs)


def _structural_accuracy(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _structural_accuracy as _impl

    return _impl(*args, **kwargs)


def _structural_memory_gain(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _structural_memory_gain as _impl

    return _impl(*args, **kwargs)


def _structural_target_source_control(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _structural_target_source_control as _impl

    return _impl(*args, **kwargs)


def _structural_matching_diagnostics(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _structural_matching_diagnostics as _impl

    return _impl(*args, **kwargs)


def _first_float(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _first_float as _impl

    return _impl(*args, **kwargs)


def _mean_numeric(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _mean_numeric as _impl

    return _impl(*args, **kwargs)


def _row_float(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _row_float as _impl

    return _impl(*args, **kwargs)


def _missing_csv_columns(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _missing_csv_columns as _impl

    return _impl(*args, **kwargs)


def _read_csv_if_exists(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _read_csv_if_exists as _impl

    return _impl(*args, **kwargs)


def _structural_trial_columns(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _structural_trial_columns as _impl

    return _impl(*args, **kwargs)


def _structural_mask_columns(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _structural_mask_columns as _impl

    return _impl(*args, **kwargs)


def _structural_raw_columns(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _structural_raw_columns as _impl

    return _impl(*args, **kwargs)


def _cue_masks_for_target(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _cue_masks_for_target as _impl

    return _impl(*args, **kwargs)


def _network_peak_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _network_peak_summary as _impl

    return _impl(*args, **kwargs)


def save_debug_figures(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.debug_figures import save_debug_figures as _impl

    return _impl(*args, **kwargs)


def _save_debug_category_plot(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.debug_figures import _save_debug_category_plot as _impl

    return _impl(*args, **kwargs)


def _write_config_files(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.output_contract import _write_config_files as _impl

    return _impl(*args, **kwargs)


def _write_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.output_contract import _write_summary as _impl

    return _impl(*args, **kwargs)


def _pairwise_image_sims(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _pairwise_image_sims as _impl

    return _impl(*args, **kwargs)


def _image_flat(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _image_flat as _impl

    return _impl(*args, **kwargs)


def _images_for_ids(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _images_for_ids as _impl

    return _impl(*args, **kwargs)


def _encode_cached(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _encode_cached as _impl

    return _impl(*args, **kwargs)


def _masked_image(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _masked_image as _impl

    return _impl(*args, **kwargs)


def _encoded_spike_count(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _encoded_spike_count as _impl

    return _impl(*args, **kwargs)


def _encode_image_tensor_cached(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _encode_image_tensor_cached as _impl

    return _impl(*args, **kwargs)


def _foreground_mask(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _foreground_mask as _impl

    return _impl(*args, **kwargs)


def _layer1_map(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _layer1_map as _impl

    return _impl(*args, **kwargs)


def _top_mask(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _top_mask as _impl

    return _impl(*args, **kwargs)


def _bottom_mask(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _bottom_mask as _impl

    return _impl(*args, **kwargs)


def _random_mask_like(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _random_mask_like as _impl

    return _impl(*args, **kwargs)


def _trim_or_expand_mask(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _trim_or_expand_mask as _impl

    return _impl(*args, **kwargs)


def _target_position(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _target_position as _impl

    return _impl(*args, **kwargs)


def _cosine_distance(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _cosine_distance as _impl

    return _impl(*args, **kwargs)


def _centered_cosine(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _centered_cosine as _impl

    return _impl(*args, **kwargs)


def _gini(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _gini as _impl

    return _impl(*args, **kwargs)


def _trial_condition_audit(*args, **kwargs):
    from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _trial_condition_audit as _impl

    return _impl(*args, **kwargs)


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_from_args(args: argparse.Namespace) -> Fig3Config:
    smoke = bool(args.smoke)
    run_all = bool(args.run_all)
    seq_lengths = tuple(int(v) for v in str(args.sequence_lengths).split(",") if str(v).strip())
    sweep = tuple(float(v) for v in str(args.partial_cue_keep_fraction_sweep).split(",") if str(v).strip())
    weak_cue_keep = tuple(float(v) for v in str(args.weak_cue_keep_fractions).split(",") if str(v).strip())
    weak_probe_keep = tuple(float(v) for v in str(args.weak_probe_keep_probs).split(",") if str(v).strip())
    region_ping_conditions = tuple(str(v).strip() for v in str(args.region_ping_conditions).split(",") if str(v).strip())
    region_ping_amp_sweep = tuple(float(v) for v in str(args.region_ping_amp_sweep).split(",") if str(v).strip())
    peak_cue_main_keep = float(args.peak_cue_main_keep_fraction)
    if not any(np.isclose(float(value), peak_cue_main_keep) for value in weak_cue_keep):
        weak_cue_keep = tuple(sorted([*weak_cue_keep, peak_cue_main_keep]))
    else:
        weak_cue_keep = tuple(sorted({float(value) for value in weak_cue_keep}))
    if smoke:
        weak_probe_keep = (0.2, 0.7)
        weak_cue_keep = (peak_cue_main_keep,)
    run_peak_cue_main = run_all or bool(args.run_peak_cue_main) or bool(args.run_structural_weak_cue)
    run_region_ping = bool(args.run_region_ping) or bool(args.run_region_ping_amp_sweep)
    return Fig3Config(
        model_path=str(args.model_path),
        dataset_root=str(args.dataset_root),
        output_root=str(args.output_root),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        sequence_lengths=seq_lengths,
        primary_sequence_length=int(args.primary_sequence_length),
        main_sequence_length=int(args.main_sequence_length),
        main_only_seq_len_10=bool(args.main_only_seq_len_10),
        sample_ms=int(args.sample_ms),
        delay_ms=int(args.delay_ms),
        ping_ms=int(args.ping_ms),
        ping_amp=float(args.ping_amp),
        ping_repeats=1 if smoke else int(args.ping_repeats),
        weak_probe_ms=int(args.weak_probe_ms),
        weak_probe_keep_probs=weak_probe_keep,
        weak_probe_repeats=min(int(args.weak_probe_repeats), 2) if smoke else int(args.weak_probe_repeats),
        weak_probe_mask_space=str(args.weak_probe_mask_space),
        weak_probe_use_same_mask_across_states=bool(args.weak_probe_use_same_mask_across_states),
        weak_probe_scale=float(args.weak_probe_scale),
        weak_probe_noise=float(args.weak_probe_noise),
        weak_probe_metric_mode=str(args.weak_probe_metric_mode),
        weak_probe_target_source=str(args.weak_probe_target_source),
        weak_probe_memory_scope=str(args.weak_probe_memory_scope),
        num_sequences=4 if smoke else int(args.num_sequences),
        batch_size=min(int(args.batch_size), 2) if smoke else int(args.batch_size),
        peak_q=float(args.peak_q),
        valley_q=float(args.valley_q),
        n_null=8 if smoke else int(args.n_null),
        weak_cue_target_source=str(args.weak_cue_target_source),
        weak_cue_keep_fractions=weak_cue_keep,
        weak_cue_repeats=2 if smoke else int(args.weak_cue_repeats),
        weak_cue_mask_mode=str(args.weak_cue_mask_mode),
        foreground_threshold=float(args.foreground_threshold),
        functional_restore_mode=str(args.functional_restore_mode),
        partial_cue_keep_fraction=float(args.partial_cue_keep_fraction),
        partial_cue_keep_fraction_sweep=sweep,
        partial_cue_repeats=2 if smoke else int(args.partial_cue_repeats),
        target_position=str(args.target_position),
        run_state_bank=run_all or bool(args.run_state_bank),
        run_progressive_update=run_all or bool(args.run_progressive_update),
        run_peak_valley_landscape=run_all or bool(args.run_peak_valley_landscape),
        run_neutral_ping=run_all or bool(args.run_neutral_ping),
        run_weak_probe=run_all or bool(args.run_weak_probe),
        run_region_ping=run_region_ping,
        run_region_ping_s0_control=bool(args.run_region_ping_s0_control),
        run_region_ping_amp_sweep=bool(args.run_region_ping_amp_sweep),
        run_peak_aligned_completion=False,
        run_peak_cue_main=run_peak_cue_main,
        run_population_morphology_supplement=bool(args.run_population_morphology_supplement),
        run_structural_weak_cue=bool(args.run_structural_weak_cue) or bool(args.run_peak_aligned_completion),
        run_structural_weak_cue_supplement=run_all or bool(args.run_structural_weak_cue_supplement) or bool(args.run_structural_weak_cue) or bool(args.run_peak_aligned_completion),
        run_supplement=run_all or bool(args.run_supplement),
        save_debug_figures=bool(args.save_debug_figures),
        save_spike_cache=bool(args.save_spike_cache),
        save_all_layer_state_bank=bool(args.save_all_layer_state_bank),
        show_progress=not bool(args.no_progress),
        use_encode_cache=not bool(args.no_encode_cache),
        enable_condition_batch=bool(args.enable_condition_batch),
        smoke=smoke,
        peak_cue_main_keep_fraction=peak_cue_main_keep,
        region_ping_q=float(args.region_ping_q),
        region_ping_support_metric=str(args.region_ping_support_metric),
        region_ping_conditions=region_ping_conditions,
        region_ping_repeats=min(int(args.region_ping_repeats), 2) if smoke else int(args.region_ping_repeats),
        region_ping_amp_sweep=region_ping_amp_sweep,
        region_ping_use_random_matched=bool(args.region_ping_use_random_matched),
        weak_probe_include_singleton=bool(args.weak_probe_include_singleton),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Fig.3 multi-item peak landscape experiment.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-state-bank", action="store_true")
    parser.add_argument("--run-progressive-update", action="store_true")
    parser.add_argument("--run-peak-valley-landscape", action="store_true")
    parser.add_argument("--run-neutral-ping", action="store_true")
    parser.add_argument("--run-weak-probe", action="store_true")
    parser.add_argument("--run-region-ping", action="store_true")
    parser.add_argument("--run-region-ping-s0-control", action="store_true")
    parser.add_argument("--run-region-ping-amp-sweep", action="store_true")
    parser.add_argument("--run-peak-aligned-completion", action="store_true")
    parser.add_argument("--run-peak-cue-main", action="store_true")
    parser.add_argument("--run-population-morphology-supplement", action="store_true")
    parser.add_argument("--run-structural-weak-cue", action="store_true")
    parser.add_argument("--run-structural-weak-cue-supplement", action="store_true")
    parser.add_argument("--run-supplement", action="store_true")
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--save-spike-cache", action="store_true")
    parser.add_argument("--save-all-layer-state-bank", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-encode-cache", action="store_true")
    parser.add_argument("--enable-condition-batch", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sequence-lengths", default="3,5,7,10")
    parser.add_argument("--primary-sequence-length", type=int, default=7)
    parser.add_argument("--main-sequence-length", type=int, default=10)
    parser.add_argument("--main-only-seq-len-10", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay-ms", type=int, default=200)
    parser.add_argument("--ping-ms", type=int, default=30)
    parser.add_argument("--ping-amp", type=float, default=1.0)
    parser.add_argument("--ping-repeats", type=int, default=1)
    parser.add_argument("--weak-probe-ms", type=int, default=100)
    parser.add_argument("--weak-probe-keep-probs", default="0.05,0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    parser.add_argument("--weak-probe-repeats", type=int, default=20)
    parser.add_argument("--weak-probe-mask-space", default="encoded_spikes", choices=["encoded_spikes"])
    parser.add_argument("--weak-probe-use-same-mask-across-states", dest="weak_probe_use_same_mask_across_states", action="store_true", default=True)
    parser.add_argument("--weak-probe-independent-masks-across-states", dest="weak_probe_use_same_mask_across_states", action="store_false")
    parser.add_argument("--weak-probe-scale", type=float, default=0.35)
    parser.add_argument("--weak-probe-noise", type=float, default=0.0)
    parser.add_argument("--weak-probe-metric-mode", default="fig2_compat", choices=["fig2_compat", "legacy"])
    parser.add_argument("--weak-probe-target-source", default="sequence_member_random", choices=["sequence_member_random", "unseen_random", "both"])
    parser.add_argument("--weak-probe-memory-scope", default="final_only", choices=["final_only", "all_prefixes"])
    parser.add_argument("--weak-probe-include-singleton", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--region-ping-q", type=float, default=0.20)
    parser.add_argument("--region-ping-support-metric", default="gain_ratio_map", choices=["gain_ratio_map", "delta_gain_map", "G_final"])
    parser.add_argument("--region-ping-conditions", default="peak,valley,random")
    parser.add_argument("--region-ping-repeats", type=int, default=5)
    parser.add_argument("--region-ping-amp-sweep", default="0.25,0.5,1.0,1.5")
    parser.add_argument("--region-ping-use-random-matched", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--peak-q", type=float, default=0.20)
    parser.add_argument("--valley-q", type=float, default=0.20)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--weak-cue-target-source", default="sequence_member_random", choices=["sequence_member_random", "unseen_random", "both"])
    parser.add_argument("--weak-cue-keep-fractions", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--peak-cue-main-keep-fraction", type=float, default=0.10)
    parser.add_argument("--weak-cue-repeats", type=int, default=20)
    parser.add_argument("--weak-cue-mask-mode", default="rank_within_target_foreground", choices=["rank_within_target_foreground"])
    parser.add_argument("--foreground-threshold", type=float, default=0.1)
    parser.add_argument("--functional-restore-mode", choices=["full_boundary", "stsp_only", "stsp_only_legacy_current_ux"], default="stsp_only")
    parser.add_argument("--partial-cue-keep-fraction", type=float, default=0.10)
    parser.add_argument("--partial-cue-keep-fraction-sweep", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--partial-cue-repeats", type=int, default=20)
    parser.add_argument("--target-position", default="K-1")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
