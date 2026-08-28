from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.core.network import SDNN_Network
from src.data.encoding import DoGSpikeEncoder
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import run_dms_snapshot_rollout, run_monitored_dms_rollout
from src.experiments.common.ping_common import (
    decode_prediction_and_fire_time_from_layer3,
    prepare_network_state,
    reset_l3_decision_window,
)
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.voltage_readout import extract_class_voltage_scores, resolve_readout_step
from src.experiments.paper_figures.common.bundle_io import (
    copy_csv_alias,
    prepare_seed_dirs,
    record_optional_missing,
    relative_to_root,
    resolve_seed_dir,
    save_csv_with_registry,
    write_artifact_manifest,
    write_empty_csv_with_warning,
    write_json_file,
    write_run_log,
)
from src.experiments.paper_figures.common.progress import ProgressTracker, planned_phases
from src.experiments.paper_figures.fig4.types import (
    ExperimentContext,
    Fig4Config,
    L3AccumulatorReplayBank,
    OverlapPerturbationCompatibleBank,
    OverlapReentryDMSBank,
    SimilarityBiasCompatibleBank,
)
from src.experiments.distractor.shared.l3_replay import (
    center_vector as _legacy_center_vector,
    make_l3_region_masks,
    nanargmax_with_default as _legacy_nanargmax_with_default,
    run_dms_with_l3_trace_capture,
    run_l3_deletion_analysis_for_pair,
    run_l3_replacement_analysis_for_pair,
    summarize_l3_mechanism_results,
    vector_similarity as _legacy_vector_similarity,
)
from src.experiments.distractor.shared.masking import (
    normalize_pattern_vector,
    run_overlap_perturbed_dms,
)
from src.plotting.common.io import apply_publication_style, save_figure_all_formats

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _progress(iterable, *, total=None, desc: str = "", enabled: bool = True):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


FIGURE_ID = "fig4_overlap_reentry"
FIG4_DESIGN_VERSION = "overlap_gated_reentry_causal_decision_dynamics"
NUM_CLASSES = 10
CORE_CONDITIONS = (
    "full_dynamic",
    "full_static",
    "sample_keep_overlap_only_dynamic",
    "sample_keep_nonoverlap_only_dynamic",
    "sample_random_matched_dynamic",
)
D_L1_STSP_CONDITIONS = (
    "full_static",
    "full_dynamic_intact",
    "l1_overlap_reset",
    "l1_nonoverlap_reset",
    "l1_random_matched_reset",
)
CONDITION_LABELS = {
    "full_dynamic": "Dynamic",
    "full_static": "Static",
    "sample_keep_overlap_only_dynamic": "Overlap support",
    "sample_keep_nonoverlap_only_dynamic": "Non-overlap support",
    "sample_random_matched_dynamic": "Random matched",
    "full_dynamic_intact": "Dynamic",
    "l1_overlap_reset": "L1 overlap reset",
    "l1_nonoverlap_reset": "L1 non-overlap reset",
    "l1_random_matched_reset": "L1 random reset",
}
D_L1_STSP_CONDITION_LABELS = {
    "full_static": "Static baseline",
    "full_dynamic_intact": "Dynamic",
    "l1_overlap_reset": "L1 overlap reset",
    "l1_nonoverlap_reset": "L1 non-overlap reset",
    "l1_random_matched_reset": "L1 random reset",
}
SAMPLE_SIDE_MASKS = {
    "full_dynamic": "full_sample",
    "full_static": "full_sample",
    "sample_keep_overlap_only_dynamic": "sample_nonoverlap_mask",
    "sample_keep_nonoverlap_only_dynamic": "sample_overlap_mask",
    "sample_random_matched_dynamic": "random_matched_keep_support",
}
FIG4_MAIN_PANELS = {
    "A": "DMS sample-delay-probe / overlap-gated re-entry schematic",
    "B": "sample-probe similarity dependence of prior-history effect",
    "C": "highest-similarity-bin overlap dependence of accuracy drop",
    "D": "pre-probe layer1 STSP overlap reset accuracy drop",
    "E": "time-resolved L3 and decision-spike displacement",
    "F": "L3 accumulator replay / decision trajectory deflection",
}
FIG4_SUMMARY_PANELS = {
    "A": "DMS / overlap-gated re-entry schematic",
    "B": "similarity dependence",
    "C": "highest-similarity overlap accuracy drop",
    "D": "L1 STSP overlap reset",
    "E": "L3 / decision-spike displacement",
    "F": "L3 accumulator replay / decision trajectory deflection",
}
FIG4_LEGACY_METHODS = {
    "B": "similarity_bias_experiment-compatible snapshot readout",
    "C": "legacy overlap localization and iso-similarity controls",
    "D": "legacy overlap_causal_input_perturbation-compatible encoded-spike sample-side perturbation",
    "E": "probe_l3_trace / s2p DPI",
    "F": "l3_accumulator_mechanism-compatible L3 region deletion/replacement replay",
}
FIG4_SUPPLEMENT_PLAN = {
    "S7": "overlap transition and similarity-dissociation controls",
    "S8": "decision-dynamics and trajectory-deflection controls",
}
FIG4_MAIN_REQUIRED_OUTPUTS = [
    "data/metrics/panel_b_similarity_entry_metrics.csv",
    "data/metrics/panel_b_similarity_bin_summary.csv",
    "data/metrics/panel_b_similarity_accuracy_drop_summary.csv",
    "data/metrics/panel_c_high_similarity_overlap_accuracy_drop.csv",
    "data/metrics/panel_c_high_similarity_overlap_accuracy_drop_summary.csv",
    "data/metrics/panel_c_high_similarity_overlap_accuracy_drop_contrast.csv",
    "data/raw/panel_d_l1_stsp_overlap_perturbation_trial_readout.csv",
    "data/metrics/panel_d_l1_stsp_overlap_perturbation_summary.csv",
    "data/metrics/panel_d_l1_stsp_overlap_perturbation_contrast.csv",
    "data/metrics/panel_d_l1_stsp_overlap_perturbation_audit.csv",
    "data/metrics/panel_e_time_resolved_l3_displacement.csv",
    "data/metrics/panel_e_decision_spike_displacement.csv",
    "data/metrics/panel_f_l3_accumulator_region_replay_metrics.csv",
    "data/metrics/panel_f_l3_accumulator_summary.csv",
]
FIG4_S7_OUTPUTS = [
    "data/metrics/supp_s7_similarity_bin_full_trend.csv",
    "data/metrics/supp_s7_overlap_matching_diagnostics.csv",
    "data/metrics/supp_s7_iso_similarity_overlap_contrast.csv",
    "data/metrics/supp_s7_iso_similarity_permutation_null.csv",
    "data/metrics/supp_s7_overlap_regression_controls.csv",
    "data/metrics/supp_s7_random_nonoverlap_perturbation_controls.csv",
]
FIG4_S8_OUTPUTS = [
    "data/metrics/supp_s8_time_resolved_l3_displacement.csv",
    "data/metrics/supp_s8_decision_spike_displacement.csv",
    "data/metrics/supp_s8_l3_accumulator_replay_metrics.csv",
    "data/metrics/supp_s8_l3_accumulator_summary.csv",
    "data/metrics/supp_s8_decision_deflection_metrics.csv",
    "data/metrics/supp_s8_decision_deflection_summary.csv",
]
FIG4_COMPATIBILITY_OUTPUTS = [
    "data/metrics/panel_c_overlap_localization_metrics.csv",
    "data/metrics/panel_c_overlap_matched_comparison.csv",
    "data/metrics/panel_d_overlap_perturbation_metrics.csv",
    "data/metrics/panel_d_overlap_perturbation_summary.csv",
    "data/metrics/panel_d_overlap_perturbation_contrast.csv",
    "data/metrics/panel_d_overlap_accuracy_pair_table.csv",
    "data/metrics/panel_d_iso_similarity_matched_pairs.csv",
    "data/metrics/panel_d_overlap_accuracy_permutation_null.csv",
    "data/metrics/panel_d_overlap_accuracy_contrast_by_network.csv",
    "data/metrics/panel_d_matching_balance_diagnostics.csv",
    "data/metrics/supp_overlap_preserving_perturbation_metrics.csv",
    "data/metrics/supp_overlap_preserving_perturbation_summary.csv",
    "data/metrics/supp_decision_deflection_metrics.csv",
]
PERTURBATION_CONDITION_MAP = {
    "overlap": "sample_keep_overlap_only_dynamic",
    "nonoverlap": "sample_keep_nonoverlap_only_dynamic",
    "random": "sample_random_matched_dynamic",
    "dynamic": "full_dynamic",
    "static": "full_static",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _config_from_args(args)
    run(cfg)
    return 0


def run(cfg: Fig4Config) -> dict[str, Any]:
    seed_everything(int(cfg.network_seed))

    seed_dir = _resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = _prepare_dirs(seed_dir)
    device = resolve_device(cfg.device)
    dataset = load_mnist_skeleton_dataset(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, NUM_CLASSES)
    warnings: list[str] = []
    max_duration_ms = max(cfg.sample_ms, cfg.probe_ms, 100)
    if Path(cfg.model_path).exists():
        net, encoder = load_model_and_encoder(
            cfg.model_path,
            device=device,
            dt=cfg.dt,
            max_duration_ms=max_duration_ms,
        )
    elif cfg.smoke:
        seed_everything(int(cfg.network_seed))
        net = SDNN_Network(device=str(device)).to(device)
        net.eval()
        encoder = DoGSpikeEncoder(dt=cfg.dt, max_duration=max_duration_ms * ms, device=str(device))
        warnings.append(
            "Model checkpoint missing; smoke mode used an untrained repo SDNN_Network instance. "
            "Fig.4 outputs validate the legacy-aligned pipeline shape but are not manuscript evidence."
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
        entry_script="src.experiments.paper_figures.fig4_overlap_reentry_experiment",
        seed=cfg.network_seed,
        dataset=f"MNIST:{cfg.split}",
        command=" ".join(sys.argv),
        model_path=cfg.model_path,
        status="running",
    )
    write_run_info(seed_dir / "meta", run_info)

    try:
        similarity_bank: SimilarityBiasCompatibleBank | None = None
        overlap_bank: OverlapPerturbationCompatibleBank | None = None
        needs_similarity_bank = bool(cfg.run_similarity_entry or cfg.run_overlap_accuracy_identification)
        needs_overlap_bank = bool(cfg.run_rollouts or cfg.run_overlap_localization or cfg.run_decision_spike_displacement or cfg.run_overlap_perturbation or cfg.run_supplement)
        progress = ProgressTracker(
            ctx,
            planned_phases(
                (
                    ("config", True),
                    ("pair_sampling", True),
                    ("similarity_entry", needs_similarity_bank),
                    ("rollouts", needs_overlap_bank),
                    ("overlap_localization", cfg.run_overlap_localization),
                    ("overlap_accuracy_identification", cfg.run_overlap_accuracy_identification),
                    ("decision_spike_displacement", cfg.run_decision_spike_displacement),
                    ("decision_deflection", cfg.run_decision_deflection or cfg.run_overlap_perturbation or cfg.run_supplement),
                    ("overlap_perturbation", cfg.run_overlap_perturbation),
                    ("supplement", cfg.run_supplement),
                    ("aliases", True),
                    ("debug_figures", cfg.save_debug_figures),
                    ("summary", True),
                )
            ),
            fig_id="fig4",
        )
        with progress.phase("config"):
            _write_config_files(ctx)
        with progress.phase("pair_sampling"):
            pair_trials, candidate_pool, perturbation_masks, mask_bank = build_pair_trials(ctx)
        if needs_similarity_bank:
            with progress.phase("similarity_entry"):
                similarity_bank = run_similarity_bias_compatible_trials(ctx, pair_trials)
        if needs_overlap_bank:
            with progress.phase("rollouts"):
                overlap_bank = run_overlap_perturbation_compatible_rollouts(ctx, pair_trials, perturbation_masks, mask_bank)
        if cfg.run_overlap_localization and overlap_bank is not None:
            with progress.phase("overlap_localization"):
                compute_overlap_localization_metrics(ctx, overlap_bank)
        if cfg.run_overlap_accuracy_identification and similarity_bank is not None:
            with progress.phase("overlap_accuracy_identification"):
                compute_overlap_accuracy_identification(ctx, similarity_bank)
        if cfg.run_decision_spike_displacement and overlap_bank is not None:
            with progress.phase("decision_spike_displacement"):
                compute_probe_l3_trace_dpi_metrics(ctx, overlap_bank)
        if cfg.run_decision_deflection or ((cfg.run_overlap_perturbation or cfg.run_supplement) and overlap_bank is not None):
            with progress.phase("decision_deflection"):
                if cfg.run_decision_deflection:
                    compute_l3_accumulator_region_replay_metrics(ctx, pair_trials)
                if (cfg.run_overlap_perturbation or cfg.run_supplement) and overlap_bank is not None:
                    compute_decision_deflection_metrics(ctx, overlap_bank)
        if cfg.run_overlap_perturbation:
            with progress.phase("overlap_perturbation"):
                if overlap_bank is not None:
                    compute_overlap_preserving_perturbation_metrics(ctx, overlap_bank)
                compute_l1_stsp_overlap_perturbation_outputs(ctx, pair_trials, mask_bank)
        if cfg.run_supplement:
            with progress.phase("supplement"):
                if overlap_bank is not None:
                    compute_supplement_outputs(ctx, overlap_bank)
                elif similarity_bank is not None:
                    _write_empty_csv(ctx, ctx.metrics_dir / "supp_decision_deflection_metrics.csv", [], "overlap_bank_not_available")
                    ctx.availability["decision_deflection_available"] = False
                    ctx.availability["decision_deflection_missing_reason"] = "overlap_bank_not_available"
        with progress.phase("aliases"):
            write_fig4_panel_aliases_and_supplement_aliases(ctx)
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


def build_pair_trials(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.pair_sampling import build_pair_trials as _impl

    return _impl(*args, **kwargs)


def run_overlap_reentry_rollouts(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.rollouts import run_overlap_reentry_rollouts as _impl

    return _impl(*args, **kwargs)


def run_similarity_bias_compatible_trials(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.rollouts import run_similarity_bias_compatible_trials as _impl

    return _impl(*args, **kwargs)


def run_overlap_perturbation_compatible_rollouts(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.rollouts import run_overlap_perturbation_compatible_rollouts as _impl

    return _impl(*args, **kwargs)


def compute_similarity_entry_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.similarity_entry import compute_similarity_entry_metrics as _impl

    return _impl(*args, **kwargs)


def compute_overlap_localization_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_localization import compute_overlap_localization_metrics as _impl

    return _impl(*args, **kwargs)


def compute_overlap_accuracy_identification(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_accuracy_identification import compute_overlap_accuracy_identification as _impl

    return _impl(*args, **kwargs)


def compute_high_similarity_overlap_accuracy_drop(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_accuracy_identification import compute_high_similarity_overlap_accuracy_drop as _impl

    return _impl(*args, **kwargs)


def compute_decision_spike_displacement(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.decision_spike_displacement import compute_decision_spike_displacement as _impl

    return _impl(*args, **kwargs)


def compute_probe_l3_trace_dpi_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.decision_spike_displacement import compute_probe_l3_trace_dpi_metrics as _impl

    return _impl(*args, **kwargs)


def compute_decision_deflection_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.decision_deflection import compute_decision_deflection_metrics as _impl

    return _impl(*args, **kwargs)


def compute_l3_accumulator_region_replay_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.decision_deflection import compute_l3_accumulator_region_replay_metrics as _impl

    return _impl(*args, **kwargs)


def compute_overlap_preserving_perturbation_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import compute_overlap_preserving_perturbation_metrics as _impl

    return _impl(*args, **kwargs)


def _overlap_perturbation_contrast(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _overlap_perturbation_contrast as _impl

    return _impl(*args, **kwargs)


def compute_l1_stsp_overlap_perturbation_outputs(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import compute_l1_stsp_overlap_perturbation_outputs as _impl

    return _impl(*args, **kwargs)


def _run_dynamic_sample_delay_to_preprobe(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _run_dynamic_sample_delay_to_preprobe as _impl

    return _impl(*args, **kwargs)


def _run_probe_with_l1_dynamic_l23_frozen(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _run_probe_with_l1_dynamic_l23_frozen as _impl

    return _impl(*args, **kwargs)


def _fig4_step_network(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _fig4_step_network as _impl

    return _impl(*args, **kwargs)


def _apply_l1_reset_for_condition(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _apply_l1_reset_for_condition as _impl

    return _impl(*args, **kwargs)


def _l1_mask_tensor(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _l1_mask_tensor as _impl

    return _impl(*args, **kwargs)


def _l1_unit_count_for_mask(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _l1_unit_count_for_mask as _impl

    return _impl(*args, **kwargs)


def _snapshot_layer_ux(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _snapshot_layer_ux as _impl

    return _impl(*args, **kwargs)


def _ux_max_abs_diff(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _ux_max_abs_diff as _impl

    return _impl(*args, **kwargs)


def _snapshot_runtime_state(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _snapshot_runtime_state as _impl

    return _impl(*args, **kwargs)


def _restore_runtime_state(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _restore_runtime_state as _impl

    return _impl(*args, **kwargs)


def _runtime_state_max_abs_diff(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _runtime_state_max_abs_diff as _impl

    return _impl(*args, **kwargs)


def _l1_stsp_row(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _l1_stsp_row as _impl

    return _impl(*args, **kwargs)


def _l1_stsp_raw_columns(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _l1_stsp_raw_columns as _impl

    return _impl(*args, **kwargs)


def _l1_stsp_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _l1_stsp_summary as _impl

    return _impl(*args, **kwargs)


def _l1_stsp_contrast(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import _l1_stsp_contrast as _impl

    return _impl(*args, **kwargs)


def compute_supplement_outputs(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.supplement import compute_supplement_outputs as _impl

    return _impl(*args, **kwargs)


def write_fig4_panel_aliases_and_supplement_aliases(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.supplement import write_fig4_panel_aliases_and_supplement_aliases as _impl

    return _impl(*args, **kwargs)


def _write_s7_similarity_bin_full_trend(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.supplement import _write_s7_similarity_bin_full_trend as _impl

    return _impl(*args, **kwargs)


def _write_s7_overlap_matching_diagnostics(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.supplement import _write_s7_overlap_matching_diagnostics as _impl

    return _impl(*args, **kwargs)


def _write_s7_overlap_regression_controls(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.supplement import _write_s7_overlap_regression_controls as _impl

    return _impl(*args, **kwargs)


def _write_s7_random_nonoverlap_perturbation_controls(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.supplement import _write_s7_random_nonoverlap_perturbation_controls as _impl

    return _impl(*args, **kwargs)


def _decision_deflection_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.supplement import _decision_deflection_summary as _impl

    return _impl(*args, **kwargs)


def save_debug_figures(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.debug_figures import save_debug_figures as _impl

    return _impl(*args, **kwargs)


def _write_config_files(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.output_contract import _write_config_files as _impl

    return _impl(*args, **kwargs)


def _write_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.output_contract import _write_summary as _impl

    return _impl(*args, **kwargs)


def _fig4c_high_similarity_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _fig4c_high_similarity_summary as _impl

    return _impl(*args, **kwargs)


def _fig4d_l1_stsp_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _fig4d_l1_stsp_summary as _impl

    return _impl(*args, **kwargs)


def _json_float(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _json_float as _impl

    return _impl(*args, **kwargs)


def _json_int(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _json_int as _impl

    return _impl(*args, **kwargs)


def _any_metric_stage(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _any_metric_stage as _impl

    return _impl(*args, **kwargs)


def _n_iso_similarity_matches(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _n_iso_similarity_matches as _impl

    return _impl(*args, **kwargs)


def _resolve_fig4_readout_step(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _resolve_fig4_readout_step as _impl

    return _impl(*args, **kwargs)


def _image_cache(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _image_cache as _impl

    return _impl(*args, **kwargs)


def _aggregate_prediction(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _aggregate_prediction as _impl

    return _impl(*args, **kwargs)


def _compute_bvec(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _compute_bvec as _impl

    return _impl(*args, **kwargs)


def _bvec_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _bvec_summary as _impl

    return _impl(*args, **kwargs)


def _cti_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _cti_summary as _impl

    return _impl(*args, **kwargs)


def _sample_input_mask_for_condition(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _sample_input_mask_for_condition as _impl

    return _impl(*args, **kwargs)


def _l3_summary_rows(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _l3_summary_rows as _impl

    return _impl(*args, **kwargs)


def _condition_sample_image(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _condition_sample_image as _impl

    return _impl(*args, **kwargs)


def _prepare_condition_batch(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _prepare_condition_batch as _impl

    return _impl(*args, **kwargs)


def _encode_batch(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _encode_batch as _impl

    return _impl(*args, **kwargs)


def _class_evidence_trace(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _class_evidence_trace as _impl

    return _impl(*args, **kwargs)


def _foreground_mask(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _foreground_mask as _impl

    return _impl(*args, **kwargs)


def _build_masks(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _build_masks as _impl

    return _impl(*args, **kwargs)


def _random_matched_mask(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _random_matched_mask as _impl

    return _impl(*args, **kwargs)


def _mask_energy(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _mask_energy as _impl

    return _impl(*args, **kwargs)


def _dice(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _dice as _impl

    return _impl(*args, **kwargs)


def _safe_div(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _safe_div as _impl

    return _impl(*args, **kwargs)


def _assign_bins(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _assign_bins as _impl

    return _impl(*args, **kwargs)


def _balanced_select_pairs(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _balanced_select_pairs as _impl

    return _impl(*args, **kwargs)


def _assign_matched_groups(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _assign_matched_groups as _impl

    return _impl(*args, **kwargs)


def _matched_pairs_table(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _matched_pairs_table as _impl

    return _impl(*args, **kwargs)


def _write_panel_a_example(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _write_panel_a_example as _impl

    return _impl(*args, **kwargs)


def _cond_row(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _cond_row as _impl

    return _impl(*args, **kwargs)


def _trace(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _trace as _impl

    return _impl(*args, **kwargs)


def _vector(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _vector as _impl

    return _impl(*args, **kwargs)


def _vec_distance(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _vec_distance as _impl

    return _impl(*args, **kwargs)


def _cosine(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _cosine as _impl

    return _impl(*args, **kwargs)


def _projection(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _projection as _impl

    return _impl(*args, **kwargs)


def _dpi_timecourse(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _dpi_timecourse as _impl

    return _impl(*args, **kwargs)


def _mean_dpi(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _mean_dpi as _impl

    return _impl(*args, **kwargs)


def _decision_deflection(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _decision_deflection as _impl

    return _impl(*args, **kwargs)


def _summary_by_bin(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _summary_by_bin as _impl

    return _impl(*args, **kwargs)


def _panel_b_accuracy_drop_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _panel_b_accuracy_drop_summary as _impl

    return _impl(*args, **kwargs)


def _pair_effect_table(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _pair_effect_table as _impl

    return _impl(*args, **kwargs)


def _panel_c_matched_comparison(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _panel_c_matched_comparison as _impl

    return _impl(*args, **kwargs)


def _overlap_regression(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _overlap_regression as _impl

    return _impl(*args, **kwargs)


def _two_by_two(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _two_by_two as _impl

    return _impl(*args, **kwargs)


def _matching_diagnostics(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _matching_diagnostics as _impl

    return _impl(*args, **kwargs)


def _accuracy_pair_table(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _accuracy_pair_table as _impl

    return _impl(*args, **kwargs)


def _build_iso_similarity_overlap_matches(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _build_iso_similarity_overlap_matches as _impl

    return _impl(*args, **kwargs)


def _high_similarity_overlap_accuracy_drop_tables(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _high_similarity_overlap_accuracy_drop_tables as _impl

    return _impl(*args, **kwargs)


def _highest_bin_label(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _highest_bin_label as _impl

    return _impl(*args, **kwargs)


def _iso_match_row(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_2 import _iso_match_row as _impl

    return _impl(*args, **kwargs)


def _matched_overlap_permutation_test(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_2 import _matched_overlap_permutation_test as _impl

    return _impl(*args, **kwargs)


def _overlap_accuracy_contrast_by_network(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_2 import _overlap_accuracy_contrast_by_network as _impl

    return _impl(*args, **kwargs)


def _matching_balance_diagnostics(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_2 import _matching_balance_diagnostics as _impl

    return _impl(*args, **kwargs)


def _compute_overlap_excess_accuracy(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_2 import _compute_overlap_excess_accuracy as _impl

    return _impl(*args, **kwargs)


def _overlap_accuracy_regression(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_2 import _overlap_accuracy_regression as _impl

    return _impl(*args, **kwargs)


def _relative_difference(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_2 import _relative_difference as _impl

    return _impl(*args, **kwargs)


def _accuracy_pair_columns(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_2 import _accuracy_pair_columns as _impl

    return _impl(*args, **kwargs)


def _iso_match_columns(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_2 import _iso_match_columns as _impl

    return _impl(*args, **kwargs)


def _random_mask_controls(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_2 import _random_mask_controls as _impl

    return _impl(*args, **kwargs)


def _condition_audit(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_2 import _condition_audit as _impl

    return _impl(*args, **kwargs)


def _from_row(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_2 import _from_row as _impl

    return _impl(*args, **kwargs)


def _finite_delta(*args, **kwargs):
    from src.experiments.paper_figures.fig4.subexperiments.helpers_2 import _finite_delta as _impl

    return _impl(*args, **kwargs)


def _write_run_log(ctx: ExperimentContext) -> None:
    write_run_log(ctx, now_text=_now())


def _save_csv(ctx: ExperimentContext, df: pd.DataFrame, path: Path) -> None:
    save_csv_with_registry(ctx, df, path)


def _csv_nonempty(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(not pd.read_csv(path).empty)
    except Exception:
        return False


def _copy_csv_alias(ctx: ExperimentContext, src: Path, dst: Path, *, empty_columns: Sequence[str], reason: str) -> None:
    copy_csv_alias(ctx, src, dst, empty_columns=empty_columns, reason=reason, message_label="Fig.4 alias")


def _write_empty_csv(ctx: ExperimentContext, dst: Path, columns: Sequence[str], reason: str) -> None:
    write_empty_csv_with_warning(ctx, dst, columns, reason, message_label="Fig.4 alias")


def _record_optional_missing(ctx: ExperimentContext, output_name: str, reason: str) -> None:
    record_optional_missing(ctx, output_name, reason, message_label="Fig.4 alias")


def _mean_existing(df: pd.DataFrame, columns: Sequence[str]) -> float:
    for column in columns:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            return float(values.mean()) if not values.empty else float("nan")
    return float("nan")


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


def _config_from_args(args: argparse.Namespace) -> Fig4Config:
    smoke = bool(args.smoke)
    run_all = bool(args.run_all)
    delay_ms = int(args.delay_ms)
    if bool(args.legacy_exact_mode):
        delay_ms = 500
    return Fig4Config(
        model_path=str(args.model_path),
        dataset_root=str(args.dataset_root),
        output_root=str(args.output_root),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        sample_ms=int(args.sample_ms),
        delay_ms=delay_ms,
        probe_ms=int(args.probe_ms),
        batch_size=min(int(args.batch_size), 4) if smoke else int(args.batch_size),
        max_pairs=16 if smoke else int(args.max_pairs),
        num_similarity_bins=int(args.num_similarity_bins),
        num_overlap_bins=int(args.num_overlap_bins),
        overlap_mask_mode=str(args.overlap_mask_mode),
        foreground_threshold=float(args.foreground_threshold),
        dilation_radius=int(args.dilation_radius),
        random_mask_candidates=min(int(args.random_mask_candidates), 8) if smoke else int(args.random_mask_candidates),
        n_null=8 if smoke else int(args.n_null),
        save_full_trace=bool(args.save_full_trace),
        save_l3_trace=not bool(args.no_save_l3_trace),
        run_pair_sampling=run_all or bool(args.run_pair_sampling),
        run_rollouts=run_all or bool(args.run_rollouts),
        run_similarity_entry=run_all or bool(args.run_similarity_entry),
        run_overlap_localization=run_all or bool(args.run_overlap_localization),
        run_overlap_accuracy_identification=run_all or bool(args.run_overlap_accuracy_identification),
        run_decision_spike_displacement=run_all or bool(args.run_decision_spike_displacement),
        run_decision_deflection=run_all or bool(args.run_decision_deflection),
        run_overlap_perturbation=run_all or bool(args.run_overlap_perturbation),
        run_supplement=run_all or bool(args.run_supplement),
        num_iso_similarity_bins=5 if smoke else int(args.num_iso_similarity_bins),
        overlap_tail_quantile=float(args.overlap_tail_quantile),
        match_similarity_caliper=float(args.match_similarity_caliper),
        match_energy_caliper=float(args.match_energy_caliper),
        match_require_probe_label=bool(args.match_require_probe_label),
        match_require_class_pair=bool(args.match_require_class_pair),
        require_distinct_pair_labels=bool(args.require_distinct_pair_labels),
        min_matches_per_network=int(args.min_matches_per_network),
        n_match_permutations=100 if smoke else int(args.n_match_permutations),
        save_debug_figures=bool(args.save_debug_figures),
        show_progress=not bool(args.no_progress),
        enable_condition_batch=bool(args.enable_condition_batch),
        use_legacy_similarity_bias_method=bool(args.use_legacy_similarity_bias_method),
        use_legacy_overlap_perturbation_method=bool(args.use_legacy_overlap_perturbation_method),
        use_legacy_l3_accumulator_method=bool(args.use_legacy_l3_accumulator_method),
        legacy_exact_mode=bool(args.legacy_exact_mode),
        l3_mask_mode=str(args.l3_mask_mode),
        l3_region_batch_size=min(int(args.l3_region_batch_size), 8) if smoke else int(args.l3_region_batch_size),
        temporal_pool=str(args.temporal_pool),
        save_case_count=min(int(args.save_case_count), 2) if smoke else int(args.save_case_count),
        run_l3_region_deletion=bool(args.run_l3_region_deletion),
        run_l3_region_replacement=bool(args.run_l3_region_replacement),
        smoke=smoke,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Fig.4 overlap re-entry DMS experiment.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-pair-sampling", action="store_true")
    parser.add_argument("--run-rollouts", action="store_true")
    parser.add_argument("--run-similarity-entry", action="store_true")
    parser.add_argument("--run-overlap-localization", action="store_true")
    parser.add_argument("--run-overlap-accuracy-identification", action="store_true")
    parser.add_argument("--run-decision-spike-displacement", action="store_true")
    parser.add_argument("--run-decision-deflection", action="store_true")
    parser.add_argument("--run-overlap-perturbation", action="store_true")
    parser.add_argument("--run-supplement", action="store_true")
    parser.add_argument("--legacy-exact-mode", action="store_true", default=True)
    parser.add_argument("--use-legacy-similarity-bias-method", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-legacy-overlap-perturbation-method", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-legacy-l3-accumulator-method", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--l3-mask-mode", choices=["1x1", "2x2"], default="1x1")
    parser.add_argument("--l3-region-batch-size", type=int, default=16)
    parser.add_argument("--temporal-pool", choices=["mean"], default="mean")
    parser.add_argument("--save-case-count", type=int, default=4)
    parser.add_argument("--run-l3-region-deletion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-l3-region-replacement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--enable-condition-batch", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay-ms", type=int, default=400)
    parser.add_argument("--probe-ms", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--num-similarity-bins", type=int, default=5)
    parser.add_argument("--num-overlap-bins", type=int, default=3)
    parser.add_argument("--overlap-mask-mode", default="encoded_spike", choices=["encoded_spike", "foreground"])
    parser.add_argument("--foreground-threshold", type=float, default=0.1)
    parser.add_argument("--dilation-radius", type=int, default=1)
    parser.add_argument("--random-mask-candidates", type=int, default=32)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--num-iso-similarity-bins", type=int, default=20)
    parser.add_argument("--overlap-tail-quantile", type=float, default=0.33)
    parser.add_argument("--match-similarity-caliper", type=float, default=0.02)
    parser.add_argument("--match-energy-caliper", type=float, default=0.15)
    parser.add_argument("--match-require-probe-label", action="store_true")
    parser.add_argument("--match-require-class-pair", action="store_true")
    parser.add_argument("--require-distinct-pair-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-matches-per-network", type=int, default=20)
    parser.add_argument("--n-match-permutations", type=int, default=2000)
    parser.add_argument("--save-full-trace", action="store_true")
    parser.add_argument("--no-save-l3-trace", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
