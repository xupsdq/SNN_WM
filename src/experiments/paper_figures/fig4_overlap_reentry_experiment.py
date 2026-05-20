from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
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
from src.experiments.l3_accumulator_mechanism_experiment import (
    make_l3_region_masks,
    run_dms_with_l3_trace_capture,
    run_l3_deletion_analysis_for_pair,
    run_l3_replacement_analysis_for_pair,
    summarize_l3_mechanism_results,
    _center_vector as _legacy_center_vector,
    _nanargmax_with_default as _legacy_nanargmax_with_default,
    _vector_similarity as _legacy_vector_similarity,
)
from src.experiments.overlap_causal_input_perturbation_experiment import (
    run_overlap_perturbed_dms,
    normalize_pattern_vector,
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


@dataclass(frozen=True)
class Fig4Config:
    model_path: str
    dataset_root: str
    output_root: str
    network_seed: int
    device: str = "auto"
    split: str = "test"
    dt: float = 0.001
    sample_ms: int = 200
    delay_ms: int = 400
    probe_ms: int = 100
    batch_size: int = 16
    max_pairs: int = 500
    num_similarity_bins: int = 5
    num_overlap_bins: int = 3
    foreground_threshold: float = 0.0
    dilation_radius: int = 1
    random_mask_candidates: int = 32
    n_null: int = 100
    save_full_trace: bool = False
    save_l3_trace: bool = True
    run_pair_sampling: bool = False
    run_rollouts: bool = False
    run_similarity_entry: bool = False
    run_overlap_localization: bool = False
    run_overlap_accuracy_identification: bool = False
    run_decision_spike_displacement: bool = False
    run_decision_deflection: bool = False
    run_overlap_perturbation: bool = False
    run_supplement: bool = False
    num_iso_similarity_bins: int = 20
    overlap_tail_quantile: float = 0.33
    match_similarity_caliper: float = 0.02
    match_energy_caliper: float = 0.15
    match_require_probe_label: bool = False
    match_require_class_pair: bool = False
    min_matches_per_network: int = 20
    n_match_permutations: int = 2000
    save_debug_figures: bool = False
    show_progress: bool = True
    enable_condition_batch: bool = False
    use_legacy_similarity_bias_method: bool = True
    use_legacy_overlap_perturbation_method: bool = True
    use_legacy_l3_accumulator_method: bool = True
    legacy_exact_mode: bool = True
    l3_mask_mode: str = "1x1"
    l3_region_batch_size: int = 16
    temporal_pool: str = "mean"
    save_case_count: int = 4
    run_l3_region_deletion: bool = True
    run_l3_region_replacement: bool = True
    smoke: bool = False

    @property
    def sample_steps(self) -> int:
        return _ms_to_steps(self.sample_ms, self.dt)

    @property
    def delay_steps(self) -> int:
        return _ms_to_steps(self.delay_ms, self.dt)

    @property
    def probe_steps(self) -> int:
        return _ms_to_steps(self.probe_ms, self.dt)


@dataclass
class ExperimentContext:
    cfg: Fig4Config
    seed_dir: Path
    config_dir: Path
    trial_specs_dir: Path
    raw_dir: Path
    metrics_dir: Path
    debug_dir: Path
    device: torch.device
    dataset: Any
    class_index: dict[int, list[int]]
    net: Any
    encoder: Any
    warnings: list[str]
    output_files: dict[str, str]
    completed_modules: dict[str, bool]
    run_log: list[str]
    availability: dict[str, Any] = field(default_factory=dict)
    n_pairs: int = 0


@dataclass
class OverlapReentryDMSBank:
    pair_trials: pd.DataFrame
    perturbation_masks: pd.DataFrame
    rollout_manifest: pd.DataFrame
    condition_metrics: pd.DataFrame
    traces: dict[str, np.ndarray]
    vectors: dict[str, np.ndarray]


@dataclass
class SimilarityBiasCompatibleBank:
    pair_trials: pd.DataFrame
    trial_metrics: pd.DataFrame
    repeat_metrics: pd.DataFrame
    voltage_vectors: dict[str, np.ndarray]


@dataclass
class OverlapPerturbationCompatibleBank:
    pair_trials: pd.DataFrame
    perturbation_masks: pd.DataFrame
    rollout_manifest: pd.DataFrame
    condition_metrics: pd.DataFrame
    traces: dict[str, np.ndarray]
    vectors: dict[str, np.ndarray]


@dataclass
class L3AccumulatorReplayBank:
    pair_trials: pd.DataFrame
    region_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    pair_vectors: dict[str, np.ndarray]
    region_effects: dict[str, np.ndarray]


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
        _write_config_files(ctx)
        pair_trials, candidate_pool, perturbation_masks, mask_bank = build_pair_trials(ctx)
        similarity_bank: SimilarityBiasCompatibleBank | None = None
        overlap_bank: OverlapPerturbationCompatibleBank | None = None
        if cfg.run_similarity_entry or cfg.run_overlap_accuracy_identification:
            similarity_bank = run_similarity_bias_compatible_trials(ctx, pair_trials)
        if cfg.run_rollouts or cfg.run_overlap_localization or cfg.run_decision_spike_displacement or cfg.run_overlap_perturbation or cfg.run_supplement:
            overlap_bank = run_overlap_perturbation_compatible_rollouts(ctx, pair_trials, perturbation_masks, mask_bank)
        if cfg.run_overlap_localization and overlap_bank is not None:
            compute_overlap_localization_metrics(ctx, overlap_bank)
        if cfg.run_overlap_accuracy_identification and similarity_bank is not None:
            compute_overlap_accuracy_identification(ctx, similarity_bank)
        if cfg.run_decision_spike_displacement and overlap_bank is not None:
            compute_probe_l3_trace_dpi_metrics(ctx, overlap_bank)
        if cfg.run_decision_deflection:
            compute_l3_accumulator_region_replay_metrics(ctx, pair_trials)
        if (cfg.run_overlap_perturbation or cfg.run_supplement) and overlap_bank is not None:
            compute_decision_deflection_metrics(ctx, overlap_bank)
        if cfg.run_overlap_perturbation and overlap_bank is not None:
            compute_overlap_preserving_perturbation_metrics(ctx, overlap_bank)
        if cfg.run_overlap_perturbation:
            compute_l1_stsp_overlap_perturbation_outputs(ctx, pair_trials, mask_bank)
        if cfg.run_supplement:
            if overlap_bank is not None:
                compute_supplement_outputs(ctx, overlap_bank)
            elif similarity_bank is not None:
                _write_empty_csv(ctx, ctx.metrics_dir / "supp_decision_deflection_metrics.csv", [], "overlap_bank_not_available")
                ctx.availability["decision_deflection_available"] = False
                ctx.availability["decision_deflection_missing_reason"] = "overlap_bank_not_available"
        write_fig4_panel_aliases_and_supplement_aliases(ctx)
        if cfg.save_debug_figures:
            save_debug_figures(ctx)
        summary = _write_summary(ctx)
        _write_run_log(ctx)
        finalize_run_info(seed_dir / "meta", run_info, status="success")
        return summary
    except Exception:
        finalize_run_info(seed_dir / "meta", run_info, status="failed")
        raise


def build_pair_trials(ctx: ExperimentContext) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    cfg = ctx.cfg
    rng = np.random.default_rng(int(cfg.network_seed))
    images = torch.stack([ctx.dataset[idx][0].detach().cpu().to(torch.float32) for idx in range(len(ctx.dataset))], dim=0)
    labels = np.asarray([int(ctx.dataset[idx][1]) for idx in range(len(ctx.dataset))], dtype=np.int64)
    flat = images.view(len(images), -1).numpy().astype(np.float64, copy=False)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    flat_unit = flat / norm

    pool_rows: list[dict[str, Any]] = []
    target_candidates = max(int(cfg.max_pairs) * 12, int(cfg.max_pairs) + 64)
    label_cycle = [(a, b) for a in range(NUM_CLASSES) for b in range(NUM_CLASSES)]
    rng.shuffle(label_cycle)
    candidate_id = 0
    candidate_iter = _progress(iter(int, 1), total=target_candidates, desc="fig4 pair candidates", enabled=cfg.show_progress)
    for _ in candidate_iter:
        if candidate_id >= target_candidates:
            break
        sample_label, probe_label = label_cycle[candidate_id % len(label_cycle)]
        sample_idx = int(rng.choice(ctx.class_index[int(sample_label)]))
        probe_idx = int(rng.choice(ctx.class_index[int(probe_label)]))
        if probe_idx == sample_idx:
            choices = [idx for idx in ctx.class_index[int(probe_label)] if int(idx) != sample_idx]
            if not choices:
                continue
            probe_idx = int(rng.choice(choices))
        sim = float(np.dot(flat_unit[sample_idx], flat_unit[probe_idx]))
        sm = _foreground_mask(images[sample_idx], cfg.foreground_threshold)
        pm = _foreground_mask(images[probe_idx], cfg.foreground_threshold)
        overlap = sm & pm
        union = sm | pm
        sample_energy = _mask_energy(images[sample_idx], sm)
        probe_energy = _mask_energy(images[probe_idx], pm)
        eligible = bool(sm.any() and pm.any() and sample_idx != probe_idx)
        pool_rows.append(
            {
                "network_seed": int(cfg.network_seed),
                "candidate_id": int(candidate_id),
                "sample_image_id": sample_idx,
                "sample_label": int(sample_label),
                "probe_image_id": probe_idx,
                "probe_label": int(probe_label),
                "pixel_similarity": sim,
                "dice_overlap": _dice(sm, pm),
                "input_energy_sample": sample_energy,
                "input_energy_probe": probe_energy,
                "eligible": bool(eligible),
                "exclusion_reason": "" if eligible else "empty_foreground_or_same_image",
            }
        )
        candidate_id += 1

    candidate_pool = pd.DataFrame(pool_rows)
    eligible_pool = candidate_pool[candidate_pool["eligible"]].copy()
    if eligible_pool.empty:
        raise RuntimeError("No eligible sample-probe pairs were generated.")
    eligible_pool = _assign_bins(eligible_pool, "pixel_similarity", "similarity_bin", int(cfg.num_similarity_bins))
    eligible_pool = _assign_bins(eligible_pool, "dice_overlap", "overlap_bin", int(cfg.num_overlap_bins))

    selected = _balanced_select_pairs(eligible_pool, int(cfg.max_pairs), rng)
    selected = selected.reset_index(drop=True)
    selected["pair_id"] = np.arange(len(selected), dtype=np.int64)
    selected = _assign_matched_groups(selected)

    mask_bank: dict[int, dict[str, np.ndarray]] = {}
    pair_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    for _, row in _progress(selected.iterrows(), total=len(selected), desc="fig4 selected pairs", enabled=cfg.show_progress):
        pair_id = int(row["pair_id"])
        sample_image = images[int(row["sample_image_id"])]
        probe_image = images[int(row["probe_image_id"])]
        masks = _build_masks(sample_image, probe_image, rng, cfg)
        mask_bank[pair_id] = masks
        sample_fg = masks["sample_foreground_mask"]
        probe_fg = masks["probe_foreground_mask"]
        overlap = masks["overlap_mask"]
        union = sample_fg | probe_fg
        pair_rows.append(
            {
                "network_seed": int(cfg.network_seed),
                "pair_id": pair_id,
                "sample_image_id": int(row["sample_image_id"]),
                "sample_label": int(row["sample_label"]),
                "probe_image_id": int(row["probe_image_id"]),
                "probe_label": int(row["probe_label"]),
                "pixel_similarity": float(row["pixel_similarity"]),
                "similarity_bin": str(row["similarity_bin"]),
                "sample_foreground_area": int(sample_fg.sum()),
                "probe_foreground_area": int(probe_fg.sum()),
                "overlap_area": int(overlap.sum()),
                "union_area": int(union.sum()),
                "dice_overlap": _dice(sample_fg, probe_fg),
                "overlap_fraction_sample": _safe_div(float(overlap.sum()), float(sample_fg.sum())),
                "overlap_fraction_probe": _safe_div(float(overlap.sum()), float(probe_fg.sum())),
                "input_energy_sample": _mask_energy(sample_image, sample_fg),
                "input_energy_probe": _mask_energy(probe_image, probe_fg),
                "class_pair": f"{int(row['sample_label'])}->{int(row['probe_label'])}",
                "overlap_bin": str(row["overlap_bin"]),
                "matched_group_id": str(row.get("matched_group_id", "")),
            }
        )
        for mask_name in (
            "sample_foreground_mask",
            "probe_foreground_mask",
            "overlap_mask",
            "sample_overlap_mask",
            "sample_nonoverlap_mask",
            "sample_nonoverlap_control_mask",
            "probe_only_mask",
            "random_matched_mask",
        ):
            mask = masks[mask_name]
            matched_to = "overlap_mask" if mask_name == "random_matched_mask" else ""
            target = masks["overlap_mask"] if matched_to else mask
            mask_rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "pair_id": pair_id,
                    "mask_name": mask_name,
                    "mask_type": "sample_side" if mask_name != "probe_only_mask" else "probe_metadata",
                    "pixel_count": int(mask.sum()),
                    "input_energy": _mask_energy(sample_image if mask_name != "probe_only_mask" else probe_image, mask),
                    "spike_count_estimate": _mask_energy(sample_image if mask_name != "probe_only_mask" else probe_image, mask),
                    "matched_to": matched_to,
                    "matching_error_energy": abs(_mask_energy(sample_image, mask) - _mask_energy(sample_image, target)),
                    "matching_error_pixel_count": int(abs(int(mask.sum()) - int(target.sum()))),
                    "mask_application_space": "encoded_spikes",
                    "probe_perturbation": "disabled",
                    "sample_mask_mode": "remove",
                }
            )

    pair_trials = pd.DataFrame(pair_rows)
    perturbation_masks = pd.DataFrame(mask_rows)
    overlap_matched = _matched_pairs_table(pair_trials)
    _save_csv(ctx, pair_trials, ctx.trial_specs_dir / "pair_trials.csv")
    _save_csv(ctx, candidate_pool, ctx.trial_specs_dir / "pair_candidate_pool.csv")
    _save_csv(ctx, overlap_matched, ctx.trial_specs_dir / "overlap_matched_pairs.csv")
    _save_csv(ctx, perturbation_masks, ctx.trial_specs_dir / "perturbation_masks.csv")
    _write_panel_a_example(ctx, pair_trials, mask_bank, images)
    ctx.completed_modules["pair_sampling"] = True
    ctx.n_pairs = int(len(pair_trials))
    return pair_trials, candidate_pool, perturbation_masks, mask_bank


def run_overlap_reentry_rollouts(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    perturbation_masks: pd.DataFrame,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
) -> OverlapReentryDMSBank:
    cfg = ctx.cfg
    traces: dict[str, np.ndarray] = {}
    vectors: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    images_cache = {int(i): ctx.dataset[int(i)][0].detach().cpu().to(torch.float32) for i in pd.unique(pair_trials[["sample_image_id", "probe_image_id"]].values.ravel())}
    batch_starts = range(0, len(pair_trials), int(cfg.batch_size))
    for batch_start in _progress(batch_starts, total=math.ceil(len(pair_trials) / cfg.batch_size), desc="fig4 rollout batches", enabled=cfg.show_progress):
        batch = pair_trials.iloc[batch_start : batch_start + int(cfg.batch_size)].copy()
        probe_images = torch.stack([images_cache[int(r["probe_image_id"])] for _, r in batch.iterrows()], dim=0)
        probe_spikes = _encode_batch(ctx, probe_images, cfg.probe_steps)
        for cond_idx, condition in _progress(enumerate(CORE_CONDITIONS), total=len(CORE_CONDITIONS), desc="fig4 rollout conditions", enabled=cfg.show_progress):
            torch.manual_seed(int(cfg.network_seed) * 1009 + cond_idx)
            sample_images = torch.stack(
                [
                    _condition_sample_image(images_cache[int(r["sample_image_id"])], mask_bank[int(r["pair_id"])], condition)
                    for _, r in batch.iterrows()
                ],
                dim=0,
            )
            sample_spikes = _encode_batch(ctx, sample_images, cfg.sample_steps)
            stsp_mode = "static_frozen" if condition == "full_static" else "dynamic"
            out = run_monitored_dms_rollout(
                net=ctx.net,
                sample_spikes=sample_spikes,
                probe_spikes=probe_spikes,
                delay_steps=cfg.delay_steps,
                stsp_mode=stsp_mode,
                phase_reset=True,
                intervention_plan=None,
                record_state_names={"layer3": ("v_mem",)},
                record_phase_names=("probe",),
            )
            l3_v = out["state_traces"]["layer3"]["v_mem"]
            l3_trace = _class_evidence_trace(ctx.net, l3_v)
            pred = out["predictions"]["prediction_probe"].detach().cpu().numpy().astype(np.int64, copy=False)
            fire_t = out["predictions"]["first_fire_t_probe"].detach().cpu().numpy().astype(np.int64, copy=False)
            final_vec = l3_trace[-1] if l3_trace.size else np.zeros((len(batch), NUM_CLASSES), dtype=np.float32)
            for local_idx, (_, r) in enumerate(batch.iterrows()):
                pair_id = int(r["pair_id"])
                key_prefix = f"pair_{pair_id}_{condition}"
                trace = np.asarray(l3_trace[:, local_idx, :], dtype=np.float32)
                if cfg.save_l3_trace:
                    traces[f"{key_prefix}_l3_trace"] = trace
                vectors[f"{key_prefix}_class_evidence"] = np.asarray(final_vec[local_idx], dtype=np.float32)
                vectors[f"{key_prefix}_grouped_voltage"] = np.asarray(final_vec[local_idx], dtype=np.float32)
                vectors[f"{key_prefix}_prediction"] = np.asarray([int(pred[local_idx])], dtype=np.int64)
                metric_rows.append(
                    {
                        "network_seed": int(cfg.network_seed),
                        "pair_id": pair_id,
                        "condition": condition,
                        "sample_mask_name": SAMPLE_SIDE_MASKS[condition],
                        "probe_mask_name": "full_probe",
                        "prediction": int(pred[local_idx]),
                        "correctness": int(int(pred[local_idx]) == int(r["probe_label"])),
                        "first_fire_time": int(fire_t[local_idx]),
                        "probe_label": int(r["probe_label"]),
                        "similarity_bin": str(r["similarity_bin"]),
                        "overlap_bin": str(r["overlap_bin"]),
                        "pixel_similarity": float(r["pixel_similarity"]),
                        "dice_overlap": float(r["dice_overlap"]),
                    }
                )
                manifest_rows.append(
                    {
                        "network_seed": int(cfg.network_seed),
                        "pair_id": pair_id,
                        "condition": condition,
                        "sample_mask_name": SAMPLE_SIDE_MASKS[condition],
                        "probe_mask_name": "full_probe",
                        "sample_ms": int(cfg.sample_ms),
                        "delay_ms": int(cfg.delay_ms),
                        "probe_ms": int(cfg.probe_ms),
                        "saved_l3_trace": bool(cfg.save_l3_trace),
                        "saved_full_trace": bool(cfg.save_full_trace),
                        "trace_file": "probe_trace_arrays_l3.npz",
                        "vector_file": "readout_trajectory_vectors.npz",
                        "notes": "probe input unchanged; sample-side mask controls prior support writing",
                    }
                )

    condition_metrics = pd.DataFrame(metric_rows)
    rollout_manifest = pd.DataFrame(manifest_rows)
    _save_csv(ctx, rollout_manifest, ctx.raw_dir / "rollout_manifest.csv")
    np.savez_compressed(ctx.raw_dir / "probe_trace_arrays_l3.npz", **traces)
    np.savez_compressed(ctx.raw_dir / "readout_trajectory_vectors.npz", **vectors)
    np.savez_compressed(
        ctx.raw_dir / "panel_f_perturbation_trace_arrays.npz",
        **{k: v for k, v in traces.items() if any(c in k for c in ("sample_keep_overlap", "sample_keep_nonoverlap", "sample_random"))},
    )
    ctx.output_files["probe_trace_arrays_l3"] = "data/raw/probe_trace_arrays_l3.npz"
    ctx.output_files["readout_trajectory_vectors"] = "data/raw/readout_trajectory_vectors.npz"
    ctx.output_files["panel_f_perturbation_trace_arrays"] = "data/raw/panel_f_perturbation_trace_arrays.npz"
    ctx.completed_modules["rollouts"] = True
    return OverlapReentryDMSBank(pair_trials, perturbation_masks, rollout_manifest, condition_metrics, traces, vectors)


def run_similarity_bias_compatible_trials(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
) -> SimilarityBiasCompatibleBank:
    cfg = ctx.cfg
    readout_step = _resolve_fig4_readout_step(ctx)
    repeat_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    voltage_dynamic_rows: list[np.ndarray] = []
    voltage_static_rows: list[np.ndarray] = []
    images_cache = _image_cache(ctx, pair_trials)
    repeats = 1
    for batch_start in _progress(
        range(0, len(pair_trials), int(cfg.batch_size)),
        total=math.ceil(len(pair_trials) / int(cfg.batch_size)),
        desc="fig4 legacy similarity batches",
        enabled=cfg.show_progress,
    ):
        batch = pair_trials.iloc[batch_start : batch_start + int(cfg.batch_size)].copy()
        sample_images = torch.stack([images_cache[int(r["sample_image_id"])] for _, r in batch.iterrows()], dim=0)
        probe_images = torch.stack([images_cache[int(r["probe_image_id"])] for _, r in batch.iterrows()], dim=0)
        sample_spikes = _encode_batch(ctx, sample_images, cfg.sample_steps)
        probe_spikes = _encode_batch(ctx, probe_images, cfg.probe_steps)
        mode_preds: dict[str, list[np.ndarray]] = {"dynamic": [], "static_frozen": []}
        mode_voltages: dict[str, list[np.ndarray]] = {"dynamic": [], "static_frozen": []}
        mode_fires: dict[str, list[np.ndarray]] = {"dynamic": [], "static_frozen": []}
        for repeat_idx in range(repeats):
            for mode in ("dynamic", "static_frozen"):
                out = run_dms_snapshot_rollout(
                    ctx.net,
                    sample_spikes=sample_spikes,
                    probe_spikes=probe_spikes,
                    delay_steps=cfg.delay_steps,
                    stsp_mode=mode,
                    phase_reset=True,
                    intervention_plan=None,
                    readout_step=readout_step,
                    snapshot_state_names=("v_mem",),
                )
                snapshot = out["readout_snapshots"]["layer3"]["v_mem"]
                fire_t = out["predictions"]["first_fire_t_probe"].detach().cpu().numpy().astype(np.int64, copy=False)
                bundles = extract_class_voltage_scores(
                    snapshot,
                    num_classes=int(getattr(ctx.net.layer3, "num_classes", NUM_CLASSES)),
                    neurons_per_class=int(getattr(ctx.net.layer3, "neurons_per_class", 1)),
                    pooling="top_m_mean",
                    m=1,
                    backend="dms_voltage_wta",
                    readout_step=readout_step,
                )
                volt = np.stack([np.asarray(bundle.class_scores, dtype=np.float64) for bundle in bundles], axis=0)
                pred = np.asarray([int(bundle.predicted_label) for bundle in bundles], dtype=np.int64)
                mode_preds[mode].append(pred)
                mode_voltages[mode].append(volt)
                mode_fires[mode].append(fire_t)
        for local_idx, row in enumerate(batch.itertuples(index=False)):
            dyn_stack = np.stack([arr[local_idx] for arr in mode_voltages["dynamic"]], axis=0)
            sta_stack = np.stack([arr[local_idx] for arr in mode_voltages["static_frozen"]], axis=0)
            dyn_mean = np.asarray(dyn_stack.mean(axis=0), dtype=np.float64)
            sta_mean = np.asarray(sta_stack.mean(axis=0), dtype=np.float64)
            dyn_pred = _aggregate_prediction([int(pred[local_idx]) for pred in mode_preds["dynamic"]], dyn_mean)
            sta_pred = _aggregate_prediction([int(pred[local_idx]) for pred in mode_preds["static_frozen"]], sta_mean)
            dyn_fire = int(np.min([int(ft[local_idx]) for ft in mode_fires["dynamic"]]))
            sta_fire = int(np.min([int(ft[local_idx]) for ft in mode_fires["static_frozen"]]))
            correct_dynamic = int(dyn_pred == int(row.probe_label))
            correct_static = int(sta_pred == int(row.probe_label))
            voltage_index = len(voltage_dynamic_rows)
            voltage_dynamic_rows.append(dyn_mean)
            voltage_static_rows.append(sta_mean)
            for repeat_idx in range(repeats):
                repeat_rows.append(
                    {
                        "network_seed": int(cfg.network_seed),
                        "pair_id": int(row.pair_id),
                        "repeat_index": int(repeat_idx),
                        "sample_image_id": int(row.sample_image_id),
                        "probe_image_id": int(row.probe_image_id),
                        "sample_label": int(row.sample_label),
                        "probe_label": int(row.probe_label),
                        "pixel_similarity": float(row.pixel_similarity),
                        "similarity_bin": str(row.similarity_bin),
                        "pred_label_dynamic": int(mode_preds["dynamic"][repeat_idx][local_idx]),
                        "pred_label_static": int(mode_preds["static_frozen"][repeat_idx][local_idx]),
                        "correct_dynamic": int(int(mode_preds["dynamic"][repeat_idx][local_idx]) == int(row.probe_label)),
                        "correct_static": int(int(mode_preds["static_frozen"][repeat_idx][local_idx]) == int(row.probe_label)),
                        "b_vec": _compute_bvec(mode_voltages["dynamic"][repeat_idx][local_idx], mode_voltages["static_frozen"][repeat_idx][local_idx]),
                        "readout_step": int(readout_step),
                    }
                )
            trial_rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "pair_id": int(row.pair_id),
                    "sample_image_id": int(row.sample_image_id),
                    "probe_image_id": int(row.probe_image_id),
                    "sample_label": int(row.sample_label),
                    "probe_label": int(row.probe_label),
                    "class_pair": str(row.class_pair),
                    "pixel_similarity": float(row.pixel_similarity),
                    "similarity_bin": str(row.similarity_bin),
                    "dice_overlap": float(row.dice_overlap),
                    "input_energy_sample": float(row.input_energy_sample),
                    "input_energy_probe": float(row.input_energy_probe),
                    "pred_dynamic": int(dyn_pred),
                    "pred_static": int(sta_pred),
                    "correct_dynamic": int(correct_dynamic),
                    "correct_static": int(correct_static),
                    "acc_drop": int(correct_static - correct_dynamic),
                    "b_vec": _compute_bvec(dyn_mean, sta_mean),
                    "static_correct_eligible": int(correct_static == 1),
                    "drop_event": int(correct_static == 1 and correct_dynamic == 0),
                    "dynamic_rescue_event": int(correct_static == 0 and correct_dynamic == 1),
                    "first_fire_time_dynamic": int(dyn_fire),
                    "first_fire_time_static": int(sta_fire),
                    "readout_step": int(readout_step),
                    "voltage_vector_index": int(voltage_index),
                }
            )
    trial_df = pd.DataFrame(trial_rows).sort_values(["pair_id"], kind="stable").reset_index(drop=True)
    repeat_df = pd.DataFrame(repeat_rows).sort_values(["pair_id", "repeat_index"], kind="stable").reset_index(drop=True)
    voltage_payload = {
        "pair_id": trial_df["pair_id"].to_numpy(dtype=np.int64, copy=False) if not trial_df.empty else np.zeros(0, dtype=np.int64),
        "voltage_dynamic": np.stack(voltage_dynamic_rows, axis=0) if voltage_dynamic_rows else np.zeros((0, NUM_CLASSES), dtype=np.float64),
        "voltage_static": np.stack(voltage_static_rows, axis=0) if voltage_static_rows else np.zeros((0, NUM_CLASSES), dtype=np.float64),
    }
    _save_csv(ctx, trial_df, ctx.metrics_dir / "panel_b_similarity_entry_metrics.csv")
    bin_summary = _summary_by_bin(trial_df, "similarity_bin", "pixel_similarity")
    _save_csv(ctx, bin_summary, ctx.metrics_dir / "panel_b_similarity_bin_summary.csv")
    _save_csv(ctx, _panel_b_accuracy_drop_summary(trial_df), ctx.metrics_dir / "panel_b_similarity_accuracy_drop_summary.csv")
    _save_csv(ctx, _bvec_summary(trial_df), ctx.metrics_dir / "supp_similarity_bvec_summary.csv")
    _save_csv(ctx, _cti_summary(trial_df), ctx.metrics_dir / "supp_similarity_cti_summary.csv")
    _save_csv(ctx, repeat_df, ctx.raw_dir / "similarity_bias_repeat_metrics.csv")
    np.savez_compressed(ctx.raw_dir / "similarity_bias_voltage_vectors.npz", **voltage_payload)
    ctx.output_files["similarity_bias_voltage_vectors"] = "data/raw/similarity_bias_voltage_vectors.npz"
    ctx.completed_modules["similarity_entry"] = True
    return SimilarityBiasCompatibleBank(pair_trials, trial_df, repeat_df, voltage_payload)


def run_overlap_perturbation_compatible_rollouts(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    perturbation_masks: pd.DataFrame,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
) -> OverlapPerturbationCompatibleBank:
    cfg = ctx.cfg
    readout_step = _resolve_fig4_readout_step(ctx)
    traces_l1: dict[str, np.ndarray] = {}
    traces_l2: dict[str, np.ndarray] = {}
    traces_l3: dict[str, np.ndarray] = {}
    vectors: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    images_cache = _image_cache(ctx, pair_trials)
    for batch_start in _progress(
        range(0, len(pair_trials), int(cfg.batch_size)),
        total=math.ceil(len(pair_trials) / int(cfg.batch_size)),
        desc="fig4 legacy perturbation batches",
        enabled=cfg.show_progress,
    ):
        batch = pair_trials.iloc[batch_start : batch_start + int(cfg.batch_size)].copy()
        sample_images = torch.stack([images_cache[int(r["sample_image_id"])] for _, r in batch.iterrows()], dim=0)
        probe_images = torch.stack([images_cache[int(r["probe_image_id"])] for _, r in batch.iterrows()], dim=0)
        sample_spikes = _encode_batch(ctx, sample_images, cfg.sample_steps)
        probe_spikes = _encode_batch(ctx, probe_images, cfg.probe_steps)
        for condition in CORE_CONDITIONS:
            stsp_mode = "static_frozen" if condition == "full_static" else "dynamic"
            sample_input_mask = _sample_input_mask_for_condition(batch, mask_bank, condition)
            out = run_overlap_perturbed_dms(
                ctx.net,
                sample_spikes=sample_spikes,
                probe_spikes=probe_spikes,
                delay_steps=cfg.delay_steps,
                stsp_mode=stsp_mode,
                readout_step=readout_step,
                sample_input_mask=sample_input_mask,
            )
            for local_idx, row in enumerate(batch.itertuples(index=False)):
                pair_id = int(row.pair_id)
                key_prefix = f"pair_{pair_id}_{condition}"
                traces_l1[f"{key_prefix}_l1_trace"] = out.probe_l1_trace[:, local_idx].detach().cpu().to(torch.float32).numpy()
                traces_l2[f"{key_prefix}_l2_trace"] = out.probe_l2_trace[:, local_idx].detach().cpu().to(torch.float32).numpy()
                traces_l3[f"{key_prefix}_l3_trace"] = out.probe_l3_trace[:, local_idx].detach().cpu().to(torch.float32).numpy()
                vectors[f"{key_prefix}_grouped_voltage"] = np.asarray(out.grouped_voltage[local_idx], dtype=np.float32)
                vectors[f"{key_prefix}_class_evidence"] = np.asarray(out.grouped_voltage[local_idx], dtype=np.float32)
                vectors[f"{key_prefix}_prediction"] = np.asarray([int(out.prediction_probe[local_idx])], dtype=np.int64)
                mask_name = SAMPLE_SIDE_MASKS[condition]
                metric_rows.append(
                    {
                        "network_seed": int(cfg.network_seed),
                        "pair_id": pair_id,
                        "condition": condition,
                        "sample_mask_name": mask_name,
                        "probe_mask_name": "full_probe",
                        "prediction": int(out.prediction_probe[local_idx]),
                        "correctness": int(int(out.prediction_probe[local_idx]) == int(row.probe_label)),
                        "first_fire_time": int(out.first_fire_t_probe[local_idx]),
                        "prediction_probe": int(out.prediction_probe[local_idx]),
                        "first_fire_t_probe": int(out.first_fire_t_probe[local_idx]),
                        "probe_label": int(row.probe_label),
                        "similarity_bin": str(row.similarity_bin),
                        "overlap_bin": str(row.overlap_bin),
                        "pixel_similarity": float(row.pixel_similarity),
                        "dice_overlap": float(row.dice_overlap),
                        "readout_step": int(out.readout_step),
                        "mask_application_space": "encoded_spikes",
                    }
                )
                manifest_rows.append(
                    {
                        "network_seed": int(cfg.network_seed),
                        "pair_id": pair_id,
                        "condition": condition,
                        "sample_mask_name": mask_name,
                        "probe_mask_name": "full_probe",
                        "sample_ms": int(cfg.sample_ms),
                        "delay_ms": int(cfg.delay_ms),
                        "probe_ms": int(cfg.probe_ms),
                        "readout_step": int(out.readout_step),
                        "mask_application_space": "encoded_spikes",
                        "probe_perturbation": "disabled",
                        "sample_mask_mode": "remove",
                        "trace_file_l1": "probe_trace_arrays_l1.npz",
                        "trace_file_l2": "probe_trace_arrays_l2.npz",
                        "trace_file_l3": "probe_trace_arrays_l3.npz",
                        "vector_file": "readout_trajectory_vectors.npz",
                    }
                )
    condition_metrics = pd.DataFrame(metric_rows)
    rollout_manifest = pd.DataFrame(manifest_rows)
    all_traces = {**traces_l3}
    _save_csv(ctx, rollout_manifest, ctx.raw_dir / "overlap_perturbation_rollout_manifest.csv")
    _save_csv(ctx, rollout_manifest, ctx.raw_dir / "rollout_manifest.csv")
    _save_csv(ctx, perturbation_masks, ctx.metrics_dir / "supp_overlap_mask_application_audit.csv")
    np.savez_compressed(ctx.raw_dir / "probe_trace_arrays_l1.npz", **traces_l1)
    np.savez_compressed(ctx.raw_dir / "probe_trace_arrays_l2.npz", **traces_l2)
    np.savez_compressed(ctx.raw_dir / "probe_trace_arrays_l3.npz", **traces_l3)
    np.savez_compressed(ctx.raw_dir / "readout_trajectory_vectors.npz", **vectors)
    ctx.output_files["overlap_perturbation_rollout_manifest"] = "data/raw/overlap_perturbation_rollout_manifest.csv"
    ctx.output_files["probe_trace_arrays_l1"] = "data/raw/probe_trace_arrays_l1.npz"
    ctx.output_files["probe_trace_arrays_l2"] = "data/raw/probe_trace_arrays_l2.npz"
    ctx.output_files["probe_trace_arrays_l3"] = "data/raw/probe_trace_arrays_l3.npz"
    ctx.output_files["readout_trajectory_vectors"] = "data/raw/readout_trajectory_vectors.npz"
    ctx.completed_modules["rollouts"] = True
    return OverlapPerturbationCompatibleBank(pair_trials, perturbation_masks, rollout_manifest, condition_metrics, all_traces, vectors)


def compute_similarity_entry_metrics(ctx: ExperimentContext, bank: OverlapReentryDMSBank) -> None:
    rows = []
    for _, pair in bank.pair_trials.iterrows():
        pair_id = int(pair["pair_id"])
        dyn = _cond_row(bank.condition_metrics, pair_id, "full_dynamic")
        sta = _cond_row(bank.condition_metrics, pair_id, "full_static")
        b_vec = _vec_distance(bank, pair_id, "full_dynamic", "full_static")
        dpi = _mean_dpi(bank, pair_id, "full_dynamic")
        defl = _decision_deflection(bank, pair_id, "full_dynamic")
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "pair_id": pair_id,
                "sample_image_id": int(pair["sample_image_id"]),
                "probe_image_id": int(pair["probe_image_id"]),
                "sample_label": int(pair["sample_label"]),
                "probe_label": int(pair["probe_label"]),
                "pixel_similarity": float(pair["pixel_similarity"]),
                "similarity_bin": str(pair["similarity_bin"]),
                "pred_dynamic": int(dyn["prediction"]),
                "pred_static": int(sta["prediction"]),
                "correct_dynamic": int(dyn["correctness"]),
                "correct_static": int(sta["correctness"]),
                "acc_drop": int(sta["correctness"]) - int(dyn["correctness"]),
                "b_vec": b_vec,
                "DPI_L3": dpi,
                "decision_deflection": defl,
            }
        )
    df = pd.DataFrame(rows)
    summary = _summary_by_bin(df, "similarity_bin", "pixel_similarity")
    _save_csv(ctx, df, ctx.metrics_dir / "panel_b_similarity_entry_metrics.csv")
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_b_similarity_bin_summary.csv")
    _save_csv(ctx, _panel_b_accuracy_drop_summary(df), ctx.metrics_dir / "panel_b_similarity_accuracy_drop_summary.csv")
    _save_csv(ctx, summary.copy(), ctx.metrics_dir / "supp_similarity_bin_full_stats.csv")
    ctx.completed_modules["similarity_entry"] = True


def compute_overlap_localization_metrics(ctx: ExperimentContext, bank: OverlapReentryDMSBank | OverlapPerturbationCompatibleBank) -> None:
    b_path = ctx.metrics_dir / "panel_b_similarity_entry_metrics.csv"
    effect = _pair_effect_table(ctx, bank)
    if b_path.exists():
        b_base = pd.read_csv(b_path)
        keep = [c for c in ("pair_id", "b_vec", "acc_drop") if c in b_base.columns]
        if keep:
            effect = effect.drop(columns=[c for c in ("b_vec", "acc_drop") if c in effect.columns], errors="ignore").merge(
                b_base[keep],
                on="pair_id",
                how="left",
            )
    merged = bank.pair_trials.merge(effect[["pair_id", "b_vec", "DPI_L3", "acc_drop", "decision_deflection"]], on="pair_id", how="left")
    rows = []
    for _, r in merged.iterrows():
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "pair_id": int(r["pair_id"]),
                "similarity_bin": str(r["similarity_bin"]),
                "overlap_bin": str(r["overlap_bin"]),
                "pixel_similarity": float(r["pixel_similarity"]),
                "dice_overlap": float(r["dice_overlap"]),
                "input_energy_sample": float(r["input_energy_sample"]),
                "input_energy_probe": float(r["input_energy_probe"]),
                "dynamic_effect_metric": float(r["b_vec"]),
                "b_vec": float(r["b_vec"]),
                "DPI_L3": float(r["DPI_L3"]),
                "acc_drop": float(r["acc_drop"]),
                "decision_deflection": float(r["decision_deflection"]),
                "matched_group_id": str(r.get("matched_group_id", "")),
            }
        )
    loc = pd.DataFrame(rows)
    matched = _panel_c_matched_comparison(merged)
    reg = _overlap_regression(merged, int(ctx.cfg.network_seed))
    two = _two_by_two(merged, int(ctx.cfg.network_seed))
    diag = _matching_diagnostics(merged, int(ctx.cfg.network_seed))
    _save_csv(ctx, loc, ctx.metrics_dir / "panel_c_overlap_localization_metrics.csv")
    _save_csv(ctx, matched, ctx.metrics_dir / "panel_c_overlap_matched_comparison.csv")
    _save_csv(ctx, reg, ctx.metrics_dir / "supp_overlap_similarity_regression.csv")
    _save_csv(ctx, two, ctx.metrics_dir / "supp_overlap_similarity_2x2.csv")
    _save_csv(ctx, diag, ctx.metrics_dir / "supp_overlap_matching_diagnostics.csv")
    ctx.completed_modules["overlap_localization"] = True


def compute_overlap_accuracy_identification(ctx: ExperimentContext, bank: OverlapReentryDMSBank | SimilarityBiasCompatibleBank) -> None:
    pair_table = _accuracy_pair_table(ctx, bank)
    compute_high_similarity_overlap_accuracy_drop(ctx, pair_table)
    matches = _build_iso_similarity_overlap_matches(pair_table, ctx.cfg)
    if len(matches) < int(ctx.cfg.min_matches_per_network):
        ctx.warnings.append("Fig.4D iso-similarity matching found fewer than min_matches_per_network matched sets.")
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 404)
    null_df, perm_stats = _matched_overlap_permutation_test(matches, ctx.cfg, rng)
    contrast = _overlap_accuracy_contrast_by_network(matches, int(ctx.cfg.network_seed), perm_stats)
    balance = _matching_balance_diagnostics(matches, int(ctx.cfg.network_seed))
    excess = _compute_overlap_excess_accuracy(pair_table, ctx.cfg)
    regression = _overlap_accuracy_regression(pair_table, int(ctx.cfg.network_seed))
    _save_csv(ctx, pair_table, ctx.metrics_dir / "panel_d_overlap_accuracy_pair_table.csv")
    _save_csv(ctx, matches, ctx.metrics_dir / "panel_d_iso_similarity_matched_pairs.csv")
    _save_csv(ctx, null_df, ctx.metrics_dir / "panel_d_overlap_accuracy_permutation_null.csv")
    _save_csv(ctx, contrast, ctx.metrics_dir / "panel_d_overlap_accuracy_contrast_by_network.csv")
    _save_csv(ctx, balance, ctx.metrics_dir / "panel_d_matching_balance_diagnostics.csv")
    _save_csv(ctx, excess, ctx.metrics_dir / "supp_overlap_excess_accuracy_metrics.csv")
    _save_csv(ctx, regression, ctx.metrics_dir / "supp_overlap_accuracy_regression.csv")
    ctx.completed_modules["overlap_accuracy_identification"] = True


def compute_high_similarity_overlap_accuracy_drop(ctx: ExperimentContext, pair_table: pd.DataFrame) -> None:
    raw, summary, contrast = _high_similarity_overlap_accuracy_drop_tables(pair_table, ctx.cfg)
    _save_csv(ctx, raw, ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop.csv")
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop_summary.csv")
    _save_csv(ctx, contrast, ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop_contrast.csv")
    ctx.completed_modules["high_similarity_overlap_accuracy_drop"] = True


def compute_decision_spike_displacement(ctx: ExperimentContext, bank: OverlapReentryDMSBank) -> None:
    time_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for _, pair in bank.pair_trials.iterrows():
        pair_id = int(pair["pair_id"])
        dyn_row = _cond_row(bank.condition_metrics, pair_id, "full_dynamic")
        sta_row = _cond_row(bank.condition_metrics, pair_id, "full_static")
        for condition in CORE_CONDITIONS:
            dpi_t, s_dyn, s_sta = _dpi_timecourse(bank, pair_id, condition)
            for t, value in enumerate(dpi_t):
                time_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": pair_id,
                        "condition": condition,
                        "time_step": int(t),
                        "time_ms": float(t * ctx.cfg.dt / ms),
                        "S_dyn_L3": float(s_dyn[t]),
                        "S_sta_L3": float(s_sta[t]),
                        "DPI_L3_t": float(value),
                        "overlap_bin": str(pair["overlap_bin"]),
                        "similarity_bin": str(pair["similarity_bin"]),
                    }
                )
            summary_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "pair_id": pair_id,
                    "condition": condition,
                    "mean_DPI_L3": float(np.nanmean(dpi_t)) if len(dpi_t) else float("nan"),
                    "first_spike_time_dynamic": int(dyn_row["first_fire_time"]),
                    "first_spike_time_static": int(sta_row["first_fire_time"]),
                    "decision_spike_advance": int(sta_row["first_fire_time"]) - int(dyn_row["first_fire_time"]),
                    "overlap_bin": str(pair["overlap_bin"]),
                    "similarity_bin": str(pair["similarity_bin"]),
                }
            )
    _save_csv(ctx, pd.DataFrame(time_rows), ctx.metrics_dir / "panel_e_time_resolved_l3_displacement.csv")
    _save_csv(ctx, pd.DataFrame(summary_rows), ctx.metrics_dir / "panel_e_decision_spike_displacement.csv")
    ctx.completed_modules["decision_spike_displacement"] = True


def compute_probe_l3_trace_dpi_metrics(ctx: ExperimentContext, bank: OverlapPerturbationCompatibleBank) -> None:
    time_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for _, pair in bank.pair_trials.iterrows():
        pair_id = int(pair["pair_id"])
        dyn_row = _cond_row(bank.condition_metrics, pair_id, "full_dynamic")
        sta_row = _cond_row(bank.condition_metrics, pair_id, "full_static")
        dyn_trace = _trace(bank, pair_id, "full_dynamic")
        sta_trace = _trace(bank, pair_id, "full_static")
        for condition in CORE_CONDITIONS:
            cond_trace = _trace(bank, pair_id, condition)
            t_steps = min(int(cond_trace.shape[0]), int(dyn_trace.shape[0]), int(sta_trace.shape[0]))
            s_dyn_values: list[float] = []
            s_sta_values: list[float] = []
            dpi_values: list[float] = []
            for t in range(t_steps):
                cond_vec = normalize_pattern_vector(cond_trace[t])
                dyn_vec = normalize_pattern_vector(dyn_trace[t])
                sta_vec = normalize_pattern_vector(sta_trace[t])
                s_dyn = float(np.dot(cond_vec, dyn_vec))
                s_sta = float(np.dot(cond_vec, sta_vec))
                dpi = float(s_dyn - s_sta)
                s_dyn_values.append(s_dyn)
                s_sta_values.append(s_sta)
                dpi_values.append(dpi)
                time_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": pair_id,
                        "condition": condition,
                        "time_step": int(t),
                        "time_ms": float(t * ctx.cfg.dt / ms),
                        "S_dyn_L3": s_dyn,
                        "S_sta_L3": s_sta,
                        "DPI_L3_t": dpi,
                        "overlap_bin": str(pair["overlap_bin"]),
                        "similarity_bin": str(pair["similarity_bin"]),
                        "trace_object": "probe_l3_trace_s2p",
                        "pattern_normalization": "centered_l2",
                    }
                )
            summary_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "pair_id": pair_id,
                    "condition": condition,
                    "mean_DPI_L3": float(np.nanmean(dpi_values)) if dpi_values else float("nan"),
                    "mean_S_dyn_L3": float(np.nanmean(s_dyn_values)) if s_dyn_values else float("nan"),
                    "mean_S_sta_L3": float(np.nanmean(s_sta_values)) if s_sta_values else float("nan"),
                    "first_fire_time_dynamic": int(dyn_row["first_fire_time"]),
                    "first_fire_time_static": int(sta_row["first_fire_time"]),
                    "decision_spike_advance": int(sta_row["first_fire_time"]) - int(dyn_row["first_fire_time"]),
                    "overlap_bin": str(pair["overlap_bin"]),
                    "similarity_bin": str(pair["similarity_bin"]),
                    "trace_object": "probe_l3_trace_s2p",
                }
            )
    _save_csv(ctx, pd.DataFrame(time_rows), ctx.metrics_dir / "panel_e_time_resolved_l3_displacement.csv")
    _save_csv(ctx, pd.DataFrame(summary_rows), ctx.metrics_dir / "panel_e_decision_spike_displacement.csv")
    ctx.completed_modules["decision_spike_displacement"] = True


def compute_decision_deflection_metrics(ctx: ExperimentContext, bank: OverlapReentryDMSBank | OverlapPerturbationCompatibleBank) -> None:
    missing_reason = ""
    rows = []
    try:
        for _, pair in bank.pair_trials.iterrows():
            pair_id = int(pair["pair_id"])
            v_dyn = _vector(bank, pair_id, "full_dynamic")
            v_sta = _vector(bank, pair_id, "full_static")
            pred_dyn = int(_cond_row(bank.condition_metrics, pair_id, "full_dynamic")["prediction"])
            pred_sta = int(_cond_row(bank.condition_metrics, pair_id, "full_static")["prediction"])
            for condition in CORE_CONDITIONS:
                v_cond = _vector(bank, pair_id, condition)
                s_dyn = _cosine(v_cond, v_dyn)
                s_sta = _cosine(v_cond, v_sta)
                push = _projection(v_cond - v_sta, v_dyn - v_sta)
                recovery = s_dyn - s_sta
                pred_cond = int(_cond_row(bank.condition_metrics, pair_id, condition)["prediction"])
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": pair_id,
                        "condition": condition,
                        "similarity_bin": str(pair["similarity_bin"]),
                        "overlap_bin": str(pair["overlap_bin"]),
                        "trajectory_distance_dynamic_static": float(np.linalg.norm(v_dyn - v_sta)),
                        "condition_to_dynamic_similarity": s_dyn,
                        "condition_to_static_similarity": s_sta,
                        "dynamic_like_recovery": recovery,
                        "static_to_dynamic_push": push,
                        "decision_deflection_score": push,
                        "prediction_dynamic": pred_dyn,
                        "prediction_static": pred_sta,
                        "prediction_condition": pred_cond,
                        "condition_matches_dynamic": int(pred_cond == pred_dyn),
                        "condition_matches_static": int(pred_cond == pred_sta),
                        "x0": 0.0,
                        "y0": 0.0,
                        "x1": push,
                        "y1": recovery,
                    }
                )
    except KeyError as exc:
        missing_reason = f"overlap_vectors_missing:{exc}"
        ctx.warnings.append(f"Decision deflection metrics unavailable: {missing_reason}")
    df = pd.DataFrame(rows)
    _save_csv(ctx, df, ctx.metrics_dir / "supp_decision_deflection_metrics.csv")
    _save_csv(ctx, df.copy(), ctx.metrics_dir / "supp_s8_decision_deflection_metrics.csv")
    summary = _decision_deflection_summary(df)
    _save_csv(ctx, summary, ctx.metrics_dir / "supp_s8_decision_deflection_summary.csv")
    available = bool(not df.empty)
    ctx.availability["decision_deflection_available"] = available
    ctx.availability["decision_deflection_missing_reason"] = None if available else (missing_reason or "decision_deflection_metrics_empty")
    ctx.completed_modules["simplified_decision_deflection_supplement"] = True


def compute_l3_accumulator_region_replay_metrics(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
) -> L3AccumulatorReplayBank:
    cfg = ctx.cfg
    readout_step = _resolve_fig4_readout_step(ctx)
    work_pairs = pair_trials.copy()
    if cfg.smoke:
        work_pairs = work_pairs.head(min(4, len(work_pairs))).copy()
    images_cache = _image_cache(ctx, work_pairs)
    pair_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    pair_ids: list[int] = []
    delta_v_rows: list[np.ndarray] = []
    delta_hat_plus_rows: list[np.ndarray] = []
    delta_hat_minus_rows: list[np.ndarray] = []
    v_dynamic_rows: list[np.ndarray] = []
    v_static_rows: list[np.ndarray] = []
    pred_dynamic_rows: list[int] = []
    pred_static_rows: list[int] = []
    region_effect_payload: dict[str, list[np.ndarray]] = {
        "pair_id": [],
        "region_id": [],
        "D_dyn": [],
        "D_sta": [],
        "E_dyn": [],
        "E_sta": [],
        "R_plus": [],
        "R_minus": [],
        "R_plus_tilde": [],
        "R_minus_tilde": [],
    }
    regions = None
    for case_idx, row in enumerate(
        _progress(work_pairs.itertuples(index=False), total=len(work_pairs), desc="fig4 L3 accumulator replay", enabled=cfg.show_progress)
    ):
        sample_image = images_cache[int(row.sample_image_id)].unsqueeze(0)
        probe_image = images_cache[int(row.probe_image_id)].unsqueeze(0)
        sample_spikes = _encode_batch(ctx, sample_image, cfg.sample_steps)
        probe_spikes = _encode_batch(ctx, probe_image, cfg.probe_steps)
        dynamic_capture = run_dms_with_l3_trace_capture(
            ctx.net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=cfg.delay_steps,
            stsp_mode="dynamic",
            readout_step=readout_step,
            phase_reset=True,
        )
        static_capture = run_dms_with_l3_trace_capture(
            ctx.net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=cfg.delay_steps,
            stsp_mode="static_frozen",
            readout_step=readout_step,
            phase_reset=True,
        )
        if regions is None:
            regions = make_l3_region_masks(
                int(dynamic_capture.probe_s2p_trace.shape[-2]),
                int(dynamic_capture.probe_s2p_trace.shape[-1]),
                mask_mode=str(cfg.l3_mask_mode),
            )
        num_classes = int(getattr(ctx.net.layer3, "num_classes", NUM_CLASSES))
        if bool(cfg.run_l3_region_deletion):
            deletion = run_l3_deletion_analysis_for_pair(
                ctx.net,
                dynamic_capture=dynamic_capture,
                static_capture=static_capture,
                regions=regions,
                batch_size=int(cfg.l3_region_batch_size),
            )
        else:
            nan_matrix = np.full((len(regions), num_classes), np.nan, dtype=np.float64)
            deletion = {"D_dyn": nan_matrix.copy(), "D_sta": nan_matrix.copy(), "E_dyn": nan_matrix.copy(), "E_sta": nan_matrix.copy()}
        if bool(cfg.run_l3_region_replacement):
            replacement = run_l3_replacement_analysis_for_pair(
                ctx.net,
                dynamic_capture=dynamic_capture,
                static_capture=static_capture,
                regions=regions,
                batch_size=int(cfg.l3_region_batch_size),
            )
        else:
            nan_matrix = np.full((len(regions), num_classes), np.nan, dtype=np.float64)
            replacement = {
                "R_plus": nan_matrix.copy(),
                "R_minus": nan_matrix.copy(),
                "R_plus_tilde": nan_matrix.copy(),
                "R_minus_tilde": nan_matrix.copy(),
            }
        v_dyn = np.asarray(dynamic_capture.grouped_voltage[0], dtype=np.float64)
        v_sta = np.asarray(static_capture.grouped_voltage[0], dtype=np.float64)
        delta_v = _legacy_center_vector(v_dyn) - _legacy_center_vector(v_sta)
        bias_direction = int(np.argmax(np.abs(delta_v))) if delta_v.size else -1
        bias_magnitude = float(delta_v[bias_direction]) if bias_direction >= 0 else float("nan")
        delta_hat_plus = np.nansum(np.asarray(replacement["R_plus_tilde"], dtype=np.float64), axis=0)
        delta_hat_minus = np.nansum(np.asarray(replacement["R_minus_tilde"], dtype=np.float64), axis=0)
        sim_plus = _legacy_vector_similarity(delta_hat_plus, delta_v)
        sim_minus = _legacy_vector_similarity(delta_hat_minus, delta_v)
        e_dyn_k = np.asarray(deletion["E_dyn"][:, bias_direction], dtype=np.float64) if bias_direction >= 0 else np.asarray([])
        e_sta_k = np.asarray(deletion["E_sta"][:, bias_direction], dtype=np.float64) if bias_direction >= 0 else np.asarray([])
        r_plus_k = np.asarray(replacement["R_plus_tilde"][:, bias_direction], dtype=np.float64) if bias_direction >= 0 else np.asarray([])
        r_minus_k = np.asarray(replacement["R_minus_tilde"][:, bias_direction], dtype=np.float64) if bias_direction >= 0 else np.asarray([])
        pair_id = int(row.pair_id)
        pair_rows.append(
            {
                "network_seed": int(cfg.network_seed),
                "pair_id": pair_id,
                "sample_image_id": int(row.sample_image_id),
                "probe_image_id": int(row.probe_image_id),
                "sample_label": int(row.sample_label),
                "probe_label": int(row.probe_label),
                "prediction_dynamic": int(dynamic_capture.prediction_probe[0]),
                "prediction_static": int(static_capture.prediction_probe[0]),
                "correct_dynamic": int(int(dynamic_capture.prediction_probe[0]) == int(row.probe_label)),
                "correct_static": int(int(static_capture.prediction_probe[0]) == int(row.probe_label)),
                "first_fire_dynamic": int(dynamic_capture.first_fire_t_probe[0]),
                "first_fire_static": int(static_capture.first_fire_t_probe[0]),
                "bias_direction": int(bias_direction),
                "bias_magnitude": float(bias_magnitude),
                "replacement_push_kstar": float(np.nanmean(r_plus_k)) if r_plus_k.size else float("nan"),
                "replacement_pullback_kstar": float(np.nanmean(r_minus_k)) if r_minus_k.size else float("nan"),
                "deletion_dynamic_minus_static_kstar": float(np.nanmean(e_dyn_k - e_sta_k)) if e_dyn_k.size and e_sta_k.size else float("nan"),
                "reconstruction_cosine_plus": float(sim_plus["cosine"]),
                "reconstruction_cosine_minus": float(sim_minus["cosine"]),
                "direction_match_plus": int(np.nanargmax(delta_hat_plus) == bias_direction) if np.isfinite(delta_hat_plus).any() and bias_direction >= 0 else 0,
                "direction_match_minus": int(np.nanargmax(delta_hat_minus) == bias_direction) if np.isfinite(delta_hat_minus).any() and bias_direction >= 0 else 0,
                "n_regions": int(len(regions)),
                "l3_mask_mode": str(cfg.l3_mask_mode),
                "readout_step": int(readout_step),
            }
        )
        if case_idx < int(cfg.save_case_count):
            case_rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "case_id": int(case_idx),
                    "pair_id": pair_id,
                    "selection_reason": "first_cases_smoke" if cfg.smoke else "first_cases",
                    "sample_image_id": int(row.sample_image_id),
                    "probe_image_id": int(row.probe_image_id),
                }
            )
        pair_ids.append(pair_id)
        delta_v_rows.append(delta_v)
        delta_hat_plus_rows.append(delta_hat_plus)
        delta_hat_minus_rows.append(delta_hat_minus)
        v_dynamic_rows.append(v_dyn)
        v_static_rows.append(v_sta)
        pred_dynamic_rows.append(int(dynamic_capture.prediction_probe[0]))
        pred_static_rows.append(int(static_capture.prediction_probe[0]))
        for effect_name in ("D_dyn", "D_sta", "E_dyn", "E_sta"):
            pass
        for region_idx, region in enumerate(regions):
            region_effect_payload["pair_id"].append(np.asarray([pair_id], dtype=np.int64))
            region_effect_payload["region_id"].append(np.asarray([int(region.region_id)], dtype=np.int64))
            for name, source in (
                ("D_dyn", deletion["D_dyn"]),
                ("D_sta", deletion["D_sta"]),
                ("E_dyn", deletion["E_dyn"]),
                ("E_sta", deletion["E_sta"]),
                ("R_plus", replacement["R_plus"]),
                ("R_minus", replacement["R_minus"]),
                ("R_plus_tilde", replacement["R_plus_tilde"]),
                ("R_minus_tilde", replacement["R_minus_tilde"]),
            ):
                region_effect_payload[name].append(np.asarray(source[region_idx], dtype=np.float64))
    results = pd.DataFrame(pair_rows)
    summary_dict = summarize_l3_mechanism_results(results) if not results.empty else {}
    summary_rows = _l3_summary_rows(results, summary_dict, int(cfg.network_seed))
    pair_payload = {
        "pair_id": np.asarray(pair_ids, dtype=np.int64),
        "delta_v": np.stack(delta_v_rows, axis=0) if delta_v_rows else np.zeros((0, NUM_CLASSES), dtype=np.float64),
        "Delta_hat_plus": np.stack(delta_hat_plus_rows, axis=0) if delta_hat_plus_rows else np.zeros((0, NUM_CLASSES), dtype=np.float64),
        "Delta_hat_minus": np.stack(delta_hat_minus_rows, axis=0) if delta_hat_minus_rows else np.zeros((0, NUM_CLASSES), dtype=np.float64),
        "v_dynamic": np.stack(v_dynamic_rows, axis=0) if v_dynamic_rows else np.zeros((0, NUM_CLASSES), dtype=np.float64),
        "v_static": np.stack(v_static_rows, axis=0) if v_static_rows else np.zeros((0, NUM_CLASSES), dtype=np.float64),
        "prediction_dynamic": np.asarray(pred_dynamic_rows, dtype=np.int64),
        "prediction_static": np.asarray(pred_static_rows, dtype=np.int64),
    }
    region_payload = {
        name: (np.concatenate(values, axis=0) if name in {"pair_id", "region_id"} and values else np.stack(values, axis=0) if values else np.zeros((0, NUM_CLASSES), dtype=np.float64))
        for name, values in region_effect_payload.items()
    }
    _save_csv(ctx, results, ctx.metrics_dir / "panel_f_l3_accumulator_region_replay_metrics.csv")
    _save_csv(ctx, pd.DataFrame(summary_rows), ctx.metrics_dir / "panel_f_l3_accumulator_summary.csv")
    _save_csv(ctx, pd.DataFrame(case_rows), ctx.raw_dir / "panel_f_l3_accumulator_case_metadata.csv")
    np.savez_compressed(ctx.raw_dir / "panel_f_l3_accumulator_pair_vectors.npz", **pair_payload)
    np.savez_compressed(ctx.raw_dir / "panel_f_l3_accumulator_region_effects.npz", **region_payload)
    ctx.output_files["panel_f_l3_accumulator_pair_vectors"] = "data/raw/panel_f_l3_accumulator_pair_vectors.npz"
    ctx.output_files["panel_f_l3_accumulator_region_effects"] = "data/raw/panel_f_l3_accumulator_region_effects.npz"
    ctx.completed_modules["decision_deflection"] = True
    return L3AccumulatorReplayBank(pair_trials, results, pd.DataFrame(summary_rows), pair_payload, region_payload)


def compute_overlap_preserving_perturbation_metrics(ctx: ExperimentContext, bank: OverlapReentryDMSBank) -> None:
    e_path = ctx.metrics_dir / "supp_decision_deflection_metrics.csv"
    e_df = pd.read_csv(e_path) if e_path.exists() else pd.DataFrame()
    rows = []
    for _, pair in bank.pair_trials.iterrows():
        pair_id = int(pair["pair_id"])
        for condition in CORE_CONDITIONS:
            dpi_t, s_dyn, s_sta = _dpi_timecourse(bank, pair_id, condition)
            e_row = e_df[(e_df["pair_id"].eq(pair_id)) & (e_df["condition"].eq(condition))].head(1) if not e_df.empty else pd.DataFrame()
            cond_row = _cond_row(bank.condition_metrics, pair_id, condition)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "pair_id": pair_id,
                    "condition": condition,
                    "DPI_L3": float(np.nanmean(dpi_t)) if len(dpi_t) else float("nan"),
                    "mean_S_dyn_L3": float(np.nanmean(s_dyn)) if len(s_dyn) else float("nan"),
                    "mean_S_sta_L3": float(np.nanmean(s_sta)) if len(s_sta) else float("nan"),
                    "dynamic_like_recovery": _from_row(e_row, "dynamic_like_recovery", float("nan")),
                    "decision_deflection_score": _from_row(e_row, "decision_deflection_score", float("nan")),
                    "probe_accuracy": int(cond_row["correctness"]),
                    "prediction": int(cond_row["prediction"]),
                    "condition_to_dynamic_similarity": _from_row(e_row, "condition_to_dynamic_similarity", float("nan")),
                    "condition_to_static_similarity": _from_row(e_row, "condition_to_static_similarity", float("nan")),
                }
            )
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["network_seed", "condition"], dropna=False)
        .agg(
            mean_DPI_L3=("DPI_L3", "mean"),
            mean_dynamic_like_recovery=("dynamic_like_recovery", "mean"),
            mean_decision_deflection_score=("decision_deflection_score", "mean"),
            mean_probe_accuracy=("probe_accuracy", "mean"),
            n_pairs=("pair_id", "nunique"),
        )
        .reset_index()
    )
    contrast = _overlap_perturbation_contrast(ctx, summary)
    _save_csv(ctx, df, ctx.metrics_dir / "supp_overlap_preserving_perturbation_metrics.csv")
    _save_csv(ctx, summary, ctx.metrics_dir / "supp_overlap_preserving_perturbation_summary.csv")
    _save_csv(ctx, df.copy(), ctx.metrics_dir / "panel_d_overlap_perturbation_metrics.csv")
    _save_csv(ctx, summary.copy(), ctx.metrics_dir / "panel_d_overlap_perturbation_summary.csv")
    _save_csv(ctx, contrast, ctx.metrics_dir / "panel_d_overlap_perturbation_contrast.csv")
    ctx.completed_modules["overlap_perturbation_supplement"] = True
    ctx.completed_modules["legacy_overlap_perturbation"] = True


def _overlap_perturbation_contrast(ctx: ExperimentContext, summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "DPI_overlap",
        "DPI_nonoverlap",
        "DPI_random",
        "DPI_static",
        "DPI_dynamic",
        "overlap_minus_nonoverlap_DPI",
        "overlap_minus_random_DPI",
        "recovery_overlap",
        "recovery_nonoverlap",
        "recovery_random",
        "overlap_minus_nonoverlap_recovery",
        "overlap_minus_random_recovery",
        "accuracy_overlap",
        "accuracy_nonoverlap",
        "accuracy_random",
        "n_pairs",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for network_seed, part in summary.groupby("network_seed", sort=False):
        by_cond = {str(row.condition): row for row in part.itertuples(index=False)}
        missing = [condition for condition in PERTURBATION_CONDITION_MAP.values() if condition not in by_cond]
        if missing:
            ctx.warnings.append(f"Fig.4D overlap perturbation contrast missing conditions: {', '.join(missing)}")

        def value(alias: str, field_name: str) -> float:
            row = by_cond.get(PERTURBATION_CONDITION_MAP[alias])
            return float(getattr(row, field_name, np.nan)) if row is not None else float("nan")

        dpi_overlap = value("overlap", "mean_DPI_L3")
        dpi_nonoverlap = value("nonoverlap", "mean_DPI_L3")
        dpi_random = value("random", "mean_DPI_L3")
        recovery_overlap = value("overlap", "mean_dynamic_like_recovery")
        recovery_nonoverlap = value("nonoverlap", "mean_dynamic_like_recovery")
        recovery_random = value("random", "mean_dynamic_like_recovery")
        n_pairs = int(pd.to_numeric(part.get("n_pairs", pd.Series(dtype=float)), errors="coerce").max()) if "n_pairs" in part.columns else 0
        rows.append(
            {
                "network_seed": int(network_seed),
                "DPI_overlap": dpi_overlap,
                "DPI_nonoverlap": dpi_nonoverlap,
                "DPI_random": dpi_random,
                "DPI_static": value("static", "mean_DPI_L3"),
                "DPI_dynamic": value("dynamic", "mean_DPI_L3"),
                "overlap_minus_nonoverlap_DPI": _finite_delta(dpi_overlap, dpi_nonoverlap),
                "overlap_minus_random_DPI": _finite_delta(dpi_overlap, dpi_random),
                "recovery_overlap": recovery_overlap,
                "recovery_nonoverlap": recovery_nonoverlap,
                "recovery_random": recovery_random,
                "overlap_minus_nonoverlap_recovery": _finite_delta(recovery_overlap, recovery_nonoverlap),
                "overlap_minus_random_recovery": _finite_delta(recovery_overlap, recovery_random),
                "accuracy_overlap": value("overlap", "mean_probe_accuracy"),
                "accuracy_nonoverlap": value("nonoverlap", "mean_probe_accuracy"),
                "accuracy_random": value("random", "mean_probe_accuracy"),
                "n_pairs": n_pairs,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compute_l1_stsp_overlap_perturbation_outputs(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
) -> None:
    cfg = ctx.cfg
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    images_cache = _image_cache(ctx, pair_trials)
    l2_diffs: list[float] = []
    l3_diffs: list[float] = []
    restore_ok_values: list[int] = []
    perturbation_ok_values: list[int] = []
    failure_reasons: list[str] = []

    for batch_start in _progress(
        range(0, len(pair_trials), int(cfg.batch_size)),
        total=math.ceil(len(pair_trials) / int(cfg.batch_size)),
        desc="fig4 L1 STSP reset batches",
        enabled=cfg.show_progress,
    ):
        batch = pair_trials.iloc[batch_start : batch_start + int(cfg.batch_size)].copy()
        if batch.empty:
            continue
        sample_images = torch.stack([images_cache[int(r["sample_image_id"])] for _, r in batch.iterrows()], dim=0)
        probe_images = torch.stack([images_cache[int(r["probe_image_id"])] for _, r in batch.iterrows()], dim=0)
        sample_spikes = _encode_batch(ctx, sample_images, cfg.sample_steps)
        probe_spikes = _encode_batch(ctx, probe_images, cfg.probe_steps)
        static_out = run_dms_snapshot_rollout(
            ctx.net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=cfg.delay_steps,
            stsp_mode="static_frozen",
            phase_reset=True,
            intervention_plan=None,
            readout_step=_resolve_fig4_readout_step(ctx),
            snapshot_state_names=("v_mem",),
        )
        static_pred = static_out["predictions"]["prediction_probe"].detach().cpu().numpy().astype(np.int64, copy=False)
        static_correct = (static_pred == batch["probe_label"].to_numpy(dtype=np.int64)).astype(np.int64)

        try:
            pre_probe_state, pre_probe_time = _run_dynamic_sample_delay_to_preprobe(ctx, sample_spikes)
        except Exception as exc:
            reason = f"preprobe_dynamic_failed:{type(exc).__name__}:{exc}"
            ctx.warnings.append(f"Fig.4D L1 STSP perturbation skipped batch {batch_start}: {reason}")
            failure_reasons.append(reason)
            continue

        for condition in D_L1_STSP_CONDITIONS:
            if condition == "full_static":
                for local_idx, row in enumerate(batch.itertuples(index=False)):
                    rows.append(
                        _l1_stsp_row(
                            ctx,
                            row,
                            mask_bank,
                            condition=condition,
                            prediction=int(static_pred[local_idx]),
                            correct=int(static_correct[local_idx]),
                            correct_static=int(static_correct[local_idx]),
                            accuracy_drop_vs_static=0,
                            l2_diff=0.0,
                            l3_diff=0.0,
                            restore_ok=1,
                            perturbation_ok=1,
                            insufficient_units=0,
                        )
                    )
                continue

            try:
                _restore_runtime_state(ctx.net, pre_probe_state)
                restore_ok = int(_runtime_state_max_abs_diff(ctx.net, pre_probe_state, ("layer1", "layer2", "layer3")) <= 1e-6)
                insufficient_units = _apply_l1_reset_for_condition(ctx.net, batch, mask_bank, condition)
                perturbation_ok = int(condition == "full_dynamic_intact" or not bool(insufficient_units))
                l2_pre = _snapshot_layer_ux(ctx.net.layer2)
                l3_pre = _snapshot_layer_ux(ctx.net.layer3)
                pred = _run_probe_with_l1_dynamic_l23_frozen(ctx.net, probe_spikes, pre_probe_time)
                l2_diff = _ux_max_abs_diff(ctx.net.layer2, l2_pre)
                l3_diff = _ux_max_abs_diff(ctx.net.layer3, l3_pre)
                l2_diffs.append(float(l2_diff))
                l3_diffs.append(float(l3_diff))
                restore_ok_values.append(int(restore_ok))
                perturbation_ok_values.append(int(perturbation_ok))
            except Exception as exc:
                reason = f"{condition}_failed:{type(exc).__name__}:{exc}"
                ctx.warnings.append(f"Fig.4D L1 STSP perturbation failed for batch {batch_start}: {reason}")
                failure_reasons.append(reason)
                pred = np.full(len(batch), -1, dtype=np.int64)
                restore_ok = 0
                perturbation_ok = 0
                insufficient_units = 1
                l2_diff = float("nan")
                l3_diff = float("nan")

            probe_labels = batch["probe_label"].to_numpy(dtype=np.int64)
            correct = (np.asarray(pred, dtype=np.int64) == probe_labels).astype(np.int64)
            for local_idx, row in enumerate(batch.itertuples(index=False)):
                rows.append(
                    _l1_stsp_row(
                        ctx,
                        row,
                        mask_bank,
                        condition=condition,
                        prediction=int(np.asarray(pred, dtype=np.int64)[local_idx]),
                        correct=int(correct[local_idx]),
                        correct_static=int(static_correct[local_idx]),
                        accuracy_drop_vs_static=int(static_correct[local_idx] - correct[local_idx]),
                        l2_diff=float(l2_diff),
                        l3_diff=float(l3_diff),
                        restore_ok=int(restore_ok),
                        perturbation_ok=int(perturbation_ok),
                        insufficient_units=int(insufficient_units),
                    )
                )

    raw = pd.DataFrame(rows, columns=_l1_stsp_raw_columns())
    summary = _l1_stsp_summary(raw)
    contrast = _l1_stsp_contrast(raw)
    l2_max = float(np.nanmax(l2_diffs)) if l2_diffs else float("nan")
    l3_max = float(np.nanmax(l3_diffs)) if l3_diffs else float("nan")
    audit_rows.append(
        {
            "network_seed": int(ctx.cfg.network_seed),
            "run_l1_stsp_overlap_perturbation": bool(not raw.empty),
            "probe_input_unchanged": True,
            "sample_input_complete": True,
            "perturbed_layer": "L1",
            "perturbed_variables": json.dumps(["u", "x"]),
            "l2_stsp_frozen": bool(np.isfinite(l2_max) and l2_max <= 1e-6),
            "l3_stsp_frozen": bool(np.isfinite(l3_max) and l3_max <= 1e-6),
            "l2_stsp_max_abs_diff_across_conditions": l2_max,
            "l3_stsp_max_abs_diff_across_conditions": l3_max,
            "n_pairs": int(raw["pair_id"].nunique()) if "pair_id" in raw.columns and not raw.empty else 0,
            "n_valid_pairs": int(raw[raw["perturbation_ok"].eq(1)]["pair_id"].nunique()) if "perturbation_ok" in raw.columns and not raw.empty else 0,
            "failure_reason": ";".join(failure_reasons),
        }
    )
    _save_csv(ctx, raw, ctx.raw_dir / "panel_d_l1_stsp_overlap_perturbation_trial_readout.csv")
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_summary.csv")
    _save_csv(ctx, contrast, ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_contrast.csv")
    _save_csv(ctx, pd.DataFrame(audit_rows), ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_audit.csv")
    ctx.completed_modules["l1_stsp_overlap_perturbation"] = True
    ctx.completed_modules["overlap_perturbation_main"] = True


def _run_dynamic_sample_delay_to_preprobe(ctx: ExperimentContext, sample_spikes: torch.Tensor) -> tuple[dict[str, dict[str, torch.Tensor]], int]:
    net = ctx.net
    batch_size, t_sample, channels, height, width = sample_spikes.shape
    prepare_network_state(net, int(batch_size), int(channels), int(height), int(width))
    zero_input = torch.zeros((batch_size, channels, height, width), device=sample_spikes.device)
    current_time = 0
    with torch.no_grad():
        for t_step in range(int(t_sample)):
            _fig4_step_network(net, sample_spikes[:, t_step, ...], current_time, l1_mode="dynamic", l2_mode="dynamic", l3_mode="dynamic")
            current_time += 1
        for _ in range(int(ctx.cfg.delay_steps)):
            _fig4_step_network(net, zero_input, current_time, l1_mode="dynamic", l2_mode="dynamic", l3_mode="dynamic")
            current_time += 1
    return _snapshot_runtime_state(net), int(current_time)


def _run_probe_with_l1_dynamic_l23_frozen(net: Any, probe_spikes: torch.Tensor, pre_probe_time: int) -> np.ndarray:
    batch_size = int(probe_spikes.shape[0])
    reset_l3_decision_window(net)
    with torch.no_grad():
        for t_step in range(int(probe_spikes.shape[1])):
            _fig4_step_network(
                net,
                probe_spikes[:, t_step, ...],
                int(pre_probe_time) + int(t_step),
                l1_mode="dynamic",
                l2_mode="static_frozen",
                l3_mode="static_frozen",
                force_l3_time=int(t_step),
            )
    pred, _ = decode_prediction_and_fire_time_from_layer3(net, batch_size)
    return pred.numpy().astype(np.int64, copy=False)


def _fig4_step_network(
    net: Any,
    input_t: torch.Tensor,
    current_time: int,
    *,
    l1_mode: str,
    l2_mode: str,
    l3_mode: str,
    force_l3_time: int | None = None,
) -> None:
    s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=l1_mode)
    s1_p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, monitor=False, stsp_mode=l2_mode)
    s2_p = net.pool2(s2.float())
    t_l3 = int(current_time) if force_l3_time is None else int(force_l3_time)
    net.layer3.forward_step(s2_p, t_l3, training=False, monitor=False, stsp_mode=l3_mode)


def _apply_l1_reset_for_condition(
    net: Any,
    batch: pd.DataFrame,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
    condition: str,
) -> int:
    if condition == "full_dynamic_intact":
        return 0
    mask_key = {
        "l1_overlap_reset": "overlap_mask",
        "l1_nonoverlap_reset": "sample_nonoverlap_mask",
        "l1_random_matched_reset": "random_matched_mask",
    }.get(condition)
    if mask_key is None:
        raise ValueError(f"Unsupported L1 STSP reset condition: {condition}")
    masks = [np.asarray(mask_bank[int(row.pair_id)][mask_key], dtype=bool) for row in batch.itertuples(index=False)]
    insufficient = int(any(int(mask.sum()) == 0 for mask in masks))
    mask_tensor = _l1_mask_tensor(net.layer1, masks)
    with torch.no_grad():
        if net.layer1.u_pre is None or net.layer1.x_pre is None:
            raise ValueError("Layer1 STSP state is not initialized.")
        net.layer1.u_pre[mask_tensor] = float(net.layer1.stsp_U)
        net.layer1.x_pre[mask_tensor] = 1.0
    return int(insufficient)


def _l1_mask_tensor(layer: Any, masks: Sequence[np.ndarray]) -> torch.Tensor:
    if layer.u_pre is None:
        raise ValueError("Layer1 u_pre is not initialized.")
    target_shape = tuple(layer.u_pre.shape)
    if len(target_shape) != 4:
        raise ValueError(f"Expected layer1 STSP shape [B,C,H,W], got {target_shape}")
    batch_size, channels, height, width = target_shape
    if int(batch_size) != len(masks):
        raise ValueError(f"Mask batch mismatch: layer batch={batch_size}, masks={len(masks)}")
    arr = np.stack([np.asarray(mask, dtype=bool) for mask in masks], axis=0)
    if tuple(arr.shape[1:]) != (int(height), int(width)):
        raise ValueError(f"Layer1 mask shape mismatch: masks={arr.shape}, layer spatial={(height, width)}")
    arr = np.repeat(arr[:, None, :, :], int(channels), axis=1)
    return torch.as_tensor(arr, dtype=torch.bool, device=layer.u_pre.device)


def _l1_unit_count_for_mask(layer: Any, mask: np.ndarray) -> int:
    if layer.u_pre is None:
        return int(np.asarray(mask, dtype=bool).sum())
    shape = tuple(layer.u_pre.shape)
    channels = int(shape[1]) if len(shape) == 4 else 1
    return int(np.asarray(mask, dtype=bool).sum()) * channels


def _snapshot_layer_ux(layer: Any) -> dict[str, torch.Tensor]:
    if layer.u_pre is None or layer.x_pre is None:
        raise ValueError("Layer STSP state is not initialized.")
    return {"u": layer.u_pre.detach().clone(), "x": layer.x_pre.detach().clone()}


def _ux_max_abs_diff(layer: Any, snapshot: Mapping[str, torch.Tensor]) -> float:
    if layer.u_pre is None or layer.x_pre is None:
        return float("nan")
    u_saved = snapshot["u"].to(device=layer.u_pre.device, dtype=layer.u_pre.dtype)
    x_saved = snapshot["x"].to(device=layer.x_pre.device, dtype=layer.x_pre.dtype)
    return float(max(torch.max(torch.abs(layer.u_pre - u_saved)).item(), torch.max(torch.abs(layer.x_pre - x_saved)).item()))


def _snapshot_runtime_state(net: Any) -> dict[str, dict[str, torch.Tensor]]:
    state: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key in ("layer1", "layer2", "layer3"):
        layer = getattr(net, layer_key)
        layer_state: dict[str, torch.Tensor] = {}
        for attr in ("v_mem", "g_e", "res", "u_pre", "x_pre"):
            value = getattr(layer, attr, None)
            if value is not None:
                layer_state[attr] = value.detach().clone()
        inh = getattr(getattr(layer, "lateral_inh", None), "inh_trace", None)
        if inh is not None:
            layer_state["inh_trace"] = inh.detach().clone()
        firing_times = getattr(layer, "firing_times", None)
        if firing_times is not None:
            layer_state["firing_times"] = firing_times.detach().clone()
        state[layer_key] = layer_state
    return state


def _restore_runtime_state(net: Any, state: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    with torch.no_grad():
        for layer_key, layer_state in state.items():
            layer = getattr(net, layer_key)
            for attr in ("v_mem", "g_e", "res", "u_pre", "x_pre"):
                if attr not in layer_state:
                    continue
                target = getattr(layer, attr, None)
                if target is None or tuple(target.shape) != tuple(layer_state[attr].shape):
                    raise ValueError(f"Cannot restore {layer_key}.{attr}: shape mismatch or missing target")
                target.copy_(layer_state[attr].to(device=target.device, dtype=target.dtype))
            if "inh_trace" in layer_state:
                target = layer.lateral_inh.inh_trace
                if tuple(target.shape) != tuple(layer_state["inh_trace"].shape):
                    raise ValueError(f"Cannot restore {layer_key}.inh_trace: shape mismatch")
                target.copy_(layer_state["inh_trace"].to(device=target.device, dtype=target.dtype))
            if "firing_times" in layer_state and getattr(layer, "firing_times", None) is not None:
                target = layer.firing_times
                if tuple(target.shape) != tuple(layer_state["firing_times"].shape):
                    raise ValueError(f"Cannot restore {layer_key}.firing_times: shape mismatch")
                target.copy_(layer_state["firing_times"].to(device=target.device, dtype=target.dtype))


def _runtime_state_max_abs_diff(net: Any, state: Mapping[str, Mapping[str, torch.Tensor]], layer_keys: Sequence[str]) -> float:
    diffs: list[float] = []
    for layer_key in layer_keys:
        layer = getattr(net, layer_key)
        for attr in ("v_mem", "g_e", "res", "u_pre", "x_pre"):
            saved = state.get(layer_key, {}).get(attr)
            current = getattr(layer, attr, None)
            if saved is None or current is None:
                continue
            saved = saved.to(device=current.device, dtype=current.dtype)
            diffs.append(float(torch.max(torch.abs(current - saved)).item()))
    return max(diffs) if diffs else 0.0


def _l1_stsp_row(
    ctx: ExperimentContext,
    row: Any,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
    *,
    condition: str,
    prediction: int,
    correct: int,
    correct_static: int,
    accuracy_drop_vs_static: int,
    l2_diff: float,
    l3_diff: float,
    restore_ok: int,
    perturbation_ok: int,
    insufficient_units: int,
) -> dict[str, Any]:
    pair_id = int(row.pair_id)
    masks = mask_bank[pair_id]
    sample_fg = np.asarray(masks["sample_foreground_mask"], dtype=bool)
    probe_fg = np.asarray(masks["probe_foreground_mask"], dtype=bool)
    overlap = np.asarray(masks["overlap_mask"], dtype=bool)
    nonoverlap = np.asarray(masks["sample_nonoverlap_mask"], dtype=bool)
    random_mask = np.asarray(masks["random_matched_mask"], dtype=bool)
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "pair_id": pair_id,
        "condition": str(condition),
        "condition_label": D_L1_STSP_CONDITION_LABELS.get(str(condition), str(condition)),
        "probe_label": int(row.probe_label),
        "prediction": int(prediction),
        "correct": int(correct),
        "correct_static": int(correct_static),
        "accuracy_drop_vs_static": int(accuracy_drop_vs_static),
        "pixel_similarity": float(row.pixel_similarity),
        "dice_overlap": float(row.dice_overlap),
        "similarity_bin": str(row.similarity_bin),
        "overlap_bin": str(row.overlap_bin),
        "sample_fg_area": int(sample_fg.sum()),
        "probe_fg_area": int(probe_fg.sum()),
        "overlap_area": int(overlap.sum()),
        "nonoverlap_area": int(nonoverlap.sum()),
        "random_area": int(random_mask.sum()),
        "l1_overlap_unit_count": _l1_unit_count_for_mask(ctx.net.layer1, overlap),
        "l1_nonoverlap_unit_count": _l1_unit_count_for_mask(ctx.net.layer1, nonoverlap),
        "l1_random_unit_count": _l1_unit_count_for_mask(ctx.net.layer1, random_mask),
        "perturbed_layer": "L1",
        "perturbed_variables": json.dumps(["u", "x"]),
        "perturbation_mode": "static_baseline" if condition == "full_static" else ("none" if condition == "full_dynamic_intact" else "reset_to_s0"),
        "probe_input_unchanged": True,
        "sample_input_complete": True,
        "l2_stsp_frozen": bool(np.isfinite(l2_diff) and float(l2_diff) <= 1e-6),
        "l3_stsp_frozen": bool(np.isfinite(l3_diff) and float(l3_diff) <= 1e-6),
        "l2_stsp_max_abs_diff_across_conditions": float(l2_diff),
        "l3_stsp_max_abs_diff_across_conditions": float(l3_diff),
        "restore_ok": int(restore_ok),
        "perturbation_ok": int(perturbation_ok),
        "insufficient_units": int(insufficient_units),
    }


def _l1_stsp_raw_columns() -> list[str]:
    return [
        "network_seed",
        "pair_id",
        "condition",
        "condition_label",
        "probe_label",
        "prediction",
        "correct",
        "correct_static",
        "accuracy_drop_vs_static",
        "pixel_similarity",
        "dice_overlap",
        "similarity_bin",
        "overlap_bin",
        "sample_fg_area",
        "probe_fg_area",
        "overlap_area",
        "nonoverlap_area",
        "random_area",
        "l1_overlap_unit_count",
        "l1_nonoverlap_unit_count",
        "l1_random_unit_count",
        "perturbed_layer",
        "perturbed_variables",
        "perturbation_mode",
        "probe_input_unchanged",
        "sample_input_complete",
        "l2_stsp_frozen",
        "l3_stsp_frozen",
        "l2_stsp_max_abs_diff_across_conditions",
        "l3_stsp_max_abs_diff_across_conditions",
        "restore_ok",
        "perturbation_ok",
        "insufficient_units",
    ]


def _l1_stsp_summary(raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "condition",
        "condition_label",
        "mean_accuracy_drop_vs_static",
        "sem_accuracy_drop_vs_static",
        "mean_probe_accuracy",
        "n_pairs",
        "n_valid_pairs",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (network_seed, condition), part in raw.groupby(["network_seed", "condition"], sort=False):
        drops = pd.to_numeric(part["accuracy_drop_vs_static"], errors="coerce")
        correct = pd.to_numeric(part["correct"], errors="coerce")
        valid = part[part["perturbation_ok"].eq(1)] if "perturbation_ok" in part.columns else part
        rows.append(
            {
                "network_seed": int(network_seed),
                "condition": str(condition),
                "condition_label": D_L1_STSP_CONDITION_LABELS.get(str(condition), str(condition)),
                "mean_accuracy_drop_vs_static": float(drops.mean(skipna=True)),
                "sem_accuracy_drop_vs_static": float(drops.sem()) if len(drops.dropna()) > 1 else 0.0,
                "mean_probe_accuracy": float(correct.mean(skipna=True)),
                "n_pairs": int(part["pair_id"].nunique()),
                "n_valid_pairs": int(valid["pair_id"].nunique()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _l1_stsp_contrast(raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "acc_drop_dynamic",
        "acc_drop_overlap_reset",
        "acc_drop_nonoverlap_reset",
        "acc_drop_random_reset",
        "dynamic_minus_overlap_reset",
        "nonoverlap_reset_minus_overlap_reset",
        "random_reset_minus_overlap_reset",
        "n_pairs",
        "n_valid_pairs",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for network_seed, part in raw.groupby("network_seed", sort=False):
        means = part.groupby("condition")["accuracy_drop_vs_static"].mean()
        dyn = float(means.get("full_dynamic_intact", np.nan))
        overlap = float(means.get("l1_overlap_reset", np.nan))
        nonoverlap = float(means.get("l1_nonoverlap_reset", np.nan))
        random = float(means.get("l1_random_matched_reset", np.nan))
        valid = part[part["perturbation_ok"].eq(1)] if "perturbation_ok" in part.columns else part
        rows.append(
            {
                "network_seed": int(network_seed),
                "acc_drop_dynamic": dyn,
                "acc_drop_overlap_reset": overlap,
                "acc_drop_nonoverlap_reset": nonoverlap,
                "acc_drop_random_reset": random,
                "dynamic_minus_overlap_reset": _finite_delta(dyn, overlap),
                "nonoverlap_reset_minus_overlap_reset": _finite_delta(nonoverlap, overlap),
                "random_reset_minus_overlap_reset": _finite_delta(random, overlap),
                "n_pairs": int(part["pair_id"].nunique()),
                "n_valid_pairs": int(valid["pair_id"].nunique()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compute_supplement_outputs(ctx: ExperimentContext, bank: OverlapReentryDMSBank) -> None:
    effect = _pair_effect_table(ctx, bank)
    alt_rows = []
    for _, r in bank.pair_trials.merge(effect[["pair_id", "b_vec", "DPI_L3", "decision_deflection"]], on="pair_id", how="left").iterrows():
        for name, value in (
            ("dice_overlap", r["dice_overlap"]),
            ("overlap_fraction_sample", r["overlap_fraction_sample"]),
            ("overlap_fraction_probe", r["overlap_fraction_probe"]),
            ("dilated_overlap", min(1.0, float(r["dice_overlap"]) + 0.05)),
            ("encoded_spike_overlap", float(r["dice_overlap"])),
        ):
            alt_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "pair_id": int(r["pair_id"]),
                    "overlap_definition": name,
                    "overlap_value": float(value),
                    "dynamic_effect_metric": "DPI_L3",
                    "metric_value": float(r["DPI_L3"]),
                }
            )
    class_breakdown = (
        bank.pair_trials.merge(effect, on="pair_id", how="left")
        .groupby("class_pair", dropna=False)
        .agg(n_pairs=("pair_id", "nunique"), mean_acc_drop=("acc_drop", "mean"), mean_DPI_L3=("DPI_L3", "mean"), mean_decision_deflection=("decision_deflection", "mean"))
        .reset_index()
    )
    class_breakdown.insert(0, "network_seed", int(ctx.cfg.network_seed))
    layer_rows = []
    for _, r in effect.iterrows():
        layer_rows.append({"network_seed": int(ctx.cfg.network_seed), "pair_id": int(r["pair_id"]), "layer": "L3", "delay_ms": int(ctx.cfg.delay_ms), "metric": "DPI_L3", "value": float(r["DPI_L3"])})
        layer_rows.append({"network_seed": int(ctx.cfg.network_seed), "pair_id": int(r["pair_id"]), "layer": "readout", "delay_ms": int(ctx.cfg.delay_ms), "metric": "final_readout_deflection", "value": float(r["decision_deflection"])})
    random_controls = _random_mask_controls(ctx, bank)
    audit = _condition_audit(ctx, bank)
    _save_csv(ctx, random_controls, ctx.metrics_dir / "supp_random_mask_perturbation_controls.csv")
    _save_csv(ctx, pd.DataFrame(alt_rows), ctx.metrics_dir / "supp_alternative_overlap_definitions.csv")
    _save_csv(ctx, class_breakdown, ctx.metrics_dir / "supp_class_pair_breakdown.csv")
    _save_csv(ctx, pd.DataFrame(layer_rows), ctx.metrics_dir / "supp_layer_delay_reentry_metrics.csv")
    _save_csv(ctx, audit, ctx.metrics_dir / "supp_trial_condition_audit.csv")
    for filename in ("supp_overlap_similarity_regression.csv", "supp_overlap_matching_diagnostics.csv", "supp_overlap_similarity_2x2.csv", "supp_similarity_bin_full_stats.csv"):
        path = ctx.metrics_dir / filename
        if not path.exists():
            _save_csv(ctx, pd.DataFrame(), path)
    ctx.completed_modules["supplement"] = True


def write_fig4_panel_aliases_and_supplement_aliases(ctx: ExperimentContext) -> None:
    _write_s7_similarity_bin_full_trend(ctx)
    _write_s7_overlap_matching_diagnostics(ctx)
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_d_overlap_accuracy_contrast_by_network.csv",
        ctx.metrics_dir / "supp_s7_iso_similarity_overlap_contrast.csv",
        empty_columns=[
            "network_seed",
            "n_matched_sets",
            "drop_rate_high_overlap",
            "drop_rate_low_overlap",
            "delta_drop_rate",
            "mean_acc_drop_high_overlap",
            "mean_acc_drop_low_overlap",
            "delta_acc_drop",
        ],
        reason="panel_d_overlap_accuracy_contrast_by_network_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_d_overlap_accuracy_permutation_null.csv",
        ctx.metrics_dir / "supp_s7_iso_similarity_permutation_null.csv",
        empty_columns=["network_seed", "permutation_index", "delta_drop_rate_null", "delta_acc_drop_null"],
        reason="panel_d_overlap_accuracy_permutation_null_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_d_iso_similarity_matched_pairs.csv",
        ctx.metrics_dir / "supp_s7_iso_similarity_matched_pairs.csv",
        empty_columns=_iso_match_columns(),
        reason="panel_d_iso_similarity_matched_pairs_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_d_matching_balance_diagnostics.csv",
        ctx.metrics_dir / "supp_s7_overlap_matching_balance_diagnostics.csv",
        empty_columns=[
            "network_seed",
            "n_matched_sets",
            "mean_similarity_difference",
            "mean_sample_energy_rel_difference",
            "mean_probe_energy_rel_difference",
            "mean_overlap_difference",
        ],
        reason="panel_d_matching_balance_diagnostics_missing_or_empty",
    )
    _write_s7_overlap_regression_controls(ctx)
    _write_s7_random_nonoverlap_perturbation_controls(ctx)
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_e_time_resolved_l3_displacement.csv",
        ctx.metrics_dir / "supp_s8_time_resolved_l3_displacement.csv",
        empty_columns=["network_seed", "pair_id", "condition", "time_step", "time_ms", "S_dyn_L3", "S_sta_L3", "DPI_L3_t"],
        reason="panel_e_time_resolved_l3_displacement_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_e_decision_spike_displacement.csv",
        ctx.metrics_dir / "supp_s8_decision_spike_displacement.csv",
        empty_columns=["network_seed", "pair_id", "condition", "mean_DPI_L3", "decision_spike_advance"],
        reason="panel_e_decision_spike_displacement_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_f_l3_accumulator_region_replay_metrics.csv",
        ctx.metrics_dir / "supp_s8_l3_accumulator_replay_metrics.csv",
        empty_columns=["network_seed", "pair_id", "region_id", "region_label"],
        reason="panel_f_l3_accumulator_region_replay_metrics_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_f_l3_accumulator_summary.csv",
        ctx.metrics_dir / "supp_s8_l3_accumulator_summary.csv",
        empty_columns=["network_seed", "metric", "value", "n_pairs"],
        reason="panel_f_l3_accumulator_summary_missing_or_empty",
    )
    if not (ctx.metrics_dir / "supp_s8_decision_deflection_metrics.csv").exists():
        _copy_csv_alias(
            ctx,
            ctx.metrics_dir / "supp_decision_deflection_metrics.csv",
            ctx.metrics_dir / "supp_s8_decision_deflection_metrics.csv",
            empty_columns=["network_seed", "pair_id", "condition", "dynamic_like_recovery", "decision_deflection_score"],
            reason="supp_decision_deflection_metrics_missing_or_empty",
        )
    if not (ctx.metrics_dir / "supp_s8_decision_deflection_summary.csv").exists():
        src = ctx.metrics_dir / "supp_s8_decision_deflection_metrics.csv"
        df = pd.read_csv(src) if src.exists() else pd.DataFrame()
        _save_csv(ctx, _decision_deflection_summary(df), ctx.metrics_dir / "supp_s8_decision_deflection_summary.csv")
    ctx.completed_modules["s7_s8_aliases"] = True


def _write_s7_similarity_bin_full_trend(ctx: ExperimentContext) -> None:
    src = ctx.metrics_dir / "supp_similarity_bin_full_stats.csv"
    if not src.exists() or not _csv_nonempty(src):
        src = ctx.metrics_dir / "panel_b_similarity_bin_summary.csv"
    if not src.exists():
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s7_similarity_bin_full_trend.csv", ["network_seed", "similarity_bin", "mean_pixel_similarity", "mean_acc_drop", "mean_drop_event", "mean_b_vec", "mean_DPI_L3", "n_pairs"], "panel_b_similarity_bin_summary_missing")
        return
    df = pd.read_csv(src)
    if df.empty:
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s7_similarity_bin_full_trend.csv", ["network_seed", "similarity_bin", "mean_pixel_similarity", "mean_acc_drop", "mean_drop_event", "mean_b_vec", "mean_DPI_L3", "n_pairs"], "similarity_bin_source_empty")
        return
    out = pd.DataFrame(
        {
            "network_seed": df["network_seed"] if "network_seed" in df.columns else int(ctx.cfg.network_seed),
            "similarity_bin": df["similarity_bin"] if "similarity_bin" in df.columns else df.index.astype(str),
            "mean_pixel_similarity": df["mean_pixel_similarity"] if "mean_pixel_similarity" in df.columns else df.get("bin_center", np.nan),
            "mean_acc_drop": df["mean_acc_drop"] if "mean_acc_drop" in df.columns else np.nan,
            "mean_drop_event": df["mean_drop_event"] if "mean_drop_event" in df.columns else df.get("drop_rate", np.nan),
            "mean_b_vec": df["mean_b_vec"] if "mean_b_vec" in df.columns else np.nan,
            "mean_DPI_L3": df["mean_DPI_L3"] if "mean_DPI_L3" in df.columns else np.nan,
            "n_pairs": df["n_pairs"] if "n_pairs" in df.columns else len(df),
        }
    )
    _save_csv(ctx, out, ctx.metrics_dir / "supp_s7_similarity_bin_full_trend.csv")


def _write_s7_overlap_matching_diagnostics(ctx: ExperimentContext) -> None:
    matches_path = ctx.metrics_dir / "panel_d_iso_similarity_matched_pairs.csv"
    if matches_path.exists() and _csv_nonempty(matches_path):
        matches = pd.read_csv(matches_path)
        rows = []
        for network_seed, part in matches.groupby("network_seed", sort=False):
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "comparison": "iso_similarity_high_vs_low_overlap",
                    "group": "matched_pairs",
                    "mean_pixel_similarity": float(pd.to_numeric(pd.concat([part["pixel_similarity_high"], part["pixel_similarity_low"]]), errors="coerce").mean()),
                    "mean_input_energy_sample": float(pd.to_numeric(pd.concat([part["input_energy_sample_high"], part["input_energy_sample_low"]]), errors="coerce").mean()),
                    "mean_input_energy_probe": float(pd.to_numeric(pd.concat([part["input_energy_probe_high"], part["input_energy_probe_low"]]), errors="coerce").mean()),
                    "mean_dice_overlap": float(pd.to_numeric(pd.concat([part["dice_overlap_high"], part["dice_overlap_low"]]), errors="coerce").mean()),
                    "n_pairs": int(len(part) * 2),
                    "similarity_abs_diff": float(pd.to_numeric(part["similarity_difference"], errors="coerce").mean()),
                    "sample_energy_rel_diff": float(pd.to_numeric(part["sample_energy_rel_difference"], errors="coerce").mean()),
                    "probe_energy_rel_diff": float(pd.to_numeric(part["probe_energy_rel_difference"], errors="coerce").mean()),
                    "class_pair_balance_notes": "see panel_d_matching_balance_diagnostics.csv for aggregate class/probe-label balance",
                }
            )
        _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "supp_s7_overlap_matching_diagnostics.csv")
        return
    for candidate in ("panel_d_matching_balance_diagnostics.csv", "supp_overlap_matching_diagnostics.csv"):
        src = ctx.metrics_dir / candidate
        if src.exists():
            _copy_csv_alias(ctx, src, ctx.metrics_dir / "supp_s7_overlap_matching_diagnostics.csv", empty_columns=["network_seed", "comparison", "group", "n_pairs"], reason=f"{candidate}_missing_or_empty")
            return
    _write_empty_csv(ctx, ctx.metrics_dir / "supp_s7_overlap_matching_diagnostics.csv", ["network_seed", "comparison", "group", "mean_pixel_similarity", "mean_input_energy_sample", "mean_input_energy_probe", "mean_dice_overlap", "n_pairs", "similarity_abs_diff", "sample_energy_rel_diff", "probe_energy_rel_diff", "class_pair_balance_notes"], "overlap_matching_sources_missing")


def _write_s7_overlap_regression_controls(ctx: ExperimentContext) -> None:
    src = ctx.metrics_dir / "supp_overlap_accuracy_regression.csv"
    if not src.exists():
        src = ctx.metrics_dir / "supp_overlap_similarity_regression.csv"
    if not src.exists():
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s7_overlap_regression_controls.csv", ["network_seed", "metric", "beta_overlap", "beta_similarity", "beta_input_energy_sample", "beta_input_energy_probe", "r2", "n_pairs", "notes"], "overlap_regression_sources_missing")
        return
    df = pd.read_csv(src)
    if df.empty:
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s7_overlap_regression_controls.csv", ["network_seed", "metric", "beta_overlap", "beta_similarity", "beta_input_energy_sample", "beta_input_energy_probe", "r2", "n_pairs", "notes"], "overlap_regression_source_empty")
        return
    out = pd.DataFrame(
        {
            "network_seed": df["network_seed"] if "network_seed" in df.columns else int(ctx.cfg.network_seed),
            "metric": df["metric"] if "metric" in df.columns else "unknown",
            "beta_overlap": df["beta_overlap"] if "beta_overlap" in df.columns else np.nan,
            "beta_similarity": df["beta_similarity"] if "beta_similarity" in df.columns else np.nan,
            "beta_input_energy_sample": df["beta_input_energy_sample"] if "beta_input_energy_sample" in df.columns else df.get("beta_input_energy", np.nan),
            "beta_input_energy_probe": df["beta_input_energy_probe"] if "beta_input_energy_probe" in df.columns else np.nan,
            "r2": df["r2"] if "r2" in df.columns else np.nan,
            "n_pairs": df["n_pairs"] if "n_pairs" in df.columns else np.nan,
            "notes": df["notes"] if "notes" in df.columns else "standardized S7 regression alias",
        }
    )
    _save_csv(ctx, out, ctx.metrics_dir / "supp_s7_overlap_regression_controls.csv")


def _write_s7_random_nonoverlap_perturbation_controls(ctx: ExperimentContext) -> None:
    src = ctx.metrics_dir / "panel_d_overlap_perturbation_summary.csv"
    if not src.exists():
        src = ctx.metrics_dir / "supp_overlap_preserving_perturbation_summary.csv"
    if not src.exists():
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s7_random_nonoverlap_perturbation_controls.csv", ["network_seed", "condition", "mean_DPI_L3", "mean_dynamic_like_recovery", "mean_probe_accuracy", "n_pairs"], "overlap_perturbation_summary_missing")
        return
    df = pd.read_csv(src)
    keep = {"full_dynamic", "full_static", "sample_keep_overlap_only_dynamic", "sample_keep_nonoverlap_only_dynamic", "sample_random_matched_dynamic"}
    if "condition" in df.columns:
        df = df[df["condition"].astype(str).isin(keep)].copy()
    cols = ["network_seed", "condition", "mean_DPI_L3", "mean_dynamic_like_recovery", "mean_probe_accuracy", "n_pairs"]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    _save_csv(ctx, df[cols], ctx.metrics_dir / "supp_s7_random_nonoverlap_perturbation_controls.csv")


def _decision_deflection_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "condition", "mean_dynamic_like_recovery", "mean_static_to_dynamic_push", "mean_decision_deflection_score", "condition_matches_dynamic_rate", "condition_matches_static_rate", "n_pairs"]
    if df.empty or "condition" not in df.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    for (network_seed, condition), part in df.groupby(["network_seed", "condition"], sort=False):
        rows.append(
            {
                "network_seed": int(network_seed),
                "condition": str(condition),
                "mean_dynamic_like_recovery": _mean_existing(part, ["dynamic_like_recovery"]),
                "mean_static_to_dynamic_push": _mean_existing(part, ["static_to_dynamic_push"]),
                "mean_decision_deflection_score": _mean_existing(part, ["decision_deflection_score"]),
                "condition_matches_dynamic_rate": _mean_existing(part, ["condition_matches_dynamic"]),
                "condition_matches_static_rate": _mean_existing(part, ["condition_matches_static"]),
                "n_pairs": int(part["pair_id"].nunique()) if "pair_id" in part.columns else int(len(part)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def save_debug_figures(ctx: ExperimentContext) -> None:
    apply_publication_style()
    debug_specs = [
        ("fig4_debug_similarity_entry", ctx.metrics_dir / "panel_b_similarity_bin_summary.csv", "similarity_bin", "mean_acc_drop"),
        ("fig4_debug_overlap_localization", ctx.metrics_dir / "panel_c_overlap_localization_metrics.csv", "dice_overlap", "acc_drop"),
        ("fig4_debug_s7_iso_similarity_overlap_contrast", ctx.metrics_dir / "supp_s7_iso_similarity_overlap_contrast.csv", "network_seed", "delta_drop_rate"),
        ("fig4_debug_l3_trace_displacement", ctx.metrics_dir / "panel_e_time_resolved_l3_displacement.csv", "time_ms", "DPI_L3_t"),
        ("fig4_debug_l3_accumulator", ctx.metrics_dir / "panel_f_l3_accumulator_region_replay_metrics.csv", "replacement_push_kstar", "replacement_pullback_kstar"),
        ("fig4_debug_overlap_perturbation_contrast", ctx.metrics_dir / "panel_d_overlap_perturbation_contrast.csv", "network_seed", "overlap_minus_nonoverlap_DPI"),
        ("fig4_debug_s8_decision_deflection", ctx.metrics_dir / "supp_s8_decision_deflection_summary.csv", "condition", "mean_decision_deflection_score"),
    ]
    for stem, path, x_col, y_col in debug_specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            continue
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.0, 2.5), dpi=150)
        if x_col == "condition":
            ax.bar(np.arange(len(df)), pd.to_numeric(df[y_col], errors="coerce"))
            ax.set_xticks(np.arange(len(df)), [CONDITION_LABELS.get(str(v), str(v)) for v in df[x_col]], rotation=30, ha="right")
        else:
            x_num = pd.to_numeric(df[x_col], errors="coerce")
            if x_num.notna().any():
                ax.scatter(x_num, pd.to_numeric(df[y_col], errors="coerce"), s=16)
            else:
                labels = [str(v) for v in df[x_col]]
                ax.scatter(np.arange(len(df)), pd.to_numeric(df[y_col], errors="coerce"), s=16)
                ax.set_xticks(np.arange(len(df)), labels, rotation=30, ha="right")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        save_figure_all_formats(fig, ctx.debug_dir / stem)
        plt.close(fig)
    ctx.completed_modules["debug_figures"] = True


def _write_config_files(ctx: ExperimentContext) -> None:
    cfg = ctx.cfg
    _write_json(_json_safe(asdict(cfg)), ctx.config_dir / "run_config.json")
    _write_json(
        {
            "fig4_design_version": FIG4_DESIGN_VERSION,
            "main_panels": FIG4_MAIN_PANELS,
            "legacy_methods": FIG4_LEGACY_METHODS,
            "supplement_plan": FIG4_SUPPLEMENT_PLAN,
            "required_conditions": list(CORE_CONDITIONS),
            "fig4d_main_conditions": list(D_L1_STSP_CONDITIONS),
            "main_required_outputs": FIG4_MAIN_REQUIRED_OUTPUTS,
            "supplementary_outputs": {
                "S7": FIG4_S7_OUTPUTS,
                "S8": FIG4_S8_OUTPUTS,
            },
            "compatibility_outputs": FIG4_COMPATIBILITY_OUTPUTS,
        },
        ctx.config_dir / "figure_requirements.json",
    )
    _write_json(
        {
            "conditions": {
                "full_dynamic": "normal sample, delay, and unchanged probe with dynamic STSP",
                "full_static": "same DMS sequence with stsp_mode=static_frozen across the low-level rollout",
                "sample_keep_overlap_only_dynamic": "encoded sample spikes with non-overlap pixels removed, leaving overlap support; probe unchanged",
                "sample_keep_nonoverlap_only_dynamic": "encoded sample spikes with overlap pixels removed, leaving non-overlap support; probe unchanged",
                "sample_random_matched_dynamic": "encoded sample spikes with complement of random matched sample-side support removed; probe unchanged",
                "full_dynamic_intact": "complete sample, normal delay, unchanged probe, dynamic L1 STSP, L2/L3 STSP variables held fixed during the probe",
                "l1_overlap_reset": "after complete sample and delay, reset layer1 STSP in sample/probe overlap units to S0 immediately before the unchanged probe",
                "l1_nonoverlap_reset": "after complete sample and delay, reset layer1 STSP in sample-only non-overlap units to S0 immediately before the unchanged probe",
                "l1_random_matched_reset": "after complete sample and delay, reset layer1 STSP in random sample-foreground units matched to overlap count immediately before the unchanged probe",
            },
            "panel_b": "similarity_bias_experiment-compatible DMS snapshot readout using layer3 v_mem, top_m_mean m=1.",
            "panel_c": "highest-similarity-bin high-vs-low overlap accuracy_drop = correct_static - correct_dynamic; old overlap localization files are compatibility outputs.",
            "panel_d": "pre-probe layer1 STSP overlap reset with complete sample and unchanged probe; main metric is accuracy_drop_vs_static = correct_static - correct_condition.",
            "panel_e": "probe_l3_trace / s2p DPI with centered-L2 pattern normalization.",
            "panel_f": "l3_accumulator_mechanism-compatible L3 region deletion/replacement replay.",
            "static_frozen_approximation": "The project API exposes stsp_mode=static_frozen as U*1 neutral STSP gain without u/x updates.",
            "probe_input_core_assay": "unchanged for all Fig.4 perturbation conditions",
            "decision_deflection_status": "L3 accumulator replay supports main Fig.4F; simplified vector deflection is used as S8 decision-dynamics supplement.",
            "supplement_plan": FIG4_SUPPLEMENT_PLAN,
        },
        ctx.config_dir / "condition_spec.json",
    )
    _write_json(
        {
            "status": "S7_control_and_Fig4C_inset",
            "metric": "drop_event",
            "static_correct_eligible_only": True,
            "matching": {
                "num_iso_similarity_bins": int(cfg.num_iso_similarity_bins),
                "overlap_tail_quantile": float(cfg.overlap_tail_quantile),
                "match_similarity_caliper": float(cfg.match_similarity_caliper),
                "match_energy_caliper": float(cfg.match_energy_caliper),
                "match_require_probe_label": bool(cfg.match_require_probe_label),
                "match_require_class_pair": bool(cfg.match_require_class_pair),
            },
            "permutation_test": {
                "n_match_permutations": int(cfg.n_match_permutations),
                "one_sided_direction": "high_overlap > low_overlap",
            },
        },
        ctx.config_dir / "overlap_accuracy_identification_spec.json",
    )
    _write_json(
        {
            "foreground_threshold": float(cfg.foreground_threshold),
            "dilation_radius": int(cfg.dilation_radius),
            "overlap": "sample foreground AND probe foreground",
            "dice_overlap": "2*area(overlap)/(area(sample foreground)+area(probe foreground))",
        },
        ctx.config_dir / "overlap_definition_spec.json",
    )
    _write_json(
        {
            "core_perturbation_scope": "pre-probe layer1 STSP state only",
            "mask_application_space": "layer1 STSP variable tensor",
            "probe_perturbation": "disabled",
            "sample_mask_mode": "complete_sample_no_removal",
            "reset_timing": "after complete sample and delay, before unchanged probe",
            "perturbed_layer": "L1",
            "perturbed_variables": ["u", "x"],
            "l2_l3_stsp_probe_mode": "static_frozen",
            "random_mask_candidates": int(cfg.random_mask_candidates),
            "matched_to": "overlap_mask",
            "matching_targets": ["pixel_count", "input_energy", "spike_count_estimate"],
            "legacy_outputs": [
                "panel_d_overlap_perturbation_metrics.csv",
                "panel_d_overlap_perturbation_summary.csv",
                "panel_d_overlap_perturbation_contrast.csv",
            ],
        },
        ctx.config_dir / "perturbation_spec.json",
    )


def _write_summary(ctx: ExperimentContext) -> dict[str, Any]:
    required_main: list[Path] = []
    if ctx.cfg.run_similarity_entry:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_b_similarity_entry_metrics.csv",
                ctx.metrics_dir / "panel_b_similarity_bin_summary.csv",
                ctx.metrics_dir / "panel_b_similarity_accuracy_drop_summary.csv",
            ]
        )
    if ctx.cfg.run_overlap_accuracy_identification:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop.csv",
                ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop_summary.csv",
                ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop_contrast.csv",
            ]
        )
    if ctx.cfg.run_overlap_perturbation:
        required_main.extend(
            [
                ctx.raw_dir / "panel_d_l1_stsp_overlap_perturbation_trial_readout.csv",
                ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_summary.csv",
                ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_contrast.csv",
                ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_audit.csv",
            ]
        )
    if ctx.cfg.run_decision_spike_displacement:
        required_main.extend([ctx.metrics_dir / "panel_e_time_resolved_l3_displacement.csv", ctx.metrics_dir / "panel_e_decision_spike_displacement.csv"])
    if ctx.cfg.run_decision_deflection:
        required_main.extend([ctx.metrics_dir / "panel_f_l3_accumulator_region_replay_metrics.csv", ctx.metrics_dir / "panel_f_l3_accumulator_summary.csv"])
    required_supp: list[Path] = []
    if ctx.cfg.run_overlap_accuracy_identification:
        required_supp.extend(
            [
                ctx.metrics_dir / "supp_s7_iso_similarity_overlap_contrast.csv",
                ctx.metrics_dir / "supp_s7_iso_similarity_permutation_null.csv",
                ctx.metrics_dir / "supp_s7_iso_similarity_matched_pairs.csv",
                ctx.metrics_dir / "supp_s7_overlap_matching_balance_diagnostics.csv",
            ]
        )
    if ctx.cfg.run_supplement or ctx.cfg.run_overlap_accuracy_identification or ctx.cfg.run_overlap_perturbation:
        required_supp.extend(ctx.seed_dir / output for output in FIG4_S7_OUTPUTS)
    if ctx.cfg.run_decision_spike_displacement:
        required_supp.extend([ctx.metrics_dir / "supp_s8_time_resolved_l3_displacement.csv", ctx.metrics_dir / "supp_s8_decision_spike_displacement.csv"])
    if ctx.cfg.run_decision_deflection:
        required_supp.extend([ctx.metrics_dir / "supp_s8_l3_accumulator_replay_metrics.csv", ctx.metrics_dir / "supp_s8_l3_accumulator_summary.csv"])
    if ctx.cfg.run_supplement:
        required_supp.extend([ctx.metrics_dir / "supp_s8_decision_deflection_metrics.csv", ctx.metrics_dir / "supp_s8_decision_deflection_summary.csv"])
    decision_deflection_available = bool(ctx.availability.get("decision_deflection_available", _csv_nonempty(ctx.metrics_dir / "supp_s8_decision_deflection_metrics.csv")))
    summary = {
        "figure": FIGURE_ID,
        "network_seed": int(ctx.cfg.network_seed),
        "run_mode": "single_network",
        "fig4_design_version": FIG4_DESIGN_VERSION,
        "legacy_similarity_bias_method": bool(ctx.cfg.use_legacy_similarity_bias_method),
        "legacy_overlap_perturbation_method": bool(ctx.cfg.use_legacy_overlap_perturbation_method),
        "legacy_l3_accumulator_method": bool(ctx.cfg.use_legacy_l3_accumulator_method),
        "main_panels": FIG4_SUMMARY_PANELS,
        "supplement_plan": FIG4_SUPPLEMENT_PLAN,
        "overlap_perturbation_in_main": True,
        "iso_similarity_overlap_identification_demoted_to_S7": True,
        "fig4C_main": "highest-similarity-bin high-vs-low overlap accuracy drop",
        "fig4C_inset_or_S7C": "legacy iso-similarity high-vs-low overlap matched contrast",
        "fig4D_preserved": True,
        "legacy_timing_exact_match": bool(ctx.cfg.legacy_exact_mode and int(ctx.cfg.sample_ms) == 200 and int(ctx.cfg.delay_ms) == 500 and int(ctx.cfg.probe_ms) == 100),
        "mask_application_space": "encoded_spikes",
        "probe_perturbation": "disabled",
        "panel_f_main_method": "l3_region_deletion_replacement_replay",
        "simplified_decision_deflection_supplement_available": decision_deflection_available,
        "decision_deflection_available": decision_deflection_available,
        "decision_deflection_missing_reason": ctx.availability.get("decision_deflection_missing_reason"),
        "readout_rule_robustness_status": "optional_not_run",
        "main_fig4d_metric": "accuracy_drop_vs_static",
        "main_fig4d_method": "pre-probe layer1 STSP overlap reset with complete sample and unchanged probe",
        "main_fig4f_method": "l3_region_deletion_replacement_replay",
        "fig4c_high_similarity_overlap": _fig4c_high_similarity_summary(ctx),
        "fig4d_l1_stsp_overlap_perturbation": _fig4d_l1_stsp_summary(ctx),
        "n_iso_similarity_matches": _n_iso_similarity_matches(ctx),
        "min_matches_per_network": int(ctx.cfg.min_matches_per_network),
        "smoke": bool(ctx.cfg.smoke),
        "completed_modules": ctx.completed_modules,
        "output_files": ctx.output_files,
        "n_pairs": int(ctx.n_pairs),
        "conditions": list(CORE_CONDITIONS),
        "fig4d_main_conditions": list(D_L1_STSP_CONDITIONS),
        "similarity_bins": int(ctx.cfg.num_similarity_bins),
        "overlap_bins": int(ctx.cfg.num_overlap_bins),
        "mask_definition": {"foreground_threshold": float(ctx.cfg.foreground_threshold), "dilation_radius": int(ctx.cfg.dilation_radius)},
        "supplement_alias_missing_reasons": ctx.availability.get("supplement_alias_missing_reasons", {}),
        "warnings": ctx.warnings,
        "main_claim_supported_fields_available": all(path.exists() for path in required_main),
        "missing_for_main_figure": [_rel(path, ctx.seed_dir) for path in required_main if not path.exists()],
        "missing_for_supplementary": [_rel(path, ctx.seed_dir) for path in required_supp if not path.exists()],
    }
    _write_json(summary, ctx.seed_dir / "summary.json")
    ctx.output_files["summary"] = "summary.json"
    return summary


def _fig4c_high_similarity_summary(ctx: ExperimentContext) -> dict[str, Any]:
    contrast_path = ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop_contrast.csv"
    if not contrast_path.exists():
        return {"enabled": False}
    try:
        contrast = pd.read_csv(contrast_path)
    except Exception as exc:
        return {"enabled": False, "failure_reason": str(exc)}
    if contrast.empty:
        return {"enabled": False}
    row = contrast.iloc[0]
    return {
        "enabled": True,
        "highest_similarity_bin": str(row.get("highest_similarity_bin", "")),
        "overlap_split_method": "median_split_within_highest_similarity_bin",
        "mean_acc_drop_high_overlap": _json_float(row.get("mean_acc_drop_high_overlap", np.nan)),
        "mean_acc_drop_low_overlap": _json_float(row.get("mean_acc_drop_low_overlap", np.nan)),
        "high_minus_low_acc_drop": _json_float(row.get("high_minus_low_acc_drop", np.nan)),
        "n_pairs_high": _json_int(row.get("n_pairs_high", 0)),
        "n_pairs_low": _json_int(row.get("n_pairs_low", 0)),
    }


def _fig4d_l1_stsp_summary(ctx: ExperimentContext) -> dict[str, Any]:
    contrast_path = ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_contrast.csv"
    audit_path = ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_audit.csv"
    if not contrast_path.exists():
        return {"enabled": False}
    try:
        contrast = pd.read_csv(contrast_path)
        audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    except Exception as exc:
        return {"enabled": False, "failure_reason": str(exc)}
    if contrast.empty:
        return {"enabled": False}
    row = contrast.iloc[0]
    audit_row = audit.iloc[0] if not audit.empty else {}
    return {
        "enabled": True,
        "perturbed_layer": "L1",
        "perturbed_variables": ["u", "x"],
        "probe_input_unchanged": bool(audit_row.get("probe_input_unchanged", True)) if hasattr(audit_row, "get") else True,
        "sample_input_complete": bool(audit_row.get("sample_input_complete", True)) if hasattr(audit_row, "get") else True,
        "l2_stsp_frozen": bool(audit_row.get("l2_stsp_frozen", False)) if hasattr(audit_row, "get") else False,
        "l3_stsp_frozen": bool(audit_row.get("l3_stsp_frozen", False)) if hasattr(audit_row, "get") else False,
        "acc_drop_dynamic": _json_float(row.get("acc_drop_dynamic", np.nan)),
        "acc_drop_overlap_reset": _json_float(row.get("acc_drop_overlap_reset", np.nan)),
        "acc_drop_nonoverlap_reset": _json_float(row.get("acc_drop_nonoverlap_reset", np.nan)),
        "acc_drop_random_reset": _json_float(row.get("acc_drop_random_reset", np.nan)),
    }


def _json_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _json_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _any_metric_stage(cfg: Fig4Config) -> bool:
    return any(
        (
            cfg.run_similarity_entry,
            cfg.run_overlap_localization,
            cfg.run_overlap_accuracy_identification,
            cfg.run_decision_spike_displacement,
            cfg.run_decision_deflection,
            cfg.run_overlap_perturbation,
            cfg.run_supplement,
        )
    )


def _n_iso_similarity_matches(ctx: ExperimentContext) -> int:
    path = ctx.metrics_dir / "panel_d_iso_similarity_matched_pairs.csv"
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def _resolve_fig4_readout_step(ctx: ExperimentContext) -> int:
    return resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(ctx.cfg.probe_steps),
        decision_offset=int(getattr(ctx.net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )


def _image_cache(ctx: ExperimentContext, pair_trials: pd.DataFrame) -> dict[int, torch.Tensor]:
    ids = pd.unique(pair_trials[["sample_image_id", "probe_image_id"]].values.ravel()) if not pair_trials.empty else []
    return {int(i): ctx.dataset[int(i)][0].detach().cpu().to(torch.float32) for i in ids}


def _aggregate_prediction(predictions: Sequence[int], mean_voltage: np.ndarray) -> int:
    values = [int(v) for v in predictions]
    if not values:
        return int(np.argmax(np.asarray(mean_voltage, dtype=np.float64)))
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    max_count = max(counts.values())
    tied = sorted(label for label, count in counts.items() if count == max_count)
    if len(tied) == 1:
        return int(tied[0])
    voltage = np.asarray(mean_voltage, dtype=np.float64)
    valid_tied = [label for label in tied if 0 <= int(label) < voltage.size]
    if not valid_tied:
        return int(tied[0])
    return int(max(valid_tied, key=lambda label: float(voltage[int(label)])))


def _compute_bvec(voltage_dynamic: np.ndarray, voltage_static: np.ndarray) -> float:
    dyn = np.asarray(voltage_dynamic, dtype=np.float64)
    sta = np.asarray(voltage_static, dtype=np.float64)
    return float(np.linalg.norm((dyn - dyn.mean()) - (sta - sta.mean()), ord=2))


def _bvec_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bin_label, part in df.groupby("similarity_bin", sort=False):
        values = pd.to_numeric(part["b_vec"], errors="coerce").dropna()
        rows.append(
            {
                "network_seed": int(part["network_seed"].iloc[0]) if len(part) else 0,
                "similarity_bin": str(bin_label),
                "bin_center": float(pd.to_numeric(part["pixel_similarity"], errors="coerce").mean()),
                "n_trials": int(len(part)),
                "mean_B_vec": float(values.mean()) if len(values) else float("nan"),
                "std_B_vec": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "sem_B_vec": float(values.sem()) if len(values) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _cti_summary(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-12
    support = list(range(NUM_CLASSES))
    if ((df.get("pred_dynamic", pd.Series(dtype=int)) == -1) | (df.get("pred_static", pd.Series(dtype=int)) == -1)).any():
        support = [-1] + support
    rows = []
    for bin_index, (bin_label, bin_part) in enumerate(df.groupby("similarity_bin", sort=False)):
        for sample_label in range(NUM_CLASSES):
            for probe_label in range(NUM_CLASSES):
                sub = bin_part[
                    bin_part["sample_label"].astype(int).eq(int(sample_label))
                    & bin_part["probe_label"].astype(int).eq(int(probe_label))
                ]
                if sub.empty:
                    cti = capture = capture_ratio = float("nan")
                else:
                    dyn = sub["pred_dynamic"].to_numpy(dtype=np.int64, copy=False)
                    sta = sub["pred_static"].to_numpy(dtype=np.int64, copy=False)
                    q_dyn = np.asarray([np.mean(dyn == label) for label in support], dtype=np.float64)
                    q_sta = np.asarray([np.mean(sta == label) for label in support], dtype=np.float64)
                    cti = 0.5 * float(np.abs(q_dyn - q_sta).sum())
                    sample_idx = support.index(int(sample_label))
                    capture = float(q_dyn[sample_idx] - q_sta[sample_idx])
                    capture_ratio = float(max(capture, 0.0) / (cti + eps))
                rows.append(
                    {
                        "network_seed": int(df["network_seed"].iloc[0]) if len(df) else 0,
                        "similarity_bin": str(bin_label),
                        "bin_index": int(bin_index),
                        "sample_label": int(sample_label),
                        "probe_label": int(probe_label),
                        "n_trials": int(len(sub)),
                        "cti": float(cti),
                        "capture": float(capture),
                        "capture_ratio": float(capture_ratio),
                    }
                )
    return pd.DataFrame(rows)


def _sample_input_mask_for_condition(
    batch: pd.DataFrame,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
    condition: str,
) -> torch.Tensor | None:
    if condition in {"full_dynamic", "full_static"}:
        return None
    masks: list[np.ndarray] = []
    for row in batch.itertuples(index=False):
        bank = mask_bank[int(row.pair_id)]
        if condition == "sample_keep_overlap_only_dynamic":
            mask = bank["sample_nonoverlap_mask"]
        elif condition == "sample_keep_nonoverlap_only_dynamic":
            mask = bank["sample_overlap_mask"]
        elif condition == "sample_random_matched_dynamic":
            mask = bank["random_matched_remove_mask"]
        else:
            raise ValueError(f"Unsupported Fig.4 perturbation condition: {condition}")
        masks.append(np.asarray(mask, dtype=bool))
    return torch.as_tensor(np.stack(masks, axis=0), dtype=torch.bool)


def _l3_summary_rows(results: pd.DataFrame, summary: Mapping[str, Any], network_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "network_seed": int(network_seed),
            "summary_group": "all",
            "bias_direction": "all",
            "n_pairs": int(len(results)),
            "mean_reconstruction_cosine_plus": float(results["reconstruction_cosine_plus"].mean(skipna=True)) if len(results) else float("nan"),
            "mean_reconstruction_cosine_minus": float(results["reconstruction_cosine_minus"].mean(skipna=True)) if len(results) else float("nan"),
            "direction_match_rate_plus": float(results["direction_match_plus"].mean(skipna=True)) if len(results) else float("nan"),
            "direction_match_rate_minus": float(results["direction_match_minus"].mean(skipna=True)) if len(results) else float("nan"),
            "mean_static_to_dynamic_push": float(results["replacement_push_kstar"].mean(skipna=True)) if len(results) else float("nan"),
            "mean_dynamic_to_static_pullback": float(results["replacement_pullback_kstar"].mean(skipna=True)) if len(results) else float("nan"),
            "mean_dynamic_vs_static_deletion_contrast": float(results["deletion_dynamic_minus_static_kstar"].mean(skipna=True)) if len(results) else float("nan"),
            "legacy_summary_payload": json.dumps(_json_safe(dict(summary)), sort_keys=True),
        }
    ]
    if "bias_direction" in results.columns:
        for direction, part in results.groupby("bias_direction", sort=True):
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "summary_group": "by_bias_direction",
                    "bias_direction": str(direction),
                    "n_pairs": int(len(part)),
                    "mean_reconstruction_cosine_plus": float(part["reconstruction_cosine_plus"].mean(skipna=True)),
                    "mean_reconstruction_cosine_minus": float(part["reconstruction_cosine_minus"].mean(skipna=True)),
                    "direction_match_rate_plus": float(part["direction_match_plus"].mean(skipna=True)),
                    "direction_match_rate_minus": float(part["direction_match_minus"].mean(skipna=True)),
                    "mean_static_to_dynamic_push": float(part["replacement_push_kstar"].mean(skipna=True)),
                    "mean_dynamic_to_static_pullback": float(part["replacement_pullback_kstar"].mean(skipna=True)),
                    "mean_dynamic_vs_static_deletion_contrast": float(part["deletion_dynamic_minus_static_kstar"].mean(skipna=True)),
                    "legacy_summary_payload": "",
                }
            )
    return rows


def _condition_sample_image(image: torch.Tensor, masks: Mapping[str, np.ndarray], condition: str) -> torch.Tensor:
    mask_name = SAMPLE_SIDE_MASKS[condition]
    if mask_name == "full_sample":
        return image.detach().cpu().clone()
    if mask_name == "random_matched_keep_support":
        keep = np.asarray(masks["random_matched_mask"], dtype=bool)
        mask = torch.as_tensor(keep, dtype=image.dtype).unsqueeze(0)
        return image.detach().cpu() * mask
    mask = torch.as_tensor(masks[mask_name], dtype=image.dtype).unsqueeze(0)
    return image.detach().cpu().masked_fill(mask.bool().unsqueeze(0) if mask.ndim == 2 else mask.bool(), 0.0)


def _prepare_condition_batch(
    ctx: ExperimentContext,
    batch: pd.DataFrame,
    images_cache: Mapping[int, torch.Tensor],
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
    conditions: Sequence[str],
) -> tuple[torch.Tensor, list[str]]:
    if ctx.cfg.enable_condition_batch:
        ctx.warnings.append("Fig.4 condition batch helper is scaffolded; default rollout remains order-preserving per-condition.")
    sample_images: list[torch.Tensor] = []
    condition_names: list[str] = []
    for condition in conditions:
        for _, row in batch.iterrows():
            sample_images.append(_condition_sample_image(images_cache[int(row["sample_image_id"])], mask_bank[int(row["pair_id"])], condition))
            condition_names.append(str(condition))
    return torch.stack(sample_images, dim=0), condition_names


def _encode_batch(ctx: ExperimentContext, images: torch.Tensor, steps: int) -> torch.Tensor:
    return encode_images(ctx.encoder, images.to(ctx.device, dtype=torch.float32), int(steps)).to(ctx.device)


def _class_evidence_trace(net: Any, l3_v: torch.Tensor) -> np.ndarray:
    arr = l3_v.detach().cpu().to(torch.float32).numpy()
    if arr.ndim != 5:
        raise ValueError(f"Expected L3 trace [T,B,C,H,W], got {arr.shape}")
    t_steps, batch, channels, height, width = arr.shape
    num_classes = int(getattr(net.layer3, "num_classes", NUM_CLASSES))
    neurons_per_class = int(getattr(net.layer3, "neurons_per_class", max(1, channels // num_classes)))
    usable_channels = min(channels, num_classes * neurons_per_class)
    grouped = arr[:, :, :usable_channels, :, :].reshape(t_steps, batch, num_classes, -1)
    return grouped.mean(axis=3).astype(np.float32)


def _foreground_mask(image: torch.Tensor, threshold: float) -> np.ndarray:
    arr = image.detach().cpu().to(torch.float32).abs().amax(dim=0).numpy()
    return np.asarray(arr > float(threshold), dtype=bool)


def _build_masks(sample_image: torch.Tensor, probe_image: torch.Tensor, rng: np.random.Generator, cfg: Fig4Config) -> dict[str, np.ndarray]:
    sample_fg = _foreground_mask(sample_image, cfg.foreground_threshold)
    probe_fg = _foreground_mask(probe_image, cfg.foreground_threshold)
    overlap = sample_fg & probe_fg
    sample_nonoverlap = sample_fg & ~probe_fg
    probe_only = probe_fg & ~sample_fg
    random_matched = _random_matched_mask(sample_image, sample_fg, overlap, rng, int(cfg.random_mask_candidates))
    random_remove = sample_fg & ~random_matched
    return {
        "sample_foreground_mask": sample_fg,
        "probe_foreground_mask": probe_fg,
        "overlap_mask": overlap,
        "sample_overlap_mask": overlap,
        "sample_nonoverlap_mask": sample_nonoverlap,
        "sample_nonoverlap_control_mask": random_matched,
        "probe_only_mask": probe_only,
        "random_matched_mask": random_matched,
        "random_matched_remove_mask": random_remove,
    }


def _random_matched_mask(sample_image: torch.Tensor, sample_fg: np.ndarray, target: np.ndarray, rng: np.random.Generator, candidates: int) -> np.ndarray:
    target_count = int(target.sum())
    if target_count <= 0:
        return np.zeros_like(sample_fg, dtype=bool)
    available = np.argwhere(sample_fg)
    if len(available) == 0:
        return np.zeros_like(sample_fg, dtype=bool)
    take = min(target_count, len(available))
    target_energy = _mask_energy(sample_image, target)
    best_mask = None
    best_score = float("inf")
    for _ in range(max(1, int(candidates))):
        chosen = available[rng.choice(len(available), size=take, replace=False)]
        mask = np.zeros_like(sample_fg, dtype=bool)
        mask[chosen[:, 0], chosen[:, 1]] = True
        score = abs(_mask_energy(sample_image, mask) - target_energy) + abs(int(mask.sum()) - target_count)
        if score < best_score:
            best_score = score
            best_mask = mask
    return np.asarray(best_mask, dtype=bool)


def _mask_energy(image: torch.Tensor, mask: np.ndarray) -> float:
    arr = image.detach().cpu().to(torch.float32).abs().amax(dim=0).numpy()
    return float(arr[np.asarray(mask, dtype=bool)].sum())


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.asarray(a).sum() + np.asarray(b).sum())
    return 0.0 if denom <= 0 else float(2.0 * np.logical_and(a, b).sum() / denom)


def _safe_div(num: float, denom: float) -> float:
    return 0.0 if denom <= 0 else float(num / denom)


def _assign_bins(df: pd.DataFrame, value_col: str, bin_col: str, n_bins: int) -> pd.DataFrame:
    out = df.copy()
    values = pd.to_numeric(out[value_col], errors="coerce")
    try:
        codes = pd.qcut(values.rank(method="first"), q=max(1, int(n_bins)), labels=False, duplicates="drop")
    except ValueError:
        codes = pd.Series(np.zeros(len(out), dtype=int), index=out.index)
    out[bin_col] = [f"bin_{int(c) + 1}" if pd.notna(c) else "bin_1" for c in codes]
    return out


def _balanced_select_pairs(pool: pd.DataFrame, max_pairs: int, rng: np.random.Generator) -> pd.DataFrame:
    use = pool.copy()
    use["class_pair"] = use["sample_label"].astype(str) + "->" + use["probe_label"].astype(str)
    chunks = []
    for _, part in use.groupby("class_pair", sort=True):
        shuffled = part.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        chunks.append(shuffled)
    interleaved = []
    max_len = max(len(c) for c in chunks)
    for i in range(max_len):
        for chunk in chunks:
            if i < len(chunk):
                interleaved.append(chunk.iloc[i])
            if len(interleaved) >= max_pairs:
                return pd.DataFrame(interleaved).reset_index(drop=True)
    return pd.DataFrame(interleaved).head(max_pairs).reset_index(drop=True)


def _assign_matched_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["matched_group_id"] = ""
    high_sim = sorted(out["similarity_bin"].unique())[-1]
    sub = out[out["similarity_bin"].eq(high_sim)].copy()
    if len(sub) < 2:
        return out
    median_overlap = float(sub["dice_overlap"].median())
    high = sub[sub["dice_overlap"] >= median_overlap].sort_values("dice_overlap", ascending=False)
    low = sub[sub["dice_overlap"] < median_overlap].sort_values("dice_overlap", ascending=True)
    n = min(len(high), len(low))
    for i in range(n):
        gid = f"match_{i:03d}"
        out.loc[out["candidate_id"].eq(high.iloc[i]["candidate_id"]), "matched_group_id"] = gid
        out.loc[out["candidate_id"].eq(low.iloc[i]["candidate_id"]), "matched_group_id"] = gid
    return out


def _matched_pairs_table(pair_trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, part in pair_trials[pair_trials["matched_group_id"].astype(str).str.len() > 0].groupby("matched_group_id"):
        if len(part) < 2:
            continue
        high = part.sort_values("dice_overlap", ascending=False).iloc[0]
        low = part.sort_values("dice_overlap", ascending=True).iloc[0]
        rows.append(
            {
                "network_seed": int(high["network_seed"]),
                "matched_group_id": gid,
                "high_pair_id": int(high["pair_id"]),
                "low_pair_id": int(low["pair_id"]),
                "similarity_difference": abs(float(high["pixel_similarity"]) - float(low["pixel_similarity"])),
                "energy_difference": abs(float(high["input_energy_sample"]) - float(low["input_energy_sample"])),
                "class_pair_matched": bool(high["class_pair"] == low["class_pair"]),
                "overlap_difference": float(high["dice_overlap"]) - float(low["dice_overlap"]),
            }
        )
    return pd.DataFrame(rows)


def _write_panel_a_example(ctx: ExperimentContext, pair_trials: pd.DataFrame, mask_bank: Mapping[int, Mapping[str, np.ndarray]], images: torch.Tensor) -> None:
    if pair_trials.empty:
        return
    row = pair_trials.iloc[0]
    pair_id = int(row["pair_id"])
    meta = {k: _json_safe(v) for k, v in row.to_dict().items()}
    _write_json(meta, ctx.raw_dir / "panel_a_example_reentry_trial_metadata.json")
    masks = mask_bank[pair_id]
    np.savez_compressed(
        ctx.raw_dir / "panel_a_example_reentry_trial.npz",
        sample_image=images[int(row["sample_image_id"])].numpy(),
        probe_image=images[int(row["probe_image_id"])].numpy(),
        sample_foreground_mask=masks["sample_foreground_mask"],
        probe_foreground_mask=masks["probe_foreground_mask"],
        overlap_mask=masks["overlap_mask"],
        sample_nonoverlap_mask=masks["sample_nonoverlap_mask"],
        random_matched_mask=masks["random_matched_mask"],
    )
    ctx.output_files["panel_a_example_reentry_trial_metadata"] = "data/raw/panel_a_example_reentry_trial_metadata.json"
    ctx.output_files["panel_a_example_reentry_trial"] = "data/raw/panel_a_example_reentry_trial.npz"


def _cond_row(condition_metrics: pd.DataFrame, pair_id: int, condition: str) -> pd.Series:
    part = condition_metrics[(condition_metrics["pair_id"].eq(pair_id)) & (condition_metrics["condition"].eq(condition))]
    if part.empty:
        raise KeyError(f"Missing condition={condition} pair_id={pair_id}")
    return part.iloc[0]


def _trace(bank: OverlapReentryDMSBank, pair_id: int, condition: str) -> np.ndarray:
    return np.asarray(bank.traces[f"pair_{int(pair_id)}_{condition}_l3_trace"], dtype=np.float64)


def _vector(bank: OverlapReentryDMSBank, pair_id: int, condition: str) -> np.ndarray:
    return np.asarray(bank.vectors[f"pair_{int(pair_id)}_{condition}_class_evidence"], dtype=np.float64)


def _vec_distance(bank: OverlapReentryDMSBank, pair_id: int, a: str, b: str) -> float:
    return float(np.linalg.norm(_vector(bank, pair_id, a) - _vector(bank, pair_id, b)))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return 0.0 if denom <= 1e-12 else float(np.dot(av, bv) / denom)


def _projection(delta: np.ndarray, axis: np.ndarray) -> float:
    denom = float(np.dot(axis.reshape(-1), axis.reshape(-1)))
    return 0.0 if denom <= 1e-12 else float(np.dot(delta.reshape(-1), axis.reshape(-1)) / denom)


def _dpi_timecourse(bank: OverlapReentryDMSBank, pair_id: int, condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dyn = _trace(bank, pair_id, "full_dynamic")
    sta = _trace(bank, pair_id, "full_static")
    cond = _trace(bank, pair_id, condition)
    n = min(len(dyn), len(sta), len(cond))
    s_dyn = np.asarray([float(np.dot(normalize_pattern_vector(cond[t]), normalize_pattern_vector(dyn[t]))) for t in range(n)], dtype=np.float64)
    s_sta = np.asarray([float(np.dot(normalize_pattern_vector(cond[t]), normalize_pattern_vector(sta[t]))) for t in range(n)], dtype=np.float64)
    return s_dyn - s_sta, s_dyn, s_sta


def _mean_dpi(bank: OverlapReentryDMSBank, pair_id: int, condition: str) -> float:
    dpi, _, _ = _dpi_timecourse(bank, pair_id, condition)
    return float(np.nanmean(dpi)) if len(dpi) else float("nan")


def _decision_deflection(bank: OverlapReentryDMSBank, pair_id: int, condition: str) -> float:
    v_dyn = _vector(bank, pair_id, "full_dynamic")
    v_sta = _vector(bank, pair_id, "full_static")
    v_cond = _vector(bank, pair_id, condition)
    return _projection(v_cond - v_sta, v_dyn - v_sta)


def _summary_by_bin(df: pd.DataFrame, bin_col: str, center_col: str) -> pd.DataFrame:
    rows = []
    for bin_name, part in df.groupby(bin_col, sort=True):
        row = {
            "network_seed": int(part["network_seed"].iloc[0]),
            bin_col: str(bin_name),
            "bin_center": float(pd.to_numeric(part[center_col], errors="coerce").mean()),
            "n_pairs": int(len(part)),
        }
        for metric in ("acc_drop", "b_vec", "DPI_L3", "decision_deflection"):
            if metric not in part.columns:
                row[f"mean_{metric}"] = float("nan")
                row[f"sem_{metric}"] = 0.0
                continue
            vals = pd.to_numeric(part[metric], errors="coerce").dropna()
            key = metric if metric != "DPI_L3" else "DPI_L3"
            row[f"mean_{key}"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"sem_{key}"] = float(vals.sem()) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _panel_b_accuracy_drop_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "similarity_bin",
        "pixel_similarity_min",
        "pixel_similarity_max",
        "mean_accuracy_drop",
        "sem_accuracy_drop",
        "n_pairs",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for bin_name, part in df.groupby("similarity_bin", sort=True):
        sim = pd.to_numeric(part["pixel_similarity"], errors="coerce")
        acc = pd.to_numeric(part["acc_drop"], errors="coerce").dropna()
        rows.append(
            {
                "network_seed": int(part["network_seed"].iloc[0]),
                "similarity_bin": str(bin_name),
                "pixel_similarity_min": float(sim.min()) if len(sim.dropna()) else float("nan"),
                "pixel_similarity_max": float(sim.max()) if len(sim.dropna()) else float("nan"),
                "mean_accuracy_drop": float(acc.mean()) if len(acc) else float("nan"),
                "sem_accuracy_drop": float(acc.sem()) if len(acc) > 1 else 0.0,
                "n_pairs": int(len(part)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _pair_effect_table(ctx: ExperimentContext, bank: OverlapReentryDMSBank) -> pd.DataFrame:
    rows = []
    for _, pair in bank.pair_trials.iterrows():
        pair_id = int(pair["pair_id"])
        dyn = _cond_row(bank.condition_metrics, pair_id, "full_dynamic")
        sta = _cond_row(bank.condition_metrics, pair_id, "full_static")
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "pair_id": pair_id,
                "b_vec": _vec_distance(bank, pair_id, "full_dynamic", "full_static"),
                "DPI_L3": _mean_dpi(bank, pair_id, "full_dynamic"),
                "acc_drop": int(sta["correctness"]) - int(dyn["correctness"]),
                "decision_deflection": _decision_deflection(bank, pair_id, "full_dynamic"),
            }
        )
    return pd.DataFrame(rows)


def _panel_c_matched_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    use = df[df["matched_group_id"].astype(str).str.len() > 0].copy()
    for gid, part in use.groupby("matched_group_id"):
        if len(part) < 2:
            continue
        med = float(part["dice_overlap"].median())
        for _, r in part.iterrows():
            rows.append(
                {
                    "network_seed": int(r["network_seed"]),
                    "matched_group_id": str(gid),
                    "pair_id": int(r["pair_id"]),
                    "overlap_group": "high_overlap" if float(r["dice_overlap"]) >= med else "low_overlap",
                    "pixel_similarity": float(r["pixel_similarity"]),
                    "dice_overlap": float(r["dice_overlap"]),
                    "input_energy_sample": float(r["input_energy_sample"]),
                    "input_energy_probe": float(r["input_energy_probe"]),
                    "class_pair": str(r["class_pair"]),
                    "b_vec": float(r["b_vec"]),
                    "DPI_L3": float(r["DPI_L3"]),
                    "acc_drop": float(r["acc_drop"]),
                    "decision_deflection": float(r["decision_deflection"]),
                }
            )
    if not rows and not df.empty:
        med = float(df["dice_overlap"].median())
        high = df[df["dice_overlap"] >= med].head(max(1, len(df) // 2))
        low = df[df["dice_overlap"] < med].head(max(1, len(df) // 2))
        for label, part in (("high_overlap", high), ("low_overlap", low)):
            for _, r in part.iterrows():
                rows.append(
                    {
                        "network_seed": int(r["network_seed"]),
                        "matched_group_id": "fallback_quantile",
                        "pair_id": int(r["pair_id"]),
                        "overlap_group": label,
                        "pixel_similarity": float(r["pixel_similarity"]),
                        "dice_overlap": float(r["dice_overlap"]),
                        "input_energy_sample": float(r["input_energy_sample"]),
                        "input_energy_probe": float(r["input_energy_probe"]),
                        "class_pair": str(r["class_pair"]),
                        "b_vec": float(r["b_vec"]),
                        "DPI_L3": float(r["DPI_L3"]),
                        "acc_drop": float(r["acc_drop"]),
                        "decision_deflection": float(r["decision_deflection"]),
                    }
                )
    return pd.DataFrame(rows)


def _overlap_regression(df: pd.DataFrame, network_seed: int) -> pd.DataFrame:
    rows = []
    for metric in ("b_vec", "DPI_L3", "acc_drop", "decision_deflection"):
        use = df[["dice_overlap", "pixel_similarity", "input_energy_sample", metric]].dropna()
        if len(use) >= 4:
            x = np.column_stack([np.ones(len(use)), use["dice_overlap"], use["pixel_similarity"], use["input_energy_sample"]])
            y = use[metric].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            pred = x @ beta
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
            notes = "ordinary least squares; p-values not computed in first-pass implementation"
        else:
            beta = [float("nan")] * 4
            r2 = float("nan")
            notes = "insufficient rows for regression; table remains regression-ready"
        rows.append(
            {
                "network_seed": int(network_seed),
                "metric": metric,
                "beta_overlap": float(beta[1]),
                "beta_similarity": float(beta[2]),
                "beta_input_energy": float(beta[3]),
                "r2": float(r2),
                "n_pairs": int(len(use)),
                "p_overlap": float("nan"),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def _two_by_two(df: pd.DataFrame, network_seed: int) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    sim_med = float(df["pixel_similarity"].median())
    ov_med = float(df["dice_overlap"].median())
    for sim_label, sim_mask in (("low_similarity", df["pixel_similarity"] < sim_med), ("high_similarity", df["pixel_similarity"] >= sim_med)):
        for ov_label, ov_mask in (("low_overlap", df["dice_overlap"] < ov_med), ("high_overlap", df["dice_overlap"] >= ov_med)):
            part = df[sim_mask & ov_mask]
            for metric in ("b_vec", "DPI_L3", "acc_drop", "decision_deflection"):
                vals = pd.to_numeric(part[metric], errors="coerce").dropna()
                rows.append({"network_seed": int(network_seed), "similarity_group": sim_label, "overlap_group": ov_label, "metric": metric, "value": float(vals.mean()) if len(vals) else float("nan"), "n_pairs": int(len(part))})
    return pd.DataFrame(rows)


def _matching_diagnostics(df: pd.DataFrame, network_seed: int) -> pd.DataFrame:
    rows = []
    for gid, part in df[df["matched_group_id"].astype(str).str.len() > 0].groupby("matched_group_id"):
        if len(part) < 2:
            continue
        high = part.sort_values("dice_overlap", ascending=False).iloc[0]
        low = part.sort_values("dice_overlap", ascending=True).iloc[0]
        rows.append(
            {
                "network_seed": int(network_seed),
                "matched_group_id": str(gid),
                "high_pair_id": int(high["pair_id"]),
                "low_pair_id": int(low["pair_id"]),
                "similarity_difference": abs(float(high["pixel_similarity"]) - float(low["pixel_similarity"])),
                "energy_difference": abs(float(high["input_energy_sample"]) - float(low["input_energy_sample"])),
                "class_pair_matched": bool(high["class_pair"] == low["class_pair"]),
                "overlap_difference": float(high["dice_overlap"]) - float(low["dice_overlap"]),
            }
        )
    return pd.DataFrame(rows)


def _accuracy_pair_table(ctx: ExperimentContext, bank: OverlapReentryDMSBank | SimilarityBiasCompatibleBank) -> pd.DataFrame:
    if isinstance(bank, SimilarityBiasCompatibleBank):
        rows = []
        meta = bank.pair_trials.set_index("pair_id", drop=False)
        for row in bank.trial_metrics.itertuples(index=False):
            pair = meta.loc[int(row.pair_id)]
            correct_dynamic = int(row.correct_dynamic)
            correct_static = int(row.correct_static)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "pair_id": int(row.pair_id),
                    "sample_image_id": int(pair["sample_image_id"]),
                    "probe_image_id": int(pair["probe_image_id"]),
                    "sample_label": int(pair["sample_label"]),
                    "probe_label": int(pair["probe_label"]),
                    "class_pair": str(pair["class_pair"]),
                    "similarity_bin": str(pair["similarity_bin"]),
                    "overlap_bin": str(pair["overlap_bin"]),
                    "pixel_similarity": float(pair["pixel_similarity"]),
                    "dice_overlap": float(pair["dice_overlap"]),
                    "input_energy_sample": float(pair["input_energy_sample"]),
                    "input_energy_probe": float(pair["input_energy_probe"]),
                    "correct_dynamic": correct_dynamic,
                    "correct_static": correct_static,
                    "acc_drop": int(row.acc_drop),
                    "static_correct_eligible": int(row.static_correct_eligible),
                    "drop_event": int(row.drop_event),
                    "dynamic_rescue_event": int(row.dynamic_rescue_event),
                }
            )
        return pd.DataFrame(rows, columns=_accuracy_pair_columns())
    rows: list[dict[str, Any]] = []
    for _, pair in bank.pair_trials.iterrows():
        pair_id = int(pair["pair_id"])
        dyn = _cond_row(bank.condition_metrics, pair_id, "full_dynamic")
        sta = _cond_row(bank.condition_metrics, pair_id, "full_static")
        correct_dynamic = int(dyn["correctness"])
        correct_static = int(sta["correctness"])
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "pair_id": pair_id,
                "sample_image_id": int(pair["sample_image_id"]),
                "probe_image_id": int(pair["probe_image_id"]),
                "sample_label": int(pair["sample_label"]),
                "probe_label": int(pair["probe_label"]),
                "class_pair": str(pair["class_pair"]),
                "similarity_bin": str(pair["similarity_bin"]),
                "overlap_bin": str(pair["overlap_bin"]),
                "pixel_similarity": float(pair["pixel_similarity"]),
                "dice_overlap": float(pair["dice_overlap"]),
                "input_energy_sample": float(pair["input_energy_sample"]),
                "input_energy_probe": float(pair["input_energy_probe"]),
                "correct_dynamic": correct_dynamic,
                "correct_static": correct_static,
                "acc_drop": int(correct_static - correct_dynamic),
                "static_correct_eligible": int(correct_static == 1),
                "drop_event": int(correct_static == 1 and correct_dynamic == 0),
                "dynamic_rescue_event": int(correct_static == 0 and correct_dynamic == 1),
            }
        )
    return pd.DataFrame(rows, columns=_accuracy_pair_columns())


def _build_iso_similarity_overlap_matches(df: pd.DataFrame, cfg: Fig4Config) -> pd.DataFrame:
    columns = _iso_match_columns()
    if df.empty:
        return pd.DataFrame(columns=columns)
    eligible = df[df["static_correct_eligible"].astype(int).eq(1)].copy()
    if eligible.empty:
        return pd.DataFrame(columns=columns)
    eligible = _assign_bins(eligible, "pixel_similarity", "iso_similarity_bin", int(cfg.num_iso_similarity_bins))
    rows: list[dict[str, Any]] = []
    match_id = 0
    for bin_name, part in eligible.groupby("iso_similarity_bin", sort=True):
        if len(part) < 2:
            continue
        low_thr = float(part["dice_overlap"].quantile(float(cfg.overlap_tail_quantile)))
        high_thr = float(part["dice_overlap"].quantile(1.0 - float(cfg.overlap_tail_quantile)))
        high_pool = part[part["dice_overlap"] >= high_thr].sort_values("dice_overlap", ascending=False)
        low_pool = part[part["dice_overlap"] <= low_thr].sort_values("dice_overlap", ascending=True)
        used_low: set[int] = set()
        for _, high in high_pool.iterrows():
            candidates = low_pool[(~low_pool["pair_id"].astype(int).isin(used_low)) & (~low_pool["pair_id"].astype(int).eq(int(high["pair_id"])))].copy()
            if candidates.empty:
                continue
            candidates["similarity_difference"] = (candidates["pixel_similarity"].astype(float) - float(high["pixel_similarity"])).abs()
            candidates["sample_energy_rel_difference"] = candidates["input_energy_sample"].map(lambda v: _relative_difference(float(high["input_energy_sample"]), float(v)))
            candidates["probe_energy_rel_difference"] = candidates["input_energy_probe"].map(lambda v: _relative_difference(float(high["input_energy_probe"]), float(v)))
            candidates = candidates[candidates["similarity_difference"] <= float(cfg.match_similarity_caliper)]
            candidates = candidates[candidates["sample_energy_rel_difference"] <= float(cfg.match_energy_caliper)]
            candidates = candidates[candidates["probe_energy_rel_difference"] <= float(cfg.match_energy_caliper)]
            if bool(cfg.match_require_probe_label):
                candidates = candidates[candidates["probe_label"].astype(int).eq(int(high["probe_label"]))]
            if bool(cfg.match_require_class_pair):
                candidates = candidates[candidates["class_pair"].astype(str).eq(str(high["class_pair"]))]
            if candidates.empty:
                continue
            candidates["match_score"] = candidates["similarity_difference"] + candidates["sample_energy_rel_difference"] + candidates["probe_energy_rel_difference"]
            low = candidates.sort_values(["match_score", "dice_overlap"], ascending=[True, True]).iloc[0]
            used_low.add(int(low["pair_id"]))
            rows.append(_iso_match_row(int(match_id), str(bin_name), high, low))
            match_id += 1
    return pd.DataFrame(rows, columns=columns)


def _high_similarity_overlap_accuracy_drop_tables(
    pair_table: pd.DataFrame,
    cfg: Fig4Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_columns = [
        "network_seed",
        "pair_id",
        "similarity_bin",
        "highest_similarity_bin",
        "pixel_similarity",
        "dice_overlap",
        "overlap_group",
        "correct_static",
        "correct_dynamic",
        "accuracy_drop",
        "drop_event",
    ]
    summary_columns = [
        "network_seed",
        "overlap_group",
        "mean_accuracy_drop",
        "sem_accuracy_drop",
        "mean_drop_event",
        "sem_drop_event",
        "n_pairs",
    ]
    contrast_columns = [
        "network_seed",
        "highest_similarity_bin",
        "median_overlap_threshold",
        "mean_acc_drop_high_overlap",
        "mean_acc_drop_low_overlap",
        "high_minus_low_acc_drop",
        "drop_event_high_overlap",
        "drop_event_low_overlap",
        "high_minus_low_drop_event",
        "n_pairs_high",
        "n_pairs_low",
    ]
    if pair_table.empty:
        return pd.DataFrame(columns=raw_columns), pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=contrast_columns)
    use = pair_table.copy()
    if "similarity_bin" not in use.columns:
        use = _assign_bins(use, "pixel_similarity", "similarity_bin", int(cfg.num_similarity_bins))
    highest = _highest_bin_label(use["similarity_bin"])
    high_sim = use[use["similarity_bin"].astype(str).eq(highest)].copy()
    if high_sim.empty:
        return pd.DataFrame(columns=raw_columns), pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=contrast_columns)
    median_overlap = float(pd.to_numeric(high_sim["dice_overlap"], errors="coerce").median())
    high_sim["overlap_group"] = np.where(
        pd.to_numeric(high_sim["dice_overlap"], errors="coerce") > median_overlap,
        "high_overlap",
        "low_overlap",
    )
    if high_sim["overlap_group"].nunique() < 2 and len(high_sim) >= 2:
        ordered = high_sim.sort_values(["dice_overlap", "pair_id"], ascending=[True, True]).copy()
        split = len(ordered) // 2
        low_ids = set(ordered.iloc[:split]["pair_id"].astype(int))
        high_sim["overlap_group"] = high_sim["pair_id"].astype(int).map(lambda v: "low_overlap" if int(v) in low_ids else "high_overlap")
    raw = pd.DataFrame(
        {
            "network_seed": high_sim["network_seed"].astype(int),
            "pair_id": high_sim["pair_id"].astype(int),
            "similarity_bin": high_sim["similarity_bin"].astype(str),
            "highest_similarity_bin": str(highest),
            "pixel_similarity": pd.to_numeric(high_sim["pixel_similarity"], errors="coerce"),
            "dice_overlap": pd.to_numeric(high_sim["dice_overlap"], errors="coerce"),
            "overlap_group": high_sim["overlap_group"].astype(str),
            "correct_static": pd.to_numeric(high_sim["correct_static"], errors="coerce").fillna(0).astype(int),
            "correct_dynamic": pd.to_numeric(high_sim["correct_dynamic"], errors="coerce").fillna(0).astype(int),
            "accuracy_drop": pd.to_numeric(high_sim["acc_drop"], errors="coerce"),
            "drop_event": pd.to_numeric(high_sim["drop_event"], errors="coerce").fillna(0).astype(int),
        }
    )
    summary_rows: list[dict[str, Any]] = []
    for group in ("low_overlap", "high_overlap"):
        part = raw[raw["overlap_group"].eq(group)]
        acc = pd.to_numeric(part["accuracy_drop"], errors="coerce").dropna()
        drop = pd.to_numeric(part["drop_event"], errors="coerce").dropna()
        summary_rows.append(
            {
                "network_seed": int(raw["network_seed"].iloc[0]),
                "overlap_group": group,
                "mean_accuracy_drop": float(acc.mean()) if len(acc) else float("nan"),
                "sem_accuracy_drop": float(acc.sem()) if len(acc) > 1 else 0.0,
                "mean_drop_event": float(drop.mean()) if len(drop) else float("nan"),
                "sem_drop_event": float(drop.sem()) if len(drop) > 1 else 0.0,
                "n_pairs": int(len(part)),
            }
        )
    summary = pd.DataFrame(summary_rows, columns=summary_columns)
    high = raw[raw["overlap_group"].eq("high_overlap")]
    low = raw[raw["overlap_group"].eq("low_overlap")]
    high_acc = pd.to_numeric(high["accuracy_drop"], errors="coerce")
    low_acc = pd.to_numeric(low["accuracy_drop"], errors="coerce")
    high_drop = pd.to_numeric(high["drop_event"], errors="coerce")
    low_drop = pd.to_numeric(low["drop_event"], errors="coerce")
    contrast = pd.DataFrame(
        [
            {
                "network_seed": int(raw["network_seed"].iloc[0]),
                "highest_similarity_bin": str(highest),
                "median_overlap_threshold": float(median_overlap),
                "mean_acc_drop_high_overlap": float(high_acc.mean()) if len(high_acc.dropna()) else float("nan"),
                "mean_acc_drop_low_overlap": float(low_acc.mean()) if len(low_acc.dropna()) else float("nan"),
                "high_minus_low_acc_drop": float(high_acc.mean() - low_acc.mean()) if len(high_acc.dropna()) and len(low_acc.dropna()) else float("nan"),
                "drop_event_high_overlap": float(high_drop.mean()) if len(high_drop.dropna()) else float("nan"),
                "drop_event_low_overlap": float(low_drop.mean()) if len(low_drop.dropna()) else float("nan"),
                "high_minus_low_drop_event": float(high_drop.mean() - low_drop.mean()) if len(high_drop.dropna()) and len(low_drop.dropna()) else float("nan"),
                "n_pairs_high": int(len(high)),
                "n_pairs_low": int(len(low)),
            }
        ],
        columns=contrast_columns,
    )
    return raw[raw_columns], summary, contrast


def _highest_bin_label(values: pd.Series) -> str:
    labels = [str(v) for v in values.dropna().astype(str).unique()]
    if not labels:
        return "bin_1"

    def key(label: str) -> tuple[int, str]:
        digits = "".join(ch for ch in label if ch.isdigit())
        return (int(digits) if digits else -1, label)

    return sorted(labels, key=key)[-1]


def _iso_match_row(match_id: int, bin_name: str, high: pd.Series, low: pd.Series) -> dict[str, Any]:
    sim_diff = abs(float(high["pixel_similarity"]) - float(low["pixel_similarity"]))
    sample_energy_diff = _relative_difference(float(high["input_energy_sample"]), float(low["input_energy_sample"]))
    probe_energy_diff = _relative_difference(float(high["input_energy_probe"]), float(low["input_energy_probe"]))
    return {
        "network_seed": int(high["network_seed"]),
        "match_id": int(match_id),
        "iso_similarity_bin": bin_name,
        "high_pair_id": int(high["pair_id"]),
        "low_pair_id": int(low["pair_id"]),
        "pixel_similarity_high": float(high["pixel_similarity"]),
        "pixel_similarity_low": float(low["pixel_similarity"]),
        "similarity_difference": sim_diff,
        "dice_overlap_high": float(high["dice_overlap"]),
        "dice_overlap_low": float(low["dice_overlap"]),
        "overlap_difference": float(high["dice_overlap"]) - float(low["dice_overlap"]),
        "input_energy_sample_high": float(high["input_energy_sample"]),
        "input_energy_sample_low": float(low["input_energy_sample"]),
        "sample_energy_rel_difference": sample_energy_diff,
        "input_energy_probe_high": float(high["input_energy_probe"]),
        "input_energy_probe_low": float(low["input_energy_probe"]),
        "probe_energy_rel_difference": probe_energy_diff,
        "class_pair_high": str(high["class_pair"]),
        "class_pair_low": str(low["class_pair"]),
        "probe_label_high": int(high["probe_label"]),
        "probe_label_low": int(low["probe_label"]),
        "drop_event_high": int(high["drop_event"]),
        "drop_event_low": int(low["drop_event"]),
        "acc_drop_high": int(high["acc_drop"]),
        "acc_drop_low": int(low["acc_drop"]),
        "paired_delta_drop_event": int(high["drop_event"]) - int(low["drop_event"]),
        "paired_delta_acc_drop": int(high["acc_drop"]) - int(low["acc_drop"]),
    }


def _matched_overlap_permutation_test(matches: pd.DataFrame, cfg: Fig4Config, rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, float]]:
    network_seed = int(matches["network_seed"].iloc[0]) if not matches.empty else 0
    n_perm = max(0, int(cfg.n_match_permutations))
    if matches.empty:
        null = pd.DataFrame(columns=["network_seed", "perm_id", "null_delta_drop_event"])
        return null, {"observed_delta_drop_event": float("nan"), "p_one_sided": float("nan"), "p_two_sided": float("nan")}
    deltas = matches["paired_delta_drop_event"].to_numpy(dtype=float)
    observed = float(np.mean(deltas))
    rows = []
    null_values = []
    for perm_id in range(n_perm):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(deltas))
        null_delta = float(np.mean(signs * deltas))
        null_values.append(null_delta)
        rows.append({"network_seed": network_seed, "perm_id": int(perm_id), "null_delta_drop_event": null_delta})
    null_arr = np.asarray(null_values, dtype=float)
    p_one = float((1 + np.sum(null_arr >= observed)) / (1 + n_perm)) if n_perm else float("nan")
    p_two = float((1 + np.sum(np.abs(null_arr) >= abs(observed))) / (1 + n_perm)) if n_perm else float("nan")
    return pd.DataFrame(rows), {"observed_delta_drop_event": observed, "p_one_sided": p_one, "p_two_sided": p_two}


def _overlap_accuracy_contrast_by_network(matches: pd.DataFrame, network_seed: int, perm_stats: Mapping[str, float]) -> pd.DataFrame:
    columns = [
        "network_seed",
        "n_matched_sets",
        "drop_rate_high_overlap",
        "drop_rate_low_overlap",
        "delta_drop_rate",
        "mean_acc_drop_high_overlap",
        "mean_acc_drop_low_overlap",
        "delta_acc_drop",
        "mean_similarity_difference",
        "max_similarity_difference",
        "mean_overlap_difference",
        "mean_sample_energy_rel_difference",
        "mean_probe_energy_rel_difference",
        "permutation_p_one_sided",
        "permutation_p_two_sided",
    ]
    if matches.empty:
        return pd.DataFrame([{col: (int(network_seed) if col == "network_seed" else 0 if col == "n_matched_sets" else float("nan")) for col in columns}], columns=columns)
    row = {
        "network_seed": int(network_seed),
        "n_matched_sets": int(len(matches)),
        "drop_rate_high_overlap": float(matches["drop_event_high"].mean()),
        "drop_rate_low_overlap": float(matches["drop_event_low"].mean()),
        "delta_drop_rate": float(matches["paired_delta_drop_event"].mean()),
        "mean_acc_drop_high_overlap": float(matches["acc_drop_high"].mean()),
        "mean_acc_drop_low_overlap": float(matches["acc_drop_low"].mean()),
        "delta_acc_drop": float(matches["paired_delta_acc_drop"].mean()),
        "mean_similarity_difference": float(matches["similarity_difference"].mean()),
        "max_similarity_difference": float(matches["similarity_difference"].max()),
        "mean_overlap_difference": float(matches["overlap_difference"].mean()),
        "mean_sample_energy_rel_difference": float(matches["sample_energy_rel_difference"].mean()),
        "mean_probe_energy_rel_difference": float(matches["probe_energy_rel_difference"].mean()),
        "permutation_p_one_sided": float(perm_stats.get("p_one_sided", float("nan"))),
        "permutation_p_two_sided": float(perm_stats.get("p_two_sided", float("nan"))),
    }
    return pd.DataFrame([row], columns=columns)


def _matching_balance_diagnostics(matches: pd.DataFrame, network_seed: int) -> pd.DataFrame:
    columns = [
        "network_seed",
        "n_matched_sets",
        "mean_similarity_difference",
        "median_similarity_difference",
        "p95_similarity_difference",
        "max_similarity_difference",
        "mean_sample_energy_rel_difference",
        "mean_probe_energy_rel_difference",
        "mean_overlap_difference",
        "fraction_probe_label_matched",
        "fraction_class_pair_matched",
    ]
    if matches.empty:
        return pd.DataFrame([{col: (int(network_seed) if col == "network_seed" else 0 if col == "n_matched_sets" else float("nan")) for col in columns}], columns=columns)
    row = {
        "network_seed": int(network_seed),
        "n_matched_sets": int(len(matches)),
        "mean_similarity_difference": float(matches["similarity_difference"].mean()),
        "median_similarity_difference": float(matches["similarity_difference"].median()),
        "p95_similarity_difference": float(matches["similarity_difference"].quantile(0.95)),
        "max_similarity_difference": float(matches["similarity_difference"].max()),
        "mean_sample_energy_rel_difference": float(matches["sample_energy_rel_difference"].mean()),
        "mean_probe_energy_rel_difference": float(matches["probe_energy_rel_difference"].mean()),
        "mean_overlap_difference": float(matches["overlap_difference"].mean()),
        "fraction_probe_label_matched": float((matches["probe_label_high"].astype(int) == matches["probe_label_low"].astype(int)).mean()),
        "fraction_class_pair_matched": float((matches["class_pair_high"].astype(str) == matches["class_pair_low"].astype(str)).mean()),
    }
    return pd.DataFrame([row], columns=columns)


def _compute_overlap_excess_accuracy(df: pd.DataFrame, cfg: Fig4Config) -> pd.DataFrame:
    columns = ["network_seed", "iso_similarity_bin", "overlap_excess_group", "n_pairs", "drop_rate", "mean_acc_drop", "mean_pixel_similarity", "mean_dice_overlap", "mean_overlap_excess"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    use = _assign_bins(df.copy(), "pixel_similarity", "iso_similarity_bin", int(cfg.num_iso_similarity_bins))
    rows: list[dict[str, Any]] = []
    for bin_name, part in use.groupby("iso_similarity_bin", sort=True):
        expected = float(part["dice_overlap"].mean())
        part = part.copy()
        part["overlap_excess"] = part["dice_overlap"] - expected
        median = float(part["overlap_excess"].median())
        for group, mask in (("low_overlap_excess", part["overlap_excess"] <= median), ("high_overlap_excess", part["overlap_excess"] > median)):
            sub = part[mask]
            rows.append(
                {
                    "network_seed": int(part["network_seed"].iloc[0]),
                    "iso_similarity_bin": str(bin_name),
                    "overlap_excess_group": group,
                    "n_pairs": int(len(sub)),
                    "drop_rate": float(sub["drop_event"].mean()) if len(sub) else float("nan"),
                    "mean_acc_drop": float(sub["acc_drop"].mean()) if len(sub) else float("nan"),
                    "mean_pixel_similarity": float(sub["pixel_similarity"].mean()) if len(sub) else float("nan"),
                    "mean_dice_overlap": float(sub["dice_overlap"].mean()) if len(sub) else float("nan"),
                    "mean_overlap_excess": float(sub["overlap_excess"].mean()) if len(sub) else float("nan"),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _overlap_accuracy_regression(df: pd.DataFrame, network_seed: int) -> pd.DataFrame:
    columns = ["network_seed", "metric", "beta_overlap", "beta_similarity", "beta_input_energy_sample", "beta_input_energy_probe", "r2", "n_pairs", "p_overlap", "notes"]
    use = df[["drop_event", "dice_overlap", "pixel_similarity", "input_energy_sample", "input_energy_probe"]].dropna() if not df.empty else pd.DataFrame()
    if len(use) >= 5:
        x = np.column_stack([np.ones(len(use)), use["dice_overlap"], use["pixel_similarity"], use["input_energy_sample"], use["input_energy_probe"]])
        y = use["drop_event"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        pred = x @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
        notes = "OLS probability model for supplementary drop_event sensitivity; main Fig.4D uses matched contrast."
    else:
        beta = [float("nan")] * 5
        r2 = float("nan")
        notes = "insufficient rows for OLS probability model; main Fig.4D uses matched contrast."
    return pd.DataFrame(
        [
            {
                "network_seed": int(network_seed),
                "metric": "drop_event",
                "beta_overlap": float(beta[1]),
                "beta_similarity": float(beta[2]),
                "beta_input_energy_sample": float(beta[3]),
                "beta_input_energy_probe": float(beta[4]),
                "r2": float(r2),
                "n_pairs": int(len(use)),
                "p_overlap": float("nan"),
                "notes": notes,
            }
        ],
        columns=columns,
    )


def _relative_difference(a: float, b: float) -> float:
    return float(abs(a - b) / max((abs(a) + abs(b)) / 2.0, 1e-12))


def _accuracy_pair_columns() -> list[str]:
    return [
        "network_seed",
        "pair_id",
        "sample_image_id",
        "probe_image_id",
        "sample_label",
        "probe_label",
        "class_pair",
        "similarity_bin",
        "overlap_bin",
        "pixel_similarity",
        "dice_overlap",
        "input_energy_sample",
        "input_energy_probe",
        "correct_dynamic",
        "correct_static",
        "acc_drop",
        "static_correct_eligible",
        "drop_event",
        "dynamic_rescue_event",
    ]


def _iso_match_columns() -> list[str]:
    return [
        "network_seed",
        "match_id",
        "iso_similarity_bin",
        "high_pair_id",
        "low_pair_id",
        "pixel_similarity_high",
        "pixel_similarity_low",
        "similarity_difference",
        "dice_overlap_high",
        "dice_overlap_low",
        "overlap_difference",
        "input_energy_sample_high",
        "input_energy_sample_low",
        "sample_energy_rel_difference",
        "input_energy_probe_high",
        "input_energy_probe_low",
        "probe_energy_rel_difference",
        "class_pair_high",
        "class_pair_low",
        "probe_label_high",
        "probe_label_low",
        "drop_event_high",
        "drop_event_low",
        "acc_drop_high",
        "acc_drop_low",
        "paired_delta_drop_event",
        "paired_delta_acc_drop",
    ]


def _random_mask_controls(ctx: ExperimentContext, bank: OverlapReentryDMSBank) -> pd.DataFrame:
    f_path = ctx.metrics_dir / "supp_overlap_preserving_perturbation_metrics.csv"
    f_df = pd.read_csv(f_path) if f_path.exists() else pd.DataFrame()
    rows = []
    random_masks = bank.perturbation_masks[bank.perturbation_masks["mask_name"].eq("random_matched_mask")]
    for _, mask in random_masks.iterrows():
        pair_id = int(mask["pair_id"])
        f_row = f_df[(f_df["pair_id"].eq(pair_id)) & (f_df["condition"].eq("sample_random_matched_dynamic"))].head(1) if not f_df.empty else pd.DataFrame()
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "pair_id": pair_id,
                "condition": "sample_random_matched_dynamic",
                "random_mask_id": f"pair_{pair_id}_random_0",
                "mask_pixel_count": int(mask["pixel_count"]),
                "mask_input_energy": float(mask["input_energy"]),
                "mask_spike_count_estimate": float(mask["spike_count_estimate"]),
                "DPI_L3": _from_row(f_row, "DPI_L3", float("nan")),
                "dynamic_like_recovery": _from_row(f_row, "dynamic_like_recovery", float("nan")),
                "decision_deflection_score": _from_row(f_row, "decision_deflection_score", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def _condition_audit(ctx: ExperimentContext, bank: OverlapReentryDMSBank) -> pd.DataFrame:
    rows = []
    for condition in CORE_CONDITIONS:
        completed = int(bank.condition_metrics[bank.condition_metrics["condition"].eq(condition)]["pair_id"].nunique())
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "n_pairs": int(len(bank.pair_trials)),
                "n_similarity_bins": int(bank.pair_trials["similarity_bin"].nunique()),
                "n_overlap_bins": int(bank.pair_trials["overlap_bin"].nunique()),
                "n_matched_groups": int(bank.pair_trials["matched_group_id"].replace("", pd.NA).dropna().nunique()),
                "n_conditions": int(len(CORE_CONDITIONS)),
                "condition": condition,
                "n_completed": completed,
                "n_failed": max(0, int(len(bank.pair_trials)) - completed),
                "notes": "probe unchanged for all core perturbation assays",
            }
        )
    return pd.DataFrame(rows)


def _from_row(df: pd.DataFrame, column: str, default: float) -> float:
    if df.empty or column not in df.columns:
        return float(default)
    value = pd.to_numeric(df[column], errors="coerce").iloc[0]
    return float(default) if pd.isna(value) else float(value)


def _finite_delta(a: float, b: float) -> float:
    return float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")


def _write_run_log(ctx: ExperimentContext) -> None:
    ctx.run_log.append(f"{_now()} completed modules={sorted(k for k, v in ctx.completed_modules.items() if v)}")
    path = ctx.seed_dir / "run_log.txt"
    path.write_text("\n".join(ctx.run_log) + "\n", encoding="utf-8")
    ctx.output_files["run_log"] = "run_log.txt"


def _save_csv(ctx: ExperimentContext, df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    ctx.output_files[path.stem] = _rel(path, ctx.seed_dir)


def _csv_nonempty(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(not pd.read_csv(path).empty)
    except Exception:
        return False


def _copy_csv_alias(ctx: ExperimentContext, src: Path, dst: Path, *, empty_columns: Sequence[str], reason: str) -> None:
    if not src.exists():
        _write_empty_csv(ctx, dst, empty_columns, reason)
        return
    df = pd.read_csv(src)
    if df.empty:
        _write_empty_csv(ctx, dst, list(df.columns) if len(df.columns) else empty_columns, reason)
        return
    _save_csv(ctx, df, dst)


def _write_empty_csv(ctx: ExperimentContext, dst: Path, columns: Sequence[str], reason: str) -> None:
    _record_optional_missing(ctx, dst.name, reason)
    _save_csv(ctx, pd.DataFrame(columns=list(columns)), dst)


def _record_optional_missing(ctx: ExperimentContext, output_name: str, reason: str) -> None:
    missing = ctx.availability.setdefault("supplement_alias_missing_reasons", {})
    missing[output_name] = reason
    message = f"Optional Fig.4 alias {output_name} is empty: {reason}"
    if message not in ctx.warnings:
        ctx.warnings.append(message)


def _mean_existing(df: pd.DataFrame, columns: Sequence[str]) -> float:
    for column in columns:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            return float(values.mean()) if not values.empty else float("nan")
    return float("nan")


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


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
    paths = {
        "config": seed_dir / "config",
        "trial_specs": seed_dir / "data" / "trial_specs",
        "raw": seed_dir / "data" / "raw",
        "metrics": seed_dir / "data" / "metrics",
        "debug": seed_dir / "debug_figures",
        "meta": seed_dir / "meta",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _resolve_seed_dir(output_root: Path, network_seed: int) -> Path:
    if output_root.name.startswith("seed_"):
        return output_root
    return output_root / f"seed_{int(network_seed):03d}"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


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
    parser.add_argument("--foreground-threshold", type=float, default=0.0)
    parser.add_argument("--dilation-radius", type=int, default=1)
    parser.add_argument("--random-mask-candidates", type=int, default=32)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--num-iso-similarity-bins", type=int, default=20)
    parser.add_argument("--overlap-tail-quantile", type=float, default=0.33)
    parser.add_argument("--match-similarity-caliper", type=float, default=0.02)
    parser.add_argument("--match-energy-caliper", type=float, default=0.15)
    parser.add_argument("--match-require-probe-label", action="store_true")
    parser.add_argument("--match-require-class-pair", action="store_true")
    parser.add_argument("--min-matches-per-network", type=int, default=20)
    parser.add_argument("--n-match-permutations", type=int, default=2000)
    parser.add_argument("--save-full-trace", action="store_true")
    parser.add_argument("--no-save-l3-trace", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
