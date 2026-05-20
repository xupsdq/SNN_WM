from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
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
FIG3_DESIGN_VERSION = "multiitem_peak_landscape_singleton_weakprobe_region_ping"


@dataclass(frozen=True)
class Fig3Config:
    model_path: str
    dataset_root: str
    output_root: str
    network_seed: int
    device: str = "auto"
    split: str = "test"
    dt: float = 0.001
    sequence_lengths: tuple[int, ...] = (10,)
    primary_sequence_length: int = 10
    main_sequence_length: int = 10
    main_only_seq_len_10: bool = True
    sample_ms: int = 200
    delay_ms: int = 200
    ping_ms: int = 30
    ping_amp: float = 1.0
    ping_repeats: int = 1
    ping_main_state_conditions: tuple[str, ...] = ("S_final", "S0")
    weak_probe_ms: int = 100
    weak_probe_keep_probs: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    weak_probe_repeats: int = 5
    weak_probe_mask_space: str = "encoded_spikes"
    weak_probe_use_same_mask_across_states: bool = True
    weak_probe_scale: float = 0.35
    weak_probe_noise: float = 0.0
    weak_probe_metric_mode: str = "fig2_compat"
    weak_probe_target_source: str = "sequence_member_random"
    weak_probe_memory_scope: str = "final_only"
    num_sequences: int = 100
    batch_size: int = 16
    peak_q: float = 0.20
    valley_q: float = 0.20
    n_null: int = 100
    weak_cue_target_source: str = "sequence_member_random"
    weak_cue_keep_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    weak_cue_repeats: int = 5
    weak_cue_mask_mode: str = "rank_within_target_foreground"
    foreground_threshold: float = 0.1
    partial_cue_keep_fraction: float = 0.10
    partial_cue_keep_fraction_sweep: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3)
    partial_cue_repeats: int = 20
    target_position: str = "K-1"
    run_state_bank: bool = False
    run_progressive_update: bool = False
    run_peak_valley_landscape: bool = False
    run_neutral_ping: bool = False
    run_weak_probe: bool = False
    run_region_ping: bool = False
    run_region_ping_s0_control: bool = False
    run_region_ping_amp_sweep: bool = False
    run_peak_aligned_completion: bool = False
    run_peak_cue_main: bool = False
    run_population_morphology_supplement: bool = False
    run_structural_weak_cue: bool = False
    run_structural_weak_cue_supplement: bool = False
    run_supplement: bool = False
    save_debug_figures: bool = False
    save_spike_cache: bool = False
    save_all_layer_state_bank: bool = False
    show_progress: bool = True
    use_encode_cache: bool = True
    enable_condition_batch: bool = False
    smoke: bool = False
    peak_cue_main_keep_fraction: float = 0.10
    region_ping_q: float = 0.10
    region_ping_support_metric: str = "delta_gain_map"
    region_ping_conditions: tuple[str, ...] = ("peak", "valley", "random")
    region_ping_repeats: int = 5
    region_ping_amp_sweep: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5)
    region_ping_use_random_matched: bool = True
    weak_probe_include_singleton: bool = True

    @property
    def sample_steps(self) -> int:
        return _ms_to_steps(self.sample_ms, self.dt)

    @property
    def delay_steps(self) -> int:
        return _ms_to_steps(self.delay_ms, self.dt)

    @property
    def ping_steps(self) -> int:
        return _ms_to_steps(self.ping_ms, self.dt)

    @property
    def weak_probe_steps(self) -> int:
        return _ms_to_steps(self.weak_probe_ms, self.dt)


@dataclass
class ExperimentContext:
    cfg: Fig3Config
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
    n_sequences: int = 0


@dataclass
class MultiItemSequenceLandscapeBank:
    sequence_trials: pd.DataFrame
    sequence_meta: pd.DataFrame
    arrays: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]]
    singleton_refs: dict[int, dict[int, dict[str, dict[str, np.ndarray]]]]
    singleton_boundaries: dict[int, dict[int, Mapping[str, Mapping[str, torch.Tensor]]]]
    boundaries: dict[int, dict[str, Mapping[str, Mapping[str, torch.Tensor]]]]
    landscapes: dict[int, dict[str, np.ndarray]]

    def get(self, sequence_id: int, state: str, layer: str, variable: str) -> np.ndarray:
        if variable == "g":
            return self.arrays[int(sequence_id)][state][layer]["g"]
        return self.arrays[int(sequence_id)][state][layer][variable]


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
        _write_config_files(ctx)
        seq_trials, singleton_trials, partial_trials = build_sequence_trial_specs(ctx)
        bank: MultiItemSequenceLandscapeBank | None = None
        if any(
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
        ):
            bank = run_multiitem_sequence_state_bank(ctx, seq_trials)
        if bank is not None and cfg.run_progressive_update:
            compute_progressive_update_metrics(ctx, bank)
        if bank is not None and (cfg.run_peak_valley_landscape or cfg.run_population_morphology_supplement or cfg.run_supplement):
            compute_final_support_landscape(ctx, bank)
        if bank is not None and cfg.run_neutral_ping:
            run_neutral_ping_readout_distribution(ctx, bank)
        if bank is not None and cfg.run_weak_probe:
            run_sequence_weak_probe_real_rollout_from_state_bank(ctx, bank)
        if bank is not None and cfg.run_region_ping:
            run_region_gated_ping_readout(ctx, bank)
        if bank is not None and cfg.run_peak_cue_main:
            run_peak_cue_main_from_state_bank(ctx, bank)
        if bank is not None and cfg.run_structural_weak_cue_supplement:
            ensure_structural_weak_cue_outputs(ctx, bank)
        if bank is not None and cfg.run_supplement:
            compute_supplementary_metrics(ctx, bank)
        if cfg.save_debug_figures:
            save_debug_figures(ctx)
        summary = _write_summary(ctx)
        _write_run_log(ctx)
        finalize_run_info(seed_dir / "meta", run_info, status="success")
        return summary
    except Exception:
        finalize_run_info(seed_dir / "meta", run_info, status="failed")
        raise


def build_sequence_trial_specs(ctx: ExperimentContext) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = ctx.cfg
    rng = np.random.default_rng(int(cfg.network_seed))
    lengths = list(cfg.sequence_lengths)
    rows: list[dict[str, Any]] = []
    singleton_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    for sequence_id in _progress(range(int(cfg.num_sequences)), total=int(cfg.num_sequences), desc="fig3 sequence specs", enabled=cfg.show_progress):
        seq_len = int(lengths[sequence_id % len(lengths)])
        labels = rng.choice(np.arange(NUM_CLASSES), size=seq_len, replace=seq_len > NUM_CLASSES)
        image_ids = [int(rng.choice(ctx.class_index[int(label)])) for label in labels]
        sims = _pairwise_image_sims(ctx.dataset, image_ids)
        sequence_seed = int(rng.integers(0, 2**31 - 1))
        for stage_k, (image_id, label) in enumerate(zip(image_ids, labels), start=1):
            rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "sequence_id": int(sequence_id),
                    "seq_len": int(seq_len),
                    "stage_k": int(stage_k),
                    "item_image_id": int(image_id),
                    "item_label": int(label),
                    "ordered_item_ids": ";".join(str(v) for v in image_ids),
                    "ordered_item_labels": ";".join(str(int(v)) for v in labels),
                    "sequence_seed": sequence_seed,
                    "mean_pairwise_image_similarity": float(np.mean(sims)) if sims else 0.0,
                    "max_pairwise_image_similarity": float(np.max(sims)) if sims else 0.0,
                    "min_pairwise_image_similarity": float(np.min(sims)) if sims else 0.0,
                }
            )
            singleton_rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "sequence_id": int(sequence_id),
                    "seq_len": int(seq_len),
                    "reference_position": int(stage_k),
                    "reference_image_id": int(image_id),
                    "reference_label": int(label),
                    "temporal_slot": int(stage_k),
                    "reference_seed": int(sequence_seed + stage_k),
                }
            )
        target_position = _target_position(seq_len, cfg.target_position)
        partial_rows.append(
            {
                "network_seed": int(cfg.network_seed),
                "sequence_id": int(sequence_id),
                "seq_len": int(seq_len),
                "target_position": int(target_position),
                "target_image_id": int(image_ids[target_position - 1]),
                "target_label": int(labels[target_position - 1]),
                "keep_fraction": float(cfg.partial_cue_keep_fraction),
            }
        )
    seq_trials = pd.DataFrame(rows)
    singleton_trials = pd.DataFrame(singleton_rows)
    partial_trials = pd.DataFrame(partial_rows)
    _save_csv(ctx, seq_trials, ctx.trial_specs_dir / "sequence_trials.csv")
    _save_csv(ctx, singleton_trials, ctx.trial_specs_dir / "singleton_reference_trials.csv")
    _save_csv(ctx, partial_trials, ctx.trial_specs_dir / "partial_cue_trials.csv")
    _save_csv(ctx, _trial_condition_audit(ctx.cfg.network_seed, seq_trials), ctx.metrics_dir / "supp_trial_condition_audit.csv")
    example = seq_trials[seq_trials["sequence_id"] == int(seq_trials["sequence_id"].iloc[0])].copy()
    _write_json(_json_safe(example.iloc[0].to_dict()), ctx.raw_dir / "panel_a_example_sequence_metadata.json")
    np.savez_compressed(
        ctx.raw_dir / "panel_a_example_sequence.npz",
        image_ids=example["item_image_id"].to_numpy(dtype=np.int64),
        labels=example["item_label"].to_numpy(dtype=np.int64),
    )
    ctx.output_files["panel_a_example_sequence_metadata"] = _rel(ctx.raw_dir / "panel_a_example_sequence_metadata.json", ctx.seed_dir)
    ctx.output_files["panel_a_example_sequence"] = _rel(ctx.raw_dir / "panel_a_example_sequence.npz", ctx.seed_dir)
    ctx.n_sequences = int(seq_trials["sequence_id"].nunique())
    ctx.completed_modules["sequence_trial_specs"] = True
    return seq_trials, singleton_trials, partial_trials


def run_multiitem_sequence_state_bank(ctx: ExperimentContext, sequence_trials: pd.DataFrame) -> MultiItemSequenceLandscapeBank:
    cfg = ctx.cfg
    arrays: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    singleton_refs: dict[int, dict[int, dict[str, dict[str, np.ndarray]]]] = {}
    singleton_boundaries: dict[int, dict[int, Mapping[str, Mapping[str, torch.Tensor]]]] = {}
    boundaries: dict[int, dict[str, Mapping[str, Mapping[str, torch.Tensor]]]] = {}
    landscapes: dict[int, dict[str, np.ndarray]] = {}
    manifest_rows: list[dict[str, Any]] = []
    l1_payload: dict[str, np.ndarray] = {}
    l3_payload: dict[str, np.ndarray] = {}
    meta_rows: list[dict[str, Any]] = []

    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    groups = list(sequence_trials.groupby("sequence_id", sort=True))
    for sequence_id, group in _progress(groups, total=len(groups), desc="fig3 state sequences", enabled=cfg.show_progress):
        seq_id = int(sequence_id)
        group = group.sort_values("stage_k")
        seq_len = int(group["seq_len"].iloc[0])
        image_ids = group["item_image_id"].to_numpy(dtype=np.int64).tolist()
        labels = group["item_label"].to_numpy(dtype=np.int64).tolist()
        spikes = _encode_cached(ctx, image_ids, cfg.sample_steps, cache=encode_cache)
        state_arrays, state_boundaries = _capture_sequence(ctx, spikes)
        refs, ref_boundaries = _capture_singleton_refs_and_boundaries(ctx, spikes)
        arrays[seq_id] = state_arrays
        singleton_refs[seq_id] = refs
        singleton_boundaries[seq_id] = ref_boundaries
        boundaries[seq_id] = {"S0": state_boundaries["S0"], "S_final": state_boundaries["S_final"]}
        landscapes[seq_id] = _landscape_for_sequence(ctx, state_arrays, group)
        meta_rows.append({"sequence_id": seq_id, "seq_len": seq_len, "ordered_item_ids": ";".join(map(str, image_ids)), "ordered_item_labels": ";".join(map(str, labels))})
        for state, layer_map in state_arrays.items():
            stage_k = 0 if state == "S0" else (seq_len if state == "S_final" else int(state.split("_")[1]))
            for layer in ("layer1", "layer3"):
                for variable in STATE_VARIABLES:
                    arr = layer_map[layer][variable].astype(np.float32, copy=False)
                    key_state = state.replace("_", "")
                    storage_file = "state_bank_layer1.npz" if layer == "layer1" else "state_bank_layer3.npz"
                    storage_key = f"sequence_{seq_id}_{key_state}_{variable}"
                    manifest_rows.append(
                        {
                            "network_seed": int(cfg.network_seed),
                            "sequence_id": seq_id,
                            "seq_len": seq_len,
                            "state_condition": state,
                            "stage_k": stage_k,
                            "layer": layer,
                            "state_variable": variable,
                            "shape": "x".join(str(v) for v in arr.shape),
                            "storage_file": storage_file,
                            "storage_key": storage_key,
                            "captured_after": "item_delay" if state not in {"S0", "S_final"} else state,
                            "sample_ms": int(cfg.sample_ms),
                            "delay_ms": int(cfg.delay_ms),
                        }
                    )
                    if layer == "layer1":
                        l1_payload[storage_key] = arr
                    else:
                        l3_payload[storage_key] = arr
        for pos, ref in refs.items():
            for layer in ("layer1", "layer3"):
                for variable in STATE_VARIABLES:
                    arr = ref[layer][variable].astype(np.float32, copy=False)
                    storage_file = "state_bank_layer1.npz" if layer == "layer1" else "state_bank_layer3.npz"
                    storage_key = f"sequence_{seq_id}_singleton_reference_{pos}_{variable}"
                    manifest_rows.append(
                        {
                            "network_seed": int(cfg.network_seed),
                            "sequence_id": seq_id,
                            "seq_len": seq_len,
                            "state_condition": "singleton_reference",
                            "stage_k": int(pos),
                            "layer": layer,
                            "state_variable": variable,
                            "shape": "x".join(str(v) for v in arr.shape),
                            "storage_file": storage_file,
                            "storage_key": storage_key,
                            "captured_after": f"temporal_slot_{pos}",
                            "sample_ms": int(cfg.sample_ms),
                            "delay_ms": int(cfg.delay_ms),
                        }
                    )
                    if layer == "layer1":
                        l1_payload[storage_key] = arr
                    else:
                        l3_payload[storage_key] = arr
            manifest_rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "sequence_id": seq_id,
                    "seq_len": seq_len,
                    "state_condition": "singleton_boundary",
                    "stage_k": int(pos),
                    "layer": "",
                    "state_variable": "",
                    "shape": "",
                    "storage_file": "",
                    "storage_key": "",
                    "captured_after": f"temporal_slot_{pos}_singleton_end",
                    "sample_ms": int(cfg.sample_ms),
                    "delay_ms": int(cfg.delay_ms),
                    "restore_mode": "reset_all_state_restore_selected_stsp",
                }
            )

    np.savez_compressed(ctx.raw_dir / "state_bank_layer1.npz", **l1_payload)
    np.savez_compressed(ctx.raw_dir / "state_bank_layer3.npz", **l3_payload)
    _save_csv(ctx, pd.DataFrame(manifest_rows), ctx.raw_dir / "state_bank_manifest.csv")
    ctx.output_files["state_bank_layer1"] = _rel(ctx.raw_dir / "state_bank_layer1.npz", ctx.seed_dir)
    ctx.output_files["state_bank_layer3"] = _rel(ctx.raw_dir / "state_bank_layer3.npz", ctx.seed_dir)
    bank = MultiItemSequenceLandscapeBank(
        sequence_trials=sequence_trials.reset_index(drop=True),
        sequence_meta=pd.DataFrame(meta_rows),
        arrays=arrays,
        singleton_refs=singleton_refs,
        singleton_boundaries=singleton_boundaries,
        boundaries=boundaries,
        landscapes=landscapes,
    )
    _save_example_landscape(ctx, bank)
    ctx.completed_modules["state_bank"] = True
    return bank


def compute_progressive_update_metrics(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    rows: list[dict[str, Any]] = []
    for _, meta in _progress(bank.sequence_meta.iterrows(), total=len(bank.sequence_meta), desc="fig3 progressive sequences", enabled=ctx.cfg.show_progress):
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        for layer in LAYER_KEYS:
            for variable in ("g", "u", "x"):
                prev = bank.get(seq_id, "S0", layer, variable)
                prev_com = 0.0
                for stage_k in range(1, seq_len + 1):
                    state = bank.get(seq_id, f"S_{stage_k}", layer, variable)
                    ref = bank.singleton_refs[seq_id][stage_k][layer][variable]
                    state_disp = _cosine_distance(state, prev)
                    ref_disp = _cosine_distance(ref, prev)
                    sims = []
                    for pos in range(1, stage_k + 1):
                        sims.append(max(0.0, _centered_cosine(state, bank.singleton_refs[seq_id][pos][layer][variable])))
                    weights = np.asarray(sims, dtype=float)
                    weights = weights / max(float(weights.sum()), 1e-12)
                    positions = np.arange(1, stage_k + 1, dtype=float)
                    anchor_com = float(np.sum(positions * weights))
                    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))) / max(math.log(max(stage_k, 2)), 1e-12))
                    rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "sequence_id": seq_id,
                            "seq_len": seq_len,
                            "stage_k": stage_k,
                            "layer": layer,
                            "state_variable": variable,
                            "state_displacement": state_disp,
                            "singleton_displacement": ref_disp,
                            "natural_decay_displacement": 0.0,
                            "stepwise_update_ratio": float(state_disp / max(ref_disp, 1e-12)),
                            "anchor_COM": anchor_com,
                            "anchor_shift": float(anchor_com - prev_com),
                            "similarity_entropy": entropy,
                        }
                    )
                    prev = state
                    prev_com = anchor_com
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "panel_b_progressive_update_metrics.csv")
    ctx.completed_modules["progressive_update"] = True


def compute_final_support_landscape(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    _save_csv(ctx, _example_landscape_summary(ctx, bank), ctx.metrics_dir / "panel_c_example_landscape_summary.csv")
    if not (ctx.cfg.run_population_morphology_supplement or ctx.cfg.run_supplement):
        ctx.completed_modules["peak_valley_landscape"] = True
        return
    contrast_rows: list[dict[str, Any]] = []
    nonflat_rows: list[dict[str, Any]] = []
    prevalence_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 505)
    for _, meta in _progress(bank.sequence_meta.iterrows(), total=len(bank.sequence_meta), desc="fig3 landscape sequences", enabled=ctx.cfg.show_progress):
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        landscape = bank.landscapes[seq_id]
        g_final = landscape["G_final"]
        peak = landscape["peak_mask"].astype(bool)
        valley = landscape["valley_mask"].astype(bool)
        random_mask = landscape["random_matched_mask"].astype(bool)
        peak_mean = float(g_final[peak].mean()) if np.any(peak) else 0.0
        valley_mean = float(g_final[valley].mean()) if np.any(valley) else 0.0
        random_mean = float(g_final[random_mask].mean()) if np.any(random_mask) else 0.0
        delta = peak_mean - valley_mean
        contrast_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": seq_id,
                "seq_len": seq_len,
                "layer": PRIMARY_LAYER,
                "state_variable": PRIMARY_STATE_VARIABLE,
                "peak_mean_support": peak_mean,
                "valley_mean_support": valley_mean,
                "random_mean_support": random_mean,
                "peak_valley_delta": delta,
                "peak_random_delta": float(peak_mean - random_mean),
                "random_valley_delta": float(random_mean - valley_mean),
                "peak_valley_ratio": float(peak_mean / max(valley_mean, 1e-12)),
            }
        )
        pos = np.clip(g_final - float(np.min(g_final)), 0.0, None)
        total = float(pos.sum())
        nonflat_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": seq_id,
                "seq_len": seq_len,
                "layer": PRIMARY_LAYER,
                "state_variable": PRIMARY_STATE_VARIABLE,
                "support_std": float(g_final.std()),
                "support_cv": float(g_final.std() / max(abs(g_final.mean()), 1e-12)),
                "support_gini": _gini(pos.reshape(-1)),
                "top_q_mass_fraction": float(pos[peak].sum() / max(total, 1e-12)) if np.any(peak) else 0.0,
                "positive_support_area": int((landscape["delta_gain_map"] > 0).sum()),
                "peak_area_fraction": float(peak.mean()),
            }
        )
        null_values = []
        flat = g_final.reshape(-1)
        peak_count = int(peak.sum())
        valley_count = int(valley.sum())
        for _ in _progress(range(int(ctx.cfg.n_null)), total=int(ctx.cfg.n_null), desc="fig3 landscape nulls", enabled=ctx.cfg.show_progress):
            perm = rng.permutation(flat.size)
            p_idx = perm[:peak_count]
            v_idx = perm[peak_count : peak_count + valley_count]
            if len(p_idx) and len(v_idx):
                null_values.append(float(flat[p_idx].mean() - flat[v_idx].mean()))
        null_arr = np.asarray(null_values, dtype=float)
        p95 = float(np.percentile(null_arr, 95)) if null_arr.size else 0.0
        prevalence_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": seq_id,
                "seq_len": seq_len,
                "layer": PRIMARY_LAYER,
                "state_variable": PRIMARY_STATE_VARIABLE,
                "observed_peak_valley_delta": delta,
                "null_peak_valley_delta_mean": float(null_arr.mean()) if null_arr.size else 0.0,
                "null_peak_valley_delta_p95": p95,
                "observed_minus_null": float(delta - (null_arr.mean() if null_arr.size else 0.0)),
                "is_structured": int(delta > p95),
                "n_null": int(ctx.cfg.n_null),
            }
        )
    contrast = pd.DataFrame(contrast_rows)
    nonflat = pd.DataFrame(nonflat_rows)
    prevalence = pd.DataFrame(prevalence_rows)
    _save_csv(ctx, contrast, ctx.metrics_dir / "supp_peak_valley_contrast.csv")
    _save_csv(ctx, nonflat, ctx.metrics_dir / "supp_landscape_nonflatness.csv")
    _save_csv(ctx, prevalence, ctx.metrics_dir / "supp_peak_valley_prevalence.csv")
    _save_csv(ctx, _network_peak_summary(ctx.cfg.network_seed, contrast, nonflat, prevalence), ctx.metrics_dir / "supp_network_peak_valley_summary.csv")
    ctx.completed_modules["peak_valley_landscape"] = True


def run_neutral_ping_from_final_state(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    run_neutral_ping_readout_distribution(ctx, bank)


def run_neutral_ping_readout_distribution(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    raw_rows: list[dict[str, Any]] = []
    main_meta = _main_sequence_meta(ctx, bank)
    for _, meta in _progress(main_meta.iterrows(), total=len(main_meta), desc="fig3 ping sequences", enabled=ctx.cfg.show_progress):
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        labels = [int(v) for v in str(meta["ordered_item_labels"]).split(";")]
        for ping_repeat in _progress(range(int(ctx.cfg.ping_repeats)), total=int(ctx.cfg.ping_repeats), desc="fig3 ping repeats", enabled=ctx.cfg.show_progress):
            ping_seed = int(ctx.cfg.network_seed) * 100000 + seq_id * 100 + ping_repeat
            for state_condition in ctx.cfg.ping_main_state_conditions:
                state_key = "S_final" if str(state_condition) == "S_final" else "S0"
                boundary = bank.boundaries[seq_id][state_key]
                pred, fire, ping_energy, ping_spike_count, restore_info = _run_ping_from_boundary(ctx, boundary)
                position = labels.index(pred) + 1 if pred in labels else -1
                silent = pred < 0
                memory_condition = "sequence_state" if state_key == "S_final" else "cue_only"
                raw_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": seq_id,
                        "seq_len": seq_len,
                        "ordered_item_labels": ";".join(str(v) for v in labels),
                        "ping_repeat": int(ping_repeat),
                        "ping_seed": int(ping_seed),
                        "state_condition": str(state_condition),
                        "memory_condition": memory_condition,
                        "predicted_label": int(pred),
                        "predicted_position": int(position),
                        "pred_is_seen_item": int(position > 0),
                        "pred_is_unseen": int((not silent) and position < 0),
                        "silent": int(silent),
                        "first_fire_time_ms": int(fire),
                        "ping_energy": float(ping_energy),
                        "ping_spike_count": float(ping_spike_count),
                        "restore_mode": "reset_all_state_restore_selected_stsp",
                        "stsp_only_restore": 1,
                        "fast_state_reset": 1,
                        "restore_ok": int(restore_info.get("restore_ok", 1)),
                    }
                )
    raw_columns = [
        "network_seed",
        "sequence_id",
        "seq_len",
        "ordered_item_labels",
        "ping_repeat",
        "ping_seed",
        "state_condition",
        "memory_condition",
        "predicted_label",
        "predicted_position",
        "pred_is_seen_item",
        "pred_is_unseen",
        "silent",
        "first_fire_time_ms",
        "ping_energy",
        "ping_spike_count",
        "restore_mode",
        "stsp_only_restore",
        "fast_state_reset",
        "restore_ok",
    ]
    raw = pd.DataFrame(raw_rows, columns=raw_columns)
    _save_csv(ctx, raw, ctx.raw_dir / "panel_d_neutral_ping_trial_readout.csv")
    _save_csv(ctx, raw.copy(), ctx.raw_dir / "panel_e_neutral_ping_trial_readout.csv")

    serial_bins = [f"pos_{idx}" for idx in range(1, int(ctx.cfg.main_sequence_length) + 1)] + ["other", "silent"]
    pos_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    recency_rows: list[dict[str, Any]] = []
    if not raw.empty:
        raw = raw.copy()
        raw["serial_bin"] = raw.apply(_serial_bin_for_ping_row, axis=1)
        raw["class_label"] = raw.apply(lambda r: "silent" if int(r.get("silent", 0)) else str(int(r.get("predicted_label", -1))), axis=1)
        for (network_seed, state_condition, seq_len), part in raw.groupby(["network_seed", "state_condition", "seq_len"], sort=True):
            for serial_bin in serial_bins:
                pos_rows.append(
                    {
                        "network_seed": int(network_seed),
                        "state_condition": str(state_condition),
                        "seq_len": int(seq_len),
                        "serial_bin": serial_bin,
                        "readout_mass": float((part["serial_bin"] == serial_bin).mean()),
                        "n_trials": int(len(part)),
                    }
                )
            for class_label in [str(v) for v in range(NUM_CLASSES)] + ["silent"]:
                class_rows.append(
                    {
                        "network_seed": int(network_seed),
                        "state_condition": str(state_condition),
                        "class_label": class_label,
                        "readout_mass": float((part["class_label"] == class_label).mean()),
                        "n_trials": int(len(part)),
                    }
                )
            seen = part[part["pred_is_seen_item"] == 1]
            positions = pd.to_numeric(part["predicted_position"], errors="coerce")
            latest = float((positions == int(seq_len)).mean())
            earlier = float(((positions > 0) & (positions < int(seq_len))).mean())
            summary_rows.append(
                {
                    "network_seed": int(network_seed),
                    "state_condition": str(state_condition),
                    "seq_len": int(seq_len),
                    "P_seen_item": float(part["pred_is_seen_item"].mean()),
                    "P_unseen": float(part["pred_is_unseen"].mean()),
                    "P_silent": float(part["silent"].mean()),
                    "mean_first_fire_time_ms": float(pd.to_numeric(part["first_fire_time_ms"], errors="coerce").replace(-1, np.nan).mean()),
                    "n_trials": int(len(part)),
                }
            )
            recency_rows.append(
                {
                    "network_seed": int(network_seed),
                    "state_condition": str(state_condition),
                    "seq_len": int(seq_len),
                    "ping_COM": float(positions[positions > 0].mean()) if not seen.empty else float("nan"),
                    "latest_item_mass": latest,
                    "earlier_item_residual_mass": earlier,
                    "earlier_item_above_null": float(earlier - ((int(seq_len) - 1) / NUM_CLASSES)),
                    "n_trials": int(len(part)),
                }
            )
    pos_df = pd.DataFrame(pos_rows)
    class_df = pd.DataFrame(class_rows)
    summary_df = pd.DataFrame(summary_rows)
    _save_csv(ctx, pos_df, ctx.metrics_dir / "panel_d_ping_position_distribution.csv")
    _save_csv(ctx, class_df, ctx.metrics_dir / "panel_d_ping_class_distribution.csv")
    _save_csv(ctx, summary_df, ctx.metrics_dir / "panel_d_ping_summary.csv")
    _save_csv(ctx, pos_df.copy(), ctx.metrics_dir / "panel_e_ping_position_distribution.csv")
    _save_csv(ctx, class_df.copy(), ctx.metrics_dir / "panel_e_ping_class_distribution.csv")
    _save_csv(ctx, summary_df.copy(), ctx.metrics_dir / "panel_e_ping_summary.csv")
    _save_csv(ctx, pd.DataFrame(recency_rows), ctx.metrics_dir / "supp_ping_recency_diagnostics.csv")
    ctx.completed_modules["neutral_ping"] = True


def _serial_bin_for_ping_row(row: pd.Series) -> str:
    if int(row.get("silent", 0)):
        return "silent"
    position = int(row.get("predicted_position", -1))
    seq_len = int(row.get("seq_len", 0))
    if 1 <= position <= seq_len:
        return f"pos_{position}"
    return "other"


def run_region_gated_ping_readout(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    if not bank.landscapes:
        raise RuntimeError("run_region_ping requires state-bank landscapes; run the state-bank path first.")
    raw = _region_gated_ping_trial_rows(
        ctx,
        bank,
        amp_values=(float(ctx.cfg.ping_amp),),
        include_s0=bool(ctx.cfg.run_region_ping_s0_control),
        repeats=int(ctx.cfg.region_ping_repeats),
        desc="fig3 region ping sequences",
    )
    _save_csv(ctx, raw, ctx.raw_dir / "panel_f_region_ping_trial_readout.csv")
    position = _region_ping_position_distribution(ctx.cfg.network_seed, raw)
    summary = _region_ping_summary(ctx.cfg.network_seed, raw)
    contrast = _region_ping_contrast(ctx.cfg.network_seed, raw)
    matching = _region_ping_current_matching(ctx.cfg.network_seed, raw)
    _save_csv(ctx, position, ctx.metrics_dir / "panel_f_region_ping_position_distribution.csv")
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_f_region_ping_summary.csv")
    _save_csv(ctx, contrast, ctx.metrics_dir / "panel_f_region_ping_contrast.csv")
    _save_csv(ctx, matching, ctx.metrics_dir / "panel_f_region_ping_current_matching.csv")
    if _region_ping_current_matching_status(matching) != "passed":
        ctx.warnings.append("Fig.3F region ping current matching failed; see panel_f_region_ping_current_matching.csv.")
    if bool(ctx.cfg.run_region_ping_amp_sweep):
        sweep_raw = _region_gated_ping_trial_rows(
            ctx,
            bank,
            amp_values=tuple(float(v) for v in ctx.cfg.region_ping_amp_sweep),
            include_s0=bool(ctx.cfg.run_region_ping_s0_control),
            repeats=int(ctx.cfg.region_ping_repeats),
            desc="fig3 region ping amp sweep",
        )
        _save_csv(ctx, sweep_raw, ctx.raw_dir / "supp_region_ping_amp_sweep_trial_readout.csv")
        _save_csv(ctx, _region_ping_amp_sweep_summary(ctx.cfg.network_seed, sweep_raw), ctx.metrics_dir / "supp_region_ping_amp_sweep_summary.csv")
        _save_csv(ctx, _region_ping_amp_sweep_latency(ctx.cfg.network_seed, sweep_raw), ctx.metrics_dir / "supp_region_ping_amp_sweep_latency.csv")
    ctx.completed_modules["region_ping"] = True


def _region_gated_ping_trial_rows(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
    *,
    amp_values: Sequence[float],
    include_s0: bool,
    repeats: int,
    desc: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 1301)
    main_meta = _main_sequence_meta(ctx, bank)
    states = [("S_final", "sequence_state")]
    if include_s0:
        states.append(("S0", "no_memory"))
    for _, meta in _progress(main_meta.iterrows(), total=len(main_meta), desc=desc, enabled=ctx.cfg.show_progress):
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        labels = [int(v) for v in str(meta["ordered_item_labels"]).split(";")]
        landscape = bank.landscapes.get(seq_id)
        if landscape is None:
            raise RuntimeError(f"Missing landscape for sequence_id={seq_id}; region ping cannot run.")
        support_metric = str(ctx.cfg.region_ping_support_metric)
        if support_metric not in {"delta_gain_map", "G_final"}:
            raise ValueError(f"Unsupported region_ping_support_metric={support_metric!r}; expected delta_gain_map or G_final.")
        support_map = np.asarray(landscape[support_metric], dtype=np.float32)
        if np.any(~np.isfinite(support_map)):
            ctx.warnings.append(f"Region ping excluded non-finite support values for sequence_id={seq_id}.")
        for ping_repeat in range(int(repeats)):
            ping_seed = int(ctx.cfg.network_seed) * 1000000 + seq_id * 1000 + ping_repeat
            masks = _make_region_ping_masks(
                support_map,
                float(ctx.cfg.region_ping_q),
                np.random.default_rng(ping_seed),
                ctx.cfg.region_ping_conditions,
            )
            for ping_amp in amp_values:
                for state_condition, memory_condition in states:
                    boundary = bank.boundaries[seq_id][state_condition]
                    for region_condition in ctx.cfg.region_ping_conditions:
                        mask = masks[str(region_condition)]
                        pred, fire_ms, total_current, active_units, restore_info = _run_masked_ping_from_boundary(
                            ctx,
                            boundary,
                            mask,
                            float(ping_amp),
                            int(ctx.cfg.ping_steps),
                        )
                        silent = int(pred < 0)
                        positions = [idx + 1 for idx, label in enumerate(labels) if int(label) == int(pred)]
                        ambiguous = int(len(positions) > 1)
                        predicted_position = int(positions[0]) if positions else -1
                        serial_bin = "silent" if silent else (f"pos_{predicted_position}" if predicted_position > 0 else "other")
                        rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "sequence_id": seq_id,
                                "seq_len": seq_len,
                                "ordered_item_labels": ";".join(str(v) for v in labels),
                                "region_condition": str(region_condition),
                                "support_metric": support_metric,
                                "region_q": float(ctx.cfg.region_ping_q),
                                "region_unit_count": int(np.asarray(mask, dtype=bool).sum()),
                                "ping_repeat": int(ping_repeat),
                                "ping_seed": int(ping_seed),
                                "ping_amp": float(ping_amp),
                                "ping_ms": int(ctx.cfg.ping_ms),
                                "active_unit_count": float(active_units),
                                "total_ping_current": float(total_current),
                                "state_condition": state_condition,
                                "memory_condition": memory_condition,
                                "predicted_label": int(pred),
                                "predicted_position": int(predicted_position),
                                "serial_bin": serial_bin,
                                "pred_is_seen_item": int((not silent) and predicted_position > 0),
                                "pred_is_latest_item": int((not silent) and predicted_position == seq_len),
                                "pred_is_recent_item": int((not silent) and predicted_position >= seq_len - 2 and predicted_position < seq_len),
                                "pred_is_earlier_item": int((not silent) and predicted_position > 0 and predicted_position < seq_len - 2),
                                "pred_is_unseen": int((not silent) and predicted_position < 0),
                                "silent": silent,
                                "label_is_ambiguous": ambiguous,
                                "first_fire_time_ms": float(fire_ms),
                                "restore_mode": "reset_all_state_restore_selected_stsp",
                                "stsp_only_restore": 1,
                                "restore_ok": int(restore_info.get("restore_ok", 0)),
                            }
                        )
    return pd.DataFrame(rows)


def _make_region_ping_masks(
    support_map: np.ndarray,
    region_q: float,
    rng: np.random.Generator,
    conditions: Sequence[str],
) -> dict[str, np.ndarray]:
    support = np.asarray(support_map, dtype=np.float64)
    if support.ndim < 2:
        raise ValueError(f"region ping support_map must be at least 2D, got shape={support.shape}")
    valid_flat = np.flatnonzero(np.isfinite(support).reshape(-1))
    if valid_flat.size == 0:
        raise ValueError("region ping support_map has no finite units.")
    q = float(np.clip(float(region_q), 0.0, 1.0))
    count = max(1, int(round(q * valid_flat.size)))
    support_flat = support.reshape(-1)
    order = valid_flat[np.argsort(support_flat[valid_flat], kind="mergesort")]
    valley_idx = order[:count]
    peak_idx = order[-count:]
    random_idx = rng.choice(valid_flat, size=count, replace=valid_flat.size < count)
    index_by_condition = {"peak": peak_idx, "valley": valley_idx, "random": random_idx}
    masks: dict[str, np.ndarray] = {}
    for condition in conditions:
        cond = str(condition)
        if cond not in index_by_condition:
            raise ValueError(f"Unsupported region ping condition={cond!r}; expected peak, valley, or random.")
        out = np.zeros(support.size, dtype=bool)
        out[index_by_condition[cond]] = True
        masks[cond] = out.reshape(support.shape)
    counts = {cond: int(mask.sum()) for cond, mask in masks.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Region ping masks are not count-matched: {counts}")
    return masks


def _run_masked_ping_from_boundary(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    region_mask: np.ndarray,
    ping_amp: float,
    ping_steps: int,
) -> tuple[int, int, float, float, dict[str, object]]:
    batch_size = 1
    restore_info = restore_condition_state_for_functional_readout(ctx, boundary, batch_size=batch_size)
    input_shape = _layer_input_shapes_for_batch(boundary, batch_size)["layer1"]
    zero = torch.zeros(input_shape, dtype=torch.float32, device=ctx.device)
    mask_arr = np.asarray(region_mask, dtype=np.float32)
    if tuple(mask_arr.shape) == tuple(input_shape[1:]):
        mask_tensor = torch.as_tensor(mask_arr, dtype=torch.float32, device=ctx.device).unsqueeze(0)
    elif len(input_shape) == 4 and tuple(mask_arr.shape) == tuple(input_shape[2:]):
        mask_tensor = torch.as_tensor(mask_arr, dtype=torch.float32, device=ctx.device).unsqueeze(0).unsqueeze(0)
        mask_tensor = mask_tensor.expand(batch_size, input_shape[1], input_shape[2], input_shape[3]).contiguous()
    else:
        raise ValueError(f"region_mask shape {mask_arr.shape} is incompatible with layer1 input shape {input_shape}")
    ping_drive = torch.as_tensor(float(ping_amp), dtype=torch.float32, device=ctx.device) * mask_tensor
    active_unit_count = float((ping_drive > 0).detach().to(torch.float32).sum().item())
    total_ping_current = float(ping_amp) * active_unit_count * int(ping_steps)
    with torch.no_grad():
        for t_idx in range(int(ping_steps)):
            _step_network_once(ctx.net, zero, int(t_idx), ping_drive=ping_drive)
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size)
    fire_step = int(fire[0].item())
    fire_ms = int(fire_step * ctx.cfg.dt / ms) if fire_step >= 0 else -1
    return int(pred[0].item()), fire_ms, total_ping_current, active_unit_count, restore_info


def run_sequence_weak_probe_real_rollout_from_state_bank(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
) -> None:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 909)
    target_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    mask_id = 0
    main_meta = _main_sequence_meta(ctx, bank)
    for _, meta in _progress(main_meta.iterrows(), total=len(main_meta), desc="fig3 weak probe sequences", enabled=ctx.cfg.show_progress):
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        item_ids = [int(v) for v in str(meta["ordered_item_ids"]).split(";")]
        labels = [int(v) for v in str(meta["ordered_item_labels"]).split(";")]
        target_sources = _weak_probe_target_sources(ctx.cfg.weak_probe_target_source)
        for target_source in target_sources:
            for repeat_id in range(int(ctx.cfg.weak_probe_repeats)):
                target_seed = int(rng.integers(0, 2**31 - 1))
                local_rng = np.random.default_rng(target_seed)
                target_position, target_image_id, target_label = _sample_weak_cue_target(
                    ctx,
                    target_source,
                    seq_len,
                    item_ids,
                    labels,
                    local_rng,
                )
                target_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": seq_id,
                        "seq_len": seq_len,
                        "target_source": target_source,
                        "target_position": int(target_position),
                        "target_image_id": int(target_image_id),
                        "target_label": int(target_label),
                        "repeat_id": int(repeat_id),
                        "target_seed": int(target_seed),
                        "ordered_item_ids": ";".join(str(v) for v in item_ids),
                        "ordered_item_labels": ";".join(str(v) for v in labels),
                    }
                )
                target_image = ctx.dataset[int(target_image_id)][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
                full_target_spikes = encode_images(ctx.encoder, target_image, ctx.cfg.weak_probe_steps).to(ctx.device)
                memory_specs = _weak_probe_memory_specs_for_target(ctx, bank, seq_id, target_position)
                boundary = concat_named_boundaries([spec[2] for spec in memory_specs], device=ctx.device)
                memory_states = [spec[0] for spec in memory_specs]
                memory_conditions = [spec[1] for spec in memory_specs]
                for keep_prob in ctx.cfg.weak_probe_keep_probs:
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    weak_spikes, mask_info = _make_weak_probe_spikes_encoded_dropout(
                        full_target_spikes,
                        float(keep_prob),
                        seed=mask_seed,
                        same_mask_count=len(memory_specs),
                        use_same_mask_across_states=True,
                        device=ctx.device,
                    )
                    mask_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "mask_id": int(mask_id),
                            "sequence_id": seq_id,
                            "seq_len": seq_len,
                            "target_source": target_source,
                            "target_position": int(target_position),
                            "target_image_id": int(target_image_id),
                            "target_label": int(target_label),
                            "keep_prob": float(keep_prob),
                            "repeat_id": int(repeat_id),
                            "mask_seed": int(mask_seed),
                            "mask_space": "encoded_spikes",
                            "same_mask_used_across_states": bool(mask_info["same_mask_used_across_states"]),
                            "same_mask_used_across_memory_conditions": bool(mask_info["same_mask_used_across_memory_conditions"]),
                            "weak_probe_scale": float(ctx.cfg.weak_probe_scale),
                            "weak_probe_noise": float(ctx.cfg.weak_probe_noise),
                            "realized_keep_fraction": float(mask_info["realized_keep_fraction"]),
                            "full_spike_count": float(mask_info["full_spike_count"]),
                            "weak_spike_count": float(mask_info["weak_spike_count"]),
                            "weak_spike_fraction": float(mask_info["weak_spike_fraction"]),
                        }
                    )
                    pred, fire = run_probe_readout_from_boundary(
                        ctx,
                        boundary,
                        weak_spikes,
                        probe_scale=float(ctx.cfg.weak_probe_scale),
                        probe_noise=float(ctx.cfg.weak_probe_noise),
                        seed=mask_seed + 17,
                    )
                    for idx, state_condition in enumerate(memory_states):
                        prediction = int(pred[idx])
                        silent = prediction < 0
                        pred_is_seen = prediction in labels
                        raw_rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "sequence_id": seq_id,
                                "seq_len": seq_len,
                                "ordered_item_ids": ";".join(str(v) for v in item_ids),
                                "ordered_item_labels": ";".join(str(v) for v in labels),
                                "target_source": target_source,
                                "target_position": int(target_position),
                                "target_position_bin": _target_position_bin(int(target_position), seq_len),
                                "relative_position": float(target_position / seq_len) if int(target_position) > 0 else float("nan"),
                                "retention_slots_after_target": int(seq_len - target_position) if int(target_position) > 0 else -1,
                                "target_image_id": int(target_image_id),
                                "target_label": int(target_label),
                                "keep_prob": float(keep_prob),
                                "repeat_id": int(repeat_id),
                                "mask_id": int(mask_id),
                                "mask_seed": int(mask_seed),
                                "state_condition": state_condition,
                                "memory_condition": memory_conditions[idx],
                                "prediction": prediction,
                                "pred_is_target": int(prediction == target_label),
                                "pred_is_seen_item": int(pred_is_seen),
                                "pred_is_unseen": int((not silent) and not pred_is_seen),
                                "pred_is_latest_item": int(prediction == labels[-1]),
                                "pred_is_other_seen_item": int(pred_is_seen and prediction != target_label),
                                "silent": int(silent),
                                "first_fire_time_ms": float(fire[idx] * ctx.cfg.dt / ms) if int(fire[idx]) >= 0 else -1.0,
                                "mask_space": "encoded_spikes",
                                "weak_probe_scale": float(ctx.cfg.weak_probe_scale),
                                "weak_probe_noise": float(ctx.cfg.weak_probe_noise),
                                "weak_probe_metric_mode": str(ctx.cfg.weak_probe_metric_mode),
                                "realized_keep_fraction": float(mask_info["realized_keep_fraction"]),
                                "full_spike_count": float(mask_info["full_spike_count"]),
                                "weak_spike_count": float(mask_info["weak_spike_count"]),
                                "weak_spike_fraction": float(mask_info["weak_spike_fraction"]),
                                "same_mask_used_across_states": bool(mask_info["same_mask_used_across_states"]),
                                "same_mask_used_across_memory_conditions": bool(mask_info["same_mask_used_across_memory_conditions"]),
                                "restore_mode": "reset_all_state_restore_selected_stsp",
                                "stsp_only_restore": 1,
                            }
                        )
                    mask_id += 1
    targets = pd.DataFrame(target_rows)
    masks = pd.DataFrame(mask_rows)
    raw = pd.DataFrame(raw_rows)
    _save_csv(ctx, targets, ctx.trial_specs_dir / "weak_probe_targets.csv")
    _save_csv(ctx, masks, ctx.trial_specs_dir / "weak_probe_masks.csv")
    _save_csv(ctx, raw, ctx.raw_dir / "panel_e_weak_probe_trial_readout.csv")
    _save_csv(ctx, raw, ctx.raw_dir / "panel_f_weak_probe_trial_readout.csv")
    metrics = compute_fig3e_weak_probe_metrics(ctx.cfg.network_seed, raw)
    auc = compute_fig3e_weak_probe_auc_metrics(ctx.cfg.network_seed, metrics)
    gain = compute_fig3e_weak_probe_memory_gain(ctx.cfg.network_seed, metrics)
    pos_metrics = compute_fig3e_weak_probe_position_stratified_metrics(ctx.cfg.network_seed, raw)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_e_weak_probe_metrics.csv")
    _save_csv(ctx, auc, ctx.metrics_dir / "panel_e_weak_probe_auc_metrics.csv")
    _save_csv(ctx, gain, ctx.metrics_dir / "panel_e_weak_probe_memory_gain.csv")
    _save_csv(ctx, pos_metrics, ctx.metrics_dir / "panel_e_weak_probe_position_stratified_metrics.csv")
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_f_weak_probe_metrics.csv")
    _save_csv(ctx, auc, ctx.metrics_dir / "panel_f_weak_probe_auc_metrics.csv")
    _save_csv(ctx, gain, ctx.metrics_dir / "panel_f_weak_probe_memory_gain.csv")
    ctx.completed_modules["weak_probe"] = True


def _make_weak_probe_spikes_encoded_dropout(
    full_probe_spikes: torch.Tensor,
    keep_prob: float,
    *,
    seed: int,
    same_mask_count: int,
    use_same_mask_across_states: bool,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    full = full_probe_spikes.to(device=device, dtype=torch.float32)
    keep_prob = float(np.clip(float(keep_prob), 0.0, 1.0))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    if use_same_mask_across_states:
        mask = (torch.rand(full.shape, generator=gen, device=device) < keep_prob).to(torch.float32)
        weak = (full * mask).repeat(int(same_mask_count), 1, 1, 1, 1)
        realized_keep_fraction = float(mask.mean().detach().cpu().item())
    else:
        expanded = full.repeat(int(same_mask_count), 1, 1, 1, 1)
        mask = (torch.rand(expanded.shape, generator=gen, device=device) < keep_prob).to(torch.float32)
        weak = expanded * mask
        realized_keep_fraction = float(mask.mean().detach().cpu().item())
    full_spike_count = float(full.sum().detach().cpu().item())
    weak_spike_count = float(weak.sum().detach().cpu().item())
    denom = full_spike_count * float(same_mask_count)
    return weak.contiguous(), {
        "keep_prob": keep_prob,
        "mask_space": "encoded_spikes",
        "realized_keep_fraction": realized_keep_fraction,
        "full_spike_count": full_spike_count,
        "weak_spike_count": weak_spike_count,
        "weak_spike_fraction": float(weak_spike_count / denom) if denom > 0.0 else 0.0,
        "same_mask_used_across_states": bool(use_same_mask_across_states),
        "same_mask_used_across_memory_conditions": bool(use_same_mask_across_states),
        "mask_seed": int(seed),
    }


def slice_boundary_state(
    boundary_state: Mapping[str, Mapping[str, torch.Tensor]],
    row_indices: Sequence[int],
    device: torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    idx = torch.as_tensor(list(row_indices), dtype=torch.long)
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key, state in boundary_state.items():
        out[layer_key] = {}
        for key, value in state.items():
            selected = value.index_select(0, idx).detach().clone()
            out[layer_key][key] = selected.to(device) if device is not None else selected
    return out


def concat_sequence_condition_boundaries(
    boundary_states: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    conditions: Sequence[str],
    device: torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    return concat_named_boundaries([boundary_states[condition] for condition in conditions], device=device)


def concat_named_boundaries(
    boundaries: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
    device: torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    sliced = [slice_boundary_state(boundary, [0], device) for boundary in boundaries]
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key in sliced[0]:
        out[layer_key] = {}
        for key in sliced[0][layer_key]:
            out[layer_key][key] = torch.cat([part[layer_key][key] for part in sliced], dim=0)
    return out


def _weak_probe_memory_specs_for_target(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
    seq_id: int,
    target_position: int,
) -> list[tuple[str, str, Mapping[str, Mapping[str, torch.Tensor]]]]:
    specs: list[tuple[str, str, Mapping[str, Mapping[str, torch.Tensor]]]] = [
        ("S0", "cue_only", bank.boundaries[int(seq_id)]["S0"]),
    ]
    if bool(ctx.cfg.weak_probe_include_singleton):
        if int(target_position) in bank.singleton_boundaries[int(seq_id)]:
            singleton_boundary = bank.singleton_boundaries[int(seq_id)][int(target_position)]
        else:
            singleton_boundary = bank.boundaries[int(seq_id)]["S0"]
            ctx.warnings.append(
                f"Weak-probe singleton boundary unavailable for sequence_id={seq_id}, "
                f"target_position={target_position}; using S0 for that non-sequence target."
            )
        specs.append(("S_singleton_slot_matched", "single_item_memory", singleton_boundary))
    specs.append(("S_final", "sequence_state", bank.boundaries[int(seq_id)]["S_final"]))
    return specs


def run_probe_readout_from_boundary(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    probe_spikes: torch.Tensor,
    *,
    probe_scale: float = 1.0,
    probe_noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = int(probe_spikes.shape[0])
    restore_condition_state_for_functional_readout(ctx, boundary, batch_size)
    gen = torch.Generator(device=ctx.device)
    gen.manual_seed(int(seed))
    with torch.no_grad():
        for t_idx in range(probe_spikes.shape[1]):
            input_t = probe_spikes[:, t_idx].to(ctx.device, dtype=torch.float32) * float(probe_scale)
            if float(probe_noise) > 0.0:
                input_t = torch.clamp(
                    input_t + torch.randn(input_t.shape, generator=gen, device=ctx.device) * float(probe_noise),
                    min=0.0,
                )
            _step_network_once(ctx.net, input_t, int(t_idx))
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size=batch_size)
    return pred.numpy().astype(np.int64, copy=False), fire.numpy().astype(np.int64, copy=False)


def _fig3f_memory_states(
    cfg: Fig3Config,
    seq_len: int,
    available: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
) -> list[str]:
    if cfg.weak_probe_memory_scope == "final_only":
        return ["S0", "S_final"]
    if cfg.weak_probe_memory_scope != "all_prefixes":
        raise ValueError(f"Unsupported weak_probe_memory_scope={cfg.weak_probe_memory_scope}")
    states = ["S0"] + [f"S_{idx}" for idx in range(1, int(seq_len) + 1)]
    if "S_final" not in states:
        states.append("S_final")
    missing = [state for state in states if state not in available]
    if missing:
        raise NotImplementedError(f"weak_probe_memory_scope=all_prefixes requested but missing boundaries: {missing}")
    return states


def _memory_condition_label(state: str) -> str:
    if state == "S0":
        return "cue_only"
    if state == "S_final":
        return "sequence_state"
    if state.startswith("S_") and state[2:].isdigit():
        return f"prefix_{state[2:]}"
    return state


def _weak_probe_target_sources(value: str) -> tuple[str, ...]:
    text = str(value).strip()
    if text == "both":
        return ("sequence_member_random", "unseen_random")
    if text not in {"sequence_member_random", "unseen_random"}:
        return ("sequence_member_random",)
    return (text,)


def run_structural_weak_cue_classification_supplement(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 707)
    trial_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    mask_id = 0
    main_meta = _main_sequence_meta(ctx, bank)
    for _, meta in _progress(main_meta.iterrows(), total=len(main_meta), desc="fig3 weak cue sequences", enabled=ctx.cfg.show_progress):
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        item_ids = [int(v) for v in str(meta["ordered_item_ids"]).split(";")]
        labels = [int(v) for v in str(meta["ordered_item_labels"]).split(";")]
        sources = _weak_cue_target_sources(ctx.cfg.weak_cue_target_source)
        for target_source in sources:
            for repeat_id in range(int(ctx.cfg.weak_cue_repeats)):
                target_seed = int(rng.integers(0, 2**31 - 1))
                local_rng = np.random.default_rng(target_seed)
                target_position, target_image_id, target_label = _sample_weak_cue_target(ctx, target_source, seq_len, item_ids, labels, local_rng)
                trial_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": seq_id,
                        "seq_len": seq_len,
                        "target_source": target_source,
                        "target_position": int(target_position),
                        "target_image_id": int(target_image_id),
                        "target_label": int(target_label),
                        "repeat_id": int(repeat_id),
                        "target_seed": int(target_seed),
                    }
                )
                support_map = _support_map_for_structural_cue(ctx, bank.landscapes[seq_id])
                target_image = ctx.dataset[int(target_image_id)][0].detach().cpu().to(torch.float32)
                for keep_fraction in ctx.cfg.weak_cue_keep_fractions:
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    masks, stats = build_ranked_foreground_masks(
                        support_map,
                        target_image,
                        float(keep_fraction),
                        np.random.default_rng(mask_seed),
                        float(ctx.cfg.foreground_threshold),
                    )
                    for cue_condition in _progress(CUE_CONDITIONS, total=len(CUE_CONDITIONS), desc="fig3 weak cue conditions", enabled=ctx.cfg.show_progress):
                        mask = masks[cue_condition]
                        masked_image = _masked_image(ctx.dataset, int(target_image_id), mask).to(ctx.device)
                        cue_spikes = _encode_image_tensor_cached(
                            ctx,
                            masked_image,
                            ctx.cfg.weak_probe_steps,
                            cache=encode_cache,
                            cache_key=("weak_cue", seq_id, int(target_position), int(target_image_id), float(keep_fraction), int(repeat_id), cue_condition, int(mask_seed)),
                        )
                        encoded_spike_count = float(cue_spikes.detach().to(torch.float32).sum().item())
                        cue_energy = float(masked_image.detach().cpu().sum().item())
                        selected = support_map[mask.astype(bool)]
                        foreground = stats["foreground_mask"].astype(bool)
                        support_fg = support_map[foreground]
                        support_quantile_mean = _selected_quantile_mean(support_fg, selected)
                        mask_row = {
                            "network_seed": int(ctx.cfg.network_seed),
                            "sequence_id": seq_id,
                            "target_source": target_source,
                            "target_position": int(target_position),
                            "target_image_id": int(target_image_id),
                            "target_label": int(target_label),
                            "keep_fraction": float(keep_fraction),
                            "cue_condition": cue_condition,
                            "repeat_id": int(repeat_id),
                            "mask_id": int(mask_id),
                            "mask_seed": int(mask_seed),
                            "target_foreground_count": int(foreground.sum()),
                            "cue_pixel_count": int(mask.sum()),
                            "cue_fraction_actual": float(mask.sum() / max(1, int(foreground.sum()))),
                            "cue_energy": cue_energy,
                            "encoded_spike_count": float(encoded_spike_count),
                            "support_mean_selected": float(np.mean(selected)) if selected.size else 0.0,
                            "support_min_selected": float(np.min(selected)) if selected.size else 0.0,
                            "support_max_selected": float(np.max(selected)) if selected.size else 0.0,
                            "support_mean_foreground": float(np.mean(support_fg)) if support_fg.size else 0.0,
                            "same_mask_used_across_memory_conditions": True,
                        }
                        mask_rows.append(mask_row)
                        for memory_condition in _progress(MEMORY_CONDITIONS, total=len(MEMORY_CONDITIONS), desc="fig3 memory states", enabled=ctx.cfg.show_progress):
                            boundary = bank.boundaries[seq_id]["S_final"] if memory_condition == "sequence_state" else bank.boundaries[seq_id]["S0"]
                            pred, fire = _run_weak_cue_spikes_from_boundary(ctx, boundary, cue_spikes)
                            silent = pred < 0
                            raw_rows.append(
                                {
                                    "network_seed": int(ctx.cfg.network_seed),
                                    "sequence_id": seq_id,
                                    "seq_len": seq_len,
                                    "target_source": target_source,
                                    "target_position": int(target_position),
                                    "target_image_id": int(target_image_id),
                                    "target_label": int(target_label),
                                    "keep_fraction": float(keep_fraction),
                                    "cue_condition": cue_condition,
                                    "repeat_id": int(repeat_id),
                                    "mask_id": int(mask_id),
                                    "memory_condition": memory_condition,
                                    "prediction": int(pred),
                                    "correct": int(pred == target_label),
                                    "pred_is_target": int(pred == target_label),
                                    "pred_is_seen_item": int(pred in labels),
                                    "pred_is_unseen": int((not silent) and pred not in labels),
                                    "silent": int(silent),
                                    "first_fire_time_ms": int(fire),
                                    "cue_pixel_count": int(mask_row["cue_pixel_count"]),
                                    "target_foreground_count": int(mask_row["target_foreground_count"]),
                                    "cue_fraction_actual": float(mask_row["cue_fraction_actual"]),
                                    "cue_energy": cue_energy,
                                    "encoded_spike_count": float(encoded_spike_count),
                                    "support_mean_selected": float(mask_row["support_mean_selected"]),
                                    "support_mean_foreground": float(mask_row["support_mean_foreground"]),
                                    "support_quantile_mean": support_quantile_mean,
                                }
                            )
                        mask_id += 1
    trials = pd.DataFrame(trial_rows, columns=_structural_trial_columns())
    masks_df = pd.DataFrame(mask_rows, columns=_structural_mask_columns())
    raw = pd.DataFrame(raw_rows, columns=_structural_raw_columns())
    _save_csv(ctx, trials, ctx.trial_specs_dir / "supp_structural_weak_cue_trials.csv")
    _save_csv(ctx, masks_df, ctx.trial_specs_dir / "supp_structural_weak_cue_masks.csv")
    _save_csv(ctx, raw, ctx.raw_dir / "supp_structural_weak_cue_trial_readout.csv")
    accuracy = _structural_accuracy(ctx.cfg.network_seed, raw)
    memory_gain = _structural_memory_gain(ctx.cfg.network_seed, accuracy)
    _save_csv(ctx, accuracy, ctx.metrics_dir / "supp_structural_weak_cue_accuracy.csv")
    _save_csv(ctx, memory_gain, ctx.metrics_dir / "supp_structural_weak_cue_memory_gain.csv")
    _save_csv(ctx, _structural_target_source_control(ctx.cfg.network_seed, raw), ctx.metrics_dir / "supp_structural_weak_cue_target_source_control.csv")
    _save_csv(ctx, _structural_matching_diagnostics(ctx.cfg.network_seed, masks_df), ctx.metrics_dir / "supp_structural_weak_cue_matching_diagnostics.csv")
    ctx.completed_modules["structural_weak_cue_supplement"] = True


def ensure_structural_weak_cue_outputs(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    required = [
        ctx.raw_dir / "supp_structural_weak_cue_trial_readout.csv",
        ctx.metrics_dir / "supp_structural_weak_cue_accuracy.csv",
        ctx.metrics_dir / "supp_structural_weak_cue_memory_gain.csv",
        ctx.metrics_dir / "supp_structural_weak_cue_matching_diagnostics.csv",
    ]
    if all(path.exists() for path in required):
        ctx.completed_modules["structural_weak_cue_supplement"] = True
        return
    run_structural_weak_cue_classification_supplement(ctx, bank)


def run_peak_cue_main_from_state_bank(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    ensure_structural_weak_cue_outputs(ctx, bank)
    raw_path = ctx.raw_dir / "supp_structural_weak_cue_trial_readout.csv"
    match_path = ctx.metrics_dir / "supp_structural_weak_cue_matching_diagnostics.csv"
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    panel_raw = _filter_peak_cue_main_raw(ctx, raw)
    wanted_raw_columns = [
        "network_seed",
        "sequence_id",
        "seq_len",
        "target_source",
        "target_position",
        "target_image_id",
        "target_label",
        "keep_fraction",
        "cue_condition",
        "repeat_id",
        "mask_id",
        "memory_condition",
        "prediction",
        "correct",
        "pred_is_target",
        "pred_is_seen_item",
        "pred_is_unseen",
        "silent",
        "first_fire_time_ms",
        "cue_pixel_count",
        "cue_fraction_actual",
        "cue_energy",
        "encoded_spike_count",
        "support_mean_selected",
        "support_mean_foreground",
        "support_quantile_mean",
    ]
    missing_optional = [column for column in wanted_raw_columns if column not in panel_raw.columns]
    if missing_optional:
        ctx.warnings.append("Panel F peak-cue optional fields missing from structural raw: " + ",".join(missing_optional))
    raw_columns = [column for column in wanted_raw_columns if column in panel_raw.columns]
    if not raw_columns:
        raw_columns = list(panel_raw.columns)
    _save_csv(ctx, panel_raw.loc[:, raw_columns].copy(), ctx.raw_dir / "panel_f_peak_cue_trial_readout.csv")
    accuracy = _peak_cue_accuracy(ctx.cfg.network_seed, panel_raw)
    gain = _peak_cue_memory_gain(ctx.cfg.network_seed, accuracy)
    matching = _peak_cue_matching_diagnostics(ctx, panel_raw, match_path)
    _save_csv(ctx, accuracy, ctx.metrics_dir / "panel_f_peak_cue_accuracy.csv")
    _save_csv(ctx, gain, ctx.metrics_dir / "panel_f_peak_cue_memory_gain.csv")
    _save_csv(ctx, matching, ctx.metrics_dir / "panel_f_peak_cue_matching_diagnostics.csv")
    compute_peak_cue_serial_position_metrics(ctx)
    ctx.completed_modules["peak_cue_main"] = True


def _filter_peak_cue_main_raw(ctx: ExperimentContext, raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    out = raw.copy()
    if "keep_fraction" in out.columns:
        keep = pd.to_numeric(out["keep_fraction"], errors="coerce").to_numpy(dtype=float)
        out = out[np.isclose(keep, float(ctx.cfg.peak_cue_main_keep_fraction))].copy()
    if "target_source" in out.columns:
        out = out[out["target_source"].astype(str).eq("sequence_member_random")].copy()
    if "cue_condition" in out.columns:
        out = out[out["cue_condition"].astype(str).isin(CUE_CONDITIONS)].copy()
    if "memory_condition" in out.columns:
        out = out[out["memory_condition"].astype(str).isin(MEMORY_CONDITIONS)].copy()
    return out.reset_index(drop=True)


def _peak_cue_accuracy(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "cue_condition",
        "memory_condition",
        "keep_fraction",
        "P_target",
        "P_seen_item",
        "P_unseen",
        "P_silent",
        "n_trials",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    grouped = raw.groupby(["network_seed", "cue_condition", "memory_condition", "keep_fraction"], sort=True)
    for (seed, cue_condition, memory_condition, keep_fraction), part in grouped:
        denom = max(1, len(part))
        rows.append(
            {
                "network_seed": int(seed) if pd.notna(seed) else int(network_seed),
                "cue_condition": str(cue_condition),
                "memory_condition": str(memory_condition),
                "keep_fraction": float(keep_fraction),
                "P_target": float(part["pred_is_target"].sum() / denom) if "pred_is_target" in part else 0.0,
                "P_seen_item": float(part["pred_is_seen_item"].sum() / denom) if "pred_is_seen_item" in part else 0.0,
                "P_unseen": float(part["pred_is_unseen"].sum() / denom) if "pred_is_unseen" in part else 0.0,
                "P_silent": float(part["silent"].sum() / denom) if "silent" in part else 0.0,
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _peak_cue_memory_gain(network_seed: int, accuracy: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "cue_condition",
        "keep_fraction",
        "P_target_sequence_state",
        "P_target_cue_only",
        "memory_gain",
        "P_seen_sequence_state",
        "P_seen_cue_only",
        "seen_item_gain",
        "P_silent_sequence_state",
        "P_silent_cue_only",
        "silent_reduction",
        "n_trials",
    ]
    if accuracy.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (seed, cue_condition, keep_fraction), part in accuracy.groupby(["network_seed", "cue_condition", "keep_fraction"], sort=True):
        seq = part[part["memory_condition"].astype(str).eq("sequence_state")]
        cue = part[part["memory_condition"].astype(str).eq("cue_only")]
        p_target_seq = _first_float(seq, "P_target")
        p_target_cue = _first_float(cue, "P_target")
        p_seen_seq = _first_float(seq, "P_seen_item")
        p_seen_cue = _first_float(cue, "P_seen_item")
        p_silent_seq = _first_float(seq, "P_silent")
        p_silent_cue = _first_float(cue, "P_silent")
        rows.append(
            {
                "network_seed": int(seed) if pd.notna(seed) else int(network_seed),
                "cue_condition": str(cue_condition),
                "keep_fraction": float(keep_fraction),
                "P_target_sequence_state": p_target_seq,
                "P_target_cue_only": p_target_cue,
                "memory_gain": float(p_target_seq - p_target_cue),
                "P_seen_sequence_state": p_seen_seq,
                "P_seen_cue_only": p_seen_cue,
                "seen_item_gain": float(p_seen_seq - p_seen_cue),
                "P_silent_sequence_state": p_silent_seq,
                "P_silent_cue_only": p_silent_cue,
                "silent_reduction": float(p_silent_cue - p_silent_seq),
                "n_trials": int(min(_first_float(seq, "n_trials"), _first_float(cue, "n_trials"))),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _peak_cue_matching_diagnostics(ctx: ExperimentContext, panel_raw: pd.DataFrame, match_path: Path) -> pd.DataFrame:
    columns = [
        "network_seed",
        "cue_condition",
        "keep_fraction",
        "cue_pixel_count",
        "cue_fraction_actual",
        "cue_energy",
        "encoded_spike_count",
        "support_mean_selected",
        "support_quantile_mean",
        "n_masks",
    ]
    match = pd.read_csv(match_path) if match_path.exists() else pd.DataFrame()
    if not match.empty and "keep_fraction" in match.columns:
        keep = pd.to_numeric(match["keep_fraction"], errors="coerce").to_numpy(dtype=float)
        match = match[np.isclose(keep, float(ctx.cfg.peak_cue_main_keep_fraction))].copy()
    if not match.empty and "cue_condition" in match.columns:
        match = match[match["cue_condition"].astype(str).isin(CUE_CONDITIONS)].copy()
    quantile_by_condition: dict[str, float] = {}
    fraction_by_condition: dict[str, float] = {}
    if not panel_raw.empty:
        for cue_condition, part in panel_raw.groupby("cue_condition", sort=True):
            quantile_by_condition[str(cue_condition)] = _mean_numeric(part, "support_quantile_mean")
            fraction_by_condition[str(cue_condition)] = _mean_numeric(part, "cue_fraction_actual")
    rows: list[dict[str, Any]] = []
    if not match.empty:
        for _, row in match.iterrows():
            cue_condition = str(row.get("cue_condition", ""))
            rows.append(
                {
                    "network_seed": int(row.get("network_seed", ctx.cfg.network_seed)),
                    "cue_condition": cue_condition,
                    "keep_fraction": float(row.get("keep_fraction", ctx.cfg.peak_cue_main_keep_fraction)),
                    "cue_pixel_count": _row_float(row, "cue_pixel_count", "cue_pixel_count_mean"),
                    "cue_fraction_actual": fraction_by_condition.get(cue_condition, _row_float(row, "cue_fraction_actual", "cue_fraction_actual_mean")),
                    "cue_energy": _row_float(row, "cue_energy", "cue_energy_mean"),
                    "encoded_spike_count": _row_float(row, "encoded_spike_count", "encoded_spike_count_mean"),
                    "support_mean_selected": _row_float(row, "support_mean_selected"),
                    "support_quantile_mean": quantile_by_condition.get(cue_condition, _row_float(row, "support_quantile_mean")),
                    "n_masks": int(row.get("n_masks", 0)),
                }
            )
    elif not panel_raw.empty:
        for cue_condition, part in panel_raw.groupby("cue_condition", sort=True):
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "cue_condition": str(cue_condition),
                    "keep_fraction": float(ctx.cfg.peak_cue_main_keep_fraction),
                    "cue_pixel_count": _mean_numeric(part, "cue_pixel_count"),
                    "cue_fraction_actual": _mean_numeric(part, "cue_fraction_actual"),
                    "cue_energy": _mean_numeric(part, "cue_energy"),
                    "encoded_spike_count": _mean_numeric(part, "encoded_spike_count"),
                    "support_mean_selected": _mean_numeric(part, "support_mean_selected"),
                    "support_quantile_mean": _mean_numeric(part, "support_quantile_mean"),
                    "n_masks": int(part[["sequence_id", "repeat_id", "mask_id"]].drop_duplicates().shape[0]) if {"sequence_id", "repeat_id", "mask_id"}.issubset(part.columns) else int(len(part)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def run_structural_weak_cue_classification(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    ctx.warnings.append("Legacy structural weak-cue flag mapped to Main Fig.3F peak/valley/random weak-cue analysis.")
    run_structural_weak_cue_classification_supplement(ctx, bank)


def run_peak_aligned_completion(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank, partial_trials: pd.DataFrame) -> None:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 606)
    cue_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    mask_id = 0
    for _, trial in partial_trials.iterrows():
        seq_id = int(trial["sequence_id"])
        seq_len = int(trial["seq_len"])
        target_position = int(trial["target_position"])
        target_label = int(trial["target_label"])
        labels = [int(v) for v in bank.sequence_meta.loc[bank.sequence_meta["sequence_id"] == seq_id, "ordered_item_labels"].iloc[0].split(";")]
        landscape = bank.landscapes[seq_id]
        masks = _cue_masks_for_target(ctx, landscape, int(trial["target_image_id"]), rng)
        for cue_condition, mask in masks.items():
            masked_image = _masked_image(ctx.dataset, int(trial["target_image_id"]), mask).to(ctx.device)
            spike_count = _encoded_spike_count(ctx, masked_image)
            cue_energy = float(masked_image.detach().cpu().sum().item())
            cue_pixel_count = int(mask.sum())
            cue_row = {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": seq_id,
                "seq_len": seq_len,
                "target_position": target_position,
                "target_label": target_label,
                "cue_condition": cue_condition,
                "mask_id": int(mask_id),
                "cue_pixel_count": cue_pixel_count,
                "cue_fraction": float(cue_pixel_count / max(1, int((_foreground_mask(ctx.dataset, int(trial["target_image_id"]))).sum()))),
                "cue_input_energy": cue_energy,
                "cue_spike_count": float(spike_count),
                "matched_to_peak_mask": int(cue_condition == "random_matched"),
                "matching_error_energy": 0.0,
                "matching_error_spike_count": 0.0,
            }
            cue_rows.append(cue_row)
            for memory_condition in MEMORY_CONDITIONS:
                boundary = bank.boundaries[seq_id]["S_final"] if memory_condition == "sequence_state" else bank.boundaries[seq_id]["S0"]
                pred, fire = _run_weak_cue_from_boundary(ctx, boundary, masked_image)
                silent = pred < 0
                raw_rows.append(
                    {
                        **cue_row,
                        "memory_condition": memory_condition,
                        "keep_fraction": float(ctx.cfg.partial_cue_keep_fraction),
                        "prediction": int(pred),
                        "pred_is_target": int(pred == target_label),
                        "pred_is_seen_item": int(pred in labels),
                        "pred_is_latest_item": int(pred == labels[-1]),
                        "pred_is_other": int((not silent) and pred != target_label),
                        "silent": int(silent),
                        "first_fire_time_ms": int(fire),
                    }
                )
            mask_id += 1
    cue_df = pd.DataFrame(cue_rows)
    raw = pd.DataFrame(raw_rows)
    _save_csv(ctx, cue_df, ctx.trial_specs_dir / "cue_masks.csv")
    _save_csv(ctx, raw, ctx.raw_dir / "panel_f_partial_cue_trial_readout.csv")
    metrics = _completion_metrics(ctx.cfg.network_seed, raw)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_f_peak_aligned_completion_metrics.csv")
    diag_cols = [
        "network_seed",
        "sequence_id",
        "target_position",
        "cue_condition",
        "mask_id",
        "cue_pixel_count",
        "cue_fraction",
        "cue_input_energy",
        "cue_spike_count",
        "matched_to_peak_mask",
        "matching_error_energy",
        "matching_error_spike_count",
    ]
    _save_csv(ctx, cue_df[diag_cols], ctx.metrics_dir / "panel_f_cue_matching_diagnostics.csv")
    ctx.completed_modules["peak_aligned_completion"] = True


def compute_supplementary_metrics(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    if (ctx.metrics_dir / "supp_partial_cue_fraction_sweep.csv").exists():
        ctx.warnings.append("Existing supp_partial_cue_fraction_sweep.csv was ignored; post-hoc scaled keep-fraction sweeps are not part of the new Fig.3 design.")
    two_rows = []
    recency_rows = []
    layer_rows = []
    for _, meta in bank.sequence_meta.iterrows():
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        final = bank.get(seq_id, "S_final", "layer1", "g")
        latest = bank.singleton_refs[seq_id][seq_len]["layer1"]["g"]
        first = bank.singleton_refs[seq_id][1]["layer1"]["g"]
        two_rows.append({"network_seed": int(ctx.cfg.network_seed), "sequence_id": seq_id, "seq_len": seq_len, "metric": "latest_minus_first_similarity", "value": float(_centered_cosine(final, latest) - _centered_cosine(final, first))})
        recency_rows.append({"network_seed": int(ctx.cfg.network_seed), "sequence_id": seq_id, "seq_len": seq_len, "metric": "earlier_residual_support", "value": float(max(0.0, _centered_cosine(final, first))), "notes": "Supplement only; not used for Fig.3D."})
        for layer in LAYER_KEYS:
            layer_rows.append({"network_seed": int(ctx.cfg.network_seed), "sequence_id": seq_id, "seq_len": seq_len, "layer": layer, "delay_ms": int(ctx.cfg.delay_ms), "metric": "progressive_update", "value": float(_cosine_distance(bank.get(seq_id, "S_final", layer, "g"), bank.get(seq_id, "S0", layer, "g")))})
    _save_csv(ctx, pd.DataFrame(two_rows), ctx.metrics_dir / "supp_two_item_imbalance_metrics.csv")
    _save_csv(ctx, pd.DataFrame(layer_rows), ctx.metrics_dir / "supp_layer_delay_multiitem_metrics.csv")
    _save_csv(ctx, pd.DataFrame(recency_rows), ctx.metrics_dir / "supp_recency_only_controls.csv")
    compute_anchor_dynamics_metrics(ctx, bank)
    compute_weak_probe_target_source_control(ctx)
    compute_peak_cue_serial_position_metrics(ctx)
    ctx.completed_modules["supplement"] = True


def compute_anchor_dynamics_metrics(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    path = ctx.metrics_dir / "panel_b_progressive_update_metrics.csv"
    if path.exists():
        source = pd.read_csv(path)
    else:
        rows: list[dict[str, Any]] = []
        for _, meta in bank.sequence_meta.iterrows():
            seq_id = int(meta["sequence_id"])
            seq_len = int(meta["seq_len"])
            for layer in LAYER_KEYS:
                for variable in ("g", "u", "x"):
                    prev_com = 0.0
                    for stage_k in range(1, seq_len + 1):
                        state = bank.get(seq_id, f"S_{stage_k}", layer, variable)
                        sims = [
                            max(0.0, _centered_cosine(state, bank.singleton_refs[seq_id][pos][layer][variable]))
                            for pos in range(1, stage_k + 1)
                        ]
                        weights = np.asarray(sims, dtype=float)
                        weights = weights / max(float(weights.sum()), 1e-12)
                        positions = np.arange(1, stage_k + 1, dtype=float)
                        anchor_com = float(np.sum(positions * weights))
                        entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))) / max(math.log(max(stage_k, 2)), 1e-12))
                        rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "sequence_id": seq_id,
                                "seq_len": seq_len,
                                "stage_k": stage_k,
                                "layer": layer,
                                "state_variable": variable,
                                "anchor_COM": anchor_com,
                                "anchor_shift": float(anchor_com - prev_com),
                                "similarity_entropy": entropy,
                            }
                        )
                        prev_com = anchor_com
        source = pd.DataFrame(rows)
    columns = [
        "network_seed",
        "sequence_id",
        "seq_len",
        "stage_k",
        "layer",
        "state_variable",
        "anchor_COM",
        "anchor_shift",
        "similarity_entropy",
        "latest_position",
        "distance_to_latest",
        "earlier_residual_proxy",
    ]
    if source.empty:
        _save_csv(ctx, pd.DataFrame(columns=columns), ctx.metrics_dir / "supp_anchor_dynamics_metrics.csv")
        return
    out = source.copy()
    out["latest_position"] = pd.to_numeric(out["stage_k"], errors="coerce")
    out["distance_to_latest"] = pd.to_numeric(out["stage_k"], errors="coerce") - pd.to_numeric(out["anchor_COM"], errors="coerce")
    out["earlier_residual_proxy"] = pd.to_numeric(out["similarity_entropy"], errors="coerce")
    _save_csv(ctx, out.loc[:, [column for column in columns if column in out.columns]], ctx.metrics_dir / "supp_anchor_dynamics_metrics.csv")


def compute_weak_probe_target_source_control(ctx: ExperimentContext) -> None:
    raw_path = ctx.raw_dir / "panel_e_weak_probe_trial_readout.csv"
    if not raw_path.exists():
        raw_path = ctx.raw_dir / "panel_f_weak_probe_trial_readout.csv"
    control_columns = ["network_seed", "target_source", "memory_condition", "keep_prob", "P_target", "P_seen_item", "P_unseen", "P_silent", "n_trials"]
    gain_columns = [
        "network_seed",
        "target_source",
        "keep_prob",
        "P_target_sequence_state",
        "P_target_cue_only",
        "target_recovery_gain",
        "P_seen_sequence_state",
        "P_seen_cue_only",
        "seen_item_gain",
        "n_trials",
    ]
    if not raw_path.exists():
        ctx.warnings.append("Weak-probe target-source control skipped because panel E/F weak-probe raw output is missing.")
        _save_csv(ctx, pd.DataFrame(columns=control_columns), ctx.metrics_dir / "supp_weak_probe_target_source_control.csv")
        _save_csv(ctx, pd.DataFrame(columns=gain_columns), ctx.metrics_dir / "supp_weak_probe_target_source_gain.csv")
        return
    raw = pd.read_csv(raw_path)
    control_rows: list[dict[str, Any]] = []
    if not raw.empty:
        for (seed, target_source, memory_condition, keep_prob), part in raw.groupby(["network_seed", "target_source", "memory_condition", "keep_prob"], sort=True):
            denom = max(1, len(part))
            control_rows.append(
                {
                    "network_seed": int(seed),
                    "target_source": str(target_source),
                    "memory_condition": str(memory_condition),
                    "keep_prob": float(keep_prob),
                    "P_target": float(part["pred_is_target"].sum() / denom),
                    "P_seen_item": float(part["pred_is_seen_item"].sum() / denom),
                    "P_unseen": float(part["pred_is_unseen"].sum() / denom),
                    "P_silent": float(part["silent"].sum() / denom),
                    "n_trials": int(len(part)),
                }
            )
    control = pd.DataFrame(control_rows, columns=control_columns)
    gain_rows: list[dict[str, Any]] = []
    if not control.empty:
        for (seed, target_source, keep_prob), part in control.groupby(["network_seed", "target_source", "keep_prob"], sort=True):
            seq = part[part["memory_condition"].astype(str).eq("sequence_state")]
            cue = part[part["memory_condition"].astype(str).eq("cue_only")]
            p_target_seq = _first_float(seq, "P_target")
            p_target_cue = _first_float(cue, "P_target")
            p_seen_seq = _first_float(seq, "P_seen_item")
            p_seen_cue = _first_float(cue, "P_seen_item")
            gain_rows.append(
                {
                    "network_seed": int(seed),
                    "target_source": str(target_source),
                    "keep_prob": float(keep_prob),
                    "P_target_sequence_state": p_target_seq,
                    "P_target_cue_only": p_target_cue,
                    "target_recovery_gain": float(p_target_seq - p_target_cue),
                    "P_seen_sequence_state": p_seen_seq,
                    "P_seen_cue_only": p_seen_cue,
                    "seen_item_gain": float(p_seen_seq - p_seen_cue),
                    "n_trials": int(min(_first_float(seq, "n_trials"), _first_float(cue, "n_trials"))),
                }
            )
    gain = pd.DataFrame(gain_rows, columns=gain_columns)
    _save_csv(ctx, control, ctx.metrics_dir / "supp_weak_probe_target_source_control.csv")
    _save_csv(ctx, gain, ctx.metrics_dir / "supp_weak_probe_target_source_gain.csv")
    ctx.completed_modules["weak_probe_target_source_control"] = True


def compute_peak_cue_serial_position_metrics(ctx: ExperimentContext) -> None:
    raw_path = ctx.raw_dir / "supp_structural_weak_cue_trial_readout.csv"
    metric_columns = [
        "network_seed",
        "seq_len",
        "target_position",
        "target_position_bin",
        "relative_position",
        "cue_condition",
        "memory_condition",
        "keep_fraction",
        "P_target",
        "P_seen_item",
        "P_unseen",
        "P_silent",
        "n_trials",
    ]
    gain_columns = [
        "network_seed",
        "seq_len",
        "target_position",
        "target_position_bin",
        "relative_position",
        "cue_condition",
        "keep_fraction",
        "P_target_sequence_state",
        "P_target_cue_only",
        "memory_gain",
        "n_trials",
    ]
    if not raw_path.exists():
        ctx.warnings.append("Peak-cue serial-position supplement skipped because structural weak-cue raw output is missing.")
        _save_csv(ctx, pd.DataFrame(columns=metric_columns), ctx.metrics_dir / "supp_peak_cue_serial_position_metrics.csv")
        _save_csv(ctx, pd.DataFrame(columns=gain_columns), ctx.metrics_dir / "supp_peak_cue_serial_position_gain.csv")
        return
    raw = pd.read_csv(raw_path)
    if raw.empty:
        _save_csv(ctx, pd.DataFrame(columns=metric_columns), ctx.metrics_dir / "supp_peak_cue_serial_position_metrics.csv")
        _save_csv(ctx, pd.DataFrame(columns=gain_columns), ctx.metrics_dir / "supp_peak_cue_serial_position_gain.csv")
        return
    raw = raw.copy()
    if "target_source" in raw.columns:
        raw = raw[raw["target_source"].astype(str).eq("sequence_member_random")].copy()
    raw = raw[raw["cue_condition"].astype(str).isin(CUE_CONDITIONS)].copy()
    raw = raw[raw["memory_condition"].astype(str).isin(MEMORY_CONDITIONS)].copy()
    raw["target_position_bin"] = raw.apply(lambda row: _target_position_bin(row.get("target_position", -1), row.get("seq_len", 0)), axis=1)
    raw["relative_position"] = pd.to_numeric(raw["target_position"], errors="coerce") / pd.to_numeric(raw["seq_len"], errors="coerce").replace(0, np.nan)
    metric_rows: list[dict[str, Any]] = []
    for keys, part in raw.groupby(["network_seed", "seq_len", "target_position", "target_position_bin", "relative_position", "cue_condition", "memory_condition", "keep_fraction"], sort=True):
        seed, seq_len, target_position, target_position_bin, relative_position, cue_condition, memory_condition, keep_fraction = keys
        denom = max(1, len(part))
        metric_rows.append(
            {
                "network_seed": int(seed),
                "seq_len": int(seq_len),
                "target_position": int(target_position),
                "target_position_bin": str(target_position_bin),
                "relative_position": float(relative_position),
                "cue_condition": str(cue_condition),
                "memory_condition": str(memory_condition),
                "keep_fraction": float(keep_fraction),
                "P_target": float(part["pred_is_target"].sum() / denom),
                "P_seen_item": float(part["pred_is_seen_item"].sum() / denom),
                "P_unseen": float(part["pred_is_unseen"].sum() / denom),
                "P_silent": float(part["silent"].sum() / denom),
                "n_trials": int(len(part)),
            }
        )
    metrics = pd.DataFrame(metric_rows, columns=metric_columns)
    gain_rows: list[dict[str, Any]] = []
    if not metrics.empty:
        for keys, part in metrics.groupby(["network_seed", "seq_len", "target_position", "target_position_bin", "relative_position", "cue_condition", "keep_fraction"], sort=True):
            seed, seq_len, target_position, target_position_bin, relative_position, cue_condition, keep_fraction = keys
            seq = part[part["memory_condition"].astype(str).eq("sequence_state")]
            cue = part[part["memory_condition"].astype(str).eq("cue_only")]
            p_target_seq = _first_float(seq, "P_target")
            p_target_cue = _first_float(cue, "P_target")
            gain_rows.append(
                {
                    "network_seed": int(seed),
                    "seq_len": int(seq_len),
                    "target_position": int(target_position),
                    "target_position_bin": str(target_position_bin),
                    "relative_position": float(relative_position),
                    "cue_condition": str(cue_condition),
                    "keep_fraction": float(keep_fraction),
                    "P_target_sequence_state": p_target_seq,
                    "P_target_cue_only": p_target_cue,
                    "memory_gain": float(p_target_seq - p_target_cue),
                    "n_trials": int(min(_first_float(seq, "n_trials"), _first_float(cue, "n_trials"))),
                }
            )
    gain = pd.DataFrame(gain_rows, columns=gain_columns)
    _save_csv(ctx, metrics, ctx.metrics_dir / "supp_peak_cue_serial_position_metrics.csv")
    _save_csv(ctx, gain, ctx.metrics_dir / "supp_peak_cue_serial_position_gain.csv")
    ctx.completed_modules["peak_cue_serial_position"] = True


def _target_position_bin(target_position: Any, seq_len: Any) -> str:
    try:
        pos = int(target_position)
        length = int(seq_len)
    except (TypeError, ValueError):
        return "unknown"
    if pos < 1 or length < 1:
        return "unknown"
    if pos == length:
        return "latest"
    if pos >= length - 2:
        return "recent"
    if pos <= 2:
        return "early"
    return "middle"


def _capture_sequence(ctx: ExperimentContext, spikes: torch.Tensor) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, Mapping[str, Mapping[str, torch.Tensor]]]]:
    cfg = ctx.cfg
    seq_len, _, channels, height, width = spikes.shape
    arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    boundaries: dict[str, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    zero_input = torch.zeros((1, channels, height, width), device=ctx.device)
    prepare_network_state(ctx.net, 1, channels, height, width)
    for _ in range(seq_len):
        for _ in range(cfg.sample_steps + cfg.delay_steps):
            _step_network_once(ctx.net, zero_input, 0)
    arrays["S0"] = _snapshot_arrays(ctx.net, 1)
    boundaries["S0"] = snapshot_boundary_state(ctx.net)

    prepare_network_state(ctx.net, 1, channels, height, width)
    current_time = 0
    for idx in range(seq_len):
        for t in range(cfg.sample_steps):
            current_time = _step_network_once(ctx.net, spikes[idx : idx + 1, t, ...], current_time)
        for _ in range(cfg.delay_steps):
            current_time = _step_network_once(ctx.net, zero_input, current_time)
        arrays[f"S_{idx + 1}"] = _snapshot_arrays(ctx.net, 1)
        boundaries[f"S_{idx + 1}"] = snapshot_boundary_state(ctx.net)
    arrays["S_final"] = arrays[f"S_{seq_len}"]
    boundaries["S_final"] = boundaries[f"S_{seq_len}"]
    return arrays, boundaries


def _capture_singleton_refs(ctx: ExperimentContext, spikes: torch.Tensor) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    refs, _ = _capture_singleton_refs_and_boundaries(ctx, spikes)
    return refs


def _capture_singleton_refs_and_boundaries(
    ctx: ExperimentContext,
    spikes: torch.Tensor,
) -> tuple[
    dict[int, dict[str, dict[str, np.ndarray]]],
    dict[int, Mapping[str, Mapping[str, torch.Tensor]]],
]:
    cfg = ctx.cfg
    seq_len, _, channels, height, width = spikes.shape
    zero_input = torch.zeros((1, channels, height, width), device=ctx.device)
    refs: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    boundaries: dict[int, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    for target_idx in range(seq_len):
        prepare_network_state(ctx.net, 1, channels, height, width)
        current_time = 0
        for idx in range(seq_len):
            for t in range(cfg.sample_steps):
                input_t = spikes[idx : idx + 1, t, ...] if idx == target_idx else zero_input
                current_time = _step_network_once(ctx.net, input_t, current_time)
            for _ in range(cfg.delay_steps):
                current_time = _step_network_once(ctx.net, zero_input, current_time)
        refs[target_idx + 1] = _snapshot_arrays(ctx.net, 1)
        boundaries[target_idx + 1] = snapshot_boundary_state(ctx.net)
    return refs, boundaries


def _snapshot_arrays(net, batch_size: int) -> dict[str, dict[str, np.ndarray]]:
    snap = snapshot_ux_state(net, batch_size)
    out: dict[str, dict[str, np.ndarray]] = {}
    for layer in LAYER_KEYS:
        u = snap[layer]["u"][0].astype(np.float32, copy=False)
        x = snap[layer]["x"][0].astype(np.float32, copy=False)
        out[layer] = {"u": u, "x": x, "g": (u * x).astype(np.float32, copy=False)}
    return out


def _landscape_for_sequence(ctx: ExperimentContext, state_arrays: Mapping[str, Any], group: pd.DataFrame) -> dict[str, np.ndarray]:
    baseline = _layer1_map(state_arrays["S0"]["layer1"]["g"])
    final = _layer1_map(state_arrays["S_final"]["layer1"]["g"])
    delta = final - baseline
    positive = delta > 1e-12
    peak_mask = _top_mask(delta, ctx.cfg.peak_q, positive=positive)
    valley_mask = _bottom_mask(delta, ctx.cfg.valley_q)
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + int(group["sequence_id"].iloc[0]))
    random_mask = _random_mask_like(peak_mask, np.ones_like(peak_mask, dtype=bool), rng)
    foreground_masks = np.stack([_foreground_mask(ctx.dataset, int(image_id)) for image_id in group.sort_values("stage_k")["item_image_id"].tolist()], axis=0)
    return {
        "G_baseline": baseline.astype(np.float32),
        "G_final": final.astype(np.float32),
        "delta_gain_map": delta.astype(np.float32),
        "peak_mask": peak_mask.astype(np.uint8),
        "valley_mask": valley_mask.astype(np.uint8),
        "random_matched_mask": random_mask.astype(np.uint8),
        "item_foreground_masks": foreground_masks.astype(np.uint8),
        "sequence_labels": group.sort_values("stage_k")["item_label"].to_numpy(dtype=np.int64),
    }


def _save_example_landscape(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    target = int(bank.sequence_meta.iloc[0]["sequence_id"])
    landscape = bank.landscapes[target]
    np.savez_compressed(ctx.raw_dir / "panel_c_example_landscape.npz", **landscape)
    ctx.output_files["panel_c_example_landscape"] = _rel(ctx.raw_dir / "panel_c_example_landscape.npz", ctx.seed_dir)
    row = bank.sequence_meta.iloc[0]
    metadata = {
        "network_seed": int(ctx.cfg.network_seed),
        "sequence_id": int(target),
        "seq_len": int(row["seq_len"]),
        "ordered_item_ids": str(row["ordered_item_ids"]),
        "ordered_item_labels": str(row["ordered_item_labels"]),
        "structural_weak_cue_target_selection": "random_sequence_member",
        "peak_q": float(ctx.cfg.peak_q),
        "valley_q": float(ctx.cfg.valley_q),
        "epsilon": 1e-12,
        "layer": PRIMARY_LAYER,
        "state_variable": PRIMARY_STATE_VARIABLE,
    }
    _write_json(metadata, ctx.raw_dir / "panel_c_example_landscape_metadata.json")
    ctx.output_files["panel_c_example_landscape_metadata"] = _rel(ctx.raw_dir / "panel_c_example_landscape_metadata.json", ctx.seed_dir)


def _example_landscape_summary(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> pd.DataFrame:
    seq_id = int(bank.sequence_meta.iloc[0]["sequence_id"])
    row = bank.sequence_meta.iloc[0]
    land = bank.landscapes[seq_id]
    g = land["G_final"]
    peak = land["peak_mask"].astype(bool)
    valley = land["valley_mask"].astype(bool)
    random = land["random_matched_mask"].astype(bool)
    return pd.DataFrame(
        [
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": seq_id,
                "seq_len": int(row["seq_len"]),
                "layer": PRIMARY_LAYER,
                "state_variable": PRIMARY_STATE_VARIABLE,
                "peak_q": float(ctx.cfg.peak_q),
                "valley_q": float(ctx.cfg.valley_q),
                "peak_pixel_count": int(peak.sum()),
                "valley_pixel_count": int(valley.sum()),
                "random_pixel_count": int(random.sum()),
                "peak_mean_support": float(g[peak].mean()) if np.any(peak) else 0.0,
                "valley_mean_support": float(g[valley].mean()) if np.any(valley) else 0.0,
                "random_mean_support": float(g[random].mean()) if np.any(random) else 0.0,
            }
        ]
    )


def boundary_state_to_restore_ux_by_layer(
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_key, state in boundary.items():
        if "u" in state and "x" in state:
            out[layer_key] = (state["u"].to(device), state["x"].to(device))
    return out


def _layer_input_shapes_from_boundary(boundary: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, tuple[int, ...]]:
    return {layer_key: tuple(state["u"].shape) for layer_key, state in boundary.items() if "u" in state}


def _layer_input_shapes_for_batch(boundary: Mapping[str, Mapping[str, torch.Tensor]], batch_size: int) -> dict[str, tuple[int, ...]]:
    shapes = _layer_input_shapes_from_boundary(boundary)
    return {layer_key: (int(batch_size),) + tuple(shape[1:]) for layer_key, shape in shapes.items()}


def restore_condition_state_for_functional_readout(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    batch_size: int,
) -> dict[str, object]:
    layer_input_shapes = _layer_input_shapes_for_batch(boundary, int(batch_size))
    restore_ux = boundary_state_to_restore_ux_by_layer(boundary, ctx.device)
    info = reset_all_state_restore_selected_stsp_in_place(ctx.net, layer_input_shapes, restore_ux)
    with torch.no_grad():
        ctx.net.layer3.reset_decision_state()
        if hasattr(ctx.net.layer3, "lateral_inh"):
            ctx.net.layer3.lateral_inh.reset_state(ctx.net.layer3.output_shape)
    out = dict(info)
    out["restore_ok"] = int(out.get("restored_stsp_layer_count", 0) > 0)
    return out


def _run_ping_from_boundary(ctx: ExperimentContext, boundary: Mapping[str, Mapping[str, torch.Tensor]]) -> tuple[int, int, float, float, dict[str, object]]:
    batch_size = int(next(iter(next(iter(boundary.values())).values())).shape[0])
    restore_info = restore_condition_state_for_functional_readout(ctx, boundary, batch_size)
    input_shape = _layer_input_shapes_for_batch(boundary, batch_size)["layer1"]
    zero = torch.zeros(input_shape, dtype=torch.float32, device=ctx.device)
    ping = torch.full_like(zero, float(ctx.cfg.ping_amp))
    ping_energy = float(ping.detach().to(torch.float32).sum().item()) * float(ctx.cfg.ping_steps)
    with torch.no_grad():
        for t_idx in range(ctx.cfg.ping_steps):
            _step_network_once(ctx.net, zero, int(t_idx), ping_drive=ping)
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size)
    ping_spike_count = ping_energy
    return int(pred[0].item()), int(fire[0].item()), ping_energy, ping_spike_count, restore_info


def _run_weak_cue_spikes_from_boundary(ctx: ExperimentContext, boundary: Mapping[str, Mapping[str, torch.Tensor]], spikes: torch.Tensor) -> tuple[int, int]:
    batch_size = int(spikes.shape[0])
    restore_condition_state_for_functional_readout(ctx, boundary, batch_size)
    with torch.no_grad():
        current_time = 0
        for t in range(spikes.shape[1]):
            current_time = _step_network_once(ctx.net, spikes[:, t, ...], current_time)
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size)
    return int(pred[0].item()), int(fire[0].item())


def _run_weak_cue_from_boundary(ctx: ExperimentContext, boundary: Mapping[str, Mapping[str, torch.Tensor]], image: torch.Tensor) -> tuple[int, int]:
    spikes = encode_images(ctx.encoder, image.unsqueeze(0).to(ctx.device), ctx.cfg.weak_probe_steps)
    return _run_weak_cue_spikes_from_boundary(ctx, boundary, spikes)


def _run_weak_cue_multi_boundary_batch(
    ctx: ExperimentContext,
    boundaries: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
    cue_spikes: torch.Tensor,
    condition_names: Sequence[str],
) -> dict[str, tuple[int, int]]:
    if ctx.cfg.enable_condition_batch:
        ctx.warnings.append("Fig.3 weak-cue boundary batch helper is scaffolded; falling back to order-preserving per-boundary rollout.")
    return {
        str(name): _run_weak_cue_spikes_from_boundary(ctx, boundary, cue_spikes)
        for name, boundary in zip(condition_names, boundaries)
    }


def _step_network_once(net, input_t: torch.Tensor, current_time: int, *, stsp_mode: str = "dynamic", ping_drive: torch.Tensor | None = None) -> int:
    s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode, ping_drive=ping_drive)
    s1p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s2p = net.pool2(s2.float())
    net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    return current_time + 1


def _restore_boundary_state(net, boundary: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    with torch.no_grad():
        for layer_key, state in boundary.items():
            layer = getattr(net, layer_key)
            for src_key, attr in (("v_mem", "v_mem"), ("g_e", "g_e"), ("res", "res")):
                if src_key in state:
                    getattr(layer, attr).copy_(state[src_key].to(device=getattr(layer, attr).device, dtype=getattr(layer, attr).dtype))
            if "inh_trace" in state:
                layer.lateral_inh.inh_trace.copy_(state["inh_trace"].to(device=layer.lateral_inh.inh_trace.device, dtype=layer.lateral_inh.inh_trace.dtype))
            if "u" in state and getattr(layer, "u_pre", None) is not None:
                layer.u_pre.copy_(state["u"].to(device=layer.u_pre.device, dtype=layer.u_pre.dtype))
            if "x" in state and getattr(layer, "x_pre", None) is not None:
                layer.x_pre.copy_(state["x"].to(device=layer.x_pre.device, dtype=layer.x_pre.dtype))


def _region_ping_serial_bins(raw: pd.DataFrame, seq_len: int | None = None) -> list[str]:
    if seq_len is None:
        max_len = int(pd.to_numeric(raw.get("seq_len", pd.Series([0])), errors="coerce").max()) if not raw.empty else int(0)
    else:
        max_len = int(seq_len)
    return [f"pos_{idx}" for idx in range(1, max_len + 1)] + ["other", "silent"]


def _region_ping_position_distribution(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = ["network_seed", "state_condition", "memory_condition", "region_condition", "seq_len", "serial_bin", "readout_mass", "n_trials"]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    for keys, part in raw.groupby(["state_condition", "memory_condition", "region_condition", "seq_len"], sort=True):
        state_condition, memory_condition, region_condition, seq_len = keys
        for serial_bin in _region_ping_serial_bins(part, int(seq_len)):
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "state_condition": str(state_condition),
                    "memory_condition": str(memory_condition),
                    "region_condition": str(region_condition),
                    "seq_len": int(seq_len),
                    "serial_bin": serial_bin,
                    "readout_mass": float((part["serial_bin"].astype(str) == serial_bin).mean()),
                    "n_trials": int(len(part)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _region_ping_summary(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if raw.empty:
        return pd.DataFrame()
    for keys, part in raw.groupby(["state_condition", "memory_condition", "region_condition"], sort=True):
        state_condition, memory_condition, region_condition = keys
        fire = pd.to_numeric(part["first_fire_time_ms"], errors="coerce").replace(-1, np.nan)
        rows.append(
            {
                "network_seed": int(network_seed),
                "state_condition": str(state_condition),
                "memory_condition": str(memory_condition),
                "region_condition": str(region_condition),
                "P_seen_item": float(part["pred_is_seen_item"].mean()),
                "P_latest_item": float(part["pred_is_latest_item"].mean()),
                "P_recent_item": float(part["pred_is_recent_item"].mean()),
                "P_earlier_item": float(part["pred_is_earlier_item"].mean()),
                "P_unseen": float(part["pred_is_unseen"].mean()),
                "P_silent": float(part["silent"].mean()),
                "mean_first_fire_time_ms": float(fire.mean()),
                "median_first_fire_time_ms": float(fire.median()),
                "n_trials": int(len(part)),
                "active_unit_count_mean": float(pd.to_numeric(part["active_unit_count"], errors="coerce").mean()),
                "total_ping_current_mean": float(pd.to_numeric(part["total_ping_current"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _region_ping_contrast(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if raw.empty:
        return pd.DataFrame()
    main = raw[
        raw["state_condition"].astype(str).eq("S_final")
        & raw["memory_condition"].astype(str).eq("sequence_state")
        & raw["region_condition"].astype(str).isin(["peak", "valley"])
    ].copy()
    if main.empty:
        return pd.DataFrame()
    group_cols = ["state_condition", "memory_condition", "support_metric", "region_q"]
    for keys, part in main.groupby(group_cols, sort=True):
        state_condition, memory_condition, support_metric, region_q = keys
        bins = _region_ping_serial_bins(part)
        peak = part[part["region_condition"].astype(str).eq("peak")]
        valley = part[part["region_condition"].astype(str).eq("valley")]
        p = _serial_distribution(peak, bins)
        q = _serial_distribution(valley, bins)
        paired = peak.merge(
            valley,
            on=["sequence_id", "ping_repeat", "state_condition", "memory_condition"],
            suffixes=("_peak", "_valley"),
        )
        if paired.empty:
            label_diff = float("nan")
        else:
            label_diff = float((paired["predicted_label_peak"].astype(int) != paired["predicted_label_valley"].astype(int)).mean())
        fire_peak = pd.to_numeric(peak["first_fire_time_ms"], errors="coerce").replace(-1, np.nan)
        fire_valley = pd.to_numeric(valley["first_fire_time_ms"], errors="coerce").replace(-1, np.nan)
        rows.append(
            {
                "network_seed": int(network_seed),
                "state_condition": str(state_condition),
                "memory_condition": str(memory_condition),
                "support_metric": str(support_metric),
                "region_q": float(region_q),
                "JS_peak_valley": _js_divergence(p, q),
                "TV_peak_valley": _tv_distance(p, q),
                "P_peak_label_differs_from_valley": label_diff,
                "P_peak_seen_minus_valley_seen": float(peak["pred_is_seen_item"].mean() - valley["pred_is_seen_item"].mean()),
                "P_peak_latest_minus_valley_latest": float(peak["pred_is_latest_item"].mean() - valley["pred_is_latest_item"].mean()),
                "latency_peak_minus_valley": float(fire_peak.median() - fire_valley.median()),
                "n_sequences": int(part["sequence_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _region_ping_current_matching(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = [
        "network_seed",
        "region_condition",
        "support_metric",
        "region_q",
        "active_unit_count_mean",
        "active_unit_count_std",
        "total_ping_current_mean",
        "total_ping_current_std",
        "n_trials",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    main = raw[raw["state_condition"].astype(str).eq("S_final")].copy()
    for keys, part in main.groupby(["region_condition", "support_metric", "region_q"], sort=True):
        region_condition, support_metric, region_q = keys
        active = pd.to_numeric(part["active_unit_count"], errors="coerce")
        current = pd.to_numeric(part["total_ping_current"], errors="coerce")
        rows.append(
            {
                "network_seed": int(network_seed),
                "region_condition": str(region_condition),
                "support_metric": str(support_metric),
                "region_q": float(region_q),
                "active_unit_count_mean": float(active.mean()),
                "active_unit_count_std": float(active.std(ddof=0)),
                "total_ping_current_mean": float(current.mean()),
                "total_ping_current_std": float(current.std(ddof=0)),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _region_ping_current_matching_status(matching: pd.DataFrame) -> str:
    if matching.empty:
        return "missing"
    required = {"peak", "valley", "random"}
    if not required.issubset(set(matching["region_condition"].astype(str))):
        return "failed"
    active = pd.to_numeric(matching["active_unit_count_mean"], errors="coerce").dropna().to_numpy(dtype=float)
    current = pd.to_numeric(matching["total_ping_current_mean"], errors="coerce").dropna().to_numpy(dtype=float)
    if active.size == 0 or current.size == 0:
        return "failed"
    if float(np.max(active) - np.min(active)) > 1e-9:
        return "failed"
    if float(np.max(current) - np.min(current)) > 1e-9:
        return "failed"
    return "passed"


def _region_ping_amp_sweep_summary(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if raw.empty:
        return pd.DataFrame()
    for keys, part in raw.groupby(["ping_amp", "region_condition", "state_condition"], sort=True):
        ping_amp, region_condition, state_condition = keys
        fire = pd.to_numeric(part["first_fire_time_ms"], errors="coerce").replace(-1, np.nan)
        rows.append(
            {
                "network_seed": int(network_seed),
                "ping_amp": float(ping_amp),
                "region_condition": str(region_condition),
                "state_condition": str(state_condition),
                "P_seen_item": float(part["pred_is_seen_item"].mean()),
                "P_latest_item": float(part["pred_is_latest_item"].mean()),
                "P_unseen": float(part["pred_is_unseen"].mean()),
                "P_silent": float(part["silent"].mean()),
                "mean_first_fire_time_ms": float(fire.mean()),
                "median_first_fire_time_ms": float(fire.median()),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def _region_ping_amp_sweep_latency(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if raw.empty:
        return pd.DataFrame()
    for keys, part in raw.groupby(["region_condition", "state_condition", "ping_amp"], sort=True):
        region_condition, state_condition, ping_amp = keys
        fire = pd.to_numeric(part["first_fire_time_ms"], errors="coerce").replace(-1, np.nan)
        rows.append(
            {
                "network_seed": int(network_seed),
                "region_condition": str(region_condition),
                "state_condition": str(state_condition),
                "ping_amp": float(ping_amp),
                "median_first_fire_time_ms": float(fire.median()),
                "P_fire_by_ping_end": float((pd.to_numeric(part["first_fire_time_ms"], errors="coerce") >= 0).mean()),
                "P_seen_item": float(part["pred_is_seen_item"].mean()),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def _serial_distribution(part: pd.DataFrame, bins: Sequence[str]) -> np.ndarray:
    if part.empty:
        return np.full(len(bins), 1.0 / max(1, len(bins)), dtype=np.float64)
    values = part["serial_bin"].astype(str)
    counts = np.asarray([(values == str(bin_name)).sum() for bin_name in bins], dtype=np.float64)
    denom = float(counts.sum())
    if denom <= 0.0:
        return np.full(len(bins), 1.0 / max(1, len(bins)), dtype=np.float64)
    return counts / denom


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    pp = np.asarray(p, dtype=np.float64)
    qq = np.asarray(q, dtype=np.float64)
    pp = pp / max(float(pp.sum()), 1e-12)
    qq = qq / max(float(qq.sum()), 1e-12)
    mm = 0.5 * (pp + qq)
    return float(0.5 * _kl_divergence(pp, mm) + 0.5 * _kl_divergence(qq, mm))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], 1e-12))))


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)).sum())


def compute_fig3e_weak_probe_metrics(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "target_source",
        "seq_len",
        "state_condition",
        "memory_condition",
        "keep_prob",
        "P_target",
        "P_seen_item",
        "P_other_seen_item",
        "P_latest_item",
        "P_unseen",
        "P_silent",
        "mean_first_fire_time_ms",
        "n_trials",
        "weak_probe_metric_mode",
        "weak_probe_mask_space",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, part in raw.groupby(["target_source", "seq_len", "state_condition", "memory_condition", "keep_prob"], sort=True):
        target_source, seq_len, state_condition, memory_condition, keep_prob = str(keys[0]), int(keys[1]), str(keys[2]), str(keys[3]), float(keys[4])
        denom = max(1, len(part))
        rows.append(
            {
                "network_seed": int(network_seed),
                "target_source": target_source,
                "seq_len": seq_len,
                "state_condition": state_condition,
                "memory_condition": memory_condition,
                "keep_prob": keep_prob,
                "P_target": float(part["pred_is_target"].sum() / denom),
                "P_seen_item": float(part["pred_is_seen_item"].sum() / denom),
                "P_other_seen_item": float(part["pred_is_other_seen_item"].sum() / denom),
                "P_latest_item": float(part["pred_is_latest_item"].sum() / denom),
                "P_unseen": float(part["pred_is_unseen"].sum() / denom),
                "P_silent": float(part["silent"].sum() / denom),
                "mean_first_fire_time_ms": float(pd.to_numeric(part["first_fire_time_ms"], errors="coerce").replace(-1, np.nan).mean()),
                "n_trials": int(len(part)),
                "weak_probe_metric_mode": _mode_value(part, "weak_probe_metric_mode", "fig2_compat"),
                "weak_probe_mask_space": _mode_value(part, "mask_space", ""),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compute_fig3e_weak_probe_auc_metrics(network_seed: int, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    for (target_source, seq_len), target_part in metrics.groupby(["target_source", "seq_len"], sort=True):
        auc_by_mem: dict[str, float] = {}
        p50_by_mem: dict[str, float] = {}
        for memory_condition, part in target_part.groupby("memory_condition", sort=True):
            ordered = part.sort_values("keep_prob")
            x = ordered["keep_prob"].to_numpy(dtype=float)
            y = ordered["P_target"].to_numpy(dtype=float)
            auc_by_mem[str(memory_condition)] = _normalized_auc(x, y)
            p50_by_mem[str(memory_condition)] = _p50_from_curve(x, y, threshold=0.5)
        for (_, row) in target_part.iterrows():
            mem = str(row["memory_condition"])
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "target_source": str(target_source),
                    "seq_len": int(seq_len),
                    "state_condition": str(row["state_condition"]),
                    "memory_condition": mem,
                    "normalized_auc_target_recovery": float(auc_by_mem.get(mem, np.nan)),
                    "p50_target_recovery_keep_prob": float(p50_by_mem.get(mem, np.nan)),
                    "sequence_vs_S0_auc_gain": float(auc_by_mem.get("sequence_state", np.nan) - auc_by_mem.get("cue_only", np.nan)),
                    "sequence_vs_S0_p50_shift": _nan_diff(p50_by_mem.get("sequence_state"), p50_by_mem.get("cue_only")),
                    "low_cue_gain": _fig3f_cue_gain(target_part, max_keep=0.1),
                    "mid_cue_gain": _fig3f_cue_gain(target_part, min_keep=0.1, max_keep=0.3),
                    "high_cue_gain": _fig3f_cue_gain(target_part, min_keep=0.3),
                    "weak_probe_metric_mode": str(row.get("weak_probe_metric_mode", "")),
                    "weak_probe_mask_space": str(row.get("weak_probe_mask_space", "")),
                    "n_trials": int(target_part["n_trials"].sum()),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["target_source", "seq_len", "state_condition", "memory_condition"]).reset_index(drop=True)


def compute_fig3e_weak_probe_memory_gain(network_seed: int, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    for (target_source, seq_len, keep_prob), part in metrics.groupby(["target_source", "seq_len", "keep_prob"], sort=True):
        seq = part[part["memory_condition"].astype(str).eq("sequence_state")]
        single = part[part["memory_condition"].astype(str).eq("single_item_memory")]
        cue = part[part["memory_condition"].astype(str).eq("cue_only")]
        rows.append(
            {
                "network_seed": int(network_seed),
                "target_source": str(target_source),
                "seq_len": int(seq_len),
                "keep_prob": float(keep_prob),
                "P_target_sequence_state": _first_float(seq, "P_target"),
                "P_target_single_item_memory": _first_float(single, "P_target"),
                "P_target_cue_only": _first_float(cue, "P_target"),
                "sequence_minus_S0": float(_first_float(seq, "P_target") - _first_float(cue, "P_target")),
                "sequence_minus_single_item": float(_first_float(seq, "P_target") - _first_float(single, "P_target")),
                "single_item_minus_S0": float(_first_float(single, "P_target") - _first_float(cue, "P_target")),
                "P_seen_sequence_state": _first_float(seq, "P_seen_item"),
                "P_seen_single_item_memory": _first_float(single, "P_seen_item"),
                "P_seen_cue_only": _first_float(cue, "P_seen_item"),
                "seen_sequence_minus_S0": float(_first_float(seq, "P_seen_item") - _first_float(cue, "P_seen_item")),
                "seen_sequence_minus_single_item": float(_first_float(seq, "P_seen_item") - _first_float(single, "P_seen_item")),
                "seen_single_item_minus_S0": float(_first_float(single, "P_seen_item") - _first_float(cue, "P_seen_item")),
                "P_silent_sequence_state": _first_float(seq, "P_silent"),
                "P_silent_single_item_memory": _first_float(single, "P_silent"),
                "P_silent_cue_only": _first_float(cue, "P_silent"),
                "silent_reduction_sequence_vs_S0": float(_first_float(cue, "P_silent") - _first_float(seq, "P_silent")),
                "silent_reduction_single_item_vs_S0": float(_first_float(cue, "P_silent") - _first_float(single, "P_silent")),
                "n_trials": int(min(_first_float(seq, "n_trials"), _first_float(single, "n_trials"), _first_float(cue, "n_trials"))),
            }
        )
    return pd.DataFrame(rows)


def compute_fig3e_weak_probe_position_stratified_metrics(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "target_source",
        "seq_len",
        "target_position_bin",
        "memory_condition",
        "keep_prob",
        "P_target",
        "P_seen_item",
        "P_silent",
        "n_trials",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, part in raw.groupby(["target_source", "seq_len", "target_position_bin", "memory_condition", "keep_prob"], sort=True):
        target_source, seq_len, position_bin, memory_condition, keep_prob = keys
        denom = max(1, len(part))
        rows.append(
            {
                "network_seed": int(network_seed),
                "target_source": str(target_source),
                "seq_len": int(seq_len),
                "target_position_bin": str(position_bin),
                "memory_condition": str(memory_condition),
                "keep_prob": float(keep_prob),
                "P_target": float(part["pred_is_target"].sum() / denom),
                "P_seen_item": float(part["pred_is_seen_item"].sum() / denom),
                "P_silent": float(part["silent"].sum() / denom),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compute_fig3f_weak_probe_metrics(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    return compute_fig3e_weak_probe_metrics(network_seed, raw)


def compute_fig3f_weak_probe_auc_metrics(network_seed: int, metrics: pd.DataFrame) -> pd.DataFrame:
    return compute_fig3e_weak_probe_auc_metrics(network_seed, metrics)


def compute_fig3f_weak_probe_memory_gain(network_seed: int, metrics: pd.DataFrame) -> pd.DataFrame:
    return compute_fig3e_weak_probe_memory_gain(network_seed, metrics)


def _normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return float("nan")
    if len(x) == 1:
        return float(y.mean())
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = float(x[-1] - x[0])
    if span <= 0.0:
        return float(np.nanmean(y))
    return float(np.trapezoid(y, x) / span)


def _p50_from_curve(x: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return float("nan")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if not np.any(y >= float(threshold)):
        return float("nan")
    first = int(np.argmax(y >= float(threshold)))
    if first == 0:
        return float(x[0])
    x0, x1 = float(x[first - 1]), float(x[first])
    y0, y1 = float(y[first - 1]), float(y[first])
    if abs(y1 - y0) <= 1e-12:
        return x1
    frac = (float(threshold) - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0))


def _nan_diff(a: Any, b: Any) -> float:
    aa = float(a) if a is not None else float("nan")
    bb = float(b) if b is not None else float("nan")
    return float(aa - bb) if math.isfinite(aa) and math.isfinite(bb) else float("nan")


def _mode_value(part: pd.DataFrame, column: str, default: str) -> str:
    if column not in part.columns or part.empty:
        return str(default)
    values = part[column].dropna().astype(str).unique()
    return str(values[0]) if len(values) else str(default)


def _fig3f_cue_gain(part: pd.DataFrame, *, min_keep: float = -np.inf, max_keep: float = np.inf) -> float:
    sub = part[(pd.to_numeric(part["keep_prob"], errors="coerce") > float(min_keep)) & (pd.to_numeric(part["keep_prob"], errors="coerce") <= float(max_keep))]
    if sub.empty:
        return float("nan")
    pivot = sub.pivot_table(index="keep_prob", columns="memory_condition", values="P_target", aggfunc="mean")
    if not {"sequence_state", "cue_only"}.issubset(pivot.columns):
        return float("nan")
    return float((pivot["sequence_state"] - pivot["cue_only"]).mean())


def _completion_metrics(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cue_condition, part in raw.groupby("cue_condition", sort=False):
        seq = part[part["memory_condition"] == "sequence_state"]
        cue = part[part["memory_condition"] == "cue_only"]
        p_seq = float(seq["pred_is_target"].mean()) if not seq.empty else 0.0
        p_cue = float(cue["pred_is_target"].mean()) if not cue.empty else 0.0
        rows.append(
            {
                "network_seed": int(network_seed),
                "cue_condition": cue_condition,
                "target_position": int(part["target_position"].mode().iloc[0]) if not part.empty else -1,
                "P_target_sequence": p_seq,
                "P_target_cue_only": p_cue,
                "completion_gain": float(p_seq - p_cue),
                "P_seen_item": float(seq["pred_is_seen_item"].mean()) if not seq.empty else 0.0,
                "P_other": float(seq["pred_is_other"].mean()) if not seq.empty else 0.0,
                "P_silent": float(seq["silent"].mean()) if not seq.empty else 0.0,
                "n_trials": int(len(seq)),
            }
        )
    return pd.DataFrame(rows)


def _main_sequence_meta(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> pd.DataFrame:
    meta = bank.sequence_meta.copy()
    if bool(ctx.cfg.main_only_seq_len_10):
        use = meta[meta["seq_len"].astype(int).eq(int(ctx.cfg.main_sequence_length))].copy()
        if not use.empty:
            return use
        ctx.warnings.append(f"No seq_len={ctx.cfg.main_sequence_length} sequences available; using all sequence lengths for main Fig.3 E/F analyses.")
    return meta


def _weak_cue_target_sources(value: str) -> tuple[str, ...]:
    text = str(value).strip()
    if text == "both":
        return ("sequence_member_random", "unseen_random")
    if text not in {"sequence_member_random", "unseen_random"}:
        return ("sequence_member_random",)
    return (text,)


def _sample_weak_cue_target(
    ctx: ExperimentContext,
    target_source: str,
    seq_len: int,
    item_ids: Sequence[int],
    labels: Sequence[int],
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    if target_source == "sequence_member_random":
        position = int(rng.integers(1, int(seq_len) + 1))
        return position, int(item_ids[position - 1]), int(labels[position - 1])
    seen_labels = {int(v) for v in labels}
    candidate_labels = [label for label in range(NUM_CLASSES) if label not in seen_labels] or list(range(NUM_CLASSES))
    label = int(rng.choice(candidate_labels))
    pool = [int(idx) for idx in ctx.class_index[label] if int(idx) not in set(int(v) for v in item_ids)]
    if not pool:
        pool = [int(idx) for idx in ctx.class_index[label]]
    image_id = int(rng.choice(pool))
    return -1, image_id, label


def _support_map_for_structural_cue(ctx: ExperimentContext, landscape: Mapping[str, np.ndarray]) -> np.ndarray:
    delta = np.asarray(landscape.get("delta_gain_map"), dtype=np.float32)
    if delta.size and np.isfinite(delta).all() and float(np.std(delta)) > 1e-12:
        return delta
    ctx.warnings.append("Structural weak-cue ranking fell back from delta_G to G_final for at least one sequence.")
    return np.asarray(landscape["G_final"], dtype=np.float32)


def build_ranked_foreground_masks(
    support_map: np.ndarray,
    target_image: torch.Tensor | np.ndarray,
    keep_fraction: float,
    rng: np.random.Generator,
    foreground_threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    support = np.asarray(support_map, dtype=float)
    image = target_image.detach().cpu().numpy() if isinstance(target_image, torch.Tensor) else np.asarray(target_image)
    image2d = np.squeeze(image).astype(float)
    foreground = image2d > float(foreground_threshold)
    if not np.any(foreground):
        foreground = image2d >= float(np.nanmax(image2d))
    fg_idx = np.flatnonzero(foreground.reshape(-1))
    count = max(1, int(round(float(keep_fraction) * max(1, fg_idx.size))))
    count = min(count, max(1, fg_idx.size))
    support_flat = support.reshape(-1)
    fg_support = support_flat[fg_idx]
    order = np.argsort(fg_support, kind="mergesort")
    valley_idx = fg_idx[order[:count]]
    peak_idx = fg_idx[order[-count:]]
    random_idx = rng.choice(fg_idx, size=count, replace=fg_idx.size < count)
    masks = {
        "peak": _mask_from_flat_indices(support.shape, peak_idx),
        "valley": _mask_from_flat_indices(support.shape, valley_idx),
        "random": _mask_from_flat_indices(support.shape, random_idx),
    }
    return masks, {"foreground_mask": foreground.astype(bool)}


def _mask_from_flat_indices(shape: Sequence[int], indices: np.ndarray) -> np.ndarray:
    out = np.zeros(int(np.prod(shape)), dtype=bool)
    out[np.asarray(indices, dtype=int)] = True
    return out.reshape(tuple(shape))


def _selected_quantile_mean(support_fg: np.ndarray, selected: np.ndarray) -> float:
    fg = np.asarray(support_fg, dtype=float).reshape(-1)
    vals = np.asarray(selected, dtype=float).reshape(-1)
    fg = fg[np.isfinite(fg)]
    vals = vals[np.isfinite(vals)]
    if fg.size == 0 or vals.size == 0:
        return 0.0
    sorted_fg = np.sort(fg)
    ranks = np.searchsorted(sorted_fg, vals, side="right") / max(1, sorted_fg.size)
    return float(np.mean(ranks))


def _structural_accuracy(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if raw.empty:
        return pd.DataFrame(columns=["network_seed", "cue_condition", "keep_fraction", "memory_condition", "accuracy", "P_target", "P_seen_item", "P_unseen", "P_silent", "mean_first_fire_time_ms", "n_trials"])
    main = raw[raw["target_source"].astype(str).eq("sequence_member_random")].copy()
    for (cue_condition, keep_fraction, memory_condition), part in main.groupby(["cue_condition", "keep_fraction", "memory_condition"], sort=True):
        rows.append(
            {
                "network_seed": int(network_seed),
                "cue_condition": str(cue_condition),
                "keep_fraction": float(keep_fraction),
                "memory_condition": str(memory_condition),
                "accuracy": float(part["correct"].mean()),
                "P_target": float(part["pred_is_target"].mean()),
                "P_seen_item": float(part["pred_is_seen_item"].mean()),
                "P_unseen": float(part["pred_is_unseen"].mean()),
                "P_silent": float(part["silent"].mean()),
                "mean_first_fire_time_ms": float(pd.to_numeric(part["first_fire_time_ms"], errors="coerce").replace(-1, np.nan).mean()),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def _structural_memory_gain(network_seed: int, accuracy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if accuracy.empty:
        return pd.DataFrame(columns=["network_seed", "cue_condition", "keep_fraction", "accuracy_sequence_state", "accuracy_cue_only", "memory_gain", "P_silent_sequence_state", "P_silent_cue_only", "n_trials"])
    for (cue_condition, keep_fraction), part in accuracy.groupby(["cue_condition", "keep_fraction"], sort=True):
        seq = part[part["memory_condition"].astype(str).eq("sequence_state")]
        cue = part[part["memory_condition"].astype(str).eq("cue_only")]
        rows.append(
            {
                "network_seed": int(network_seed),
                "cue_condition": str(cue_condition),
                "keep_fraction": float(keep_fraction),
                "accuracy_sequence_state": _first_float(seq, "accuracy"),
                "accuracy_cue_only": _first_float(cue, "accuracy"),
                "memory_gain": float(_first_float(seq, "accuracy") - _first_float(cue, "accuracy")),
                "P_silent_sequence_state": _first_float(seq, "P_silent"),
                "P_silent_cue_only": _first_float(cue, "P_silent"),
                "n_trials": int(min(_first_float(seq, "n_trials"), _first_float(cue, "n_trials"))),
            }
        )
    return pd.DataFrame(rows)


def _structural_target_source_control(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if raw.empty:
        return pd.DataFrame(columns=["network_seed", "target_source", "cue_condition", "keep_fraction", "memory_condition", "accuracy", "memory_gain", "n_trials"])
    grouped = raw.groupby(["target_source", "cue_condition", "keep_fraction", "memory_condition"], sort=True)
    acc_rows = []
    for keys, part in grouped:
        target_source, cue_condition, keep_fraction, memory_condition = keys
        acc_rows.append(
            {
                "target_source": str(target_source),
                "cue_condition": str(cue_condition),
                "keep_fraction": float(keep_fraction),
                "memory_condition": str(memory_condition),
                "accuracy": float(part["correct"].mean()),
                "n_trials": int(len(part)),
            }
        )
    acc = pd.DataFrame(acc_rows)
    gains: dict[tuple[str, str, float], float] = {}
    for (target_source, cue_condition, keep_fraction), part in acc.groupby(["target_source", "cue_condition", "keep_fraction"], sort=True):
        seq = part[part["memory_condition"].eq("sequence_state")]
        cue = part[part["memory_condition"].eq("cue_only")]
        gains[(str(target_source), str(cue_condition), float(keep_fraction))] = float(_first_float(seq, "accuracy") - _first_float(cue, "accuracy"))
    for _, row in acc.iterrows():
        rows.append(
            {
                "network_seed": int(network_seed),
                "target_source": row["target_source"],
                "cue_condition": row["cue_condition"],
                "keep_fraction": float(row["keep_fraction"]),
                "memory_condition": row["memory_condition"],
                "accuracy": float(row["accuracy"]),
                "memory_gain": gains.get((str(row["target_source"]), str(row["cue_condition"]), float(row["keep_fraction"])), 0.0),
                "n_trials": int(row["n_trials"]),
            }
        )
    return pd.DataFrame(rows)


def _structural_matching_diagnostics(network_seed: int, masks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if masks.empty:
        return pd.DataFrame(columns=["network_seed", "cue_condition", "keep_fraction", "cue_pixel_count_mean", "cue_energy_mean", "encoded_spike_count_mean", "support_mean_selected", "support_mean_foreground", "n_masks"])
    main = masks[masks["target_source"].astype(str).eq("sequence_member_random")].copy()
    for (cue_condition, keep_fraction), part in main.groupby(["cue_condition", "keep_fraction"], sort=True):
        rows.append(
            {
                "network_seed": int(network_seed),
                "cue_condition": str(cue_condition),
                "keep_fraction": float(keep_fraction),
                "cue_pixel_count_mean": float(pd.to_numeric(part["cue_pixel_count"], errors="coerce").mean()),
                "cue_energy_mean": float(pd.to_numeric(part["cue_energy"], errors="coerce").mean()),
                "encoded_spike_count_mean": float(pd.to_numeric(part["encoded_spike_count"], errors="coerce").mean()),
                "support_mean_selected": float(pd.to_numeric(part["support_mean_selected"], errors="coerce").mean()),
                "support_mean_foreground": float(pd.to_numeric(part["support_mean_foreground"], errors="coerce").mean()),
                "n_masks": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def _first_float(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    value = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(value.iloc[0]) if not value.empty else 0.0


def _mean_numeric(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def _row_float(row: pd.Series, *columns: str) -> float:
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            return float(row[column])
    return 0.0


def _missing_csv_columns(path: Path, columns: Sequence[str]) -> list[str]:
    if not path.exists():
        return list(columns)
    try:
        present = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return list(columns)
    return [column for column in columns if column not in present]


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _structural_trial_columns() -> list[str]:
    return ["network_seed", "sequence_id", "seq_len", "target_source", "target_position", "target_image_id", "target_label", "repeat_id", "target_seed"]


def _structural_mask_columns() -> list[str]:
    return [
        "network_seed",
        "sequence_id",
        "target_source",
        "target_position",
        "target_image_id",
        "target_label",
        "keep_fraction",
        "cue_condition",
        "repeat_id",
        "mask_id",
        "mask_seed",
        "target_foreground_count",
        "cue_pixel_count",
        "cue_fraction_actual",
        "cue_energy",
        "encoded_spike_count",
        "support_mean_selected",
        "support_min_selected",
        "support_max_selected",
        "support_mean_foreground",
        "same_mask_used_across_memory_conditions",
    ]


def _structural_raw_columns() -> list[str]:
    return [
        "network_seed",
        "sequence_id",
        "seq_len",
        "target_source",
        "target_position",
        "target_image_id",
        "target_label",
        "keep_fraction",
        "cue_condition",
        "repeat_id",
        "mask_id",
        "memory_condition",
        "prediction",
        "correct",
        "pred_is_target",
        "pred_is_seen_item",
        "pred_is_unseen",
        "silent",
        "first_fire_time_ms",
        "cue_pixel_count",
        "target_foreground_count",
        "cue_fraction_actual",
        "cue_energy",
        "encoded_spike_count",
        "support_mean_selected",
        "support_mean_foreground",
        "support_quantile_mean",
    ]


def _cue_masks_for_target(ctx: ExperimentContext, landscape: Mapping[str, np.ndarray], image_id: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    foreground = _foreground_mask(ctx.dataset, image_id)
    peak = landscape["peak_mask"].astype(bool) & foreground
    valley = landscape["valley_mask"].astype(bool) & foreground
    delta = landscape["delta_gain_map"]
    if not np.any(peak):
        peak = _top_mask(delta, ctx.cfg.partial_cue_keep_fraction, positive=foreground)
    if not np.any(valley):
        valley = _bottom_mask(np.where(foreground, delta, np.inf), ctx.cfg.partial_cue_keep_fraction)
    target_count = max(1, int(round(float(ctx.cfg.partial_cue_keep_fraction) * max(1, int(foreground.sum())))))
    peak = _trim_or_expand_mask(peak, foreground, target_count, rng)
    valley = _trim_or_expand_mask(valley, foreground, target_count, rng)
    random = _random_mask_like(peak, foreground, rng)
    return {"peak_aligned": peak, "valley_aligned": valley, "random_matched": random}


def _network_peak_summary(network_seed: int, contrast: pd.DataFrame, nonflat: pd.DataFrame, prevalence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seq_len, part in contrast.groupby("seq_len", sort=True):
        nf = nonflat[nonflat["seq_len"] == seq_len]
        pv = prevalence[prevalence["seq_len"] == seq_len]
        vals = part["peak_valley_delta"].to_numpy(dtype=float)
        rows.append(
            {
                "network_seed": int(network_seed),
                "seq_len": int(seq_len),
                "mean_peak_valley_delta": float(np.mean(vals)) if vals.size else 0.0,
                "sem_peak_valley_delta": float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0,
                "fraction_structured_sequences": float(pv["is_structured"].mean()) if not pv.empty else 0.0,
                "mean_top_q_mass_fraction": float(nf["top_q_mass_fraction"].mean()) if not nf.empty else 0.0,
                "mean_support_gini": float(nf["support_gini"].mean()) if not nf.empty else 0.0,
                "n_sequences": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def save_debug_figures(ctx: ExperimentContext) -> None:
    apply_publication_style()
    jobs = [
        ("panel_b_progressive_update_metrics.csv", "stepwise_update_ratio", "fig3_debug_progressive_update"),
        ("panel_c_example_landscape_summary.csv", "peak_mean_support", "fig3_debug_example_landscape"),
        ("panel_d_ping_position_distribution.csv", "readout_mass", "fig3_debug_ping_distribution"),
        ("panel_e_weak_probe_metrics.csv", "P_target", "fig3_debug_weak_probe_target"),
        ("panel_e_weak_probe_memory_gain.csv", "target_recovery_gain", "fig3_debug_weak_probe_gain"),
    ]
    for filename, column, stem in jobs:
        path = ctx.metrics_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if column not in df.columns or df.empty:
            continue
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=150)
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        ax.hist(values, bins=min(20, max(3, len(values))), color="#4C78A8", alpha=0.8)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        save_figure_all_formats(fig, ctx.debug_dir / stem)
        plt.close(fig)
    _save_debug_category_plot(ctx, "panel_f_peak_cue_memory_gain.csv", "cue_condition", "memory_gain", "panel_f_peak_cue_memory_gain")
    matching_path = ctx.metrics_dir / "panel_f_peak_cue_matching_diagnostics.csv"
    if matching_path.exists():
        matching = pd.read_csv(matching_path)
        y_column = "cue_energy" if "cue_energy" in matching.columns else "encoded_spike_count"
        if y_column in matching.columns:
            _save_debug_category_plot(ctx, "panel_f_peak_cue_matching_diagnostics.csv", "cue_condition", y_column, "panel_f_peak_cue_matching")
    serial_path = ctx.metrics_dir / "supp_peak_cue_serial_position_gain.csv"
    if serial_path.exists():
        serial = pd.read_csv(serial_path)
        x_column = "target_position_bin" if "target_position_bin" in serial.columns else "relative_position"
        if x_column in serial.columns and "memory_gain" in serial.columns:
            _save_debug_category_plot(ctx, "supp_peak_cue_serial_position_gain.csv", x_column, "memory_gain", "supp_peak_cue_serial_position_gain")
    ctx.completed_modules["debug_figures"] = True


def _save_debug_category_plot(ctx: ExperimentContext, filename: str, x_column: str, y_column: str, stem: str) -> None:
    path = ctx.metrics_dir / filename
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or x_column not in df.columns or y_column not in df.columns:
        return
    import matplotlib.pyplot as plt

    plot_df = df.copy()
    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[y_column])
    if plot_df.empty:
        return
    grouped = plot_df.groupby(x_column, sort=True)[y_column].mean().reset_index()
    fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=150)
    ax.bar(grouped[x_column].astype(str), grouped[y_column].astype(float), color="#4C78A8", alpha=0.85)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.tick_params(axis="x", rotation=30)
    save_figure_all_formats(fig, ctx.debug_dir / stem)
    plt.close(fig)


def _write_config_files(ctx: ExperimentContext) -> None:
    cfg = ctx.cfg
    _write_json(_json_safe(asdict(cfg)), ctx.config_dir / "run_config.json")
    _write_json(
        {
            "main_panels": {
                "A": "multi-item sequence protocol",
                "B": "progressive state update / update weakening",
                "C": "peak-structured final STSP landscape",
                "D": "neutral ping serial-position readout",
                "E": "singleton-matched sequence-state weak-probe completion",
                "F": "STSP-region-gated ping serial-position readout",
            },
            "primary_layer": PRIMARY_LAYER,
            "primary_state_variable": PRIMARY_STATE_VARIABLE,
            "state_bank_required": True,
            "structural_weak_cue_in_main": False,
            "main_required_outputs": [
                "data/metrics/panel_b_progressive_update_metrics.csv",
                "data/metrics/panel_c_example_landscape_summary.csv",
                "data/metrics/panel_d_ping_position_distribution.csv",
                "data/metrics/panel_d_ping_summary.csv",
                "data/raw/panel_e_weak_probe_trial_readout.csv",
                "data/metrics/panel_e_weak_probe_metrics.csv",
                "data/metrics/panel_e_weak_probe_memory_gain.csv",
                "data/metrics/panel_e_weak_probe_position_stratified_metrics.csv",
                "data/raw/panel_f_region_ping_trial_readout.csv",
                "data/metrics/panel_f_region_ping_position_distribution.csv",
                "data/metrics/panel_f_region_ping_summary.csv",
                "data/metrics/panel_f_region_ping_contrast.csv",
                "data/metrics/panel_f_region_ping_current_matching.csv",
            ],
            "supplementary_outputs": {
                "S5": [
                    "data/metrics/supp_peak_valley_contrast.csv",
                    "data/metrics/supp_landscape_nonflatness.csv",
                    "data/metrics/supp_peak_valley_prevalence.csv",
                    "data/metrics/supp_network_peak_valley_summary.csv",
                    "data/metrics/supp_anchor_dynamics_metrics.csv",
                    "data/metrics/supp_recency_only_controls.csv",
                ],
                "S6": [
                    "data/metrics/supp_ping_recency_diagnostics.csv",
                    "data/metrics/supp_weak_probe_target_source_control.csv",
                    "data/metrics/supp_weak_probe_target_source_gain.csv",
                    "data/metrics/supp_structural_weak_cue_matching_diagnostics.csv",
                    "data/metrics/supp_structural_weak_cue_accuracy.csv",
                    "data/metrics/supp_structural_weak_cue_memory_gain.csv",
                    "data/metrics/supp_peak_cue_serial_position_metrics.csv",
                    "data/metrics/supp_peak_cue_serial_position_gain.csv",
                    "data/raw/supp_region_ping_amp_sweep_trial_readout.csv",
                    "data/metrics/supp_region_ping_amp_sweep_summary.csv",
                    "data/metrics/supp_region_ping_amp_sweep_latency.csv",
                ],
            },
        },
        ctx.config_dir / "figure_requirements.json",
    )
    _write_json(
        {
            "state_conditions": ["S0", "S_1..S_K", "S_final", "singleton_reference", "singleton_boundary", "decay_counterfactual"],
            "primary_representation": {"layer": PRIMARY_LAYER, "state_variable": PRIMARY_STATE_VARIABLE},
            "distance": "centered_cosine_distance",
            "fig3_boundary": "Exploratory multi-item STSP morphology and functional readout; mechanism proof is reserved for Fig.6.",
            "population_morphology_diagnostics": "supplementary",
            "panel_d_neutral_ping": "STSP-only restore of S0/S_final followed by neutral constant-drive ping; readout is serial-position distribution.",
            "panel_e_weak_probe": "Fig.2F-compatible encoded-spike dropout weak probe; same degraded spike probe across cue_only, slot-matched singleton, and final sequence STSP states.",
            "panel_f_region_ping": "Main Fig.3F; peak/valley/random STSP regions are pinged directly from final landscape without target images or weak cue construction.",
            "panel_f_structural_weak_cue": "Legacy/supplement only; not the main Fig.3F source for this design version.",
            "region_ping": {
                "enabled": bool(cfg.run_region_ping),
                "support_metric": str(cfg.region_ping_support_metric),
                "region_q": float(cfg.region_ping_q),
                "conditions": list(cfg.region_ping_conditions),
                "repeats": int(cfg.region_ping_repeats),
                "s0_control": bool(cfg.run_region_ping_s0_control),
                "amp_sweep": bool(cfg.run_region_ping_amp_sweep),
                "amp_values": list(cfg.region_ping_amp_sweep),
            },
        },
        ctx.config_dir / "condition_spec.json",
    )
    _write_json(
        {
            "restore_mode": "reset_all_state_restore_selected_stsp",
            "weak_probe_mask_space": str(cfg.weak_probe_mask_space),
            "weak_probe_use_same_mask_across_states": bool(cfg.weak_probe_use_same_mask_across_states),
            "weak_probe_scale": float(cfg.weak_probe_scale),
            "weak_probe_noise": float(cfg.weak_probe_noise),
            "weak_probe_metric_mode": str(cfg.weak_probe_metric_mode),
            "weak_probe_memory_scope": str(cfg.weak_probe_memory_scope),
            "weak_probe_target_source": str(cfg.weak_probe_target_source),
            "weak_probe_include_singleton": bool(cfg.weak_probe_include_singleton),
            "weak_probe_memory_conditions": list(MEMORY_CONDITIONS),
            "fig2F_compat_enabled": bool(cfg.weak_probe_mask_space == "encoded_spikes" and cfg.weak_probe_metric_mode == "fig2_compat"),
            "fig4_weak_probe_method_compat_enabled": bool(cfg.weak_probe_mask_space == "encoded_spikes"),
            "structural_weak_cue_in_main": False,
            "panel_e": "encoded-spike dropout weak-probe completion",
            "panel_f": "STSP-region-gated masked ping readout",
            "region_ping": {
                "support_metric": str(cfg.region_ping_support_metric),
                "region_q": float(cfg.region_ping_q),
                "conditions": list(cfg.region_ping_conditions),
                "ping_amp": float(cfg.ping_amp),
                "ping_ms": int(cfg.ping_ms),
                "s0_control": bool(cfg.run_region_ping_s0_control),
                "amp_sweep_enabled": bool(cfg.run_region_ping_amp_sweep),
                "amp_sweep": list(cfg.region_ping_amp_sweep),
            },
        },
        ctx.config_dir / "functional_readout_spec.json",
    )
    _write_json(
        {
            "definition": "Masks are defined from final pre-cue STSP support landscape before weak-cue presentation.",
            "support": "delta_G = G_final - G_baseline",
            "peak_q": float(cfg.peak_q),
            "valley_q": float(cfg.valley_q),
            "cue_conditions": list(CUE_CONDITIONS),
            "mask_mode": str(cfg.weak_cue_mask_mode),
            "foreground_threshold": float(cfg.foreground_threshold),
            "pre_cue": True,
        },
        ctx.config_dir / "mask_definition_spec.json",
    )
    _write_json(
        {
            "purpose": "Exploratory unbiased neutral-ping readout distribution from final multi-item STSP state.",
            "main_state_condition": "S_final",
            "baseline_state_condition": "S0",
            "decoder": "decode_prediction_and_fire_time_from_layer3",
            "main_metric": "position_distribution",
            "not_main_claim": ["recent_bias", "latest_item_mass", "ping_COM"],
            "ping_ms": int(cfg.ping_ms),
            "ping_amp": float(cfg.ping_amp),
            "state_conditions": list(cfg.ping_main_state_conditions),
        },
        ctx.config_dir / "ping_readout_spec.json",
    )
    _write_json(
        {
            "purpose": "Test whether target-foreground locations with high, low, or random final STSP support differ in weak-cue classification efficacy.",
            "main_or_supplement": "main_fig3f_with_supplemental_controls",
            "target_source_main": "sequence_member_random",
            "mask_mode": str(cfg.weak_cue_mask_mode),
            "support_map_for_ranking": "delta_G; fallback to G_final if delta_G is non-finite or flat, with warning recorded in summary.json",
            "cue_conditions": list(CUE_CONDITIONS),
            "memory_conditions": list(MEMORY_CONDITIONS),
            "primary_metric": "accuracy and memory_gain",
            "keep_fractions": list(cfg.weak_cue_keep_fractions),
            "main_keep_fraction": float(cfg.peak_cue_main_keep_fraction),
            "same_mask_used_across_memory_conditions": True,
            "main_panel_claim": "Main Fig.3F; peak/valley/random matched foreground masks test whether high-support peak regions provide stronger memory-dependent weak-cue recovery.",
        },
        ctx.config_dir / "structural_weak_cue_spec.json",
    )


def _write_summary(ctx: ExperimentContext) -> dict[str, Any]:
    required_main: list[Path] = []
    if ctx.cfg.run_progressive_update:
        required_main.append(ctx.metrics_dir / "panel_b_progressive_update_metrics.csv")
    if ctx.cfg.run_peak_valley_landscape:
        required_main.append(ctx.metrics_dir / "panel_c_example_landscape_summary.csv")
    if ctx.cfg.run_neutral_ping:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_d_ping_position_distribution.csv",
                ctx.metrics_dir / "panel_d_ping_summary.csv",
            ]
        )
    if ctx.cfg.run_weak_probe:
        required_main.extend(
            [
                ctx.raw_dir / "panel_e_weak_probe_trial_readout.csv",
                ctx.metrics_dir / "panel_e_weak_probe_metrics.csv",
                ctx.metrics_dir / "panel_e_weak_probe_auc_metrics.csv",
                ctx.metrics_dir / "panel_e_weak_probe_memory_gain.csv",
                ctx.metrics_dir / "panel_e_weak_probe_position_stratified_metrics.csv",
            ]
        )
    if ctx.cfg.run_region_ping:
        required_main.extend(
            [
                ctx.raw_dir / "panel_f_region_ping_trial_readout.csv",
                ctx.metrics_dir / "panel_f_region_ping_position_distribution.csv",
                ctx.metrics_dir / "panel_f_region_ping_summary.csv",
                ctx.metrics_dir / "panel_f_region_ping_contrast.csv",
                ctx.metrics_dir / "panel_f_region_ping_current_matching.csv",
            ]
        )
    if ctx.cfg.run_peak_cue_main:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_f_peak_cue_accuracy.csv",
                ctx.metrics_dir / "panel_f_peak_cue_memory_gain.csv",
                ctx.metrics_dir / "panel_f_peak_cue_matching_diagnostics.csv",
            ]
        )
    required_supp: list[Path] = []
    if ctx.cfg.run_supplement or ctx.cfg.run_population_morphology_supplement:
        required_supp.extend(
            [
                ctx.metrics_dir / "supp_peak_valley_contrast.csv",
                ctx.metrics_dir / "supp_landscape_nonflatness.csv",
                ctx.metrics_dir / "supp_peak_valley_prevalence.csv",
                ctx.metrics_dir / "supp_network_peak_valley_summary.csv",
                ctx.metrics_dir / "supp_anchor_dynamics_metrics.csv",
                ctx.metrics_dir / "supp_recency_only_controls.csv",
            ]
        )
    if ctx.cfg.run_supplement or ctx.cfg.run_neutral_ping:
        required_supp.append(ctx.metrics_dir / "supp_ping_recency_diagnostics.csv")
    if ctx.cfg.run_supplement or ctx.cfg.run_weak_probe:
        required_supp.extend(
            [
                ctx.metrics_dir / "supp_weak_probe_target_source_control.csv",
                ctx.metrics_dir / "supp_weak_probe_target_source_gain.csv",
            ]
        )
    if ctx.cfg.run_supplement or ctx.cfg.run_peak_cue_main or ctx.cfg.run_structural_weak_cue_supplement:
        required_supp.extend(
            [
                ctx.metrics_dir / "supp_structural_weak_cue_accuracy.csv",
                ctx.metrics_dir / "supp_structural_weak_cue_memory_gain.csv",
                ctx.metrics_dir / "supp_structural_weak_cue_matching_diagnostics.csv",
                ctx.metrics_dir / "supp_peak_cue_serial_position_metrics.csv",
                ctx.metrics_dir / "supp_peak_cue_serial_position_gain.csv",
            ]
        )
    if ctx.cfg.run_region_ping_amp_sweep:
        required_supp.extend(
            [
                ctx.raw_dir / "supp_region_ping_amp_sweep_trial_readout.csv",
                ctx.metrics_dir / "supp_region_ping_amp_sweep_summary.csv",
                ctx.metrics_dir / "supp_region_ping_amp_sweep_latency.csv",
            ]
        )
    panel_f_optional = [
        "cue_fraction_actual",
        "cue_energy",
        "encoded_spike_count",
        "support_mean_selected",
        "support_mean_foreground",
        "support_quantile_mean",
    ]
    panel_f_raw_path = ctx.raw_dir / "panel_f_peak_cue_trial_readout.csv"
    missing_panel_f_optional = _missing_csv_columns(panel_f_raw_path, panel_f_optional)
    target_control_path = ctx.metrics_dir / "supp_weak_probe_target_source_control.csv"
    unseen_target_control_available = False
    if target_control_path.exists():
        try:
            target_control = pd.read_csv(target_control_path)
            unseen_target_control_available = bool("target_source" in target_control.columns and "unseen_random" in set(target_control["target_source"].astype(str)))
        except Exception:
            unseen_target_control_available = False
    region_matching = _read_csv_if_exists(ctx.metrics_dir / "panel_f_region_ping_current_matching.csv")
    region_contrast = _read_csv_if_exists(ctx.metrics_dir / "panel_f_region_ping_contrast.csv")
    region_raw = _read_csv_if_exists(ctx.raw_dir / "panel_f_region_ping_trial_readout.csv")
    weak_metrics = _read_csv_if_exists(ctx.metrics_dir / "panel_e_weak_probe_metrics.csv")
    region_current_status = _region_ping_current_matching_status(region_matching)
    legacy_peak_available = bool((ctx.metrics_dir / "panel_f_peak_cue_memory_gain.csv").exists())
    region_available = bool((ctx.metrics_dir / "panel_f_region_ping_summary.csv").exists())
    summary = {
        "figure": FIGURE_ID,
        "network_seed": int(ctx.cfg.network_seed),
        "run_mode": SINGLE_NETWORK_MODE,
        "fig3_design_version": FIG3_DESIGN_VERSION,
        "main_panels": {
            "A": "sequence protocol",
            "B": "progressive update / update weakening",
            "C": "peak-structured final STSP landscape",
            "D": "neutral ping serial-position readout",
            "E": "singleton-matched sequence-state weak-probe completion",
            "F": "STSP-region-gated ping serial-position readout",
        },
        "demoted_to_supplement": ["population peak-valley prevalence", "ping recency diagnostics", "target-source and serial-position controls"],
        "supplement_plan": {
            "S5": "morphology and anchor controls",
            "S6": "functional controls for multi-item and peak-guided access",
        },
        "weak_cue_mask_mode": str(ctx.cfg.weak_cue_mask_mode),
        "weak_cue_target_source_main": "sequence_member_random",
        "panel_d_restore_mode": "reset_all_state_restore_selected_stsp",
        "panel_d_stsp_only_restore": True,
        "panel_e_weak_probe_mask_space": str(ctx.cfg.weak_probe_mask_space),
        "panel_e_weak_probe_scale": float(ctx.cfg.weak_probe_scale),
        "panel_e_weak_probe_metric_mode": str(ctx.cfg.weak_probe_metric_mode),
        "panel_e_fig2F_compatible": bool(ctx.cfg.weak_probe_mask_space == "encoded_spikes" and ctx.cfg.weak_probe_metric_mode == "fig2_compat"),
        "fig2F_weak_probe_compatible": bool(ctx.cfg.weak_probe_mask_space == "encoded_spikes" and ctx.cfg.weak_probe_metric_mode == "fig2_compat"),
        "structural_weak_cue_in_main": False,
        "peak_cue_main_keep_fraction": float(ctx.cfg.peak_cue_main_keep_fraction),
        "panel_e_general_weak_probe_available": bool(ctx.completed_modules.get("weak_probe", False)),
        "fig3e_singleton_weak_probe": {
            "enabled": bool(ctx.cfg.run_weak_probe),
            "has_single_item_memory": bool("memory_condition" in weak_metrics.columns and "single_item_memory" in set(weak_metrics.get("memory_condition", pd.Series(dtype=str)).astype(str))),
            "n_memory_conditions": int(weak_metrics["memory_condition"].nunique()) if "memory_condition" in weak_metrics.columns else 0,
            "memory_conditions": sorted(weak_metrics["memory_condition"].dropna().astype(str).unique().tolist()) if "memory_condition" in weak_metrics.columns else [],
            "position_stratified_available": bool((ctx.metrics_dir / "panel_e_weak_probe_position_stratified_metrics.csv").exists()),
        },
        "fig3f_region_ping": {
            "enabled": bool(ctx.cfg.run_region_ping),
            "support_metric": str(ctx.cfg.region_ping_support_metric),
            "region_q": float(ctx.cfg.region_ping_q),
            "region_conditions": list(ctx.cfg.region_ping_conditions),
            "n_region_conditions": len(tuple(ctx.cfg.region_ping_conditions)),
            "s0_control_available": bool("state_condition" in region_raw.columns and "S0" in set(region_raw.get("state_condition", pd.Series(dtype=str)).astype(str))),
            "amp_sweep_available": bool((ctx.metrics_dir / "supp_region_ping_amp_sweep_summary.csv").exists()),
            "current_matching_status": region_current_status,
            "JS_peak_valley": _first_float(region_contrast, "JS_peak_valley"),
            "TV_peak_valley": _first_float(region_contrast, "TV_peak_valley"),
            "P_peak_label_differs_from_valley": _first_float(region_contrast, "P_peak_label_differs_from_valley"),
            "ambiguous_label_count": int(pd.to_numeric(region_raw.get("label_is_ambiguous", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not region_raw.empty else 0,
        },
        "legacy_peak_cue_outputs_available": legacy_peak_available,
        "region_ping_outputs_available": region_available,
        "main_fig3f_source": "region_ping" if bool(ctx.cfg.run_region_ping) else "legacy_or_missing",
        "panel_f_peak_cue_available": bool(ctx.completed_modules.get("peak_cue_main", False)),
        "panel_f_peak_cue_missing_optional_fields": missing_panel_f_optional,
        "unseen_target_control_available": unseen_target_control_available,
        "structural_weak_cue_supplement_available": bool((ctx.metrics_dir / "supp_structural_weak_cue_accuracy.csv").exists()),
        "keep_fraction_sweep_is_real_rollout": True,
        "fixed_K_minus_1_target_used_for_main": False,
        "smoke": bool(ctx.cfg.smoke),
        "completed_modules": ctx.completed_modules,
        "output_files": ctx.output_files,
        "n_sequences": int(ctx.n_sequences),
        "sequence_lengths": list(ctx.cfg.sequence_lengths),
        "state_conditions": ["S0", "S_1..S_K", "S_final"],
        "cue_conditions": list(CUE_CONDITIONS),
        "memory_conditions": list(MEMORY_CONDITIONS),
        "mask_definition": {"peak_q": float(ctx.cfg.peak_q), "valley_q": float(ctx.cfg.valley_q), "pre_cue": True, "mode": str(ctx.cfg.weak_cue_mask_mode)},
        "warnings": ctx.warnings,
        "main_claim_supported_fields_available": all(path.exists() for path in required_main),
        "missing_for_main_figure": [_rel(path, ctx.seed_dir) for path in required_main if not path.exists()],
        "missing_for_supplementary": [_rel(path, ctx.seed_dir) for path in required_supp if not path.exists()],
    }
    _write_json(summary, ctx.seed_dir / "summary.json")
    ctx.output_files["summary"] = "summary.json"
    return summary


def _pairwise_image_sims(dataset, image_ids: Sequence[int]) -> list[float]:
    flats = [_image_flat(dataset, idx) for idx in image_ids]
    sims = []
    for i in range(len(flats)):
        for j in range(i + 1, len(flats)):
            sims.append(_centered_cosine(flats[i], flats[j]))
    return sims


def _image_flat(dataset, image_id: int) -> np.ndarray:
    return dataset[int(image_id)][0].detach().cpu().to(torch.float32).reshape(-1).numpy().astype(np.float64, copy=False)


def _images_for_ids(dataset, image_ids: Iterable[int]) -> torch.Tensor:
    return torch.stack([dataset[int(idx)][0].detach().to(torch.float32) for idx in image_ids], dim=0)


def _encode_cached(ctx: ExperimentContext, image_ids: Iterable[int], steps: int, *, cache: dict[tuple[Any, ...], torch.Tensor]) -> torch.Tensor:
    ids = tuple(int(v) for v in image_ids)
    key = (ids, int(steps), str(ctx.device))
    if (not ctx.cfg.use_encode_cache) or key not in cache:
        images = _images_for_ids(ctx.dataset, ids).to(ctx.device)
        spikes = encode_images(ctx.encoder, images, int(steps))
        if not ctx.cfg.use_encode_cache:
            return spikes
        cache[key] = spikes
    return cache[key]


def _masked_image(dataset, image_id: int, mask: np.ndarray) -> torch.Tensor:
    image = dataset[int(image_id)][0].detach().to(torch.float32).clone()
    mask_t = torch.as_tensor(mask.astype(np.float32), dtype=image.dtype)
    return image * mask_t.unsqueeze(0)


def _encoded_spike_count(ctx: ExperimentContext, image: torch.Tensor) -> float:
    spikes = encode_images(ctx.encoder, image.unsqueeze(0).to(ctx.device), ctx.cfg.weak_probe_steps)
    return float(spikes.detach().to(torch.float32).sum().item())


def _encode_image_tensor_cached(
    ctx: ExperimentContext,
    image: torch.Tensor,
    steps: int,
    *,
    cache: dict[tuple[Any, ...], torch.Tensor],
    cache_key: tuple[Any, ...],
) -> torch.Tensor:
    key = tuple(cache_key) + (int(steps), str(ctx.device))
    if (not ctx.cfg.use_encode_cache) or key not in cache:
        spikes = encode_images(ctx.encoder, image.unsqueeze(0).to(ctx.device), int(steps))
        if not ctx.cfg.use_encode_cache:
            return spikes
        cache[key] = spikes
    return cache[key]


def _foreground_mask(dataset, image_id: int) -> np.ndarray:
    image = dataset[int(image_id)][0].detach().cpu().to(torch.float32).squeeze(0).numpy()
    return image > 0.1


def _layer1_map(flat_g: np.ndarray) -> np.ndarray:
    arr = np.asarray(flat_g, dtype=np.float32).reshape(2, 28, 28)
    return arr.mean(axis=0)


def _top_mask(values: np.ndarray, q: float, *, positive: np.ndarray | None = None) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    valid = np.ones_like(vals, dtype=bool) if positive is None else positive.astype(bool)
    candidates = vals[valid]
    if candidates.size == 0:
        return np.zeros_like(vals, dtype=bool)
    k = max(1, int(round(float(q) * candidates.size)))
    thresh = np.partition(candidates.reshape(-1), -k)[-k]
    return valid & (vals >= thresh)


def _bottom_mask(values: np.ndarray, q: float) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    candidates = vals[finite]
    if candidates.size == 0:
        return np.zeros_like(vals, dtype=bool)
    k = max(1, int(round(float(q) * candidates.size)))
    thresh = np.partition(candidates.reshape(-1), k - 1)[k - 1]
    return finite & (vals <= thresh)


def _random_mask_like(reference: np.ndarray, pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = max(1, int(reference.sum()))
    choices = np.flatnonzero(pool.reshape(-1))
    if choices.size == 0:
        choices = np.arange(reference.size)
    selected = rng.choice(choices, size=min(count, choices.size), replace=choices.size < count)
    out = np.zeros(reference.size, dtype=bool)
    out[selected] = True
    return out.reshape(reference.shape)


def _trim_or_expand_mask(mask: np.ndarray, pool: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.flatnonzero(mask.reshape(-1))
    pool_flat = np.flatnonzero(pool.reshape(-1))
    if flat.size >= count:
        selected = rng.choice(flat, size=count, replace=False)
    else:
        extra_pool = np.setdiff1d(pool_flat, flat)
        need = max(0, count - flat.size)
        extra = rng.choice(extra_pool if extra_pool.size else pool_flat, size=need, replace=(extra_pool.size if extra_pool.size else pool_flat.size) < need)
        selected = np.concatenate([flat, extra])
    out = np.zeros(mask.size, dtype=bool)
    out[selected] = True
    return out.reshape(mask.shape)


def _target_position(seq_len: int, target_position: str) -> int:
    if str(target_position).upper() == "K-1":
        return max(1, int(seq_len) - 1)
    return max(1, min(int(seq_len), int(target_position)))


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - _centered_cosine(a, b))


def _centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), 1e-12))


def _gini(values: np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or np.sum(arr) <= 1e-12:
        return 0.0
    idx = np.arange(1, arr.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * arr) / (arr.size * np.sum(arr))) - ((arr.size + 1.0) / arr.size))


def _trial_condition_audit(network_seed: int, trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seq_meta = trials.drop_duplicates("sequence_id")
    for seq_len, count in seq_meta["seq_len"].value_counts().sort_index().items():
        rows.append({"network_seed": int(network_seed), "audit_type": "seq_len_count", "label": int(seq_len), "count": int(count), "value": float(count)})
    for label, count in trials["item_label"].value_counts().sort_index().items():
        rows.append({"network_seed": int(network_seed), "audit_type": "item_label_count", "label": int(label), "count": int(count), "value": float(count)})
    return pd.DataFrame(rows)


def _write_run_log(ctx: ExperimentContext) -> None:
    ctx.run_log.append(f"{_now()} completed modules={sorted(k for k, v in ctx.completed_modules.items() if v)}")
    path = ctx.seed_dir / "run_log.txt"
    path.write_text("\n".join(ctx.run_log) + "\n", encoding="utf-8")
    ctx.output_files["run_log"] = "run_log.txt"


def _save_csv(ctx: ExperimentContext, df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    ctx.output_files[path.stem] = _rel(path, ctx.seed_dir)


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
    run_region_ping = run_all or bool(args.run_region_ping) or bool(args.run_region_ping_amp_sweep)
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
    parser.add_argument("--region-ping-q", type=float, default=0.10)
    parser.add_argument("--region-ping-support-metric", default="delta_gain_map", choices=["delta_gain_map", "G_final"])
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
    parser.add_argument("--partial-cue-keep-fraction", type=float, default=0.10)
    parser.add_argument("--partial-cue-keep-fraction-sweep", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--partial-cue-repeats", type=int, default=20)
    parser.add_argument("--target-position", default="K-1")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
