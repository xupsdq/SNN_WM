from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.paths import DEFAULT_PATH_CONFIG
from src.config.units import ms
from src.experiments.common.dataset import build_class_index, build_dataset_arrays, encode_images
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import build_layer_input_shapes
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_run_config, save_summary_json
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.seed import mix_seed
from src.plotting.common.io import apply_publication_style, save_figure_all_formats, save_tidy_csv
from src.plotting.common.style import DYNAMIC_COLOR, NOISE_COLOR, SAMPLE_COLOR, SHUFFLE_COLOR


EXPERIMENT_ID = "chunk_stsp_layer2_peak_boost_attraction"
LAYER_KEYS: tuple[str, ...] = ("layer1", "layer2", "layer3")
SMALL_EPS = 1e-12


@dataclass(frozen=True)
class ExperimentConfig:
    model_path: str
    dataset_root: str
    split: str
    device: str
    seed: int
    output_dir: str
    sample_ms: float
    delay_ms: float
    batch_size: int
    max_sequences: int
    sequence_lengths: tuple[int, ...]
    samples_per_label: int
    intervention_stages: tuple[int, ...]
    epsilon: float
    peak_q: float
    candidate_pool_size: int
    selected_targets_per_trial: int
    target_overlap_mode: str
    boost_levels: tuple[float, ...]
    include_intact: bool
    include_nonpeak_boost_control: bool
    include_shuffle_boost_control: bool
    skip_figures: bool
    smoke: bool
    nonbase_outside_strategy: str = "preserve_prefix"
    boost_state_strategy: str = "intact_prefix_peak_only_no_nonbase_flatten"
    boost_formula: str = "target_u_x = intact_u_x + lambda * (intact_u_x - nonbase_mean_u_x)"
    projection_score: str = "abs_weighted_conv"
    projection_peak_rule: str = "top_peak_q_positive_peak_support_within_nonbase_supported"
    projection_support_threshold: float = 0.0
    dt: float = 1.0 * ms

    @property
    def sample_steps(self) -> int:
        return ms_to_steps(self.sample_ms, self.dt)

    @property
    def delay_steps(self) -> int:
        return ms_to_steps(self.delay_ms, self.dt)

    @property
    def max_duration_ms(self) -> float:
        return max(float(self.sample_ms), 100.0)


@dataclass(frozen=True)
class SequenceTrial:
    trial_id: int
    trial_index_within_seq_len: int
    seq_len: int
    ordered_item_ids: tuple[int, ...]
    ordered_item_labels: tuple[int, ...]
    sequence_seed: int


@dataclass(frozen=True)
class SequenceBatch:
    batch_id: int
    seq_len: int
    trials: tuple[SequenceTrial, ...]
    item_spikes: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class LayerRuntimeState:
    v_mem: torch.Tensor
    g_e: torch.Tensor
    res: torch.Tensor
    inh_trace: torch.Tensor
    u_pre: torch.Tensor | None
    x_pre: torch.Tensor | None
    pre_trace: torch.Tensor | None
    input_trace: torch.Tensor | None
    eligibility_trace: torch.Tensor | None
    firing_times: torch.Tensor | None


@dataclass(frozen=True)
class BoundarySnapshot:
    stage_k: int
    current_time: int
    layer_input_shapes: dict[str, tuple[int, ...]]
    full_state_by_layer: dict[str, LayerRuntimeState]
    layer2_u: np.ndarray
    layer2_x: np.ndarray
    layer2_g: np.ndarray


@dataclass(frozen=True)
class InputRegionMasks:
    nonbase: np.ndarray
    peak: np.ndarray
    nonpeak: np.ndarray
    valid: np.ndarray
    invalid_reason: tuple[str, ...]
    num_nonbase: np.ndarray
    num_peak: np.ndarray
    num_nonpeak: np.ndarray


@dataclass(frozen=True)
class OutputRegionMasks:
    mask_mode: str
    nonbase: np.ndarray
    peak: np.ndarray
    nonpeak: np.ndarray
    valid: np.ndarray
    invalid_reason: tuple[str, ...]
    num_nonbase: np.ndarray
    num_peak: np.ndarray
    num_nonpeak: np.ndarray
    projection_info: dict[str, Any]


def ms_to_steps(duration_ms: float, dt: float) -> int:
    return int(round((float(duration_ms) * ms) / float(dt)))


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layer 2 STSP peak boost attraction experiment.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_PATH_CONFIG.model_path))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_PATH_CONFIG.results_root / EXPERIMENT_ID))
    parser.add_argument("--sample-ms", type=float, default=180.0)
    parser.add_argument("--delay-ms", type=float, default=200.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[10])
    parser.add_argument("--samples-per-label", type=int, default=200)
    parser.add_argument("--intervention-stages", type=int, nargs="+", default=[5])
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--peak-q", type=float, default=0.20)
    parser.add_argument("--candidate-pool-size", type=int, default=40)
    parser.add_argument("--selected-targets-per-trial", type=int, default=1)
    parser.add_argument("--target-overlap-mode", type=str, default="low", choices=["low", "high", "both"])
    parser.add_argument("--boost-levels", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0])
    parser.add_argument("--include-intact", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--include-nonpeak-boost-control", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--include-shuffle-boost-control", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def normalize_config(args: argparse.Namespace) -> ExperimentConfig:
    sequence_lengths = tuple(dict.fromkeys(int(item) for item in args.sequence_lengths))
    intervention_stages = tuple(dict.fromkeys(int(item) for item in args.intervention_stages))
    boost_levels = tuple(dict.fromkeys(float(item) for item in args.boost_levels))
    if not sequence_lengths or not intervention_stages:
        raise ValueError("sequence-lengths and intervention-stages must not be empty.")
    cfg = ExperimentConfig(
        model_path=str(args.model_path),
        dataset_root=str(args.dataset_root),
        split=str(args.split),
        device=str(args.device),
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        sample_ms=float(args.sample_ms),
        delay_ms=float(args.delay_ms),
        batch_size=int(args.batch_size),
        max_sequences=int(args.max_sequences),
        sequence_lengths=sequence_lengths,
        samples_per_label=int(args.samples_per_label),
        intervention_stages=intervention_stages,
        epsilon=float(args.epsilon),
        peak_q=float(args.peak_q),
        candidate_pool_size=int(args.candidate_pool_size),
        selected_targets_per_trial=int(args.selected_targets_per_trial),
        target_overlap_mode=str(args.target_overlap_mode),
        boost_levels=boost_levels,
        include_intact=bool(args.include_intact),
        include_nonpeak_boost_control=bool(args.include_nonpeak_boost_control),
        include_shuffle_boost_control=bool(args.include_shuffle_boost_control),
        skip_figures=bool(args.skip_figures),
        smoke=bool(args.smoke),
    )
    if cfg.smoke:
        cfg = ExperimentConfig(
            **{
                **asdict(cfg),
                "output_dir": str(Path(cfg.output_dir) / "smoke"),
                "batch_size": min(int(cfg.batch_size), 2),
                "max_sequences": min(int(cfg.max_sequences), 4),
                "sequence_lengths": (6,),
                "intervention_stages": (5,),
                "samples_per_label": min(int(cfg.samples_per_label), 8),
                "sample_ms": min(float(cfg.sample_ms), 15.0),
                "delay_ms": min(float(cfg.delay_ms), 10.0),
                "candidate_pool_size": min(int(cfg.candidate_pool_size), 6),
                "selected_targets_per_trial": 1,
                "boost_levels": (0.0, 1.0, 2.0),
            }
        )
    if min(cfg.sequence_lengths) < 2 or max(cfg.sequence_lengths) > 10:
        raise ValueError("--sequence-lengths must be in [2, 10].")
    if min(cfg.intervention_stages) < 1:
        raise ValueError("--intervention-stages must be >= 1.")
    if not any(stage < seq_len for stage in cfg.intervention_stages for seq_len in cfg.sequence_lengths):
        raise ValueError("At least one intervention stage must be smaller than a sequence length.")
    if min(cfg.batch_size, cfg.max_sequences, cfg.candidate_pool_size, cfg.selected_targets_per_trial) <= 0:
        raise ValueError("batch, sequence, candidate, and selected-target counts must be positive.")
    if not (0.0 < cfg.peak_q < 1.0):
        raise ValueError("--peak-q must be in (0, 1).")
    if cfg.epsilon < 0.0:
        raise ValueError("--epsilon must be non-negative.")
    if min(cfg.sample_steps, cfg.delay_steps) <= 0:
        raise ValueError("--sample-ms and --delay-ms must map to positive step counts.")
    return cfg


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def log_and_print(lines: list[str], message: str) -> None:
    print(message, flush=True)
    lines.append(str(message))


def safe_mean(values: Sequence[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    return float(arr[valid].mean()) if np.count_nonzero(valid) else None


def safe_sem(values: Sequence[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    if np.count_nonzero(valid) <= 1:
        return None
    arr = arr[valid]
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def build_label_candidate_pools(class_index: Mapping[int, Sequence[int]], cfg: ExperimentConfig) -> dict[int, np.ndarray]:
    pools: dict[int, np.ndarray] = {}
    for label in sorted(class_index):
        rng = np.random.default_rng(mix_seed(cfg.seed, int(label), 991))
        ids = np.asarray([int(idx) for idx in class_index[int(label)]], dtype=np.int64)
        permuted = rng.permutation(ids)
        limit = len(permuted) if cfg.samples_per_label < 0 else min(int(cfg.samples_per_label), len(permuted))
        pool = permuted[:limit].astype(np.int64, copy=False)
        if pool.size <= 0:
            raise ValueError(f"Label {label} has no available images after pooling.")
        pools[int(label)] = pool
    return pools


def build_sequence_trials(
    flat_normalized: np.ndarray,
    class_index: Mapping[int, Sequence[int]],
    cfg: ExperimentConfig,
) -> tuple[list[SequenceTrial], pd.DataFrame]:
    label_pools = build_label_candidate_pools(class_index, cfg)
    all_labels = np.asarray(sorted(label_pools.keys()), dtype=np.int64)
    trials: list[SequenceTrial] = []
    rows: list[dict[str, object]] = []
    trial_id = 0
    for seq_len in cfg.sequence_lengths:
        if int(seq_len) > len(all_labels):
            raise ValueError(f"seq_len={seq_len} exceeds available label count={len(all_labels)}.")
        for within_len_idx in range(int(cfg.max_sequences)):
            trial_seed = mix_seed(cfg.seed, int(seq_len), int(within_len_idx), 907)
            rng = np.random.default_rng(trial_seed)
            chosen_labels = rng.choice(all_labels, size=int(seq_len), replace=False)
            chosen_item_ids = np.asarray([int(rng.choice(label_pools[int(label)])) for label in chosen_labels], dtype=np.int64)
            order = rng.permutation(int(seq_len))
            ordered_labels = chosen_labels[order].astype(np.int64, copy=False)
            ordered_ids = chosen_item_ids[order].astype(np.int64, copy=False)
            trial = SequenceTrial(
                trial_id=int(trial_id),
                trial_index_within_seq_len=int(within_len_idx),
                seq_len=int(seq_len),
                ordered_item_ids=tuple(int(item) for item in ordered_ids.tolist()),
                ordered_item_labels=tuple(int(item) for item in ordered_labels.tolist()),
                sequence_seed=int(trial_seed),
            )
            trials.append(trial)
            ordered_ids_str = "|".join(str(int(item)) for item in trial.ordered_item_ids)
            ordered_labels_str = "|".join(str(int(item)) for item in trial.ordered_item_labels)
            sim = flat_normalized[ordered_ids] @ flat_normalized[ordered_ids].T
            pair_vals = sim[~np.eye(int(seq_len), dtype=bool)]
            for item_index, (image_id, item_label) in enumerate(zip(trial.ordered_item_ids, trial.ordered_item_labels), start=1):
                rows.append(
                    {
                        "trial_id": int(trial.trial_id),
                        "trial_index_within_seq_len": int(trial.trial_index_within_seq_len),
                        "seq_len": int(trial.seq_len),
                        "item_index": int(item_index),
                        "image_id": int(image_id),
                        "item_label": int(item_label),
                        "ordered_item_ids": ordered_ids_str,
                        "ordered_item_labels": ordered_labels_str,
                        "sequence_seed": int(trial.sequence_seed),
                        "mean_pairwise_image_similarity": float(pair_vals.mean()) if pair_vals.size else 1.0,
                    }
                )
            trial_id += 1
    return trials, pd.DataFrame(rows)


def build_candidate_targets(
    trials: Sequence[SequenceTrial],
    labels: np.ndarray,
    cfg: ExperimentConfig,
) -> dict[tuple[int, int], tuple[int, ...]]:
    all_ids = np.arange(int(labels.shape[0]), dtype=np.int64)
    out: dict[tuple[int, int], tuple[int, ...]] = {}
    for trial in trials:
        for stage in cfg.intervention_stages:
            if int(stage) >= int(trial.seq_len):
                continue
            excluded = set(int(item) for item in trial.ordered_item_ids[: int(stage)])
            available = np.asarray([int(idx) for idx in all_ids if int(idx) not in excluded], dtype=np.int64)
            rng = np.random.default_rng(mix_seed(cfg.seed, int(trial.trial_id), int(stage), 4021))
            chosen = rng.choice(available, size=min(int(cfg.candidate_pool_size), int(available.size)), replace=False)
            out[(int(trial.trial_id), int(stage))] = tuple(int(item) for item in chosen.tolist())
    return out


def build_spike_lookup(
    images: torch.Tensor,
    encoder: Any,
    image_ids: Iterable[int],
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    unique_ids = sorted({int(image_id) for image_id in image_ids})
    if not unique_ids:
        return {}
    spike_bank = encode_images(
        encoder,
        images[unique_ids].to(device=device, dtype=torch.float32),
        steps=int(cfg.sample_steps),
    )
    return {int(image_id): spike_bank[idx] for idx, image_id in enumerate(unique_ids)}


def build_batches(trials: Sequence[SequenceTrial], spike_lookup: Mapping[int, torch.Tensor], cfg: ExperimentConfig) -> Iterator[SequenceBatch]:
    by_seq_len: dict[int, list[SequenceTrial]] = {}
    for trial in trials:
        by_seq_len.setdefault(int(trial.seq_len), []).append(trial)
    for seq_len in sorted(by_seq_len):
        trial_list = by_seq_len[int(seq_len)]
        for batch_id, start in enumerate(range(0, len(trial_list), int(cfg.batch_size))):
            batch_trials = tuple(trial_list[start : start + int(cfg.batch_size)])
            item_spikes = tuple(
                torch.stack([spike_lookup[int(trial.ordered_item_ids[item_pos])] for trial in batch_trials], dim=0)
                for item_pos in range(int(seq_len))
            )
            yield SequenceBatch(batch_id=int(batch_id), seq_len=int(seq_len), trials=batch_trials, item_spikes=item_spikes)


def prepare_clean_network_state(net: Any, batch_size: int, channels: int, height: int, width: int) -> dict[str, tuple[int, ...]]:
    layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            getattr(net, layer_key).reset_state(layer_input_shapes[layer_key])
    return {str(key): tuple(value) for key, value in layer_input_shapes.items()}


def forward_three_layers_capture_layer2(net: Any, input_t: torch.Tensor, t_step: int) -> torch.Tensor:
    s1, _ = net.layer1.forward_step(input_t, t_step, training=False, monitor=False, stsp_mode="dynamic")
    s1_p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1_p, t_step, training=False, monitor=False, stsp_mode="dynamic")
    s2_p = net.pool2(s2.float())
    net.layer3.forward_step(s2_p, t_step, training=False, monitor=False, stsp_mode="dynamic")
    return s2


def forward_three_layers(net: Any, input_t: torch.Tensor, t_step: int) -> None:
    forward_three_layers_capture_layer2(net, input_t, t_step)


def _clone_optional_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().cpu().clone()


def clone_layer_runtime_state(state: LayerRuntimeState) -> LayerRuntimeState:
    return LayerRuntimeState(
        v_mem=state.v_mem.clone(),
        g_e=state.g_e.clone(),
        res=state.res.clone(),
        inh_trace=state.inh_trace.clone(),
        u_pre=None if state.u_pre is None else state.u_pre.clone(),
        x_pre=None if state.x_pre is None else state.x_pre.clone(),
        pre_trace=None if state.pre_trace is None else state.pre_trace.clone(),
        input_trace=None if state.input_trace is None else state.input_trace.clone(),
        eligibility_trace=None if state.eligibility_trace is None else state.eligibility_trace.clone(),
        firing_times=None if state.firing_times is None else state.firing_times.clone(),
    )


def capture_layer_runtime_state(layer: Any) -> LayerRuntimeState:
    return LayerRuntimeState(
        v_mem=layer.v_mem.detach().cpu().clone(),
        g_e=layer.g_e.detach().cpu().clone(),
        res=layer.res.detach().cpu().clone(),
        inh_trace=layer.lateral_inh.inh_trace.detach().cpu().clone(),
        u_pre=_clone_optional_tensor(getattr(layer, "u_pre", None)),
        x_pre=_clone_optional_tensor(getattr(layer, "x_pre", None)),
        pre_trace=_clone_optional_tensor(getattr(layer, "pre_trace", None)),
        input_trace=_clone_optional_tensor(getattr(layer, "input_trace", None)),
        eligibility_trace=_clone_optional_tensor(getattr(layer, "eligibility_trace", None)),
        firing_times=_clone_optional_tensor(getattr(layer, "firing_times", None)),
    )


def snapshot_boundary_state(
    net: Any,
    *,
    batch_size: int,
    stage_k: int,
    current_time: int,
    layer_input_shapes: Mapping[str, tuple[int, ...]],
) -> BoundarySnapshot:
    full_state = {str(layer_key): capture_layer_runtime_state(getattr(net, layer_key)) for layer_key in LAYER_KEYS}
    layer2 = net.layer2
    if layer2.u_pre is None or layer2.x_pre is None:
        raise ValueError("layer2 is missing STSP state.")
    u = layer2.u_pre.detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=True)
    x = layer2.x_pre.detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=True)
    g = (layer2.u_pre * layer2.x_pre).detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=True)
    return BoundarySnapshot(
        stage_k=int(stage_k),
        current_time=int(current_time),
        layer_input_shapes={str(key): tuple(value) for key, value in layer_input_shapes.items()},
        full_state_by_layer=full_state,
        layer2_u=u,
        layer2_x=x,
        layer2_g=g,
    )


def clone_boundary_snapshot(snapshot: BoundarySnapshot) -> BoundarySnapshot:
    return BoundarySnapshot(
        stage_k=int(snapshot.stage_k),
        current_time=int(snapshot.current_time),
        layer_input_shapes={str(key): tuple(value) for key, value in snapshot.layer_input_shapes.items()},
        full_state_by_layer={str(key): clone_layer_runtime_state(value) for key, value in snapshot.full_state_by_layer.items()},
        layer2_u=np.asarray(snapshot.layer2_u, dtype=np.float32).copy(),
        layer2_x=np.asarray(snapshot.layer2_x, dtype=np.float32).copy(),
        layer2_g=np.asarray(snapshot.layer2_g, dtype=np.float32).copy(),
    )


def _copy_tensor_in_place(target: torch.Tensor, source: torch.Tensor) -> None:
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def restore_network_boundary_state(net: Any, snapshot: BoundarySnapshot) -> None:
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            getattr(net, layer_key).reset_state(snapshot.layer_input_shapes[layer_key])
        for layer_key, layer_state in snapshot.full_state_by_layer.items():
            layer = getattr(net, layer_key)
            _copy_tensor_in_place(layer.v_mem, layer_state.v_mem)
            _copy_tensor_in_place(layer.g_e, layer_state.g_e)
            _copy_tensor_in_place(layer.res, layer_state.res)
            _copy_tensor_in_place(layer.lateral_inh.inh_trace, layer_state.inh_trace)
            if layer_state.u_pre is not None and getattr(layer, "u_pre", None) is not None:
                _copy_tensor_in_place(layer.u_pre, layer_state.u_pre)
            if layer_state.x_pre is not None and getattr(layer, "x_pre", None) is not None:
                _copy_tensor_in_place(layer.x_pre, layer_state.x_pre)
            if layer_state.pre_trace is not None and getattr(layer, "pre_trace", None) is not None:
                _copy_tensor_in_place(layer.pre_trace, layer_state.pre_trace)
            if layer_state.input_trace is not None and getattr(layer, "input_trace", None) is not None:
                _copy_tensor_in_place(layer.input_trace, layer_state.input_trace)
            if layer_state.eligibility_trace is not None and getattr(layer, "eligibility_trace", None) is not None:
                _copy_tensor_in_place(layer.eligibility_trace, layer_state.eligibility_trace)
            if layer_state.firing_times is not None and getattr(layer, "firing_times", None) is not None:
                _copy_tensor_in_place(layer.firing_times, layer_state.firing_times)


def capture_prefix_snapshots(net: Any, batch: SequenceBatch, cfg: ExperimentConfig) -> tuple[BoundarySnapshot, dict[int, BoundarySnapshot]]:
    first_sequence = batch.item_spikes[0]
    batch_size, _, channels, height, width = first_sequence.shape
    requested = {int(stage) for stage in cfg.intervention_stages if int(stage) < int(batch.seq_len)}
    max_stage = max(requested) if requested else 0
    snapshots: dict[int, BoundarySnapshot] = {}
    with torch.no_grad():
        layer_input_shapes = prepare_clean_network_state(net, batch_size, channels, height, width)
        zero_input = torch.zeros((batch_size, channels, height, width), dtype=first_sequence.dtype, device=first_sequence.device)
        current_time = 0
        initial_snapshot = snapshot_boundary_state(
            net,
            batch_size=batch_size,
            stage_k=0,
            current_time=current_time,
            layer_input_shapes=layer_input_shapes,
        )
        for position in range(1, max_stage + 1):
            item_spikes = batch.item_spikes[position - 1]
            for step_idx in range(int(item_spikes.shape[1])):
                forward_three_layers(net, item_spikes[:, step_idx, ...], current_time)
                current_time += 1
            for _ in range(int(cfg.delay_steps)):
                forward_three_layers(net, zero_input, current_time)
                current_time += 1
            if position in requested:
                snapshots[int(position)] = snapshot_boundary_state(
                    net,
                    batch_size=batch_size,
                    stage_k=int(position),
                    current_time=current_time,
                    layer_input_shapes=layer_input_shapes,
                )
    return initial_snapshot, snapshots


def derive_input_region_masks(snapshot: BoundarySnapshot, *, baseline_gain: float, epsilon: float, peak_q: float) -> InputRegionMasks:
    g = np.asarray(snapshot.layer2_g, dtype=np.float64)
    delta = g - float(baseline_gain)
    nonbase = np.abs(delta) > float(epsilon)
    peak = np.zeros_like(nonbase, dtype=bool)
    nonpeak = np.zeros_like(nonbase, dtype=bool)
    valid = np.zeros(g.shape[0], dtype=bool)
    invalid_reason: list[str] = []
    num_nonbase = nonbase.sum(axis=1).astype(np.int64)
    num_peak = np.zeros(g.shape[0], dtype=np.int64)
    num_nonpeak = np.zeros(g.shape[0], dtype=np.int64)
    for row_idx in range(g.shape[0]):
        row_nonbase = nonbase[row_idx]
        count = int(row_nonbase.sum())
        if count <= 0:
            invalid_reason.append("empty_input_nonbase")
            continue
        peak_count = max(1, min(int(math.ceil(float(count) * float(peak_q))), count))
        nonbase_indices = np.flatnonzero(row_nonbase)
        ranked = nonbase_indices[np.argsort(delta[row_idx, nonbase_indices], kind="stable")]
        peak[row_idx, ranked[-peak_count:]] = True
        nonpeak[row_idx] = row_nonbase & ~peak[row_idx]
        num_peak[row_idx] = int(peak[row_idx].sum())
        num_nonpeak[row_idx] = int(nonpeak[row_idx].sum())
        if num_peak[row_idx] <= 0:
            invalid_reason.append("empty_input_peak")
        elif num_nonpeak[row_idx] <= 0:
            invalid_reason.append("empty_input_nonpeak")
        else:
            valid[row_idx] = True
            invalid_reason.append("")
    return InputRegionMasks(nonbase, peak, nonpeak, valid, tuple(invalid_reason), num_nonbase, num_peak, num_nonpeak)


def project_input_masks_to_layer2_outputs(net: Any, snapshot: BoundarySnapshot, masks: InputRegionMasks, cfg: ExperimentConfig) -> OutputRegionMasks:
    batch_size = int(snapshot.layer2_g.shape[0])
    input_shape = tuple(int(v) for v in snapshot.layer_input_shapes["layer2"][1:])
    output_shape = tuple(int(v) for v in snapshot.full_state_by_layer["layer2"].v_mem.shape[1:])
    input_flat = int(np.prod(input_shape))
    output_flat = int(np.prod(output_shape))
    if input_flat == output_flat and input_shape == output_shape:
        return OutputRegionMasks(
            mask_mode="direct",
            nonbase=masks.nonbase.copy(),
            peak=masks.peak.copy(),
            nonpeak=masks.nonpeak.copy(),
            valid=masks.valid.copy(),
            invalid_reason=masks.invalid_reason,
            num_nonbase=masks.num_nonbase.copy(),
            num_peak=masks.num_peak.copy(),
            num_nonpeak=masks.num_nonpeak.copy(),
            projection_info={"mask_mode": "direct", "input_shape": input_shape, "output_shape": output_shape},
        )

    weight = net.layer2.kernels.detach().abs().cpu().to(torch.float32)
    nonbase_out = np.zeros((batch_size, output_flat), dtype=bool)
    peak_out = np.zeros((batch_size, output_flat), dtype=bool)
    nonpeak_out = np.zeros((batch_size, output_flat), dtype=bool)
    valid = np.zeros(batch_size, dtype=bool)
    reasons: list[str] = []
    num_nonbase = np.zeros(batch_size, dtype=np.int64)
    num_peak = np.zeros(batch_size, dtype=np.int64)
    num_nonpeak = np.zeros(batch_size, dtype=np.int64)
    for row_idx in range(batch_size):
        if not bool(masks.valid[row_idx]):
            reasons.append(masks.invalid_reason[row_idx])
            continue
        peak_tensor = torch.as_tensor(masks.peak[row_idx].reshape((1, *input_shape)), dtype=torch.float32)
        nonbase_tensor = torch.as_tensor(masks.nonbase[row_idx].reshape((1, *input_shape)), dtype=torch.float32)
        peak_score = F.conv2d(peak_tensor, weight, stride=int(net.layer2.stride), padding=int(net.layer2.padding)).reshape(-1).numpy()
        nonbase_score = F.conv2d(nonbase_tensor, weight, stride=int(net.layer2.stride), padding=int(net.layer2.padding)).reshape(-1).numpy()
        supported = nonbase_score > float(cfg.projection_support_threshold)
        candidate = supported & (peak_score > float(cfg.projection_support_threshold))
        if int(supported.sum()) <= 0:
            reasons.append("empty_projected_nonbase")
            continue
        if int(candidate.sum()) <= 0:
            reasons.append("empty_projected_peak")
            continue
        peak_count = max(1, min(int(math.ceil(float(supported.sum()) * float(cfg.peak_q))), int(candidate.sum())))
        candidate_indices = np.flatnonzero(candidate)
        ranked = candidate_indices[np.argsort(peak_score[candidate_indices], kind="stable")]
        row_peak = np.zeros(output_flat, dtype=bool)
        row_peak[ranked[-peak_count:]] = True
        row_nonpeak = supported & ~row_peak
        if int(row_nonpeak.sum()) <= 0:
            reasons.append("empty_projected_nonpeak")
            continue
        nonbase_out[row_idx] = supported
        peak_out[row_idx] = row_peak
        nonpeak_out[row_idx] = row_nonpeak
        num_nonbase[row_idx] = int(supported.sum())
        num_peak[row_idx] = int(row_peak.sum())
        num_nonpeak[row_idx] = int(row_nonpeak.sum())
        valid[row_idx] = True
        reasons.append("")
    return OutputRegionMasks(
        mask_mode="projected",
        nonbase=nonbase_out,
        peak=peak_out,
        nonpeak=nonpeak_out,
        valid=valid,
        invalid_reason=tuple(reasons),
        num_nonbase=num_nonbase,
        num_peak=num_peak,
        num_nonpeak=num_nonpeak,
        projection_info={
            "mask_mode": "projected",
            "projection_score": cfg.projection_score,
            "projection_peak_rule": cfg.projection_peak_rule,
            "projection_support_threshold": float(cfg.projection_support_threshold),
            "input_shape": input_shape,
            "output_shape": output_shape,
            "layer2_kernel_shape": tuple(int(v) for v in net.layer2.kernels.shape),
            "layer2_stride": int(net.layer2.stride),
            "layer2_padding": int(net.layer2.padding),
        },
    )


def compute_natural_contrast_rows(batch: SequenceBatch, stage_k: int, snapshot: BoundarySnapshot, masks: InputRegionMasks) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    g = np.asarray(snapshot.layer2_g, dtype=np.float64)
    u = np.asarray(snapshot.layer2_u, dtype=np.float64)
    x = np.asarray(snapshot.layer2_x, dtype=np.float64)
    for row_idx, trial in enumerate(batch.trials):
        valid = bool(masks.valid[row_idx])
        if valid:
            peak = masks.peak[row_idx]
            nonpeak = masks.nonpeak[row_idx]
            nonbase = masks.nonbase[row_idx]
            g_peak = g[row_idx, peak]
            g_nonpeak = g[row_idx, nonpeak]
            pooled = math.sqrt(0.5 * (float(np.var(g_peak)) + float(np.var(g_nonpeak))) + SMALL_EPS)
            cohen_d = float((np.mean(g_peak) - np.mean(g_nonpeak)) / pooled)
            row = {
                "mean_g_peak": float(np.mean(g_peak)),
                "mean_g_nonpeak": float(np.mean(g_nonpeak)),
                "mean_g_nonbase": float(np.mean(g[row_idx, nonbase])),
                "peak_minus_nonpeak_g": float(np.mean(g_peak) - np.mean(g_nonpeak)),
                "peak_over_nonpeak_g": float(np.mean(g_peak) / (np.mean(g_nonpeak) + SMALL_EPS)),
                "cohen_d_peak_vs_nonpeak_g": cohen_d,
                "std_g_peak": float(np.std(g_peak)),
                "std_g_nonpeak": float(np.std(g_nonpeak)),
                "mean_u_peak": float(np.mean(u[row_idx, peak])),
                "mean_u_nonpeak": float(np.mean(u[row_idx, nonpeak])),
                "peak_minus_nonpeak_u": float(np.mean(u[row_idx, peak]) - np.mean(u[row_idx, nonpeak])),
                "mean_x_peak": float(np.mean(x[row_idx, peak])),
                "mean_x_nonpeak": float(np.mean(x[row_idx, nonpeak])),
                "peak_minus_nonpeak_x": float(np.mean(x[row_idx, peak]) - np.mean(x[row_idx, nonpeak])),
            }
        else:
            row = {
                "mean_g_peak": np.nan,
                "mean_g_nonpeak": np.nan,
                "mean_g_nonbase": np.nan,
                "peak_minus_nonpeak_g": np.nan,
                "peak_over_nonpeak_g": np.nan,
                "cohen_d_peak_vs_nonpeak_g": np.nan,
                "std_g_peak": np.nan,
                "std_g_nonpeak": np.nan,
                "mean_u_peak": np.nan,
                "mean_u_nonpeak": np.nan,
                "peak_minus_nonpeak_u": np.nan,
                "mean_x_peak": np.nan,
                "mean_x_nonpeak": np.nan,
                "peak_minus_nonpeak_x": np.nan,
            }
        rows.append(
            {
                "trial_id": int(trial.trial_id),
                "seq_len": int(trial.seq_len),
                "intervention_stage": int(stage_k),
                "num_nonbase": int(masks.num_nonbase[row_idx]),
                "num_peak": int(masks.num_peak[row_idx]),
                "num_nonpeak": int(masks.num_nonpeak[row_idx]),
                **row,
                "valid": int(valid),
                "invalid_reason": str(masks.invalid_reason[row_idx]),
            }
        )
    return rows


def run_target_and_capture_layer2_spikes(net: Any, start_snapshot: BoundarySnapshot, target_spikes: torch.Tensor) -> np.ndarray:
    restore_network_boundary_state(net, start_snapshot)
    current_time = int(start_snapshot.current_time)
    captured: list[torch.Tensor] = []
    with torch.no_grad():
        for step_idx in range(int(target_spikes.shape[1])):
            s2 = forward_three_layers_capture_layer2(net, target_spikes[:, step_idx, ...], current_time)
            captured.append(s2.detach().to(torch.float32).cpu())
            current_time += 1
    spike_tensor = torch.stack(captured, dim=0).sum(dim=0)
    return spike_tensor.view(spike_tensor.shape[0], -1).numpy().astype(np.float32, copy=False)


def spike_metrics_for_row(counts: np.ndarray, mask: OutputRegionMasks, row_idx: int) -> dict[str, float | int | str]:
    if not bool(mask.valid[row_idx]):
        return {
            "valid": 0,
            "invalid_reason": str(mask.invalid_reason[row_idx]),
            "total_spike_count": float(np.sum(counts)),
            "peak_spike_count": np.nan,
            "nonpeak_spike_count": np.nan,
            "peak_spike_fraction": np.nan,
            "peak_spike_density": np.nan,
            "nonpeak_spike_density": np.nan,
            "spike_enrichment": np.nan,
        }
    peak = mask.peak[row_idx]
    nonpeak = mask.nonpeak[row_idx]
    total = float(np.sum(counts))
    peak_count = float(np.sum(counts[peak]))
    nonpeak_count = float(np.sum(counts[nonpeak]))
    peak_density = float(peak_count / max(int(mask.num_peak[row_idx]), 1))
    nonpeak_density = float(nonpeak_count / max(int(mask.num_nonpeak[row_idx]), 1))
    return {
        "valid": 1,
        "invalid_reason": "",
        "total_spike_count": total,
        "peak_spike_count": peak_count,
        "nonpeak_spike_count": nonpeak_count,
        "peak_spike_fraction": float(peak_count / (total + SMALL_EPS)),
        "peak_spike_density": peak_density,
        "nonpeak_spike_density": nonpeak_density,
        "spike_enrichment": float(peak_density / (nonpeak_density + SMALL_EPS)),
    }


def select_targets_for_batch(
    net: Any,
    batch: SequenceBatch,
    stage_k: int,
    initial_snapshot: BoundarySnapshot,
    output_masks: OutputRegionMasks,
    candidate_by_trial_stage: Mapping[tuple[int, int], tuple[int, ...]],
    spike_lookup: Mapping[int, torch.Tensor],
    labels: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidate_rows: list[dict[str, object]] = []
    batch_size = len(batch.trials)
    max_candidates = min(int(cfg.candidate_pool_size), max(len(candidate_by_trial_stage[(int(t.trial_id), int(stage_k))]) for t in batch.trials))
    for cand_pos in range(max_candidates):
        target_spikes = torch.stack(
            [spike_lookup[int(candidate_by_trial_stage[(int(trial.trial_id), int(stage_k))][cand_pos])] for trial in batch.trials],
            dim=0,
        )
        spike_counts = run_target_and_capture_layer2_spikes(net, initial_snapshot, target_spikes)
        for row_idx, trial in enumerate(batch.trials):
            candidate_id = int(candidate_by_trial_stage[(int(trial.trial_id), int(stage_k))][cand_pos])
            metrics = spike_metrics_for_row(spike_counts[row_idx], output_masks, row_idx)
            candidate_rows.append(
                {
                    "trial_id": int(trial.trial_id),
                    "seq_len": int(trial.seq_len),
                    "intervention_stage": int(stage_k),
                    "candidate_image_id": candidate_id,
                    "candidate_label": int(labels[candidate_id]),
                    "candidate_pool_position": int(cand_pos),
                    "candidate_rank_by_overlap": -1,
                    "selected": 0,
                    "target_overlap_group": "unselected",
                    "overlap_fraction": float(metrics["peak_spike_fraction"]) if int(metrics["valid"]) == 1 else np.nan,
                    "overlap_enrichment": float(metrics["spike_enrichment"]) if int(metrics["valid"]) == 1 else np.nan,
                    "baseline_total_spike_count": float(metrics["total_spike_count"]),
                    "baseline_peak_spike_fraction": float(metrics["peak_spike_fraction"]) if int(metrics["valid"]) == 1 else np.nan,
                    "baseline_spike_enrichment": float(metrics["spike_enrichment"]) if int(metrics["valid"]) == 1 else np.nan,
                }
            )

    selected_rows: list[dict[str, object]] = []
    df = pd.DataFrame(candidate_rows)
    for trial_id, sub in df.groupby("trial_id", sort=True):
        sub_valid = sub[np.isfinite(sub["overlap_fraction"].to_numpy(dtype=np.float64))].copy()
        if sub_valid.empty:
            continue
        sub_valid = sub_valid.sort_values(["overlap_fraction", "candidate_image_id"], kind="stable").reset_index(drop=True)
        ranks = {int(row.candidate_image_id): int(idx + 1) for idx, row in sub_valid.iterrows()}
        for row_idx, row in df[df["trial_id"] == trial_id].iterrows():
            candidate_rows[int(row_idx)]["candidate_rank_by_overlap"] = ranks.get(int(row["candidate_image_id"]), -1)
        groups: list[tuple[str, pd.DataFrame]] = []
        if cfg.target_overlap_mode in {"low", "both"}:
            groups.append(("low_overlap", sub_valid.head(int(cfg.selected_targets_per_trial))))
        if cfg.target_overlap_mode in {"high", "both"}:
            groups.append(("high_overlap", sub_valid.tail(int(cfg.selected_targets_per_trial)).sort_values(["overlap_fraction"], ascending=False)))
        for group_name, chosen in groups:
            for selected_idx, (_, row) in enumerate(chosen.iterrows()):
                selected = row.to_dict()
                selected["selected_target_index"] = int(selected_idx)
                selected["target_overlap_group"] = group_name
                selected_rows.append(selected)
                for cand_row in candidate_rows:
                    if int(cand_row["trial_id"]) == int(trial_id) and int(cand_row["candidate_image_id"]) == int(row["candidate_image_id"]):
                        cand_row["selected"] = 1
                        cand_row["target_overlap_group"] = group_name
    return candidate_rows, selected_rows


def make_flatten_or_boost_snapshot(
    snapshot: BoundarySnapshot,
    masks: InputRegionMasks,
    row_idx: int,
    *,
    boost_level: float,
    boost_type: str,
    seed: int,
) -> tuple[BoundarySnapshot, dict[str, float]]:
    out = clone_boundary_snapshot(snapshot)
    state = out.full_state_by_layer["layer2"]
    if state.u_pre is None or state.x_pre is None:
        raise ValueError("layer2 state missing u/x.")
    u_new = state.u_pre.clone()
    x_new = state.x_pre.clone()
    flat_u = u_new.view(u_new.shape[0], -1)
    flat_x = x_new.view(x_new.shape[0], -1)
    nonbase_np = masks.nonbase[row_idx]
    peak_np = masks.peak[row_idx]
    nonpeak_np = masks.nonpeak[row_idx]
    if not nonbase_np.any():
        return out, {"u_clip_fraction_peak": np.nan, "x_clip_fraction_peak": np.nan, "any_clip_fraction_peak": np.nan}
    nonbase = torch.as_tensor(nonbase_np, dtype=torch.bool, device=flat_u.device)
    peak = torch.as_tensor(peak_np, dtype=torch.bool, device=flat_u.device)
    u_orig = flat_u[row_idx].clone()
    x_orig = flat_x[row_idx].clone()
    u_bar = flat_u[row_idx, nonbase].mean()
    x_bar = flat_x[row_idx, nonbase].mean()

    if boost_type == "flatten":
        flat_u[row_idx, nonbase] = u_bar
        flat_x[row_idx, nonbase] = x_bar
        out.full_state_by_layer["layer2"] = LayerRuntimeState(
            v_mem=state.v_mem,
            g_e=state.g_e,
            res=state.res,
            inh_trace=state.inh_trace,
            u_pre=u_new,
            x_pre=x_new,
            pre_trace=state.pre_trace,
            input_trace=state.input_trace,
            eligibility_trace=state.eligibility_trace,
            firing_times=state.firing_times,
        )
        flat_u_out = u_new.view(u_new.shape[0], -1)
        flat_x_out = x_new.view(x_new.shape[0], -1)
        out.layer2_u[:, :] = flat_u_out.cpu().numpy().astype(np.float32, copy=False)
        out.layer2_x[:, :] = flat_x_out.cpu().numpy().astype(np.float32, copy=False)
        out.layer2_g[:, :] = (flat_u_out * flat_x_out).cpu().numpy().astype(np.float32, copy=False)
        return out, {
            "u_clip_fraction_peak": 0.0,
            "x_clip_fraction_peak": 0.0,
            "any_clip_fraction_peak": 0.0,
        }

    target_mask = peak
    if boost_type == "nonpeak_boost":
        rng = np.random.default_rng(seed)
        nonpeak_indices = np.flatnonzero(nonpeak_np)
        peak_count = int(np.count_nonzero(peak_np))
        chosen = rng.choice(nonpeak_indices, size=min(peak_count, len(nonpeak_indices)), replace=False)
        target_np = np.zeros_like(peak_np, dtype=bool)
        target_np[chosen] = True
        target_mask = torch.as_tensor(target_np, dtype=torch.bool, device=flat_u.device)
    elif boost_type == "shuffled_boost":
        rng = np.random.default_rng(seed)
        nonbase_indices = np.flatnonzero(nonbase_np)
        peak_count = int(np.count_nonzero(peak_np))
        chosen = rng.choice(nonbase_indices, size=min(peak_count, len(nonbase_indices)), replace=False)
        target_np = np.zeros_like(peak_np, dtype=bool)
        target_np[chosen] = True
        target_mask = torch.as_tensor(target_np, dtype=torch.bool, device=flat_u.device)
    proposed_u = u_orig[target_mask] + float(boost_level) * (u_orig[target_mask] - u_bar)
    proposed_x = x_orig[target_mask] + float(boost_level) * (x_orig[target_mask] - x_bar)
    clipped_u = torch.clamp(proposed_u, 0.0, 1.0)
    clipped_x = torch.clamp(proposed_x, 0.0, 1.0)
    flat_u[row_idx, target_mask] = clipped_u
    flat_x[row_idx, target_mask] = clipped_x
    u_clip = (clipped_u != proposed_u)
    x_clip = (clipped_x != proposed_x)
    out.full_state_by_layer["layer2"] = LayerRuntimeState(
        v_mem=state.v_mem,
        g_e=state.g_e,
        res=state.res,
        inh_trace=state.inh_trace,
        u_pre=u_new,
        x_pre=x_new,
        pre_trace=state.pre_trace,
        input_trace=state.input_trace,
        eligibility_trace=state.eligibility_trace,
        firing_times=state.firing_times,
    )
    flat_u_out = u_new.view(u_new.shape[0], -1)
    flat_x_out = x_new.view(x_new.shape[0], -1)
    out.layer2_u[:, :] = flat_u_out.cpu().numpy().astype(np.float32, copy=False)
    out.layer2_x[:, :] = flat_x_out.cpu().numpy().astype(np.float32, copy=False)
    out.layer2_g[:, :] = (flat_u_out * flat_x_out).cpu().numpy().astype(np.float32, copy=False)
    target_count = max(int(target_mask.sum().item()), 1)
    return out, {
        "u_clip_fraction_peak": float(u_clip.float().mean().item()) if target_count else np.nan,
        "x_clip_fraction_peak": float(x_clip.float().mean().item()) if target_count else np.nan,
        "any_clip_fraction_peak": float((u_clip | x_clip).float().mean().item()) if target_count else np.nan,
    }


def condition_row(
    trial: SequenceTrial,
    selected: Mapping[str, object],
    condition: str,
    boost_type: str,
    boost_level: float,
    metrics: Mapping[str, Any],
    output_masks: OutputRegionMasks,
    row_idx: int,
    clip_info: Mapping[str, float],
    cfg: ExperimentConfig,
) -> dict[str, object]:
    return {
        "trial_id": int(trial.trial_id),
        "seq_len": int(trial.seq_len),
        "intervention_stage": int(selected["intervention_stage"]),
        "selected_target_index": int(selected["selected_target_index"]),
        "selected_target_image_id": int(selected["candidate_image_id"]),
        "selected_target_label": int(selected["candidate_label"]),
        "target_overlap_group": str(selected["target_overlap_group"]),
        "overlap_fraction": float(selected["overlap_fraction"]),
        "overlap_enrichment": float(selected["overlap_enrichment"]),
        "condition": str(condition),
        "boost_type": str(boost_type),
        "boost_level": float(boost_level),
        "valid": int(metrics["valid"]),
        "invalid_reason": str(metrics["invalid_reason"]),
        "mask_mode": str(output_masks.mask_mode),
        "num_nonbase_output": int(output_masks.num_nonbase[row_idx]),
        "num_peak_output": int(output_masks.num_peak[row_idx]),
        "num_nonpeak_output": int(output_masks.num_nonpeak[row_idx]),
        "total_spike_count": float(metrics["total_spike_count"]),
        "peak_spike_count": float(metrics["peak_spike_count"]) if int(metrics["valid"]) == 1 else np.nan,
        "nonpeak_spike_count": float(metrics["nonpeak_spike_count"]) if int(metrics["valid"]) == 1 else np.nan,
        "peak_spike_fraction": float(metrics["peak_spike_fraction"]) if int(metrics["valid"]) == 1 else np.nan,
        "peak_spike_density": float(metrics["peak_spike_density"]) if int(metrics["valid"]) == 1 else np.nan,
        "nonpeak_spike_density": float(metrics["nonpeak_spike_density"]) if int(metrics["valid"]) == 1 else np.nan,
        "spike_enrichment": float(metrics["spike_enrichment"]) if int(metrics["valid"]) == 1 else np.nan,
        "u_clip_fraction_peak": float(clip_info.get("u_clip_fraction_peak", 0.0)),
        "x_clip_fraction_peak": float(clip_info.get("x_clip_fraction_peak", 0.0)),
        "any_clip_fraction_peak": float(clip_info.get("any_clip_fraction_peak", 0.0)),
        "target_window_steps": int(cfg.sample_steps),
    }


def run_selected_target_conditions(
    net: Any,
    batch: SequenceBatch,
    stage_k: int,
    initial_snapshot: BoundarySnapshot,
    intact_snapshot: BoundarySnapshot,
    input_masks: InputRegionMasks,
    output_masks: OutputRegionMasks,
    selected_rows: Sequence[Mapping[str, object]],
    spike_lookup: Mapping[int, torch.Tensor],
    cfg: ExperimentConfig,
) -> list[dict[str, object]]:
    trial_rows: list[dict[str, object]] = []
    row_by_trial = {int(trial.trial_id): idx for idx, trial in enumerate(batch.trials)}
    for selected in selected_rows:
        trial_id = int(selected["trial_id"])
        if trial_id not in row_by_trial:
            continue
        row_idx = row_by_trial[trial_id]
        trial = batch.trials[row_idx]
        target = spike_lookup[int(selected["candidate_image_id"])].unsqueeze(0).repeat(len(batch.trials), 1, 1, 1, 1)

        baseline_counts = run_target_and_capture_layer2_spikes(net, initial_snapshot, target)
        baseline_metrics = spike_metrics_for_row(baseline_counts[row_idx], output_masks, row_idx)
        trial_rows.append(condition_row(trial, selected, "baseline", "baseline", np.nan, baseline_metrics, output_masks, row_idx, {}, cfg))

        flat_snapshot, flat_clip = make_flatten_or_boost_snapshot(intact_snapshot, input_masks, row_idx, boost_level=0.0, boost_type="flatten", seed=0)
        flat_counts = run_target_and_capture_layer2_spikes(net, flat_snapshot, target)
        flat_metrics = spike_metrics_for_row(flat_counts[row_idx], output_masks, row_idx)
        trial_rows.append(condition_row(trial, selected, "flatten_lambda0", "flatten", 0.0, flat_metrics, output_masks, row_idx, flat_clip, cfg))

        if cfg.include_intact:
            intact_counts = run_target_and_capture_layer2_spikes(net, intact_snapshot, target)
            intact_metrics = spike_metrics_for_row(intact_counts[row_idx], output_masks, row_idx)
            trial_rows.append(condition_row(trial, selected, "intact_natural", "intact", 1.0, intact_metrics, output_masks, row_idx, {}, cfg))

        for level in cfg.boost_levels:
            boost_snapshot, clip_info = make_flatten_or_boost_snapshot(intact_snapshot, input_masks, row_idx, boost_level=float(level), boost_type="peak_boost", seed=0)
            boost_counts = run_target_and_capture_layer2_spikes(net, boost_snapshot, target)
            boost_metrics = spike_metrics_for_row(boost_counts[row_idx], output_masks, row_idx)
            trial_rows.append(condition_row(trial, selected, f"peak_boost_lambda{float(level):g}", "peak_boost", float(level), boost_metrics, output_masks, row_idx, clip_info, cfg))

            if cfg.include_nonpeak_boost_control and float(level) > 0.0:
                ctrl_snapshot, ctrl_clip = make_flatten_or_boost_snapshot(
                    intact_snapshot,
                    input_masks,
                    row_idx,
                    boost_level=float(level),
                    boost_type="nonpeak_boost",
                    seed=mix_seed(cfg.seed, trial_id, int(stage_k), int(float(level) * 1000), 17),
                )
                ctrl_counts = run_target_and_capture_layer2_spikes(net, ctrl_snapshot, target)
                ctrl_metrics = spike_metrics_for_row(ctrl_counts[row_idx], output_masks, row_idx)
                trial_rows.append(condition_row(trial, selected, f"nonpeak_boost_lambda{float(level):g}", "nonpeak_boost", float(level), ctrl_metrics, output_masks, row_idx, ctrl_clip, cfg))

            if cfg.include_shuffle_boost_control and float(level) > 0.0:
                shuf_snapshot, shuf_clip = make_flatten_or_boost_snapshot(
                    intact_snapshot,
                    input_masks,
                    row_idx,
                    boost_level=float(level),
                    boost_type="shuffled_boost",
                    seed=mix_seed(cfg.seed, trial_id, int(stage_k), int(float(level) * 1000), 29),
                )
                shuf_counts = run_target_and_capture_layer2_spikes(net, shuf_snapshot, target)
                shuf_metrics = spike_metrics_for_row(shuf_counts[row_idx], output_masks, row_idx)
                trial_rows.append(condition_row(trial, selected, f"shuffled_boost_lambda{float(level):g}", "shuffled_boost", float(level), shuf_metrics, output_masks, row_idx, shuf_clip, cfg))
    return trial_rows


def compute_boost_effects(trial_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trial_id",
        "seq_len",
        "intervention_stage",
        "selected_target_image_id",
        "selected_target_label",
        "target_overlap_group",
        "overlap_fraction",
        "overlap_enrichment",
        "condition",
        "boost_type",
        "boost_level",
        "baseline_peak_spike_fraction",
        "flatten_peak_spike_fraction",
        "condition_peak_spike_fraction",
        "baseline_spike_enrichment",
        "flatten_spike_enrichment",
        "condition_spike_enrichment",
        "delta_fraction_vs_flatten",
        "delta_enrichment_vs_flatten",
        "delta_fraction_vs_baseline",
        "delta_enrichment_vs_baseline",
        "total_spike_count_condition",
        "total_spike_count_flatten",
        "total_spike_count_baseline",
    ]
    rows: list[dict[str, object]] = []
    if trial_df.empty:
        return pd.DataFrame(columns=columns)
    group_cols = ["trial_id", "seq_len", "intervention_stage", "selected_target_image_id", "selected_target_label", "target_overlap_group"]
    for key, sub in trial_df[trial_df["valid"] == 1].groupby(group_cols, dropna=False):
        baseline = sub[sub["condition"] == "baseline"]
        flatten = sub[sub["condition"] == "flatten_lambda0"]
        if baseline.empty or flatten.empty:
            continue
        b = baseline.iloc[0]
        f = flatten.iloc[0]
        for _, row in sub[sub["condition"] != "baseline"].iterrows():
            rows.append(
                {
                    "trial_id": int(key[0]),
                    "seq_len": int(key[1]),
                    "intervention_stage": int(key[2]),
                    "selected_target_image_id": int(key[3]),
                    "selected_target_label": int(key[4]),
                    "target_overlap_group": str(key[5]),
                    "overlap_fraction": float(row["overlap_fraction"]),
                    "overlap_enrichment": float(row["overlap_enrichment"]),
                    "condition": str(row["condition"]),
                    "boost_type": str(row["boost_type"]),
                    "boost_level": float(row["boost_level"]) if pd.notna(row["boost_level"]) else np.nan,
                    "baseline_peak_spike_fraction": float(b["peak_spike_fraction"]),
                    "flatten_peak_spike_fraction": float(f["peak_spike_fraction"]),
                    "condition_peak_spike_fraction": float(row["peak_spike_fraction"]),
                    "baseline_spike_enrichment": float(b["spike_enrichment"]),
                    "flatten_spike_enrichment": float(f["spike_enrichment"]),
                    "condition_spike_enrichment": float(row["spike_enrichment"]),
                    "delta_fraction_vs_flatten": float(row["peak_spike_fraction"] - f["peak_spike_fraction"]),
                    "delta_enrichment_vs_flatten": float(row["spike_enrichment"] - f["spike_enrichment"]),
                    "delta_fraction_vs_baseline": float(row["peak_spike_fraction"] - b["peak_spike_fraction"]),
                    "delta_enrichment_vs_baseline": float(row["spike_enrichment"] - b["spike_enrichment"]),
                    "total_spike_count_condition": float(row["total_spike_count"]),
                    "total_spike_count_flatten": float(f["total_spike_count"]),
                    "total_spike_count_baseline": float(b["total_spike_count"]),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def compute_dose_response(trial_df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["trial_id", "seq_len", "intervention_stage", "selected_target_image_id", "selected_target_label", "target_overlap_group"]
    for key, sub in trial_df[(trial_df["valid"] == 1) & (trial_df["boost_type"] == "peak_boost")].groupby(group_cols, dropna=False):
        sub = sub.sort_values("boost_level")
        lambdas = sub["boost_level"].to_numpy(dtype=np.float64)
        fractions = sub["peak_spike_fraction"].to_numpy(dtype=np.float64)
        enrichments = sub["spike_enrichment"].to_numpy(dtype=np.float64)
        valid = np.isfinite(lambdas) & np.isfinite(fractions) & np.isfinite(enrichments)
        frac_slope = np.nan
        enrich_slope = np.nan
        if np.count_nonzero(valid) >= 2 and float(np.std(lambdas[valid])) > 0:
            frac_slope = float(np.polyfit(lambdas[valid], fractions[valid], deg=1)[0])
            enrich_slope = float(np.polyfit(lambdas[valid], enrichments[valid], deg=1)[0])
        row = {
            "trial_id": int(key[0]),
            "seq_len": int(key[1]),
            "intervention_stage": int(key[2]),
            "selected_target_image_id": int(key[3]),
            "selected_target_label": int(key[4]),
            "target_overlap_group": str(key[5]),
            "overlap_fraction": float(sub["overlap_fraction"].iloc[0]),
            "overlap_enrichment": float(sub["overlap_enrichment"].iloc[0]),
            "fraction_slope_per_lambda": frac_slope,
            "enrichment_slope_per_lambda": enrich_slope,
            "mean_clip_fraction_across_boosts": safe_mean(sub["any_clip_fraction_peak"].to_numpy(dtype=np.float64)),
        }
        for level in cfg.boost_levels:
            level_sub = sub[np.isclose(sub["boost_level"].to_numpy(dtype=np.float64), float(level))]
            key_suffix = f"lambda{float(level):g}".replace(".", "p")
            row[f"fraction_{key_suffix}"] = float(level_sub["peak_spike_fraction"].iloc[0]) if not level_sub.empty else np.nan
            row[f"enrichment_{key_suffix}"] = float(level_sub["spike_enrichment"].iloc[0]) if not level_sub.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_figures(
    natural_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    effects_df: pd.DataFrame,
    dose_df: pd.DataFrame,
    layout: Any,
) -> dict[str, dict[str, str]]:
    apply_publication_style()
    figures: dict[str, dict[str, str]] = {}

    valid_nat = natural_df[natural_df["valid"] == 1].copy()
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    means = [safe_mean(valid_nat["mean_g_peak"].to_numpy(dtype=np.float64)), safe_mean(valid_nat["mean_g_nonpeak"].to_numpy(dtype=np.float64))]
    sems = [safe_sem(valid_nat["mean_g_peak"].to_numpy(dtype=np.float64)) or 0.0, safe_sem(valid_nat["mean_g_nonpeak"].to_numpy(dtype=np.float64)) or 0.0]
    ax.bar([0, 1], means, yerr=sems, color=[DYNAMIC_COLOR, NOISE_COLOR], edgecolor="black", linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["peak", "nonpeak"])
    ax.set_ylabel("mean Layer 2 STSP g")
    ax.set_title("Natural peak contrast")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    figures["natural_peak_contrast"] = save_figure_all_formats(fig, layout.figure_base("natural_peak_contrast"))

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if not selected_df.empty:
        for group, sub in selected_df.groupby("target_overlap_group"):
            ax.scatter(np.arange(len(sub)), sub["overlap_fraction"].to_numpy(dtype=np.float64), label=str(group), alpha=0.8)
    ax.set_ylabel("selected target overlap fraction")
    ax.set_title("Selected target overlap")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    figures["selected_target_overlap_distribution"] = save_figure_all_formats(fig, layout.figure_base("selected_target_overlap_distribution"))

    peak_boost = trial_df[(trial_df["valid"] == 1) & (trial_df["boost_type"] == "peak_boost")]
    for metric, stem, ylabel in [
        ("peak_spike_fraction", "peak_fraction_dose_response", "peak spike fraction"),
        ("spike_enrichment", "spike_enrichment_dose_response", "spike enrichment"),
    ]:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        grouped = peak_boost.groupby("boost_level", as_index=False)[metric].agg(["mean", "sem"]).reset_index()
        if not grouped.empty:
            ax.errorbar(grouped["boost_level"], grouped["mean"], yerr=grouped["sem"].fillna(0.0), marker="o", color=DYNAMIC_COLOR)
        baseline_mean = safe_mean(trial_df.loc[trial_df["condition"] == "baseline", metric].to_numpy(dtype=np.float64))
        intact_mean = safe_mean(trial_df.loc[trial_df["condition"] == "intact_natural", metric].to_numpy(dtype=np.float64))
        if baseline_mean is not None:
            ax.axhline(baseline_mean, color=NOISE_COLOR, linestyle="--", linewidth=1.2, label="baseline")
        if intact_mean is not None:
            ax.axhline(intact_mean, color=SAMPLE_COLOR, linestyle=":", linewidth=1.5, label="intact")
        ax.set_xlabel("boost level lambda")
        ax.set_ylabel(ylabel)
        ax.set_title(stem.replace("_", " "))
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        figures[stem] = save_figure_all_formats(fig, layout.figure_base(stem))

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    boost_effects = effects_df[effects_df["boost_type"] == "peak_boost"].copy()
    grouped = boost_effects.groupby("boost_level", as_index=False)["delta_fraction_vs_flatten"].agg(["mean", "sem"]).reset_index()
    if not grouped.empty:
        ax.errorbar(grouped["boost_level"], grouped["mean"], yerr=grouped["sem"].fillna(0.0), marker="o", color=DYNAMIC_COLOR)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("boost level lambda")
    ax.set_ylabel("delta fraction vs flatten")
    ax.set_title("Boost effect vs flatten")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    figures["boost_effect_vs_flatten"] = save_figure_all_formats(fig, layout.figure_base("boost_effect_vs_flatten"))

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    means = [
        safe_mean(dose_df["fraction_slope_per_lambda"].to_numpy(dtype=np.float64)) if not dose_df.empty else None,
        safe_mean(dose_df["enrichment_slope_per_lambda"].to_numpy(dtype=np.float64)) if not dose_df.empty else None,
    ]
    sems = [
        safe_sem(dose_df["fraction_slope_per_lambda"].to_numpy(dtype=np.float64)) or 0.0 if not dose_df.empty else 0.0,
        safe_sem(dose_df["enrichment_slope_per_lambda"].to_numpy(dtype=np.float64)) or 0.0 if not dose_df.empty else 0.0,
    ]
    ax.bar([0, 1], [0.0 if item is None else item for item in means], yerr=sems, color=[DYNAMIC_COLOR, SAMPLE_COLOR], edgecolor="black", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["fraction", "enrichment"])
    ax.set_ylabel("slope per lambda")
    ax.set_title("Dose response slope")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    figures["dose_response_slope_distribution"] = save_figure_all_formats(fig, layout.figure_base("dose_response_slope_distribution"))

    if {"nonpeak_boost", "shuffled_boost"}.intersection(set(effects_df["boost_type"].astype(str))) if not effects_df.empty else False:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        grouped = effects_df.groupby("boost_type", as_index=False)["delta_fraction_vs_flatten"].mean(numeric_only=True)
        ax.bar(np.arange(len(grouped)), grouped["delta_fraction_vs_flatten"], color=DYNAMIC_COLOR, edgecolor="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(grouped)))
        ax.set_xticklabels(grouped["boost_type"], rotation=20, ha="right")
        ax.set_ylabel("delta fraction vs flatten")
        ax.set_title("Boost control comparison")
        figures["optional_nonpeak_or_shuffle_control"] = save_figure_all_formats(fig, layout.figure_base("optional_nonpeak_or_shuffle_control"))

    plt.close("all")
    return figures


def summarize_by_boost_level(effects_df: pd.DataFrame, column: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for level, sub in effects_df[effects_df["boost_type"] == "peak_boost"].groupby("boost_level"):
        out[f"lambda_{float(level):g}"] = safe_mean(sub[column].to_numpy(dtype=np.float64))
    return out


def build_summary(
    cfg: ExperimentConfig,
    natural_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    effects_df: pd.DataFrame,
    dose_df: pd.DataFrame,
    projection_infos: Sequence[Mapping[str, Any]],
    exported_files: Mapping[str, str],
    figure_paths: Mapping[str, Any],
) -> dict[str, Any]:
    valid_natural = natural_df[natural_df["valid"] == 1].copy()
    valid_trials = trial_df[trial_df["valid"] == 1].copy()
    mask_counts = trial_df["mask_mode"].value_counts(dropna=False).to_dict() if "mask_mode" in trial_df.columns else {}
    clip_by_level: dict[str, float | None] = {}
    for level, sub in trial_df[trial_df["boost_type"] == "peak_boost"].groupby("boost_level"):
        clip_by_level[f"lambda_{float(level):g}"] = safe_mean(sub["any_clip_fraction_peak"].to_numpy(dtype=np.float64))
    return {
        "experiment_id": EXPERIMENT_ID,
        "config": json_safe(asdict(cfg)),
        "number_of_trials": int(natural_df["trial_id"].nunique()) if "trial_id" in natural_df.columns else 0,
        "valid_trials": int(valid_natural["trial_id"].nunique()) if "trial_id" in valid_natural.columns else 0,
        "number_of_selected_targets": int(selected_df[selected_df["selected"] == 1][["trial_id", "candidate_image_id"]].drop_duplicates().shape[0]) if not selected_df.empty else 0,
        "target_overlap_mode": str(cfg.target_overlap_mode),
        "boost_levels": [float(item) for item in cfg.boost_levels],
        "boost_state_strategy": str(cfg.boost_state_strategy),
        "boost_formula": str(cfg.boost_formula),
        "peak_boost_lambda0_semantics": "intact_prefix_no_extra_boost",
        "mean_natural_peak_minus_nonpeak_g": safe_mean(valid_natural["peak_minus_nonpeak_g"].to_numpy(dtype=np.float64)) if not valid_natural.empty else None,
        "mean_peak_over_nonpeak_g": safe_mean(valid_natural["peak_over_nonpeak_g"].to_numpy(dtype=np.float64)) if not valid_natural.empty else None,
        "mean_cohen_d_peak_vs_nonpeak_g": safe_mean(valid_natural["cohen_d_peak_vs_nonpeak_g"].to_numpy(dtype=np.float64)) if not valid_natural.empty else None,
        "mean_selected_target_overlap_fraction": safe_mean(selected_df.loc[selected_df["selected"] == 1, "overlap_fraction"].to_numpy(dtype=np.float64)) if not selected_df.empty else None,
        "mean_selected_target_overlap_enrichment": safe_mean(selected_df.loc[selected_df["selected"] == 1, "overlap_enrichment"].to_numpy(dtype=np.float64)) if not selected_df.empty else None,
        "mean_delta_fraction_vs_flatten_by_boost_level": summarize_by_boost_level(effects_df, "delta_fraction_vs_flatten"),
        "mean_delta_enrichment_vs_flatten_by_boost_level": summarize_by_boost_level(effects_df, "delta_enrichment_vs_flatten"),
        "mean_fraction_slope_per_lambda": safe_mean(dose_df["fraction_slope_per_lambda"].to_numpy(dtype=np.float64)) if not dose_df.empty else None,
        "mean_enrichment_slope_per_lambda": safe_mean(dose_df["enrichment_slope_per_lambda"].to_numpy(dtype=np.float64)) if not dose_df.empty else None,
        "mean_clipping_fraction_per_boost_level": clip_by_level,
        "mean_clipping_fraction_all_boosts": safe_mean(valid_trials["any_clip_fraction_peak"].to_numpy(dtype=np.float64)) if not valid_trials.empty else None,
        "mask_mode_counts": {str(key): int(value) for key, value in mask_counts.items()},
        "projection_info_examples": json_safe(list(projection_infos[:3])),
        "artifact_paths": json_safe(exported_files),
        "figures": json_safe(figure_paths),
        "smoke_mode": bool(cfg.smoke),
    }


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    layout = prepare_result_layout(cfg.output_dir)
    log_lines: list[str] = []
    run_info_payload = build_run_info(
        experiment_name=EXPERIMENT_ID,
        output_dir=layout.root,
        entry_script=str(Path(__file__).resolve()),
        seed=int(cfg.seed),
        dataset=str(cfg.dataset_root),
        command=" ".join([Path(sys.executable).name, *sys.argv]),
        model_path=str(cfg.model_path),
        status="running",
    )
    write_run_info(layout.meta_dir, run_info_payload)
    seed_everything(int(cfg.seed))
    device = resolve_device(cfg.device)
    log_and_print(log_lines, f"[Init] device={device} smoke={cfg.smoke}")

    dataset = load_mnist_skeleton_dataset(str(cfg.dataset_root), split=str(cfg.split))
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    class_index = build_class_index(dataset, num_classes=int(labels.max()) + 1)
    trials, sequences_df = build_sequence_trials(flat_normalized, class_index, cfg)
    candidate_by_trial_stage = build_candidate_targets(trials, labels, cfg)
    log_and_print(log_lines, f"[Data] generated {len(trials)} trials.")

    net, encoder = load_model_and_encoder(cfg.model_path, device=device, dt=float(cfg.dt), max_duration_ms=float(cfg.max_duration_ms))
    all_image_ids = [item for trial in trials for item in trial.ordered_item_ids]
    for values in candidate_by_trial_stage.values():
        all_image_ids.extend(values)
    spike_lookup = build_spike_lookup(images, encoder, all_image_ids, cfg, device)
    baseline_gain = float(net.layer2.stsp_U)

    natural_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    trial_summary_rows: list[dict[str, object]] = []
    projection_infos: list[Mapping[str, Any]] = []
    for batch in build_batches(trials, spike_lookup, cfg):
        log_and_print(log_lines, f"[Batch] seq_len={batch.seq_len} batch_id={batch.batch_id} batch_size={len(batch.trials)}")
        initial_snapshot, prefix_by_stage = capture_prefix_snapshots(net, batch, cfg)
        for stage_k, intact_snapshot in sorted(prefix_by_stage.items()):
            input_masks = derive_input_region_masks(intact_snapshot, baseline_gain=baseline_gain, epsilon=cfg.epsilon, peak_q=cfg.peak_q)
            output_masks = project_input_masks_to_layer2_outputs(net, intact_snapshot, input_masks, cfg)
            projection_infos.append(output_masks.projection_info)
            natural_rows.extend(compute_natural_contrast_rows(batch, int(stage_k), intact_snapshot, input_masks))
            candidate_rows, selected_rows = select_targets_for_batch(
                net,
                batch,
                int(stage_k),
                initial_snapshot,
                output_masks,
                candidate_by_trial_stage,
                spike_lookup,
                labels,
                cfg,
            )
            selection_rows.extend(candidate_rows)
            trial_summary_rows.extend(
                run_selected_target_conditions(
                    net,
                    batch,
                    int(stage_k),
                    initial_snapshot,
                    intact_snapshot,
                    input_masks,
                    output_masks,
                    selected_rows,
                    spike_lookup,
                    cfg,
                )
            )

    natural_df = pd.DataFrame(natural_rows)
    selection_df = pd.DataFrame(selection_rows)
    trial_df = pd.DataFrame(trial_summary_rows)
    effects_df = compute_boost_effects(trial_df)
    dose_df = compute_dose_response(trial_df, cfg)

    sequences_csv = save_tidy_csv(sequences_df, layout.data_file("sequences.csv"), sort_by=["seq_len", "trial_id", "item_index"])
    natural_csv = save_tidy_csv(natural_df, layout.data_file("layer2_peak_boost_natural_contrast.csv"), sort_by=["seq_len", "trial_id", "intervention_stage"])
    selection_csv = save_tidy_csv(selection_df, layout.data_file("layer2_peak_boost_target_selection.csv"), sort_by=["seq_len", "trial_id", "intervention_stage", "selected", "candidate_rank_by_overlap"])
    trial_csv = save_tidy_csv(trial_df, layout.data_file("layer2_peak_boost_trial_summary.csv"), sort_by=["seq_len", "trial_id", "intervention_stage", "selected_target_image_id", "boost_type", "boost_level"])
    effects_csv = save_tidy_csv(effects_df, layout.data_file("layer2_peak_boost_effects.csv"), sort_by=["seq_len", "trial_id", "intervention_stage", "selected_target_image_id", "boost_type", "boost_level"])
    dose_csv = save_tidy_csv(dose_df, layout.data_file("layer2_peak_boost_dose_response.csv"), sort_by=["seq_len", "trial_id", "intervention_stage", "selected_target_image_id"])

    figure_paths: dict[str, dict[str, str]] = {}
    if not cfg.skip_figures:
        figure_paths = make_figures(natural_df, selection_df[selection_df["selected"] == 1].copy(), trial_df, effects_df, dose_df, layout)

    exported_files = {
        "sequences": sequences_csv,
        "natural_contrast": natural_csv,
        "target_selection": selection_csv,
        "trial_summary": trial_csv,
        "effects": effects_csv,
        "dose_response": dose_csv,
    }
    summary = build_summary(cfg, natural_df, selection_df, trial_df, effects_df, dose_df, projection_infos, exported_files, figure_paths)
    summary_path = save_summary_json(summary, layout.root)
    run_config_path = save_run_config(json_safe(asdict(cfg)), layout.root)
    manifest_path = save_summary_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "artifacts": json_safe({**exported_files, "summary": str(summary_path), "run_config": str(run_config_path), "run_info": str(layout.meta_file("run_info.json")), "figures": figure_paths}),
        },
        layout.root,
        filename="artifact_manifest.json",
    )
    log_and_print(log_lines, f"[Done] Wrote results to {layout.root}")
    log_path = save_log_lines(log_lines, layout.log_dir)
    run_info_path = finalize_run_info(layout.meta_dir, run_info_payload, status="success")
    return {
        "layout": str(layout.root),
        "summary": str(summary_path),
        "run_config": str(run_config_path),
        "manifest": str(manifest_path),
        "run_info": str(run_info_path),
        "log": str(log_path),
        **exported_files,
        "figures": figure_paths,
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = build_argparser()
    cfg = normalize_config(parser.parse_args(argv))
    return run_experiment(cfg)


if __name__ == "__main__":
    main()
