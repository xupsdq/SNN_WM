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


EXPERIMENT_ID = "chunk_stsp_layer2_peak_spiking_intervention"
LAYER_KEYS: tuple[str, ...] = ("layer1", "layer2", "layer3")
FLATTEN_LAYER_KEYS: tuple[str, ...] = LAYER_KEYS
CONDITIONS: tuple[str, ...] = ("baseline", "flatten_memory", "intact_memory")
REGIONS: tuple[str, ...] = ("peak", "nonpeak")
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
    skip_figures: bool
    smoke: bool
    projection_score: str = "abs_weighted_conv"
    projection_peak_rule: str = "top_peak_q_positive_peak_support_within_nonbase_supported"
    projection_support_threshold: float = 0.0
    flatten_layers: tuple[str, ...] = FLATTEN_LAYER_KEYS
    flatten_scope: str = "all_layers_stsp_state_variables"
    overlap_metric: str = "overlap_fraction"
    num_overlap_bins: int = 3
    skip_overlap_analysis: bool = False
    run_overlap_target_selection: bool = False
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
    mean_pairwise_image_similarity: float
    max_pairwise_image_similarity: float
    min_pairwise_image_similarity: float


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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Layer 2 STSP peak-spiking intervention experiment.",
    )
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_PATH_CONFIG.model_path))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_PATH_CONFIG.results_root / EXPERIMENT_ID))
    parser.add_argument("--sample-ms", type=float, default=180.0)
    parser.add_argument("--delay-ms", type=float, default=200.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-sequences", type=int, default=500)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[10])
    parser.add_argument("--samples-per-label", type=int, default=200)
    parser.add_argument("--intervention-stages", type=int, nargs="+", default=[5])
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--peak-q", type=float, default=0.20)
    parser.add_argument("--overlap-metric", type=str, default="overlap_fraction", choices=["overlap_fraction", "overlap_enrichment"])
    parser.add_argument("--num-overlap-bins", type=int, default=3)
    parser.add_argument("--skip-overlap-analysis", action="store_true")
    parser.add_argument("--run-overlap-target-selection", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def normalize_config(args: argparse.Namespace) -> ExperimentConfig:
    sequence_lengths = tuple(dict.fromkeys(int(item) for item in args.sequence_lengths))
    intervention_stages = tuple(dict.fromkeys(int(item) for item in args.intervention_stages))
    if not sequence_lengths:
        raise ValueError("--sequence-lengths must not be empty.")
    if not intervention_stages:
        raise ValueError("--intervention-stages must not be empty.")
    if min(sequence_lengths) < 2:
        raise ValueError("--sequence-lengths must be >= 2.")
    if max(sequence_lengths) > 10:
        raise ValueError("--sequence-lengths must be <= 10 because labels are sampled without replacement.")
    if min(intervention_stages) < 1:
        raise ValueError("--intervention-stages must be >= 1.")
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
        overlap_metric=str(args.overlap_metric),
        num_overlap_bins=int(args.num_overlap_bins),
        skip_overlap_analysis=bool(args.skip_overlap_analysis),
        run_overlap_target_selection=bool(args.run_overlap_target_selection),
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
                "samples_per_label": min(int(cfg.samples_per_label), 8),
                "sample_ms": min(float(cfg.sample_ms), 15.0),
                "delay_ms": min(float(cfg.delay_ms), 10.0),
                "intervention_stages": (5,),
            }
        )
    if cfg.batch_size <= 0 or cfg.max_sequences <= 0:
        raise ValueError("--batch-size and --max-sequences must be positive.")
    if cfg.samples_per_label == 0:
        raise ValueError("--samples-per-label must be non-zero.")
    if min(cfg.sample_steps, cfg.delay_steps) <= 0:
        raise ValueError("--sample-ms and --delay-ms must map to at least one step.")
    if cfg.epsilon < 0.0:
        raise ValueError("--epsilon must be non-negative.")
    if not (0.0 < cfg.peak_q < 1.0):
        raise ValueError("--peak-q must be in (0, 1).")
    if cfg.num_overlap_bins < 2:
        raise ValueError("--num-overlap-bins must be >= 2.")
    if cfg.run_overlap_target_selection:
        raise ValueError("--run-overlap-target-selection is reserved for a future network-resampling experiment.")
    valid_stage_exists = any(int(stage) < int(seq_len) for seq_len in cfg.sequence_lengths for stage in cfg.intervention_stages)
    if not valid_stage_exists:
        raise ValueError("At least one intervention stage must be smaller than one sequence length.")
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
    if np.count_nonzero(valid) == 0:
        return None
    return float(arr[valid].mean())


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
    labels: np.ndarray,
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
            trial_seed = mix_seed(cfg.seed, int(seq_len), int(within_len_idx), 761)
            rng = np.random.default_rng(trial_seed)
            chosen_labels = rng.choice(all_labels, size=int(seq_len), replace=False)
            chosen_item_ids = np.asarray([int(rng.choice(label_pools[int(label)])) for label in chosen_labels], dtype=np.int64)
            order = rng.permutation(int(seq_len))
            ordered_labels = chosen_labels[order].astype(np.int64, copy=False)
            ordered_ids = chosen_item_ids[order].astype(np.int64, copy=False)
            image_sim = flat_normalized[ordered_ids] @ flat_normalized[ordered_ids].T
            mask = ~np.eye(int(seq_len), dtype=bool)
            pairwise = image_sim[mask]
            trial = SequenceTrial(
                trial_id=int(trial_id),
                trial_index_within_seq_len=int(within_len_idx),
                seq_len=int(seq_len),
                ordered_item_ids=tuple(int(item) for item in ordered_ids.tolist()),
                ordered_item_labels=tuple(int(item) for item in ordered_labels.tolist()),
                sequence_seed=int(trial_seed),
                mean_pairwise_image_similarity=float(pairwise.mean()) if pairwise.size > 0 else 1.0,
                max_pairwise_image_similarity=float(pairwise.max()) if pairwise.size > 0 else 1.0,
                min_pairwise_image_similarity=float(pairwise.min()) if pairwise.size > 0 else 1.0,
            )
            trials.append(trial)
            ordered_ids_str = "|".join(str(int(item)) for item in trial.ordered_item_ids)
            ordered_labels_str = "|".join(str(int(item)) for item in trial.ordered_item_labels)
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
                        "mean_pairwise_image_similarity": float(trial.mean_pairwise_image_similarity),
                        "max_pairwise_image_similarity": float(trial.max_pairwise_image_similarity),
                        "min_pairwise_image_similarity": float(trial.min_pairwise_image_similarity),
                    }
                )
            trial_id += 1
    return trials, pd.DataFrame(rows)


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


def build_batches(
    trials: Sequence[SequenceTrial],
    spike_lookup: Mapping[int, torch.Tensor],
    cfg: ExperimentConfig,
) -> Iterator[SequenceBatch]:
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


def forward_three_layers_capture_layer2(net: Any, input_t: torch.Tensor, t_step: int) -> tuple[torch.Tensor, torch.Tensor]:
    s1, _ = net.layer1.forward_step(input_t, t_step, training=False, monitor=False, stsp_mode="dynamic")
    s1_p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1_p, t_step, training=False, monitor=False, stsp_mode="dynamic")
    s2_p = net.pool2(s2.float())
    net.layer3.forward_step(s2_p, t_step, training=False, monitor=False, stsp_mode="dynamic")
    return s2, s2_p


def forward_three_layers(net: Any, input_t: torch.Tensor, t_step: int) -> None:
    forward_three_layers_capture_layer2(net, input_t, t_step)


def _clone_optional_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().cpu().clone()


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
    full_state_by_layer = {str(layer_key): capture_layer_runtime_state(getattr(net, layer_key)) for layer_key in LAYER_KEYS}
    layer2 = net.layer2
    if layer2.u_pre is None or layer2.x_pre is None:
        raise ValueError("layer2 is missing STSP state at the requested boundary.")
    layer2_u = layer2.u_pre.detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=True)
    layer2_x = layer2.x_pre.detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=True)
    layer2_g = (layer2.u_pre * layer2.x_pre).detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=True)
    return BoundarySnapshot(
        stage_k=int(stage_k),
        current_time=int(current_time),
        layer_input_shapes={str(key): tuple(value) for key, value in layer_input_shapes.items()},
        full_state_by_layer=full_state_by_layer,
        layer2_u=layer2_u,
        layer2_x=layer2_x,
        layer2_g=layer2_g,
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
    delta_g = g - float(baseline_gain)
    nonbase = np.abs(delta_g) > float(epsilon)
    peak = np.zeros_like(nonbase, dtype=bool)
    nonpeak = np.zeros_like(nonbase, dtype=bool)
    valid = np.zeros(g.shape[0], dtype=bool)
    invalid_reason: list[str] = []
    num_nonbase = nonbase.sum(axis=1).astype(np.int64)
    num_peak = np.zeros(g.shape[0], dtype=np.int64)
    num_nonpeak = np.zeros(g.shape[0], dtype=np.int64)
    for row_idx in range(g.shape[0]):
        row_nonbase = nonbase[row_idx]
        count_nonbase = int(row_nonbase.sum())
        if count_nonbase <= 0:
            invalid_reason.append("empty_input_nonbase")
            continue
        peak_count = max(1, min(int(math.ceil(float(count_nonbase) * float(peak_q))), count_nonbase))
        nonbase_indices = np.flatnonzero(row_nonbase)
        ranked = nonbase_indices[np.argsort(delta_g[row_idx, nonbase_indices], kind="stable")]
        peak_indices = ranked[-peak_count:]
        peak[row_idx, peak_indices] = True
        nonpeak[row_idx] = row_nonbase & ~peak[row_idx]
        num_peak[row_idx] = int(peak[row_idx].sum())
        num_nonpeak[row_idx] = int(nonpeak[row_idx].sum())
        if num_peak[row_idx] <= 0:
            invalid_reason.append("empty_input_peak")
            continue
        if num_nonpeak[row_idx] <= 0:
            invalid_reason.append("empty_input_nonpeak")
            continue
        valid[row_idx] = True
        invalid_reason.append("")
    return InputRegionMasks(
        nonbase=nonbase,
        peak=peak,
        nonpeak=nonpeak,
        valid=valid,
        invalid_reason=tuple(invalid_reason),
        num_nonbase=num_nonbase,
        num_peak=num_peak,
        num_nonpeak=num_nonpeak,
    )


def project_input_masks_to_layer2_outputs(
    net: Any,
    snapshot: BoundarySnapshot,
    input_masks: InputRegionMasks,
    cfg: ExperimentConfig,
) -> OutputRegionMasks:
    batch_size = int(snapshot.layer2_g.shape[0])
    input_shape = tuple(int(v) for v in snapshot.layer_input_shapes["layer2"][1:])
    output_shape = tuple(int(v) for v in snapshot.full_state_by_layer["layer2"].v_mem.shape[1:])
    input_flat_size = int(np.prod(input_shape))
    output_flat_size = int(np.prod(output_shape))
    if int(snapshot.layer2_g.shape[1]) != input_flat_size:
        raise ValueError(f"Layer2 STSP flat size mismatch: {snapshot.layer2_g.shape[1]} != {input_flat_size}")
    if input_flat_size == output_flat_size and input_shape == output_shape:
        valid = input_masks.valid.copy()
        return OutputRegionMasks(
            mask_mode="direct",
            nonbase=input_masks.nonbase.copy(),
            peak=input_masks.peak.copy(),
            nonpeak=input_masks.nonpeak.copy(),
            valid=valid,
            invalid_reason=input_masks.invalid_reason,
            num_nonbase=input_masks.num_nonbase.copy(),
            num_peak=input_masks.num_peak.copy(),
            num_nonpeak=input_masks.num_nonpeak.copy(),
            projection_info={
                "mask_mode": "direct",
                "input_shape": input_shape,
                "output_shape": output_shape,
                "shape_check": "layer2_stsp_and_layer2_spikes_flat_shapes_match",
            },
        )

    weight = net.layer2.kernels.detach().abs().cpu().to(torch.float32)
    nonbase_out = np.zeros((batch_size, output_flat_size), dtype=bool)
    peak_out = np.zeros((batch_size, output_flat_size), dtype=bool)
    nonpeak_out = np.zeros((batch_size, output_flat_size), dtype=bool)
    valid = np.zeros(batch_size, dtype=bool)
    invalid_reason: list[str] = []
    num_nonbase = np.zeros(batch_size, dtype=np.int64)
    num_peak = np.zeros(batch_size, dtype=np.int64)
    num_nonpeak = np.zeros(batch_size, dtype=np.int64)

    for row_idx in range(batch_size):
        if not bool(input_masks.valid[row_idx]):
            invalid_reason.append(input_masks.invalid_reason[row_idx])
            continue
        peak_tensor = torch.as_tensor(input_masks.peak[row_idx].reshape((1, *input_shape)), dtype=torch.float32)
        nonpeak_tensor = torch.as_tensor(input_masks.nonpeak[row_idx].reshape((1, *input_shape)), dtype=torch.float32)
        nonbase_tensor = torch.as_tensor(input_masks.nonbase[row_idx].reshape((1, *input_shape)), dtype=torch.float32)
        peak_score = F.conv2d(peak_tensor, weight, stride=int(net.layer2.stride), padding=int(net.layer2.padding)).reshape(-1).numpy()
        nonpeak_score = F.conv2d(nonpeak_tensor, weight, stride=int(net.layer2.stride), padding=int(net.layer2.padding)).reshape(-1).numpy()
        nonbase_score = F.conv2d(nonbase_tensor, weight, stride=int(net.layer2.stride), padding=int(net.layer2.padding)).reshape(-1).numpy()
        if peak_score.size != output_flat_size:
            raise ValueError(f"Projected output size mismatch: {peak_score.size} != {output_flat_size}")
        supported = nonbase_score > float(cfg.projection_support_threshold)
        positive_peak = peak_score > float(cfg.projection_support_threshold)
        supported_count = int(supported.sum())
        if supported_count <= 0:
            invalid_reason.append("empty_projected_nonbase")
            continue
        candidate = supported & positive_peak
        candidate_count = int(candidate.sum())
        if candidate_count <= 0:
            invalid_reason.append("empty_projected_peak_support")
            continue
        peak_count = max(1, min(int(math.ceil(float(supported_count) * float(cfg.peak_q))), candidate_count))
        candidate_indices = np.flatnonzero(candidate)
        ranked = candidate_indices[np.argsort(peak_score[candidate_indices], kind="stable")]
        chosen_peak = ranked[-peak_count:]
        row_peak = np.zeros(output_flat_size, dtype=bool)
        row_peak[chosen_peak] = True
        row_nonbase = supported
        row_nonpeak = row_nonbase & ~row_peak
        if int(row_nonpeak.sum()) <= 0:
            invalid_reason.append("empty_projected_nonpeak")
            continue
        nonbase_out[row_idx] = row_nonbase
        peak_out[row_idx] = row_peak
        nonpeak_out[row_idx] = row_nonpeak
        num_nonbase[row_idx] = int(row_nonbase.sum())
        num_peak[row_idx] = int(row_peak.sum())
        num_nonpeak[row_idx] = int(row_nonpeak.sum())
        valid[row_idx] = True
        invalid_reason.append("")

    return OutputRegionMasks(
        mask_mode="projected",
        nonbase=nonbase_out,
        peak=peak_out,
        nonpeak=nonpeak_out,
        valid=valid,
        invalid_reason=tuple(invalid_reason),
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


def derive_nonbase_mask_from_layer_state(
    layer_state: LayerRuntimeState,
    *,
    baseline_gain: float,
    epsilon: float,
) -> np.ndarray:
    if layer_state.u_pre is None or layer_state.x_pre is None:
        raise ValueError("Cannot derive nonbase mask from a layer without u_pre/x_pre.")
    gain = (layer_state.u_pre * layer_state.x_pre).view(layer_state.u_pre.shape[0], -1)
    delta = gain.detach().cpu().numpy().astype(np.float64, copy=False) - float(baseline_gain)
    return np.abs(delta) > float(epsilon)


def flatten_layer_state_variables(
    layer_state: LayerRuntimeState,
    nonbase_mask: np.ndarray,
) -> tuple[LayerRuntimeState, dict[str, Any]]:
    if layer_state.u_pre is None or layer_state.x_pre is None:
        raise ValueError("Cannot flatten a layer without u_pre/x_pre.")
    u_new = layer_state.u_pre.clone()
    x_new = layer_state.x_pre.clone()
    flat_u = u_new.view(u_new.shape[0], -1)
    flat_x = x_new.view(x_new.shape[0], -1)
    mask_arr = np.asarray(nonbase_mask, dtype=bool)
    if mask_arr.shape != tuple(flat_u.shape):
        raise ValueError(f"Flatten mask shape mismatch: mask={mask_arr.shape}, state={tuple(flat_u.shape)}")
    nonbase_counts: list[int] = []
    for row_idx in range(flat_u.shape[0]):
        row_mask_np = mask_arr[row_idx]
        nonbase_counts.append(int(row_mask_np.sum()))
        if not row_mask_np.any():
            continue
        row_mask = torch.as_tensor(row_mask_np, dtype=torch.bool, device=flat_u.device)
        flat_u[row_idx, row_mask] = flat_u[row_idx, row_mask].mean()
        flat_x[row_idx, row_mask] = flat_x[row_idx, row_mask].mean()
    return (
        LayerRuntimeState(
            v_mem=layer_state.v_mem,
            g_e=layer_state.g_e,
            res=layer_state.res,
            inh_trace=layer_state.inh_trace,
            u_pre=u_new,
            x_pre=x_new,
            pre_trace=layer_state.pre_trace,
            input_trace=layer_state.input_trace,
            eligibility_trace=layer_state.eligibility_trace,
            firing_times=layer_state.firing_times,
        ),
        {
            "trials_with_nonbase": int(np.count_nonzero(np.asarray(nonbase_counts) > 0)),
            "total_nonbase_elements": int(np.sum(nonbase_counts)),
            "min_nonbase_elements_per_trial": int(np.min(nonbase_counts)) if nonbase_counts else 0,
            "max_nonbase_elements_per_trial": int(np.max(nonbase_counts)) if nonbase_counts else 0,
        },
    )


def apply_all_layers_state_variable_flatten(
    net: Any,
    snapshot: BoundarySnapshot,
    layer2_input_masks: InputRegionMasks,
    cfg: ExperimentConfig,
) -> tuple[BoundarySnapshot, dict[str, Any]]:
    flattened = clone_boundary_snapshot(snapshot)
    layer_infos: dict[str, Any] = {}
    for layer_key in cfg.flatten_layers:
        if layer_key not in LAYER_KEYS:
            raise ValueError(f"Unsupported flatten layer: {layer_key}")
        layer_state = flattened.full_state_by_layer[str(layer_key)]
        if layer_key == "layer2":
            nonbase_mask = np.asarray(layer2_input_masks.nonbase, dtype=bool)
            mask_source = "layer2_intact_prefix_nonbase_mask"
        else:
            nonbase_mask = derive_nonbase_mask_from_layer_state(
                layer_state,
                baseline_gain=float(getattr(net, layer_key).stsp_U),
                epsilon=float(cfg.epsilon),
            )
            mask_source = f"{layer_key}_intact_prefix_nonbase_mask"
        new_state, layer_info = flatten_layer_state_variables(layer_state, nonbase_mask)
        flattened.full_state_by_layer[str(layer_key)] = new_state
        layer_infos[str(layer_key)] = {
            **layer_info,
            "mask_source": mask_source,
            "state_shape": tuple(int(v) for v in new_state.u_pre.shape) if new_state.u_pre is not None else None,
        }

    layer2_state = flattened.full_state_by_layer["layer2"]
    if layer2_state.u_pre is None or layer2_state.x_pre is None:
        raise ValueError("layer2 snapshot is missing u_pre/x_pre after flatten intervention.")
    flat_u = layer2_state.u_pre.view(layer2_state.u_pre.shape[0], -1)
    flat_x = layer2_state.x_pre.view(layer2_state.x_pre.shape[0], -1)
    flattened.layer2_u[:, :] = flat_u.cpu().numpy().astype(np.float32, copy=False)
    flattened.layer2_x[:, :] = flat_x.cpu().numpy().astype(np.float32, copy=False)
    flattened.layer2_g[:, :] = (flat_u * flat_x).cpu().numpy().astype(np.float32, copy=False)
    return (
        flattened,
        {
            "flatten_scope": str(cfg.flatten_scope),
            "flatten_layers": [str(layer) for layer in cfg.flatten_layers],
            "layer_infos": layer_infos,
        },
    )


def run_target_and_capture_layer2_spikes(
    net: Any,
    start_snapshot: BoundarySnapshot,
    target_spikes: torch.Tensor,
) -> np.ndarray:
    restore_network_boundary_state(net, start_snapshot)
    current_time = int(start_snapshot.current_time)
    captured: list[torch.Tensor] = []
    with torch.no_grad():
        for step_idx in range(int(target_spikes.shape[1])):
            s2, _ = forward_three_layers_capture_layer2(net, target_spikes[:, step_idx, ...], current_time)
            captured.append(s2.detach().to(torch.float32).cpu())
            current_time += 1
    spike_tensor = torch.stack(captured, dim=0).sum(dim=0)
    return spike_tensor.view(spike_tensor.shape[0], -1).numpy().astype(np.float32, copy=False)


def compute_condition_trial_rows(
    batch: SequenceBatch,
    cfg: ExperimentConfig,
    *,
    condition: str,
    stage_k: int,
    target_item_index: int,
    spike_counts: np.ndarray,
    output_masks: OutputRegionMasks,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_idx, trial in enumerate(batch.trials):
        target_image_id = int(trial.ordered_item_ids[target_item_index - 1])
        target_label = int(trial.ordered_item_labels[target_item_index - 1])
        counts = np.asarray(spike_counts[row_idx], dtype=np.float64)
        total = float(counts.sum())
        is_valid = bool(output_masks.valid[row_idx])
        if is_valid:
            peak_mask = output_masks.peak[row_idx]
            nonpeak_mask = output_masks.nonpeak[row_idx]
            peak_count = float(counts[peak_mask].sum())
            nonpeak_count = float(counts[nonpeak_mask].sum())
            peak_density = float(peak_count / max(int(output_masks.num_peak[row_idx]), 1))
            nonpeak_density = float(nonpeak_count / max(int(output_masks.num_nonpeak[row_idx]), 1))
            peak_fraction = float(peak_count / (total + SMALL_EPS))
            enrichment = float(peak_density / (nonpeak_density + SMALL_EPS))
        else:
            peak_count = np.nan
            nonpeak_count = np.nan
            peak_density = np.nan
            nonpeak_density = np.nan
            peak_fraction = np.nan
            enrichment = np.nan
        rows.append(
            {
                "trial_id": int(trial.trial_id),
                "seq_len": int(trial.seq_len),
                "intervention_stage": int(stage_k),
                "target_item_index": int(target_item_index),
                "target_image_id": target_image_id,
                "target_label": target_label,
                "condition": str(condition),
                "valid": int(is_valid),
                "invalid_reason": str(output_masks.invalid_reason[row_idx]),
                "mask_mode": str(output_masks.mask_mode),
                "num_nonbase": int(output_masks.num_nonbase[row_idx]),
                "num_peak": int(output_masks.num_peak[row_idx]),
                "num_nonpeak": int(output_masks.num_nonpeak[row_idx]),
                "total_spike_count": total,
                "peak_spike_count": peak_count,
                "nonpeak_spike_count": nonpeak_count,
                "peak_spike_fraction": peak_fraction,
                "peak_spike_density": peak_density,
                "nonpeak_spike_density": nonpeak_density,
                "spike_enrichment": enrichment,
                "mean_spike_count_all": float(counts.mean()) if counts.size else np.nan,
                "target_window_steps": int(cfg.sample_steps),
            }
        )
    return rows


def compute_paired_effects(trial_df: pd.DataFrame) -> pd.DataFrame:
    valid_df = trial_df[trial_df["valid"] == 1].copy()
    columns = [
        "trial_id",
        "seq_len",
        "intervention_stage",
        "target_item_index",
        "delta_fraction_intact_vs_baseline",
        "delta_fraction_flatten_vs_baseline",
        "delta_fraction_intact_minus_flatten",
        "delta_enrichment_intact_vs_baseline",
        "delta_enrichment_flatten_vs_baseline",
        "delta_enrichment_intact_minus_flatten",
        "ratio_enrichment_flatten_over_intact",
        "ratio_fraction_flatten_over_intact",
    ]
    if valid_df.empty:
        return pd.DataFrame(columns=columns)
    index_cols = ["trial_id", "seq_len", "intervention_stage", "target_item_index"]
    pivot = valid_df.pivot_table(
        index=index_cols,
        columns="condition",
        values=["peak_spike_fraction", "spike_enrichment"],
        aggfunc="first",
    )
    rows: list[dict[str, object]] = []
    required = {"baseline", "intact_memory", "flatten_memory"}
    for key, row in pivot.iterrows():
        available = set(pivot.columns.get_level_values("condition"))
        if not required.issubset(available):
            continue
        try:
            f_base = float(row[("peak_spike_fraction", "baseline")])
            f_intact = float(row[("peak_spike_fraction", "intact_memory")])
            f_flatten = float(row[("peak_spike_fraction", "flatten_memory")])
            e_base = float(row[("spike_enrichment", "baseline")])
            e_intact = float(row[("spike_enrichment", "intact_memory")])
            e_flatten = float(row[("spike_enrichment", "flatten_memory")])
        except KeyError:
            continue
        if not all(np.isfinite([f_base, f_intact, f_flatten, e_base, e_intact, e_flatten])):
            continue
        trial_id, seq_len, intervention_stage, target_item_index = key
        fraction_intact_bias = f_intact - f_base
        fraction_flatten_bias = f_flatten - f_base
        enrichment_intact_bias = e_intact - e_base
        enrichment_flatten_bias = e_flatten - e_base
        rows.append(
            {
                "trial_id": int(trial_id),
                "seq_len": int(seq_len),
                "intervention_stage": int(intervention_stage),
                "target_item_index": int(target_item_index),
                "delta_fraction_intact_vs_baseline": float(fraction_intact_bias),
                "delta_fraction_flatten_vs_baseline": float(fraction_flatten_bias),
                "delta_fraction_intact_minus_flatten": float(f_intact - f_flatten),
                "delta_enrichment_intact_vs_baseline": float(enrichment_intact_bias),
                "delta_enrichment_flatten_vs_baseline": float(enrichment_flatten_bias),
                "delta_enrichment_intact_minus_flatten": float(e_intact - e_flatten),
                "ratio_enrichment_flatten_over_intact": float(enrichment_flatten_bias / (enrichment_intact_bias + SMALL_EPS)),
                "ratio_fraction_flatten_over_intact": float(fraction_flatten_bias / (fraction_intact_bias + SMALL_EPS)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_stage_summary(trial_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["condition", "region", "mean_spike_density", "mean_spike_fraction", "mean_total_spike_count"]
    if trial_df.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for _, row in trial_df[trial_df["valid"] == 1].iterrows():
        total = float(row["total_spike_count"])
        peak_count = float(row["peak_spike_count"])
        nonpeak_count = float(row["nonpeak_spike_count"])
        rows.append(
            {
                "condition": row["condition"],
                "region": "peak",
                "spike_density": float(row["peak_spike_density"]),
                "spike_fraction": peak_count / (total + SMALL_EPS),
                "total_spike_count": total,
            }
        )
        rows.append(
            {
                "condition": row["condition"],
                "region": "nonpeak",
                "spike_density": float(row["nonpeak_spike_density"]),
                "spike_fraction": nonpeak_count / (total + SMALL_EPS),
                "total_spike_count": total,
            }
        )
    region_df = pd.DataFrame(rows)
    if region_df.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        region_df.groupby(["condition", "region"], as_index=False)
        .agg(
            mean_spike_density=("spike_density", "mean"),
            mean_spike_fraction=("spike_fraction", "mean"),
            mean_total_spike_count=("total_spike_count", "mean"),
        )
        .loc[:, columns]
    )
    return grouped


def _finite_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.count_nonzero(valid) < 2:
        return None
    x_valid = x_arr[valid]
    y_valid = y_arr[valid]
    if float(np.std(x_valid)) <= 0.0 or float(np.std(y_valid)) <= 0.0:
        return None
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def _spearman_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_rank = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(np.asarray(y, dtype=np.float64)).rank(method="average").to_numpy(dtype=np.float64)
    return _finite_corr(x_rank, y_rank)


def compute_overlap_trial_effects(trial_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trial_id",
        "seq_len",
        "intervention_stage",
        "target_item_index",
        "target_image_id",
        "target_label",
        "mask_mode",
        "valid",
        "overlap_fraction",
        "overlap_enrichment",
        "baseline_peak_spike_fraction",
        "flatten_peak_spike_fraction",
        "intact_peak_spike_fraction",
        "baseline_spike_enrichment",
        "flatten_spike_enrichment",
        "intact_spike_enrichment",
        "fraction_intact_bias",
        "fraction_flatten_bias",
        "fraction_peakvalley_effect",
        "enrichment_intact_bias",
        "enrichment_flatten_bias",
        "enrichment_peakvalley_effect",
        "total_spike_count_baseline",
        "total_spike_count_flatten",
        "total_spike_count_intact",
    ]
    if trial_df.empty:
        return pd.DataFrame(columns=columns)
    index_cols = ["trial_id", "seq_len", "intervention_stage", "target_item_index"]
    required_conditions = {"baseline", "flatten_memory", "intact_memory"}
    rows: list[dict[str, object]] = []
    for key, sub in trial_df.groupby(index_cols, dropna=False, sort=True):
        by_condition = {str(row["condition"]): row for _, row in sub.iterrows()}
        if not required_conditions.issubset(by_condition):
            continue
        baseline = by_condition["baseline"]
        flatten = by_condition["flatten_memory"]
        intact = by_condition["intact_memory"]
        valid = int(int(baseline["valid"]) == 1 and int(flatten["valid"]) == 1 and int(intact["valid"]) == 1)
        if valid != 1:
            continue
        trial_id, seq_len, intervention_stage, target_item_index = key
        baseline_fraction = float(baseline["peak_spike_fraction"])
        flatten_fraction = float(flatten["peak_spike_fraction"])
        intact_fraction = float(intact["peak_spike_fraction"])
        baseline_enrichment = float(baseline["spike_enrichment"])
        flatten_enrichment = float(flatten["spike_enrichment"])
        intact_enrichment = float(intact["spike_enrichment"])
        values = [
            baseline_fraction,
            flatten_fraction,
            intact_fraction,
            baseline_enrichment,
            flatten_enrichment,
            intact_enrichment,
        ]
        if not all(np.isfinite(values)):
            continue
        rows.append(
            {
                "trial_id": int(trial_id),
                "seq_len": int(seq_len),
                "intervention_stage": int(intervention_stage),
                "target_item_index": int(target_item_index),
                "target_image_id": int(baseline["target_image_id"]),
                "target_label": int(baseline["target_label"]),
                "mask_mode": str(baseline["mask_mode"]),
                "valid": int(valid),
                "overlap_fraction": float(baseline_fraction),
                "overlap_enrichment": float(baseline_enrichment),
                "baseline_peak_spike_fraction": float(baseline_fraction),
                "flatten_peak_spike_fraction": float(flatten_fraction),
                "intact_peak_spike_fraction": float(intact_fraction),
                "baseline_spike_enrichment": float(baseline_enrichment),
                "flatten_spike_enrichment": float(flatten_enrichment),
                "intact_spike_enrichment": float(intact_enrichment),
                "fraction_intact_bias": float(intact_fraction - baseline_fraction),
                "fraction_flatten_bias": float(flatten_fraction - baseline_fraction),
                "fraction_peakvalley_effect": float(intact_fraction - flatten_fraction),
                "enrichment_intact_bias": float(intact_enrichment - baseline_enrichment),
                "enrichment_flatten_bias": float(flatten_enrichment - baseline_enrichment),
                "enrichment_peakvalley_effect": float(intact_enrichment - flatten_enrichment),
                "total_spike_count_baseline": float(baseline["total_spike_count"]),
                "total_spike_count_flatten": float(flatten["total_spike_count"]),
                "total_spike_count_intact": float(intact["total_spike_count"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def assign_overlap_bins(overlap_df: pd.DataFrame, metric: str, num_bins: int) -> pd.DataFrame:
    out = overlap_df.copy()
    if out.empty:
        out["overlap_bin"] = pd.Series(dtype="object")
        return out
    values = pd.to_numeric(out[metric], errors="coerce")
    valid_mask = values.replace([np.inf, -np.inf], np.nan).notna()
    out["overlap_bin"] = "invalid_overlap"
    valid_count = int(valid_mask.sum())
    if valid_count <= 0:
        return out
    bins = max(1, min(int(num_bins), valid_count))
    ranked = values[valid_mask].rank(method="first")
    try:
        codes = pd.qcut(ranked, q=bins, labels=False, duplicates="drop")
    except ValueError:
        order = np.argsort(ranked.to_numpy(dtype=np.float64), kind="stable")
        codes_arr = np.zeros(valid_count, dtype=np.int64)
        for bin_idx, chunk in enumerate(np.array_split(order, bins)):
            codes_arr[chunk] = int(bin_idx)
        codes = pd.Series(codes_arr, index=ranked.index)
    codes = pd.Series(codes, index=ranked.index).astype(int)
    observed_bins = sorted(int(code) for code in pd.unique(codes))
    if len(observed_bins) == 3:
        label_map = {observed_bins[0]: "low_overlap", observed_bins[1]: "mid_overlap", observed_bins[2]: "high_overlap"}
    else:
        label_map = {code: f"overlap_bin_{idx + 1}" for idx, code in enumerate(observed_bins)}
        if len(observed_bins) >= 2:
            label_map[observed_bins[0]] = "low_overlap"
            label_map[observed_bins[-1]] = "high_overlap"
    out.loc[valid_mask, "overlap_bin"] = codes.map(label_map).to_numpy()
    return out


def summarize_overlap_bins(overlap_df: pd.DataFrame, metric: str, num_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "overlap_metric",
        "overlap_bin",
        "n_trials",
        "overlap_min",
        "overlap_max",
        "overlap_mean",
        "mean_fraction_intact_bias",
        "mean_fraction_flatten_bias",
        "mean_fraction_peakvalley_effect",
        "sem_fraction_peakvalley_effect",
        "mean_enrichment_intact_bias",
        "mean_enrichment_flatten_bias",
        "mean_enrichment_peakvalley_effect",
        "sem_enrichment_peakvalley_effect",
        "mean_baseline_peak_spike_fraction",
        "mean_flatten_peak_spike_fraction",
        "mean_intact_peak_spike_fraction",
        "mean_baseline_spike_enrichment",
        "mean_flatten_spike_enrichment",
        "mean_intact_spike_enrichment",
    ]
    binned = assign_overlap_bins(overlap_df, metric, num_bins)
    if binned.empty:
        return binned, pd.DataFrame(columns=columns)
    order = ["low_overlap", "mid_overlap", "high_overlap"]
    rows: list[dict[str, object]] = []
    for overlap_bin, sub in binned[binned["overlap_bin"] != "invalid_overlap"].groupby("overlap_bin", sort=False):
        metric_values = sub[metric].to_numpy(dtype=np.float64)
        fraction_peakvalley = sub["fraction_peakvalley_effect"].to_numpy(dtype=np.float64)
        enrichment_peakvalley = sub["enrichment_peakvalley_effect"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "overlap_metric": str(metric),
                "overlap_bin": str(overlap_bin),
                "n_trials": int(len(sub)),
                "overlap_min": float(np.nanmin(metric_values)),
                "overlap_max": float(np.nanmax(metric_values)),
                "overlap_mean": float(np.nanmean(metric_values)),
                "mean_fraction_intact_bias": safe_mean(sub["fraction_intact_bias"].to_numpy(dtype=np.float64)),
                "mean_fraction_flatten_bias": safe_mean(sub["fraction_flatten_bias"].to_numpy(dtype=np.float64)),
                "mean_fraction_peakvalley_effect": safe_mean(fraction_peakvalley),
                "sem_fraction_peakvalley_effect": safe_sem(fraction_peakvalley),
                "mean_enrichment_intact_bias": safe_mean(sub["enrichment_intact_bias"].to_numpy(dtype=np.float64)),
                "mean_enrichment_flatten_bias": safe_mean(sub["enrichment_flatten_bias"].to_numpy(dtype=np.float64)),
                "mean_enrichment_peakvalley_effect": safe_mean(enrichment_peakvalley),
                "sem_enrichment_peakvalley_effect": safe_sem(enrichment_peakvalley),
                "mean_baseline_peak_spike_fraction": safe_mean(sub["baseline_peak_spike_fraction"].to_numpy(dtype=np.float64)),
                "mean_flatten_peak_spike_fraction": safe_mean(sub["flatten_peak_spike_fraction"].to_numpy(dtype=np.float64)),
                "mean_intact_peak_spike_fraction": safe_mean(sub["intact_peak_spike_fraction"].to_numpy(dtype=np.float64)),
                "mean_baseline_spike_enrichment": safe_mean(sub["baseline_spike_enrichment"].to_numpy(dtype=np.float64)),
                "mean_flatten_spike_enrichment": safe_mean(sub["flatten_spike_enrichment"].to_numpy(dtype=np.float64)),
                "mean_intact_spike_enrichment": safe_mean(sub["intact_spike_enrichment"].to_numpy(dtype=np.float64)),
            }
        )
    summary = pd.DataFrame(rows, columns=columns)
    if not summary.empty:
        summary["_sort"] = summary["overlap_bin"].map({label: idx for idx, label in enumerate(order)}).fillna(99)
        summary = summary.sort_values(["_sort", "overlap_mean"], kind="stable").drop(columns=["_sort"]).reset_index(drop=True)
    return binned, summary


def compute_overlap_correlations(overlap_df: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "overlap_fraction_with_intact_fraction": ("overlap_fraction", "intact_peak_spike_fraction"),
        "overlap_fraction_with_flatten_fraction": ("overlap_fraction", "flatten_peak_spike_fraction"),
        "overlap_fraction_with_peakvalley_fraction_effect": ("overlap_fraction", "fraction_peakvalley_effect"),
        "overlap_enrichment_with_intact_enrichment": ("overlap_enrichment", "intact_spike_enrichment"),
        "overlap_enrichment_with_flatten_enrichment": ("overlap_enrichment", "flatten_spike_enrichment"),
        "overlap_enrichment_with_peakvalley_enrichment_effect": ("overlap_enrichment", "enrichment_peakvalley_effect"),
    }
    row: dict[str, object] = {"n_trials": int(len(overlap_df))}
    for name, (left, right) in specs.items():
        pearson_value = _finite_corr(overlap_df[left].to_numpy(dtype=np.float64), overlap_df[right].to_numpy(dtype=np.float64))
        row[f"corr_{name}"] = pearson_value
        row[f"pearson_corr_{name}"] = pearson_value
        row[f"spearman_corr_{name}"] = _spearman_corr(overlap_df[left].to_numpy(dtype=np.float64), overlap_df[right].to_numpy(dtype=np.float64))
    return pd.DataFrame([row])


def build_overlap_analysis_summary(
    cfg: ExperimentConfig,
    overlap_df: pd.DataFrame,
    bin_summary_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    artifact_paths: Mapping[str, str],
) -> dict[str, Any]:
    high = bin_summary_df[bin_summary_df["overlap_bin"] == "high_overlap"] if not bin_summary_df.empty else pd.DataFrame()
    low = bin_summary_df[bin_summary_df["overlap_bin"] == "low_overlap"] if not bin_summary_df.empty else pd.DataFrame()
    high_minus_low_fraction = None
    high_minus_low_enrichment = None
    if not high.empty and not low.empty:
        high_minus_low_fraction = float(high["mean_fraction_peakvalley_effect"].iloc[0] - low["mean_fraction_peakvalley_effect"].iloc[0])
        high_minus_low_enrichment = float(high["mean_enrichment_peakvalley_effect"].iloc[0] - low["mean_enrichment_peakvalley_effect"].iloc[0])
    corr_payload = corr_df.iloc[0].to_dict() if not corr_df.empty else {}
    return {
        "overlap_metric_used": str(cfg.overlap_metric),
        "num_overlap_bins": int(cfg.num_overlap_bins),
        "mean_overlap_fraction": safe_mean(overlap_df["overlap_fraction"].to_numpy(dtype=np.float64)) if not overlap_df.empty else None,
        "mean_overlap_enrichment": safe_mean(overlap_df["overlap_enrichment"].to_numpy(dtype=np.float64)) if not overlap_df.empty else None,
        "mean_fraction_peakvalley_effect": safe_mean(overlap_df["fraction_peakvalley_effect"].to_numpy(dtype=np.float64)) if not overlap_df.empty else None,
        "mean_enrichment_peakvalley_effect": safe_mean(overlap_df["enrichment_peakvalley_effect"].to_numpy(dtype=np.float64)) if not overlap_df.empty else None,
        "correlation_summaries": json_safe(corr_payload),
        "high_vs_low_overlap_comparisons": {
            "high_minus_low_fraction_peakvalley_effect": high_minus_low_fraction,
            "high_minus_low_enrichment_peakvalley_effect": high_minus_low_enrichment,
        },
        "artifact_paths": json_safe(artifact_paths),
        "smoke_mode": bool(cfg.smoke),
    }


def build_summary(
    cfg: ExperimentConfig,
    trial_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    projection_infos: Sequence[Mapping[str, Any]],
    flatten_infos: Sequence[Mapping[str, Any]],
    overlap_analysis: Mapping[str, Any] | None,
    exported_files: Mapping[str, Any],
    figure_paths: Mapping[str, Any],
) -> dict[str, Any]:
    valid_df = trial_df[trial_df["valid"] == 1].copy() if not trial_df.empty else pd.DataFrame()

    def condition_mean(condition: str, column: str) -> float | None:
        if valid_df.empty:
            return None
        return safe_mean(valid_df.loc[valid_df["condition"] == condition, column].to_numpy(dtype=np.float64))

    mask_mode_counts = trial_df["mask_mode"].value_counts(dropna=False).to_dict() if "mask_mode" in trial_df.columns else {}
    return {
        "experiment_id": EXPERIMENT_ID,
        "config": json_safe(asdict(cfg)),
        "number_of_trials": int(trial_df["trial_id"].nunique()) if "trial_id" in trial_df.columns else 0,
        "valid_trials": int(valid_df["trial_id"].nunique()) if "trial_id" in valid_df.columns else 0,
        "invalid_trials": int(trial_df.loc[trial_df["valid"] != 1, "trial_id"].nunique()) if "valid" in trial_df.columns else 0,
        "trial_condition_rows": int(len(trial_df)),
        "valid_trial_condition_rows": int(len(valid_df)),
        "intervention_stages": [int(stage) for stage in cfg.intervention_stages],
        "mask_mode_counts": {str(key): int(value) for key, value in mask_mode_counts.items()},
        "projection_info_examples": json_safe(list(projection_infos[:3])),
        "flatten_scope": str(cfg.flatten_scope),
        "flatten_layers": [str(layer) for layer in cfg.flatten_layers],
        "flatten_info_examples": json_safe(list(flatten_infos[:3])),
        "mean_baseline_peak_spike_fraction": condition_mean("baseline", "peak_spike_fraction"),
        "mean_intact_peak_spike_fraction": condition_mean("intact_memory", "peak_spike_fraction"),
        "mean_flatten_peak_spike_fraction": condition_mean("flatten_memory", "peak_spike_fraction"),
        "mean_delta_fraction_intact_minus_flatten": safe_mean(paired_df["delta_fraction_intact_minus_flatten"].to_numpy()) if not paired_df.empty else None,
        "sem_delta_fraction_intact_minus_flatten": safe_sem(paired_df["delta_fraction_intact_minus_flatten"].to_numpy()) if not paired_df.empty else None,
        "mean_baseline_spike_enrichment": condition_mean("baseline", "spike_enrichment"),
        "mean_intact_spike_enrichment": condition_mean("intact_memory", "spike_enrichment"),
        "mean_flatten_spike_enrichment": condition_mean("flatten_memory", "spike_enrichment"),
        "mean_delta_enrichment_intact_minus_flatten": safe_mean(paired_df["delta_enrichment_intact_minus_flatten"].to_numpy()) if not paired_df.empty else None,
        "sem_delta_enrichment_intact_minus_flatten": safe_sem(paired_df["delta_enrichment_intact_minus_flatten"].to_numpy()) if not paired_df.empty else None,
        "smoke_mode": bool(cfg.smoke),
        "artifact_paths": json_safe(exported_files),
        "figures": json_safe(figure_paths),
        "overlap_analysis": json_safe(overlap_analysis) if overlap_analysis is not None else {"skipped": True},
    }


def plot_condition_bar(
    trial_df: pd.DataFrame,
    *,
    column: str,
    ylabel: str,
    title: str,
    base_path: Path,
) -> dict[str, str]:
    valid_df = trial_df[trial_df["valid"] == 1].copy()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    colors = [NOISE_COLOR, SHUFFLE_COLOR, DYNAMIC_COLOR]
    x = np.arange(len(CONDITIONS), dtype=np.float64)
    means: list[float] = []
    sems: list[float] = []
    for condition in CONDITIONS:
        values = valid_df.loc[valid_df["condition"] == condition, column].to_numpy(dtype=np.float64)
        means.append(float(np.nanmean(values)) if values.size else np.nan)
        sem = safe_sem(values)
        sems.append(float(sem) if sem is not None else 0.0)
    ax.bar(x, means, color=colors, yerr=sems, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["baseline", "flatten", "intact"], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save_figure_all_formats(fig, base_path)


def plot_memory_bias(paired_df: pd.DataFrame, *, base_path: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    specs = [
        ("delta_fraction_flatten_vs_baseline", "delta_fraction_intact_vs_baseline", "Peak fraction bias"),
        ("delta_enrichment_flatten_vs_baseline", "delta_enrichment_intact_vs_baseline", "Spike enrichment bias"),
    ]
    for ax, (flatten_col, intact_col, title) in zip(axes, specs):
        if not paired_df.empty:
            for _, row in paired_df.iterrows():
                ax.plot([0, 1], [row[flatten_col], row[intact_col]], color=NOISE_COLOR, alpha=0.35, linewidth=0.9)
            means = [float(paired_df[flatten_col].mean()), float(paired_df[intact_col].mean())]
            ax.plot([0, 1], means, color=SAMPLE_COLOR, linewidth=2.4, marker="o", markersize=5)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["flatten", "intact"])
        ax.set_title(title)
        ax.set_ylabel("condition - baseline")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return save_figure_all_formats(fig, base_path)


def plot_density_peak_vs_nonpeak(trial_df: pd.DataFrame, *, base_path: Path) -> dict[str, str]:
    valid_df = trial_df[trial_df["valid"] == 1].copy()
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    x = np.arange(len(CONDITIONS), dtype=np.float64)
    width = 0.34
    for offset, region, column, color in [
        (-width / 2, "peak", "peak_spike_density", DYNAMIC_COLOR),
        (width / 2, "nonpeak", "nonpeak_spike_density", NOISE_COLOR),
    ]:
        means: list[float] = []
        sems: list[float] = []
        for condition in CONDITIONS:
            values = valid_df.loc[valid_df["condition"] == condition, column].to_numpy(dtype=np.float64)
            means.append(float(np.nanmean(values)) if values.size else np.nan)
            sem = safe_sem(values)
            sems.append(float(sem) if sem is not None else 0.0)
        ax.bar(x + offset, means, width=width, label=region, color=color, yerr=sems, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["baseline", "flatten", "intact"], rotation=15, ha="right")
    ax.set_ylabel("Layer 2 spike density")
    ax.set_title("Peak vs nonpeak spike density")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save_figure_all_formats(fig, base_path)


def plot_overlap_bin_bias(
    bin_summary_df: pd.DataFrame,
    *,
    flatten_col: str,
    intact_col: str,
    peakvalley_col: str,
    ylabel: str,
    title: str,
    base_path: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    if not bin_summary_df.empty:
        labels = bin_summary_df["overlap_bin"].astype(str).tolist()
        x = np.arange(len(labels), dtype=np.float64)
        width = 0.25
        ax.bar(x - width, bin_summary_df[flatten_col].to_numpy(dtype=np.float64), width=width, label="flatten - baseline", color=SHUFFLE_COLOR, edgecolor="black", linewidth=0.8)
        ax.bar(x, bin_summary_df[intact_col].to_numpy(dtype=np.float64), width=width, label="intact - baseline", color=DYNAMIC_COLOR, edgecolor="black", linewidth=0.8)
        ax.bar(x + width, bin_summary_df[peakvalley_col].to_numpy(dtype=np.float64), width=width, label="intact - flatten", color=SAMPLE_COLOR, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save_figure_all_formats(fig, base_path)


def plot_overlap_scatter_peakvalley_effect(
    overlap_df: pd.DataFrame,
    *,
    metric: str,
    base_path: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if not overlap_df.empty:
        x = overlap_df[metric].to_numpy(dtype=np.float64)
        y = overlap_df["fraction_peakvalley_effect"].to_numpy(dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], color=DYNAMIC_COLOR, alpha=0.75, edgecolor="black", linewidth=0.4)
        if np.count_nonzero(valid) >= 2 and float(np.std(x[valid])) > 0.0:
            slope, intercept = np.polyfit(x[valid], y[valid], deg=1)
            xs = np.linspace(float(np.min(x[valid])), float(np.max(x[valid])), 100)
            ax.plot(xs, slope * xs + intercept, color=SHUFFLE_COLOR, linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(metric)
    ax.set_ylabel("fraction peak-valley effect")
    ax.set_title("Overlap vs peak-valley effect")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save_figure_all_formats(fig, base_path)


def plot_baseline_overlap_vs_intact_response(overlap_df: pd.DataFrame, *, base_path: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if not overlap_df.empty:
        x = overlap_df["baseline_peak_spike_fraction"].to_numpy(dtype=np.float64)
        y = overlap_df["intact_peak_spike_fraction"].to_numpy(dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], color=DYNAMIC_COLOR, alpha=0.75, edgecolor="black", linewidth=0.4)
        if np.count_nonzero(valid) >= 1:
            lo = float(min(np.min(x[valid]), np.min(y[valid])))
            hi = float(max(np.max(x[valid]), np.max(y[valid])))
            ax.plot([lo, hi], [lo, hi], color=NOISE_COLOR, linewidth=1.2, linestyle="--")
    ax.set_xlabel("baseline peak spike fraction")
    ax.set_ylabel("intact peak spike fraction")
    ax.set_title("Baseline overlap vs intact response")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save_figure_all_formats(fig, base_path)


def make_overlap_figures(
    overlap_df: pd.DataFrame,
    bin_summary_df: pd.DataFrame,
    cfg: ExperimentConfig,
    layout: Any,
) -> dict[str, dict[str, str]]:
    figure_paths: dict[str, dict[str, str]] = {}
    figure_paths["overlap_bins_peak_fraction_bias"] = plot_overlap_bin_bias(
        bin_summary_df,
        flatten_col="mean_fraction_flatten_bias",
        intact_col="mean_fraction_intact_bias",
        peakvalley_col="mean_fraction_peakvalley_effect",
        ylabel="peak fraction bias",
        title="Overlap bins: peak fraction bias",
        base_path=layout.figure_base("overlap_bins_peak_fraction_bias"),
    )
    figure_paths["overlap_bins_spike_enrichment_bias"] = plot_overlap_bin_bias(
        bin_summary_df,
        flatten_col="mean_enrichment_flatten_bias",
        intact_col="mean_enrichment_intact_bias",
        peakvalley_col="mean_enrichment_peakvalley_effect",
        ylabel="spike enrichment bias",
        title="Overlap bins: spike enrichment bias",
        base_path=layout.figure_base("overlap_bins_spike_enrichment_bias"),
    )
    figure_paths["overlap_scatter_peakvalley_effect"] = plot_overlap_scatter_peakvalley_effect(
        overlap_df,
        metric=str(cfg.overlap_metric),
        base_path=layout.figure_base("overlap_scatter_peakvalley_effect"),
    )
    figure_paths["baseline_overlap_vs_intact_response"] = plot_baseline_overlap_vs_intact_response(
        overlap_df,
        base_path=layout.figure_base("baseline_overlap_vs_intact_response"),
    )
    return figure_paths


def make_figures(trial_df: pd.DataFrame, paired_df: pd.DataFrame, layout: Any) -> dict[str, dict[str, str]]:
    apply_publication_style()
    figure_paths: dict[str, dict[str, str]] = {}
    figure_paths["peak_spike_fraction_by_condition"] = plot_condition_bar(
        trial_df,
        column="peak_spike_fraction",
        ylabel="peak spike fraction",
        title="Layer 2 peak spike fraction",
        base_path=layout.figure_base("peak_spike_fraction_by_condition"),
    )
    figure_paths["spike_enrichment_by_condition"] = plot_condition_bar(
        trial_df,
        column="spike_enrichment",
        ylabel="peak / nonpeak spike density",
        title="Layer 2 spike enrichment",
        base_path=layout.figure_base("spike_enrichment_by_condition"),
    )
    figure_paths["memory_induced_peak_bias"] = plot_memory_bias(
        paired_df,
        base_path=layout.figure_base("memory_induced_peak_bias"),
    )
    figure_paths["layer2_peak_spike_density_peak_vs_nonpeak"] = plot_density_peak_vs_nonpeak(
        trial_df,
        base_path=layout.figure_base("layer2_peak_spike_density_peak_vs_nonpeak"),
    )
    plt.close("all")
    return figure_paths


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

    log_and_print(log_lines, "[Data] Loading dataset.")
    dataset = load_mnist_skeleton_dataset(str(cfg.dataset_root), split=str(cfg.split))
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    num_classes = int(labels.max()) + 1
    class_index = build_class_index(dataset, num_classes=num_classes)
    trials, sequences_df = build_sequence_trials(labels, flat_normalized, class_index, cfg)
    log_and_print(log_lines, f"[Data] generated {len(trials)} trials across seq_len={list(cfg.sequence_lengths)}")

    log_and_print(log_lines, "[Model] Loading model and encoder.")
    net, encoder = load_model_and_encoder(
        cfg.model_path,
        device=device,
        dt=float(cfg.dt),
        max_duration_ms=float(cfg.max_duration_ms),
    )
    image_ids = [item_id for trial in trials for item_id in trial.ordered_item_ids]
    spike_lookup = build_spike_lookup(images, encoder, image_ids, cfg, device)
    baseline_gain = float(net.layer2.stsp_U)

    trial_rows: list[dict[str, object]] = []
    projection_infos: list[Mapping[str, Any]] = []
    flatten_infos: list[Mapping[str, Any]] = []
    for batch in build_batches(trials, spike_lookup, cfg):
        log_and_print(log_lines, f"[Batch] seq_len={batch.seq_len} batch_id={batch.batch_id} batch_size={len(batch.trials)}")
        initial_snapshot, prefix_by_stage = capture_prefix_snapshots(net, batch, cfg)
        for stage_k, intact_snapshot in sorted(prefix_by_stage.items()):
            target_item_index = int(stage_k) + 1
            target_spikes = batch.item_spikes[target_item_index - 1]
            input_masks = derive_input_region_masks(
                intact_snapshot,
                baseline_gain=baseline_gain,
                epsilon=float(cfg.epsilon),
                peak_q=float(cfg.peak_q),
            )
            output_masks = project_input_masks_to_layer2_outputs(net, intact_snapshot, input_masks, cfg)
            projection_infos.append(output_masks.projection_info)
            flatten_snapshot, flatten_info = apply_all_layers_state_variable_flatten(net, intact_snapshot, input_masks, cfg)
            flatten_infos.append(flatten_info)
            condition_snapshots = {
                "baseline": clone_boundary_snapshot(initial_snapshot),
                "intact_memory": clone_boundary_snapshot(intact_snapshot),
                "flatten_memory": flatten_snapshot,
            }
            for condition in CONDITIONS:
                spike_counts = run_target_and_capture_layer2_spikes(net, condition_snapshots[condition], target_spikes)
                trial_rows.extend(
                    compute_condition_trial_rows(
                        batch,
                        cfg,
                        condition=condition,
                        stage_k=int(stage_k),
                        target_item_index=target_item_index,
                        spike_counts=spike_counts,
                        output_masks=output_masks,
                    )
                )

    trial_df = pd.DataFrame(trial_rows)
    paired_df = compute_paired_effects(trial_df)
    stage_summary_df = build_stage_summary(trial_df)

    sequences_csv = save_tidy_csv(sequences_df, layout.data_file("sequences.csv"), sort_by=["seq_len", "trial_id", "item_index"])
    trial_csv = save_tidy_csv(
        trial_df,
        layout.data_file("layer2_peak_spiking_trial_summary.csv"),
        sort_by=["seq_len", "trial_id", "intervention_stage", "condition"],
    )
    paired_csv = save_tidy_csv(
        paired_df,
        layout.data_file("layer2_peak_spiking_paired_effects.csv"),
        sort_by=["seq_len", "trial_id", "intervention_stage", "target_item_index"],
    )
    stage_csv = save_tidy_csv(
        stage_summary_df,
        layout.data_file("layer2_peak_spiking_stage_summary.csv"),
        sort_by=["condition", "region"],
    )

    figure_paths: dict[str, dict[str, str]] = {}
    overlap_analysis: dict[str, Any] | None = None
    overlap_files: dict[str, str] = {}
    overlap_figure_paths: dict[str, dict[str, str]] = {}
    if not cfg.skip_overlap_analysis:
        overlap_df = compute_overlap_trial_effects(trial_df)
        binned_overlap_df, overlap_bin_summary_df = summarize_overlap_bins(
            overlap_df,
            metric=str(cfg.overlap_metric),
            num_bins=int(cfg.num_overlap_bins),
        )
        overlap_corr_df = compute_overlap_correlations(overlap_df)
        overlap_trial_csv = save_tidy_csv(
            binned_overlap_df,
            layout.data_file("overlap_trial_effects.csv"),
            sort_by=["seq_len", "trial_id", "intervention_stage", "target_item_index"],
        )
        overlap_bin_csv = save_tidy_csv(
            overlap_bin_summary_df,
            layout.data_file("overlap_bin_summary.csv"),
        )
        overlap_corr_csv = save_tidy_csv(
            overlap_corr_df,
            layout.data_file("overlap_correlation_summary.csv"),
        )
        overlap_files = {
            "overlap_trial_effects": overlap_trial_csv,
            "overlap_bin_summary": overlap_bin_csv,
            "overlap_correlation_summary": overlap_corr_csv,
        }
        overlap_analysis = build_overlap_analysis_summary(
            cfg,
            overlap_df,
            overlap_bin_summary_df,
            overlap_corr_df,
            overlap_files,
        )
        if not cfg.skip_figures:
            overlap_figure_paths = make_overlap_figures(binned_overlap_df, overlap_bin_summary_df, cfg, layout)

    if not cfg.skip_figures:
        figure_paths = make_figures(trial_df, paired_df, layout)
        figure_paths.update(overlap_figure_paths)

    exported_files = {
        "sequences": sequences_csv,
        "trial_summary": trial_csv,
        "paired_effects": paired_csv,
        "stage_summary": stage_csv,
        **overlap_files,
    }
    summary = build_summary(cfg, trial_df, paired_df, projection_infos, flatten_infos, overlap_analysis, exported_files, figure_paths)
    summary_path = save_summary_json(summary, layout.root)
    run_config_path = save_run_config(json_safe(asdict(cfg)), layout.root)
    manifest_path = save_summary_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "artifacts": json_safe(
                {
                    **exported_files,
                    "summary": str(summary_path),
                    "run_config": str(run_config_path),
                    "run_info": str(layout.meta_file("run_info.json")),
                    "figures": figure_paths,
                }
            ),
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
