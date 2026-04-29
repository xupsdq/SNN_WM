from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
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
from src.plotting.common.style import DYNAMIC_COLOR, NOISE_COLOR, SAMPLE_COLOR, SHUFFLE_COLOR, STATIC_COLOR


EXPERIMENT_ID = "chunk_stsp_layer1_overlap_peak_formation"
LAYER_KEYS: tuple[str, ...] = ("layer1", "layer2", "layer3")
SMALL_EPS = 1e-12
FIG6D_GROUP_ORDER: tuple[str, ...] = (
    "nonpeak_low_overlap",
    "peak_low_overlap",
    "nonpeak_high_overlap",
    "peak_high_overlap",
)
FIG6D_CONDITION_ORDER: tuple[str, ...] = ("peak_flattened", "intact_final", "peak_boosted")
FIG6D_X_AXIS_GROUPS: tuple[str, ...] = ("low_peak_overlap", "high_peak_overlap")


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
    epsilon: float
    peak_q: float
    multi_hit_threshold: int
    recent_window: int
    overlap_mode: str
    probe_candidate_pool_size: int
    selected_probes_per_trial: int
    probe_overlap_mode: str
    boost_levels: tuple[float, ...]
    include_peak_flatten: bool
    include_peak_boost: bool
    include_low_overlap_probe: bool
    include_matched_nonpeak_probe: bool
    save_element_table: bool
    fast_no_decay_reference: bool
    skip_figures: bool
    smoke: bool
    dt: float = 1.0 * ms
    current_proxy_mode: str = "dynamic_stsp_gain"
    probe_stsp_update_mode: str = "dynamic_after_initial_manipulation"

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
class NetworkSnapshot:
    current_time: int
    layer_input_shapes: dict[str, tuple[int, ...]]
    state_by_layer: dict[str, LayerRuntimeState]


@dataclass(frozen=True)
class InputMasks:
    nonbase: np.ndarray
    peak: np.ndarray
    nonpeak: np.ndarray
    valid: np.ndarray
    invalid_reason: tuple[str, ...]
    num_nonbase: np.ndarray
    num_peak: np.ndarray
    num_nonpeak: np.ndarray


@dataclass(frozen=True)
class OutputMasks:
    mask_mode: str
    peak: np.ndarray
    nonpeak: np.ndarray
    valid: np.ndarray
    invalid_reason: tuple[str, ...]
    num_peak: np.ndarray
    num_nonpeak: np.ndarray
    projection_info: dict[str, Any]


def ms_to_steps(duration_ms: float, dt: float) -> int:
    return int(round((float(duration_ms) * ms) / float(dt)))


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Figure 6 Layer 1 overlap-gated STSP peak formation experiment.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_PATH_CONFIG.model_path))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_PATH_CONFIG.results_root / EXPERIMENT_ID))
    parser.add_argument("--sample-ms", type=float, default=180.0)
    parser.add_argument("--delay-ms", type=float, default=200.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-sequences", type=int, default=128)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[10])
    parser.add_argument("--samples-per-label", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--peak-q", type=float, default=0.20)
    parser.add_argument("--multi-hit-threshold", type=int, default=2)
    parser.add_argument("--recent-window", type=int, default=2)
    parser.add_argument("--overlap-mode", type=str, default="input_spike_binary", choices=["input_spike_binary", "input_spike_count"])
    parser.add_argument("--probe-candidate-pool-size", type=int, default=40)
    parser.add_argument("--selected-probes-per-trial", type=int, default=1)
    parser.add_argument("--probe-overlap-mode", type=str, default="both", choices=["both", "fraction", "enrichment"])
    parser.add_argument("--boost-levels", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    parser.add_argument("--include-peak-flatten", type=str_to_bool, default=True)
    parser.add_argument("--include-peak-boost", type=str_to_bool, default=True)
    parser.add_argument("--include-low-overlap-probe", type=str_to_bool, default=True)
    parser.add_argument("--include-matched-nonpeak-probe", type=str_to_bool, default=True)
    parser.add_argument("--save-element-table", type=str_to_bool, default=True)
    parser.add_argument("--fast-no-decay-reference", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def normalize_config(args: argparse.Namespace) -> ExperimentConfig:
    sequence_lengths = tuple(dict.fromkeys(int(item) for item in args.sequence_lengths))
    boost_levels = tuple(dict.fromkeys(float(item) for item in args.boost_levels))
    if not sequence_lengths:
        raise ValueError("--sequence-lengths must not be empty.")
    if min(sequence_lengths) < 2:
        raise ValueError("--sequence-lengths must be >= 2.")
    if max(sequence_lengths) > 10:
        raise ValueError("--sequence-lengths must be <= 10 because labels are sampled without replacement.")
    if not boost_levels:
        raise ValueError("--boost-levels must not be empty.")
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
        epsilon=float(args.epsilon),
        peak_q=float(args.peak_q),
        multi_hit_threshold=int(args.multi_hit_threshold),
        recent_window=int(args.recent_window),
        overlap_mode=str(args.overlap_mode),
        probe_candidate_pool_size=int(args.probe_candidate_pool_size),
        selected_probes_per_trial=int(args.selected_probes_per_trial),
        probe_overlap_mode=str(args.probe_overlap_mode),
        boost_levels=boost_levels,
        include_peak_flatten=bool(args.include_peak_flatten),
        include_peak_boost=bool(args.include_peak_boost),
        include_low_overlap_probe=bool(args.include_low_overlap_probe),
        include_matched_nonpeak_probe=bool(args.include_matched_nonpeak_probe),
        save_element_table=bool(args.save_element_table),
        fast_no_decay_reference=bool(args.fast_no_decay_reference),
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
                "probe_candidate_pool_size": min(int(cfg.probe_candidate_pool_size), 6),
                "selected_probes_per_trial": 1,
                "boost_levels": (0.0, 1.0),
                "save_element_table": True,
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
    if cfg.multi_hit_threshold < 1 or cfg.recent_window < 1:
        raise ValueError("--multi-hit-threshold and --recent-window must be >= 1.")
    if cfg.probe_candidate_pool_size <= 0 or cfg.selected_probes_per_trial <= 0:
        raise ValueError("--probe-candidate-pool-size and --selected-probes-per-trial must be positive.")
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
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def safe_mean(values: Sequence[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    if np.count_nonzero(valid) == 0:
        return None
    return float(arr[valid].mean())


def finite_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
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


def linear_r2(features: np.ndarray, y: np.ndarray) -> float | None:
    x_arr = np.asarray(features, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    valid = np.isfinite(y_arr) & np.all(np.isfinite(x_arr), axis=1)
    if np.count_nonzero(valid) <= x_arr.shape[1] + 1:
        return None
    x_valid = x_arr[valid]
    y_valid = y_arr[valid]
    if float(np.var(y_valid)) <= 0.0:
        return None
    design = np.column_stack([np.ones(x_valid.shape[0], dtype=np.float64), x_valid])
    beta, *_ = np.linalg.lstsq(design, y_valid, rcond=None)
    pred = design @ beta
    ss_res = float(np.sum((y_valid - pred) ** 2))
    ss_tot = float(np.sum((y_valid - y_valid.mean()) ** 2))
    return float(1.0 - ss_res / (ss_tot + SMALL_EPS))


def linear_betas(features: np.ndarray, y: np.ndarray) -> list[float | None]:
    x_arr = np.asarray(features, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(y_arr) & np.all(np.isfinite(x_arr), axis=1)
    if np.count_nonzero(valid) <= x_arr.shape[1] + 1:
        return [None for _ in range(x_arr.shape[1])]
    design = np.column_stack([np.ones(np.count_nonzero(valid), dtype=np.float64), x_arr[valid]])
    beta, *_ = np.linalg.lstsq(design, y_arr[valid], rcond=None)
    return [float(item) if math.isfinite(float(item)) else None for item in beta[1:]]


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
                    }
                )
            trial_id += 1
    return trials, pd.DataFrame(rows)


def build_probe_candidates(
    trials: Sequence[SequenceTrial],
    labels: np.ndarray,
    cfg: ExperimentConfig,
) -> dict[int, tuple[int, ...]]:
    all_ids = np.arange(labels.shape[0], dtype=np.int64)
    out: dict[int, tuple[int, ...]] = {}
    for trial in trials:
        used = set(int(item) for item in trial.ordered_item_ids)
        available = np.asarray([int(item) for item in all_ids.tolist() if int(item) not in used], dtype=np.int64)
        rng = np.random.default_rng(mix_seed(cfg.seed, int(trial.trial_id), 5333))
        if available.size <= 0:
            raise ValueError("No probe candidates remain after excluding sequence items.")
        chosen = rng.choice(
            available,
            size=min(int(cfg.probe_candidate_pool_size), int(available.size)),
            replace=False,
        )
        out[int(trial.trial_id)] = tuple(int(item) for item in chosen.tolist())
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


def forward_three_layers(net: Any, input_t: torch.Tensor, t_step: int) -> None:
    s1, _ = net.layer1.forward_step(input_t, t_step, training=False, monitor=False, stsp_mode="dynamic")
    s1_p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1_p, t_step, training=False, monitor=False, stsp_mode="dynamic")
    s2_p = net.pool2(s2.float())
    net.layer3.forward_step(s2_p, t_step, labels=None, training=False, monitor=False, stsp_mode="dynamic")


def clone_optional_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().cpu().clone()


def capture_layer_runtime_state(layer: Any) -> LayerRuntimeState:
    return LayerRuntimeState(
        v_mem=layer.v_mem.detach().cpu().clone(),
        g_e=layer.g_e.detach().cpu().clone(),
        res=layer.res.detach().cpu().clone(),
        inh_trace=layer.lateral_inh.inh_trace.detach().cpu().clone(),
        u_pre=clone_optional_tensor(getattr(layer, "u_pre", None)),
        x_pre=clone_optional_tensor(getattr(layer, "x_pre", None)),
        pre_trace=clone_optional_tensor(getattr(layer, "pre_trace", None)),
        input_trace=clone_optional_tensor(getattr(layer, "input_trace", None)),
        eligibility_trace=clone_optional_tensor(getattr(layer, "eligibility_trace", None)),
        firing_times=clone_optional_tensor(getattr(layer, "firing_times", None)),
    )


def snapshot_network(
    net: Any,
    *,
    current_time: int,
    layer_input_shapes: Mapping[str, tuple[int, ...]],
) -> NetworkSnapshot:
    return NetworkSnapshot(
        current_time=int(current_time),
        layer_input_shapes={str(key): tuple(value) for key, value in layer_input_shapes.items()},
        state_by_layer={str(layer_key): capture_layer_runtime_state(getattr(net, layer_key)) for layer_key in LAYER_KEYS},
    )


def copy_tensor_in_place(target: torch.Tensor, source: torch.Tensor) -> None:
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def restore_network_snapshot(net: Any, snapshot: NetworkSnapshot) -> None:
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            getattr(net, layer_key).reset_state(snapshot.layer_input_shapes[layer_key])
        for layer_key, layer_state in snapshot.state_by_layer.items():
            layer = getattr(net, layer_key)
            copy_tensor_in_place(layer.v_mem, layer_state.v_mem)
            copy_tensor_in_place(layer.g_e, layer_state.g_e)
            copy_tensor_in_place(layer.res, layer_state.res)
            copy_tensor_in_place(layer.lateral_inh.inh_trace, layer_state.inh_trace)
            if layer_state.u_pre is not None and getattr(layer, "u_pre", None) is not None:
                copy_tensor_in_place(layer.u_pre, layer_state.u_pre)
            if layer_state.x_pre is not None and getattr(layer, "x_pre", None) is not None:
                copy_tensor_in_place(layer.x_pre, layer_state.x_pre)
            if layer_state.pre_trace is not None and getattr(layer, "pre_trace", None) is not None:
                copy_tensor_in_place(layer.pre_trace, layer_state.pre_trace)
            if layer_state.input_trace is not None and getattr(layer, "input_trace", None) is not None:
                copy_tensor_in_place(layer.input_trace, layer_state.input_trace)
            if layer_state.eligibility_trace is not None and getattr(layer, "eligibility_trace", None) is not None:
                copy_tensor_in_place(layer.eligibility_trace, layer_state.eligibility_trace)
            if layer_state.firing_times is not None and getattr(layer, "firing_times", None) is not None:
                copy_tensor_in_place(layer.firing_times, layer_state.firing_times)


def run_input_window(net: Any, spikes: torch.Tensor, current_time: int) -> int:
    with torch.no_grad():
        for step_idx in range(int(spikes.shape[1])):
            forward_three_layers(net, spikes[:, step_idx, ...], current_time)
            current_time += 1
    return int(current_time)


def run_zero_window(net: Any, zero_input: torch.Tensor, steps: int, current_time: int) -> int:
    with torch.no_grad():
        for _ in range(int(steps)):
            forward_three_layers(net, zero_input, current_time)
            current_time += 1
    return int(current_time)


def layer1_gain_flat(net: Any) -> np.ndarray:
    layer = net.layer1
    if layer.u_pre is None or layer.x_pre is None:
        raise ValueError("Layer 1 is missing STSP u_pre/x_pre.")
    return (layer.u_pre * layer.x_pre).detach().view(layer.u_pre.shape[0], -1).cpu().numpy().astype(np.float32, copy=True)


def layer1_u_x_g_flat_from_snapshot(snapshot: NetworkSnapshot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = snapshot.state_by_layer["layer1"]
    if state.u_pre is None or state.x_pre is None:
        raise ValueError("Layer 1 snapshot is missing STSP state.")
    u = state.u_pre.view(state.u_pre.shape[0], -1).numpy().astype(np.float32, copy=True)
    x = state.x_pre.view(state.x_pre.shape[0], -1).numpy().astype(np.float32, copy=True)
    return u, x, (u * x).astype(np.float32, copy=False)


def derive_final_peak_masks(g_final: np.ndarray, baseline_gain: float, epsilon: float, peak_q: float) -> InputMasks:
    delta = np.asarray(g_final, dtype=np.float64) - float(baseline_gain)
    nonbase = np.abs(delta) > float(epsilon)
    peak = np.zeros_like(nonbase, dtype=bool)
    nonpeak = np.zeros_like(nonbase, dtype=bool)
    valid = np.zeros(delta.shape[0], dtype=bool)
    reasons: list[str] = []
    num_nonbase = nonbase.sum(axis=1).astype(np.int64)
    num_peak = np.zeros(delta.shape[0], dtype=np.int64)
    num_nonpeak = np.zeros(delta.shape[0], dtype=np.int64)
    for row_idx in range(delta.shape[0]):
        row_nonbase = nonbase[row_idx]
        if int(row_nonbase.sum()) <= 0:
            reasons.append("empty_final_nonbase")
            continue
        count_nonbase = int(row_nonbase.sum())
        peak_count = max(1, min(int(math.ceil(count_nonbase * float(peak_q))), count_nonbase))
        candidates = np.flatnonzero(row_nonbase)
        ranked = candidates[np.argsort(delta[row_idx, candidates], kind="stable")]
        chosen = ranked[-peak_count:]
        peak[row_idx, chosen] = True
        nonpeak[row_idx] = row_nonbase & ~peak[row_idx]
        num_peak[row_idx] = int(peak[row_idx].sum())
        num_nonpeak[row_idx] = int(nonpeak[row_idx].sum())
        if num_nonpeak[row_idx] <= 0:
            reasons.append("empty_final_nonpeak")
            continue
        valid[row_idx] = True
        reasons.append("")
    return InputMasks(
        nonbase=nonbase,
        peak=peak,
        nonpeak=nonpeak,
        valid=valid,
        invalid_reason=tuple(reasons),
        num_nonbase=num_nonbase,
        num_peak=num_peak,
        num_nonpeak=num_nonpeak,
    )


def project_layer1_input_masks_to_outputs(
    net: Any,
    snapshot: NetworkSnapshot,
    input_masks: InputMasks,
    cfg: ExperimentConfig,
) -> OutputMasks:
    input_shape = tuple(int(v) for v in snapshot.layer_input_shapes["layer1"][1:])
    output_shape = tuple(int(v) for v in snapshot.state_by_layer["layer1"].v_mem.shape[1:])
    input_flat_size = int(np.prod(input_shape))
    output_flat_size = int(np.prod(output_shape))
    batch_size = int(input_masks.peak.shape[0])
    if input_flat_size == output_flat_size and input_shape == output_shape:
        return OutputMasks(
            mask_mode="direct",
            peak=input_masks.peak.copy(),
            nonpeak=input_masks.nonpeak.copy(),
            valid=input_masks.valid.copy(),
            invalid_reason=input_masks.invalid_reason,
            num_peak=input_masks.num_peak.copy(),
            num_nonpeak=input_masks.num_nonpeak.copy(),
            projection_info={"mask_mode": "direct", "input_shape": input_shape, "output_shape": output_shape},
        )
    weight = net.layer1.kernels.detach().abs().cpu().to(torch.float32)
    peak_out = np.zeros((batch_size, output_flat_size), dtype=bool)
    nonpeak_out = np.zeros((batch_size, output_flat_size), dtype=bool)
    valid = np.zeros(batch_size, dtype=bool)
    reasons: list[str] = []
    num_peak = np.zeros(batch_size, dtype=np.int64)
    num_nonpeak = np.zeros(batch_size, dtype=np.int64)
    for row_idx in range(batch_size):
        if not bool(input_masks.valid[row_idx]):
            reasons.append(input_masks.invalid_reason[row_idx])
            continue
        peak_tensor = torch.as_tensor(input_masks.peak[row_idx].reshape((1, *input_shape)), dtype=torch.float32)
        nonpeak_tensor = torch.as_tensor(input_masks.nonpeak[row_idx].reshape((1, *input_shape)), dtype=torch.float32)
        peak_score = F.conv2d(peak_tensor, weight, stride=int(net.layer1.stride), padding=int(net.layer1.padding)).reshape(-1).numpy()
        nonpeak_score = F.conv2d(nonpeak_tensor, weight, stride=int(net.layer1.stride), padding=int(net.layer1.padding)).reshape(-1).numpy()
        if peak_score.size != output_flat_size:
            raise ValueError(f"Layer 1 projection size mismatch: {peak_score.size} != {output_flat_size}")
        supported = (peak_score + nonpeak_score) > 0.0
        peak_supported = supported & (peak_score >= nonpeak_score) & (peak_score > 0.0)
        nonpeak_supported = supported & ~peak_supported
        if int(peak_supported.sum()) <= 0:
            reasons.append("empty_projected_peak")
            continue
        if int(nonpeak_supported.sum()) <= 0:
            reasons.append("empty_projected_nonpeak")
            continue
        peak_out[row_idx] = peak_supported
        nonpeak_out[row_idx] = nonpeak_supported
        num_peak[row_idx] = int(peak_supported.sum())
        num_nonpeak[row_idx] = int(nonpeak_supported.sum())
        valid[row_idx] = True
        reasons.append("")
    return OutputMasks(
        mask_mode="projected",
        peak=peak_out,
        nonpeak=nonpeak_out,
        valid=valid,
        invalid_reason=tuple(reasons),
        num_peak=num_peak,
        num_nonpeak=num_nonpeak,
        projection_info={
            "mask_mode": "projected",
            "projection_score": "abs_weighted_conv_peak_vs_nonpeak_support",
            "input_shape": input_shape,
            "output_shape": output_shape,
            "layer1_kernel_shape": tuple(int(v) for v in net.layer1.kernels.shape),
            "layer1_stride": int(net.layer1.stride),
            "layer1_padding": int(net.layer1.padding),
        },
    )


def layer1_element_coordinates(input_shape: Sequence[int]) -> list[tuple[int, int, int]]:
    channels, height, width = (int(input_shape[0]), int(input_shape[1]), int(input_shape[2]))
    coords: list[tuple[int, int, int]] = []
    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                coords.append((int(channel), int(y), int(x)))
    return coords


def compute_item_activity(item_spikes: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    counts = item_spikes.detach().sum(dim=1).view(item_spikes.shape[0], -1).cpu().numpy().astype(np.float32, copy=False)
    hits = counts > 0.0
    return hits, counts


def build_element_rows(
    batch: SequenceBatch,
    input_shape: Sequence[int],
    *,
    overlap_count: np.ndarray,
    input_spike_count: np.ndarray,
    update_count: np.ndarray,
    g_final: np.ndarray,
    masks: InputMasks,
    last_input_hit_stage: np.ndarray,
    last_update_stage: np.ndarray,
    cfg: ExperimentConfig,
) -> list[dict[str, object]]:
    coords = layer1_element_coordinates(input_shape)
    delta = np.asarray(g_final, dtype=np.float64) - float(0.2)
    rows: list[dict[str, object]] = []
    for row_idx, trial in enumerate(batch.trials):
        for element_index, (channel, y, x) in enumerate(coords):
            li = float(last_input_hit_stage[row_idx, element_index])
            lu = float(last_update_stage[row_idx, element_index])
            tsi = np.nan if not np.isfinite(li) else float(int(trial.seq_len) - li)
            tsu = np.nan if not np.isfinite(lu) else float(int(trial.seq_len) - lu)
            rows.append(
                {
                    "trial_id": int(trial.trial_id),
                    "seq_len": int(trial.seq_len),
                    "element_index": int(element_index),
                    "input_channel": int(channel),
                    "input_y": int(y),
                    "input_x": int(x),
                    "overlap_count": float(overlap_count[row_idx, element_index]),
                    "input_spike_count": float(input_spike_count[row_idx, element_index]),
                    "update_count": float(update_count[row_idx, element_index]),
                    "final_g": float(g_final[row_idx, element_index]),
                    "final_delta_g": float(delta[row_idx, element_index]),
                    "final_nonbase": int(masks.nonbase[row_idx, element_index]),
                    "final_peak": int(masks.peak[row_idx, element_index]),
                    "final_nonpeak": int(masks.nonpeak[row_idx, element_index]),
                    "last_input_hit_stage": None if not np.isfinite(li) else int(li),
                    "last_update_stage": None if not np.isfinite(lu) else int(lu),
                    "time_since_last_input_hit": None if not np.isfinite(tsi) else float(tsi),
                    "time_since_last_update": None if not np.isfinite(tsu) else float(tsu),
                    "recent_input_hit": int(np.isfinite(tsi) and tsi < int(cfg.recent_window)),
                    "recent_update": int(np.isfinite(tsu) and tsu < int(cfg.recent_window)),
                    "multi_overlap": int(float(overlap_count[row_idx, element_index]) >= int(cfg.multi_hit_threshold)),
                    "multi_update": int(float(update_count[row_idx, element_index]) >= int(cfg.multi_hit_threshold)),
                    "valid": int(bool(masks.valid[row_idx])),
                }
            )
    return rows


def safe_region_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) <= 0:
        return float("nan")
    return float(np.asarray(values, dtype=np.float64)[mask].mean())


def ratio_safe(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
    return float(num / (den + SMALL_EPS))


def build_formation_trial_rows(
    batch: SequenceBatch,
    *,
    overlap_count: np.ndarray,
    update_count: np.ndarray,
    g_final: np.ndarray,
    masks: InputMasks,
    time_since_last_update: np.ndarray,
    cfg: ExperimentConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_idx, trial in enumerate(batch.trials):
        peak = masks.peak[row_idx]
        nonpeak = masks.nonpeak[row_idx]
        multi_overlap = overlap_count[row_idx] >= int(cfg.multi_hit_threshold)
        multi_update = update_count[row_idx] >= int(cfg.multi_hit_threshold)
        peak_frac_multi_overlap = safe_region_mean(multi_overlap.astype(np.float64), peak)
        nonpeak_frac_multi_overlap = safe_region_mean(multi_overlap.astype(np.float64), nonpeak)
        peak_frac_multi_update = safe_region_mean(multi_update.astype(np.float64), peak)
        nonpeak_frac_multi_update = safe_region_mean(multi_update.astype(np.float64), nonpeak)
        rows.append(
            {
                "trial_id": int(trial.trial_id),
                "seq_len": int(trial.seq_len),
                "num_elements": int(g_final.shape[1]),
                "num_nonbase_final": int(masks.num_nonbase[row_idx]),
                "num_peak_final": int(masks.num_peak[row_idx]),
                "mean_overlap_count_peak": safe_region_mean(overlap_count[row_idx], peak),
                "mean_overlap_count_nonpeak": safe_region_mean(overlap_count[row_idx], nonpeak),
                "mean_update_count_peak": safe_region_mean(update_count[row_idx], peak),
                "mean_update_count_nonpeak": safe_region_mean(update_count[row_idx], nonpeak),
                "mean_final_g_peak": safe_region_mean(g_final[row_idx], peak),
                "mean_final_g_nonpeak": safe_region_mean(g_final[row_idx], nonpeak),
                "peak_over_nonpeak_final_g": ratio_safe(safe_region_mean(g_final[row_idx], peak), safe_region_mean(g_final[row_idx], nonpeak)),
                "enrichment_peak_for_multi_overlap": ratio_safe(peak_frac_multi_overlap, nonpeak_frac_multi_overlap),
                "enrichment_peak_for_multi_update": ratio_safe(peak_frac_multi_update, nonpeak_frac_multi_update),
                "corr_overlap_update_count": finite_corr(overlap_count[row_idx], update_count[row_idx]),
                "corr_overlap_count_final_g": finite_corr(overlap_count[row_idx], g_final[row_idx]),
                "corr_update_count_final_g": finite_corr(update_count[row_idx], g_final[row_idx]),
                "corr_time_since_last_update_final_g": finite_corr(time_since_last_update[row_idx], g_final[row_idx]),
                "valid": int(bool(masks.valid[row_idx])),
            }
        )
    return rows


def build_recency_group_rows(
    batch: SequenceBatch,
    *,
    overlap_count: np.ndarray,
    update_count: np.ndarray,
    g_final: np.ndarray,
    masks: InputMasks,
    recent_input_hit: np.ndarray,
    recent_update: np.ndarray,
    cfg: ExperimentConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    variants = (
        ("update_count", update_count, recent_update, "recent_update"),
        ("overlap_count", overlap_count, recent_input_hit, "recent_input_hit"),
    )
    for row_idx, trial in enumerate(batch.trials):
        for group_by, counts, recent_mask, recent_definition in variants:
            row_count = counts[row_idx]
            row_recent = recent_mask[row_idx]
            row_multi = row_count >= int(cfg.multi_hit_threshold)
            groups = {
                "single_old": (~row_multi) & (~row_recent),
                "single_recent": (~row_multi) & row_recent,
                "multi_old": row_multi & (~row_recent),
                "multi_recent": row_multi & row_recent,
            }
            for group_name, group_mask in groups.items():
                group_mask = group_mask & masks.nonbase[row_idx]
                n_elements = int(group_mask.sum())
                peak_fraction = safe_region_mean(masks.peak[row_idx].astype(np.float64), group_mask)
                rows.append(
                    {
                        "trial_id": int(trial.trial_id),
                        "group_by": str(group_by),
                        "recent_definition": str(recent_definition),
                        "group_name": str(group_name),
                        "n_elements": int(n_elements),
                        "mean_final_g": safe_region_mean(g_final[row_idx], group_mask),
                        "mean_final_delta_g": safe_region_mean(g_final[row_idx] - float(0.2), group_mask),
                        "mean_update_count": safe_region_mean(update_count[row_idx], group_mask),
                        "mean_overlap_count": safe_region_mean(overlap_count[row_idx], group_mask),
                        "peak_fraction_in_group": peak_fraction,
                    }
                )
    return rows


def build_prediction_rows(
    batch: SequenceBatch,
    *,
    overlap_count: np.ndarray,
    update_count: np.ndarray,
    g_final: np.ndarray,
    time_since_last_update: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_idx, trial in enumerate(batch.trials):
        features = np.column_stack([overlap_count[row_idx], update_count[row_idx], time_since_last_update[row_idx]])
        beta_overlap, beta_update, beta_recency = linear_betas(features, g_final[row_idx])
        rows.append(
            {
                "trial_id": int(trial.trial_id),
                "seq_len": int(trial.seq_len),
                "corr_overlap_count_final_g": finite_corr(overlap_count[row_idx], g_final[row_idx]),
                "corr_update_count_final_g": finite_corr(update_count[row_idx], g_final[row_idx]),
                "corr_time_since_last_update_final_g": finite_corr(time_since_last_update[row_idx], g_final[row_idx]),
                "beta_overlap_count": beta_overlap,
                "beta_update_count": beta_update,
                "beta_recency": beta_recency,
                "beta_time_since_last_update": beta_recency,
                "r2_overlap_only": linear_r2(overlap_count[row_idx], g_final[row_idx]),
                "r2_update_only": linear_r2(update_count[row_idx], g_final[row_idx]),
                "r2_update_plus_recency": linear_r2(np.column_stack([update_count[row_idx], time_since_last_update[row_idx]]), g_final[row_idx]),
                "valid": 1,
            }
        )
    return rows


def select_probe_rows_for_trial(
    trial: SequenceTrial,
    candidate_ids: Sequence[int],
    labels: np.ndarray,
    spike_lookup: Mapping[int, torch.Tensor],
    peak_mask: np.ndarray,
    nonpeak_mask: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    for candidate_id in candidate_ids:
        spikes = spike_lookup[int(candidate_id)]
        hit = spikes.detach().sum(dim=0).view(-1).cpu().numpy() > 0.0
        spike_count = float(spikes.detach().sum().item())
        denom = float(hit.sum()) + SMALL_EPS
        peak_hits = float(np.logical_and(hit, peak_mask).sum())
        nonpeak_hits = float(np.logical_and(hit, nonpeak_mask).sum())
        peak_fraction = float(peak_hits / denom)
        nonpeak_fraction = float(nonpeak_hits / denom)
        peak_density = float(peak_hits / (float(np.asarray(peak_mask, dtype=bool).sum()) + SMALL_EPS))
        nonpeak_density = float(nonpeak_hits / (float(np.asarray(nonpeak_mask, dtype=bool).sum()) + SMALL_EPS))
        peak_vs_nonpeak = float(peak_density / (nonpeak_density + SMALL_EPS))
        nonpeak_vs_peak = float(nonpeak_density / (peak_density + SMALL_EPS))
        rows.append(
            {
                "trial_id": int(trial.trial_id),
                "candidate_image_id": int(candidate_id),
                "candidate_label": int(labels[int(candidate_id)]),
                "candidate_spike_count": spike_count,
                "probe_peak_overlap_fraction": peak_fraction,
                "probe_nonpeak_overlap_fraction": nonpeak_fraction,
                "probe_peak_overlap_density": peak_density,
                "probe_nonpeak_overlap_density": nonpeak_density,
                "probe_peak_vs_nonpeak_overlap_enrichment": peak_vs_nonpeak,
                "probe_nonpeak_vs_peak_overlap_enrichment": nonpeak_vs_peak,
                "probe_peak_overlap_enrichment": peak_vs_nonpeak,
                "probe_peak_minus_nonpeak_overlap": float(peak_fraction - nonpeak_fraction),
                "probe_nonpeak_minus_peak_overlap": float(nonpeak_fraction - peak_fraction),
                "selected": 0,
                "probe_group": "unselected",
                "target_region": "unselected",
                "overlap_level": "unselected",
                "selection_score": np.nan,
                "duplicate_selected_candidate": 0,
                "valid": 1,
            }
        )
    if not rows:
        return rows, [], {"duplicate_probe_group_selection_count": 0}

    table = pd.DataFrame(rows)
    scores = pd.DataFrame(index=table.index)
    scores["peak_high_overlap"] = (
        table["probe_peak_overlap_fraction"].rank(method="first", ascending=False)
        + table["probe_peak_minus_nonpeak_overlap"].rank(method="first", ascending=False)
    )
    scores["nonpeak_high_overlap"] = (
        table["probe_nonpeak_overlap_fraction"].rank(method="first", ascending=False)
        + table["probe_nonpeak_minus_peak_overlap"].rank(method="first", ascending=False)
    )
    scores["peak_low_overlap"] = table["probe_peak_overlap_fraction"].rank(method="first", ascending=True)
    scores["nonpeak_low_overlap"] = table["probe_nonpeak_overlap_fraction"].rank(method="first", ascending=True)

    selection_specs = (
        ("peak_high_overlap", "peak", "high"),
        ("nonpeak_high_overlap", "nonpeak", "high"),
        ("peak_low_overlap", "peak", "low"),
        ("nonpeak_low_overlap", "nonpeak", "low"),
    )
    used_indices: set[int] = set()
    selected_indices: list[int] = []
    selected_records: list[dict[str, object]] = []
    duplicate_count = 0

    for group_name, target_region, overlap_level in selection_specs:
        order = list(scores.sort_values(group_name, ascending=True, kind="stable").index.astype(int))
        chosen: list[int] = []
        for idx in order:
            if idx in used_indices:
                continue
            chosen.append(int(idx))
            used_indices.add(int(idx))
            if len(chosen) >= int(cfg.selected_probes_per_trial):
                break
        if len(chosen) < int(cfg.selected_probes_per_trial):
            for idx in order:
                if len(chosen) >= int(cfg.selected_probes_per_trial):
                    break
                duplicate_count += 1
                chosen.append(int(idx))
        for idx in chosen:
            row = dict(rows[idx])
            row["selected"] = 1
            row["probe_group"] = group_name
            row["target_region"] = target_region
            row["overlap_level"] = overlap_level
            row["selection_score"] = float(scores.loc[idx, group_name])
            row["duplicate_selected_candidate"] = int(selected_indices.count(int(idx)) > 0)
            selected_indices.append(int(idx))
            selected_records.append(row)

    selected_index_set = set(selected_indices)
    output_rows = [dict(row) for idx, row in enumerate(rows) if idx not in selected_index_set]
    output_rows.extend(selected_records)
    return output_rows, selected_records, {"duplicate_probe_group_selection_count": int(duplicate_count)}


def manipulate_layer1_peak(
    net: Any,
    row_idx: int,
    peak_mask: np.ndarray,
    nonpeak_mask: np.ndarray,
    *,
    condition: str,
    boost_level: float,
) -> float:
    layer = net.layer1
    if layer.u_pre is None or layer.x_pre is None:
        raise ValueError("Layer 1 STSP state is unavailable for manipulation.")
    flat_u = layer.u_pre.view(layer.u_pre.shape[0], -1)
    flat_x = layer.x_pre.view(layer.x_pre.shape[0], -1)
    peak = torch.as_tensor(peak_mask, dtype=torch.bool, device=flat_u.device)
    nonpeak = torch.as_tensor(nonpeak_mask, dtype=torch.bool, device=flat_u.device)
    if int(peak.sum().item()) <= 0 or int(nonpeak.sum().item()) <= 0:
        return float("nan")
    with torch.no_grad():
        mean_u = flat_u[row_idx, nonpeak].mean()
        mean_x = flat_x[row_idx, nonpeak].mean()
        before_u = flat_u[row_idx, peak].clone()
        before_x = flat_x[row_idx, peak].clone()
        if condition == "peak_flattened":
            new_u = torch.full_like(before_u, float(mean_u.item()))
            new_x = torch.full_like(before_x, float(mean_x.item()))
        elif condition == "peak_boosted":
            level = float(boost_level)
            new_u = before_u + level * (before_u - mean_u)
            new_x = before_x + level * (before_x - mean_x)
        else:
            return 0.0
        clipped_u = torch.clamp(new_u, 0.0, 1.0)
        clipped_x = torch.clamp(new_x, 0.0, 1.0)
        clipping = torch.cat([(new_u != clipped_u).float(), (new_x != clipped_x).float()])
        flat_u[row_idx, peak] = clipped_u
        flat_x[row_idx, peak] = clipped_x
    return float(clipping.mean().item()) if clipping.numel() else 0.0


def run_probe_readout(
    net: Any,
    snapshot: NetworkSnapshot,
    probe_spikes_single: torch.Tensor,
    row_idx: int,
    input_peak_mask: np.ndarray,
    input_nonpeak_mask: np.ndarray,
    output_peak_mask: np.ndarray,
    output_nonpeak_mask: np.ndarray,
    cfg: ExperimentConfig,
) -> dict[str, float]:
    restore_network_snapshot(net, snapshot)
    device = probe_spikes_single.device
    batch_size = int(snapshot.layer_input_shapes["layer1"][0])
    channels, height, width = (int(v) for v in snapshot.layer_input_shapes["layer1"][1:])
    probe_batch = torch.zeros((batch_size, int(probe_spikes_single.shape[0]), channels, height, width), dtype=probe_spikes_single.dtype, device=device)
    probe_batch[row_idx] = probe_spikes_single
    peak_in = torch.as_tensor(input_peak_mask, dtype=torch.bool, device=device)
    nonpeak_in = torch.as_tensor(input_nonpeak_mask, dtype=torch.bool, device=device)
    peak_out = torch.as_tensor(output_peak_mask, dtype=torch.bool, device=device)
    nonpeak_out = torch.as_tensor(output_nonpeak_mask, dtype=torch.bool, device=device)
    peak_current = 0.0
    nonpeak_current = 0.0
    voltage_peak_values: list[float] = []
    voltage_nonpeak_values: list[float] = []
    spike_counts_peak = 0.0
    spike_counts_nonpeak = 0.0
    total_spikes = 0.0
    first_peak_latency: float | None = None
    first_nonpeak_latency: float | None = None
    current_time = int(snapshot.current_time)
    with torch.no_grad():
        for step_idx in range(int(probe_batch.shape[1])):
            input_t = probe_batch[:, step_idx, ...]
            spikes, monitor_data = net.layer1.forward_step(
                input_t,
                current_time,
                training=False,
                monitor=True,
                stsp_mode="dynamic",
            )
            gain = monitor_data.get("stsp_gain")
            if gain is None:
                gain = net.layer1.u_pre * net.layer1.x_pre
            row_input = input_t[row_idx].reshape(-1).to(torch.float32)
            row_gain = gain[row_idx].reshape(-1).to(torch.float32)
            peak_current += float((row_input[peak_in] * row_gain[peak_in]).sum().item())
            nonpeak_current += float((row_input[nonpeak_in] * row_gain[nonpeak_in]).sum().item())
            v_flat = monitor_data["v_mem_snapshot"][row_idx].reshape(-1).to(torch.float32)
            s_flat = spikes[row_idx].reshape(-1).to(torch.float32)
            if int(peak_out.sum().item()) > 0:
                voltage_peak_values.append(float(v_flat[peak_out].mean().item()))
                peak_step_spikes = float(s_flat[peak_out].sum().item())
                spike_counts_peak += peak_step_spikes
                if peak_step_spikes > 0.0 and first_peak_latency is None:
                    first_peak_latency = float(step_idx)
            if int(nonpeak_out.sum().item()) > 0:
                voltage_nonpeak_values.append(float(v_flat[nonpeak_out].mean().item()))
                nonpeak_step_spikes = float(s_flat[nonpeak_out].sum().item())
                spike_counts_nonpeak += nonpeak_step_spikes
                if nonpeak_step_spikes > 0.0 and first_nonpeak_latency is None:
                    first_nonpeak_latency = float(step_idx)
            total_spikes += float(s_flat.sum().item())
            current_time += 1
    peak_density_current = peak_current / max(int(np.asarray(input_peak_mask, dtype=bool).sum()), 1)
    nonpeak_density_current = nonpeak_current / max(int(np.asarray(input_nonpeak_mask, dtype=bool).sum()), 1)
    peak_spike_density = spike_counts_peak / max(int(np.asarray(output_peak_mask, dtype=bool).sum()), 1)
    nonpeak_spike_density = spike_counts_nonpeak / max(int(np.asarray(output_nonpeak_mask, dtype=bool).sum()), 1)
    peak_v = float(np.mean(voltage_peak_values)) if voltage_peak_values else float("nan")
    nonpeak_v = float(np.mean(voltage_nonpeak_values)) if voltage_nonpeak_values else float("nan")
    return {
        "peak_current_proxy": float(peak_current),
        "nonpeak_current_proxy": float(nonpeak_current),
        "current_enrichment": float(peak_density_current / (nonpeak_density_current + SMALL_EPS)),
        "peak_supported_voltage_mean": peak_v,
        "nonpeak_supported_voltage_mean": nonpeak_v,
        "voltage_advantage_peak_minus_nonpeak": float(peak_v - nonpeak_v) if np.isfinite([peak_v, nonpeak_v]).all() else float("nan"),
        "total_spike_count": float(total_spikes),
        "peak_supported_spike_count": float(spike_counts_peak),
        "nonpeak_supported_spike_count": float(spike_counts_nonpeak),
        "peak_supported_spike_fraction": float(spike_counts_peak / (total_spikes + SMALL_EPS)),
        "peak_supported_spike_density": float(peak_spike_density),
        "nonpeak_supported_spike_density": float(nonpeak_spike_density),
        "spike_enrichment": float(peak_spike_density / (nonpeak_spike_density + SMALL_EPS)),
        "first_spike_latency_peak_supported": first_peak_latency,
        "first_spike_latency_nonpeak_supported": first_nonpeak_latency,
    }


def run_probe_condition(
    net: Any,
    snapshot: NetworkSnapshot,
    probe_spikes_single: torch.Tensor,
    row_idx: int,
    input_masks: InputMasks,
    output_masks: OutputMasks,
    *,
    condition: str,
    boost_level: float,
    cfg: ExperimentConfig,
) -> tuple[dict[str, float], float, int, str]:
    restore_network_snapshot(net, snapshot)
    clipping_fraction = 0.0
    if condition in {"peak_flattened", "peak_boosted"}:
        clipping_fraction = manipulate_layer1_peak(
            net,
            row_idx,
            input_masks.peak[row_idx],
            input_masks.nonpeak[row_idx],
            condition=condition,
            boost_level=boost_level,
        )
        manipulated_snapshot = snapshot_network(net, current_time=snapshot.current_time, layer_input_shapes=snapshot.layer_input_shapes)
    else:
        manipulated_snapshot = snapshot
    is_valid = bool(input_masks.valid[row_idx]) and bool(output_masks.valid[row_idx])
    invalid_reason = ""
    if not is_valid:
        invalid_reason = input_masks.invalid_reason[row_idx] or output_masks.invalid_reason[row_idx]
        return {}, clipping_fraction, 0, invalid_reason
    metrics = run_probe_readout(
        net,
        manipulated_snapshot,
        probe_spikes_single,
        row_idx,
        input_masks.peak[row_idx],
        input_masks.nonpeak[row_idx],
        output_masks.peak[row_idx],
        output_masks.nonpeak[row_idx],
        cfg,
    )
    return metrics, clipping_fraction, 1, invalid_reason


def build_probe_effects(probe_summary_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trial_id",
        "seq_len",
        "probe_image_id",
        "probe_label",
        "probe_group",
        "target_region",
        "overlap_level",
        "input_peak_overlap_fraction",
        "input_nonpeak_overlap_fraction",
        "delta_current_enrichment_intact_vs_flattened",
        "delta_voltage_advantage_intact_vs_flattened",
        "delta_spike_enrichment_intact_vs_flattened",
        "boost_level",
        "delta_current_enrichment_boost_vs_intact",
        "delta_voltage_advantage_boost_vs_intact",
        "delta_spike_enrichment_boost_vs_intact",
        "delta_current_enrichment_boost_vs_flattened",
        "delta_voltage_advantage_boost_vs_flattened",
        "delta_spike_enrichment_boost_vs_flattened",
    ]
    if probe_summary_df.empty:
        return pd.DataFrame(columns=columns)
    valid_df = probe_summary_df[probe_summary_df["valid"] == 1].copy()
    if valid_df.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    index_cols = [
        "trial_id",
        "seq_len",
        "probe_image_id",
        "probe_label",
        "probe_group",
        "target_region",
        "overlap_level",
        "input_peak_overlap_fraction",
        "input_nonpeak_overlap_fraction",
    ]
    for key, sub in valid_df.groupby(index_cols, dropna=False, sort=False):
        by_cond = {(str(row["condition"]), float(row["boost_level"])): row for _, row in sub.iterrows()}
        intact = by_cond.get(("intact_final", 0.0))
        flattened = by_cond.get(("peak_flattened", 0.0))
        if intact is None:
            continue
        flat_current = float(flattened["current_enrichment"]) if flattened is not None else np.nan
        flat_voltage = float(flattened["voltage_advantage_peak_minus_nonpeak"]) if flattened is not None else np.nan
        flat_spike = float(flattened["spike_enrichment"]) if flattened is not None else np.nan
        (
            trial_id,
            seq_len,
            probe_image_id,
            probe_label,
            probe_group,
            target_region,
            overlap_level,
            peak_overlap_fraction,
            nonpeak_overlap_fraction,
        ) = key
        boost_rows = [row for (condition, _level), row in by_cond.items() if condition == "peak_boosted"]
        if not boost_rows:
            boost_rows = [intact]
        for boost_row in boost_rows:
            boost_level = float(boost_row["boost_level"]) if str(boost_row["condition"]) == "peak_boosted" else 0.0
            rows.append(
                {
                    "trial_id": int(trial_id),
                    "seq_len": int(seq_len),
                    "probe_image_id": int(probe_image_id),
                    "probe_label": int(probe_label),
                    "probe_group": str(probe_group),
                    "target_region": str(target_region),
                    "overlap_level": str(overlap_level),
                    "input_peak_overlap_fraction": float(peak_overlap_fraction),
                    "input_nonpeak_overlap_fraction": float(nonpeak_overlap_fraction),
                    "delta_current_enrichment_intact_vs_flattened": float(float(intact["current_enrichment"]) - flat_current),
                    "delta_voltage_advantage_intact_vs_flattened": float(float(intact["voltage_advantage_peak_minus_nonpeak"]) - flat_voltage),
                    "delta_spike_enrichment_intact_vs_flattened": float(float(intact["spike_enrichment"]) - flat_spike),
                    "boost_level": float(boost_level),
                    "delta_current_enrichment_boost_vs_intact": float(float(boost_row["current_enrichment"]) - float(intact["current_enrichment"])),
                    "delta_voltage_advantage_boost_vs_intact": float(float(boost_row["voltage_advantage_peak_minus_nonpeak"]) - float(intact["voltage_advantage_peak_minus_nonpeak"])),
                    "delta_spike_enrichment_boost_vs_intact": float(float(boost_row["spike_enrichment"]) - float(intact["spike_enrichment"])),
                    "delta_current_enrichment_boost_vs_flattened": float(float(boost_row["current_enrichment"]) - flat_current),
                    "delta_voltage_advantage_boost_vs_flattened": float(float(boost_row["voltage_advantage_peak_minus_nonpeak"]) - flat_voltage),
                    "delta_spike_enrichment_boost_vs_flattened": float(float(boost_row["spike_enrichment"]) - flat_spike),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def aggregate_mean_sem(df: pd.DataFrame, group_cols: Sequence[str], value_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, "mean", "sem", "n"])
    rows: list[dict[str, object]] = []
    for key, sub in df.groupby(list(group_cols), sort=False, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        values = sub[value_col].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        row = {str(col): key_tuple[idx] for idx, col in enumerate(group_cols)}
        row.update(
            {
                "mean": float(finite.mean()) if finite.size else np.nan,
                "sem": sem(finite),
                "n": int(finite.size),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _warn_figure(message: str) -> None:
    print(f"[{EXPERIMENT_ID}] figure warning: {message}", flush=True)


def _insufficient_axis(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()


def cleanup_legacy_main_figure_outputs(figures_dir: Path) -> None:
    old_stems = (
        "overlap_count_map_examples",
        "overlap_vs_update_count",
        "final_peak_enrichment",
        "probe_overlap_selection",
        "peak_function_current",
        "peak_function_voltage",
        "peak_function_spiking",
        "overlap_conditioned_effect",
        "anchor_prediction",
        "update_recency_group_final_g",
    )
    for stem in old_stems:
        for ext in ("png", "pdf", "svg"):
            path = figures_dir / f"{stem}.{ext}"
            if path.exists():
                path.unlink()


def make_fig6B_update_recency(group_df: pd.DataFrame, out_base: Path) -> tuple[dict[str, str], dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    stats: dict[str, Any] = {}
    required = {"trial_id", "group_by", "group_name", "mean_final_g"}
    if group_df.empty or not required.issubset(group_df.columns):
        _warn_figure("Fig6B missing group summary columns.")
        _insufficient_axis(ax, "insufficient data")
        return save_figure_all_formats(fig, out_base), stats

    plot_df = group_df[(group_df["group_by"] == "update_count") & group_df["group_name"].isin(["single_old", "single_recent", "multi_old", "multi_recent"])].copy()
    if plot_df.empty:
        _warn_figure("Fig6B has no update_count groups.")
        _insufficient_axis(ax, "insufficient data")
        return save_figure_all_formats(fig, out_base), stats

    plot_df["update_class"] = np.where(plot_df["group_name"].astype(str).str.startswith("multi"), "Repeated", "Single")
    plot_df["recency_class"] = np.where(plot_df["group_name"].astype(str).str.endswith("recent"), "recent", "old")
    plot_df = plot_df[np.isfinite(plot_df["mean_final_g"].to_numpy(dtype=np.float64))]
    agg = aggregate_mean_sem(plot_df, ["update_class", "recency_class"], "mean_final_g")
    x_lookup = {"old": 0.0, "recent": 1.0}
    colors = {"Single": STATIC_COLOR, "Repeated": DYNAMIC_COLOR}
    offsets = {"Single": -0.035, "Repeated": 0.035}
    for update_class in ("Single", "Repeated"):
        sub = agg[agg["update_class"] == update_class].copy()
        if sub.empty:
            _warn_figure(f"Fig6B missing {update_class} update class.")
            continue
        sub["x"] = sub["recency_class"].map(x_lookup)
        sub = sub.dropna(subset=["x"]).sort_values("x")
        if sub.empty:
            continue
        ax.errorbar(
            sub["x"].to_numpy(dtype=np.float64) + offsets[update_class],
            sub["mean"].to_numpy(dtype=np.float64),
            yerr=sub["sem"].to_numpy(dtype=np.float64),
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=colors[update_class],
            label=update_class,
        )
        point_df = plot_df[plot_df["update_class"] == update_class]
        point_x = point_df["recency_class"].map(x_lookup).to_numpy(dtype=np.float64) + offsets[update_class]
        ax.scatter(point_x, point_df["mean_final_g"], s=18, color=colors[update_class], alpha=0.25, linewidths=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["old", "recent"])
    ax.set_xlabel("Recency")
    ax.set_ylabel("Final STSP gain")
    ax.set_title("Repeated and recent updates create the strongest STSP peaks")
    ax.legend(frameon=False, title="")
    ax.margins(x=0.18)
    return save_figure_all_formats(fig, out_base), stats


def make_fig6C_anchor_prediction(prediction_df: pd.DataFrame, out_base: Path) -> tuple[dict[str, str], dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    stats: dict[str, Any] = {"fig6C_delta_r2_mean": None}
    required = {"trial_id", "r2_overlap_only", "r2_update_plus_recency"}
    if prediction_df.empty or not required.issubset(prediction_df.columns):
        _warn_figure("Fig6C missing prediction summary columns.")
        _insufficient_axis(ax, "insufficient data")
        return save_figure_all_formats(fig, out_base), stats
    plot_df = prediction_df.loc[:, ["trial_id", "r2_overlap_only", "r2_update_plus_recency"]].copy()
    for col in ("r2_overlap_only", "r2_update_plus_recency"):
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df["r2_overlap_only"]) & np.isfinite(plot_df["r2_update_plus_recency"])]
    if plot_df.empty:
        _warn_figure("Fig6C has no finite paired R2 rows.")
        _insufficient_axis(ax, "insufficient data")
        return save_figure_all_formats(fig, out_base), stats
    if len(plot_df) < 2:
        _warn_figure("Fig6C has fewer than 2 paired trials.")
    for _, row in plot_df.iterrows():
        ax.plot([0, 1], [row["r2_overlap_only"], row["r2_update_plus_recency"]], color=NOISE_COLOR, alpha=0.28, linewidth=1.0, zorder=1)
    ax.scatter(np.zeros(len(plot_df)), plot_df["r2_overlap_only"], s=22, color=STATIC_COLOR, alpha=0.45, linewidths=0, zorder=2)
    ax.scatter(np.ones(len(plot_df)), plot_df["r2_update_plus_recency"], s=22, color=DYNAMIC_COLOR, alpha=0.55, linewidths=0, zorder=2)
    means = [float(plot_df["r2_overlap_only"].mean()), float(plot_df["r2_update_plus_recency"].mean())]
    errors = [sem(plot_df["r2_overlap_only"]), sem(plot_df["r2_update_plus_recency"])]
    ax.errorbar([0, 1], means, yerr=errors, fmt="o", markersize=8, capsize=4, linewidth=0, elinewidth=1.7, color="black", zorder=3)
    delta = plot_df["r2_update_plus_recency"] - plot_df["r2_overlap_only"]
    stats["fig6C_delta_r2_mean"] = safe_mean(delta.to_numpy(dtype=np.float64))
    if stats["fig6C_delta_r2_mean"] is not None:
        ax.text(0.5, 0.04, f"mean delta R^2 = {stats['fig6C_delta_r2_mean']:.3f}", transform=ax.transAxes, ha="center", va="bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Overlap only", "Update + recency"])
    ax.set_ylabel("Anchor prediction R^2")
    ax.set_title("Update + recency improves anchor prediction")
    ax.margins(x=0.22)
    return save_figure_all_formats(fig, out_base), stats


def assign_peak_overlap_bin(df: pd.DataFrame, overlap_col: str) -> tuple[pd.DataFrame, float | None]:
    out = df.copy()
    overlap = pd.to_numeric(out[overlap_col], errors="coerce")
    finite = overlap[np.isfinite(overlap)]
    if finite.empty:
        out["peak_overlap_bin"] = np.nan
        return out, None
    median_value = float(finite.median())
    out["peak_overlap_bin"] = np.where(overlap <= median_value, "low_peak_overlap", "high_peak_overlap")
    out.loc[~np.isfinite(overlap), "peak_overlap_bin"] = np.nan
    return out, median_value


def make_fig6D_peak_function_spiking(probe_summary_df: pd.DataFrame, out_base: Path) -> tuple[dict[str, str], dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    stats: dict[str, Any] = {
        "fig6D_plot_type": "peak_overlap_bin_by_peak_strength",
        "fig6D_overlap_bin_method": "median_split",
        "fig6D_condition_order": list(FIG6D_CONDITION_ORDER),
        "fig6D_x_axis_groups": list(FIG6D_X_AXIS_GROUPS),
        "fig6D_max_boost_level_used": None,
        "fig6D_overlap_median": None,
        "fig6D_overlap_field": None,
        "fig6D_mean_spike_enrichment_by_overlap_and_condition": {},
    }
    overlap_col = "input_peak_overlap_fraction" if "input_peak_overlap_fraction" in probe_summary_df.columns else "probe_peak_overlap_fraction"
    required = {"condition", "boost_level", "spike_enrichment", overlap_col}
    if probe_summary_df.empty or not required.issubset(probe_summary_df.columns):
        _warn_figure("Fig6D missing probe summary columns.")
        _insufficient_axis(ax, "insufficient data")
        return save_figure_all_formats(fig, out_base), stats
    valid = probe_summary_df.copy()
    if "valid" in valid.columns:
        valid = valid[valid["valid"].isin([1, True, "1", "True", "true"])].copy()
    if "trial_id" not in valid.columns:
        _warn_figure("Fig6D missing trial_id; falling back to row-level aggregation.")
        valid["trial_id"] = np.arange(len(valid), dtype=np.int64)
    valid["boost_level"] = pd.to_numeric(valid["boost_level"], errors="coerce")
    valid["spike_enrichment"] = pd.to_numeric(valid["spike_enrichment"], errors="coerce")
    valid[overlap_col] = pd.to_numeric(valid[overlap_col], errors="coerce")
    valid = valid[np.isfinite(valid["spike_enrichment"]) & np.isfinite(valid[overlap_col])].copy()
    valid, overlap_median = assign_peak_overlap_bin(valid, overlap_col)
    valid = valid[valid["peak_overlap_bin"].isin(FIG6D_X_AXIS_GROUPS)].copy()
    stats["fig6D_overlap_median"] = overlap_median
    stats["fig6D_overlap_field"] = overlap_col
    if valid.empty:
        _warn_figure("Fig6D has no finite valid rows after peak-overlap binning.")
        _insufficient_axis(ax, "insufficient data")
        return save_figure_all_formats(fig, out_base), stats
    boost_levels = valid.loc[valid["condition"] == "peak_boosted", "boost_level"].dropna()
    max_boost = float(boost_levels.max()) if not boost_levels.empty else np.nan
    stats["fig6D_max_boost_level_used"] = max_boost if np.isfinite(max_boost) else None
    group_order = list(FIG6D_X_AXIS_GROUPS)
    label_map = {"low_peak_overlap": "Low overlap", "high_peak_overlap": "High overlap"}
    condition_specs = [
        ("peak_flattened", 0.0, "Peak flattened", STATIC_COLOR),
        ("intact_final", 0.0, "Intact", SAMPLE_COLOR),
        ("peak_boosted", max_boost, "Peak boosted", DYNAMIC_COLOR),
    ]
    rows: list[pd.DataFrame] = []
    for condition, level, label, _color in condition_specs:
        if not np.isfinite(level):
            _warn_figure(f"Fig6D missing {label}.")
            continue
        sub = valid[valid["condition"] == condition].copy()
        if condition == "peak_boosted":
            sub = sub[np.isclose(sub["boost_level"].to_numpy(dtype=np.float64), float(level))]
        else:
            sub = sub[np.isclose(sub["boost_level"].fillna(0.0).to_numpy(dtype=np.float64), 0.0)]
        if sub.empty:
            _warn_figure(f"Fig6D missing condition {label}.")
            continue
        sub["condition_label"] = label
        sub["condition_key"] = condition
        rows.append(sub)
    if not rows:
        _insufficient_axis(ax, "insufficient data")
        return save_figure_all_formats(fig, out_base), stats
    plot_df = pd.concat(rows, ignore_index=True)
    trial_df = (
        plot_df.groupby(["trial_id", "peak_overlap_bin", "condition_key", "condition_label"], as_index=False)
        .agg(spike_enrichment=("spike_enrichment", "mean"))
    )
    agg = aggregate_mean_sem(trial_df, ["peak_overlap_bin", "condition_label"], "spike_enrichment")
    fig6d_means: dict[str, dict[str, float | None]] = {group: {} for group in group_order}
    for group in group_order:
        for condition, _level, label, _color in condition_specs:
            sub = trial_df[(trial_df["peak_overlap_bin"] == group) & (trial_df["condition_key"] == condition)]
            fig6d_means[group][condition] = safe_mean(sub["spike_enrichment"].to_numpy(dtype=np.float64)) if not sub.empty else None
    stats["fig6D_mean_spike_enrichment_by_overlap_and_condition"] = fig6d_means
    x_lookup = {name: idx for idx, name in enumerate(group_order)}
    for label, color in [("Peak flattened", STATIC_COLOR), ("Intact", SAMPLE_COLOR), ("Peak boosted", DYNAMIC_COLOR)]:
        sub = agg[agg["condition_label"] == label].copy()
        if sub.empty:
            continue
        sub["x"] = sub["peak_overlap_bin"].map(x_lookup)
        sub = sub.dropna(subset=["x"]).sort_values("x")
        ax.errorbar(
            sub["x"].to_numpy(dtype=np.float64),
            sub["mean"].to_numpy(dtype=np.float64),
            yerr=sub["sem"].to_numpy(dtype=np.float64),
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=color,
            label=label,
        )
    ax.set_xticks(np.arange(len(group_order)))
    ax.set_xticklabels([label_map[item] for item in group_order], rotation=0)
    ax.set_xlabel("Probe-peak overlap")
    ax.set_ylabel("Spike enrichment")
    ax.set_title("Peak strength amplifies spiking under probe-peak overlap")
    ax.legend(frameon=False, title="")
    ax.margins(x=0.18)
    return save_figure_all_formats(fig, out_base), stats


def _equal_frequency_bins(x: np.ndarray) -> np.ndarray | None:
    finite_x = x[np.isfinite(x)]
    n_unique = int(np.unique(finite_x).size)
    if finite_x.size < 2 or n_unique < 2:
        return None
    q = 5 if finite_x.size >= 5 and n_unique >= 5 else min(3, n_unique)
    try:
        return pd.qcut(x, q=q, labels=False, duplicates="drop")
    except ValueError:
        ranks = pd.Series(x).rank(method="first")
        return pd.qcut(ranks, q=q, labels=False, duplicates="drop")


def make_fig6E_overlap_conditioned_effect(paired_df: pd.DataFrame, out_base: Path) -> tuple[dict[str, str], dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    stats: dict[str, Any] = {"fig6E_overlap_effect_r": None, "fig6E_overlap_effect_slope": None, "fig6E_n_points": 0}
    required = {"input_peak_overlap_fraction", "delta_spike_enrichment_intact_vs_flattened"}
    if paired_df.empty or not required.issubset(paired_df.columns):
        _warn_figure("Fig6E missing paired effect columns.")
        _insufficient_axis(ax, "insufficient data")
        return save_figure_all_formats(fig, out_base), stats
    plot_df = paired_df.loc[:, list(required)].copy()
    plot_df["input_peak_overlap_fraction"] = pd.to_numeric(plot_df["input_peak_overlap_fraction"], errors="coerce")
    plot_df["delta_spike_enrichment_intact_vs_flattened"] = pd.to_numeric(plot_df["delta_spike_enrichment_intact_vs_flattened"], errors="coerce")
    plot_df = plot_df[
        np.isfinite(plot_df["input_peak_overlap_fraction"])
        & np.isfinite(plot_df["delta_spike_enrichment_intact_vs_flattened"])
    ].copy()
    if plot_df.empty:
        _warn_figure("Fig6E has no finite x/y rows.")
        _insufficient_axis(ax, "insufficient data")
        return save_figure_all_formats(fig, out_base), stats
    x = plot_df["input_peak_overlap_fraction"].to_numpy(dtype=np.float64)
    y = plot_df["delta_spike_enrichment_intact_vs_flattened"].to_numpy(dtype=np.float64)
    stats["fig6E_n_points"] = int(x.size)
    ax.scatter(x, y, s=26, color=DYNAMIC_COLOR, alpha=0.45, linewidths=0)
    ax.axhline(0.0, color=NOISE_COLOR, linewidth=1.0, linestyle="--", alpha=0.8)
    if x.size >= 2 and float(np.std(x)) > 0.0:
        slope, intercept = np.polyfit(x, y, deg=1)
        stats["fig6E_overlap_effect_slope"] = float(slope)
        stats["fig6E_overlap_effect_r"] = finite_corr(x, y)
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), num=100)
        ax.plot(x_line, slope * x_line + intercept, color=DYNAMIC_COLOR, linewidth=1.8)
    else:
        _warn_figure("Fig6E has insufficient x variation for regression.")
    bins = _equal_frequency_bins(x)
    if bins is not None:
        tmp = plot_df.copy()
        tmp["bin"] = bins
        binned = (
            tmp.groupby("bin", as_index=False)
            .agg(
                x_mean=("input_peak_overlap_fraction", "mean"),
                y_mean=("delta_spike_enrichment_intact_vs_flattened", "mean"),
                y_sem=("delta_spike_enrichment_intact_vs_flattened", sem),
            )
            .sort_values("x_mean")
        )
        ax.errorbar(
            binned["x_mean"],
            binned["y_mean"],
            yerr=binned["y_sem"],
            marker="o",
            markersize=7,
            linewidth=1.4,
            capsize=3,
            color="black",
        )
    if stats["fig6E_overlap_effect_r"] is not None:
        ax.text(0.04, 0.94, f"Pearson r = {stats['fig6E_overlap_effect_r']:.3f}", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel("Probe-peak overlap fraction")
    ax.set_ylabel("Delta spike enrichment (intact - flattened)")
    ax.set_title("Peak benefit scales with probe-peak overlap")
    return save_figure_all_formats(fig, out_base), stats


def generate_figures(
    layout: Any,
    *,
    element_df: pd.DataFrame,
    formation_df: pd.DataFrame,
    group_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    probe_selection_df: pd.DataFrame,
    probe_summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
) -> dict[str, Any]:
    apply_publication_style()
    cleanup_legacy_main_figure_outputs(layout.figures_dir)
    main_figures: dict[str, dict[str, str]] = {}
    figure_stats: dict[str, Any] = {}

    paths, stats = make_fig6B_update_recency(group_df, layout.figure_base("fig6B_update_recency_final_g"))
    main_figures["fig6B_update_recency_final_g"] = paths
    figure_stats.update(stats)
    plt.close("all")

    paths, stats = make_fig6C_anchor_prediction(prediction_df, layout.figure_base("fig6C_anchor_prediction_model_comparison"))
    main_figures["fig6C_anchor_prediction_model_comparison"] = paths
    figure_stats.update(stats)
    plt.close("all")

    paths, stats = make_fig6D_peak_function_spiking(probe_summary_df, layout.figure_base("fig6D_peak_function_spiking"))
    main_figures["fig6D_peak_function_spiking"] = paths
    figure_stats.update(stats)
    plt.close("all")

    paths, stats = make_fig6E_overlap_conditioned_effect(paired_df, layout.figure_base("fig6E_overlap_conditioned_spike_effect"))
    main_figures["fig6E_overlap_conditioned_spike_effect"] = paths
    figure_stats.update(stats)
    plt.close("all")

    return {"main_figures": main_figures, "figure_stats": figure_stats}


def rel_files(root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in sorted(root.rglob("*")) if path.is_file()]


def write_artifact_manifest(layout: Any, artifact_paths: Mapping[str, Any]) -> Path:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "files": rel_files(layout.root),
        "artifact_paths": json_safe(artifact_paths),
    }
    path = layout.root / "artifact_manifest.json"
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def summarize_results(
    cfg: ExperimentConfig,
    *,
    formation_df: pd.DataFrame,
    group_df: pd.DataFrame,
    probe_selection_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    probe_summary_df: pd.DataFrame,
    mask_mode_counts: Mapping[str, int],
    artifact_paths: Mapping[str, Any],
    duplicate_probe_group_selection_count: int = 0,
) -> dict[str, Any]:
    group_update = group_df[group_df["group_by"] == "update_count"] if not group_df.empty else pd.DataFrame()

    def group_mean(name: str) -> float | None:
        if group_update.empty:
            return None
        return safe_mean(group_update.loc[group_update["group_name"] == name, "mean_final_g"].to_numpy(dtype=np.float64))

    selected_probe_df = probe_selection_df[probe_selection_df["selected"] == 1].copy() if not probe_selection_df.empty and "selected" in probe_selection_df.columns else pd.DataFrame()
    high_sel = selected_probe_df[selected_probe_df["overlap_level"] == "high"] if not selected_probe_df.empty and "overlap_level" in selected_probe_df.columns else pd.DataFrame()
    low_sel = selected_probe_df[selected_probe_df["overlap_level"] == "low"] if not selected_probe_df.empty and "overlap_level" in selected_probe_df.columns else pd.DataFrame()
    high_effect = paired_df[paired_df["overlap_level"] == "high"] if not paired_df.empty and "overlap_level" in paired_df.columns else pd.DataFrame()
    low_effect = paired_df[paired_df["overlap_level"] == "low"] if not paired_df.empty and "overlap_level" in paired_df.columns else pd.DataFrame()
    probe_group_counts = {
        group: int((selected_probe_df["probe_group"] == group).sum()) if not selected_probe_df.empty and "probe_group" in selected_probe_df.columns else 0
        for group in FIG6D_GROUP_ORDER
    }
    mean_peak_overlap_by_group = {
        group: safe_mean(selected_probe_df.loc[selected_probe_df["probe_group"] == group, "probe_peak_overlap_fraction"].to_numpy(dtype=np.float64))
        if not selected_probe_df.empty and "probe_peak_overlap_fraction" in selected_probe_df.columns
        else None
        for group in FIG6D_GROUP_ORDER
    }
    mean_nonpeak_overlap_by_group = {
        group: safe_mean(selected_probe_df.loc[selected_probe_df["probe_group"] == group, "probe_nonpeak_overlap_fraction"].to_numpy(dtype=np.float64))
        if not selected_probe_df.empty and "probe_nonpeak_overlap_fraction" in selected_probe_df.columns
        else None
        for group in FIG6D_GROUP_ORDER
    }
    fig6d_means: dict[str, dict[str, float | None]] = {}
    if not probe_summary_df.empty and {"probe_group", "condition", "boost_level", "spike_enrichment", "valid"}.issubset(probe_summary_df.columns):
        valid_probe_summary = probe_summary_df[probe_summary_df["valid"] == 1].copy()
        valid_probe_summary["boost_level"] = pd.to_numeric(valid_probe_summary["boost_level"], errors="coerce")
        boost_levels = valid_probe_summary.loc[valid_probe_summary["condition"] == "peak_boosted", "boost_level"].dropna()
        max_boost = float(boost_levels.max()) if not boost_levels.empty else np.nan
        for group in FIG6D_GROUP_ORDER:
            fig6d_means[group] = {}
            for condition in FIG6D_CONDITION_ORDER:
                condition_df = valid_probe_summary[
                    (valid_probe_summary["probe_group"] == group)
                    & (valid_probe_summary["condition"] == condition)
                ]
                if condition == "peak_boosted" and np.isfinite(max_boost):
                    condition_df = condition_df[np.isclose(condition_df["boost_level"].to_numpy(dtype=np.float64), max_boost)]
                elif condition != "peak_boosted":
                    condition_df = condition_df[np.isclose(condition_df["boost_level"].to_numpy(dtype=np.float64), 0.0)]
                fig6d_means[group][condition] = safe_mean(condition_df["spike_enrichment"].to_numpy(dtype=np.float64)) if not condition_df.empty else None
    boost_effect: dict[str, Any] = {}
    clipping: dict[str, Any] = {}
    if not paired_df.empty:
        for level, sub in paired_df.groupby("boost_level", sort=True):
            boost_effect[str(float(level))] = {
                "delta_current_enrichment_boost_vs_intact": safe_mean(sub["delta_current_enrichment_boost_vs_intact"].to_numpy(dtype=np.float64)),
                "delta_voltage_advantage_boost_vs_intact": safe_mean(sub["delta_voltage_advantage_boost_vs_intact"].to_numpy(dtype=np.float64)),
                "delta_spike_enrichment_boost_vs_intact": safe_mean(sub["delta_spike_enrichment_boost_vs_intact"].to_numpy(dtype=np.float64)),
            }
    if not probe_summary_df.empty and "clipping_fraction" in probe_summary_df.columns:
        boosted = probe_summary_df[probe_summary_df["condition"] == "peak_boosted"]
        for level, sub in boosted.groupby("boost_level", sort=True):
            clipping[str(float(level))] = safe_mean(sub["clipping_fraction"].to_numpy(dtype=np.float64))
    figure_payload = artifact_paths.get("figures", {}) if isinstance(artifact_paths, Mapping) else {}
    main_figures = figure_payload.get("main_figures", {}) if isinstance(figure_payload, Mapping) else {}
    figure_stats = figure_payload.get("figure_stats", {}) if isinstance(figure_payload, Mapping) else {}
    return {
        "experiment_id": EXPERIMENT_ID,
        "config": json_safe(asdict(cfg)),
        "number_of_trials": int(len(formation_df)) if not formation_df.empty else 0,
        "valid_trials": int(formation_df["valid"].sum()) if not formation_df.empty and "valid" in formation_df.columns else 0,
        "sequence_lengths": list(cfg.sequence_lengths),
        "sample_ms": float(cfg.sample_ms),
        "delay_ms": float(cfg.delay_ms),
        "peak_q": float(cfg.peak_q),
        "epsilon": float(cfg.epsilon),
        "part_i": {
            "mean_corr_overlap_update_count": safe_mean(formation_df["corr_overlap_update_count"].to_numpy(dtype=np.float64)) if not formation_df.empty and "corr_overlap_update_count" in formation_df.columns else None,
            "mean_corr_update_final_g": safe_mean(formation_df["corr_update_count_final_g"].to_numpy(dtype=np.float64)) if not formation_df.empty else None,
            "mean_corr_overlap_final_g": safe_mean(formation_df["corr_overlap_count_final_g"].to_numpy(dtype=np.float64)) if not formation_df.empty else None,
            "mean_corr_recency_final_g": safe_mean(formation_df["corr_time_since_last_update_final_g"].to_numpy(dtype=np.float64)) if not formation_df.empty else None,
            "mean_final_g_single_old": group_mean("single_old"),
            "mean_final_g_single_recent": group_mean("single_recent"),
            "mean_final_g_multi_old": group_mean("multi_old"),
            "mean_final_g_multi_recent": group_mean("multi_recent"),
            "mean_peak_over_nonpeak_update_count": safe_mean((formation_df["mean_update_count_peak"] / (formation_df["mean_update_count_nonpeak"] + SMALL_EPS)).to_numpy(dtype=np.float64)) if not formation_df.empty else None,
            "mean_peak_over_nonpeak_overlap_count": safe_mean((formation_df["mean_overlap_count_peak"] / (formation_df["mean_overlap_count_nonpeak"] + SMALL_EPS)).to_numpy(dtype=np.float64)) if not formation_df.empty else None,
            "mean_peak_over_nonpeak_final_g": safe_mean(formation_df["peak_over_nonpeak_final_g"].to_numpy(dtype=np.float64)) if not formation_df.empty else None,
        },
        "part_ii": {
            "num_selected_high_overlap_probes": int(len(high_sel)),
            "num_selected_low_overlap_probes": int(len(low_sel)),
            "mean_high_probe_overlap_fraction": safe_mean(high_sel["probe_peak_overlap_fraction"].to_numpy(dtype=np.float64)) if not high_sel.empty else None,
            "mean_low_probe_overlap_fraction": safe_mean(low_sel["probe_peak_overlap_fraction"].to_numpy(dtype=np.float64)) if not low_sel.empty else None,
            "mean_delta_current_enrichment_intact_vs_flattened_high": safe_mean(high_effect["delta_current_enrichment_intact_vs_flattened"].to_numpy(dtype=np.float64)) if not high_effect.empty else None,
            "mean_delta_current_enrichment_intact_vs_flattened_low": safe_mean(low_effect["delta_current_enrichment_intact_vs_flattened"].to_numpy(dtype=np.float64)) if not low_effect.empty else None,
            "mean_delta_voltage_advantage_intact_vs_flattened_high": safe_mean(high_effect["delta_voltage_advantage_intact_vs_flattened"].to_numpy(dtype=np.float64)) if not high_effect.empty else None,
            "mean_delta_voltage_advantage_intact_vs_flattened_low": safe_mean(low_effect["delta_voltage_advantage_intact_vs_flattened"].to_numpy(dtype=np.float64)) if not low_effect.empty else None,
            "mean_delta_spike_enrichment_intact_vs_flattened_high": safe_mean(high_effect["delta_spike_enrichment_intact_vs_flattened"].to_numpy(dtype=np.float64)) if not high_effect.empty else None,
            "mean_delta_spike_enrichment_intact_vs_flattened_low": safe_mean(low_effect["delta_spike_enrichment_intact_vs_flattened"].to_numpy(dtype=np.float64)) if not low_effect.empty else None,
            "mean_boost_effect_by_lambda": boost_effect,
            "clipping_fraction_by_lambda": clipping,
        },
        "main_figures": json_safe(main_figures),
        "figure_stats": json_safe(figure_stats),
        "probe_group_counts": probe_group_counts,
        "mean_peak_overlap_fraction_by_group": mean_peak_overlap_by_group,
        "mean_nonpeak_overlap_fraction_by_group": mean_nonpeak_overlap_by_group,
        "duplicate_probe_group_selection_count": int(duplicate_probe_group_selection_count),
        "fig6D_plot_type": figure_stats.get("fig6D_plot_type", "peak_overlap_bin_by_peak_strength") if isinstance(figure_stats, Mapping) else "peak_overlap_bin_by_peak_strength",
        "fig6D_overlap_bin_method": figure_stats.get("fig6D_overlap_bin_method", "median_split") if isinstance(figure_stats, Mapping) else "median_split",
        "fig6D_condition_order": figure_stats.get("fig6D_condition_order", list(FIG6D_CONDITION_ORDER)) if isinstance(figure_stats, Mapping) else list(FIG6D_CONDITION_ORDER),
        "fig6D_x_axis_groups": figure_stats.get("fig6D_x_axis_groups", list(FIG6D_X_AXIS_GROUPS)) if isinstance(figure_stats, Mapping) else list(FIG6D_X_AXIS_GROUPS),
        "fig6D_max_boost_level_used": figure_stats.get("fig6D_max_boost_level_used") if isinstance(figure_stats, Mapping) else None,
        "fig6D_mean_spike_enrichment_by_overlap_and_condition": json_safe(
            figure_stats.get("fig6D_mean_spike_enrichment_by_overlap_and_condition", {}) if isinstance(figure_stats, Mapping) else {}
        ),
        "mask_mode_counts": dict(mask_mode_counts),
        "artifact_paths": json_safe(artifact_paths),
        "smoke_mode": bool(cfg.smoke),
        "current_proxy_mode": cfg.current_proxy_mode,
        "probe_stsp_update_mode": cfg.probe_stsp_update_mode,
    }


def log_and_print(lines: list[str], message: str) -> None:
    print(message, flush=True)
    lines.append(str(message))


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    seed_everything(int(cfg.seed))
    device = resolve_device(cfg.device)
    layout = prepare_result_layout(cfg.output_dir)
    log_lines: list[str] = []
    run_info = build_run_info(
        experiment_name=EXPERIMENT_ID,
        output_dir=layout.root,
        entry_script=f"python {Path(__file__).as_posix()}",
        seed=int(cfg.seed),
        dataset=str(cfg.dataset_root),
        command=subprocess.list2cmdline(sys.argv),
        model_path=str(cfg.model_path),
    )
    write_run_info(layout.meta_dir, run_info)
    status = "failed"
    try:
        log_and_print(log_lines, f"[{EXPERIMENT_ID}] loading dataset/model on {device}")
        dataset = load_mnist_skeleton_dataset(cfg.dataset_root, cfg.split)
        images, labels, _flat = build_dataset_arrays(dataset)
        class_index = build_class_index(dataset, 10)
        trials, sequence_df = build_sequence_trials(class_index, cfg)
        probe_candidates = build_probe_candidates(trials, labels, cfg)
        all_image_ids: set[int] = set()
        for trial in trials:
            all_image_ids.update(int(item) for item in trial.ordered_item_ids)
            all_image_ids.update(int(item) for item in probe_candidates[int(trial.trial_id)])
        net, encoder = load_model_and_encoder(cfg.model_path, device=device, dt=cfg.dt, max_duration_ms=cfg.max_duration_ms)
        spike_lookup = build_spike_lookup(images, encoder, all_image_ids, cfg, device)
        baseline_gain = float(net.layer1.stsp_U)
        log_and_print(log_lines, f"[{EXPERIMENT_ID}] trials={len(trials)} sample_steps={cfg.sample_steps} delay_steps={cfg.delay_steps}")

        element_rows: list[dict[str, object]] = []
        formation_rows: list[dict[str, object]] = []
        group_rows: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        probe_selection_rows: list[dict[str, object]] = []
        probe_summary_rows: list[dict[str, object]] = []
        mask_mode_counts: dict[str, int] = {}
        projection_info_last: dict[str, Any] = {}
        duplicate_probe_group_selection_count = 0

        for batch in build_batches(trials, spike_lookup, cfg):
            first_sequence = batch.item_spikes[0]
            batch_size, _steps, channels, height, width = first_sequence.shape
            layer_input_shapes = prepare_clean_network_state(net, batch_size, channels, height, width)
            input_shape = tuple(int(v) for v in layer_input_shapes["layer1"][1:])
            flat_size = int(np.prod(input_shape))
            zero_input = torch.zeros((batch_size, channels, height, width), dtype=first_sequence.dtype, device=first_sequence.device)
            overlap_count = np.zeros((batch_size, flat_size), dtype=np.float32)
            input_spike_count = np.zeros((batch_size, flat_size), dtype=np.float32)
            update_count = np.zeros((batch_size, flat_size), dtype=np.float32)
            last_input_hit_stage = np.full((batch_size, flat_size), np.nan, dtype=np.float32)
            last_update_stage = np.full((batch_size, flat_size), np.nan, dtype=np.float32)
            current_time = 0
            final_snapshot: NetworkSnapshot | None = None

            for item_pos, item_spikes in enumerate(batch.item_spikes, start=1):
                item_hits, item_counts = compute_item_activity(item_spikes)
                overlap_count += item_hits.astype(np.float32)
                input_spike_count += item_counts.astype(np.float32)
                last_input_hit_stage[item_hits] = float(item_pos)
                pre_snapshot = snapshot_network(net, current_time=current_time, layer_input_shapes=layer_input_shapes)
                current_time = run_input_window(net, item_spikes, current_time)
                current_time = run_zero_window(net, zero_input, int(cfg.delay_steps), current_time)
                actual_g = layer1_gain_flat(net)
                actual_snapshot = snapshot_network(net, current_time=current_time, layer_input_shapes=layer_input_shapes)
                if cfg.fast_no_decay_reference:
                    decay_g = pre_snapshot.state_by_layer["layer1"].u_pre.view(batch_size, -1).numpy() * pre_snapshot.state_by_layer["layer1"].x_pre.view(batch_size, -1).numpy()
                else:
                    restore_network_snapshot(net, pre_snapshot)
                    decay_time = int(pre_snapshot.current_time)
                    zero_seq = zero_input.unsqueeze(1).expand(batch_size, int(item_spikes.shape[1]), channels, height, width).contiguous()
                    decay_time = run_input_window(net, zero_seq, decay_time)
                    decay_time = run_zero_window(net, zero_input, int(cfg.delay_steps), decay_time)
                    decay_g = layer1_gain_flat(net)
                delta = actual_g - decay_g
                update_hits = delta > float(cfg.epsilon)
                update_count += update_hits.astype(np.float32)
                last_update_stage[update_hits] = float(item_pos)
                restore_network_snapshot(net, actual_snapshot)
                current_time = int(actual_snapshot.current_time)
                final_snapshot = actual_snapshot

            if final_snapshot is None:
                continue
            _u_final, _x_final, g_final = layer1_u_x_g_flat_from_snapshot(final_snapshot)
            input_masks = derive_final_peak_masks(g_final, baseline_gain=baseline_gain, epsilon=cfg.epsilon, peak_q=cfg.peak_q)
            output_masks = project_layer1_input_masks_to_outputs(net, final_snapshot, input_masks, cfg)
            mask_mode_counts[output_masks.mask_mode] = mask_mode_counts.get(output_masks.mask_mode, 0) + int(batch_size)
            projection_info_last = output_masks.projection_info
            time_since_last_input = np.where(np.isfinite(last_input_hit_stage), int(batch.seq_len) - last_input_hit_stage, np.nan)
            time_since_last_update = np.where(np.isfinite(last_update_stage), int(batch.seq_len) - last_update_stage, np.nan)
            recent_input = np.isfinite(time_since_last_input) & (time_since_last_input < int(cfg.recent_window))
            recent_update = np.isfinite(time_since_last_update) & (time_since_last_update < int(cfg.recent_window))

            if cfg.save_element_table:
                element_rows.extend(
                    build_element_rows(
                        batch,
                        input_shape,
                        overlap_count=overlap_count,
                        input_spike_count=input_spike_count,
                        update_count=update_count,
                        g_final=g_final,
                        masks=input_masks,
                        last_input_hit_stage=last_input_hit_stage,
                        last_update_stage=last_update_stage,
                        cfg=cfg,
                    )
                )
            formation_rows.extend(
                build_formation_trial_rows(
                    batch,
                    overlap_count=overlap_count,
                    update_count=update_count,
                    g_final=g_final,
                    masks=input_masks,
                    time_since_last_update=time_since_last_update,
                    cfg=cfg,
                )
            )
            group_rows.extend(
                build_recency_group_rows(
                    batch,
                    overlap_count=overlap_count,
                    update_count=update_count,
                    g_final=g_final,
                    masks=input_masks,
                    recent_input_hit=recent_input,
                    recent_update=recent_update,
                    cfg=cfg,
                )
            )
            prediction_rows.extend(
                build_prediction_rows(
                    batch,
                    overlap_count=overlap_count,
                    update_count=update_count,
                    g_final=g_final,
                    time_since_last_update=time_since_last_update,
                )
            )

            for row_idx, trial in enumerate(batch.trials):
                candidate_rows, selected_rows, selection_info = select_probe_rows_for_trial(
                    trial,
                    probe_candidates[int(trial.trial_id)],
                    labels,
                    spike_lookup,
                    input_masks.peak[row_idx],
                    input_masks.nonpeak[row_idx],
                    cfg,
                )
                duplicate_probe_group_selection_count += int(selection_info.get("duplicate_probe_group_selection_count", 0))
                probe_selection_rows.extend(candidate_rows)
                for selected in selected_rows:
                    probe_id = int(selected["candidate_image_id"])
                    conditions: list[tuple[str, float]] = [("intact_final", 0.0)]
                    if cfg.include_peak_flatten:
                        conditions.append(("peak_flattened", 0.0))
                    if cfg.include_peak_boost:
                        conditions.extend(("peak_boosted", float(level)) for level in cfg.boost_levels)
                    for condition, boost_level in conditions:
                        metrics, clipping_fraction, valid, invalid_reason = run_probe_condition(
                            net,
                            final_snapshot,
                            spike_lookup[probe_id],
                            row_idx,
                            input_masks,
                            output_masks,
                            condition=condition,
                            boost_level=boost_level,
                            cfg=cfg,
                        )
                        row = {
                            "trial_id": int(trial.trial_id),
                            "seq_len": int(trial.seq_len),
                            "probe_image_id": int(probe_id),
                            "probe_label": int(labels[probe_id]),
                            "probe_group": str(selected["probe_group"]),
                            "target_region": str(selected["target_region"]),
                            "overlap_level": str(selected["overlap_level"]),
                            "condition": str(condition),
                            "boost_level": float(boost_level),
                            "input_peak_overlap_fraction": float(selected["probe_peak_overlap_fraction"]),
                            "input_nonpeak_overlap_fraction": float(selected["probe_nonpeak_overlap_fraction"]),
                            "input_peak_overlap_enrichment": float(selected["probe_peak_overlap_enrichment"]),
                            "input_peak_vs_nonpeak_overlap_enrichment": float(selected["probe_peak_vs_nonpeak_overlap_enrichment"]),
                            "input_nonpeak_vs_peak_overlap_enrichment": float(selected["probe_nonpeak_vs_peak_overlap_enrichment"]),
                            "clipping_fraction": float(clipping_fraction) if np.isfinite(clipping_fraction) else np.nan,
                            "valid": int(valid),
                            "invalid_reason": str(invalid_reason),
                            "mask_mode": str(output_masks.mask_mode),
                        }
                        for metric_name in (
                            "peak_current_proxy",
                            "nonpeak_current_proxy",
                            "current_enrichment",
                            "peak_supported_voltage_mean",
                            "nonpeak_supported_voltage_mean",
                            "voltage_advantage_peak_minus_nonpeak",
                            "total_spike_count",
                            "peak_supported_spike_count",
                            "nonpeak_supported_spike_count",
                            "peak_supported_spike_fraction",
                            "peak_supported_spike_density",
                            "nonpeak_supported_spike_density",
                            "spike_enrichment",
                            "first_spike_latency_peak_supported",
                            "first_spike_latency_nonpeak_supported",
                        ):
                            row[metric_name] = metrics.get(metric_name, np.nan)
                        probe_summary_rows.append(row)
            log_and_print(log_lines, f"[{EXPERIMENT_ID}] finished batch {batch.batch_id} seq_len={batch.seq_len}")

        sequence_csv = save_tidy_csv(sequence_df, layout.data_file("layer1_sequence_items.csv"), sort_by=["trial_id", "item_index"])
        element_df = pd.DataFrame(element_rows)
        formation_df = pd.DataFrame(formation_rows)
        group_df = pd.DataFrame(group_rows)
        prediction_df = pd.DataFrame(prediction_rows)
        probe_selection_df = pd.DataFrame(probe_selection_rows)
        probe_summary_df = pd.DataFrame(probe_summary_rows)
        paired_df = build_probe_effects(probe_summary_df)

        artifact_paths: dict[str, Any] = {
            "layer1_sequence_items": sequence_csv,
            "projection_info": projection_info_last,
        }
        if cfg.save_element_table:
            artifact_paths["layer1_overlap_update_element_summary"] = save_tidy_csv(
                element_df,
                layout.data_file("layer1_overlap_update_element_summary.csv"),
                sort_by=["trial_id", "element_index"] if not element_df.empty else None,
            )
        artifact_paths["layer1_peak_formation_trial_summary"] = save_tidy_csv(
            formation_df,
            layout.data_file("layer1_peak_formation_trial_summary.csv"),
            sort_by=["trial_id"] if not formation_df.empty else None,
        )
        artifact_paths["layer1_recency_update_group_summary"] = save_tidy_csv(
            group_df,
            layout.data_file("layer1_recency_update_group_summary.csv"),
            sort_by=["trial_id", "group_by", "group_name"] if not group_df.empty else None,
        )
        artifact_paths["layer1_anchor_prediction_summary"] = save_tidy_csv(
            prediction_df,
            layout.data_file("layer1_anchor_prediction_summary.csv"),
            sort_by=["trial_id"] if not prediction_df.empty else None,
        )
        artifact_paths["layer1_probe_selection"] = save_tidy_csv(
            probe_selection_df,
            layout.data_file("layer1_probe_selection.csv"),
            sort_by=["trial_id", "selected", "probe_group"] if not probe_selection_df.empty else None,
        )
        artifact_paths["layer1_peak_function_probe_summary"] = save_tidy_csv(
            probe_summary_df,
            layout.data_file("layer1_peak_function_probe_summary.csv"),
            sort_by=["trial_id", "probe_image_id", "condition", "boost_level"] if not probe_summary_df.empty else None,
        )
        artifact_paths["layer1_peak_function_paired_effects"] = save_tidy_csv(
            paired_df,
            layout.data_file("layer1_peak_function_paired_effects.csv"),
            sort_by=["trial_id", "probe_image_id", "boost_level"] if not paired_df.empty else None,
        )

        if not cfg.skip_figures:
            artifact_paths["figures"] = generate_figures(
                layout,
                element_df=element_df,
                formation_df=formation_df,
                group_df=group_df,
                prediction_df=prediction_df,
                probe_selection_df=probe_selection_df,
                probe_summary_df=probe_summary_df,
                paired_df=paired_df,
            )

        summary = summarize_results(
            cfg,
            formation_df=formation_df,
            group_df=group_df,
            probe_selection_df=probe_selection_df,
            paired_df=paired_df,
            probe_summary_df=probe_summary_df,
            mask_mode_counts=mask_mode_counts,
            artifact_paths=artifact_paths,
            duplicate_probe_group_selection_count=duplicate_probe_group_selection_count,
        )
        artifact_paths["run_config"] = str(save_run_config(json_safe(asdict(cfg)), layout.root))
        artifact_paths["summary"] = str(save_summary_json(summary, layout.root))
        save_summary_json({"projection_info": projection_info_last, "mask_mode_counts": mask_mode_counts}, layout.meta_dir, filename="projection_info.json")
        save_log_lines(log_lines, layout.logs_dir, filename="run.log")
        manifest_path = write_artifact_manifest(layout, artifact_paths)
        summary["artifact_paths"]["artifact_manifest"] = str(manifest_path)
        save_summary_json(summary, layout.root)
        status = "success"
        return summary
    finally:
        finalize_run_info(layout.meta_dir, run_info, status=status)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    cfg = normalize_config(parser.parse_args(argv))
    run_experiment(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
