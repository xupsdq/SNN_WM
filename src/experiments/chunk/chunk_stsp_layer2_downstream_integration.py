from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_run_config, save_summary_json
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.seed import mix_seed
from src.plotting.common.io import apply_publication_style, save_figure_all_formats, save_tidy_csv
from src.plotting.common.style import DYNAMIC_COLOR, NOISE_COLOR, SAMPLE_COLOR, STATIC_COLOR


EXPERIMENT_ID = "chunk_stsp_layer2_downstream_integration"
LAYER_KEYS: tuple[str, ...] = ("layer1", "layer2", "layer3")
L1_CONDITIONS: tuple[str, ...] = ("l1_peak_flattened", "l1_intact", "l1_peak_boosted")
L2_MEMORY_CONDITIONS: tuple[str, ...] = ("l2_reset", "l2_intact")
PROBE_GROUPS: tuple[str, ...] = ("low_peak_overlap", "high_peak_overlap")
SMALL_EPS = 1e-12
L2_RESET_SCOPE = "layer2_only_runtime_and_stsp"


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
    sequence_length: int
    samples_per_label: int
    epsilon: float
    peak_q: float
    boost_level: float
    probe_candidate_pool_size: int
    selected_probes_per_trial: int
    probe_overlap_bin_method: str
    save_update_maps: bool
    skip_figures: bool
    smoke: bool
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
    trial_index: int
    seq_len: int
    ordered_item_ids: tuple[int, ...]
    ordered_item_labels: tuple[int, ...]
    sequence_seed: int


@dataclass(frozen=True)
class SequenceBatch:
    batch_id: int
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
    empty_mask: torch.Tensor | None


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
    mean_g_peak: np.ndarray
    mean_g_nonpeak: np.ndarray
    peak_nonpeak_ratio: np.ndarray


@dataclass(frozen=True)
class RegionMasks:
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
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fig6 Layer 2 downstream STSP integration experiment.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_PATH_CONFIG.model_path))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_PATH_CONFIG.results_root / EXPERIMENT_ID))
    parser.add_argument("--sample-ms", type=float, default=180.0)
    parser.add_argument("--delay-ms", type=float, default=200.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-sequences", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--samples-per-label", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--peak-q", type=float, default=0.20)
    parser.add_argument("--boost-level", type=float, default=1.0)
    parser.add_argument("--probe-candidate-pool-size", type=int, default=40)
    parser.add_argument("--selected-probes-per-trial", type=int, default=1)
    parser.add_argument("--probe-overlap-bin-method", type=str, default="median", choices=["median", "rank"])
    parser.add_argument("--save-update-maps", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def normalize_config(args: argparse.Namespace) -> ExperimentConfig:
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
        sequence_length=int(args.sequence_length),
        samples_per_label=int(args.samples_per_label),
        epsilon=float(args.epsilon),
        peak_q=float(args.peak_q),
        boost_level=float(args.boost_level),
        probe_candidate_pool_size=int(args.probe_candidate_pool_size),
        selected_probes_per_trial=int(args.selected_probes_per_trial),
        probe_overlap_bin_method=str(args.probe_overlap_bin_method),
        save_update_maps=bool(args.save_update_maps),
        skip_figures=bool(args.skip_figures),
        smoke=bool(args.smoke),
    )
    if cfg.smoke:
        smoke_name = "smoke_skip_figures_check" if cfg.skip_figures else "smoke"
        cfg = ExperimentConfig(
            **{
                **asdict(cfg),
                "output_dir": str(Path(cfg.output_dir) / smoke_name),
                "max_sequences": min(int(cfg.max_sequences), 4),
                "batch_size": min(int(cfg.batch_size), 2),
                "sequence_length": min(int(cfg.sequence_length), 6),
                "sample_ms": min(float(cfg.sample_ms), 15.0),
                "delay_ms": min(float(cfg.delay_ms), 10.0),
                "probe_candidate_pool_size": min(int(cfg.probe_candidate_pool_size), 6),
                "selected_probes_per_trial": 1,
                "boost_level": 1.0,
            }
        )
    if cfg.sequence_length < 2:
        raise ValueError("--sequence-length must be >= 2.")
    if cfg.sequence_length > 10:
        raise ValueError("--sequence-length must be <= 10 for the 10-label MNIST sequence design.")
    if min(cfg.batch_size, cfg.max_sequences, cfg.samples_per_label, cfg.probe_candidate_pool_size, cfg.selected_probes_per_trial) <= 0:
        raise ValueError("batch, sequence, sample, candidate, and selected-probe counts must be positive.")
    if not (0.0 < cfg.peak_q < 1.0):
        raise ValueError("--peak-q must be in (0, 1).")
    if cfg.epsilon < 0.0:
        raise ValueError("--epsilon must be non-negative.")
    if cfg.sample_steps <= 0 or cfg.delay_steps <= 0:
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
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def finite_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(valid.sum()) < 2:
        return None
    x_valid = x_arr[valid]
    y_valid = y_arr[valid]
    if float(np.std(x_valid)) <= 0.0 or float(np.std(y_valid)) <= 0.0:
        return None
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def safe_mean(values: Sequence[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def ratio_safe(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
    return float(num / (den + SMALL_EPS))


def region_mean(values: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask, dtype=bool)
    if int(mask_bool.sum()) <= 0:
        return float("nan")
    return float(np.asarray(values, dtype=np.float64)[mask_bool].mean())


def log_and_print(lines: list[str], message: str) -> None:
    print(message, flush=True)
    lines.append(str(message))


def build_label_candidate_pools(class_index: Mapping[int, Sequence[int]], cfg: ExperimentConfig) -> dict[int, np.ndarray]:
    pools: dict[int, np.ndarray] = {}
    for label in sorted(class_index):
        rng = np.random.default_rng(mix_seed(cfg.seed, int(label), 211))
        ids = np.asarray([int(idx) for idx in class_index[int(label)]], dtype=np.int64)
        permuted = rng.permutation(ids)
        limit = min(int(cfg.samples_per_label), int(permuted.size))
        pools[int(label)] = permuted[:limit].astype(np.int64, copy=False)
        if pools[int(label)].size <= 0:
            raise ValueError(f"Label {label} has no available samples.")
    return pools


def build_sequence_trials(
    class_index: Mapping[int, Sequence[int]],
    labels: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[list[SequenceTrial], pd.DataFrame]:
    label_pools = build_label_candidate_pools(class_index, cfg)
    all_labels = np.asarray(sorted(label_pools.keys()), dtype=np.int64)
    trials: list[SequenceTrial] = []
    rows: list[dict[str, object]] = []
    for trial_id in range(int(cfg.max_sequences)):
        trial_seed = mix_seed(cfg.seed, int(trial_id), int(cfg.sequence_length), 607)
        rng = np.random.default_rng(trial_seed)
        chosen_labels = rng.choice(all_labels, size=int(cfg.sequence_length), replace=False)
        chosen_ids = np.asarray([int(rng.choice(label_pools[int(label)])) for label in chosen_labels], dtype=np.int64)
        order = rng.permutation(int(cfg.sequence_length))
        ordered_ids = chosen_ids[order]
        ordered_labels = np.asarray(labels[ordered_ids], dtype=np.int64)
        trial = SequenceTrial(
            trial_id=int(trial_id),
            trial_index=int(trial_id),
            seq_len=int(cfg.sequence_length),
            ordered_item_ids=tuple(int(item) for item in ordered_ids.tolist()),
            ordered_item_labels=tuple(int(item) for item in ordered_labels.tolist()),
            sequence_seed=int(trial_seed),
        )
        trials.append(trial)
        ids_text = "|".join(str(int(item)) for item in trial.ordered_item_ids)
        labels_text = "|".join(str(int(item)) for item in trial.ordered_item_labels)
        for item_index, (image_id, item_label) in enumerate(zip(trial.ordered_item_ids, trial.ordered_item_labels), start=1):
            rows.append(
                {
                    "trial_id": int(trial.trial_id),
                    "trial_index": int(trial.trial_index),
                    "seq_len": int(trial.seq_len),
                    "item_index": int(item_index),
                    "image_id": int(image_id),
                    "item_label": int(item_label),
                    "ordered_item_ids": ids_text,
                    "ordered_item_labels": labels_text,
                    "sequence_seed": int(trial.sequence_seed),
                }
            )
    return trials, pd.DataFrame(rows)


def build_probe_candidates(trials: Sequence[SequenceTrial], labels: np.ndarray, cfg: ExperimentConfig) -> dict[int, tuple[int, ...]]:
    all_ids = np.arange(int(labels.shape[0]), dtype=np.int64)
    out: dict[int, tuple[int, ...]] = {}
    for trial in trials:
        excluded = set(int(item) for item in trial.ordered_item_ids)
        available = np.asarray([int(idx) for idx in all_ids if int(idx) not in excluded], dtype=np.int64)
        rng = np.random.default_rng(mix_seed(cfg.seed, int(trial.trial_id), 9919))
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


def build_batches(trials: Sequence[SequenceTrial], spike_lookup: Mapping[int, torch.Tensor], cfg: ExperimentConfig) -> Iterable[SequenceBatch]:
    for batch_id, start in enumerate(range(0, len(trials), int(cfg.batch_size))):
        batch_trials = tuple(trials[start : start + int(cfg.batch_size)])
        item_spikes = tuple(
            torch.stack([spike_lookup[int(trial.ordered_item_ids[item_pos])] for trial in batch_trials], dim=0)
            for item_pos in range(int(cfg.sequence_length))
        )
        yield SequenceBatch(batch_id=int(batch_id), trials=batch_trials, item_spikes=item_spikes)


def build_layer_input_shapes(net: Any, batch_size: int, channels: int, height: int, width: int) -> dict[str, tuple[int, ...]]:
    h1 = (height + 2 * int(net.layer1.padding) - int(net.layer1.kernel_size)) // int(net.layer1.stride) + 1
    w1 = (width + 2 * int(net.layer1.padding) - int(net.layer1.kernel_size)) // int(net.layer1.stride) + 1
    h1_p, w1_p = h1 // 2, w1 // 2
    h2 = (h1_p + 2 * int(net.layer2.padding) - int(net.layer2.kernel_size)) // int(net.layer2.stride) + 1
    w2 = (w1_p + 2 * int(net.layer2.padding) - int(net.layer2.kernel_size)) // int(net.layer2.stride) + 1
    h2_p, w2_p = h2 // 2, w2 // 2
    return {
        "layer1": (int(batch_size), int(channels), int(height), int(width)),
        "layer2": (int(batch_size), int(net.layer1.out_channels), int(h1_p), int(w1_p)),
        "layer3": (int(batch_size), int(net.layer2.out_channels), int(h2_p), int(w2_p)),
    }


def prepare_clean_network_state(net: Any, batch_size: int, channels: int, height: int, width: int) -> dict[str, tuple[int, ...]]:
    layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            getattr(net, layer_key).reset_state(layer_input_shapes[layer_key])
    return {str(key): tuple(value) for key, value in layer_input_shapes.items()}


def forward_three_layers_capture(net: Any, input_t: torch.Tensor, t_step: int) -> tuple[torch.Tensor, torch.Tensor]:
    s1, _ = net.layer1.forward_step(input_t, t_step, training=False, monitor=False, stsp_mode="dynamic")
    s1_p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1_p, t_step, training=False, monitor=False, stsp_mode="dynamic")
    s2_p = net.pool2(s2.float())
    net.layer3.forward_step(s2_p, t_step, labels=None, training=False, monitor=False, stsp_mode="dynamic")
    return s1, s2


def run_input_window(net: Any, spikes: torch.Tensor, current_time: int) -> int:
    with torch.no_grad():
        for step_idx in range(int(spikes.shape[1])):
            forward_three_layers_capture(net, spikes[:, step_idx, ...], current_time)
            current_time += 1
    return int(current_time)


def run_zero_window(net: Any, zero_input: torch.Tensor, steps: int, current_time: int) -> int:
    with torch.no_grad():
        for _ in range(int(steps)):
            forward_three_layers_capture(net, zero_input, current_time)
            current_time += 1
    return int(current_time)


def clone_optional_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().cpu().clone()


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
        empty_mask=clone_optional_tensor(getattr(layer, "empty_mask", None)),
    )


def snapshot_model_state(net: Any, *, current_time: int, layer_input_shapes: Mapping[str, tuple[int, ...]]) -> NetworkSnapshot:
    return NetworkSnapshot(
        current_time=int(current_time),
        layer_input_shapes={str(key): tuple(int(v) for v in value) for key, value in layer_input_shapes.items()},
        state_by_layer={str(layer_key): capture_layer_runtime_state(getattr(net, layer_key)) for layer_key in LAYER_KEYS},
    )


def copy_tensor_in_place(target: torch.Tensor, source: torch.Tensor) -> None:
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def restore_model_state(net: Any, snapshot: NetworkSnapshot) -> None:
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
            if layer_state.empty_mask is not None and getattr(layer, "empty_mask", None) is not None:
                copy_tensor_in_place(layer.empty_mask, layer_state.empty_mask)


def layer_gain_flat(net: Any, layer_key: str) -> np.ndarray:
    layer = getattr(net, layer_key)
    if layer.u_pre is None or layer.x_pre is None:
        raise ValueError(f"{layer_key} is missing STSP u_pre/x_pre.")
    return (layer.u_pre * layer.x_pre).detach().view(layer.u_pre.shape[0], -1).cpu().numpy().astype(np.float32, copy=True)


def layer_u_x_g_flat_from_snapshot(snapshot: NetworkSnapshot, layer_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = snapshot.state_by_layer[layer_key]
    if state.u_pre is None or state.x_pre is None:
        raise ValueError(f"{layer_key} snapshot is missing STSP u_pre/x_pre.")
    u = state.u_pre.view(state.u_pre.shape[0], -1).numpy().astype(np.float32, copy=True)
    x = state.x_pre.view(state.x_pre.shape[0], -1).numpy().astype(np.float32, copy=True)
    return u, x, (u * x).astype(np.float32, copy=False)


def compute_layer1_peak_masks(g_final: np.ndarray, *, baseline_gain: float, epsilon: float, peak_q: float) -> InputMasks:
    delta = np.asarray(g_final, dtype=np.float64) - float(baseline_gain)
    nonbase = np.abs(delta) > float(epsilon)
    peak = np.zeros_like(nonbase, dtype=bool)
    nonpeak = np.zeros_like(nonbase, dtype=bool)
    valid = np.zeros(delta.shape[0], dtype=bool)
    reasons: list[str] = []
    num_nonbase = nonbase.sum(axis=1).astype(np.int64)
    num_peak = np.zeros(delta.shape[0], dtype=np.int64)
    num_nonpeak = np.zeros(delta.shape[0], dtype=np.int64)
    mean_g_peak = np.full(delta.shape[0], np.nan, dtype=np.float64)
    mean_g_nonpeak = np.full(delta.shape[0], np.nan, dtype=np.float64)
    peak_nonpeak_ratio = np.full(delta.shape[0], np.nan, dtype=np.float64)
    for row_idx in range(delta.shape[0]):
        row_nonbase = nonbase[row_idx]
        nonbase_count = int(row_nonbase.sum())
        if nonbase_count <= 0:
            reasons.append("empty_l1_nonbase")
            continue
        peak_count = max(1, min(int(math.ceil(nonbase_count * float(peak_q))), nonbase_count))
        candidates = np.flatnonzero(row_nonbase)
        ranked = candidates[np.argsort(delta[row_idx, candidates], kind="stable")]
        peak[row_idx, ranked[-peak_count:]] = True
        nonpeak[row_idx] = row_nonbase & ~peak[row_idx]
        num_peak[row_idx] = int(peak[row_idx].sum())
        num_nonpeak[row_idx] = int(nonpeak[row_idx].sum())
        if num_nonpeak[row_idx] <= 0:
            reasons.append("empty_l1_nonpeak")
            continue
        mean_g_peak[row_idx] = region_mean(g_final[row_idx], peak[row_idx])
        mean_g_nonpeak[row_idx] = region_mean(g_final[row_idx], nonpeak[row_idx])
        peak_nonpeak_ratio[row_idx] = ratio_safe(mean_g_peak[row_idx], mean_g_nonpeak[row_idx])
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
        mean_g_peak=mean_g_peak,
        mean_g_nonpeak=mean_g_nonpeak,
        peak_nonpeak_ratio=peak_nonpeak_ratio,
    )


def project_l1_input_mask_to_l1_output(net: Any, snapshot: NetworkSnapshot, input_masks: InputMasks) -> RegionMasks:
    input_shape = tuple(int(v) for v in snapshot.layer_input_shapes["layer1"][1:])
    output_shape = tuple(int(v) for v in snapshot.state_by_layer["layer1"].v_mem.shape[1:])
    input_flat = int(np.prod(input_shape))
    output_flat = int(np.prod(output_shape))
    batch_size = int(input_masks.peak.shape[0])
    if input_shape == output_shape and input_flat == output_flat:
        return RegionMasks(
            mask_mode="direct",
            peak=input_masks.peak.copy(),
            nonpeak=input_masks.nonpeak.copy(),
            valid=input_masks.valid.copy(),
            invalid_reason=input_masks.invalid_reason,
            num_peak=input_masks.num_peak.copy(),
            num_nonpeak=input_masks.num_nonpeak.copy(),
            projection_info={"projection_mode": "direct", "input_shape": input_shape, "output_shape": output_shape},
        )

    weight = net.layer1.kernels.detach().abs().cpu().to(torch.float32)
    peak_out = np.zeros((batch_size, output_flat), dtype=bool)
    nonpeak_out = np.zeros((batch_size, output_flat), dtype=bool)
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
        supported = (peak_score + nonpeak_score) > 0.0
        peak_supported = supported & (peak_score >= nonpeak_score) & (peak_score > 0.0)
        nonpeak_supported = supported & ~peak_supported
        if int(peak_supported.sum()) <= 0:
            reasons.append("empty_l1_projected_peak")
            continue
        if int(nonpeak_supported.sum()) <= 0:
            reasons.append("empty_l1_projected_nonpeak")
            continue
        peak_out[row_idx] = peak_supported
        nonpeak_out[row_idx] = nonpeak_supported
        num_peak[row_idx] = int(peak_supported.sum())
        num_nonpeak[row_idx] = int(nonpeak_supported.sum())
        valid[row_idx] = True
        reasons.append("")
    return RegionMasks(
        mask_mode="weight_projected",
        peak=peak_out,
        nonpeak=nonpeak_out,
        valid=valid,
        invalid_reason=tuple(reasons),
        num_peak=num_peak,
        num_nonpeak=num_nonpeak,
        projection_info={
            "projection_mode": "weight_projected",
            "projection_score": "abs_weighted_conv_peak_vs_nonpeak_support",
            "input_shape": input_shape,
            "output_shape": output_shape,
            "layer1_kernel_shape": tuple(int(v) for v in net.layer1.kernels.shape),
            "layer1_stride": int(net.layer1.stride),
            "layer1_padding": int(net.layer1.padding),
        },
    )


def align_l1_output_mask_to_l2_input(snapshot: NetworkSnapshot, l1_output_masks: RegionMasks) -> RegionMasks:
    source_shape = tuple(int(v) for v in snapshot.state_by_layer["layer1"].v_mem.shape[1:])
    target_shape = tuple(int(v) for v in snapshot.layer_input_shapes["layer2"][1:])
    source_flat = int(np.prod(source_shape))
    target_flat = int(np.prod(target_shape))
    batch_size = int(l1_output_masks.peak.shape[0])
    if source_shape == target_shape and source_flat == target_flat:
        return RegionMasks(
            mask_mode="direct",
            peak=l1_output_masks.peak.copy(),
            nonpeak=l1_output_masks.nonpeak.copy(),
            valid=l1_output_masks.valid.copy(),
            invalid_reason=l1_output_masks.invalid_reason,
            num_peak=l1_output_masks.num_peak.copy(),
            num_nonpeak=l1_output_masks.num_nonpeak.copy(),
            projection_info={"projection_mode": "direct", "input_shape": source_shape, "output_shape": target_shape},
        )
    if len(source_shape) == 3 and len(target_shape) == 3 and source_shape[0] == target_shape[0] and source_shape[1] // 2 == target_shape[1] and source_shape[2] // 2 == target_shape[2]:
        peak_out = np.zeros((batch_size, target_flat), dtype=bool)
        nonpeak_out = np.zeros((batch_size, target_flat), dtype=bool)
        valid = np.zeros(batch_size, dtype=bool)
        reasons: list[str] = []
        num_peak = np.zeros(batch_size, dtype=np.int64)
        num_nonpeak = np.zeros(batch_size, dtype=np.int64)
        for row_idx in range(batch_size):
            if not bool(l1_output_masks.valid[row_idx]):
                reasons.append(l1_output_masks.invalid_reason[row_idx])
                continue
            peak_tensor = torch.as_tensor(l1_output_masks.peak[row_idx].reshape((1, *source_shape)), dtype=torch.float32)
            nonpeak_tensor = torch.as_tensor(l1_output_masks.nonpeak[row_idx].reshape((1, *source_shape)), dtype=torch.float32)
            peak_score = F.avg_pool2d(peak_tensor, kernel_size=2, stride=2).reshape(-1).numpy()
            nonpeak_score = F.avg_pool2d(nonpeak_tensor, kernel_size=2, stride=2).reshape(-1).numpy()
            supported = (peak_score + nonpeak_score) > 0.0
            peak_aligned = supported & (peak_score >= nonpeak_score) & (peak_score > 0.0)
            nonpeak_aligned = supported & ~peak_aligned
            if int(peak_aligned.sum()) <= 0:
                reasons.append("empty_l2_input_peak_alignment")
                continue
            if int(nonpeak_aligned.sum()) <= 0:
                reasons.append("empty_l2_input_nonpeak_alignment")
                continue
            peak_out[row_idx] = peak_aligned
            nonpeak_out[row_idx] = nonpeak_aligned
            num_peak[row_idx] = int(peak_aligned.sum())
            num_nonpeak[row_idx] = int(nonpeak_aligned.sum())
            valid[row_idx] = True
            reasons.append("")
        return RegionMasks(
            mask_mode="pool1_score_aligned",
            peak=peak_out,
            nonpeak=nonpeak_out,
            valid=valid,
            invalid_reason=tuple(reasons),
            num_peak=num_peak,
            num_nonpeak=num_nonpeak,
            projection_info={
                "projection_mode": "pool1_score_aligned",
                "input_shape": source_shape,
                "output_shape": target_shape,
                "pool_kernel": 2,
                "pool_stride": 2,
            },
        )
    if source_flat == target_flat:
        return RegionMasks(
            mask_mode="flat_reshape",
            peak=l1_output_masks.peak.copy(),
            nonpeak=l1_output_masks.nonpeak.copy(),
            valid=l1_output_masks.valid.copy(),
            invalid_reason=l1_output_masks.invalid_reason,
            num_peak=l1_output_masks.num_peak.copy(),
            num_nonpeak=l1_output_masks.num_nonpeak.copy(),
            projection_info={"projection_mode": "flat_reshape", "input_shape": source_shape, "output_shape": target_shape},
        )
    reasons = tuple("l2_input_alignment_shape_mismatch" for _ in range(batch_size))
    return RegionMasks(
        mask_mode="invalid_shape_mismatch",
        peak=np.zeros((batch_size, target_flat), dtype=bool),
        nonpeak=np.zeros((batch_size, target_flat), dtype=bool),
        valid=np.zeros(batch_size, dtype=bool),
        invalid_reason=reasons,
        num_peak=np.zeros(batch_size, dtype=np.int64),
        num_nonpeak=np.zeros(batch_size, dtype=np.int64),
        projection_info={"projection_mode": "invalid_shape_mismatch", "input_shape": source_shape, "output_shape": target_shape},
    )


def project_l2_input_mask_to_l2_output(net: Any, snapshot: NetworkSnapshot, l2_input_masks: RegionMasks) -> RegionMasks:
    input_shape = tuple(int(v) for v in snapshot.layer_input_shapes["layer2"][1:])
    output_shape = tuple(int(v) for v in snapshot.state_by_layer["layer2"].v_mem.shape[1:])
    input_flat = int(np.prod(input_shape))
    output_flat = int(np.prod(output_shape))
    batch_size = int(l2_input_masks.peak.shape[0])
    if input_shape == output_shape and input_flat == output_flat:
        return RegionMasks(
            mask_mode="direct",
            peak=l2_input_masks.peak.copy(),
            nonpeak=l2_input_masks.nonpeak.copy(),
            valid=l2_input_masks.valid.copy(),
            invalid_reason=l2_input_masks.invalid_reason,
            num_peak=l2_input_masks.num_peak.copy(),
            num_nonpeak=l2_input_masks.num_nonpeak.copy(),
            projection_info={"projection_mode": "direct", "input_shape": input_shape, "output_shape": output_shape},
        )
    weight = net.layer2.kernels.detach().abs().cpu().to(torch.float32)
    peak_out = np.zeros((batch_size, output_flat), dtype=bool)
    nonpeak_out = np.zeros((batch_size, output_flat), dtype=bool)
    valid = np.zeros(batch_size, dtype=bool)
    reasons: list[str] = []
    num_peak = np.zeros(batch_size, dtype=np.int64)
    num_nonpeak = np.zeros(batch_size, dtype=np.int64)
    for row_idx in range(batch_size):
        if not bool(l2_input_masks.valid[row_idx]):
            reasons.append(l2_input_masks.invalid_reason[row_idx])
            continue
        peak_tensor = torch.as_tensor(l2_input_masks.peak[row_idx].reshape((1, *input_shape)), dtype=torch.float32)
        nonpeak_tensor = torch.as_tensor(l2_input_masks.nonpeak[row_idx].reshape((1, *input_shape)), dtype=torch.float32)
        peak_score = F.conv2d(peak_tensor, weight, stride=int(net.layer2.stride), padding=int(net.layer2.padding)).reshape(-1).numpy()
        nonpeak_score = F.conv2d(nonpeak_tensor, weight, stride=int(net.layer2.stride), padding=int(net.layer2.padding)).reshape(-1).numpy()
        supported = (peak_score + nonpeak_score) > 0.0
        peak_supported = supported & (peak_score >= nonpeak_score) & (peak_score > 0.0)
        nonpeak_supported = supported & ~peak_supported
        if int(peak_supported.sum()) <= 0:
            reasons.append("empty_l2_projected_peak")
            continue
        if int(nonpeak_supported.sum()) <= 0:
            reasons.append("empty_l2_projected_nonpeak")
            continue
        peak_out[row_idx] = peak_supported
        nonpeak_out[row_idx] = nonpeak_supported
        num_peak[row_idx] = int(peak_supported.sum())
        num_nonpeak[row_idx] = int(nonpeak_supported.sum())
        valid[row_idx] = True
        reasons.append("")
    return RegionMasks(
        mask_mode="weight_projected",
        peak=peak_out,
        nonpeak=nonpeak_out,
        valid=valid,
        invalid_reason=tuple(reasons),
        num_peak=num_peak,
        num_nonpeak=num_nonpeak,
        projection_info={
            "projection_mode": "weight_projected",
            "projection_score": "abs_weighted_conv_peak_vs_nonpeak_support",
            "input_shape": input_shape,
            "output_shape": output_shape,
            "layer2_kernel_shape": tuple(int(v) for v in net.layer2.kernels.shape),
            "layer2_stride": int(net.layer2.stride),
            "layer2_padding": int(net.layer2.padding),
        },
    )


def select_low_high_peak_overlap_probes(
    trial: SequenceTrial,
    candidate_ids: Sequence[int],
    labels: np.ndarray,
    spike_lookup: Mapping[int, torch.Tensor],
    peak_mask: np.ndarray,
    nonpeak_mask: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    masks_valid = bool(np.asarray(peak_mask, dtype=bool).sum() > 0 and np.asarray(nonpeak_mask, dtype=bool).sum() > 0)
    for candidate_id in candidate_ids:
        spikes = spike_lookup[int(candidate_id)]
        hit = spikes.detach().sum(dim=0).view(-1).cpu().numpy() > 0.0
        spike_count = float(spikes.detach().sum().item())
        denom = float(hit.sum()) + SMALL_EPS
        peak_fraction = float(np.logical_and(hit, peak_mask).sum() / denom)
        nonpeak_fraction = float(np.logical_and(hit, nonpeak_mask).sum() / denom)
        valid = int(masks_valid and spike_count > 0.0)
        rows.append(
            {
                "trial_id": int(trial.trial_id),
                "candidate_image_id": int(candidate_id),
                "candidate_label": int(labels[int(candidate_id)]),
                "candidate_spike_count": spike_count,
                "probe_peak_overlap_fraction": peak_fraction,
                "probe_nonpeak_overlap_fraction": nonpeak_fraction,
                "selected": 0,
                "probe_overlap_group": "unselected",
                "selection_rank": -1,
                "duplicate_selected_candidate": 0,
                "valid": valid,
                "invalid_reason": "" if valid else ("empty_masks" if not masks_valid else "zero_spike_probe"),
            }
        )
    if not rows:
        return rows, [], {"duplicate_probe_selection_count": 0, "selection_warning_count": 1}

    table = pd.DataFrame(rows)
    valid_table = table[table["valid"] == 1].copy()
    if valid_table.empty:
        return rows, [], {"duplicate_probe_selection_count": 0, "selection_warning_count": 1}
    if cfg.probe_overlap_bin_method == "median":
        median_value = float(valid_table["probe_peak_overlap_fraction"].median())
        low_pool = valid_table[valid_table["probe_peak_overlap_fraction"] <= median_value].copy()
        high_pool = valid_table[valid_table["probe_peak_overlap_fraction"] >= median_value].copy()
        if low_pool.empty:
            low_pool = valid_table.copy()
        if high_pool.empty:
            high_pool = valid_table.copy()
    else:
        low_pool = valid_table.copy()
        high_pool = valid_table.copy()

    low_order = list(low_pool.sort_values(["probe_peak_overlap_fraction", "candidate_image_id"], ascending=[True, True], kind="stable").index.astype(int))
    high_order = list(high_pool.sort_values(["probe_peak_overlap_fraction", "candidate_image_id"], ascending=[False, True], kind="stable").index.astype(int))
    used: set[int] = set()
    selected_indices: list[tuple[int, str, int, int]] = []
    duplicate_count = 0
    warning_count = 0

    for group_name, order in (("low_peak_overlap", low_order), ("high_peak_overlap", high_order)):
        chosen: list[int] = []
        for idx in order:
            if idx in used:
                continue
            chosen.append(int(idx))
            used.add(int(idx))
            if len(chosen) >= int(cfg.selected_probes_per_trial):
                break
        if len(chosen) < int(cfg.selected_probes_per_trial):
            warning_count += 1
            for idx in order:
                if len(chosen) >= int(cfg.selected_probes_per_trial):
                    break
                duplicate_count += int(idx in used)
                chosen.append(int(idx))
                used.add(int(idx))
        for rank, idx in enumerate(chosen, start=1):
            selected_indices.append((int(idx), group_name, int(rank), int(sum(1 for prev_idx, _g, _r, _d in selected_indices if prev_idx == idx) > 0)))

    selected_rows: list[dict[str, object]] = []
    for idx, group_name, rank, duplicate in selected_indices:
        rows[idx]["selected"] = 1
        rows[idx]["probe_overlap_group"] = group_name
        rows[idx]["selection_rank"] = int(rank)
        rows[idx]["duplicate_selected_candidate"] = int(duplicate)
        selected_rows.append(dict(rows[idx]))
    return rows, selected_rows, {"duplicate_probe_selection_count": int(duplicate_count), "selection_warning_count": int(warning_count)}


def apply_l1_peak_condition(
    net: Any,
    row_idx: int,
    input_masks: InputMasks,
    condition: str,
    boost_level: float,
) -> tuple[float, str]:
    if condition == "l1_intact":
        return 0.0, ""
    if not bool(input_masks.valid[row_idx]):
        return float("nan"), str(input_masks.invalid_reason[row_idx])
    layer = net.layer1
    if layer.u_pre is None or layer.x_pre is None:
        return float("nan"), "missing_l1_stsp_state"
    flat_u = layer.u_pre.view(layer.u_pre.shape[0], -1)
    flat_x = layer.x_pre.view(layer.x_pre.shape[0], -1)
    peak = torch.as_tensor(input_masks.peak[row_idx], dtype=torch.bool, device=flat_u.device)
    nonpeak = torch.as_tensor(input_masks.nonpeak[row_idx], dtype=torch.bool, device=flat_u.device)
    if int(peak.sum().item()) <= 0 or int(nonpeak.sum().item()) <= 0:
        return float("nan"), "empty_l1_condition_mask"
    with torch.no_grad():
        mean_u_nonpeak = flat_u[row_idx, nonpeak].mean()
        mean_x_nonpeak = flat_x[row_idx, nonpeak].mean()
        original_u = flat_u[row_idx, peak].clone()
        original_x = flat_x[row_idx, peak].clone()
        if condition == "l1_peak_flattened":
            proposed_u = torch.full_like(original_u, float(mean_u_nonpeak.item()))
            proposed_x = torch.full_like(original_x, float(mean_x_nonpeak.item()))
        elif condition == "l1_peak_boosted":
            proposed_u = original_u + float(boost_level) * (original_u - mean_u_nonpeak)
            proposed_x = original_x + float(boost_level) * (original_x - mean_x_nonpeak)
        else:
            return float("nan"), f"unknown_l1_condition:{condition}"
        clipped_u = torch.clamp(proposed_u, 0.0, 1.0)
        clipped_x = torch.clamp(proposed_x, 0.0, 1.0)
        flat_u[row_idx, peak] = clipped_u
        flat_x[row_idx, peak] = clipped_x
        clipping = torch.cat([(clipped_u != proposed_u).float(), (clipped_x != proposed_x).float()])
    return (float(clipping.mean().item()) if clipping.numel() else 0.0), ""


def apply_l2_memory_condition(net: Any, row_idx: int, memory_condition: str) -> str:
    if memory_condition == "l2_intact":
        return ""
    if memory_condition != "l2_reset":
        return f"unknown_l2_memory_condition:{memory_condition}"
    layer = net.layer2
    with torch.no_grad():
        layer.v_mem[row_idx].fill_(float(layer.V_L))
        layer.g_e[row_idx].zero_()
        layer.res[row_idx].zero_()
        layer.lateral_inh.inh_trace[row_idx].zero_()
        if getattr(layer, "empty_mask", None) is not None:
            layer.empty_mask[row_idx].zero_()
        if getattr(layer, "pre_trace", None) is not None:
            layer.pre_trace[row_idx].zero_()
        if getattr(layer, "input_trace", None) is not None:
            layer.input_trace[row_idx].zero_()
        if getattr(layer, "eligibility_trace", None) is not None:
            layer.eligibility_trace[row_idx].zero_()
        if getattr(layer, "firing_times", None) is not None:
            layer.firing_times[row_idx].fill_(float("inf"))
        if layer.u_pre is not None:
            layer.u_pre[row_idx].fill_(float(layer.stsp_U))
        if layer.x_pre is not None:
            layer.x_pre[row_idx].fill_(1.0)
    return ""


def build_condition_snapshot(
    net: Any,
    final_snapshot: NetworkSnapshot,
    row_idx: int,
    input_masks: InputMasks,
    *,
    l1_condition: str,
    l2_memory_condition: str,
    cfg: ExperimentConfig,
) -> tuple[NetworkSnapshot, float, str]:
    restore_model_state(net, final_snapshot)
    clipping_fraction, l1_reason = apply_l1_peak_condition(
        net,
        row_idx,
        input_masks,
        condition=l1_condition,
        boost_level=float(cfg.boost_level),
    )
    l2_reason = apply_l2_memory_condition(net, row_idx, l2_memory_condition)
    invalid_reason = ";".join(item for item in (l1_reason, l2_reason) if item)
    condition_snapshot = snapshot_model_state(
        net,
        current_time=int(final_snapshot.current_time),
        layer_input_shapes=final_snapshot.layer_input_shapes,
    )
    return condition_snapshot, clipping_fraction, invalid_reason


def compute_spike_metrics(counts: np.ndarray, masks: RegionMasks, row_idx: int, *, prefix: str) -> dict[str, float | int | str]:
    total = float(np.asarray(counts, dtype=np.float64).sum())
    if not bool(masks.valid[row_idx]):
        return {
            f"{prefix}_valid": 0,
            f"{prefix}_invalid_reason": str(masks.invalid_reason[row_idx]),
            f"{prefix}_total_spikes": total,
            f"{prefix}_peak_spike_count": np.nan,
            f"{prefix}_nonpeak_spike_count": np.nan,
            f"{prefix}_peak_spike_fraction": np.nan,
            f"{prefix}_spike_enrichment": np.nan,
        }
    peak = masks.peak[row_idx]
    nonpeak = masks.nonpeak[row_idx]
    peak_count = float(counts[peak].sum())
    nonpeak_count = float(counts[nonpeak].sum())
    peak_density = peak_count / max(int(masks.num_peak[row_idx]), 1)
    nonpeak_density = nonpeak_count / max(int(masks.num_nonpeak[row_idx]), 1)
    return {
        f"{prefix}_valid": 1,
        f"{prefix}_invalid_reason": "",
        f"{prefix}_total_spikes": total,
        f"{prefix}_peak_spike_count": peak_count,
        f"{prefix}_nonpeak_spike_count": nonpeak_count,
        f"{prefix}_peak_spike_fraction": float(peak_count / (total + SMALL_EPS)),
        f"{prefix}_peak_spike_density": float(peak_density),
        f"{prefix}_nonpeak_spike_density": float(nonpeak_density),
        f"{prefix}_spike_enrichment": float(peak_density / (nonpeak_density + SMALL_EPS)),
    }


def compute_l2_update_metrics(delta_g: np.ndarray, masks: RegionMasks, row_idx: int, epsilon: float) -> dict[str, float | int | str]:
    if not bool(masks.valid[row_idx]):
        return {
            "l2_update_valid": 0,
            "l2_update_invalid_reason": str(masks.invalid_reason[row_idx]),
            "l2_update_enrichment": np.nan,
            "l2_update_difference": np.nan,
            "l2_peak_input_update_mean": np.nan,
            "l2_nonpeak_input_update_mean": np.nan,
            "l2_peak_input_signed_update_mean": np.nan,
            "l2_nonpeak_input_signed_update_mean": np.nan,
            "l2_signed_update_difference": np.nan,
            "l2_update_count_enrichment": np.nan,
            "l2_peak_input_update_count_density": np.nan,
            "l2_nonpeak_input_update_count_density": np.nan,
        }
    peak = masks.peak[row_idx]
    nonpeak = masks.nonpeak[row_idx]
    signed = np.asarray(delta_g, dtype=np.float64)
    positive = np.maximum(signed, 0.0)
    peak_update = region_mean(positive, peak)
    nonpeak_update = region_mean(positive, nonpeak)
    peak_signed = region_mean(signed, peak)
    nonpeak_signed = region_mean(signed, nonpeak)
    update_hit = signed > float(epsilon)
    peak_count_density = region_mean(update_hit.astype(np.float64), peak)
    nonpeak_count_density = region_mean(update_hit.astype(np.float64), nonpeak)
    return {
        "l2_update_valid": 1,
        "l2_update_invalid_reason": "",
        "l2_update_enrichment": ratio_safe(peak_update, nonpeak_update),
        "l2_update_difference": float(peak_update - nonpeak_update) if np.isfinite([peak_update, nonpeak_update]).all() else np.nan,
        "l2_peak_input_update_mean": peak_update,
        "l2_nonpeak_input_update_mean": nonpeak_update,
        "l2_peak_input_signed_update_mean": peak_signed,
        "l2_nonpeak_input_signed_update_mean": nonpeak_signed,
        "l2_signed_update_difference": float(peak_signed - nonpeak_signed) if np.isfinite([peak_signed, nonpeak_signed]).all() else np.nan,
        "l2_update_count_enrichment": ratio_safe(peak_count_density, nonpeak_count_density),
        "l2_peak_input_update_count_density": peak_count_density,
        "l2_nonpeak_input_update_count_density": nonpeak_count_density,
    }


def run_probe_actual_and_decay(
    net: Any,
    condition_snapshot: NetworkSnapshot,
    probe_spikes_single: torch.Tensor,
    row_idx: int,
    l1_output_masks: RegionMasks,
    l2_input_masks: RegionMasks,
    l2_output_masks: RegionMasks,
    cfg: ExperimentConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    batch_size = int(condition_snapshot.layer_input_shapes["layer1"][0])
    channels, height, width = (int(v) for v in condition_snapshot.layer_input_shapes["layer1"][1:])
    device = probe_spikes_single.device
    probe_batch = torch.zeros(
        (batch_size, int(probe_spikes_single.shape[0]), channels, height, width),
        dtype=probe_spikes_single.dtype,
        device=device,
    )
    probe_batch[row_idx] = probe_spikes_single
    l1_counts: np.ndarray | None = None
    l2_counts: np.ndarray | None = None

    restore_model_state(net, condition_snapshot)
    current_time = int(condition_snapshot.current_time)
    with torch.no_grad():
        for step_idx in range(int(probe_batch.shape[1])):
            s1, s2 = forward_three_layers_capture(net, probe_batch[:, step_idx, ...], current_time)
            s1_row = s1[row_idx].detach().to(torch.float32).cpu().view(-1).numpy()
            s2_row = s2[row_idx].detach().to(torch.float32).cpu().view(-1).numpy()
            l1_counts = s1_row.copy() if l1_counts is None else l1_counts + s1_row
            l2_counts = s2_row.copy() if l2_counts is None else l2_counts + s2_row
            current_time += 1
    actual_g_l2 = layer_gain_flat(net, "layer2")[row_idx]

    restore_model_state(net, condition_snapshot)
    current_time = int(condition_snapshot.current_time)
    zero_input = torch.zeros((batch_size, channels, height, width), dtype=probe_spikes_single.dtype, device=device)
    with torch.no_grad():
        for _ in range(int(probe_batch.shape[1])):
            forward_three_layers_capture(net, zero_input, current_time)
            current_time += 1
    decay_g_l2 = layer_gain_flat(net, "layer2")[row_idx]
    delta_g = actual_g_l2 - decay_g_l2

    if l1_counts is None:
        l1_counts = np.zeros_like(l1_output_masks.peak[row_idx], dtype=np.float64)
    if l2_counts is None:
        l2_counts = np.zeros_like(l2_output_masks.peak[row_idx], dtype=np.float64)

    l1_metrics = compute_spike_metrics(l1_counts, l1_output_masks, row_idx, prefix="l1")
    l2_update_metrics = compute_l2_update_metrics(delta_g, l2_input_masks, row_idx, epsilon=float(cfg.epsilon))
    l2_spike_metrics = compute_spike_metrics(l2_counts, l2_output_masks, row_idx, prefix="l2")
    metrics = {**l1_metrics, **l2_update_metrics, **l2_spike_metrics}
    update_map = {
        "l2_delta_g_positive_sum": float(np.maximum(delta_g, 0.0).sum()),
        "l2_delta_g_signed_sum": float(np.asarray(delta_g, dtype=np.float64).sum()),
        "l2_delta_g_positive_count": int((delta_g > float(cfg.epsilon)).sum()),
        "l2_delta_g_max": float(np.max(delta_g)) if delta_g.size else np.nan,
        "l2_delta_g_min": float(np.min(delta_g)) if delta_g.size else np.nan,
    }
    return metrics, update_map


def build_paired_effects(condition_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trial_id",
        "probe_image_id",
        "probe_label",
        "probe_overlap_group",
        "probe_peak_overlap_fraction",
        "probe_nonpeak_overlap_fraction",
        "delta_l1_spike_enrichment_intact_vs_flattened",
        "delta_l1_spike_enrichment_boosted_vs_flattened",
        "delta_l2_update_enrichment_intact_vs_flattened",
        "delta_l2_update_enrichment_boosted_vs_flattened",
        "delta_l2_update_difference_intact_vs_flattened",
        "delta_l2_update_difference_boosted_vs_flattened",
        "delta_l2_update_enrichment_intact_vs_flattened_l2reset",
        "delta_l2_update_enrichment_intact_vs_flattened_l2intact",
        "delta_l2_update_enrichment_boosted_vs_flattened_l2reset",
        "delta_l2_update_enrichment_boosted_vs_flattened_l2intact",
        "delta_l2_update_difference_intact_vs_flattened_l2reset",
        "delta_l2_update_difference_intact_vs_flattened_l2intact",
        "delta_l2_update_difference_boosted_vs_flattened_l2reset",
        "delta_l2_update_difference_boosted_vs_flattened_l2intact",
        "delta_l2_update_enrichment_l2intact_vs_l2reset",
        "delta_l2_update_difference_l2intact_vs_l2reset",
        "delta_l2_update_enrichment_l2intact_vs_l2reset_flattened",
        "delta_l2_update_enrichment_l2intact_vs_l2reset_intact",
        "delta_l2_update_enrichment_l2intact_vs_l2reset_boosted",
        "delta_l2_update_difference_l2intact_vs_l2reset_flattened",
        "delta_l2_update_difference_l2intact_vs_l2reset_intact",
        "delta_l2_update_difference_l2intact_vs_l2reset_boosted",
        "interaction_l1_intact_effect_by_l2_memory",
        "interaction_l1_boosted_effect_by_l2_memory",
        "valid",
        "invalid_reason",
    ]
    if condition_df.empty:
        return pd.DataFrame(columns=columns)
    valid_df = condition_df[condition_df["valid"] == 1].copy()
    if valid_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    index_cols = [
        "trial_id",
        "probe_image_id",
        "probe_label",
        "probe_overlap_group",
        "probe_peak_overlap_fraction",
        "probe_nonpeak_overlap_fraction",
    ]
    for key, sub in valid_df.groupby(index_cols, sort=False, dropna=False):
        by_cond = {(str(row["l1_condition"]), str(row["l2_memory_condition"])): row for _, row in sub.iterrows()}

        def value(l1_condition: str, l2_condition: str, field: str) -> float:
            row = by_cond.get((l1_condition, l2_condition))
            if row is None:
                return float("nan")
            return float(row[field])

        def diff(l1_a: str, l1_b: str, l2_condition: str, field: str) -> float:
            return value(l1_a, l2_condition, field) - value(l1_b, l2_condition, field)

        intact_l2reset = diff("l1_intact", "l1_peak_flattened", "l2_reset", "l2_update_enrichment")
        intact_l2intact = diff("l1_intact", "l1_peak_flattened", "l2_intact", "l2_update_enrichment")
        boosted_l2reset = diff("l1_peak_boosted", "l1_peak_flattened", "l2_reset", "l2_update_enrichment")
        boosted_l2intact = diff("l1_peak_boosted", "l1_peak_flattened", "l2_intact", "l2_update_enrichment")
        intact_diff_l2reset = diff("l1_intact", "l1_peak_flattened", "l2_reset", "l2_update_difference")
        intact_diff_l2intact = diff("l1_intact", "l1_peak_flattened", "l2_intact", "l2_update_difference")
        boosted_diff_l2reset = diff("l1_peak_boosted", "l1_peak_flattened", "l2_reset", "l2_update_difference")
        boosted_diff_l2intact = diff("l1_peak_boosted", "l1_peak_flattened", "l2_intact", "l2_update_difference")
        l1_spike_intact = diff("l1_intact", "l1_peak_flattened", "l2_intact", "l1_spike_enrichment")
        if not np.isfinite(l1_spike_intact):
            l1_spike_intact = diff("l1_intact", "l1_peak_flattened", "l2_reset", "l1_spike_enrichment")
        l1_spike_boosted = diff("l1_peak_boosted", "l1_peak_flattened", "l2_intact", "l1_spike_enrichment")
        if not np.isfinite(l1_spike_boosted):
            l1_spike_boosted = diff("l1_peak_boosted", "l1_peak_flattened", "l2_reset", "l1_spike_enrichment")

        (
            trial_id,
            probe_image_id,
            probe_label,
            probe_overlap_group,
            peak_overlap,
            nonpeak_overlap,
        ) = key
        rows.append(
            {
                "trial_id": int(trial_id),
                "probe_image_id": int(probe_image_id),
                "probe_label": int(probe_label),
                "probe_overlap_group": str(probe_overlap_group),
                "probe_peak_overlap_fraction": float(peak_overlap),
                "probe_nonpeak_overlap_fraction": float(nonpeak_overlap),
                "delta_l1_spike_enrichment_intact_vs_flattened": l1_spike_intact,
                "delta_l1_spike_enrichment_boosted_vs_flattened": l1_spike_boosted,
                "delta_l2_update_enrichment_intact_vs_flattened": intact_l2intact,
                "delta_l2_update_enrichment_boosted_vs_flattened": boosted_l2intact,
                "delta_l2_update_difference_intact_vs_flattened": intact_diff_l2intact,
                "delta_l2_update_difference_boosted_vs_flattened": boosted_diff_l2intact,
                "delta_l2_update_enrichment_intact_vs_flattened_l2reset": intact_l2reset,
                "delta_l2_update_enrichment_intact_vs_flattened_l2intact": intact_l2intact,
                "delta_l2_update_enrichment_boosted_vs_flattened_l2reset": boosted_l2reset,
                "delta_l2_update_enrichment_boosted_vs_flattened_l2intact": boosted_l2intact,
                "delta_l2_update_difference_intact_vs_flattened_l2reset": intact_diff_l2reset,
                "delta_l2_update_difference_intact_vs_flattened_l2intact": intact_diff_l2intact,
                "delta_l2_update_difference_boosted_vs_flattened_l2reset": boosted_diff_l2reset,
                "delta_l2_update_difference_boosted_vs_flattened_l2intact": boosted_diff_l2intact,
                "delta_l2_update_enrichment_l2intact_vs_l2reset": value("l1_intact", "l2_intact", "l2_update_enrichment") - value("l1_intact", "l2_reset", "l2_update_enrichment"),
                "delta_l2_update_difference_l2intact_vs_l2reset": value("l1_intact", "l2_intact", "l2_update_difference") - value("l1_intact", "l2_reset", "l2_update_difference"),
                "delta_l2_update_enrichment_l2intact_vs_l2reset_flattened": value("l1_peak_flattened", "l2_intact", "l2_update_enrichment") - value("l1_peak_flattened", "l2_reset", "l2_update_enrichment"),
                "delta_l2_update_enrichment_l2intact_vs_l2reset_intact": value("l1_intact", "l2_intact", "l2_update_enrichment") - value("l1_intact", "l2_reset", "l2_update_enrichment"),
                "delta_l2_update_enrichment_l2intact_vs_l2reset_boosted": value("l1_peak_boosted", "l2_intact", "l2_update_enrichment") - value("l1_peak_boosted", "l2_reset", "l2_update_enrichment"),
                "delta_l2_update_difference_l2intact_vs_l2reset_flattened": value("l1_peak_flattened", "l2_intact", "l2_update_difference") - value("l1_peak_flattened", "l2_reset", "l2_update_difference"),
                "delta_l2_update_difference_l2intact_vs_l2reset_intact": value("l1_intact", "l2_intact", "l2_update_difference") - value("l1_intact", "l2_reset", "l2_update_difference"),
                "delta_l2_update_difference_l2intact_vs_l2reset_boosted": value("l1_peak_boosted", "l2_intact", "l2_update_difference") - value("l1_peak_boosted", "l2_reset", "l2_update_difference"),
                "interaction_l1_intact_effect_by_l2_memory": intact_l2intact - intact_l2reset,
                "interaction_l1_boosted_effect_by_l2_memory": boosted_l2intact - boosted_l2reset,
                "valid": 1,
                "invalid_reason": "",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def aggregate_mean_sem(df: pd.DataFrame, group_cols: Sequence[str], value_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, "mean", "sem", "n"])
    for key, sub in df.groupby(list(group_cols), sort=False, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        values = sub[value_col].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        row = {str(col): key_tuple[idx] for idx, col in enumerate(group_cols)}
        row.update({"mean": float(finite.mean()) if finite.size else np.nan, "sem": sem(finite), "n": int(finite.size)})
        rows.append(row)
    return pd.DataFrame(rows)


def _insufficient_axis(ax: plt.Axes, message: str = "insufficient data") -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()


def make_condition_line_figure(
    condition_df: pd.DataFrame,
    out_base: Path,
    *,
    value_col: str,
    ylabel: str,
    title: str,
    high_only: bool,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    required = {"trial_id", "probe_overlap_group", "l1_condition", "l2_memory_condition", value_col, "valid"}
    if condition_df.empty or not required.issubset(condition_df.columns):
        _insufficient_axis(ax)
        return save_figure_all_formats(fig, out_base)
    plot_df = condition_df[condition_df["valid"] == 1].copy()
    if high_only:
        plot_df = plot_df[plot_df["probe_overlap_group"] == "high_peak_overlap"].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df[value_col])].copy()
    if plot_df.empty:
        _insufficient_axis(ax)
        return save_figure_all_formats(fig, out_base)
    trial_df = (
        plot_df.groupby(["trial_id", "l1_condition", "l2_memory_condition"], as_index=False)
        .agg(value=(value_col, "mean"))
    )
    agg = aggregate_mean_sem(trial_df, ["l1_condition", "l2_memory_condition"], "value")
    x_lookup = {"l2_reset": 0.0, "l2_intact": 1.0}
    labels = {
        "l1_peak_flattened": "L1 peak flattened",
        "l1_intact": "L1 intact",
        "l1_peak_boosted": "L1 peak boosted",
    }
    colors = {
        "l1_peak_flattened": STATIC_COLOR,
        "l1_intact": SAMPLE_COLOR,
        "l1_peak_boosted": DYNAMIC_COLOR,
    }
    for condition in L1_CONDITIONS:
        sub = agg[agg["l1_condition"] == condition].copy()
        if sub.empty:
            continue
        sub["x"] = sub["l2_memory_condition"].map(x_lookup)
        sub = sub.dropna(subset=["x"]).sort_values("x")
        ax.errorbar(
            sub["x"].to_numpy(dtype=np.float64),
            sub["mean"].to_numpy(dtype=np.float64),
            yerr=sub["sem"].to_numpy(dtype=np.float64),
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=colors[condition],
            label=labels[condition],
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["L2 reset", "L2 intact"])
    ax.set_xlabel("Layer 2 memory state")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.margins(x=0.15)
    return save_figure_all_formats(fig, out_base)


def make_bias_correlation_figure(paired_df: pd.DataFrame, out_base: Path) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    if paired_df.empty:
        _insufficient_axis(ax)
        return {"paths": save_figure_all_formats(fig, out_base), "pearson_r": None}
    rows: list[dict[str, object]] = []
    for _, row in paired_df.iterrows():
        for effect_label, x_col, y_reset_col, y_intact_col, marker in (
            (
                "intact",
                "delta_l1_spike_enrichment_intact_vs_flattened",
                "delta_l2_update_enrichment_intact_vs_flattened_l2reset",
                "delta_l2_update_enrichment_intact_vs_flattened_l2intact",
                "o",
            ),
            (
                "boosted",
                "delta_l1_spike_enrichment_boosted_vs_flattened",
                "delta_l2_update_enrichment_boosted_vs_flattened_l2reset",
                "delta_l2_update_enrichment_boosted_vs_flattened_l2intact",
                "s",
            ),
        ):
            rows.append({"x": row.get(x_col, np.nan), "y": row.get(y_reset_col, np.nan), "memory": "l2_reset", "effect": effect_label, "marker": marker})
            rows.append({"x": row.get(x_col, np.nan), "y": row.get(y_intact_col, np.nan), "memory": "l2_intact", "effect": effect_label, "marker": marker})
    plot_df = pd.DataFrame(rows)
    plot_df["x"] = pd.to_numeric(plot_df["x"], errors="coerce")
    plot_df["y"] = pd.to_numeric(plot_df["y"], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df["x"]) & np.isfinite(plot_df["y"])].copy()
    if plot_df.empty:
        _insufficient_axis(ax)
        return {"paths": save_figure_all_formats(fig, out_base), "pearson_r": None}
    colors = {"l2_reset": STATIC_COLOR, "l2_intact": DYNAMIC_COLOR}
    for memory in L2_MEMORY_CONDITIONS:
        sub = plot_df[plot_df["memory"] == memory]
        for effect, marker in (("intact", "o"), ("boosted", "s")):
            part = sub[sub["effect"] == effect]
            if part.empty:
                continue
            ax.scatter(part["x"], part["y"], s=28, color=colors[memory], alpha=0.55, linewidths=0, marker=marker, label=f"{memory} {effect}")
    x = plot_df["x"].to_numpy(dtype=np.float64)
    y = plot_df["y"].to_numpy(dtype=np.float64)
    r = finite_corr(x, y)
    if x.size >= 2 and float(np.std(x)) > 0.0:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), num=100)
        ax.plot(x_line, slope * x_line + intercept, color=NOISE_COLOR, linewidth=1.6)
    if r is not None:
        ax.text(0.04, 0.94, f"Pearson r = {r:.3f}", transform=ax.transAxes, ha="left", va="top")
    ax.axhline(0.0, color=NOISE_COLOR, linewidth=1.0, linestyle="--", alpha=0.65)
    ax.axvline(0.0, color=NOISE_COLOR, linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_xlabel("Delta Layer 1 spike enrichment")
    ax.set_ylabel("Delta Layer 2 update enrichment")
    ax.set_title("Layer 1 spike bias predicts Layer 2 update bias")
    ax.legend(frameon=False, fontsize=8)
    return {"paths": save_figure_all_formats(fig, out_base), "pearson_r": r}


def make_probe_overlap_distribution_figure(probe_selection_df: pd.DataFrame, out_base: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    required = {"probe_peak_overlap_fraction", "probe_overlap_group", "selected"}
    if probe_selection_df.empty or not required.issubset(probe_selection_df.columns):
        _insufficient_axis(ax)
        return save_figure_all_formats(fig, out_base)
    df = probe_selection_df.copy()
    df["probe_peak_overlap_fraction"] = pd.to_numeric(df["probe_peak_overlap_fraction"], errors="coerce")
    df = df[np.isfinite(df["probe_peak_overlap_fraction"])].copy()
    if df.empty:
        _insufficient_axis(ax)
        return save_figure_all_formats(fig, out_base)
    ax.hist(df["probe_peak_overlap_fraction"], bins=16, color=NOISE_COLOR, alpha=0.28, label="Candidates")
    selected = df[df["selected"] == 1]
    colors = {"low_peak_overlap": STATIC_COLOR, "high_peak_overlap": DYNAMIC_COLOR}
    for group_name in PROBE_GROUPS:
        sub = selected[selected["probe_overlap_group"] == group_name]
        if sub.empty:
            continue
        ax.scatter(sub["probe_peak_overlap_fraction"], np.zeros(len(sub)), color=colors[group_name], s=32, alpha=0.75, label=group_name)
    ax.set_xlabel("Probe-peak overlap fraction")
    ax.set_ylabel("Candidate count")
    ax.set_title("Probe selection spans low and high peak overlap")
    ax.legend(frameon=False)
    return save_figure_all_formats(fig, out_base)


def generate_figures(
    layout: Any,
    *,
    probe_selection_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    paired_df: pd.DataFrame,
) -> dict[str, Any]:
    apply_publication_style()
    figures: dict[str, Any] = {}
    figures["fig6E_layer2_update_enrichment_by_memory_state"] = make_condition_line_figure(
        condition_df,
        layout.figure_base("fig6E_layer2_update_enrichment_by_memory_state"),
        value_col="l2_update_enrichment",
        ylabel="Layer 2 update enrichment",
        title="Layer 2 memory integrates Layer 1 peak-biased input",
        high_only=True,
    )
    plt.close("all")
    figures["low_vs_high_overlap_layer2_update_enrichment"] = make_condition_line_figure(
        condition_df,
        layout.figure_base("low_vs_high_overlap_layer2_update_enrichment"),
        value_col="l2_update_enrichment",
        ylabel="Layer 2 update enrichment",
        title="Layer 2 update enrichment across selected overlap groups",
        high_only=False,
    )
    plt.close("all")
    figures["fig6E_l1_spike_bias_predicts_l2_update_bias"] = make_bias_correlation_figure(
        paired_df,
        layout.figure_base("fig6E_l1_spike_bias_predicts_l2_update_bias"),
    )
    plt.close("all")
    figures["fig6E_layer2_update_difference_by_condition"] = make_condition_line_figure(
        condition_df,
        layout.figure_base("fig6E_layer2_update_difference_by_condition"),
        value_col="l2_update_difference",
        ylabel="Layer 2 update difference",
        title="Peak-biased input increases downstream update magnitude",
        high_only=True,
    )
    plt.close("all")
    figures["l2_update_count_enrichment_by_condition"] = make_condition_line_figure(
        condition_df,
        layout.figure_base("l2_update_count_enrichment_by_condition"),
        value_col="l2_update_count_enrichment",
        ylabel="Layer 2 update count enrichment",
        title="Peak-biased input increases downstream update count",
        high_only=True,
    )
    plt.close("all")
    figures["l2_spike_enrichment_by_condition"] = make_condition_line_figure(
        condition_df,
        layout.figure_base("l2_spike_enrichment_by_condition"),
        value_col="l2_spike_enrichment",
        ylabel="Layer 2 spike enrichment",
        title="Layer 2 spike enrichment by downstream condition",
        high_only=True,
    )
    plt.close("all")
    figures["probe_selection_overlap_distribution"] = make_probe_overlap_distribution_figure(
        probe_selection_df,
        layout.figure_base("probe_selection_overlap_distribution"),
    )
    plt.close("all")
    return figures


def nested_group_mean(df: pd.DataFrame, group_cols: Sequence[str], value_col: str) -> dict[str, Any]:
    if df.empty or value_col not in df.columns:
        return {}
    out: dict[str, Any] = {}
    valid_df = df.copy()
    valid_df[value_col] = pd.to_numeric(valid_df[value_col], errors="coerce")
    valid_df = valid_df[np.isfinite(valid_df[value_col])].copy()
    if "valid" in valid_df.columns:
        valid_df = valid_df[valid_df["valid"] == 1].copy()
    for key, sub in valid_df.groupby(list(group_cols), sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        cursor = out
        for part in key_tuple[:-1]:
            cursor = cursor.setdefault(str(part), {})
        cursor[str(key_tuple[-1])] = safe_mean(sub[value_col].to_numpy(dtype=np.float64))
    return out


def value_counts_dict(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def summarize_results(
    cfg: ExperimentConfig,
    *,
    mask_df: pd.DataFrame,
    probe_selection_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    artifact_paths: Mapping[str, Any],
    duplicate_probe_selection_count: int,
    selection_warning_count: int,
) -> dict[str, Any]:
    selected_df = probe_selection_df[probe_selection_df["selected"] == 1].copy() if not probe_selection_df.empty else pd.DataFrame()
    valid_condition = condition_df[condition_df["valid"] == 1].copy() if not condition_df.empty and "valid" in condition_df.columns else pd.DataFrame()
    corr_delta_x: list[float] = []
    corr_delta_y: list[float] = []
    if not paired_df.empty:
        for x_col, y_col in (
            ("delta_l1_spike_enrichment_intact_vs_flattened", "delta_l2_update_enrichment_intact_vs_flattened_l2reset"),
            ("delta_l1_spike_enrichment_intact_vs_flattened", "delta_l2_update_enrichment_intact_vs_flattened_l2intact"),
            ("delta_l1_spike_enrichment_boosted_vs_flattened", "delta_l2_update_enrichment_boosted_vs_flattened_l2reset"),
            ("delta_l1_spike_enrichment_boosted_vs_flattened", "delta_l2_update_enrichment_boosted_vs_flattened_l2intact"),
        ):
            corr_delta_x.extend(pd.to_numeric(paired_df[x_col], errors="coerce").to_list())
            corr_delta_y.extend(pd.to_numeric(paired_df[y_col], errors="coerce").to_list())
    invalid_reasons = value_counts_dict(condition_df.loc[condition_df["valid"] == 0, "invalid_reason"].fillna("")) if not condition_df.empty else {}
    return {
        "experiment_id": EXPERIMENT_ID,
        "config": json_safe(asdict(cfg)),
        "counts": {
            "num_trials": int(mask_df["trial_id"].nunique()) if not mask_df.empty else 0,
            "valid_trials": int(mask_df["valid"].sum()) if not mask_df.empty and "valid" in mask_df.columns else 0,
            "num_selected_low_overlap_probes": int((selected_df["probe_overlap_group"] == "low_peak_overlap").sum()) if not selected_df.empty else 0,
            "num_selected_high_overlap_probes": int((selected_df["probe_overlap_group"] == "high_peak_overlap").sum()) if not selected_df.empty else 0,
            "invalid_condition_rows": int((condition_df["valid"] == 0).sum()) if not condition_df.empty else 0,
            "duplicate_probe_selection_count": int(duplicate_probe_selection_count),
            "selection_warning_count": int(selection_warning_count),
            "invalid_reason_counts": invalid_reasons,
        },
        "mask_info": {
            "l1_peak_mask_size_mean": safe_mean(mask_df["l1_peak_mask_size"].to_numpy(dtype=np.float64)) if not mask_df.empty else None,
            "l1_nonpeak_mask_size_mean": safe_mean(mask_df["l1_nonpeak_mask_size"].to_numpy(dtype=np.float64)) if not mask_df.empty else None,
            "mean_g_peak": safe_mean(mask_df["mean_g_peak"].to_numpy(dtype=np.float64)) if not mask_df.empty else None,
            "mean_g_nonpeak": safe_mean(mask_df["mean_g_nonpeak"].to_numpy(dtype=np.float64)) if not mask_df.empty else None,
            "peak_nonpeak_ratio_mean": safe_mean(mask_df["peak_nonpeak_ratio"].to_numpy(dtype=np.float64)) if not mask_df.empty else None,
            "l1_output_projection_mode_counts": value_counts_dict(mask_df["l1_output_projection_mode"].to_list()) if not mask_df.empty else {},
            "l2_input_mask_alignment_mode_counts": value_counts_dict(mask_df["l2_input_alignment_mode"].to_list()) if not mask_df.empty else {},
            "l2_output_projection_mode_counts": value_counts_dict(mask_df["l2_output_projection_mode"].to_list()) if not mask_df.empty else {},
        },
        "main_results": {
            "mean_l1_spike_enrichment_by_overlap_l1condition": nested_group_mean(condition_df, ["probe_overlap_group", "l1_condition"], "l1_spike_enrichment"),
            "mean_l2_update_enrichment_by_l2memory_l1condition": nested_group_mean(condition_df, ["l2_memory_condition", "l1_condition"], "l2_update_enrichment"),
            "mean_l2_update_difference_by_l2memory_l1condition": nested_group_mean(condition_df, ["l2_memory_condition", "l1_condition"], "l2_update_difference"),
            "mean_l2_update_count_enrichment_by_l2memory_l1condition": nested_group_mean(condition_df, ["l2_memory_condition", "l1_condition"], "l2_update_count_enrichment"),
            "mean_l2_spike_enrichment_by_l2memory_l1condition": nested_group_mean(condition_df, ["l2_memory_condition", "l1_condition"], "l2_spike_enrichment"),
        },
        "paired_results": {
            "mean_delta_l2_update_enrichment_intact_vs_flattened_l2reset": safe_mean(paired_df["delta_l2_update_enrichment_intact_vs_flattened_l2reset"].to_numpy(dtype=np.float64)) if not paired_df.empty else None,
            "mean_delta_l2_update_enrichment_intact_vs_flattened_l2intact": safe_mean(paired_df["delta_l2_update_enrichment_intact_vs_flattened_l2intact"].to_numpy(dtype=np.float64)) if not paired_df.empty else None,
            "mean_delta_l2_update_enrichment_boosted_vs_flattened_l2reset": safe_mean(paired_df["delta_l2_update_enrichment_boosted_vs_flattened_l2reset"].to_numpy(dtype=np.float64)) if not paired_df.empty else None,
            "mean_delta_l2_update_enrichment_boosted_vs_flattened_l2intact": safe_mean(paired_df["delta_l2_update_enrichment_boosted_vs_flattened_l2intact"].to_numpy(dtype=np.float64)) if not paired_df.empty else None,
            "mean_interaction_l1_intact_effect_by_l2_memory": safe_mean(paired_df["interaction_l1_intact_effect_by_l2_memory"].to_numpy(dtype=np.float64)) if not paired_df.empty else None,
            "mean_interaction_l1_boosted_effect_by_l2_memory": safe_mean(paired_df["interaction_l1_boosted_effect_by_l2_memory"].to_numpy(dtype=np.float64)) if not paired_df.empty else None,
        },
        "correlations": {
            "corr_delta_l1_spike_enrichment_vs_delta_l2_update_enrichment": finite_corr(corr_delta_x, corr_delta_y),
            "corr_l1_spike_enrichment_vs_l2_update_enrichment": finite_corr(valid_condition["l1_spike_enrichment"].to_numpy(dtype=np.float64), valid_condition["l2_update_enrichment"].to_numpy(dtype=np.float64)) if not valid_condition.empty else None,
        },
        "artifacts": json_safe(artifact_paths),
        "reset_scope": L2_RESET_SCOPE,
        "smoke_mode": bool(cfg.smoke),
    }


def rel_files(root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in sorted(root.rglob("*")) if path.is_file()]


def write_artifact_manifest(layout: Any, artifact_paths: Mapping[str, Any]) -> Path:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "files": rel_files(layout.root),
        "artifact_paths": json_safe(artifact_paths),
    }
    out_path = layout.root / "artifact_manifest.json"
    out_path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


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
        trials, sequence_df = build_sequence_trials(class_index, labels, cfg)
        probe_candidates = build_probe_candidates(trials, labels, cfg)
        all_image_ids: set[int] = set()
        for trial in trials:
            all_image_ids.update(int(item) for item in trial.ordered_item_ids)
            all_image_ids.update(int(item) for item in probe_candidates[int(trial.trial_id)])
        net, encoder = load_model_and_encoder(cfg.model_path, device=device, dt=cfg.dt, max_duration_ms=cfg.max_duration_ms)
        spike_lookup = build_spike_lookup(images, encoder, all_image_ids, cfg, device)
        baseline_gain_l1 = float(net.layer1.stsp_U)
        log_and_print(
            log_lines,
            f"[{EXPERIMENT_ID}] trials={len(trials)} sample_steps={cfg.sample_steps} delay_steps={cfg.delay_steps}",
        )

        mask_rows: list[dict[str, object]] = []
        probe_selection_rows: list[dict[str, object]] = []
        condition_rows: list[dict[str, object]] = []
        update_map_rows: list[dict[str, object]] = []
        duplicate_probe_selection_count = 0
        selection_warning_count = 0
        projection_info_last: dict[str, Any] = {}

        for batch in build_batches(trials, spike_lookup, cfg):
            first_item = batch.item_spikes[0]
            batch_size, _steps, channels, height, width = first_item.shape
            layer_input_shapes = prepare_clean_network_state(net, batch_size, channels, height, width)
            zero_input = torch.zeros((batch_size, channels, height, width), dtype=first_item.dtype, device=first_item.device)
            current_time = 0
            for item_spikes in batch.item_spikes:
                current_time = run_input_window(net, item_spikes, current_time)
                current_time = run_zero_window(net, zero_input, int(cfg.delay_steps), current_time)
            final_snapshot = snapshot_model_state(net, current_time=current_time, layer_input_shapes=layer_input_shapes)
            _u_l1, _x_l1, g_l1 = layer_u_x_g_flat_from_snapshot(final_snapshot, "layer1")
            l1_input_masks = compute_layer1_peak_masks(
                g_l1,
                baseline_gain=baseline_gain_l1,
                epsilon=float(cfg.epsilon),
                peak_q=float(cfg.peak_q),
            )
            l1_output_masks = project_l1_input_mask_to_l1_output(net, final_snapshot, l1_input_masks)
            l2_input_masks = align_l1_output_mask_to_l2_input(final_snapshot, l1_output_masks)
            l2_output_masks = project_l2_input_mask_to_l2_output(net, final_snapshot, l2_input_masks)
            projection_info_last = {
                "l1_output_projection": l1_output_masks.projection_info,
                "l2_input_alignment": l2_input_masks.projection_info,
                "l2_output_projection": l2_output_masks.projection_info,
            }

            for row_idx, trial in enumerate(batch.trials):
                valid_trial = bool(l1_input_masks.valid[row_idx]) and bool(l1_output_masks.valid[row_idx]) and bool(l2_input_masks.valid[row_idx])
                invalid_reason = ";".join(
                    item
                    for item in (
                        l1_input_masks.invalid_reason[row_idx],
                        l1_output_masks.invalid_reason[row_idx],
                        l2_input_masks.invalid_reason[row_idx],
                    )
                    if item
                )
                mask_rows.append(
                    {
                        "trial_id": int(trial.trial_id),
                        "seq_len": int(trial.seq_len),
                        "l1_peak_mask_size": int(l1_input_masks.num_peak[row_idx]),
                        "l1_nonpeak_mask_size": int(l1_input_masks.num_nonpeak[row_idx]),
                        "mean_g_peak": float(l1_input_masks.mean_g_peak[row_idx]),
                        "mean_g_nonpeak": float(l1_input_masks.mean_g_nonpeak[row_idx]),
                        "peak_nonpeak_ratio": float(l1_input_masks.peak_nonpeak_ratio[row_idx]),
                        "l1_output_projection_mode": str(l1_output_masks.mask_mode),
                        "l1_peak_output_units": int(l1_output_masks.num_peak[row_idx]),
                        "l1_nonpeak_output_units": int(l1_output_masks.num_nonpeak[row_idx]),
                        "l2_input_alignment_mode": str(l2_input_masks.mask_mode),
                        "l2_peak_input_units": int(l2_input_masks.num_peak[row_idx]),
                        "l2_nonpeak_input_units": int(l2_input_masks.num_nonpeak[row_idx]),
                        "l2_output_projection_mode": str(l2_output_masks.mask_mode),
                        "l2_peak_output_units": int(l2_output_masks.num_peak[row_idx]),
                        "l2_nonpeak_output_units": int(l2_output_masks.num_nonpeak[row_idx]),
                        "valid": int(valid_trial),
                        "invalid_reason": invalid_reason,
                    }
                )
                candidate_rows, selected_rows, selection_info = select_low_high_peak_overlap_probes(
                    trial,
                    probe_candidates[int(trial.trial_id)],
                    labels,
                    spike_lookup,
                    l1_input_masks.peak[row_idx],
                    l1_input_masks.nonpeak[row_idx],
                    cfg,
                )
                duplicate_probe_selection_count += int(selection_info.get("duplicate_probe_selection_count", 0))
                selection_warning_count += int(selection_info.get("selection_warning_count", 0))
                probe_selection_rows.extend(candidate_rows)

                for selected in selected_rows:
                    probe_id = int(selected["candidate_image_id"])
                    for l1_condition in L1_CONDITIONS:
                        for l2_memory_condition in L2_MEMORY_CONDITIONS:
                            condition_snapshot, clipping_fraction, condition_invalid = build_condition_snapshot(
                                net,
                                final_snapshot,
                                row_idx,
                                l1_input_masks,
                                l1_condition=l1_condition,
                                l2_memory_condition=l2_memory_condition,
                                cfg=cfg,
                            )
                            metrics: dict[str, Any] = {}
                            update_map: dict[str, Any] = {}
                            valid = int(valid_trial and not condition_invalid)
                            invalid_parts = [invalid_reason, condition_invalid]
                            if valid:
                                metrics, update_map = run_probe_actual_and_decay(
                                    net,
                                    condition_snapshot,
                                    spike_lookup[probe_id],
                                    row_idx,
                                    l1_output_masks,
                                    l2_input_masks,
                                    l2_output_masks,
                                    cfg,
                                )
                                valid = int(
                                    int(metrics.get("l1_valid", 0)) == 1
                                    and int(metrics.get("l2_update_valid", 0)) == 1
                                )
                                invalid_parts.extend(
                                    [
                                        str(metrics.get("l1_invalid_reason", "")),
                                        str(metrics.get("l2_update_invalid_reason", "")),
                                    ]
                                )
                            invalid_reason_row = ";".join(item for item in invalid_parts if item)
                            condition_row = {
                                "trial_id": int(trial.trial_id),
                                "probe_image_id": int(probe_id),
                                "probe_label": int(labels[probe_id]),
                                "probe_overlap_group": str(selected["probe_overlap_group"]),
                                "probe_peak_overlap_fraction": float(selected["probe_peak_overlap_fraction"]),
                                "probe_nonpeak_overlap_fraction": float(selected["probe_nonpeak_overlap_fraction"]),
                                "l1_condition": str(l1_condition),
                                "l1_boost_level": float(cfg.boost_level) if l1_condition == "l1_peak_boosted" else 0.0,
                                "l2_memory_condition": str(l2_memory_condition),
                                "l2_reset_scope": L2_RESET_SCOPE if l2_memory_condition == "l2_reset" else "none",
                                "clipping_fraction": float(clipping_fraction) if np.isfinite(clipping_fraction) else np.nan,
                                "valid": int(valid),
                                "invalid_reason": invalid_reason_row,
                            }
                            condition_row.update(
                                {
                                    "l1_spike_enrichment": metrics.get("l1_spike_enrichment", np.nan),
                                    "l1_peak_spike_fraction": metrics.get("l1_peak_spike_fraction", np.nan),
                                    "l1_total_spikes": metrics.get("l1_total_spikes", np.nan),
                                    "l2_update_enrichment": metrics.get("l2_update_enrichment", np.nan),
                                    "l2_update_difference": metrics.get("l2_update_difference", np.nan),
                                    "l2_peak_input_update_mean": metrics.get("l2_peak_input_update_mean", np.nan),
                                    "l2_nonpeak_input_update_mean": metrics.get("l2_nonpeak_input_update_mean", np.nan),
                                    "l2_peak_input_signed_update_mean": metrics.get("l2_peak_input_signed_update_mean", np.nan),
                                    "l2_nonpeak_input_signed_update_mean": metrics.get("l2_nonpeak_input_signed_update_mean", np.nan),
                                    "l2_signed_update_difference": metrics.get("l2_signed_update_difference", np.nan),
                                    "l2_update_count_enrichment": metrics.get("l2_update_count_enrichment", np.nan),
                                    "l2_peak_input_update_count_density": metrics.get("l2_peak_input_update_count_density", np.nan),
                                    "l2_nonpeak_input_update_count_density": metrics.get("l2_nonpeak_input_update_count_density", np.nan),
                                    "l2_spike_enrichment": metrics.get("l2_spike_enrichment", np.nan),
                                    "l2_total_spikes": metrics.get("l2_total_spikes", np.nan),
                                    "l2_spike_valid": metrics.get("l2_valid", 0),
                                    "l2_spike_invalid_reason": metrics.get("l2_invalid_reason", ""),
                                }
                            )
                            condition_rows.append(condition_row)
                            if cfg.save_update_maps:
                                update_map_rows.append(
                                    {
                                        "trial_id": int(trial.trial_id),
                                        "probe_image_id": int(probe_id),
                                        "probe_overlap_group": str(selected["probe_overlap_group"]),
                                        "l1_condition": str(l1_condition),
                                        "l2_memory_condition": str(l2_memory_condition),
                                        **update_map,
                                    }
                                )
            log_and_print(log_lines, f"[{EXPERIMENT_ID}] finished batch {batch.batch_id}")

        sequence_csv = save_tidy_csv(sequence_df, layout.data_file("layer2_downstream_sequence_items.csv"), sort_by=["trial_id", "item_index"])
        mask_df = pd.DataFrame(mask_rows)
        probe_selection_df = pd.DataFrame(probe_selection_rows)
        condition_df = pd.DataFrame(condition_rows)
        paired_df = build_paired_effects(condition_df)
        update_map_df = pd.DataFrame(update_map_rows)

        artifact_paths: dict[str, Any] = {
            "layer2_downstream_sequence_items": sequence_csv,
            "projection_info": projection_info_last,
            "layer2_downstream_mask_summary": save_tidy_csv(
                mask_df,
                layout.data_file("layer2_downstream_mask_summary.csv"),
                sort_by=["trial_id"] if not mask_df.empty else None,
            ),
            "layer2_downstream_probe_selection": save_tidy_csv(
                probe_selection_df,
                layout.data_file("layer2_downstream_probe_selection.csv"),
                sort_by=["trial_id", "selected", "probe_overlap_group", "selection_rank"] if not probe_selection_df.empty else None,
            ),
            "layer2_downstream_condition_summary": save_tidy_csv(
                condition_df,
                layout.data_file("layer2_downstream_condition_summary.csv"),
                sort_by=["trial_id", "probe_image_id", "probe_overlap_group", "l1_condition", "l2_memory_condition"] if not condition_df.empty else None,
            ),
            "layer2_downstream_paired_effects": save_tidy_csv(
                paired_df,
                layout.data_file("layer2_downstream_paired_effects.csv"),
                sort_by=["trial_id", "probe_image_id", "probe_overlap_group"] if not paired_df.empty else None,
            ),
        }
        if cfg.save_update_maps:
            artifact_paths["layer2_downstream_update_map_summary"] = save_tidy_csv(
                update_map_df,
                layout.data_file("layer2_downstream_update_map_summary.csv"),
                sort_by=["trial_id", "probe_image_id", "l1_condition", "l2_memory_condition"] if not update_map_df.empty else None,
            )
        if not cfg.skip_figures:
            artifact_paths["figures"] = generate_figures(
                layout,
                probe_selection_df=probe_selection_df,
                condition_df=condition_df,
                paired_df=paired_df,
            )
        summary = summarize_results(
            cfg,
            mask_df=mask_df,
            probe_selection_df=probe_selection_df,
            condition_df=condition_df,
            paired_df=paired_df,
            artifact_paths=artifact_paths,
            duplicate_probe_selection_count=duplicate_probe_selection_count,
            selection_warning_count=selection_warning_count,
        )
        artifact_paths["run_config"] = str(save_run_config(json_safe(asdict(cfg)), layout.root))
        artifact_paths["summary"] = str(save_summary_json(summary, layout.root))
        save_summary_json({"projection_info": projection_info_last}, layout.meta_dir, filename="projection_info.json")
        save_log_lines(log_lines, layout.logs_dir, filename="run.log")
        manifest_path = write_artifact_manifest(layout, artifact_paths)
        artifact_paths["artifact_manifest"] = str(manifest_path)
        summary["artifacts"] = json_safe(artifact_paths)
        save_summary_json(summary, layout.root)
        status = "success"
        return summary
    finally:
        finalize_run_info(layout.meta_dir, run_info, status=status)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    cfg = normalize_config(parser.parse_args(argv))
    run_experiment(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
