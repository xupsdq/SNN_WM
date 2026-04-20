from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.paths import DEFAULT_PATH_CONFIG
from src.config.units import ms
from src.experiments.common.dataset import build_class_index, build_dataset_arrays, encode_images
from src.experiments.common.distractor_triplets import build_triplet_specs, load_mnist_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.common.runtime import seed_everything
from src.plotting.common.io import (
    COLOR_DISTRACTOR,
    COLOR_DYNAMIC,
    COLOR_SAMPLE_ALIGNED,
    COLOR_STATIC,
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
)

CONDITION_BASELINE = "baseline_intact"
CONDITION_DISTRACTOR_ONLY = "distractor_only_reference"
CONDITION_SAMPLE_ONLY = "sample_only_reference"
CONDITION_FORMATION_CLAMP = "formation_clamp"
CONDITION_ORDER = (
    CONDITION_BASELINE,
    CONDITION_DISTRACTOR_ONLY,
    CONDITION_SAMPLE_ONLY,
    CONDITION_FORMATION_CLAMP,
)
LAYER_ORDER = ("layer2", "layer3")
REWRITE_LAYER = "layer2"
PATTERN_WINDOW_NAMES = ("early", "late")
PATTERN_POOL_SHAPE = (2, 2)
EXAMPLE_TOP_GROUPS = 24


@dataclass(frozen=True)
class ExperimentConfig:
    model_path: str
    dataset_root: str
    split: str
    device_request: str
    seed: int
    output_dir: str
    sample_ms: float
    delay1_ms: float
    distractor_ms: float
    delay2_ms: float
    batch_size: int
    max_probes: int
    samples_per_probe: int
    max_triplets: int
    num_sim_bins: int
    skip_figures: bool
    smoke: bool
    dt: float = 1.0 * ms

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def delay1_steps(self) -> int:
        return int(round((self.delay1_ms * ms) / self.dt))

    @property
    def distractor_steps(self) -> int:
        return int(round((self.distractor_ms * ms) / self.dt))

    @property
    def delay2_steps(self) -> int:
        return int(round((self.delay2_ms * ms) / self.dt))

    @property
    def max_duration_ms(self) -> float:
        return float(max(self.sample_ms, self.distractor_ms, 100.0))

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).resolve()

    @property
    def distractor_time_ms(self) -> np.ndarray:
        return np.arange(self.distractor_steps, dtype=np.float64) * (self.dt / ms)


@dataclass(frozen=True)
class BatchSpec:
    batch_df: pd.DataFrame
    sample_spikes: torch.Tensor
    distractor_spikes: torch.Tensor
    zero_sample_spikes: torch.Tensor
    zero_distractor_spikes: torch.Tensor


@dataclass(frozen=True)
class RolloutCapture:
    condition: str
    rewrite_features: Dict[str, np.ndarray]
    preprobe: Dict[str, Dict[str, np.ndarray]]
    intervention_record: Dict[str, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 4 chunk-formation experiment.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_PATH_CONFIG.model_path))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="results/chunk_step4_formation")
    parser.add_argument("--sample-ms", type=float, default=100.0)
    parser.add_argument("--delay1-ms", type=float, default=100.0)
    parser.add_argument("--distractor-ms", type=float, default=100.0)
    parser.add_argument("--delay2-ms", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-probes", type=int, default=20)
    parser.add_argument("--samples-per-probe", type=int, default=5)
    parser.add_argument("--max-triplets", type=int, default=100)
    parser.add_argument("--num-sim-bins", type=int, default=4)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig(
        model_path=str(args.model_path),
        dataset_root=str(args.dataset_root),
        split=str(args.split),
        device_request=str(args.device),
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        sample_ms=float(args.sample_ms),
        delay1_ms=float(args.delay1_ms),
        distractor_ms=float(args.distractor_ms),
        delay2_ms=float(args.delay2_ms),
        batch_size=int(args.batch_size),
        max_probes=int(args.max_probes),
        samples_per_probe=int(args.samples_per_probe),
        max_triplets=int(args.max_triplets),
        num_sim_bins=int(args.num_sim_bins),
        skip_figures=bool(args.skip_figures),
        smoke=bool(args.smoke),
    )
    if config.smoke:
        config = ExperimentConfig(
            **{
                **asdict(config),
                "batch_size": min(config.batch_size, 2),
                "max_probes": min(config.max_probes, 2),
                "samples_per_probe": min(config.samples_per_probe, 1),
                "max_triplets": min(config.max_triplets, 4),
            }
        )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if not Path(config.model_path).exists():
        raise FileNotFoundError(f"Model not found: {config.model_path}")
    if not Path(config.dataset_root).exists():
        raise FileNotFoundError(f"Dataset root not found: {config.dataset_root}")
    if config.split not in {"train", "test"}:
        raise ValueError("--split must be train or test")
    numeric_positive = {
        "sample_ms": config.sample_ms,
        "delay1_ms": config.delay1_ms,
        "distractor_ms": config.distractor_ms,
        "delay2_ms": config.delay2_ms,
        "batch_size": config.batch_size,
        "max_probes": config.max_probes,
        "samples_per_probe": config.samples_per_probe,
        "max_triplets": config.max_triplets,
        "num_sim_bins": config.num_sim_bins,
    }
    for key, value in numeric_positive.items():
        if float(value) <= 0:
            raise ValueError(f"{key} must be positive, got {value}")
    if min(config.sample_steps, config.delay1_steps, config.distractor_steps, config.delay2_steps) <= 0:
        raise ValueError("All phase durations must resolve to at least one step.")


def emit(message: str, log_lines: list[str]) -> None:
    line = str(message)
    print(line, flush=True)
    log_lines.append(line)


def resolve_device_with_fallback(device_request: str, log_lines: list[str]) -> torch.device:
    if device_request == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emit(f"[Init] device_request=auto resolved_device={device}", log_lines)
        return device
    requested = torch.device(device_request)
    if requested.type == "cuda" and not torch.cuda.is_available():
        emit("[Init] CUDA requested but unavailable; falling back to CPU", log_lines)
        return torch.device("cpu")
    emit(f"[Init] device_request={device_request} resolved_device={requested}", log_lines)
    return requested


def print_config_summary(config: ExperimentConfig, device: torch.device, log_lines: list[str]) -> None:
    emit(
        (
            "[Config] "
            f"split={config.split} smoke={int(config.smoke)} "
            f"batch_size={config.batch_size} max_triplets={config.max_triplets} "
            f"max_probes={config.max_probes} samples_per_probe={config.samples_per_probe} "
            f"timing_ms=({config.sample_ms},{config.delay1_ms},{config.distractor_ms},{config.delay2_ms}) "
            f"device={device}"
        ),
        log_lines,
    )


def _stack_encoded_ids(
    image_ids: Sequence[int],
    *,
    images: torch.Tensor,
    encoder,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    batch_images = images[[int(idx) for idx in image_ids]].to(device=device, dtype=torch.float32)
    return encode_images(encoder, batch_images, steps=int(steps))


def prepare_triplet_batches(
    df_triplets: pd.DataFrame,
    images: torch.Tensor,
    *,
    encoder,
    config: ExperimentConfig,
    device: torch.device,
) -> Iterator[BatchSpec]:
    for start in range(0, len(df_triplets), config.batch_size):
        batch_df = df_triplets.iloc[start : start + config.batch_size].copy().reset_index(drop=True)
        sample_ids = batch_df["sample_id"].astype(int).tolist()
        distractor_ids = batch_df["distractor_id"].astype(int).tolist()
        unique_sample_ids = list(dict.fromkeys(sample_ids))
        unique_distractor_ids = list(dict.fromkeys(distractor_ids))
        encoded_samples = _stack_encoded_ids(
            unique_sample_ids,
            images=images,
            encoder=encoder,
            steps=config.sample_steps,
            device=device,
        )
        encoded_distractors = _stack_encoded_ids(
            unique_distractor_ids,
            images=images,
            encoder=encoder,
            steps=config.distractor_steps,
            device=device,
        )
        sample_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_sample_ids)}
        distractor_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_distractor_ids)}
        sample_select = torch.as_tensor([sample_lookup[int(idx)] for idx in sample_ids], dtype=torch.long, device=device)
        distractor_select = torch.as_tensor(
            [distractor_lookup[int(idx)] for idx in distractor_ids],
            dtype=torch.long,
            device=device,
        )
        sample_spikes = encoded_samples.index_select(0, sample_select)
        distractor_spikes = encoded_distractors.index_select(0, distractor_select)
        yield BatchSpec(
            batch_df=batch_df,
            sample_spikes=sample_spikes,
            distractor_spikes=distractor_spikes,
            zero_sample_spikes=torch.zeros_like(sample_spikes),
            zero_distractor_spikes=torch.zeros_like(distractor_spikes),
        )


def _window_slices(num_steps: int, window_names: Sequence[str] = PATTERN_WINDOW_NAMES) -> list[tuple[str, int, int]]:
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive.")
    names = list(window_names)
    edges = np.linspace(0, int(num_steps), num=len(names) + 1, dtype=np.int64)
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1
    edges[-1] = int(num_steps)
    windows: list[tuple[str, int, int]] = []
    for idx, name in enumerate(names):
        start = int(edges[idx])
        end = int(edges[idx + 1])
        if end <= start:
            end = min(int(num_steps), start + 1)
        windows.append((str(name), start, end))
    return windows


def _window_lookup_map(num_steps: int) -> tuple[list[tuple[str, int, int]], np.ndarray]:
    windows = _window_slices(num_steps)
    lookup = np.empty(int(num_steps), dtype=np.int64)
    for win_idx, (_, start, end) in enumerate(windows):
        lookup[start:end] = int(win_idx)
    return windows, lookup


def _spatial_bin_edges(size: int, bins: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, int(size), num=int(bins) + 1, dtype=np.int64)
    ranges: list[tuple[int, int]] = []
    for idx in range(int(bins)):
        start = int(edges[idx])
        end = int(edges[idx + 1])
        if end <= start:
            end = min(int(size), start + 1)
        ranges.append((start, end))
    return ranges


def _pool_layer2_spikes(spikes: torch.Tensor, pool_shape: tuple[int, int] = PATTERN_POOL_SHAPE) -> torch.Tensor:
    batch_size, channels, height, width = spikes.shape
    row_bins = _spatial_bin_edges(height, int(pool_shape[0]))
    col_bins = _spatial_bin_edges(width, int(pool_shape[1]))
    blocks: list[torch.Tensor] = []
    spikes_f = spikes.to(torch.float32)
    for row_start, row_end in row_bins:
        for col_start, col_end in col_bins:
            block_sum = spikes_f[:, :, row_start:row_end, col_start:col_end].sum(dim=(-1, -2))
            blocks.append(block_sum)
    if not blocks:
        raise RuntimeError("No pooled spike blocks were created for layer2.")
    stacked = torch.stack(blocks, dim=2)
    return stacked.reshape(batch_size, channels * len(blocks))


def _to_numpy_float(x: torch.Tensor) -> np.ndarray:
    return x.detach().to(torch.float32).cpu().numpy().astype(np.float64, copy=False)


def _snapshot_layer_state(layer) -> Dict[str, np.ndarray]:
    if getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
        raise ValueError("Requested pre-probe snapshot from a layer without STSP state.")
    u = layer.u_pre.detach().to(torch.float32).cpu().reshape(layer.u_pre.shape[0], -1).numpy().astype(np.float64, copy=False)
    x = layer.x_pre.detach().to(torch.float32).cpu().reshape(layer.x_pre.shape[0], -1).numpy().astype(np.float64, copy=False)
    ux = (layer.u_pre * layer.x_pre).detach().to(torch.float32).cpu().reshape(layer.u_pre.shape[0], -1).numpy().astype(
        np.float64,
        copy=False,
    )
    feature = np.concatenate([u, x, ux], axis=1)
    return {"u": u, "x": x, "ux": ux, "feature": feature}


def build_clamp_sample_trace_before_distractor_fn(
    target_layers: Sequence[str] = ("layer2", "layer3"),
) -> Callable[[torch.nn.Module, Mapping[str, object]], Dict[str, float]]:
    valid_layers = tuple(str(layer_key) for layer_key in target_layers)

    def _clamp_fn(net, ctx: Mapping[str, object]) -> Dict[str, float]:
        del ctx
        record: Dict[str, float] = {"clamp_applied": 1.0}
        with torch.no_grad():
            for layer_key in valid_layers:
                layer = getattr(net, layer_key, None)
                if layer is None or getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
                    record[f"{layer_key}_clamped"] = 0.0
                    continue
                record[f"{layer_key}_u_mean_before"] = float(layer.u_pre.mean().item())
                record[f"{layer_key}_x_mean_before"] = float(layer.x_pre.mean().item())
                layer.u_pre.fill_(float(layer.stsp_U))
                layer.x_pre.fill_(1.0)
                record[f"{layer_key}_u_mean_after"] = float(layer.u_pre.mean().item())
                record[f"{layer_key}_x_mean_after"] = float(layer.x_pre.mean().item())
                record[f"{layer_key}_clamped"] = 1.0
        return record

    return _clamp_fn


def run_formation_rollout_capture(
    net,
    sample_spikes: torch.Tensor,
    distractor_spikes: torch.Tensor,
    *,
    condition: str,
    config: ExperimentConfig,
    before_distractor_fn: Callable[[torch.nn.Module, Mapping[str, object]], Dict[str, float]] | None = None,
    stsp_mode: str = "dynamic",
    rewrite_capture_phase: str = "distractor",
) -> RolloutCapture:
    batch_size, _, channels, height, width = sample_spikes.shape
    prepare_network_state(net, batch_size, channels, height, width)
    zero_input = torch.zeros((batch_size, channels, height, width), dtype=sample_spikes.dtype, device=sample_spikes.device)
    current_time = 0
    intervention_record: Dict[str, float] = {}
    capture_phase = str(rewrite_capture_phase).strip().lower()
    if capture_phase not in {"sample", "distractor"}:
        raise ValueError(f"Unsupported rewrite_capture_phase: {rewrite_capture_phase}")
    capture_steps = config.sample_steps if capture_phase == "sample" else config.distractor_steps
    windows, window_lookup = _window_lookup_map(capture_steps)
    pattern_counts: torch.Tensor | None = None
    total_counts: torch.Tensor | None = None
    first_spike_step: torch.Tensor | None = None

    def step_network(input_t: torch.Tensor, *, capture_step_index: int | None = None) -> None:
        nonlocal current_time
        nonlocal pattern_counts
        nonlocal total_counts
        nonlocal first_spike_step
        with torch.no_grad():
            s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=True, stsp_mode=stsp_mode)
            s1_p = net.pool1(s1.float())
            s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, monitor=True, stsp_mode=stsp_mode)
            s2_p = net.pool2(s2.float())
            _ = net.layer3.forward_step(s2_p, current_time, training=False, monitor=True, stsp_mode=stsp_mode)
            if capture_step_index is not None:
                pooled = _pool_layer2_spikes(s2)
                if pattern_counts is None:
                    num_groups = int(pooled.shape[1])
                    pattern_counts = torch.zeros(
                        (batch_size, len(windows), num_groups),
                        dtype=torch.float32,
                        device=pooled.device,
                    )
                    total_counts = torch.zeros((batch_size, num_groups), dtype=torch.float32, device=pooled.device)
                    first_spike_step = torch.full(
                        (batch_size, num_groups),
                        fill_value=float(config.distractor_steps),
                        dtype=torch.float32,
                        device=pooled.device,
                    )
                window_idx = int(window_lookup[int(capture_step_index)])
                pattern_counts[:, window_idx, :] += pooled
                total_counts += pooled
                active_mask = pooled > 0.0
                step_fill = torch.full_like(first_spike_step, fill_value=float(capture_step_index))
                first_spike_step = torch.where(active_mask, torch.minimum(first_spike_step, step_fill), first_spike_step)
        current_time += 1

    for t_step in range(sample_spikes.shape[1]):
        step_network(
            sample_spikes[:, t_step, ...],
            capture_step_index=(int(t_step) if capture_phase == "sample" else None),
        )
    for _ in range(config.delay1_steps):
        step_network(zero_input, capture_step_index=None)
    if before_distractor_fn is not None:
        intervention_record.update(
            before_distractor_fn(
                net,
                {
                    "condition": condition,
                    "current_time": current_time,
                    "delay1_steps": config.delay1_steps,
                    "distractor_steps": config.distractor_steps,
                },
            )
        )
    for t_step in range(distractor_spikes.shape[1]):
        step_network(
            distractor_spikes[:, t_step, ...],
            capture_step_index=(int(t_step) if capture_phase == "distractor" else None),
        )
    for _ in range(config.delay2_steps):
        step_network(zero_input, capture_step_index=None)

    preprobe = {
        "layer2": _snapshot_layer_state(net.layer2),
        "layer3": _snapshot_layer_state(net.layer3),
    }
    if pattern_counts is None or total_counts is None or first_spike_step is None:
        raise RuntimeError("Distractor-phase spike-pattern capture failed to initialize.")
    pattern_vector = pattern_counts.reshape(batch_size, -1)
    active_mask = total_counts > 0.0
    first_spike_step = torch.where(
        active_mask,
        first_spike_step,
        torch.full_like(first_spike_step, fill_value=-1.0),
    )
    rewrite_features = {
        "window_counts": _to_numpy_float(pattern_counts),
        "pattern_vector": _to_numpy_float(pattern_vector),
        "total_counts": _to_numpy_float(total_counts),
        "first_spike_step": _to_numpy_float(first_spike_step),
        "active_mask": _to_numpy_float(active_mask.to(torch.float32)),
        "window_total_counts": _to_numpy_float(pattern_counts.sum(dim=2)),
        "window_index_lookup": window_lookup.astype(np.int64, copy=False),
        "capture_phase": np.asarray([capture_phase] * batch_size, dtype=object),
    }
    return RolloutCapture(
        condition=condition,
        rewrite_features=rewrite_features,
        preprobe=preprobe,
        intervention_record=intervention_record,
    )


def _concat_rows(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def collect_condition_captures(
    net,
    images: torch.Tensor,
    df_triplets: pd.DataFrame,
    *,
    encoder,
    config: ExperimentConfig,
    device: torch.device,
    log_lines: list[str],
) -> dict[str, Dict[str, object]]:
    bank: dict[str, Dict[str, object]] = {
        condition: {
            "triplet_ids": [],
            "rewrite": {
                key: []
                for key in (
                    "window_counts",
                    "pattern_vector",
                    "total_counts",
                    "first_spike_step",
                    "active_mask",
                    "window_total_counts",
                    "capture_phase",
                )
            },
            "preprobe": {
                layer_key: {state_name: [] for state_name in ("u", "x", "ux", "feature")}
                for layer_key in LAYER_ORDER
            },
            "intervention_rows": [],
        }
        for condition in CONDITION_ORDER
    }
    clamp_fn = build_clamp_sample_trace_before_distractor_fn()
    for batch_index, batch in enumerate(prepare_triplet_batches(df_triplets, images, encoder=encoder, config=config, device=device), start=1):
        emit(
            f"[Run] batch={batch_index} triplets={len(batch.batch_df)} conditions={','.join(CONDITION_ORDER)}",
            log_lines,
        )
        condition_inputs = {
            CONDITION_BASELINE: (batch.sample_spikes, batch.distractor_spikes, None, "distractor"),
            CONDITION_DISTRACTOR_ONLY: (batch.zero_sample_spikes, batch.distractor_spikes, None, "distractor"),
            CONDITION_SAMPLE_ONLY: (batch.sample_spikes, batch.zero_distractor_spikes, None, "sample"),
            CONDITION_FORMATION_CLAMP: (batch.sample_spikes, batch.distractor_spikes, clamp_fn, "distractor"),
        }
        for condition, (sample_in, distractor_in, intervention_fn, rewrite_capture_phase) in condition_inputs.items():
            capture = run_formation_rollout_capture(
                net,
                sample_in,
                distractor_in,
                condition=condition,
                config=config,
                before_distractor_fn=intervention_fn,
                rewrite_capture_phase=rewrite_capture_phase,
            )
            bank[condition]["triplet_ids"].append(batch.batch_df["triplet_id"].to_numpy(dtype=np.int64, copy=False))
            for key in bank[condition]["rewrite"]:
                bank[condition]["rewrite"][key].append(capture.rewrite_features[key])
            for layer_key in LAYER_ORDER:
                for state_name in ("u", "x", "ux", "feature"):
                    bank[condition]["preprobe"][layer_key][state_name].append(capture.preprobe[layer_key][state_name])
            for triplet_id in batch.batch_df["triplet_id"].astype(int).tolist():
                row = {"triplet_id": int(triplet_id), "condition": condition}
                for key, value in capture.intervention_record.items():
                    row[key] = float(value)
                bank[condition]["intervention_rows"].append(row)
    for condition in CONDITION_ORDER:
        bank[condition]["triplet_ids"] = _concat_rows(bank[condition]["triplet_ids"]).astype(np.int64, copy=False)
        for key in bank[condition]["rewrite"]:
            bank[condition]["rewrite"][key] = _concat_rows(bank[condition]["rewrite"][key])
        bank[condition]["rewrite"]["window_names"] = np.asarray(PATTERN_WINDOW_NAMES, dtype=object)
        for layer_key in LAYER_ORDER:
            for state_name in ("u", "x", "ux", "feature"):
                bank[condition]["preprobe"][layer_key][state_name] = _concat_rows(bank[condition]["preprobe"][layer_key][state_name])
        bank[condition]["intervention_rows"] = pd.DataFrame(bank[condition]["intervention_rows"])
    return bank


def _rowwise_projection_index(mixed: np.ndarray, distractor_only: np.ndarray, sample_only: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    sample_axis = sample_only - distractor_only
    mixed_shift = mixed - distractor_only
    numerator = np.sum(mixed_shift * sample_axis, axis=1)
    denominator = np.sum(sample_axis * sample_axis, axis=1) + float(eps)
    return numerator / denominator


def _rowwise_cosine_safe(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    numerator = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / np.maximum(denom, float(eps))


def compute_reshaping_metrics(
    df_triplets: pd.DataFrame,
    condition_bank: Mapping[str, Dict[str, object]],
    *,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timeseries_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    triplet_ids = df_triplets["triplet_id"].to_numpy(dtype=np.int64, copy=False)
    mixed_counts = np.asarray(condition_bank[CONDITION_BASELINE]["rewrite"]["window_counts"], dtype=np.float64)
    distractor_counts = np.asarray(condition_bank[CONDITION_DISTRACTOR_ONLY]["rewrite"]["window_counts"], dtype=np.float64)
    sample_counts = np.asarray(condition_bank[CONDITION_SAMPLE_ONLY]["rewrite"]["window_counts"], dtype=np.float64)
    mixed_vector = np.asarray(condition_bank[CONDITION_BASELINE]["rewrite"]["pattern_vector"], dtype=np.float64)
    distractor_vector = np.asarray(condition_bank[CONDITION_DISTRACTOR_ONLY]["rewrite"]["pattern_vector"], dtype=np.float64)
    sample_vector = np.asarray(condition_bank[CONDITION_SAMPLE_ONLY]["rewrite"]["pattern_vector"], dtype=np.float64)
    mixed_total = np.asarray(condition_bank[CONDITION_BASELINE]["rewrite"]["total_counts"], dtype=np.float64)
    distractor_total = np.asarray(condition_bank[CONDITION_DISTRACTOR_ONLY]["rewrite"]["total_counts"], dtype=np.float64)
    sample_total = np.asarray(condition_bank[CONDITION_SAMPLE_ONLY]["rewrite"]["total_counts"], dtype=np.float64)
    mixed_first = np.asarray(condition_bank[CONDITION_BASELINE]["rewrite"]["first_spike_step"], dtype=np.float64)
    distractor_first = np.asarray(condition_bank[CONDITION_DISTRACTOR_ONLY]["rewrite"]["first_spike_step"], dtype=np.float64)
    active_sample = sample_total > 0.0
    if mixed_counts.shape != distractor_counts.shape or mixed_counts.shape != sample_counts.shape:
        raise ValueError("Window-count shape mismatch across rewrite conditions.")

    sppi = _rowwise_projection_index(mixed_vector, distractor_vector, sample_vector)
    sppc = _rowwise_cosine_safe(mixed_vector - distractor_vector, sample_vector - distractor_vector)
    sim_s = _rowwise_cosine_safe(mixed_vector, sample_vector)
    sim_d = _rowwise_cosine_safe(mixed_vector, distractor_vector)
    delta_sim = sim_s - sim_d
    sppi_early = _rowwise_projection_index(mixed_counts[:, 0, :], distractor_counts[:, 0, :], sample_counts[:, 0, :])
    window_pull = _rowwise_projection_index(
        mixed_counts.reshape(-1, mixed_counts.shape[-1]),
        distractor_counts.reshape(-1, distractor_counts.shape[-1]),
        sample_counts.reshape(-1, sample_counts.shape[-1]),
    ).reshape(len(df_triplets), len(PATTERN_WINDOW_NAMES))
    window_dir = _rowwise_cosine_safe(
        (mixed_counts - distractor_counts).reshape(-1, mixed_counts.shape[-1]),
        (sample_counts - distractor_counts).reshape(-1, sample_counts.shape[-1]),
    ).reshape(len(df_triplets), len(PATTERN_WINDOW_NAMES))
    recruit_mask = (distractor_total <= 0.0) & (mixed_total > 0.0) & active_sample
    active_both = (distractor_total > 0.0) & (mixed_total > 0.0) & active_sample
    advance_mask = active_both & (mixed_first >= 0.0) & (distractor_first >= 0.0) & (mixed_first < distractor_first)
    sample_active_count = np.maximum(active_sample.sum(axis=1), 1)
    recruit_ratio = recruit_mask.sum(axis=1) / sample_active_count
    advance_ratio = advance_mask.sum(axis=1) / sample_active_count
    pattern_norm = np.linalg.norm(mixed_vector, axis=1)
    early_total = mixed_counts[:, 0, :].sum(axis=1)
    late_total = mixed_counts[:, -1, :].sum(axis=1)
    early_late_ratio = early_total / np.maximum(late_total, 1e-8)

    for trial_index, triplet_id in enumerate(triplet_ids.tolist()):
        for window_index, window_name in enumerate(PATTERN_WINDOW_NAMES):
            timeseries_rows.append(
                {
                    "triplet_id": int(triplet_id),
                    "layer": REWRITE_LAYER,
                    "window_index": int(window_index),
                    "window_name": str(window_name),
                    "SPPI_window_L2": float(window_pull[trial_index, window_index]),
                    "SPPC_window_L2": float(window_dir[trial_index, window_index]),
                    "mixed_window_count_total": float(mixed_counts[trial_index, window_index, :].sum()),
                    "distractor_window_count_total": float(distractor_counts[trial_index, window_index, :].sum()),
                    "sample_window_count_total": float(sample_counts[trial_index, window_index, :].sum()),
                }
            )
        summary_rows.append(
            {
                "triplet_id": int(triplet_id),
                "SPPI_L2": float(sppi[trial_index]),
                "SPPC_L2": float(sppc[trial_index]),
                "DeltaSim_L2": float(delta_sim[trial_index]),
                "SPPI_early_L2": float(sppi_early[trial_index]),
                "pattern_norm_L2": float(pattern_norm[trial_index]),
                "early_late_activity_ratio_L2": float(early_late_ratio[trial_index]),
                "P_recruit_sample_L2": float(recruit_ratio[trial_index]),
                "P_advance_sample_L2": float(advance_ratio[trial_index]),
            }
        )
    df_time = pd.DataFrame(timeseries_rows).sort_values(["triplet_id", "layer", "window_index"], kind="stable").reset_index(drop=True)
    df_summary = pd.DataFrame(summary_rows).sort_values("triplet_id", kind="stable").reset_index(drop=True)
    return df_time, df_summary


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, float(eps))


def _rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = _normalize_rows(a)
    b_norm = _normalize_rows(b)
    return np.sum(a_norm * b_norm, axis=1)


def compute_preprobe_fusion_metrics(
    df_triplets: pd.DataFrame,
    condition_bank: Mapping[str, Dict[str, object]],
    *,
    evaluated_conditions: Sequence[str] = (CONDITION_BASELINE, CONDITION_FORMATION_CLAMP),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    triplet_ids = df_triplets["triplet_id"].to_numpy(dtype=np.int64, copy=False)
    for condition in evaluated_conditions:
        for layer_key in LAYER_ORDER:
            fused = np.asarray(condition_bank[condition]["preprobe"][layer_key]["feature"], dtype=np.float64)
            sample_ref = np.asarray(condition_bank[CONDITION_SAMPLE_ONLY]["preprobe"][layer_key]["feature"], dtype=np.float64)
            distract_ref = np.asarray(condition_bank[CONDITION_DISTRACTOR_ONLY]["preprobe"][layer_key]["feature"], dtype=np.float64)
            sim_to_sample = _rowwise_cosine(fused, sample_ref)
            sim_to_distractor = _rowwise_cosine(fused, distract_ref)
            fusion_dual = 0.5 * (sim_to_sample + sim_to_distractor) - 0.5 * np.abs(sim_to_sample - sim_to_distractor)
            fusion_imbalance = np.abs(sim_to_sample - sim_to_distractor)
            for idx, triplet_id in enumerate(triplet_ids.tolist()):
                rows.append(
                    {
                        "triplet_id": int(triplet_id),
                        "condition": condition,
                        "layer": layer_key,
                        "sim_to_sample": float(sim_to_sample[idx]),
                        "sim_to_distractor": float(sim_to_distractor[idx]),
                        "fusion_dual_score": float(fusion_dual[idx]),
                        "fusion_imbalance": float(fusion_imbalance[idx]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["condition", "triplet_id", "layer"], kind="stable").reset_index(drop=True)


def compute_fusion_specificity_metrics(
    df_triplets: pd.DataFrame,
    condition_bank: Mapping[str, Dict[str, object]],
    *,
    evaluated_conditions: Sequence[str] = (CONDITION_BASELINE, CONDITION_FORMATION_CLAMP),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pair_df = df_triplets[["sample_id", "distractor_id"]].drop_duplicates(ignore_index=True)
    pair_keys = [(int(row.sample_id), int(row.distractor_id)) for row in pair_df.itertuples(index=False)]
    pair_index = {pair_key: idx for idx, pair_key in enumerate(pair_keys)}
    for layer_key in LAYER_ORDER:
        sample_feature = np.asarray(condition_bank[CONDITION_SAMPLE_ONLY]["preprobe"][layer_key]["feature"], dtype=np.float64)
        distractor_feature = np.asarray(condition_bank[CONDITION_DISTRACTOR_ONLY]["preprobe"][layer_key]["feature"], dtype=np.float64)
        proto_matrix = np.zeros((len(pair_df), sample_feature.shape[1]), dtype=np.float64)
        for proto_idx, pair_row in enumerate(pair_df.itertuples(index=False)):
            match = df_triplets.index[
                (df_triplets["sample_id"] == int(pair_row.sample_id))
                & (df_triplets["distractor_id"] == int(pair_row.distractor_id))
            ]
            anchor_idx = int(match[0])
            proto_matrix[proto_idx] = 0.5 * (
                _normalize_rows(sample_feature[[anchor_idx]])[0] + _normalize_rows(distractor_feature[[anchor_idx]])[0]
            )
        proto_matrix = _normalize_rows(proto_matrix)
        for condition in evaluated_conditions:
            fused = np.asarray(condition_bank[condition]["preprobe"][layer_key]["feature"], dtype=np.float64)
            fused_norm = _normalize_rows(fused)
            scores = fused_norm @ proto_matrix.T
            means = scores.mean(axis=1)
            stds = np.maximum(scores.std(axis=1), 1e-8)
            for triplet_idx, row in enumerate(df_triplets.itertuples(index=False)):
                true_idx = pair_index[(int(row.sample_id), int(row.distractor_id))]
                true_score = float(scores[triplet_idx, true_idx])
                true_rank = int(1 + np.sum(scores[triplet_idx] > true_score))
                if scores.shape[1] <= 1:
                    true_percentile = 1.0
                else:
                    true_percentile = float(1.0 - (true_rank - 1) / (scores.shape[1] - 1))
                rows.append(
                    {
                        "triplet_id": int(row.triplet_id),
                        "condition": condition,
                        "layer": layer_key,
                        "true_pair_score": true_score,
                        "true_pair_rank": true_rank,
                        "true_pair_percentile": true_percentile,
                        "true_pair_z": float((true_score - means[triplet_idx]) / stds[triplet_idx]),
                        "true_pair_top1": int(true_rank == 1),
                    }
                )
    return pd.DataFrame(rows).sort_values(["condition", "triplet_id", "layer"], kind="stable").reset_index(drop=True)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _zscore_vector(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    std = float(arr.std())
    if std <= 1e-12:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / std


def _standardized_ols(
    df: pd.DataFrame,
    *,
    outcome: str,
    predictors: Sequence[str],
) -> pd.DataFrame:
    sub = df[[outcome, *predictors]].dropna().reset_index(drop=True)
    if len(sub) <= len(predictors) + 1:
        return pd.DataFrame(
            [
                {
                    "analysis": "standardized_ols",
                    "outcome": outcome,
                    "predictor": predictor,
                    "n": int(len(sub)),
                    "beta": np.nan,
                    "stderr": np.nan,
                    "t_value": np.nan,
                    "r2": np.nan,
                }
                for predictor in predictors
            ]
        )
    x = np.column_stack([_zscore_vector(sub[predictor].to_numpy(dtype=np.float64, copy=False)) for predictor in predictors])
    y = _zscore_vector(sub[outcome].to_numpy(dtype=np.float64, copy=False))
    design = np.column_stack([np.ones(len(sub), dtype=np.float64), x])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ coeffs
    resid = y - y_hat
    dof = len(sub) - design.shape[1]
    sse = float(np.dot(resid, resid))
    y_centered = y - float(y.mean())
    denom = float(np.dot(y_centered, y_centered))
    r2 = float(1.0 - sse / denom) if denom > 0.0 else float("nan")
    if dof > 0:
        cov = (sse / dof) * np.linalg.pinv(design.T @ design)
        stderr = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    else:
        stderr = np.full(design.shape[1], np.nan, dtype=np.float64)
    rows = []
    for idx, predictor in enumerate(predictors, start=1):
        se = float(stderr[idx]) if idx < len(stderr) else float("nan")
        beta = float(coeffs[idx])
        rows.append(
            {
                "analysis": "standardized_ols",
                "outcome": outcome,
                "predictor": predictor,
                "n": int(len(sub)),
                "beta": beta,
                "stderr": se,
                "t_value": float(beta / se) if np.isfinite(se) and se > 0.0 else np.nan,
                "r2": r2,
            }
        )
    return pd.DataFrame(rows)


def compute_rewriting_to_fusion_bridge(
    df_triplets: pd.DataFrame,
    df_rewrite_summary: pd.DataFrame,
    df_fusion: pd.DataFrame,
    df_specificity: pd.DataFrame,
) -> pd.DataFrame:
    fusion_l2 = df_fusion[(df_fusion["condition"] == CONDITION_BASELINE) & (df_fusion["layer"] == REWRITE_LAYER)].copy()
    spec_l2 = df_specificity[
        (df_specificity["condition"] == CONDITION_BASELINE) & (df_specificity["layer"] == REWRITE_LAYER)
    ].copy()
    merged = (
        df_triplets[
            [
                "triplet_id",
                "sample_id",
                "distractor_id",
                "probe_id",
                "sp_similarity",
                "dp_similarity",
                "sd_similarity",
            ]
        ]
        .merge(df_rewrite_summary[["triplet_id", "SPPI_L2", "SPPC_L2", "DeltaSim_L2", "SPPI_early_L2"]], on="triplet_id", how="inner")
        .merge(
            fusion_l2[["triplet_id", "fusion_dual_score", "sim_to_sample", "sim_to_distractor"]],
            on="triplet_id",
            how="inner",
        )
        .merge(spec_l2[["triplet_id", "true_pair_z"]], on="triplet_id", how="inner")
        .sort_values("triplet_id", kind="stable")
        .reset_index(drop=True)
    )
    rows = [
        {
            "analysis": "pearson_correlation",
            "outcome": "fusion_dual_score_L2",
            "predictor": "SPPI_L2",
            "n": int(len(merged)),
            "value": _safe_corr(
                merged["SPPI_L2"].to_numpy(dtype=np.float64, copy=False),
                merged["fusion_dual_score"].to_numpy(dtype=np.float64, copy=False),
            ),
        },
        {
            "analysis": "pearson_correlation",
            "outcome": "true_pair_z_L2",
            "predictor": "SPPI_L2",
            "n": int(len(merged)),
            "value": _safe_corr(
                merged["SPPI_L2"].to_numpy(dtype=np.float64, copy=False),
                merged["true_pair_z"].to_numpy(dtype=np.float64, copy=False),
            ),
        },
    ]
    regression_predictors = ["SPPI_L2", "sp_similarity", "dp_similarity", "sd_similarity"]
    df_reg_fusion = _standardized_ols(
        merged.rename(columns={"fusion_dual_score": "fusion_dual_score_L2"}),
        outcome="fusion_dual_score_L2",
        predictors=regression_predictors,
    )
    df_reg_true_pair = _standardized_ols(
        merged.rename(columns={"true_pair_z": "true_pair_z_L2"}),
        outcome="true_pair_z_L2",
        predictors=regression_predictors,
    )
    return pd.concat([pd.DataFrame(rows), df_reg_fusion, df_reg_true_pair], axis=0, ignore_index=True, sort=False)


def compute_formation_intervention_metrics(
    df_rewrite_summary_intact: pd.DataFrame,
    df_rewrite_summary_clamp: pd.DataFrame,
    df_fusion: pd.DataFrame,
    df_specificity: pd.DataFrame,
) -> pd.DataFrame:
    intact_rewrite = df_rewrite_summary_intact[["triplet_id", "SPPI_L2", "SPPI_early_L2"]].copy().rename(
        columns={"SPPI_L2": "SPPI_L2_intact", "SPPI_early_L2": "SPPI_early_L2_intact"}
    )
    clamp_rewrite = df_rewrite_summary_clamp[["triplet_id", "SPPI_L2", "SPPI_early_L2"]].copy().rename(
        columns={"SPPI_L2": "SPPI_L2_clamp", "SPPI_early_L2": "SPPI_early_L2_clamp"}
    )
    intact_fusion_l2 = df_fusion[
        (df_fusion["condition"] == CONDITION_BASELINE) & (df_fusion["layer"] == REWRITE_LAYER)
    ][["triplet_id", "fusion_dual_score"]].rename(columns={"fusion_dual_score": "fusion_dual_score_L2_intact"})
    clamp_fusion_l2 = df_fusion[
        (df_fusion["condition"] == CONDITION_FORMATION_CLAMP) & (df_fusion["layer"] == REWRITE_LAYER)
    ][["triplet_id", "fusion_dual_score"]].rename(columns={"fusion_dual_score": "fusion_dual_score_L2_clamp"})
    intact_spec_l2 = df_specificity[
        (df_specificity["condition"] == CONDITION_BASELINE) & (df_specificity["layer"] == REWRITE_LAYER)
    ][["triplet_id", "true_pair_z"]].rename(columns={"true_pair_z": "true_pair_z_L2_intact"})
    clamp_spec_l2 = df_specificity[
        (df_specificity["condition"] == CONDITION_FORMATION_CLAMP) & (df_specificity["layer"] == REWRITE_LAYER)
    ][["triplet_id", "true_pair_z"]].rename(columns={"true_pair_z": "true_pair_z_L2_clamp"})

    merged = (
        intact_rewrite.merge(clamp_rewrite, on="triplet_id", how="inner")
        .merge(intact_fusion_l2, on="triplet_id", how="left")
        .merge(clamp_fusion_l2, on="triplet_id", how="left")
        .merge(intact_spec_l2, on="triplet_id", how="left")
        .merge(clamp_spec_l2, on="triplet_id", how="left")
        .sort_values("triplet_id", kind="stable")
        .reset_index(drop=True)
    )
    merged["delta_SPPI_L2"] = merged["SPPI_L2_intact"] - merged["SPPI_L2_clamp"]
    merged["delta_SPPI_early_L2"] = merged["SPPI_early_L2_intact"] - merged["SPPI_early_L2_clamp"]
    merged["delta_fusion_dual_score_L2"] = merged["fusion_dual_score_L2_intact"] - merged["fusion_dual_score_L2_clamp"]
    merged["delta_true_pair_z_L2"] = merged["true_pair_z_L2_intact"] - merged["true_pair_z_L2_clamp"]
    merged["summary_type"] = "triplet"
    summary = {
        "triplet_id": -1,
        "summary_type": "aggregate_mean",
        "delta_SPPI_L2": float(merged["delta_SPPI_L2"].mean()),
        "delta_SPPI_early_L2": float(merged["delta_SPPI_early_L2"].mean()),
        "delta_fusion_dual_score_L2": float(merged["delta_fusion_dual_score_L2"].mean()),
        "delta_true_pair_z_L2": float(merged["delta_true_pair_z_L2"].mean()),
    }
    return pd.concat([merged, pd.DataFrame([summary])], axis=0, ignore_index=True, sort=False)


def select_example_triplet(df_rewrite_summary: pd.DataFrame) -> int:
    scores = df_rewrite_summary[["triplet_id", "SPPI_L2"]].dropna().copy()
    if scores.empty:
        raise RuntimeError("No valid SPPI_L2 values available for example-triplet selection.")
    target = float(scores["SPPI_L2"].quantile(0.75))
    scores["distance"] = np.abs(scores["SPPI_L2"] - target)
    selected = scores.sort_values(["distance", "SPPI_L2"], ascending=[True, False], kind="stable").iloc[0]
    return int(selected["triplet_id"])


def _add_regression_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if len(x) < 2 or np.allclose(x, x[0]):
        return
    slope, intercept = np.polyfit(x, y, deg=1)
    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), num=100)
    ax.plot(x_grid, slope * x_grid + intercept, color=color, linewidth=1.5, alpha=0.9)


def _example_pattern_matrices(
    condition_bank: Mapping[str, Dict[str, object]],
    *,
    triplet_index: int,
    top_groups: int = EXAMPLE_TOP_GROUPS,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    sample_counts = np.asarray(condition_bank[CONDITION_SAMPLE_ONLY]["rewrite"]["window_counts"][triplet_index], dtype=np.float64)
    ranking = np.argsort(sample_counts.sum(axis=0))[::-1]
    if ranking.size == 0:
        raise RuntimeError("No layer2 spike-pattern groups were recorded.")
    select = ranking[: min(int(top_groups), int(ranking.size))]
    matrices: dict[str, np.ndarray] = {}
    for condition in (CONDITION_BASELINE, CONDITION_DISTRACTOR_ONLY, CONDITION_SAMPLE_ONLY):
        matrices[condition] = np.asarray(
            condition_bank[condition]["rewrite"]["window_counts"][triplet_index][:, select],
            dtype=np.float64,
        ).T
    return matrices, select


def save_rewriting_panel(
    output_dir: Path,
    df_summary: pd.DataFrame,
    *,
    example_triplet_id: int,
    condition_bank: Mapping[str, Dict[str, object]],
    config: ExperimentConfig,
) -> Dict[str, str]:
    apply_publication_style()
    triplet_index = int(df_summary.index[df_summary["triplet_id"] == int(example_triplet_id)][0])
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.2))
    matrices, _ = _example_pattern_matrices(condition_bank, triplet_index=triplet_index)
    vmax = max(float(matrix.max()) for matrix in matrices.values()) if matrices else 1.0
    for condition, color, label in (
        (CONDITION_BASELINE, COLOR_DYNAMIC, "mixed"),
        (CONDITION_DISTRACTOR_ONLY, COLOR_DISTRACTOR, "distractor-only"),
        (CONDITION_SAMPLE_ONLY, COLOR_SAMPLE_ALIGNED, "sample-only"),
    ):
        ax = axes[(CONDITION_BASELINE, CONDITION_DISTRACTOR_ONLY, CONDITION_SAMPLE_ONLY).index(condition)]
        im = ax.imshow(matrices[condition], aspect="auto", cmap="magma", vmin=0.0, vmax=max(vmax, 1.0))
        ax.set_title(f"{label}")
        ax.set_xticks(range(len(PATTERN_WINDOW_NAMES)))
        ax.set_xticklabels(PATTERN_WINDOW_NAMES)
        ax.set_xlabel("Window")
        ax.set_ylabel("Layer2 pooled group")
    fig.suptitle(f"Panel A  Example spike-pattern rewriting (triplet {example_triplet_id})")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="Spike count")
    return save_figure_all_formats(fig, output_dir / "panel_rewriting")


def save_bridge_panel(output_dir: Path, df_bridge_input: pd.DataFrame) -> Dict[str, str]:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    x = df_bridge_input["SPPI_L2"].to_numpy(dtype=np.float64, copy=False)
    y1 = df_bridge_input["fusion_dual_score"].to_numpy(dtype=np.float64, copy=False)
    y2 = df_bridge_input["true_pair_z"].to_numpy(dtype=np.float64, copy=False)
    axes[0].scatter(x, y1, color=COLOR_DYNAMIC, alpha=0.8)
    axes[1].scatter(x, y2, color=COLOR_SAMPLE_ALIGNED, alpha=0.8)
    _add_regression_line(axes[0], x, y1, COLOR_DYNAMIC)
    _add_regression_line(axes[1], x, y2, COLOR_SAMPLE_ALIGNED)
    axes[0].set_title("Panel C  SPPI_L2 vs fusion_dual_score_L2")
    axes[1].set_title("Panel C  SPPI_L2 vs true_pair_z_L2")
    axes[0].set_xlabel("SPPI_L2")
    axes[1].set_xlabel("SPPI_L2")
    axes[0].set_ylabel("fusion_dual_score_L2")
    axes[1].set_ylabel("true_pair_z_L2")
    axes[0].text(0.05, 0.95, f"r={_safe_corr(x, y1):.3f}", transform=axes[0].transAxes, ha="left", va="top")
    axes[1].text(0.05, 0.95, f"r={_safe_corr(x, y2):.3f}", transform=axes[1].transAxes, ha="left", va="top")
    return save_figure_all_formats(fig, output_dir / "panel_bridge")


def save_formation_panel(output_dir: Path, df_intervention: pd.DataFrame) -> Dict[str, str]:
    apply_publication_style()
    triplets = df_intervention[df_intervention["summary_type"] == "triplet"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    metric_map = [
        ("delta_SPPI_L2", "delta_SPPI_L2"),
        ("delta_SPPI_early_L2", "delta_SPPI_early_L2"),
        ("delta_true_pair_z_L2", "delta_true_pair_z_L2"),
    ]
    for ax, (column, title) in zip(axes, metric_map):
        values = triplets[column].to_numpy(dtype=np.float64, copy=False)
        x_jitter = np.linspace(-0.08, 0.08, num=max(len(values), 1))
        ax.bar([0.0], [float(np.mean(values))], width=0.45, color=COLOR_DYNAMIC, alpha=0.35)
        ax.scatter(x_jitter[: len(values)], values, color=COLOR_DYNAMIC, alpha=0.85)
        ax.axhline(0.0, color=COLOR_STATIC, linewidth=1.0, alpha=0.6)
        ax.set_xlim(-0.45, 0.45)
        ax.set_xticks([0.0])
        ax.set_xticklabels(["intact - clamp"])
        ax.set_title(f"Panel D  {title}")
    return save_figure_all_formats(fig, output_dir / "panel_formation")


def save_main_figure(
    output_dir: Path,
    *,
    config: ExperimentConfig,
    df_timeseries: pd.DataFrame,
    df_summary: pd.DataFrame,
    bridge_input: pd.DataFrame,
    df_intervention: pd.DataFrame,
    condition_bank: Mapping[str, Dict[str, object]],
    example_triplet_id: int,
) -> Dict[str, str]:
    apply_publication_style()
    fig = plt.figure(figsize=(13.8, 10.4))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    triplet_index = int(df_summary.index[df_summary["triplet_id"] == int(example_triplet_id)][0])
    matrices, _ = _example_pattern_matrices(condition_bank, triplet_index=triplet_index, top_groups=18)
    concat_matrix = np.concatenate(
        [
            matrices[CONDITION_BASELINE],
            matrices[CONDITION_DISTRACTOR_ONLY],
            matrices[CONDITION_SAMPLE_ONLY],
        ],
        axis=1,
    )
    im = ax_a.imshow(concat_matrix, aspect="auto", cmap="magma")
    ax_a.set_title(f"Panel A  Example spike-pattern rewriting (triplet {example_triplet_id})")
    ax_a.set_xlabel("Condition windows")
    ax_a.set_ylabel("Layer2 pooled group")
    ax_a.set_xticks(np.arange(concat_matrix.shape[1]))
    ax_a.set_xticklabels(list(PATTERN_WINDOW_NAMES) * 3, rotation=45, ha="right")
    ax_a.axvline(len(PATTERN_WINDOW_NAMES) - 0.5, color="white", linewidth=1.2)
    ax_a.axvline(2 * len(PATTERN_WINDOW_NAMES) - 0.5, color="white", linewidth=1.2)
    ax_a.text(0.17, 1.02, "mixed", transform=ax_a.transAxes, ha="center")
    ax_a.text(0.50, 1.02, "distractor-only", transform=ax_a.transAxes, ha="center")
    ax_a.text(0.83, 1.02, "sample-only", transform=ax_a.transAxes, ha="center")
    fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.02)

    mean_pull = (
        df_timeseries.groupby(["layer", "window_index", "window_name"], as_index=False)["SPPI_window_L2"]
        .mean()
        .sort_values(["layer", "window_index"], kind="stable")
    )
    sub = mean_pull[mean_pull["layer"] == REWRITE_LAYER].copy()
    ax_b.plot(sub["window_index"], sub["SPPI_window_L2"], color=COLOR_DISTRACTOR, marker="o")
    sppi_values = df_summary["SPPI_L2"].to_numpy(dtype=np.float64, copy=False)
    jitter = np.linspace(-0.12, 0.12, num=max(len(sppi_values), 1))
    ax_b.scatter(np.full(len(sppi_values), 1.6) + jitter[: len(sppi_values)], sppi_values, color=COLOR_DYNAMIC, alpha=0.75, s=28)
    ax_b.axhline(0.0, color=COLOR_STATIC, linewidth=1.0, alpha=0.5)
    ax_b.set_title("Panel B  SPPI summary")
    ax_b.set_ylabel("SPPI statistic")
    ax_b.set_xlim(-0.4, 2.0)
    ax_b.set_xticks([0.0, 1.0, 1.6])
    ax_b.set_xticklabels(["early", "late", "SPPI_L2"])

    x = bridge_input["SPPI_L2"].to_numpy(dtype=np.float64, copy=False)
    y = bridge_input["fusion_dual_score"].to_numpy(dtype=np.float64, copy=False)
    ax_c.scatter(x, y, color=COLOR_DYNAMIC, alpha=0.85)
    _add_regression_line(ax_c, x, y, COLOR_DYNAMIC)
    ax_c.set_title("Panel C  rewriting to fusion bridge")
    ax_c.set_xlabel("SPPI_L2")
    ax_c.set_ylabel("fusion_dual_score_L2")
    ax_c.text(0.04, 0.95, f"r={_safe_corr(x, y):.3f}", transform=ax_c.transAxes, ha="left", va="top")

    triplets = df_intervention[df_intervention["summary_type"] == "triplet"].copy()
    delta_columns = ["delta_SPPI_L2", "delta_SPPI_early_L2", "delta_true_pair_z_L2"]
    delta_means = [float(triplets[col].mean()) for col in delta_columns]
    ax_d.bar(np.arange(len(delta_columns)), delta_means, color=COLOR_DYNAMIC, alpha=0.35)
    for idx, column in enumerate(delta_columns):
        values = triplets[column].to_numpy(dtype=np.float64, copy=False)
        jitter = np.linspace(-0.08, 0.08, num=max(len(values), 1))
        ax_d.scatter(np.full(len(values), idx) + jitter[: len(values)], values, color=COLOR_DYNAMIC, alpha=0.85)
    ax_d.axhline(0.0, color=COLOR_STATIC, linewidth=1.0, alpha=0.6)
    ax_d.set_xticks(np.arange(len(delta_columns)))
    ax_d.set_xticklabels(["delta_SPPI_L2", "delta_SPPI_early_L2", "delta_true_pair_z_L2"], rotation=20, ha="right")
    ax_d.set_title("Panel D  formation intervention")
    ax_d.set_ylabel("Intact - clamp")
    return save_figure_all_formats(fig, output_dir / "figure_main")


def save_optional_npz(
    output_dir: Path,
    df_triplets: pd.DataFrame,
    condition_bank: Mapping[str, Dict[str, object]],
    *,
    example_triplet_id: int,
) -> None:
    state_payload: dict[str, np.ndarray] = {
        "triplet_id": df_triplets["triplet_id"].to_numpy(dtype=np.int64, copy=False),
    }
    for condition in CONDITION_ORDER:
        for layer_key in LAYER_ORDER:
            for state_name in ("u", "x", "ux", "feature"):
                state_payload[f"{condition}_{layer_key}_{state_name}"] = np.asarray(
                    condition_bank[condition]["preprobe"][layer_key][state_name],
                    dtype=np.float32,
                )
        for key in ("window_counts", "pattern_vector", "total_counts", "first_spike_step", "active_mask", "window_total_counts"):
            state_payload[f"{condition}_{REWRITE_LAYER}_{key}"] = np.asarray(condition_bank[condition]["rewrite"][key], dtype=np.float32)
    np.savez_compressed(output_dir / "state_bank_preprobe.npz", **state_payload)

    triplet_index = int(df_triplets.index[df_triplets["triplet_id"] == int(example_triplet_id)][0])
    example_payload: dict[str, np.ndarray] = {
        "triplet_id": np.asarray([int(example_triplet_id)], dtype=np.int64),
    }
    example_payload["window_names"] = np.asarray(PATTERN_WINDOW_NAMES, dtype=object)
    for condition in (CONDITION_BASELINE, CONDITION_DISTRACTOR_ONLY, CONDITION_SAMPLE_ONLY, CONDITION_FORMATION_CLAMP):
        example_payload[f"{condition}_{REWRITE_LAYER}_window_counts"] = np.asarray(
            condition_bank[condition]["rewrite"]["window_counts"][triplet_index],
            dtype=np.float32,
        )
        example_payload[f"{condition}_{REWRITE_LAYER}_pattern_vector"] = np.asarray(
            condition_bank[condition]["rewrite"]["pattern_vector"][triplet_index],
            dtype=np.float32,
        )
    np.savez_compressed(output_dir / "example_trajectory.npz", **example_payload)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)
    log_lines: list[str] = []
    output_dir = config.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device_with_fallback(config.device_request, log_lines)
    print_config_summary(config, device, log_lines)
    seed_everything(config.seed)

    dataset = load_mnist_dataset(config.dataset_root, config.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    num_classes = int(len(np.unique(labels)))
    class_index = build_class_index(dataset, num_classes=num_classes)
    emit(f"[Data] split={config.split} num_samples={len(dataset)} num_classes={num_classes}", log_lines)

    net, encoder = load_model_and_encoder(
        model_path=config.model_path,
        device=device,
        dt=config.dt,
        max_duration_ms=config.max_duration_ms,
    )
    emit(f"[Model] loaded={config.model_path}", log_lines)

    df_triplets = build_triplet_specs(
        images=images,
        labels=labels,
        flat_normalized=flat_normalized,
        class_index=class_index,
        max_probes=config.max_probes,
        samples_per_probe=config.samples_per_probe,
        num_bins=config.num_sim_bins,
        max_triplets=config.max_triplets,
        seed=config.seed,
    ).copy()
    emit(f"[Triplets] generated={len(df_triplets)} unique_probes={df_triplets['probe_id'].nunique()}", log_lines)

    condition_bank = collect_condition_captures(
        net,
        images,
        df_triplets,
        encoder=encoder,
        config=config,
        device=device,
        log_lines=log_lines,
    )

    df_rewrite_time, df_rewrite_summary = compute_reshaping_metrics(df_triplets, condition_bank, config=config)
    df_fusion = compute_preprobe_fusion_metrics(df_triplets, condition_bank)
    df_specificity = compute_fusion_specificity_metrics(df_triplets, condition_bank)
    df_bridge = compute_rewriting_to_fusion_bridge(df_triplets, df_rewrite_summary, df_fusion, df_specificity)

    _, clamp_rewrite_summary = compute_reshaping_metrics(
        df_triplets,
        {
            CONDITION_BASELINE: condition_bank[CONDITION_FORMATION_CLAMP],
            CONDITION_DISTRACTOR_ONLY: condition_bank[CONDITION_DISTRACTOR_ONLY],
            CONDITION_SAMPLE_ONLY: condition_bank[CONDITION_SAMPLE_ONLY],
        },
        config=config,
    )
    df_intervention = compute_formation_intervention_metrics(
        df_rewrite_summary,
        clamp_rewrite_summary,
        df_fusion,
        df_specificity,
    )

    example_triplet_id = select_example_triplet(df_rewrite_summary)
    bridge_input = (
        df_triplets[["triplet_id", "sp_similarity", "dp_similarity", "sd_similarity"]]
        .merge(df_rewrite_summary[["triplet_id", "SPPI_L2", "SPPC_L2", "DeltaSim_L2", "SPPI_early_L2"]], on="triplet_id", how="inner")
        .merge(
            df_fusion[(df_fusion["condition"] == CONDITION_BASELINE) & (df_fusion["layer"] == REWRITE_LAYER)][
                ["triplet_id", "fusion_dual_score"]
            ],
            on="triplet_id",
            how="inner",
        )
        .merge(
            df_specificity[(df_specificity["condition"] == CONDITION_BASELINE) & (df_specificity["layer"] == REWRITE_LAYER)][
                ["triplet_id", "true_pair_z"]
            ],
            on="triplet_id",
            how="inner",
        )
        .sort_values("triplet_id", kind="stable")
        .reset_index(drop=True)
    )

    triplet_output = df_triplets[
        [
            "triplet_id",
            "sample_id",
            "distractor_id",
            "probe_id",
            "sample_label",
            "distractor_label",
            "probe_label",
            "sp_similarity",
            "dp_similarity",
            "sd_similarity",
            "sp_bin",
            "dp_bin",
            "sp_bin_index",
            "dp_bin_index",
        ]
    ].copy()
    triplet_output["probe_bin"] = triplet_output["sp_bin"]
    triplet_output["sample_selection_bin"] = triplet_output["sp_bin"]
    triplet_output["distractor_selection_bin"] = triplet_output["dp_bin"]

    save_tidy_csv(triplet_output, output_dir / "triplets.csv", sort_by=["triplet_id"])
    save_tidy_csv(df_rewrite_time, output_dir / "distractor_pull_timeseries.csv", sort_by=["triplet_id", "layer", "window_index"])
    save_tidy_csv(df_rewrite_summary, output_dir / "distractor_pull_summary.csv", sort_by=["triplet_id"])
    save_tidy_csv(df_fusion, output_dir / "preprobe_fusion_metrics.csv", sort_by=["condition", "triplet_id", "layer"])
    save_tidy_csv(
        df_specificity,
        output_dir / "fusion_specificity_metrics.csv",
        sort_by=["condition", "triplet_id", "layer"],
    )
    save_tidy_csv(df_bridge, output_dir / "rewriting_fusion_bridge.csv")
    save_tidy_csv(
        condition_bank[CONDITION_FORMATION_CLAMP]["intervention_rows"],
        output_dir / "formation_clamp_intervention_log.csv",
        sort_by=["triplet_id", "condition"],
    )
    save_tidy_csv(df_intervention, output_dir / "formation_intervention_metrics.csv")

    save_optional_npz(output_dir, df_triplets, condition_bank, example_triplet_id=example_triplet_id)
    save_run_config(
        {
            **asdict(config),
            "resolved_device": str(device),
            "num_classes": int(num_classes),
            "example_triplet_id": int(example_triplet_id),
        },
        output_dir,
    )

    if not config.skip_figures:
        save_rewriting_panel(
            output_dir,
            df_rewrite_summary,
            example_triplet_id=example_triplet_id,
            condition_bank=condition_bank,
            config=config,
        )
        save_bridge_panel(output_dir, bridge_input)
        save_formation_panel(output_dir, df_intervention)
        save_main_figure(
            output_dir,
            config=config,
            df_timeseries=df_rewrite_time,
            df_summary=df_rewrite_summary,
            bridge_input=bridge_input,
            df_intervention=df_intervention,
            condition_bank=condition_bank,
            example_triplet_id=example_triplet_id,
        )

    emit(f"[Done] outputs={output_dir}", log_lines)


if __name__ == "__main__":
    main()
