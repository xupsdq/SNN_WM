from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.paths import DEFAULT_PATH_CONFIG
from src.config.units import ms
from src.experiments.common.dataset import build_class_index, build_dataset_arrays, encode_images
from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
from src.experiments.common.distractor_triplets import load_mnist_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import (
    build_layer_input_shapes,
    reset_all_state_restore_selected_stsp_in_place,
)
from src.experiments.common.ping_common import LAYER_KEYS, prepare_network_state
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_run_config, save_summary_json
from src.experiments.common.runtime import seed_everything
from src.experiments.common.seed import mix_seed
from src.plotting.common.io import apply_publication_style, save_figure_all_formats, save_tidy_csv
from src.plotting.common.style import DYNAMIC_COLOR, NOISE_COLOR, SAMPLE_COLOR, SHUFFLE_COLOR
from src.plotting.experiments.chunk_stsp_layer3_anchor_drift_mechanism_plot_lib import (
    render_panels_from_results as render_plot_only_panels,
    write_plot_bundle_manifest,
)


TOP_Q_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10)
TOP_Q_LABELS: tuple[str, ...] = ("1pct", "5pct", "10pct")


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
    ping_ms: float
    ping_amp: float
    batch_size: int
    max_sequences: int
    sequence_lengths: tuple[int, ...]
    samples_per_label: int
    epsilon: float
    epsilon_sweep: tuple[float, ...]
    null_permutations: int
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
    def ping_steps(self) -> int:
        return ms_to_steps(self.ping_ms, self.dt)

    @property
    def max_duration_ms(self) -> float:
        return max(float(self.sample_ms), float(self.ping_ms), 100.0)

    @property
    def max_seq_len(self) -> int:
        return max(int(value) for value in self.sequence_lengths)


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
    boundary_name: str
    current_time: int
    layer_input_shapes: dict[str, tuple[int, ...]]
    full_state_by_layer: dict[str, LayerRuntimeState]
    restore_ux_by_layer: dict[str, tuple[torch.Tensor, torch.Tensor]]
    layer3_u: np.ndarray
    layer3_x: np.ndarray
    layer3_g: np.ndarray


@dataclass(frozen=True)
class CounterfactualSnapshot:
    stage_k: int
    boundary_name: str
    current_time: int
    layer3_u_decay: np.ndarray
    layer3_x_decay: np.ndarray
    layer3_g_decay: np.ndarray


@dataclass(frozen=True)
class PingReadout:
    stage_k: int
    boundary_name: str
    first_fire_pred: np.ndarray
    first_fire_t: np.ndarray
    silent_mask: np.ndarray
    predicted_item_index: np.ndarray
    predicted_item_label: np.ndarray
    ping_seen_item_hit: np.ndarray
    ping_nonmember_first: np.ndarray
    ping_first_item_index: np.ndarray
    ping_normalized_recency: np.ndarray
    ping_latest_item_hit_raw: np.ndarray
    ping_latest_item_hit_chance_corrected: np.ndarray
    ping_recent_window_hit: np.ndarray
    ping_lag_from_latest: np.ndarray


@dataclass(frozen=True)
class SingletonReferenceSnapshot:
    stage_k: int
    item_index: int
    layer3_g: np.ndarray


@dataclass(frozen=True)
class SequenceRollout:
    initial_snapshot: BoundarySnapshot
    stage_snapshots: tuple[BoundarySnapshot, ...]
    decay_snapshots: tuple[CounterfactualSnapshot, ...]
    reference_bank: dict[int, dict[int, SingletonReferenceSnapshot]]


def ms_to_steps(duration_ms: float, dt: float) -> int:
    return int(round((float(duration_ms) * ms) / float(dt)))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Layer3 STSP anchor-drift mechanism experiment with natural-decay counterfactuals.",
    )
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_PATH_CONFIG.model_path))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_PATH_CONFIG.results_root / "chunk_stsp_layer3_anchor_drift_mechanism"),
    )
    parser.add_argument("--sample-ms", type=float, default=180.0)
    parser.add_argument("--delay-ms", type=float, default=200.0)
    parser.add_argument("--ping-ms", type=float, default=30.0)
    parser.add_argument("--ping-amp", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[3, 5, 7, 10])
    parser.add_argument("--samples-per-label", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--epsilon-sweep", type=float, nargs="*", default=None)
    parser.add_argument("--null-permutations", type=int, default=200)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def normalize_epsilon_sweep(epsilon: float, epsilon_sweep: Sequence[float] | None, smoke: bool) -> tuple[float, ...]:
    values = [float(epsilon)]
    if epsilon_sweep:
        values.extend(float(item) for item in epsilon_sweep)
    cleaned = tuple(sorted({float(item) for item in values if float(item) > 0.0}))
    if not cleaned:
        raise ValueError("At least one positive epsilon is required.")
    if smoke and len(cleaned) > 2:
        return cleaned[:2]
    return cleaned


def normalize_config(args: argparse.Namespace) -> ExperimentConfig:
    sequence_lengths = tuple(dict.fromkeys(int(item) for item in args.sequence_lengths))
    if not sequence_lengths:
        raise ValueError("--sequence-lengths must not be empty.")
    if min(sequence_lengths) < 1:
        raise ValueError("--sequence-lengths must be >= 1.")
    if max(sequence_lengths) > 10:
        raise ValueError("--sequence-lengths must be <= 10 because labels are sampled without replacement.")
    cfg = ExperimentConfig(
        model_path=str(args.model_path),
        dataset_root=str(args.dataset_root),
        split=str(args.split),
        device=str(args.device),
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        sample_ms=float(args.sample_ms),
        delay_ms=float(args.delay_ms),
        ping_ms=float(args.ping_ms),
        ping_amp=float(args.ping_amp),
        batch_size=int(args.batch_size),
        max_sequences=int(args.max_sequences),
        sequence_lengths=sequence_lengths,
        samples_per_label=int(args.samples_per_label),
        epsilon=float(args.epsilon),
        epsilon_sweep=normalize_epsilon_sweep(float(args.epsilon), args.epsilon_sweep, bool(args.smoke)),
        null_permutations=int(args.null_permutations),
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
                "sequence_lengths": (3,),
                "samples_per_label": min(int(cfg.samples_per_label), 8),
                "sample_ms": min(float(cfg.sample_ms), 15.0),
                "delay_ms": min(float(cfg.delay_ms), 10.0),
                "ping_ms": min(float(cfg.ping_ms), 10.0),
                "null_permutations": min(int(cfg.null_permutations), 24),
            }
        )
    if cfg.batch_size <= 0 or cfg.max_sequences <= 0:
        raise ValueError("--batch-size and --max-sequences must be positive.")
    if cfg.samples_per_label == 0:
        raise ValueError("--samples-per-label must be non-zero.")
    if cfg.null_permutations <= 0:
        raise ValueError("--null-permutations must be positive.")
    if min(cfg.sample_steps, cfg.delay_steps, cfg.ping_steps) <= 0:
        raise ValueError("sample-ms, delay-ms, and ping-ms must map to at least one step.")
    if cfg.epsilon < 0.0:
        raise ValueError("--epsilon must be non-negative.")
    return cfg


def resolve_device_with_fallback(device_arg: str) -> tuple[torch.device, str]:
    raw = str(device_arg).strip().lower()
    if raw in ("", "auto"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, f"[Runtime] device=auto resolved to {device.type}."
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu"), "[Runtime] CUDA unavailable on 2026-04-20; falling back to CPU."
    return device, f"[Runtime] Using device={device.type}."


def log_and_print(log_lines: list[str], message: str) -> None:
    print(message, flush=True)
    log_lines.append(str(message))


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return safe_float(value)
    if pd.isna(value):
        return None
    return value


def safe_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.count_nonzero(valid) < 2:
        return None
    x_valid = x_arr[valid]
    y_valid = y_arr[valid]
    if float(np.nanstd(x_valid)) <= 0.0 or float(np.nanstd(y_valid)) <= 0.0:
        return None
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def summarize_trial_table(
    rows: Sequence[Mapping[str, object]],
    *,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["record_type", *group_cols, *value_cols])
    df = pd.DataFrame(rows)
    if "record_type" not in df.columns:
        df.insert(0, "record_type", "trial_level")
    df_trials = df[df["record_type"] == "trial_level"].copy()
    if df_trials.empty:
        return df
    df_summary = df_trials.groupby(list(group_cols), dropna=False, as_index=False)[list(value_cols)].mean(numeric_only=True)
    df_summary.insert(0, "record_type", "stage_summary")
    return pd.concat([df, df_summary], axis=0, ignore_index=True, sort=False)


def build_label_candidate_pools(
    class_index: Mapping[int, Sequence[int]],
    cfg: ExperimentConfig,
) -> dict[int, np.ndarray]:
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
            trial_seed = mix_seed(cfg.seed, int(seq_len), int(within_len_idx), 101)
            rng = np.random.default_rng(trial_seed)
            chosen_labels = rng.choice(all_labels, size=int(seq_len), replace=False)
            chosen_item_ids = np.asarray(
                [int(rng.choice(label_pools[int(label)])) for label in chosen_labels],
                dtype=np.int64,
            )
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
            for item_index, (image_id, item_label) in enumerate(
                zip(trial.ordered_item_ids, trial.ordered_item_labels),
                start=1,
            ):
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
                torch.stack(
                    [spike_lookup[int(trial.ordered_item_ids[item_pos])] for trial in batch_trials],
                    dim=0,
                )
                for item_pos in range(int(seq_len))
            )
            yield SequenceBatch(
                batch_id=int(batch_id),
                seq_len=int(seq_len),
                trials=batch_trials,
                item_spikes=item_spikes,
            )


def forward_three_layers(
    net: Any,
    input_t: torch.Tensor,
    t_step: int,
    *,
    ping_drive: torch.Tensor | None = None,
) -> None:
    s1, _ = net.layer1.forward_step(
        input_t,
        t_step,
        training=False,
        monitor=False,
        stsp_mode="dynamic",
        ping_drive=ping_drive,
    )
    s1_p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1_p, t_step, training=False, monitor=False, stsp_mode="dynamic")
    s2_p = net.pool2(s2.float())
    net.layer3.forward_step(s2_p, t_step, training=False, monitor=False, stsp_mode="dynamic")


def _clone_optional_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().cpu().clone()


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
    boundary_name: str,
    current_time: int,
    layer_input_shapes: Mapping[str, tuple[int, ...]],
) -> BoundarySnapshot:
    full_state_by_layer: dict[str, LayerRuntimeState] = {}
    restore_ux_by_layer: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key)
        full_state_by_layer[str(layer_key)] = capture_layer_runtime_state(layer)
        if getattr(layer, "u_pre", None) is not None and getattr(layer, "x_pre", None) is not None:
            restore_ux_by_layer[str(layer_key)] = (
                layer.u_pre.detach().cpu().clone(),
                layer.x_pre.detach().cpu().clone(),
            )
    layer3 = net.layer3
    if layer3.u_pre is None or layer3.x_pre is None:
        raise ValueError("layer3 is missing STSP state at the requested boundary.")
    layer3_u = layer3.u_pre.detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=False)
    layer3_x = layer3.x_pre.detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=False)
    layer3_g = (layer3.u_pre * layer3.x_pre).detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=False)
    return BoundarySnapshot(
        stage_k=int(stage_k),
        boundary_name=str(boundary_name),
        current_time=int(current_time),
        layer_input_shapes={str(key): tuple(value) for key, value in layer_input_shapes.items()},
        full_state_by_layer=full_state_by_layer,
        restore_ux_by_layer=restore_ux_by_layer,
        layer3_u=layer3_u,
        layer3_x=layer3_x,
        layer3_g=layer3_g,
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


def run_sequence_and_capture_boundaries(
    net: Any,
    batch: SequenceBatch,
    cfg: ExperimentConfig,
) -> SequenceRollout:
    first_sequence = batch.item_spikes[0]
    batch_size, _, channels, height, width = first_sequence.shape
    with torch.no_grad():
        prepare_network_state(net, batch_size, channels, height, width)
        layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)
        zero_input = torch.zeros(
            (batch_size, channels, height, width),
            dtype=first_sequence.dtype,
            device=first_sequence.device,
        )
        current_time = 0
        initial_snapshot = snapshot_boundary_state(
            net,
            batch_size=batch_size,
            stage_k=0,
            boundary_name="stage0_start",
            current_time=current_time,
            layer_input_shapes=layer_input_shapes,
        )
        stage_snapshots: list[BoundarySnapshot] = []
        for position, item_spikes in enumerate(batch.item_spikes, start=1):
            for step_idx in range(int(item_spikes.shape[1])):
                forward_three_layers(net, item_spikes[:, step_idx, ...], current_time)
                current_time += 1
            for _ in range(int(cfg.delay_steps)):
                forward_three_layers(net, zero_input, current_time)
                current_time += 1
            stage_snapshots.append(
                snapshot_boundary_state(
                    net,
                    batch_size=batch_size,
                    stage_k=int(position),
                    boundary_name=f"stage_{position}_boundary",
                    current_time=current_time,
                    layer_input_shapes=layer_input_shapes,
                )
            )
    decay_snapshots = run_natural_decay_counterfactuals(net, cfg, initial_snapshot, tuple(stage_snapshots))
    reference_bank = run_lag_matched_singleton_reference_bank(net, batch, cfg)
    return SequenceRollout(
        initial_snapshot=initial_snapshot,
        stage_snapshots=tuple(stage_snapshots),
        decay_snapshots=decay_snapshots,
        reference_bank=reference_bank,
    )


def run_natural_decay_counterfactuals(
    net: Any,
    cfg: ExperimentConfig,
    initial_snapshot: BoundarySnapshot,
    stage_snapshots: Sequence[BoundarySnapshot],
) -> tuple[CounterfactualSnapshot, ...]:
    decay_snapshots: list[CounterfactualSnapshot] = []
    previous_snapshot = initial_snapshot
    for actual_snapshot in stage_snapshots:
        restore_network_boundary_state(net, previous_snapshot)
        zero_input = torch.zeros(
            previous_snapshot.layer_input_shapes["layer1"],
            dtype=torch.float32,
            device=net.layer1.v_mem.device,
        )
        current_time = int(previous_snapshot.current_time)
        with torch.no_grad():
            for _ in range(int(cfg.sample_steps)):
                forward_three_layers(net, zero_input, current_time)
                current_time += 1
            for _ in range(int(cfg.delay_steps)):
                forward_three_layers(net, zero_input, current_time)
                current_time += 1
        if current_time != int(actual_snapshot.current_time):
            raise RuntimeError(
                f"Counterfactual boundary time mismatch for stage {actual_snapshot.stage_k}: "
                f"{current_time} != {actual_snapshot.current_time}"
            )
        layer3 = net.layer3
        decay_snapshots.append(
            CounterfactualSnapshot(
                stage_k=int(actual_snapshot.stage_k),
                boundary_name=str(actual_snapshot.boundary_name),
                current_time=int(current_time),
                layer3_u_decay=layer3.u_pre.detach().view(zero_input.shape[0], -1).cpu().numpy().astype(np.float32, copy=False),
                layer3_x_decay=layer3.x_pre.detach().view(zero_input.shape[0], -1).cpu().numpy().astype(np.float32, copy=False),
                layer3_g_decay=(layer3.u_pre * layer3.x_pre)
                .detach()
                .view(zero_input.shape[0], -1)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False),
            )
        )
        previous_snapshot = actual_snapshot
    return tuple(decay_snapshots)


def centered_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    a_centered = a_arr - a_arr.mean(axis=1, keepdims=True)
    b_centered = b_arr - b_arr.mean(axis=1, keepdims=True)
    numerator = np.sum(a_centered * b_centered, axis=1)
    denominator = np.maximum(
        np.linalg.norm(a_centered, axis=1) * np.linalg.norm(b_centered, axis=1),
        float(eps),
    )
    return numerator / denominator


def normalize_nonnegative_weights(scores: Sequence[float], eps: float = 1e-12) -> tuple[np.ndarray, bool]:
    arr = np.clip(np.asarray(scores, dtype=np.float64), 0.0, None)
    total = float(arr.sum())
    if total <= float(eps):
        return np.zeros_like(arr), False
    return arr / total, True


def compute_similarity_profile_summary(weights: np.ndarray) -> dict[str, float | int | None]:
    weights_arr = np.asarray(weights, dtype=np.float64)
    if weights_arr.size == 0 or float(weights_arr.sum()) <= 0.0:
        return {
            "com_sim": None,
            "sim_effective_count": 0.0,
            "similarity_top1_index": None,
            "similarity_top1_mass": 0.0,
        }
    positions = np.arange(1, len(weights_arr) + 1, dtype=np.float64)
    return {
        "com_sim": float(np.sum(positions * weights_arr)),
        "sim_effective_count": float(1.0 / np.sum(np.square(weights_arr))),
        "similarity_top1_index": int(np.argmax(weights_arr) + 1),
        "similarity_top1_mass": float(np.max(weights_arr)),
    }


def run_lag_matched_singleton_reference_bank(
    net: Any,
    batch: SequenceBatch,
    cfg: ExperimentConfig,
) -> dict[int, dict[int, SingletonReferenceSnapshot]]:
    seq_len = int(batch.seq_len)
    first_sequence = batch.item_spikes[0]
    batch_size, _, channels, height, width = first_sequence.shape
    reference_bank: dict[int, dict[int, SingletonReferenceSnapshot]] = {}
    with torch.no_grad():
        for item_index in range(1, seq_len + 1):
            prepare_network_state(net, batch_size, channels, height, width)
            zero_input = torch.zeros(
                (batch_size, channels, height, width),
                dtype=first_sequence.dtype,
                device=first_sequence.device,
            )
            current_time = 0
            for position, item_spikes in enumerate(batch.item_spikes, start=1):
                for step_idx in range(int(item_spikes.shape[1])):
                    input_t = item_spikes[:, step_idx, ...] if position == item_index else zero_input
                    forward_three_layers(net, input_t, current_time)
                    current_time += 1
                for _ in range(int(cfg.delay_steps)):
                    forward_three_layers(net, zero_input, current_time)
                    current_time += 1
                if item_index <= position:
                    layer3_g = (
                        (net.layer3.u_pre * net.layer3.x_pre)
                        .detach()
                        .view(batch_size, -1)
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    reference_bank.setdefault(int(position), {})[int(item_index)] = SingletonReferenceSnapshot(
                        stage_k=int(position),
                        item_index=int(item_index),
                        layer3_g=layer3_g,
                    )
    return reference_bank


def run_neutral_ping_from_snapshot(
    net: Any,
    cfg: ExperimentConfig,
    snapshot: BoundarySnapshot,
    batch: SequenceBatch,
) -> PingReadout:
    batch_size = len(batch.trials)
    label_matrix = np.asarray([trial.ordered_item_labels[: snapshot.stage_k] for trial in batch.trials], dtype=np.int64)
    stage_k = int(snapshot.stage_k)
    with torch.no_grad():
        reset_all_state_restore_selected_stsp_in_place(
            net,
            snapshot.layer_input_shapes,
            restore_ux_by_layer=snapshot.restore_ux_by_layer,
        )
        net.layer3.reset_decision_state()
        net.layer3.v_mem.fill_(net.layer3.V_L)
        net.layer3.lateral_inh.reset_state(net.layer3.output_shape)
        zero_input = torch.zeros(
            snapshot.layer_input_shapes["layer1"],
            dtype=torch.float32,
            device=net.layer1.v_mem.device,
        )
        ping_drive = torch.full_like(zero_input, float(cfg.ping_amp))
        for t_idx in range(int(cfg.ping_steps)):
            forward_three_layers(net, zero_input, t_idx, ping_drive=ping_drive)
    first_fire_pred_t, first_fire_t_t = decode_prediction_and_fire_time_from_layer3(net, batch_size=batch_size)
    first_fire_pred = first_fire_pred_t.numpy().astype(np.int64, copy=False)
    first_fire_t = first_fire_t_t.numpy().astype(np.int64, copy=False)
    silent_mask = (first_fire_pred < 0).astype(np.int64, copy=False)
    predicted_item_index = np.full(batch_size, -1, dtype=np.int64)
    predicted_item_label = np.full(batch_size, -1, dtype=np.int64)
    ping_seen_item_hit = np.zeros(batch_size, dtype=np.int64)
    ping_nonmember_first = np.zeros(batch_size, dtype=np.int64)
    ping_first_item_index = np.full(batch_size, np.nan, dtype=np.float64)
    ping_normalized_recency = np.full(batch_size, np.nan, dtype=np.float64)
    ping_latest_item_hit_raw = np.zeros(batch_size, dtype=np.int64)
    ping_latest_item_hit_chance_corrected = np.full(batch_size, np.nan, dtype=np.float64)
    ping_recent_window_hit = np.zeros(batch_size, dtype=np.int64)
    ping_lag_from_latest = np.full(batch_size, np.nan, dtype=np.float64)
    for row_idx in range(batch_size):
        pred_label = int(first_fire_pred[row_idx])
        if pred_label < 0:
            ping_latest_item_hit_chance_corrected[row_idx] = -1.0 / float(max(stage_k, 1))
            continue
        predicted_item_label[row_idx] = pred_label
        matches = np.where(label_matrix[row_idx] == pred_label)[0]
        if matches.size > 0:
            item_index = int(matches[0] + 1)
            predicted_item_index[row_idx] = item_index
            ping_seen_item_hit[row_idx] = 1
            ping_first_item_index[row_idx] = float(item_index)
            ping_normalized_recency[row_idx] = float((item_index - 1) / float(max(stage_k - 1, 1)))
            ping_latest_item_hit_raw[row_idx] = int(item_index == stage_k)
            ping_latest_item_hit_chance_corrected[row_idx] = float(ping_latest_item_hit_raw[row_idx] - (1.0 / float(max(stage_k, 1))))
            recent_window = max(2, int(math.ceil(0.2 * float(stage_k))))
            recent_start = max(1, stage_k - recent_window + 1)
            ping_recent_window_hit[row_idx] = int(item_index >= recent_start)
            ping_lag_from_latest[row_idx] = float(stage_k - item_index)
        else:
            ping_nonmember_first[row_idx] = 1
            ping_latest_item_hit_chance_corrected[row_idx] = -1.0 / float(max(stage_k, 1))
    return PingReadout(
        stage_k=stage_k,
        boundary_name=str(snapshot.boundary_name),
        first_fire_pred=first_fire_pred,
        first_fire_t=first_fire_t,
        silent_mask=silent_mask,
        predicted_item_index=predicted_item_index,
        predicted_item_label=predicted_item_label,
        ping_seen_item_hit=ping_seen_item_hit,
        ping_nonmember_first=ping_nonmember_first,
        ping_first_item_index=ping_first_item_index,
        ping_normalized_recency=ping_normalized_recency,
        ping_latest_item_hit_raw=ping_latest_item_hit_raw,
        ping_latest_item_hit_chance_corrected=ping_latest_item_hit_chance_corrected,
        ping_recent_window_hit=ping_recent_window_hit,
        ping_lag_from_latest=ping_lag_from_latest,
    )


def _null_top_occupancy_stats(
    active_indices: np.ndarray,
    top_indices: np.ndarray,
    sample_size: int,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if active_indices.size <= 0 or top_indices.size <= 0 or sample_size <= 0 or permutations <= 0:
        return 0.0, 0.0
    sample_n = min(int(sample_size), int(active_indices.size))
    top_set = set(int(item) for item in top_indices.tolist())
    draws: list[float] = []
    for _ in range(int(permutations)):
        sampled = rng.choice(active_indices, size=sample_n, replace=False)
        hit_count = sum(1 for item in sampled.tolist() if int(item) in top_set)
        draws.append(float(hit_count / float(sample_n)))
    if not draws:
        return 0.0, 0.0
    return float(np.mean(draws)), float(np.std(draws))


def compute_rank_topness_metrics(
    actual_g: np.ndarray,
    decay_g: np.ndarray,
    delta_g: np.ndarray,
    *,
    baseline_gain: float,
    epsilon: float,
    null_permutations: int,
    rng_seed: int,
) -> dict[str, float | int | None]:
    actual = np.asarray(actual_g, dtype=np.float64).reshape(-1)
    decay = np.asarray(decay_g, dtype=np.float64).reshape(-1)
    delta = np.asarray(delta_g, dtype=np.float64).reshape(-1)
    positive_changed_mask = delta > float(epsilon)
    negative_changed_mask = delta < -float(epsilon)
    changed_mask = np.abs(delta) > float(epsilon)
    positive_indices = np.where(positive_changed_mask)[0]
    negative_indices = np.where(negative_changed_mask)[0]
    active_mask = np.abs(actual - float(baseline_gain)) > float(epsilon)
    active_indices = np.where(active_mask)[0]
    decay_active_mask = np.abs(decay - float(baseline_gain)) > float(epsilon)
    decay_active_indices = np.where(decay_active_mask)[0]
    positive_delta = np.maximum(delta, 0.0)
    absolute_changed_mass = np.sum(np.abs(delta[changed_mask]))
    active_gain_mass = float(np.sum(actual[active_mask]))
    result: dict[str, float | int | None] = {
        "active_synapse_count": int(active_indices.size),
        "baseline_nontrivial_active_fraction": float(active_mask.mean()) if actual.size > 0 else 0.0,
        "changed_rank_percentile_mean": None,
        "changed_rank_percentile_median": None,
        "changed_rank_percentile_std": None,
        "negative_changed_rank_percentile_mean": None,
        "changed_top_1pct_occupancy": 0.0,
        "changed_top_5pct_occupancy": 0.0,
        "changed_top_10pct_occupancy": 0.0,
        "changed_top_1pct_enrichment": 0.0,
        "changed_top_5pct_enrichment": 0.0,
        "changed_top_10pct_enrichment": 0.0,
        "changed_top_1pct_zscore": None,
        "changed_top_5pct_zscore": None,
        "changed_top_10pct_zscore": None,
        "changed_positive_mass_in_top_1pct": 0.0,
        "changed_positive_mass_in_top_5pct": 0.0,
        "changed_positive_mass_in_top_10pct": 0.0,
        "positive_change_mass_ratio_active": 0.0,
        "positive_change_mass_ratio_top5": 0.0,
        "positive_change_mass_ratio_changed": 0.0,
        "changed_positive_mass_in_top_5pct_ratio": 0.0,
        "old_top_5pct_suppression_mass": 0.0,
        "old_top_5pct_suppression_ratio": 0.0,
        "net_top_5pct_reweight_score": 0.0,
    }
    if active_indices.size <= 0:
        return result

    actual_values = actual[active_indices]
    active_order = np.lexsort((active_indices.astype(np.int64), -actual_values))
    ordered_active_indices = active_indices[active_order]
    actual_rank_lookup = {int(index): int(rank) for rank, index in enumerate(ordered_active_indices.tolist())}
    if positive_indices.size > 0:
        changed_active_indices = np.asarray(
            [int(index) for index in positive_indices.tolist() if int(index) in actual_rank_lookup],
            dtype=np.int64,
        )
        if changed_active_indices.size > 0:
            changed_positions = np.asarray(
                [actual_rank_lookup[int(index)] for index in changed_active_indices.tolist()],
                dtype=np.int64,
            )
            if ordered_active_indices.size == 1:
                percentiles = np.ones(changed_positions.size, dtype=np.float64)
            else:
                percentiles = 1.0 - (
                    changed_positions.astype(np.float64) / float(max(ordered_active_indices.size - 1, 1))
                )
            result["changed_rank_percentile_mean"] = float(np.mean(percentiles))
            result["changed_rank_percentile_median"] = float(np.median(percentiles))
            result["changed_rank_percentile_std"] = float(np.std(percentiles))
            rng = np.random.default_rng(int(rng_seed))
            for q_value, q_label in zip(TOP_Q_LEVELS, TOP_Q_LABELS):
                top_k = max(1, int(math.ceil(float(q_value) * float(ordered_active_indices.size))))
                top_indices = ordered_active_indices[:top_k]
                in_top = changed_positions < top_k
                occupancy = float(np.mean(in_top)) if in_top.size > 0 else 0.0
                result[f"changed_top_{q_label}_occupancy"] = occupancy
                result[f"changed_top_{q_label}_enrichment"] = float(occupancy / float(q_value))
                top_hits = changed_active_indices[in_top]
                result[f"changed_positive_mass_in_top_{q_label}"] = float(np.sum(positive_delta[top_hits])) if top_hits.size > 0 else 0.0
                null_mean, null_std = _null_top_occupancy_stats(
                    active_indices=active_indices,
                    top_indices=top_indices,
                    sample_size=int(changed_active_indices.size),
                    permutations=int(null_permutations),
                    rng=rng,
                )
                if null_std > 0.0:
                    result[f"changed_top_{q_label}_zscore"] = float((occupancy - null_mean) / null_std)
            top5_k = max(1, int(math.ceil(0.05 * float(ordered_active_indices.size))))
            top5_indices = ordered_active_indices[:top5_k]
            top5_mass = float(np.sum(actual[top5_indices]))
            positive_mass = float(np.sum(positive_delta))
            result["positive_change_mass_ratio_active"] = float(positive_mass / max(active_gain_mass, 1e-12))
            result["positive_change_mass_ratio_top5"] = float(positive_mass / max(top5_mass, 1e-12))
            result["positive_change_mass_ratio_changed"] = float(positive_mass / max(float(absolute_changed_mass), 1e-12))
            result["changed_positive_mass_in_top_5pct_ratio"] = float(
                result["changed_positive_mass_in_top_5pct"] / max(positive_mass, 1e-12)
            )

    if decay_active_indices.size > 0 and negative_indices.size > 0:
        decay_values = decay[decay_active_indices]
        decay_order = np.lexsort((decay_active_indices.astype(np.int64), -decay_values))
        ordered_decay_active_indices = decay_active_indices[decay_order]
        decay_rank_lookup = {int(index): int(rank) for rank, index in enumerate(ordered_decay_active_indices.tolist())}
        negative_decay_indices = np.asarray(
            [int(index) for index in negative_indices.tolist() if int(index) in decay_rank_lookup],
            dtype=np.int64,
        )
        if negative_decay_indices.size > 0:
            negative_positions = np.asarray(
                [decay_rank_lookup[int(index)] for index in negative_decay_indices.tolist()],
                dtype=np.int64,
            )
            if ordered_decay_active_indices.size == 1:
                negative_percentiles = np.ones(negative_positions.size, dtype=np.float64)
            else:
                negative_percentiles = 1.0 - (
                    negative_positions.astype(np.float64) / float(max(ordered_decay_active_indices.size - 1, 1))
                )
            result["negative_changed_rank_percentile_mean"] = float(np.mean(negative_percentiles))
            old_top5_k = max(1, int(math.ceil(0.05 * float(ordered_decay_active_indices.size))))
            old_top5_indices = ordered_decay_active_indices[:old_top5_k]
            negative_delta_mass = np.maximum(-delta, 0.0)
            result["old_top_5pct_suppression_mass"] = float(np.sum(negative_delta_mass[old_top5_indices]))
            old_top5_decay_mass = float(np.sum(decay[old_top5_indices]))
            result["old_top_5pct_suppression_ratio"] = float(
                result["old_top_5pct_suppression_mass"] / max(old_top5_decay_mass, 1e-12)
            )
            result["net_top_5pct_reweight_score"] = float(
                float(result["changed_top_5pct_enrichment"]) - float(result["old_top_5pct_suppression_ratio"])
            )
    return result


def compute_state_similarity_metrics(
    actual_g: np.ndarray,
    reference_snapshots: Mapping[int, SingletonReferenceSnapshot],
    *,
    stage_k: int,
) -> dict[str, float | int | None]:
    similarity_scores: list[float] = []
    for item_index in range(1, int(stage_k) + 1):
        reference = reference_snapshots[int(item_index)]
        similarity = centered_cosine_similarity(
            np.asarray(actual_g, dtype=np.float64).reshape(1, -1),
            np.asarray(reference.layer3_g, dtype=np.float64).reshape(1, -1),
        )[0]
        similarity_scores.append(float(similarity))
    weights, has_positive = normalize_nonnegative_weights(similarity_scores)
    summary = compute_similarity_profile_summary(weights)
    output: dict[str, float | int | None] = {
        "com_sim": summary["com_sim"],
        "sim_effective_count": float(summary["sim_effective_count"]),
        "similarity_top1_index": summary["similarity_top1_index"],
        "similarity_top1_mass": float(summary["similarity_top1_mass"]),
    }
    for item_index in range(1, int(stage_k) + 1):
        output[f"similarity_item_{item_index}"] = float(similarity_scores[item_index - 1])
        output[f"similarity_weight_{item_index}"] = float(weights[item_index - 1]) if has_positive else 0.0
    return output


def compute_trial_stage_rows(
    batch: SequenceBatch,
    rollout: SequenceRollout,
    baseline_gain: float,
    epsilon: float,
    null_permutations: int,
    ping_by_stage: Mapping[int, PingReadout],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    changed_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    ping_rows: list[dict[str, object]] = []
    decay_lookup = {int(snapshot.stage_k): snapshot for snapshot in rollout.decay_snapshots}
    for actual_snapshot in rollout.stage_snapshots:
        stage_k = int(actual_snapshot.stage_k)
        decay_snapshot = decay_lookup[stage_k]
        ping = ping_by_stage[stage_k]
        reference_stage_bank = rollout.reference_bank[stage_k]
        delta_g = np.asarray(actual_snapshot.layer3_g, dtype=np.float64) - np.asarray(decay_snapshot.layer3_g_decay, dtype=np.float64)
        for row_idx, trial in enumerate(batch.trials):
            actual_g = np.asarray(actual_snapshot.layer3_g[row_idx], dtype=np.float64)
            actual_u = np.asarray(actual_snapshot.layer3_u[row_idx], dtype=np.float64)
            actual_x = np.asarray(actual_snapshot.layer3_x[row_idx], dtype=np.float64)
            decay_g = np.asarray(decay_snapshot.layer3_g_decay[row_idx], dtype=np.float64)
            decay_u = np.asarray(decay_snapshot.layer3_u_decay[row_idx], dtype=np.float64)
            decay_x = np.asarray(decay_snapshot.layer3_x_decay[row_idx], dtype=np.float64)
            delta = np.asarray(delta_g[row_idx], dtype=np.float64)
            changed_mask = np.abs(delta) > float(epsilon)
            positive_mask = delta > float(epsilon)
            negative_mask = delta < -float(epsilon)
            positive_values = delta[positive_mask]
            positive_mass = float(np.sum(positive_values)) if positive_values.size > 0 else 0.0
            actual_total_gain = float(np.sum(actual_g))
            rank_metrics = compute_rank_topness_metrics(
                actual_g,
                decay_g,
                delta,
                baseline_gain=float(baseline_gain),
                epsilon=float(epsilon),
                null_permutations=int(null_permutations),
                rng_seed=mix_seed(int(trial.sequence_seed), int(stage_k), int(row_idx), int(round(float(epsilon) * 1e9))),
            )
            similarity_metrics = compute_state_similarity_metrics(
                actual_g,
                {item_index: SingletonReferenceSnapshot(stage_k=stage_k, item_index=item_index, layer3_g=reference_stage_bank[item_index].layer3_g[row_idx]) for item_index in range(1, stage_k + 1)},
                stage_k=stage_k,
            )
            current_active_mask = np.abs(actual_g - float(baseline_gain)) > float(epsilon)
            current_active_indices = np.where(current_active_mask)[0]
            current_active_values = actual_g[current_active_indices]
            current_active_order = np.lexsort((current_active_indices.astype(np.int64), -current_active_values)) if current_active_indices.size > 0 else np.asarray([], dtype=np.int64)
            current_top5_indices = current_active_indices[current_active_order[: max(1, int(math.ceil(0.05 * float(max(current_active_indices.size, 1)))))]]
            local_top5_mass = float(np.sum(actual_g[current_top5_indices])) if current_top5_indices.size > 0 else 0.0
            base_row = {
                "record_type": "trial_level",
                "trial_id": int(trial.trial_id),
                "trial_index_within_seq_len": int(trial.trial_index_within_seq_len),
                "seq_len": int(trial.seq_len),
                "stage_k": int(stage_k),
                "epsilon": float(epsilon),
                "boundary_name": str(actual_snapshot.boundary_name),
                "latest_item_index": int(stage_k),
                "latest_item_label": int(trial.ordered_item_labels[stage_k - 1]),
                "sequence_seed": int(trial.sequence_seed),
                "current_time": int(actual_snapshot.current_time),
                "layer3_synapse_count": int(actual_g.size),
                "layer3_actual_total_gain": float(actual_total_gain),
                "layer3_decay_total_gain": float(np.sum(decay_g)),
                "layer3_delta_gain_sum": float(np.sum(delta)),
                "layer3_actual_u_mean": float(np.mean(actual_u)),
                "layer3_actual_x_mean": float(np.mean(actual_x)),
                "layer3_decay_u_mean": float(np.mean(decay_u)),
                "layer3_decay_x_mean": float(np.mean(decay_x)),
                "com_sim": similarity_metrics["com_sim"],
                "sim_effective_count": similarity_metrics["sim_effective_count"],
                "similarity_top1_index": similarity_metrics["similarity_top1_index"],
                "similarity_top1_mass": similarity_metrics["similarity_top1_mass"],
            }
            changed_row = {
                **base_row,
                "changed_synapse_fraction": float(changed_mask.mean()),
                "changed_synapse_count": int(np.count_nonzero(changed_mask)),
                "positive_changed_count": int(np.count_nonzero(positive_mask)),
                "negative_changed_count": int(np.count_nonzero(negative_mask)),
                "positive_change_mass": float(positive_mass),
                "positive_change_mass_ratio": float(positive_mass / max(actual_total_gain, float(epsilon), 1e-12)),
                "positive_change_mass_ratio_active": float(rank_metrics["positive_change_mass_ratio_active"]),
                "positive_change_mass_ratio_top5": float(rank_metrics["positive_change_mass_ratio_top5"]),
                "positive_change_mass_ratio_changed": float(rank_metrics["positive_change_mass_ratio_changed"]),
                "mean_positive_delta_g": float(np.mean(positive_values)) if positive_values.size > 0 else None,
                "median_positive_delta_g": float(np.median(positive_values)) if positive_values.size > 0 else None,
                "baseline_nontrivial_active_fraction": rank_metrics["baseline_nontrivial_active_fraction"],
                "changed_positive_mass_in_top_1pct": rank_metrics["changed_positive_mass_in_top_1pct"],
                "changed_positive_mass_in_top_5pct": rank_metrics["changed_positive_mass_in_top_5pct"],
                "changed_positive_mass_in_top_10pct": rank_metrics["changed_positive_mass_in_top_10pct"],
                "changed_positive_mass_in_top_5pct_ratio": float(rank_metrics["changed_positive_mass_in_top_5pct_ratio"]),
                "top5_active_gain_mass": float(local_top5_mass),
            }
            rank_row = {
                **base_row,
                "active_synapse_count": rank_metrics["active_synapse_count"],
                "baseline_nontrivial_active_fraction": rank_metrics["baseline_nontrivial_active_fraction"],
                "changed_rank_percentile_mean": rank_metrics["changed_rank_percentile_mean"],
                "changed_rank_percentile_median": rank_metrics["changed_rank_percentile_median"],
                "changed_rank_percentile_std": rank_metrics["changed_rank_percentile_std"],
                "negative_changed_rank_percentile_mean": rank_metrics["negative_changed_rank_percentile_mean"],
                "changed_top_1pct_occupancy": rank_metrics["changed_top_1pct_occupancy"],
                "changed_top_5pct_occupancy": rank_metrics["changed_top_5pct_occupancy"],
                "changed_top_10pct_occupancy": rank_metrics["changed_top_10pct_occupancy"],
                "changed_top_1pct_enrichment": rank_metrics["changed_top_1pct_enrichment"],
                "changed_top_5pct_enrichment": rank_metrics["changed_top_5pct_enrichment"],
                "changed_top_10pct_enrichment": rank_metrics["changed_top_10pct_enrichment"],
                "changed_top_1pct_zscore": rank_metrics["changed_top_1pct_zscore"],
                "changed_top_5pct_zscore": rank_metrics["changed_top_5pct_zscore"],
                "changed_top_10pct_zscore": rank_metrics["changed_top_10pct_zscore"],
                "changed_positive_mass_in_top_1pct": rank_metrics["changed_positive_mass_in_top_1pct"],
                "changed_positive_mass_in_top_5pct": rank_metrics["changed_positive_mass_in_top_5pct"],
                "changed_positive_mass_in_top_10pct": rank_metrics["changed_positive_mass_in_top_10pct"],
                "old_top_5pct_suppression_mass": rank_metrics["old_top_5pct_suppression_mass"],
                "old_top_5pct_suppression_ratio": rank_metrics["old_top_5pct_suppression_ratio"],
                "net_top_5pct_reweight_score": rank_metrics["net_top_5pct_reweight_score"],
            }
            ping_row = {
                **base_row,
                "positive_change_mass_ratio_active": float(rank_metrics["positive_change_mass_ratio_active"]),
                "changed_positive_mass_in_top_5pct_ratio": float(rank_metrics["changed_positive_mass_in_top_5pct_ratio"]),
                "changed_rank_percentile_mean": rank_metrics["changed_rank_percentile_mean"],
                "changed_top_5pct_occupancy": rank_metrics["changed_top_5pct_occupancy"],
                "changed_top_5pct_enrichment": rank_metrics["changed_top_5pct_enrichment"],
                "changed_top_5pct_zscore": rank_metrics["changed_top_5pct_zscore"],
                "changed_topness_default": rank_metrics["changed_top_5pct_enrichment"],
                "ping_predicted_item_index": None
                if int(ping.predicted_item_index[row_idx]) < 0
                else int(ping.predicted_item_index[row_idx]),
                "ping_predicted_item_label": None
                if int(ping.predicted_item_label[row_idx]) < 0
                else int(ping.predicted_item_label[row_idx]),
                "ping_seen_item_hit": int(ping.ping_seen_item_hit[row_idx]),
                "ping_nonmember_first": int(ping.ping_nonmember_first[row_idx]),
                "ping_silent": int(ping.silent_mask[row_idx]),
                "ping_first_fire_pred": int(ping.first_fire_pred[row_idx]),
                "ping_first_fire_t": int(ping.first_fire_t[row_idx]),
                "ping_first_item_index": safe_float(ping.ping_first_item_index[row_idx]),
                "ping_com": safe_float(ping.ping_first_item_index[row_idx]),
                "ping_normalized_recency": safe_float(ping.ping_normalized_recency[row_idx]),
                "ping_latest_item_hit": int(ping.ping_latest_item_hit_raw[row_idx]),
                "ping_latest_item_hit_raw": int(ping.ping_latest_item_hit_raw[row_idx]),
                "ping_latest_item_hit_chance_corrected": safe_float(ping.ping_latest_item_hit_chance_corrected[row_idx]),
                "ping_recent_window_hit": int(ping.ping_recent_window_hit[row_idx]),
                "ping_lag_from_latest": safe_float(ping.ping_lag_from_latest[row_idx]),
            }
            for item_index in range(1, stage_k + 1):
                ping_row[f"similarity_item_{item_index}"] = similarity_metrics[f"similarity_item_{item_index}"]
                ping_row[f"similarity_weight_{item_index}"] = similarity_metrics[f"similarity_weight_{item_index}"]
            changed_rows.append(changed_row)
            rank_rows.append(rank_row)
            ping_rows.append(ping_row)
    return changed_rows, rank_rows, ping_rows


def add_anchor_shift_columns(df_ping: pd.DataFrame) -> pd.DataFrame:
    if df_ping.empty:
        return df_ping.copy()
    df = df_ping.copy()
    df["stage_to_stage_anchor_shift"] = np.nan
    trial_mask = df["record_type"] == "trial_level"
    for _, sub_idx in df[trial_mask].groupby(["trial_id", "seq_len"], sort=False).groups.items():
        ordered_idx = df.loc[list(sub_idx)].sort_values("stage_k", kind="stable").index.to_list()
        previous_com_sim: float | None = None
        for idx in ordered_idx:
            current_com_sim = safe_float(df.at[idx, "com_sim"])
            if current_com_sim is not None and previous_com_sim is not None:
                df.at[idx, "stage_to_stage_anchor_shift"] = float(current_com_sim - previous_com_sim)
            previous_com_sim = current_com_sim
    summary_mask = df["record_type"] == "stage_summary"
    if summary_mask.any():
        summary = (
            df[trial_mask]
            .groupby(["seq_len", "stage_k"], as_index=False)["stage_to_stage_anchor_shift"]
            .mean(numeric_only=True)
        )
        df = df.merge(summary, on=["seq_len", "stage_k"], how="left", suffixes=("", "_summary"))
        df.loc[summary_mask, "stage_to_stage_anchor_shift"] = df.loc[summary_mask, "stage_to_stage_anchor_shift_summary"]
        df = df.drop(columns=["stage_to_stage_anchor_shift_summary"])
    return df


def build_example_state_payload(
    batch: SequenceBatch,
    rollout: SequenceRollout,
    *,
    max_trials: int = 2,
) -> dict[str, np.ndarray]:
    keep_trials = min(int(max_trials), len(batch.trials))
    payload: dict[str, np.ndarray] = {
        "trial_ids": np.asarray([int(trial.trial_id) for trial in batch.trials[:keep_trials]], dtype=np.int64),
        "seq_len": np.asarray([int(batch.seq_len)], dtype=np.int64),
        "ordered_item_labels": np.asarray(
            [trial.ordered_item_labels for trial in batch.trials[:keep_trials]],
            dtype=np.int64,
        ),
    }
    decay_lookup = {int(snapshot.stage_k): snapshot for snapshot in rollout.decay_snapshots}
    for snapshot in rollout.stage_snapshots:
        stage_k = int(snapshot.stage_k)
        decay_snapshot = decay_lookup[stage_k]
        payload[f"stage_{stage_k}_actual_u"] = snapshot.layer3_u[:keep_trials].astype(np.float32, copy=False)
        payload[f"stage_{stage_k}_actual_x"] = snapshot.layer3_x[:keep_trials].astype(np.float32, copy=False)
        payload[f"stage_{stage_k}_actual_g"] = snapshot.layer3_g[:keep_trials].astype(np.float32, copy=False)
        payload[f"stage_{stage_k}_decay_u"] = decay_snapshot.layer3_u_decay[:keep_trials].astype(np.float32, copy=False)
        payload[f"stage_{stage_k}_decay_x"] = decay_snapshot.layer3_x_decay[:keep_trials].astype(np.float32, copy=False)
        payload[f"stage_{stage_k}_decay_g"] = decay_snapshot.layer3_g_decay[:keep_trials].astype(np.float32, copy=False)
        payload[f"stage_{stage_k}_delta_g"] = (
            snapshot.layer3_g[:keep_trials] - decay_snapshot.layer3_g_decay[:keep_trials]
        ).astype(np.float32, copy=False)
    return payload


def residualize_against_stage_and_seq(df: pd.DataFrame, value_col: str) -> np.ndarray:
    y = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=np.float64)
    stage = pd.to_numeric(df["stage_k"], errors="coerce").to_numpy(dtype=np.float64)
    seq_len = pd.to_numeric(df["seq_len"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(stage) & np.isfinite(seq_len)
    residuals = np.full(y.shape, np.nan, dtype=np.float64)
    if np.count_nonzero(valid) < 2:
        return residuals
    x = np.column_stack(
        [
            np.ones(np.count_nonzero(valid), dtype=np.float64),
            stage[valid],
            seq_len[valid],
        ]
    )
    beta, _, _, _ = np.linalg.lstsq(x, y[valid], rcond=None)
    fitted = x @ beta
    residuals[valid] = y[valid] - fitted
    return residuals


def build_metric_correlation_rows(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] | None,
    topness_cols: Sequence[str],
    outcome_cols: Sequence[str],
    record_type: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_iterable: Iterable[tuple[object, pd.DataFrame]]
    if group_cols:
        group_iterable = df.groupby(list(group_cols), dropna=False, sort=True)
    else:
        group_iterable = [(None, df)]
    for group_key, sub in group_iterable:
        if group_cols:
            group_values = group_key if isinstance(group_key, tuple) else (group_key,)
        else:
            group_values = tuple()
        for topness_col in topness_cols:
            for outcome_col in outcome_cols:
                corr_value = safe_corr(
                    pd.to_numeric(sub[topness_col], errors="coerce").to_numpy(dtype=np.float64),
                    pd.to_numeric(sub[outcome_col], errors="coerce").to_numpy(dtype=np.float64),
                )
                row = {
                    "record_type": str(record_type),
                    "topness_metric": str(topness_col),
                    "outcome_metric": str(outcome_col),
                    "trial_count": int(len(sub)),
                    "correlation": corr_value,
                }
                for idx, group_col in enumerate(group_cols or ()):
                    row[str(group_col)] = json_safe(group_values[idx])
                rows.append(row)
    return rows


def build_stage_controlled_correlation_rows(
    df: pd.DataFrame,
    *,
    topness_cols: Sequence[str],
    outcome_cols: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for topness_col in topness_cols:
        x_resid = residualize_against_stage_and_seq(df, topness_col)
        for outcome_col in outcome_cols:
            y_resid = residualize_against_stage_and_seq(df, outcome_col)
            rows.append(
                {
                    "record_type": "stage_controlled_summary",
                    "method": "residualized_linear_stage_seq",
                    "topness_metric": str(topness_col),
                    "outcome_metric": str(outcome_col),
                    "trial_count": int(len(df)),
                    "correlation": safe_corr(x_resid, y_resid),
                }
            )
    return rows


def build_epsilon_robustness_rows(
    batch: SequenceBatch,
    rollout: SequenceRollout,
    ping_by_stage: Mapping[int, PingReadout],
    *,
    baseline_gain: float,
    epsilon_values: Sequence[float],
    null_permutations: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    decay_lookup = {int(snapshot.stage_k): snapshot for snapshot in rollout.decay_snapshots}
    for actual_snapshot in rollout.stage_snapshots:
        stage_k = int(actual_snapshot.stage_k)
        decay_snapshot = decay_lookup[stage_k]
        ping = ping_by_stage[stage_k]
        reference_stage_bank = rollout.reference_bank[stage_k]
        for row_idx, trial in enumerate(batch.trials):
            actual_g = np.asarray(actual_snapshot.layer3_g[row_idx], dtype=np.float64)
            decay_g = np.asarray(decay_snapshot.layer3_g_decay[row_idx], dtype=np.float64)
            delta = actual_g - decay_g
            similarity_metrics = compute_state_similarity_metrics(
                actual_g,
                {
                    item_index: SingletonReferenceSnapshot(
                        stage_k=stage_k,
                        item_index=item_index,
                        layer3_g=reference_stage_bank[item_index].layer3_g[row_idx],
                    )
                    for item_index in range(1, stage_k + 1)
                },
                stage_k=stage_k,
            )
            for epsilon in epsilon_values:
                rank_metrics = compute_rank_topness_metrics(
                    actual_g,
                    decay_g,
                    delta,
                    baseline_gain=float(baseline_gain),
                    epsilon=float(epsilon),
                    null_permutations=int(null_permutations),
                    rng_seed=mix_seed(int(trial.sequence_seed), int(stage_k), int(row_idx), int(round(float(epsilon) * 1e9)), 701),
                )
                changed_mask = np.abs(delta) > float(epsilon)
                rows.append(
                    {
                        "record_type": "trial_level",
                        "epsilon": float(epsilon),
                        "trial_id": int(trial.trial_id),
                        "seq_len": int(trial.seq_len),
                        "stage_k": int(stage_k),
                        "changed_synapse_fraction": float(changed_mask.mean()),
                        "changed_rank_percentile_mean": rank_metrics["changed_rank_percentile_mean"],
                        "changed_top_5pct_enrichment": rank_metrics["changed_top_5pct_enrichment"],
                        "ping_normalized_recency": safe_float(ping.ping_normalized_recency[row_idx]),
                        "com_sim": similarity_metrics["com_sim"],
                    }
                )
    return rows


def summarize_epsilon_robustness(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["stage_to_stage_anchor_shift"] = np.nan
    for _, sub_idx in work.groupby(["epsilon", "trial_id", "seq_len"], sort=False).groups.items():
        ordered_idx = work.loc[list(sub_idx)].sort_values("stage_k", kind="stable").index.to_list()
        previous_com_sim: float | None = None
        for idx in ordered_idx:
            current_com_sim = safe_float(work.at[idx, "com_sim"])
            if current_com_sim is not None and previous_com_sim is not None:
                work.at[idx, "stage_to_stage_anchor_shift"] = float(current_com_sim - previous_com_sim)
            previous_com_sim = current_com_sim
    summary_rows: list[dict[str, object]] = []
    for epsilon, sub in work.groupby("epsilon", sort=True):
        summary_rows.append(
            {
                "record_type": "epsilon_summary",
                "epsilon": float(epsilon),
                "mean_changed_synapse_fraction": safe_float(sub["changed_synapse_fraction"].mean()),
                "mean_changed_rank_percentile_mean": safe_float(sub["changed_rank_percentile_mean"].mean()),
                "mean_changed_top_5pct_enrichment": safe_float(sub["changed_top_5pct_enrichment"].mean()),
                "changed_topness_vs_normalized_recency_corr": safe_corr(
                    sub["changed_top_5pct_enrichment"].to_numpy(dtype=np.float64, copy=False),
                    sub["ping_normalized_recency"].to_numpy(dtype=np.float64, copy=False),
                ),
                "changed_topness_vs_anchor_shift_corr": safe_corr(
                    sub["changed_top_5pct_enrichment"].to_numpy(dtype=np.float64, copy=False),
                    sub["stage_to_stage_anchor_shift"].to_numpy(dtype=np.float64, copy=False),
                ),
            }
        )
    return pd.concat([work, pd.DataFrame(summary_rows)], axis=0, ignore_index=True, sort=False)


def _summary_only(df: pd.DataFrame) -> pd.DataFrame:
    if "record_type" not in df.columns:
        return df.copy()
    summary = df[df["record_type"] == "stage_summary"].copy()
    return summary if not summary.empty else df.copy()


def _plot_stage_lines(
    df_summary: pd.DataFrame,
    *,
    y_col: str,
    title: str,
    ylabel: str,
    legend_title: str = "seq_len",
    y_limits: tuple[float, float] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    color_cycle = [SAMPLE_COLOR, DYNAMIC_COLOR, SHUFFLE_COLOR, NOISE_COLOR]
    for idx, (seq_len, sub) in enumerate(df_summary.groupby("seq_len", sort=True)):
        sub_sorted = sub.sort_values("stage_k", kind="stable")
        ax.plot(
            sub_sorted["stage_k"].to_numpy(dtype=np.int64, copy=False),
            sub_sorted[y_col].to_numpy(dtype=np.float64, copy=False),
            marker="o",
            linewidth=1.8,
            color=color_cycle[idx % len(color_cycle)],
            label=f"{legend_title}={int(seq_len)}",
        )
    ax.set_xlabel("stage_k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_rank_figure(df_rank_summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=False)
    ax_left, ax_right = axes
    colors = [SAMPLE_COLOR, DYNAMIC_COLOR, SHUFFLE_COLOR, NOISE_COLOR]
    for idx, (seq_len, sub) in enumerate(df_rank_summary.groupby("seq_len", sort=True)):
        sub_sorted = sub.sort_values("stage_k", kind="stable")
        color = colors[idx % len(colors)]
        ax_left.plot(
            sub_sorted["stage_k"].to_numpy(dtype=np.int64, copy=False),
            sub_sorted["changed_rank_percentile_mean"].to_numpy(dtype=np.float64, copy=False),
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"seq_len={int(seq_len)}",
        )
        ax_right.plot(
            sub_sorted["stage_k"].to_numpy(dtype=np.int64, copy=False),
            sub_sorted["changed_top_5pct_enrichment"].to_numpy(dtype=np.float64, copy=False),
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"seq_len={int(seq_len)}",
        )
    ax_left.set_xlabel("stage_k")
    ax_left.set_ylabel("mean percentile")
    ax_left.set_ylim(0.0, 1.0)
    ax_left.set_title("Changed-synapse rank percentile")
    ax_right.set_xlabel("stage_k")
    ax_right.set_ylabel("enrichment")
    ax_right.set_title("Changed top-5% enrichment")
    ax_left.legend(frameon=False)
    fig.tight_layout()
    return fig


def _binned_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_bins: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    x_valid = x[valid]
    y_valid = y[valid]
    quantiles = np.linspace(0.0, 1.0, num=max(2, int(num_bins) + 1))
    edges = np.quantile(x_valid, quantiles)
    edges = np.unique(edges)
    if edges.size <= 1:
        return np.asarray([float(np.mean(x_valid))], dtype=np.float64), np.asarray([float(np.mean(y_valid))], dtype=np.float64)
    mids: list[float] = []
    means: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        mask = (x_valid >= left) & (x_valid <= right if right == edges[-1] else x_valid < right)
        if not np.any(mask):
            continue
        mids.append(float(np.mean(x_valid[mask])))
        means.append(float(np.mean(y_valid[mask])))
    return np.asarray(mids, dtype=np.float64), np.asarray(means, dtype=np.float64)


def _plot_ping_coupling(df_ping_trials: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax_recency, ax_anchor = axes
    colors = [SAMPLE_COLOR, DYNAMIC_COLOR, SHUFFLE_COLOR, NOISE_COLOR]
    for idx, (seq_len, sub) in enumerate(df_ping_trials.groupby("seq_len", sort=True)):
        color = colors[idx % len(colors)]
        x = sub["changed_topness_default"].to_numpy(dtype=np.float64, copy=False)
        y_recency = sub["ping_normalized_recency"].to_numpy(dtype=np.float64, copy=False)
        y_anchor = sub["stage_to_stage_anchor_shift"].to_numpy(dtype=np.float64, copy=False)
        ax_recency.scatter(x, y_recency, s=20, alpha=0.35, color=color, label=f"seq_len={int(seq_len)}")
        ax_anchor.scatter(x, y_anchor, s=20, alpha=0.35, color=color, label=f"seq_len={int(seq_len)}")
        bx_recency, by_recency = _binned_curve(x, y_recency)
        bx_anchor, by_anchor = _binned_curve(x, y_anchor)
        if bx_recency.size > 0:
            ax_recency.plot(bx_recency, by_recency, linewidth=2.0, color=color)
        if bx_anchor.size > 0:
            ax_anchor.plot(bx_anchor, by_anchor, linewidth=2.0, color=color)
    ax_recency.set_xlabel("changed-topness (top-5% enrichment)")
    ax_recency.set_ylabel("ping normalized recency")
    ax_recency.set_ylim(-0.05, 1.05)
    ax_recency.set_title("Changed-topness vs normalized recency")
    ax_anchor.set_xlabel("changed-topness (top-5% enrichment)")
    ax_anchor.set_ylabel("state-based anchor shift")
    ax_anchor.set_title("Changed-topness vs state-based anchor shift")
    ax_recency.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_latest_hit_auxiliary(df_ping_trials: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = [SAMPLE_COLOR, DYNAMIC_COLOR, SHUFFLE_COLOR, NOISE_COLOR]
    for idx, (seq_len, sub) in enumerate(df_ping_trials.groupby("seq_len", sort=True)):
        color = colors[idx % len(colors)]
        x = sub["changed_topness_default"].to_numpy(dtype=np.float64, copy=False)
        y = sub["ping_latest_item_hit_chance_corrected"].to_numpy(dtype=np.float64, copy=False)
        ax.scatter(x, y, s=20, alpha=0.35, color=color, label=f"seq_len={int(seq_len)}")
        bx, by = _binned_curve(x, y)
        if bx.size > 0:
            ax.plot(bx, by, linewidth=2.0, color=color)
    ax.set_xlabel("changed-topness (top-5% enrichment)")
    ax.set_ylabel("chance-corrected latest hit")
    ax.set_title("Changed-topness vs chance-corrected latest hit")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def save_figures(
    layout: Any,
    *,
    df_changed: pd.DataFrame,
    df_rank: pd.DataFrame,
    df_ping: pd.DataFrame,
) -> dict[str, object]:
    apply_publication_style()
    figure_paths: dict[str, object] = {}
    changed_summary = _summary_only(df_changed)
    rank_summary = _summary_only(df_rank)
    ping_trials = df_ping[df_ping["record_type"] == "trial_level"].copy()
    fig_changed = _plot_stage_lines(
        changed_summary,
        y_col="changed_synapse_fraction",
        title="Changed synapse fraction vs stage",
        ylabel="changed fraction",
        y_limits=(0.0, 1.0),
    )
    figure_paths["changed_synapse_fraction_vs_stage"] = save_figure_all_formats(
        fig_changed,
        layout.figure_base("changed_synapse_fraction_vs_stage"),
    )
    plt.close(fig_changed)
    fig_mass = _plot_stage_lines(
        changed_summary,
        y_col="positive_change_mass_ratio_active",
        title="Positive change mass active-ratio vs stage",
        ylabel="positive mass / active gain",
        y_limits=(0.0, 1.0),
    )
    figure_paths["positive_change_mass_vs_stage"] = save_figure_all_formats(
        fig_mass,
        layout.figure_base("positive_change_mass_vs_stage"),
    )
    plt.close(fig_mass)
    fig_rank = _plot_rank_figure(rank_summary)
    figure_paths["changed_rank_enrichment"] = save_figure_all_formats(
        fig_rank,
        layout.figure_base("changed_rank_enrichment"),
    )
    plt.close(fig_rank)
    fig_ping = _plot_ping_coupling(ping_trials)
    figure_paths["ping_coupling_with_changed_topness"] = save_figure_all_formats(
        fig_ping,
        layout.figure_base("ping_coupling_with_changed_topness"),
    )
    plt.close(fig_ping)
    fig_latest = _plot_latest_hit_auxiliary(ping_trials)
    figure_paths["changed_topness_vs_chance_corrected_latest_hit"] = save_figure_all_formats(
        fig_latest,
        layout.figure_base("changed_topness_vs_chance_corrected_latest_hit"),
    )
    plt.close(fig_latest)
    return figure_paths


def build_summary_payload(
    cfg: ExperimentConfig,
    *,
    device_requested: str,
    device_resolved: str,
    baseline_gain: float,
    sequence_count: int,
    df_changed: pd.DataFrame,
    df_rank: pd.DataFrame,
    df_ping: pd.DataFrame,
    df_raw_global_corr: pd.DataFrame,
    df_stage_matched_corr: pd.DataFrame,
    df_stage_controlled_corr: pd.DataFrame,
    df_epsilon: pd.DataFrame,
    exported_files: Mapping[str, object],
) -> dict[str, object]:
    changed_trials = df_changed[df_changed["record_type"] == "trial_level"].copy()
    rank_trials = df_rank[df_rank["record_type"] == "trial_level"].copy()
    ping_trials = df_ping[df_ping["record_type"] == "trial_level"].copy()
    final_stage_summary_by_seq_len: dict[str, object] = {}
    for seq_len in sorted(changed_trials["seq_len"].dropna().unique().tolist()):
        seq_len_int = int(seq_len)
        final_stage = seq_len_int
        seq_changed = changed_trials[(changed_trials["seq_len"] == seq_len_int) & (changed_trials["stage_k"] == final_stage)].copy()
        seq_rank = rank_trials[(rank_trials["seq_len"] == seq_len_int) & (rank_trials["stage_k"] == final_stage)].copy()
        seq_ping = ping_trials[(ping_trials["seq_len"] == seq_len_int) & (ping_trials["stage_k"] == final_stage)].copy()
        final_stage_summary_by_seq_len[str(seq_len_int)] = {
            "final_stage_k": int(final_stage),
            "mean_changed_synapse_fraction": safe_float(seq_changed["changed_synapse_fraction"].mean()) if not seq_changed.empty else None,
            "mean_positive_change_mass_ratio_active": safe_float(seq_changed["positive_change_mass_ratio_active"].mean()) if not seq_changed.empty else None,
            "mean_changed_rank_percentile": safe_float(seq_rank["changed_rank_percentile_mean"].mean()) if not seq_rank.empty else None,
            "mean_changed_top_5pct_enrichment": safe_float(seq_rank["changed_top_5pct_enrichment"].mean()) if not seq_rank.empty else None,
            "mean_ping_normalized_recency": safe_float(seq_ping["ping_normalized_recency"].mean()) if not seq_ping.empty else None,
            "mean_ping_latest_item_hit_chance_corrected": safe_float(seq_ping["ping_latest_item_hit_chance_corrected"].mean()) if not seq_ping.empty else None,
            "mean_state_based_anchor_shift": safe_float(seq_ping["stage_to_stage_anchor_shift"].mean()) if not seq_ping.empty else None,
        }
    epsilon_summary = df_epsilon[df_epsilon["record_type"] == "epsilon_summary"].copy() if not df_epsilon.empty else pd.DataFrame()
    return {
        "experiment_name": "layer3_stsp_anchor_drift_mechanism",
        "scientific_focus": "layer3 anchor drift mechanism via natural-decay counterfactual, changed synapses, high-rank STSP support, and ping coupling",
        "config": json_safe(asdict(cfg)),
        "device_requested": str(device_requested),
        "device_resolved": str(device_resolved),
        "baseline_gain": float(baseline_gain),
        "sequence_count": int(sequence_count),
        "final_stage_summary_by_seq_len": json_safe(final_stage_summary_by_seq_len),
        "raw_global_correlations": json_safe(df_raw_global_corr.to_dict(orient="records")),
        "stage_matched_correlations": json_safe(df_stage_matched_corr.to_dict(orient="records")),
        "stage_controlled_correlations": json_safe(df_stage_controlled_corr.to_dict(orient="records")),
        "epsilon_robustness": json_safe(epsilon_summary.to_dict(orient="records")),
        "exported_files": json_safe(exported_files),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = normalize_config(args)
    layout = prepare_result_layout(cfg.output_dir)
    data_dir = layout.data_dir
    metrics_dir = layout.metrics_dir
    meta_dir = layout.meta_dir
    log_lines: list[str] = []

    device, device_message = resolve_device_with_fallback(cfg.device)
    log_and_print(log_lines, device_message)
    run_config_payload = json_safe(asdict(cfg))
    save_run_config(run_config_payload, layout.root)
    save_run_config(run_config_payload, meta_dir, filename="run_config.snapshot.json")

    seed_everything(int(cfg.seed))
    log_and_print(log_lines, f"[Setup] seed={cfg.seed}")
    log_and_print(log_lines, f"[Setup] loading dataset split={cfg.split} from {cfg.dataset_root}")
    dataset = load_mnist_dataset(cfg.dataset_root, cfg.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    class_index = build_class_index(dataset, num_classes=int(len(np.unique(labels))))
    trials, df_sequences = build_sequence_trials(labels, flat_normalized, class_index, cfg)
    sequences_csv = save_tidy_csv(
        df_sequences,
        data_dir / "sequences.csv",
        sort_by=["seq_len", "trial_id", "item_index"],
    )
    log_and_print(log_lines, f"[Data] generated {len(trials)} trials across seq_len={list(cfg.sequence_lengths)}")

    image_ids = [item_id for trial in trials for item_id in trial.ordered_item_ids]
    net, encoder = load_model_and_encoder(
        cfg.model_path,
        device=device,
        dt=cfg.dt,
        max_duration_ms=cfg.max_duration_ms,
    )
    baseline_gain = float(net.layer3.stsp_U)
    log_and_print(log_lines, f"[Model] loaded {cfg.model_path}")
    log_and_print(log_lines, f"[Model] layer3 baseline gain={baseline_gain:.6f}")

    spike_lookup = build_spike_lookup(images, encoder, image_ids, cfg, device)
    changed_trial_rows: list[dict[str, object]] = []
    rank_trial_rows: list[dict[str, object]] = []
    ping_trial_rows: list[dict[str, object]] = []
    epsilon_robustness_rows: list[dict[str, object]] = []
    example_payload: dict[str, np.ndarray] = {}

    for batch_counter, batch in enumerate(build_batches(trials, spike_lookup, cfg), start=1):
        log_and_print(log_lines, f"[Run] batch={batch_counter} seq_len={batch.seq_len} batch_size={len(batch.trials)}")
        rollout = run_sequence_and_capture_boundaries(net, batch, cfg)
        ping_by_stage = {
            int(snapshot.stage_k): run_neutral_ping_from_snapshot(net, cfg, snapshot, batch)
            for snapshot in rollout.stage_snapshots
        }
        batch_changed_rows, batch_rank_rows, batch_ping_rows = compute_trial_stage_rows(
            batch,
            rollout,
            baseline_gain=float(baseline_gain),
            epsilon=float(cfg.epsilon),
            null_permutations=int(cfg.null_permutations),
            ping_by_stage=ping_by_stage,
        )
        changed_trial_rows.extend(batch_changed_rows)
        rank_trial_rows.extend(batch_rank_rows)
        ping_trial_rows.extend(batch_ping_rows)
        epsilon_robustness_rows.extend(
            build_epsilon_robustness_rows(
                batch,
                rollout,
                ping_by_stage,
                baseline_gain=float(baseline_gain),
                epsilon_values=cfg.epsilon_sweep,
                null_permutations=int(cfg.null_permutations),
            )
        )
        if not example_payload:
            example_payload = build_example_state_payload(batch, rollout, max_trials=2)

    changed_value_cols = [
        "changed_synapse_fraction",
        "changed_synapse_count",
        "positive_changed_count",
        "negative_changed_count",
        "positive_change_mass",
        "positive_change_mass_ratio",
        "positive_change_mass_ratio_active",
        "positive_change_mass_ratio_top5",
        "positive_change_mass_ratio_changed",
        "baseline_nontrivial_active_fraction",
        "changed_positive_mass_in_top_1pct",
        "changed_positive_mass_in_top_5pct",
        "changed_positive_mass_in_top_10pct",
        "changed_positive_mass_in_top_5pct_ratio",
    ]
    rank_value_cols = [
        "active_synapse_count",
        "baseline_nontrivial_active_fraction",
        "changed_rank_percentile_mean",
        "changed_rank_percentile_median",
        "changed_rank_percentile_std",
        "negative_changed_rank_percentile_mean",
        "changed_top_1pct_occupancy",
        "changed_top_5pct_occupancy",
        "changed_top_10pct_occupancy",
        "changed_top_1pct_enrichment",
        "changed_top_5pct_enrichment",
        "changed_top_10pct_enrichment",
        "changed_top_1pct_zscore",
        "changed_top_5pct_zscore",
        "changed_top_10pct_zscore",
        "changed_positive_mass_in_top_1pct",
        "changed_positive_mass_in_top_5pct",
        "changed_positive_mass_in_top_10pct",
        "old_top_5pct_suppression_mass",
        "old_top_5pct_suppression_ratio",
        "net_top_5pct_reweight_score",
    ]
    ping_value_cols = [
        "positive_change_mass_ratio_active",
        "changed_positive_mass_in_top_5pct_ratio",
        "changed_rank_percentile_mean",
        "changed_top_5pct_occupancy",
        "changed_top_5pct_enrichment",
        "changed_top_5pct_zscore",
        "changed_topness_default",
        "com_sim",
        "sim_effective_count",
        "similarity_top1_index",
        "similarity_top1_mass",
        "ping_first_item_index",
        "ping_latest_item_hit",
        "ping_latest_item_hit_raw",
        "ping_latest_item_hit_chance_corrected",
        "ping_normalized_recency",
        "ping_recent_window_hit",
        "ping_lag_from_latest",
        "ping_seen_item_hit",
        "ping_nonmember_first",
        "ping_silent",
        "ping_first_fire_pred",
        "ping_first_fire_t",
        "ping_com",
    ]
    df_changed = summarize_trial_table(
        changed_trial_rows,
        group_cols=["seq_len", "stage_k"],
        value_cols=changed_value_cols,
    )
    df_rank = summarize_trial_table(
        rank_trial_rows,
        group_cols=["seq_len", "stage_k"],
        value_cols=rank_value_cols,
    )
    df_ping = summarize_trial_table(
        ping_trial_rows,
        group_cols=["seq_len", "stage_k"],
        value_cols=ping_value_cols,
    )
    df_ping = add_anchor_shift_columns(df_ping)
    df_epsilon = summarize_epsilon_robustness(pd.DataFrame(epsilon_robustness_rows))

    ping_trials = df_ping[df_ping["record_type"] == "trial_level"].copy()
    topness_cols = [
        "changed_top_5pct_enrichment",
        "changed_rank_percentile_mean",
        "positive_change_mass_ratio_active",
        "changed_positive_mass_in_top_5pct_ratio",
    ]
    outcome_cols = [
        "ping_latest_item_hit_raw",
        "ping_normalized_recency",
        "ping_latest_item_hit_chance_corrected",
        "ping_recent_window_hit",
        "stage_to_stage_anchor_shift",
    ]
    raw_global_corr_df = pd.DataFrame(
        build_metric_correlation_rows(
            ping_trials,
            group_cols=None,
            topness_cols=topness_cols,
            outcome_cols=outcome_cols,
            record_type="raw_global_summary",
        )
    )
    stage_matched_corr_df = pd.DataFrame(
        build_metric_correlation_rows(
            ping_trials,
            group_cols=["seq_len", "stage_k"],
            topness_cols=topness_cols,
            outcome_cols=outcome_cols,
            record_type="stage_matched_summary",
        )
    )
    stage_controlled_corr_df = pd.DataFrame(
        build_stage_controlled_correlation_rows(
            ping_trials,
            topness_cols=topness_cols,
            outcome_cols=outcome_cols,
        )
    )
    anchor_state_df = df_ping[
        [
            column
            for column in df_ping.columns
            if column in {
                "record_type",
                "trial_id",
                "seq_len",
                "stage_k",
                "com_sim",
                "sim_effective_count",
                "similarity_top1_index",
                "similarity_top1_mass",
                "stage_to_stage_anchor_shift",
            }
            or column.startswith("similarity_item_")
            or column.startswith("similarity_weight_")
        ]
    ].copy()

    changed_csv = save_tidy_csv(
        df_changed,
        metrics_dir / "layer3_changed_synapse_metrics.csv",
        sort_by=["record_type", "seq_len", "trial_id", "stage_k"],
    )
    rank_csv = save_tidy_csv(
        df_rank,
        metrics_dir / "layer3_changed_rank_metrics.csv",
        sort_by=["record_type", "seq_len", "trial_id", "stage_k"],
    )
    ping_csv = save_tidy_csv(
        df_ping,
        metrics_dir / "layer3_ping_coupling_metrics.csv",
        sort_by=["record_type", "seq_len", "trial_id", "stage_k"],
    )
    anchor_csv = save_tidy_csv(
        anchor_state_df,
        metrics_dir / "layer3_state_anchor_metrics.csv",
        sort_by=["record_type", "seq_len", "trial_id", "stage_k"],
    )
    raw_global_corr_csv = save_tidy_csv(
        raw_global_corr_df,
        metrics_dir / "raw_global_correlations.csv",
        sort_by=["topness_metric", "outcome_metric"],
    )
    stage_matched_corr_csv = save_tidy_csv(
        stage_matched_corr_df,
        metrics_dir / "stage_matched_correlations.csv",
        sort_by=["seq_len", "stage_k", "topness_metric", "outcome_metric"],
    )
    stage_controlled_corr_csv = save_tidy_csv(
        stage_controlled_corr_df,
        metrics_dir / "stage_controlled_correlations.csv",
        sort_by=["topness_metric", "outcome_metric"],
    )
    epsilon_csv = save_tidy_csv(
        df_epsilon,
        metrics_dir / "epsilon_robustness_summary.csv",
        sort_by=["record_type", "epsilon", "trial_id", "stage_k"],
    )
    example_npz_path = data_dir / "example_layer3_states.npz"
    np.savez_compressed(example_npz_path, **example_payload)

    figure_paths: dict[str, object] = {}
    write_plot_bundle_manifest(meta_dir)
    if not cfg.skip_figures:
        figure_paths = render_plot_only_panels(
            df_changed=df_changed,
            df_rank=df_rank,
            df_ping=df_ping,
            figures_dir=layout.figure_dir,
        )

    exported_files = {
        "sequences_csv": str(sequences_csv),
        "layer3_changed_synapse_metrics_csv": str(changed_csv),
        "layer3_changed_rank_metrics_csv": str(rank_csv),
        "layer3_ping_coupling_metrics_csv": str(ping_csv),
        "layer3_state_anchor_metrics_csv": str(anchor_csv),
        "raw_global_correlations_csv": str(raw_global_corr_csv),
        "stage_matched_correlations_csv": str(stage_matched_corr_csv),
        "stage_controlled_correlations_csv": str(stage_controlled_corr_csv),
        "epsilon_robustness_summary_csv": str(epsilon_csv),
        "example_layer3_states_npz": str(example_npz_path),
        "figure_paths": figure_paths,
    }
    summary_payload = build_summary_payload(
            cfg,
            device_requested=cfg.device,
            device_resolved=device.type,
            baseline_gain=float(baseline_gain),
            sequence_count=len(trials),
            df_changed=df_changed,
            df_rank=df_rank,
            df_ping=df_ping,
            df_raw_global_corr=raw_global_corr_df,
            df_stage_matched_corr=stage_matched_corr_df,
            df_stage_controlled_corr=stage_controlled_corr_df,
            df_epsilon=df_epsilon,
            exported_files=exported_files,
        )
    summary_path = save_summary_json(summary_payload, layout.root)
    save_summary_json(summary_payload, metrics_dir, filename="summary.json")
    save_run_config(
        {
            "experiment_name": "chunk_stsp_layer3_anchor_drift_mechanism",
            "sequence_count": int(len(trials)),
            "layer3_changed_rank_metrics_csv": str(rank_csv),
            "layer3_ping_coupling_metrics_csv": str(ping_csv),
            "layer3_state_anchor_metrics_csv": str(anchor_csv),
            "epsilon_robustness_summary_csv": str(epsilon_csv),
        },
        metrics_dir,
        filename="main_metrics.json",
    )
    # TODO: example_layer3_states.npz remains under data/ because it is a replay/debug snapshot rather than a compact metric.
    log_path = save_log_lines(log_lines, layout.log_dir)
    log_and_print(log_lines, f"[Done] summary={summary_path}")
    log_and_print(log_lines, f"[Done] log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
