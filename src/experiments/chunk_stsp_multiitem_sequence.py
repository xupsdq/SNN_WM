from __future__ import annotations

import argparse
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
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_run_config, save_summary_json
from src.experiments.common.runtime import seed_everything
from src.experiments.common.seed import mix_seed
from src.plotting.common.io import apply_publication_style, save_figure_all_formats, save_tidy_csv
from src.plotting.experiments.chunk_stsp_multiitem_sequence_plot import (
    _plot_anchor_position_vs_stage,
    _plot_item_similarity_heatmap,
    _plot_ping_retrieval_profile,
    _plot_stepwise_update_ratio,
)

LAYER_KEYS: tuple[str, ...] = ("layer1", "layer2", "layer3")


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
    cluster_sim_threshold: float
    update_distance_metric: str
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
        return max(self.sample_ms, self.ping_ms, 100.0)


@dataclass(frozen=True)
class SequenceTrial:
    trial_id: int
    seq_len: int
    ordered_item_ids: tuple[int, ...]
    ordered_item_labels: tuple[int, ...]
    mean_pairwise_image_similarity: float
    max_pairwise_image_similarity: float
    min_pairwise_image_similarity: float
    sequence_seed: int


@dataclass(frozen=True)
class SequenceBatch:
    batch_id: int
    seq_len: int
    trials: tuple[SequenceTrial, ...]
    item_spikes: tuple[torch.Tensor, ...]


@dataclass
class SnapshotBatch:
    stage_k: int
    layer_input_shapes: dict[str, tuple[int, ...]]
    restore_ux_by_layer: dict[str, tuple[torch.Tensor, torch.Tensor]]
    state_by_layer: dict[str, dict[str, np.ndarray]]


@dataclass
class PingBatchReadout:
    first_fire_pred: np.ndarray
    first_fire_t: np.ndarray
    silent_mask: np.ndarray
    predicted_item_index: np.ndarray
    predicted_item_label: np.ndarray
    hit_any_seen_item: np.ndarray
    unseen_label_rate: np.ndarray
    ping_weight_matrix: np.ndarray


def ms_to_steps(duration_ms: float, dt: float) -> int:
    return int(round((float(duration_ms) * ms) / float(dt)))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-input STSP chunk compression and ping-readout experiment.",
    )
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_PATH_CONFIG.model_path))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_PATH_CONFIG.results_root / "chunk_stsp_multiitem_sequence"),
    )
    parser.add_argument("--sample-ms", type=float, default=180.0)
    parser.add_argument("--delay-ms", type=float, default=200.0)
    parser.add_argument("--ping-ms", type=float, default=30.0)
    parser.add_argument("--ping-amp", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[3, 5, 7, 10])
    parser.add_argument("--samples-per-label", type=int, default=200)
    parser.add_argument("--cluster-sim-threshold", type=float, default=0.65)
    parser.add_argument(
        "--update-distance-metric",
        type=str,
        default="centered_cosine",
        choices=["centered_cosine", "euclidean"],
    )
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


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
        cluster_sim_threshold=float(args.cluster_sim_threshold),
        update_distance_metric=str(args.update_distance_metric),
        skip_figures=bool(args.skip_figures),
        smoke=bool(args.smoke),
    )
    if cfg.smoke:
        cfg = ExperimentConfig(
            **{
                **asdict(cfg),
                "batch_size": min(int(cfg.batch_size), 2),
                "max_sequences": min(int(cfg.max_sequences), 4),
                "sequence_lengths": (3,),
                "samples_per_label": min(int(cfg.samples_per_label), 8),
                "sample_ms": min(float(cfg.sample_ms), 30.0),
                "delay_ms": min(float(cfg.delay_ms), 30.0),
                "ping_ms": min(float(cfg.ping_ms), 15.0),
            }
        )
    if min(cfg.sample_steps, cfg.ping_steps) <= 0:
        raise ValueError("sample-ms and ping-ms must map to at least one step.")
    if cfg.delay_steps < 0:
        raise ValueError("delay-ms must be non-negative.")
    if cfg.max_sequences <= 0 or cfg.batch_size <= 0:
        raise ValueError("max-sequences and batch-size must be positive.")
    if not (0.0 <= cfg.cluster_sim_threshold <= 1.0):
        raise ValueError("--cluster-sim-threshold must be in [0, 1].")
    return cfg


def resolve_device_with_fallback(device_arg: str) -> tuple[torch.device, str]:
    raw = str(device_arg).strip().lower()
    if raw in ("auto", ""):
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
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return safe_float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def centered_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    a_centered = a_arr - a_arr.mean(axis=1, keepdims=True)
    b_centered = b_arr - b_arr.mean(axis=1, keepdims=True)
    numerator = np.sum(a_centered * b_centered, axis=1)
    denom = np.linalg.norm(a_centered, axis=1) * np.linalg.norm(b_centered, axis=1)
    return numerator / np.maximum(denom, float(eps))


def raw_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    numerator = np.sum(a_arr * b_arr, axis=1)
    denom = np.linalg.norm(a_arr, axis=1) * np.linalg.norm(b_arr, axis=1)
    return numerator / np.maximum(denom, float(eps))


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
            "sim_entropy": 0.0,
            "sim_effective_count": 0.0,
            "similarity_top1_index": None,
            "similarity_top1_mass": 0.0,
        }
    positions = np.arange(1, len(weights_arr) + 1, dtype=np.float64)
    positive = weights_arr[weights_arr > 0.0]
    return {
        "com_sim": float(np.sum(positions * weights_arr)),
        "sim_entropy": float(-np.sum(positive * np.log(np.maximum(positive, 1e-12)))),
        "sim_effective_count": float(1.0 / np.sum(np.square(weights_arr))),
        "similarity_top1_index": int(np.argmax(weights_arr) + 1),
        "similarity_top1_mass": float(np.max(weights_arr)),
    }


def compute_ping_profile_summary(weights: np.ndarray) -> dict[str, float | int | None]:
    weights_arr = np.asarray(weights, dtype=np.float64)
    if weights_arr.size == 0 or float(weights_arr.sum()) <= 0.0:
        return {"ping_com": None, "ping_top1_index": None, "ping_top1_mass": 0.0}
    positions = np.arange(1, len(weights_arr) + 1, dtype=np.float64)
    return {
        "ping_com": float(np.sum(positions * weights_arr)),
        "ping_top1_index": int(np.argmax(weights_arr) + 1),
        "ping_top1_mass": float(np.max(weights_arr)),
    }


def distance_between_rows(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    if metric == "centered_cosine":
        return 1.0 - centered_cosine_similarity(a, b)
    if metric == "euclidean":
        return np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), axis=1)
    raise ValueError(f"Unsupported distance metric: {metric}")


def build_label_candidate_pools(
    class_index: Mapping[int, Sequence[int]],
    cfg: ExperimentConfig,
) -> dict[int, np.ndarray]:
    pools: dict[int, np.ndarray] = {}
    for label in sorted(class_index):
        rng = np.random.default_rng(mix_seed(cfg.seed, int(label), 991))
        ids = np.asarray([int(idx) for idx in class_index[int(label)]], dtype=np.int64)
        permuted = rng.permutation(ids)
        limit = min(int(cfg.samples_per_label), len(permuted)) if cfg.samples_per_label > 0 else len(permuted)
        pool = permuted[:limit].astype(np.int64, copy=False)
        if pool.size <= 0:
            raise ValueError(f"Label {label} has no available samples after pooling.")
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
        for trial_index_within_len in range(int(cfg.max_sequences)):
            trial_seed = mix_seed(cfg.seed, int(seq_len), int(trial_index_within_len), 17)
            rng = np.random.default_rng(trial_seed)
            chosen_labels = rng.choice(all_labels, size=int(seq_len), replace=False)
            chosen_item_ids = np.asarray(
                [int(rng.choice(label_pools[int(label)])) for label in chosen_labels],
                dtype=np.int64,
            )
            order = rng.permutation(int(seq_len))
            ordered_labels = chosen_labels[order].astype(np.int64, copy=False)
            ordered_ids = chosen_item_ids[order].astype(np.int64, copy=False)
            sub_sim = flat_normalized[ordered_ids] @ flat_normalized[ordered_ids].T
            if int(seq_len) > 1:
                mask = ~np.eye(int(seq_len), dtype=bool)
                pairwise_values = sub_sim[mask]
                mean_pair = float(pairwise_values.mean())
                max_pair = float(pairwise_values.max())
                min_pair = float(pairwise_values.min())
            else:
                mean_pair = 1.0
                max_pair = 1.0
                min_pair = 1.0
            trial = SequenceTrial(
                trial_id=int(trial_id),
                seq_len=int(seq_len),
                ordered_item_ids=tuple(int(item) for item in ordered_ids.tolist()),
                ordered_item_labels=tuple(int(label) for label in ordered_labels.tolist()),
                mean_pairwise_image_similarity=mean_pair,
                max_pairwise_image_similarity=max_pair,
                min_pairwise_image_similarity=min_pair,
                sequence_seed=int(trial_seed),
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
                        "trial_index_within_seq_len": int(trial_index_within_len),
                        "seq_len": int(trial.seq_len),
                        "item_index": int(item_index),
                        "image_id": int(image_id),
                        "item_label": int(item_label),
                        "ordered_item_ids": ordered_ids_str,
                        "ordered_item_labels": ordered_labels_str,
                        "mean_pairwise_image_similarity": float(mean_pair),
                        "max_pairwise_image_similarity": float(max_pair),
                        "min_pairwise_image_similarity": float(min_pair),
                        "sequence_seed": int(trial_seed),
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


def snapshot_stsp_state_batch(
    net: Any,
    *,
    batch_size: int,
    layer_input_shapes: Mapping[str, tuple[int, ...]],
) -> SnapshotBatch:
    state_by_layer: dict[str, dict[str, np.ndarray]] = {}
    restore_ux_by_layer: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key)
        if getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
            raise ValueError(f"{layer_key} is missing STSP state at the requested snapshot boundary.")
        u_pre = layer.u_pre.detach().view(batch_size, -1)
        x_pre = layer.x_pre.detach().view(batch_size, -1)
        g_pre = (layer.u_pre * layer.x_pre).detach().view(batch_size, -1)
        state_by_layer[str(layer_key)] = {
            "u_pre": u_pre.cpu().numpy().astype(np.float32, copy=False),
            "x_pre": x_pre.cpu().numpy().astype(np.float32, copy=False),
            "g_pre": g_pre.cpu().numpy().astype(np.float32, copy=False),
        }
        restore_ux_by_layer[str(layer_key)] = (
            layer.u_pre.detach().cpu().clone(),
            layer.x_pre.detach().cpu().clone(),
        )
    return SnapshotBatch(
        stage_k=-1,
        layer_input_shapes={str(key): tuple(value) for key, value in layer_input_shapes.items()},
        restore_ux_by_layer=restore_ux_by_layer,
        state_by_layer=state_by_layer,
    )


def run_sequence_stage_snapshots(
    net: Any,
    batch: SequenceBatch,
    cfg: ExperimentConfig,
    *,
    active_positions: set[int] | None = None,
    stop_stage: int | None = None,
) -> list[SnapshotBatch]:
    seq_len = int(batch.seq_len)
    final_stage = seq_len if stop_stage is None else int(stop_stage)
    if final_stage < 1 or final_stage > seq_len:
        raise ValueError("stop_stage must be in [1, seq_len].")
    first_sequence = batch.item_spikes[0]
    batch_size, _, channels, height, width = first_sequence.shape
    snapshots: list[SnapshotBatch] = []
    active = set(range(1, seq_len + 1)) if active_positions is None else {int(pos) for pos in active_positions}
    with torch.no_grad():
        prepare_network_state(net, batch_size, channels, height, width)
        layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)
        zero_input = torch.zeros(
            (batch_size, channels, height, width),
            dtype=first_sequence.dtype,
            device=first_sequence.device,
        )
        current_time = 0
        for position in range(1, final_stage + 1):
            sequence = batch.item_spikes[position - 1]
            for step_index in range(int(sequence.shape[1])):
                input_t = sequence[:, step_index, ...] if position in active else zero_input
                forward_three_layers(net, input_t, current_time)
                current_time += 1
            for _ in range(int(cfg.delay_steps)):
                forward_three_layers(net, zero_input, current_time)
                current_time += 1
            snapshot = snapshot_stsp_state_batch(
                net,
                batch_size=batch_size,
                layer_input_shapes=layer_input_shapes,
            )
            snapshot.stage_k = int(position)
            snapshots.append(snapshot)
    return snapshots


def build_lag_matched_singleton_reference_bank(
    net: Any,
    batch: SequenceBatch,
    cfg: ExperimentConfig,
) -> dict[int, dict[int, SnapshotBatch]]:
    reference_bank: dict[int, dict[int, SnapshotBatch]] = {}
    for item_index in range(1, int(batch.seq_len) + 1):
        item_snapshots = run_sequence_stage_snapshots(
            net,
            batch,
            cfg,
            active_positions={int(item_index)},
            stop_stage=int(batch.seq_len),
        )
        for snapshot in item_snapshots:
            if item_index > int(snapshot.stage_k):
                continue
            reference_bank.setdefault(int(snapshot.stage_k), {})[int(item_index)] = snapshot
    return reference_bank


def run_neutral_ping_from_snapshot(
    net: Any,
    cfg: ExperimentConfig,
    snapshot: SnapshotBatch,
    *,
    batch: SequenceBatch,
) -> PingBatchReadout:
    batch_size = len(batch.trials)
    label_matrix = np.asarray([trial.ordered_item_labels for trial in batch.trials], dtype=np.int64)
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
    hit_any_seen_item = np.zeros(batch_size, dtype=np.int64)
    unseen_label_rate = np.zeros(batch_size, dtype=np.int64)
    ping_weight_matrix = np.zeros((batch_size, int(batch.seq_len)), dtype=np.float64)
    for row_idx in range(batch_size):
        pred_label = int(first_fire_pred[row_idx])
        if pred_label < 0:
            continue
        predicted_item_label[row_idx] = pred_label
        matches = np.where(label_matrix[row_idx] == pred_label)[0]
        if matches.size > 0:
            predicted_item_index[row_idx] = int(matches[0] + 1)
            hit_any_seen_item[row_idx] = 1
            ping_weight_matrix[row_idx, int(matches[0])] = 1.0
        else:
            unseen_label_rate[row_idx] = 1
    return PingBatchReadout(
        first_fire_pred=first_fire_pred,
        first_fire_t=first_fire_t,
        silent_mask=silent_mask,
        predicted_item_index=predicted_item_index,
        predicted_item_label=predicted_item_label,
        hit_any_seen_item=hit_any_seen_item,
        unseen_label_rate=unseen_label_rate,
        ping_weight_matrix=ping_weight_matrix,
    )


def compute_batch_metrics(
    batch: SequenceBatch,
    actual_snapshots: Sequence[SnapshotBatch],
    reference_bank: Mapping[int, Mapping[int, SnapshotBatch]],
    ping_by_stage: Mapping[int, PingBatchReadout],
    cfg: ExperimentConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    item_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []
    for snapshot in actual_snapshots:
        stage_k = int(snapshot.stage_k)
        ping = ping_by_stage[stage_k]
        for batch_row, trial in enumerate(batch.trials):
            stage_labels = np.asarray(trial.ordered_item_labels[:stage_k], dtype=np.int64)
            ping_weights = ping.ping_weight_matrix[batch_row, :stage_k].astype(np.float64, copy=False)
            ping_summary = compute_ping_profile_summary(ping_weights)
            for layer_key in LAYER_KEYS:
                centered_scores: list[float] = []
                raw_scores: list[float] = []
                for item_index in range(1, stage_k + 1):
                    actual_state = np.asarray(
                        snapshot.state_by_layer[layer_key]["g_pre"][batch_row : batch_row + 1],
                        dtype=np.float64,
                    )
                    reference_state = np.asarray(
                        reference_bank[stage_k][item_index].state_by_layer[layer_key]["g_pre"][batch_row : batch_row + 1],
                        dtype=np.float64,
                    )
                    centered_scores.append(float(centered_cosine_similarity(actual_state, reference_state)[0]))
                    raw_scores.append(float(raw_cosine_similarity(actual_state, reference_state)[0]))
                sim_weights, has_positive_similarity = normalize_nonnegative_weights(centered_scores)
                sim_summary = compute_similarity_profile_summary(sim_weights)
                for item_index in range(1, stage_k + 1):
                    item_rows.append(
                        {
                            "trial_id": int(trial.trial_id),
                            "seq_len": int(trial.seq_len),
                            "stage_k": int(stage_k),
                            "layer": str(layer_key),
                            "item_index": int(item_index),
                            "item_label": int(stage_labels[item_index - 1]),
                            "similarity_centered": float(centered_scores[item_index - 1]),
                            "similarity_raw": float(raw_scores[item_index - 1]),
                            "similarity_weight_nonnegative": float(sim_weights[item_index - 1]),
                            "has_positive_similarity": int(has_positive_similarity),
                        }
                    )
                summary_rows.append(
                    {
                        "trial_id": int(trial.trial_id),
                        "seq_len": int(trial.seq_len),
                        "stage_k": int(stage_k),
                        "layer": str(layer_key),
                        "com_sim": sim_summary["com_sim"],
                        "sim_entropy": float(sim_summary["sim_entropy"]),
                        "sim_effective_count": float(sim_summary["sim_effective_count"]),
                        "similarity_top1_index": sim_summary["similarity_top1_index"],
                        "similarity_top1_mass": float(sim_summary["similarity_top1_mass"]),
                        "has_positive_similarity": int(has_positive_similarity),
                        "ping_com": ping_summary["ping_com"],
                        "ping_top1_index": ping_summary["ping_top1_index"],
                        "ping_top1_mass": float(ping_summary["ping_top1_mass"]),
                        "ping_seen_item_hit_rate": int(ping.hit_any_seen_item[batch_row]),
                        "unseen_label_rate": int(ping.unseen_label_rate[batch_row]),
                    }
                )
                if stage_k >= 2:
                    previous_snapshot = actual_snapshots[stage_k - 2]
                    newonly_snapshot = reference_bank[stage_k][stage_k]
                    prev_state = np.asarray(previous_snapshot.state_by_layer[layer_key]["g_pre"][batch_row : batch_row + 1], dtype=np.float64)
                    current_state = np.asarray(snapshot.state_by_layer[layer_key]["g_pre"][batch_row : batch_row + 1], dtype=np.float64)
                    newonly_state = np.asarray(newonly_snapshot.state_by_layer[layer_key]["g_pre"][batch_row : batch_row + 1], dtype=np.float64)
                    update_distance_real = float(distance_between_rows(current_state, prev_state, cfg.update_distance_metric)[0])
                    update_distance_newonly = float(distance_between_rows(newonly_state, prev_state, cfg.update_distance_metric)[0])
                    update_rows.append(
                        {
                            "trial_id": int(trial.trial_id),
                            "seq_len": int(trial.seq_len),
                            "stage_k": int(stage_k),
                            "layer": str(layer_key),
                            "stepwise_update_ratio": float(update_distance_real / max(update_distance_newonly, 1e-12)),
                            "update_distance_real": float(update_distance_real),
                            "update_distance_newonly": float(update_distance_newonly),
                            "update_distance_metric": str(cfg.update_distance_metric),
                        }
                    )
    return item_rows, summary_rows, update_rows


def build_ping_rows(
    batch: SequenceBatch,
    ping_by_stage: Mapping[int, PingBatchReadout],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage_k, ping in ping_by_stage.items():
        for batch_row, trial in enumerate(batch.trials):
            for item_index in range(1, int(stage_k) + 1):
                rows.append(
                    {
                        "trial_id": int(trial.trial_id),
                        "seq_len": int(trial.seq_len),
                        "stage_k": int(stage_k),
                        "layer": "layer3",
                        "item_index": int(item_index),
                        "item_label": int(trial.ordered_item_labels[item_index - 1]),
                        "ping_weight": float(ping.ping_weight_matrix[batch_row, item_index - 1]),
                        "first_fire_pred": int(ping.first_fire_pred[batch_row]),
                        "first_fire_t": int(ping.first_fire_t[batch_row]),
                        "silent": int(ping.silent_mask[batch_row]),
                        "predicted_item_index": None if int(ping.predicted_item_index[batch_row]) < 0 else int(ping.predicted_item_index[batch_row]),
                        "predicted_item_label": None if int(ping.predicted_item_label[batch_row]) < 0 else int(ping.predicted_item_label[batch_row]),
                        "hit_any_seen_item": int(ping.hit_any_seen_item[batch_row]),
                        "hit_item_i": int(ping.ping_weight_matrix[batch_row, item_index - 1] > 0.0),
                        "unseen_label_rate": int(ping.unseen_label_rate[batch_row]),
                    }
                )
    return rows


def generate_figures(
    item_similarity_df: pd.DataFrame,
    similarity_summary_df: pd.DataFrame,
    ping_df: pd.DataFrame,
    update_df: pd.DataFrame,
    *,
    figures_dir: Path,
) -> dict[str, dict[str, str]]:
    apply_publication_style()
    figure_paths: dict[str, dict[str, str]] = {}
    figures = {
        "anchor_position_vs_stage": _plot_anchor_position_vs_stage(similarity_summary_df),
        "item_similarity_heatmap": _plot_item_similarity_heatmap(item_similarity_df),
        "ping_retrieval_profile": _plot_ping_retrieval_profile(ping_df),
        "stepwise_update_ratio": _plot_stepwise_update_ratio(update_df),
    }
    for stem, fig in figures.items():
        figure_paths[stem] = save_figure_all_formats(fig, figures_dir / stem)
        plt.close(fig)
    return figure_paths


def build_summary(
    similarity_summary_df: pd.DataFrame,
    ping_df: pd.DataFrame,
    update_df: pd.DataFrame,
    exported_files: Mapping[str, Any],
    cfg: ExperimentConfig,
) -> dict[str, object]:
    ping_trial_df = ping_df.drop_duplicates(subset=["trial_id", "seq_len", "stage_k"]).copy()
    final_similarity = similarity_summary_df[similarity_summary_df["stage_k"] == similarity_summary_df["seq_len"]].copy()
    final_ping = ping_trial_df[ping_trial_df["stage_k"] == ping_trial_df["seq_len"]].copy()
    seq_len_summary: dict[str, object] = {}
    for seq_len in sorted(final_similarity["seq_len"].unique().tolist()):
        seq_similarity = final_similarity[final_similarity["seq_len"] == int(seq_len)].copy()
        seq_ping = final_ping[final_ping["seq_len"] == int(seq_len)].copy()
        layer_summary: dict[str, object] = {}
        for layer_key, sub in seq_similarity.groupby("layer", sort=True):
            top_counts = sub["similarity_top1_index"].dropna().astype(int).value_counts(normalize=True, sort=False).sort_index().to_dict()
            layer_summary[str(layer_key)] = {
                "average_com_sim": safe_float(sub["com_sim"].mean()),
                "average_sim_entropy": safe_float(sub["sim_entropy"].mean()),
                "average_sim_effective_count": safe_float(sub["sim_effective_count"].mean()),
                "average_similarity_top1_mass": safe_float(sub["similarity_top1_mass"].mean()),
                "top_position_distribution": {str(int(key)): float(value) for key, value in top_counts.items()},
            }
        ping_top_counts = seq_ping["predicted_item_index"].dropna().astype(int).value_counts(normalize=True, sort=False).sort_index().to_dict()
        seq_update = update_df[(update_df["seq_len"] == int(seq_len)) & (update_df["stage_k"] == int(seq_len))].copy()
        seq_len_summary[str(int(seq_len))] = {
            "final_stage": int(seq_len),
            "layers": layer_summary,
            "ping_seen_item_hit_rate": safe_float(seq_ping["hit_any_seen_item"].mean()) if not seq_ping.empty else None,
            "unseen_label_rate": safe_float(seq_ping["unseen_label_rate"].mean()) if not seq_ping.empty else None,
            "ping_top_position_distribution": {str(int(key)): float(value) for key, value in ping_top_counts.items()},
            "ping_average_com": safe_float(seq_ping["predicted_item_index"].dropna().mean()) if not seq_ping.empty else None,
            "stepwise_update_ratio_mean": safe_float(seq_update["stepwise_update_ratio"].mean()) if not seq_update.empty else None,
        }
    return {
        "experiment_name": "multiitem_stsp_chunk_compression_ping_readout",
        "scientific_focus": [
            "compression",
            "asymmetry",
            "anchor_drift",
            "ping_readable_latent_state",
        ],
        "config": json_safe(asdict(cfg)),
        "sequence_length_summary": seq_len_summary,
        "exported_files": json_safe(exported_files),
    }


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    cfg = normalize_config(args)
    layout = prepare_result_layout(cfg.output_dir)
    log_lines: list[str] = []

    log_and_print(log_lines, "[Stage] Starting multi-input STSP chunk compression experiment.")
    seed_everything(int(cfg.seed))
    device, device_message = resolve_device_with_fallback(cfg.device)
    log_and_print(log_lines, device_message)
    log_and_print(
        log_lines,
        f"[Config] sample_steps={cfg.sample_steps}, delay_steps={cfg.delay_steps}, ping_steps={cfg.ping_steps}, "
        f"sequence_lengths={list(cfg.sequence_lengths)}, max_sequences_per_len={cfg.max_sequences}, batch_size={cfg.batch_size}",
    )

    dataset = load_mnist_dataset(cfg.dataset_root, cfg.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    class_index = build_class_index(dataset, num_classes=int(len(np.unique(labels))))
    trials, df_sequences = build_sequence_trials(labels, flat_normalized, class_index, cfg)
    if not trials:
        raise RuntimeError("No sequences were generated.")
    log_and_print(log_lines, f"[Data] Generated {len(trials)} trials across seq_len={list(cfg.sequence_lengths)}.")

    net, encoder = load_model_and_encoder(
        cfg.model_path,
        device=device,
        dt=cfg.dt,
        max_duration_ms=cfg.max_duration_ms,
    )
    spike_lookup = build_spike_lookup(images, encoder, [image_id for trial in trials for image_id in trial.ordered_item_ids], cfg, device)
    log_and_print(log_lines, f"[Data] Encoded {len(spike_lookup)} unique item images.")

    item_similarity_rows: list[dict[str, object]] = []
    similarity_summary_rows: list[dict[str, object]] = []
    ping_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []
    for batch in build_batches(trials, spike_lookup, cfg):
        log_and_print(log_lines, f"[Batch] seq_len={batch.seq_len} batch_id={batch.batch_id} batch_size={len(batch.trials)}")
        actual_snapshots = run_sequence_stage_snapshots(net, batch, cfg)
        reference_bank = build_lag_matched_singleton_reference_bank(net, batch, cfg)
        ping_by_stage = {int(snapshot.stage_k): run_neutral_ping_from_snapshot(net, cfg, snapshot, batch=batch) for snapshot in actual_snapshots}
        item_rows, summary_rows, update_batch_rows = compute_batch_metrics(
            batch,
            actual_snapshots,
            reference_bank,
            ping_by_stage,
            cfg,
        )
        item_similarity_rows.extend(item_rows)
        similarity_summary_rows.extend(summary_rows)
        ping_rows.extend(build_ping_rows(batch, ping_by_stage))
        update_rows.extend(update_batch_rows)

    item_similarity_df = pd.DataFrame(item_similarity_rows)
    similarity_summary_df = pd.DataFrame(similarity_summary_rows)
    ping_df = pd.DataFrame(ping_rows)
    update_df = pd.DataFrame(update_rows)

    item_similarity_csv = save_tidy_csv(item_similarity_df, layout.data_file("item_similarity_metrics.csv"), sort_by=["seq_len", "trial_id", "stage_k", "layer", "item_index"])
    similarity_summary_csv = save_tidy_csv(similarity_summary_df, layout.data_file("similarity_summary_metrics.csv"), sort_by=["seq_len", "trial_id", "stage_k", "layer"])
    ping_csv = save_tidy_csv(ping_df, layout.data_file("ping_retrieval_metrics.csv"), sort_by=["seq_len", "trial_id", "stage_k", "item_index"])
    update_csv = save_tidy_csv(update_df, layout.data_file("stepwise_update_metrics.csv"), sort_by=["seq_len", "trial_id", "stage_k", "layer"])

    figure_paths: dict[str, Any] = {}
    if not cfg.skip_figures:
        figure_paths = generate_figures(
            item_similarity_df,
            similarity_summary_df,
            ping_df,
            update_df,
            figures_dir=layout.figure_dir,
        )
        log_and_print(log_lines, f"[Output] Generated {len(figure_paths)} figure groups.")
    else:
        log_and_print(log_lines, "[Output] Figure generation skipped.")

    exported_files = {
        "item_similarity_metrics_csv": item_similarity_csv,
        "similarity_summary_metrics_csv": similarity_summary_csv,
        "ping_retrieval_metrics_csv": ping_csv,
        "stepwise_update_metrics_csv": update_csv,
        "figures": figure_paths,
    }
    summary_path = save_summary_json(
        build_summary(similarity_summary_df, ping_df, update_df, exported_files, cfg),
        layout.root,
        filename="summary.json",
    )
    run_config_path = save_run_config(
        {
            **json_safe(asdict(cfg)),
            "resolved_device": str(device),
        },
        layout.root,
    )
    log_and_print(log_lines, f"[Output] summary.json -> {summary_path}")
    log_and_print(log_lines, f"[Output] run_config.json -> {run_config_path}")
    log_path = save_log_lines(log_lines, layout.log_dir, filename="run.log")
    print(f"[Output] log -> {log_path}", flush=True)


if __name__ == "__main__":
    main()
