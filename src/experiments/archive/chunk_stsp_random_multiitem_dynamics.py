from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.paths import DEFAULT_PATH_CONFIG
from src.config.units import ms
from src.experiments.common.dataset import build_dataset_arrays, encode_images
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
from src.plotting.common.style import DYNAMIC_COLOR, NOISE_COLOR, SAMPLE_COLOR, SHUFFLE_COLOR


@dataclass(frozen=True)
class RandomMultiItemDynamicsConfig:
    model_path: str
    dataset_root: str
    split: str
    device: str
    seed: int
    output_dir: str
    n_items_list: tuple[int, ...]
    timecourse_n: int
    num_trials_per_n: int
    sample_ms: float
    delay_ms: float
    final_delay_ms: float
    ping_ms: float
    ping_amp: float
    batch_size: int
    unique_threshold: float
    epsilon: float
    skip_figures: bool
    save_example_states: bool
    smoke: bool
    dt: float = 1.0 * ms

    @property
    def sample_steps(self) -> int:
        return ms_to_steps(self.sample_ms, self.dt)

    @property
    def delay_steps(self) -> int:
        return ms_to_steps(self.delay_ms, self.dt)

    @property
    def final_delay_steps(self) -> int:
        return ms_to_steps(self.final_delay_ms, self.dt)

    @property
    def ping_steps(self) -> int:
        return ms_to_steps(self.ping_ms, self.dt)

    @property
    def max_duration_ms(self) -> float:
        return max(self.sample_ms, self.ping_ms, 100.0)

    @property
    def max_n_items(self) -> int:
        return max(int(n_items) for n_items in self.n_items_list)


@dataclass
class MultiItemBatch:
    batch_id: int
    n_items: int
    df: pd.DataFrame
    item_spikes: tuple[torch.Tensor, ...]


@dataclass
class Layer3Snapshot:
    snapshot_name: str
    snapshot_index: int
    layer_input_shapes: Dict[str, tuple[int, ...]]
    restore_ux_by_layer: Dict[str, tuple[torch.Tensor, torch.Tensor]]
    state: Dict[str, np.ndarray]


@dataclass
class SequenceRollout:
    ordered_snapshots: tuple[Layer3Snapshot, ...]
    named_snapshots: Dict[str, Layer3Snapshot]


@dataclass
class PingReadout:
    first_fire_pred: np.ndarray
    first_fire_t: np.ndarray
    silent_mask: np.ndarray
    nonmember_first: np.ndarray
    ambiguous_member_first: np.ndarray
    target_first: Dict[int, np.ndarray]
    ping_winner: np.ndarray


def ms_to_steps(duration_ms: float, dt: float) -> int:
    return int(round((float(duration_ms) * ms) / float(dt)))


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
    if not values:
        raise ValueError("n-items-list must not be empty.")
    if any(value < 2 for value in values):
        raise ValueError("Each n in n-items-list must be at least 2.")
    return tuple(dict.fromkeys(values))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Random multi-item layer3-only STSP dynamics experiment.",
    )
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_PATH_CONFIG.model_path))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_PATH_CONFIG.results_root / "chunk_stsp_random_multiitem_dynamics"),
    )
    parser.add_argument("--n-items-list", type=str, default="3,4,5,6")
    parser.add_argument("--timecourse-n", type=int, default=5)
    parser.add_argument("--num-trials-per-n", type=int, default=24)
    parser.add_argument("--sample-ms", type=float, default=50.0)
    parser.add_argument("--delay-ms", type=float, default=50.0)
    parser.add_argument("--final-delay-ms", type=float, default=50.0)
    parser.add_argument("--ping-ms", type=float, default=30.0)
    parser.add_argument("--ping-amp", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--unique-threshold", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--save-example-states", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def normalize_config(args: argparse.Namespace) -> RandomMultiItemDynamicsConfig:
    cfg = RandomMultiItemDynamicsConfig(
        model_path=str(args.model_path),
        dataset_root=str(args.dataset_root),
        split=str(args.split),
        device=str(args.device),
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        n_items_list=parse_int_list(args.n_items_list),
        timecourse_n=int(args.timecourse_n),
        num_trials_per_n=int(args.num_trials_per_n),
        sample_ms=float(args.sample_ms),
        delay_ms=float(args.delay_ms),
        final_delay_ms=float(args.final_delay_ms),
        ping_ms=float(args.ping_ms),
        ping_amp=float(args.ping_amp),
        batch_size=int(args.batch_size),
        unique_threshold=float(args.unique_threshold),
        epsilon=float(args.epsilon),
        skip_figures=bool(args.skip_figures),
        save_example_states=bool(args.save_example_states),
        smoke=bool(args.smoke),
    )
    if cfg.timecourse_n not in cfg.n_items_list:
        raise ValueError("timecourse-n must be included in n-items-list.")
    if cfg.smoke:
        cfg = RandomMultiItemDynamicsConfig(
            **{
                **asdict(cfg),
                "num_trials_per_n": min(int(cfg.num_trials_per_n), 2),
                "batch_size": min(int(cfg.batch_size), 2),
                "save_example_states": True,
            }
        )
    if min(cfg.sample_steps, cfg.delay_steps, cfg.final_delay_steps, cfg.ping_steps) <= 0:
        raise ValueError("Sample, delay, final-delay, and ping durations must map to at least one step.")
    return cfg


def resolve_device_with_fallback(device_arg: str) -> tuple[torch.device, str]:
    device_str = str(device_arg).strip().lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu"), "[Runtime] CUDA unavailable on 2026-04-19; falling back to CPU."
    return torch.device(device_str), f"[Runtime] Using device={device_str}."


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
    if isinstance(value, (np.floating, float)):
        return safe_float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def _summary_key(value: object) -> str:
    if value is None or pd.isna(value):
        return "null"
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if np.isfinite(numeric) and float(numeric).is_integer():
            return str(int(round(numeric)))
        return str(numeric)
    return str(value)


def build_random_multiitem_sequences(
    labels: np.ndarray,
    cfg: RandomMultiItemDynamicsConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg.seed))
    max_n = int(cfg.max_n_items)
    all_ids = np.arange(len(labels), dtype=np.int64)
    rows: list[dict[str, object]] = []
    trial_id = 0
    for n_items in cfg.n_items_list:
        for within_n_trial in range(int(cfg.num_trials_per_n)):
            chosen_ids = rng.choice(all_ids, size=int(n_items), replace=False)
            row: dict[str, object] = {
                "trial_id": int(trial_id),
                "trial_index_within_n": int(within_n_trial),
                "n_items": int(n_items),
                "sequence_seed": int(mix_seed(cfg.seed, int(n_items), int(within_n_trial))),
            }
            for position in range(1, max_n + 1):
                id_key = f"item{position}_id"
                label_key = f"item{position}_label"
                if position <= int(n_items):
                    image_id = int(chosen_ids[position - 1])
                    row[id_key] = image_id
                    row[label_key] = int(labels[image_id])
                else:
                    row[id_key] = np.nan
                    row[label_key] = np.nan
            rows.append(row)
            trial_id += 1
    return pd.DataFrame(rows).sort_values(["n_items", "trial_index_within_n"], kind="stable").reset_index(drop=True)


def build_spike_lookup(
    images: torch.Tensor,
    encoder,
    ids: Iterable[int],
    *,
    steps: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    unique_ids = sorted({int(image_id) for image_id in ids})
    spike_bank = encode_images(
        encoder,
        images[unique_ids].to(device=device, dtype=torch.float32),
        steps=int(steps),
    )
    return {int(image_id): spike_bank[idx] for idx, image_id in enumerate(unique_ids)}


def prepare_multiitem_batches(
    df_sequences: pd.DataFrame,
    images: torch.Tensor,
    encoder,
    cfg: RandomMultiItemDynamicsConfig,
    device: torch.device,
) -> Iterator[MultiItemBatch]:
    for n_items in cfg.n_items_list:
        df_n = df_sequences[df_sequences["n_items"] == int(n_items)].copy().reset_index(drop=True)
        if df_n.empty:
            continue
        for batch_id, start in enumerate(range(0, len(df_n), int(cfg.batch_size))):
            batch_df = df_n.iloc[start : start + int(cfg.batch_size)].copy().reset_index(drop=True)
            ids: list[int] = []
            for position in range(1, int(n_items) + 1):
                ids.extend(batch_df[f"item{position}_id"].astype(int).tolist())
            spike_lookup = build_spike_lookup(images, encoder, ids, steps=int(cfg.sample_steps), device=device)
            item_spikes = tuple(
                torch.stack(
                    [spike_lookup[int(image_id)] for image_id in batch_df[f"item{position}_id"].astype(int).tolist()],
                    dim=0,
                )
                for position in range(1, int(n_items) + 1)
            )
            yield MultiItemBatch(
                batch_id=int(batch_id),
                n_items=int(n_items),
                df=batch_df,
                item_spikes=item_spikes,
            )


def forward_three_layers(
    net,
    input_t: torch.Tensor,
    t_step: int,
    *,
    stsp_mode: str,
    ping_drive: torch.Tensor | None = None,
) -> None:
    s1, _ = net.layer1.forward_step(
        input_t,
        t_step,
        training=False,
        monitor=False,
        stsp_mode=stsp_mode,
        ping_drive=ping_drive,
    )
    s1_p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1_p, t_step, training=False, monitor=False, stsp_mode=stsp_mode)
    s2_p = net.pool2(s2.float())
    net.layer3.forward_step(s2_p, t_step, training=False, monitor=False, stsp_mode=stsp_mode)


def snapshot_layer3_state(
    net,
    *,
    batch_size: int,
    snapshot_name: str,
    snapshot_index: int,
    layer_input_shapes: Mapping[str, tuple[int, ...]],
) -> Layer3Snapshot:
    layer = net.layer3
    if getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
        raise ValueError("layer3 is missing STSP state at the requested snapshot.")
    u = layer.u_pre.detach().view(batch_size, -1)
    x = layer.x_pre.detach().view(batch_size, -1)
    g = (layer.u_pre * layer.x_pre).detach().view(batch_size, -1)
    return Layer3Snapshot(
        snapshot_name=str(snapshot_name),
        snapshot_index=int(snapshot_index),
        layer_input_shapes={str(key): tuple(value) for key, value in layer_input_shapes.items()},
        restore_ux_by_layer={
            "layer3": (
                layer.u_pre.detach().cpu().clone(),
                layer.x_pre.detach().cpu().clone(),
            )
        },
        state={
            "u": u.cpu().numpy().astype(np.float32, copy=False),
            "x": x.cpu().numpy().astype(np.float32, copy=False),
            "g": g.cpu().numpy().astype(np.float32, copy=False),
        },
    )


def run_sequence_with_snapshots(
    net,
    batch: MultiItemBatch,
    cfg: RandomMultiItemDynamicsConfig,
) -> SequenceRollout:
    first_sequence = batch.item_spikes[0]
    batch_size, _, channels, height, width = first_sequence.shape
    ordered_snapshots: list[Layer3Snapshot] = []
    named_snapshots: dict[str, Layer3Snapshot] = {}
    with torch.no_grad():
        prepare_network_state(net, batch_size, channels, height, width)
        layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)
        zero_input = torch.zeros(
            (batch_size, channels, height, width),
            dtype=first_sequence.dtype,
            device=first_sequence.device,
        )
        current_time = 0

        def capture(name: str) -> None:
            snapshot = snapshot_layer3_state(
                net,
                batch_size=batch_size,
                snapshot_name=name,
                snapshot_index=len(ordered_snapshots),
                layer_input_shapes=layer_input_shapes,
            )
            ordered_snapshots.append(snapshot)
            named_snapshots[str(name)] = snapshot

        capture("pre_1")
        for position, sequence in enumerate(batch.item_spikes, start=1):
            for step_idx in range(int(sequence.shape[1])):
                forward_three_layers(net, sequence[:, step_idx, ...], current_time, stsp_mode="dynamic")
                current_time += 1
            capture(f"post_{position}")
            delay_steps = int(cfg.delay_steps if position < batch.n_items else cfg.final_delay_steps)
            for _ in range(delay_steps):
                forward_three_layers(net, zero_input, current_time, stsp_mode="dynamic")
                current_time += 1
            if position < batch.n_items:
                capture(f"before_{position + 1}")
            else:
                capture("final_pre_ping")

    for position in range(2, batch.n_items + 1):
        named_snapshots[f"pre_{position}"] = named_snapshots[f"before_{position}"]
    return SequenceRollout(
        ordered_snapshots=tuple(ordered_snapshots),
        named_snapshots=named_snapshots,
    )


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


def _winner_from_scores(scores: Sequence[float], *, require_positive: bool = False) -> int:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size <= 0 or not np.any(np.isfinite(arr)):
        return 0
    if require_positive and float(np.nanmax(arr)) <= 0.0:
        return 0
    return int(np.nanargmax(arr) + 1)


def compute_item_unique_synapse_sets(
    rollout: SequenceRollout,
    *,
    n_items: int,
    unique_threshold: float,
) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    prior_union: np.ndarray | None = None
    for position in range(1, int(n_items) + 1):
        pre_snapshot_name = "pre_1" if position == 1 else f"before_{position}"
        pre_g = np.asarray(rollout.named_snapshots[pre_snapshot_name].state["g"], dtype=np.float64)
        post_g = np.asarray(rollout.named_snapshots[f"post_{position}"].state["g"], dtype=np.float64)
        delta = post_g - pre_g
        mask = delta > float(unique_threshold)
        if prior_union is None:
            unique_mask = mask.copy()
            prior_union = mask.copy()
        else:
            unique_mask = mask & (~prior_union)
            prior_union = prior_union | mask
        out[int(position)] = {
            "mask": mask,
            "unique_mask": unique_mask,
            "delta": delta.astype(np.float32, copy=False),
        }
    return out


def _trial_base_row(batch_df: pd.DataFrame, row_idx: int) -> dict[str, object]:
    record = batch_df.iloc[int(row_idx)].to_dict()
    return {str(key): json_safe(value) for key, value in record.items()}


def _fill_position_columns(
    row: dict[str, object],
    *,
    prefix: str,
    values: Sequence[float],
    max_n: int,
) -> None:
    for position in range(1, int(max_n) + 1):
        column = f"{prefix}_{position}"
        row[column] = safe_float(values[position - 1]) if position <= len(values) else None


def compute_shape_metrics_over_time(
    batch_df: pd.DataFrame,
    rollout: SequenceRollout,
    *,
    n_items: int,
    max_n: int,
    eps: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    templates = {
        position: np.asarray(rollout.named_snapshots[f"post_{position}"].state["g"], dtype=np.float64)
        for position in range(1, int(n_items) + 1)
    }
    for snapshot in rollout.ordered_snapshots:
        current_g = np.asarray(snapshot.state["g"], dtype=np.float64)
        similarities = {
            position: centered_cosine_similarity(current_g, template, eps=eps)
            for position, template in templates.items()
        }
        for row_idx in range(len(batch_df)):
            row = _trial_base_row(batch_df, row_idx)
            row["record_type"] = "trial_level"
            row["snapshot_index"] = int(snapshot.snapshot_index)
            row["snapshot_name"] = str(snapshot.snapshot_name)
            shape_values = [float(similarities[position][row_idx]) for position in range(1, int(n_items) + 1)]
            _fill_position_columns(row, prefix="ShapeSim", values=shape_values, max_n=max_n)
            row["ShapeWinner"] = int(_winner_from_scores(shape_values, require_positive=False))
            rows.append(row)
    return rows


def compute_strength_metrics_over_time(
    batch_df: pd.DataFrame,
    rollout: SequenceRollout,
    unique_synapse_sets: Mapping[int, Mapping[str, np.ndarray]],
    *,
    n_items: int,
    max_n: int,
    baseline_u: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    mask_bank = {position: np.asarray(item["mask"], dtype=bool) for position, item in unique_synapse_sets.items()}
    unique_mask_bank = {
        position: np.asarray(item["unique_mask"], dtype=bool) for position, item in unique_synapse_sets.items()
    }
    support_counts = {
        position: np.sum(mask_bank[position], axis=1).astype(np.int64, copy=False)
        for position in range(1, int(n_items) + 1)
    }
    unique_support_counts = {
        position: np.sum(unique_mask_bank[position], axis=1).astype(np.int64, copy=False)
        for position in range(1, int(n_items) + 1)
    }
    for snapshot in rollout.ordered_snapshots:
        current_g = np.asarray(snapshot.state["g"], dtype=np.float64)
        positive_mass = np.maximum(current_g - float(baseline_u), 0.0)
        strengths = {
            position: np.sum(positive_mass * mask_bank[position], axis=1)
            for position in range(1, int(n_items) + 1)
        }
        total_strength = np.zeros(len(batch_df), dtype=np.float64)
        for position in range(1, int(n_items) + 1):
            total_strength += strengths[position]
        pstrengths = {
            position: np.divide(
                strengths[position],
                total_strength,
                out=np.full(len(batch_df), np.nan, dtype=np.float64),
                where=total_strength > 0.0,
            )
            for position in range(1, int(n_items) + 1)
        }
        for row_idx in range(len(batch_df)):
            row = _trial_base_row(batch_df, row_idx)
            row["record_type"] = "trial_level"
            row["snapshot_index"] = int(snapshot.snapshot_index)
            row["snapshot_name"] = str(snapshot.snapshot_name)
            strength_values = [float(strengths[position][row_idx]) for position in range(1, int(n_items) + 1)]
            pstrength_values = [float(pstrengths[position][row_idx]) for position in range(1, int(n_items) + 1)]
            _fill_position_columns(row, prefix="Strength", values=strength_values, max_n=max_n)
            _fill_position_columns(row, prefix="PStrength", values=pstrength_values, max_n=max_n)
            for position in range(1, int(max_n) + 1):
                row[f"MaskCount_{position}"] = (
                    int(support_counts[position][row_idx]) if position <= int(n_items) else None
                )
                row[f"UniqueMaskCount_{position}"] = (
                    int(unique_support_counts[position][row_idx]) if position <= int(n_items) else None
                )
            row["StrengthWinner"] = int(_winner_from_scores(strength_values, require_positive=True))
            rows.append(row)
    return rows


def run_neutral_ping_from_snapshot(
    net,
    cfg: RandomMultiItemDynamicsConfig,
    snapshot: Layer3Snapshot,
    *,
    item_labels: Mapping[int, np.ndarray],
) -> PingReadout:
    batch_size = int(len(next(iter(item_labels.values()))))
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
        for step_idx in range(int(cfg.ping_steps)):
            forward_three_layers(net, zero_input, step_idx, stsp_mode="dynamic", ping_drive=ping_drive)
    first_fire_pred, first_fire_t = decode_prediction_and_fire_time_from_layer3(net, batch_size=batch_size)
    pred_np = first_fire_pred.numpy().astype(np.int64, copy=False)
    fire_t_np = first_fire_t.numpy().astype(np.int64, copy=False)
    silent_mask = (pred_np < 0).astype(np.int64, copy=False)
    target_first = {
        int(position): (((silent_mask == 0) & (pred_np == labels)).astype(np.int64, copy=False))
        for position, labels in item_labels.items()
    }
    stacked = np.stack([target_first[position] for position in sorted(target_first)], axis=1)
    member_counts = np.sum(stacked, axis=1)
    nonmember_first = (((silent_mask == 0) & (member_counts == 0)).astype(np.int64, copy=False))
    ambiguous_member_first = (member_counts > 1).astype(np.int64, copy=False)
    ping_winner = np.zeros(batch_size, dtype=np.int64)
    unique_member_mask = member_counts == 1
    if np.any(unique_member_mask):
        ping_winner[unique_member_mask] = np.argmax(stacked[unique_member_mask], axis=1).astype(np.int64) + 1
    return PingReadout(
        first_fire_pred=pred_np,
        first_fire_t=fire_t_np,
        silent_mask=silent_mask,
        nonmember_first=nonmember_first,
        ambiguous_member_first=ambiguous_member_first,
        target_first=target_first,
        ping_winner=ping_winner,
    )


def compute_ping_metrics_over_time(
    net,
    cfg: RandomMultiItemDynamicsConfig,
    batch_df: pd.DataFrame,
    rollout: SequenceRollout,
    *,
    n_items: int,
    max_n: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    item_labels = {
        position: batch_df[f"item{position}_label"].to_numpy(dtype=np.int64, copy=False)
        for position in range(1, int(n_items) + 1)
    }
    for snapshot in rollout.ordered_snapshots:
        ping = run_neutral_ping_from_snapshot(net, cfg, snapshot, item_labels=item_labels)
        for row_idx in range(len(batch_df)):
            row = _trial_base_row(batch_df, row_idx)
            row["record_type"] = "trial_level"
            row["snapshot_index"] = int(snapshot.snapshot_index)
            row["snapshot_name"] = str(snapshot.snapshot_name)
            ping_values = [int(ping.target_first[position][row_idx]) for position in range(1, int(n_items) + 1)]
            _fill_position_columns(row, prefix="PingProb", values=ping_values, max_n=max_n)
            row["PingWinner"] = int(ping.ping_winner[row_idx])
            row["first_fire_pred"] = int(ping.first_fire_pred[row_idx])
            row["first_fire_t"] = int(ping.first_fire_t[row_idx])
            row["silent_prob"] = int(ping.silent_mask[row_idx])
            row["other_first_prob"] = int(ping.nonmember_first[row_idx])
            row["ambiguous_member_prob"] = int(ping.ambiguous_member_first[row_idx])
            rows.append(row)
    return rows


def _winner_fraction_records(
    df_trials: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    winner_col: str,
    prefix: str,
    max_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_values, sub_df in df_trials.groupby(list(group_cols), dropna=False, as_index=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {str(group_cols[idx]): json_safe(group_values[idx]) for idx in range(len(group_cols))}
        winners = sub_df[winner_col].to_numpy(dtype=np.int64, copy=False)
        n_items = int(sub_df["n_items"].iloc[0])
        total = max(len(sub_df), 1)
        for position in range(1, int(max_n) + 1):
            row[f"{prefix}_item{position}_frac"] = float(np.mean(winners == position))
        row[f"{prefix}_none_frac"] = float(np.mean(winners == 0))
        row[f"{prefix}_recent_frac"] = float(np.mean(winners == n_items))
        row[f"{prefix}_count"] = int(total)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metric_table(
    rows: Sequence[Mapping[str, object]],
    *,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
    winner_col: str,
    winner_prefix: str,
    max_n: int,
) -> pd.DataFrame:
    if len(rows) <= 0:
        return pd.DataFrame(columns=["record_type", *group_cols, *value_cols, winner_col])
    df = pd.DataFrame(rows)
    if "record_type" not in df.columns:
        df.insert(0, "record_type", "trial_level")
    df_trials = df[df["record_type"] == "trial_level"].copy()
    if df_trials.empty:
        return df
    summary_mean = (
        df_trials.groupby(list(group_cols), dropna=False, as_index=False)[list(value_cols)].mean(numeric_only=True)
    )
    summary_winners = _winner_fraction_records(
        df_trials,
        group_cols=group_cols,
        winner_col=winner_col,
        prefix=winner_prefix,
        max_n=max_n,
    )
    summary_df = summary_mean.merge(summary_winners, on=list(group_cols), how="left")
    summary_df.insert(0, "record_type", "summary")
    summary_df[winner_col] = np.nan
    return pd.concat([df, summary_df], axis=0, ignore_index=True, sort=False)


def compute_shape_strength_ping_dissociation(
    df_shape: pd.DataFrame,
    df_strength: pd.DataFrame,
    df_ping: pd.DataFrame,
    *,
    max_n: int,
) -> pd.DataFrame:
    shape_trials = df_shape[df_shape["record_type"] == "trial_level"][
        ["trial_id", "n_items", "snapshot_index", "snapshot_name", "ShapeWinner"]
    ].copy()
    strength_trials = df_strength[df_strength["record_type"] == "trial_level"][
        ["trial_id", "n_items", "snapshot_index", "snapshot_name", "StrengthWinner"]
    ].copy()
    ping_trials = df_ping[df_ping["record_type"] == "trial_level"][
        ["trial_id", "n_items", "snapshot_index", "snapshot_name", "PingWinner", "silent_prob", "other_first_prob"]
    ].copy()
    merged = shape_trials.merge(
        strength_trials,
        on=["trial_id", "n_items", "snapshot_index", "snapshot_name"],
        how="inner",
    ).merge(
        ping_trials,
        on=["trial_id", "n_items", "snapshot_index", "snapshot_name"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "record_type",
                "trial_id",
                "n_items",
                "snapshot_index",
                "snapshot_name",
                "ping_matches_shape",
                "ping_matches_strength",
                "shape_matches_strength",
            ]
        )
    merged["record_type"] = "trial_level"
    merged["ping_matches_shape"] = (
        (merged["PingWinner"].to_numpy(dtype=np.int64) > 0)
        & (merged["PingWinner"].to_numpy(dtype=np.int64) == merged["ShapeWinner"].to_numpy(dtype=np.int64))
    ).astype(np.int64)
    merged["ping_matches_strength"] = (
        (merged["PingWinner"].to_numpy(dtype=np.int64) > 0)
        & (merged["PingWinner"].to_numpy(dtype=np.int64) == merged["StrengthWinner"].to_numpy(dtype=np.int64))
    ).astype(np.int64)
    merged["shape_matches_strength"] = (
        (merged["StrengthWinner"].to_numpy(dtype=np.int64) > 0)
        & (merged["ShapeWinner"].to_numpy(dtype=np.int64) == merged["StrengthWinner"].to_numpy(dtype=np.int64))
    ).astype(np.int64)
    summary = (
        merged.groupby(["n_items", "snapshot_index", "snapshot_name"], dropna=False, as_index=False)[
            ["ping_matches_shape", "ping_matches_strength", "shape_matches_strength", "silent_prob", "other_first_prob"]
        ]
        .mean(numeric_only=True)
        .copy()
    )
    summary.insert(0, "record_type", "summary")
    for prefix in ("ShapeWinner", "StrengthWinner", "PingWinner"):
        for position in range(1, int(max_n) + 1):
            summary[f"{prefix}_item{position}_frac"] = np.nan
        summary[f"{prefix}_none_frac"] = np.nan
        summary[f"{prefix}_recent_frac"] = np.nan
    return pd.concat([merged, summary], axis=0, ignore_index=True, sort=False)


def compute_final_winner_metrics(
    df_shape: pd.DataFrame,
    df_strength: pd.DataFrame,
    df_ping: pd.DataFrame,
    *,
    max_n: int,
) -> pd.DataFrame:
    shape_final = df_shape[
        (df_shape["record_type"] == "trial_level") & (df_shape["snapshot_name"] == "final_pre_ping")
    ][["trial_id", "n_items", "ShapeWinner"]].copy()
    strength_final = df_strength[
        (df_strength["record_type"] == "trial_level") & (df_strength["snapshot_name"] == "final_pre_ping")
    ][["trial_id", "n_items", "StrengthWinner"]].copy()
    ping_final = df_ping[
        (df_ping["record_type"] == "trial_level") & (df_ping["snapshot_name"] == "final_pre_ping")
    ][["trial_id", "n_items", "PingWinner", "silent_prob", "other_first_prob", "ambiguous_member_prob"]].copy()
    merged = shape_final.merge(strength_final, on=["trial_id", "n_items"], how="inner").merge(
        ping_final,
        on=["trial_id", "n_items"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=["record_type", "trial_id", "n_items"])
    merged.insert(0, "record_type", "trial_level")
    summary = merged.groupby(["n_items"], dropna=False, as_index=False)[
        ["silent_prob", "other_first_prob", "ambiguous_member_prob"]
    ].mean(numeric_only=True)
    for prefix, winner_col in (
        ("ShapeWinner", "ShapeWinner"),
        ("StrengthWinner", "StrengthWinner"),
        ("PingWinner", "PingWinner"),
    ):
        winner_summary = _winner_fraction_records(
            merged,
            group_cols=["n_items"],
            winner_col=winner_col,
            prefix=prefix,
            max_n=max_n,
        )
        summary = summary.merge(winner_summary, on=["n_items"], how="left")
    summary.insert(0, "record_type", "summary")
    return pd.concat([merged, summary], axis=0, ignore_index=True, sort=False)


def _position_palette(max_n: int) -> list[str]:
    base = [SAMPLE_COLOR, DYNAMIC_COLOR, SHUFFLE_COLOR, NOISE_COLOR]
    if max_n <= len(base):
        return base[:max_n]
    cmap = plt.get_cmap("tab10")
    extra = [cmap(idx % 10) for idx in range(max_n - len(base))]
    return [*base, *extra]


def _summary_only(df: pd.DataFrame) -> pd.DataFrame:
    if "record_type" not in df.columns:
        return df.copy()
    summary = df[df["record_type"] == "summary"].copy()
    return summary if not summary.empty else df.copy()


def _plot_timecourse_lines(
    df_summary: pd.DataFrame,
    *,
    n_items: int,
    value_prefix: str,
    title: str,
    ylabel: str,
    max_n: int,
    ylim: tuple[float, float] | None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    subset = df_summary[df_summary["n_items"] == int(n_items)].sort_values("snapshot_index", kind="stable")
    x = subset["snapshot_index"].to_numpy(dtype=np.int64, copy=False)
    labels = subset["snapshot_name"].astype(str).tolist()
    colors = _position_palette(max_n)
    for position in range(1, int(n_items) + 1):
        column = f"{value_prefix}_{position}"
        if column not in subset.columns:
            continue
        y = subset[column].to_numpy(dtype=np.float64, copy=False)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            color=colors[position - 1],
            label=f"item{position}",
        )
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_xlabel("Snapshot")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, ncol=min(int(n_items), 5))
    fig.tight_layout()
    return fig


def _plot_final_winner_distribution(
    df_final_summary: pd.DataFrame,
    *,
    max_n: int,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    winner_specs = [
        ("ShapeWinner", axes[0], "Final shape winner"),
        ("StrengthWinner", axes[1], "Final strength winner"),
        ("PingWinner", axes[2], "Final ping winner"),
    ]
    summary = df_final_summary.sort_values("n_items", kind="stable")
    n_values = summary["n_items"].to_numpy(dtype=np.int64, copy=False)
    colors = _position_palette(max_n)
    for prefix, ax, title in winner_specs:
        bottom = np.zeros(len(summary), dtype=np.float64)
        for position in range(1, int(max_n) + 1):
            column = f"{prefix}_item{position}_frac"
            heights = summary[column].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            ax.bar(
                n_values,
                heights,
                bottom=bottom,
                width=0.6,
                color=colors[position - 1],
                label=f"item{position}",
                alpha=0.92,
            )
            bottom += heights
        ax.plot(
            n_values,
            summary[f"{prefix}_none_frac"].fillna(0.0).to_numpy(dtype=np.float64, copy=False),
            color="black",
            marker="o",
            linewidth=1.4,
            label="none",
        )
        ax.set_title(title)
        ax.set_xlabel("n_items")
        ax.set_xticks(n_values)
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Winner fraction")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return fig


def _plot_dissociation_summary(df_summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    summary = df_summary[df_summary["snapshot_name"] == "final_pre_ping"].sort_values("n_items", kind="stable")
    x = summary["n_items"].to_numpy(dtype=np.int64, copy=False)
    ax.plot(
        x,
        summary["ping_matches_shape"].to_numpy(dtype=np.float64, copy=False),
        marker="o",
        color=SAMPLE_COLOR,
        linewidth=1.8,
        label="ping matches shape",
    )
    ax.plot(
        x,
        summary["ping_matches_strength"].to_numpy(dtype=np.float64, copy=False),
        marker="o",
        color=DYNAMIC_COLOR,
        linewidth=1.8,
        label="ping matches strength",
    )
    ax.plot(
        x,
        summary["shape_matches_strength"].to_numpy(dtype=np.float64, copy=False),
        marker="o",
        color=SHUFFLE_COLOR,
        linewidth=1.8,
        label="shape matches strength",
    )
    ax.set_xlabel("n_items")
    ax.set_ylabel("Match rate")
    ax.set_title("Final shape-strength-ping dissociation")
    ax.set_xticks(x)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def save_random_multiitem_figures(
    layout,
    *,
    cfg: RandomMultiItemDynamicsConfig,
    df_shape: pd.DataFrame,
    df_strength: pd.DataFrame,
    df_ping: pd.DataFrame,
    df_final_winners: pd.DataFrame,
    df_dissociation: pd.DataFrame,
) -> dict[str, object]:
    apply_publication_style()
    figure_paths: dict[str, object] = {}
    max_n = int(cfg.max_n_items)

    shape_summary = _summary_only(df_shape)
    fig_shape = _plot_timecourse_lines(
        shape_summary,
        n_items=int(cfg.timecourse_n),
        value_prefix="ShapeSim",
        title=f"Shape trajectory (n={cfg.timecourse_n})",
        ylabel="Centered cosine similarity",
        max_n=max_n,
        ylim=None,
    )
    figure_paths["shape_trajectory"] = save_figure_all_formats(fig_shape, layout.figure_base("shape_trajectory"))
    plt.close(fig_shape)

    strength_summary = _summary_only(df_strength)
    fig_strength = _plot_timecourse_lines(
        strength_summary,
        n_items=int(cfg.timecourse_n),
        value_prefix="PStrength",
        title=f"Strength trajectory (n={cfg.timecourse_n})",
        ylabel="Normalized strength share",
        max_n=max_n,
        ylim=(0.0, 1.0),
    )
    figure_paths["strength_trajectory"] = save_figure_all_formats(
        fig_strength,
        layout.figure_base("strength_trajectory"),
    )
    plt.close(fig_strength)

    ping_summary = _summary_only(df_ping)
    fig_ping = _plot_timecourse_lines(
        ping_summary,
        n_items=int(cfg.timecourse_n),
        value_prefix="PingProb",
        title=f"Ping trajectory (n={cfg.timecourse_n})",
        ylabel="First-fire probability",
        max_n=max_n,
        ylim=(0.0, 1.0),
    )
    figure_paths["ping_trajectory"] = save_figure_all_formats(fig_ping, layout.figure_base("ping_trajectory"))
    plt.close(fig_ping)

    final_summary = _summary_only(df_final_winners)
    fig_final = _plot_final_winner_distribution(final_summary, max_n=max_n)
    figure_paths["final_winner_distribution"] = save_figure_all_formats(
        fig_final,
        layout.figure_base("final_winner_distribution"),
    )
    plt.close(fig_final)

    dissociation_summary = _summary_only(df_dissociation)
    fig_dissociation = _plot_dissociation_summary(dissociation_summary)
    figure_paths["dissociation_summary"] = save_figure_all_formats(
        fig_dissociation,
        layout.figure_base("dissociation_summary"),
    )
    plt.close(fig_dissociation)
    return figure_paths


def build_nested_summary(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> dict[str, object]:
    summary_df = _summary_only(df)
    if summary_df.empty:
        return {}
    out: dict[str, object] = {}
    for record in summary_df[list(group_cols) + list(value_cols)].to_dict(orient="records"):
        cursor = out
        for group_col in group_cols[:-1]:
            cursor = cursor.setdefault(_summary_key(record[group_col]), {})
        leaf_key = _summary_key(record[group_cols[-1]])
        cursor[leaf_key] = {str(col): json_safe(record[col]) for col in value_cols}
    return out


def build_summary_payload(
    cfg: RandomMultiItemDynamicsConfig,
    *,
    device_requested: str,
    device_resolved: str,
    sequence_count: int,
    df_shape: pd.DataFrame,
    df_strength: pd.DataFrame,
    df_ping: pd.DataFrame,
    df_final_winners: pd.DataFrame,
    df_dissociation: pd.DataFrame,
    exported_files: Mapping[str, object],
) -> dict[str, object]:
    max_n = int(cfg.max_n_items)
    shape_value_cols = ["snapshot_name", *[f"ShapeSim_{position}" for position in range(1, max_n + 1)]]
    shape_value_cols.extend([f"ShapeWinner_item{position}_frac" for position in range(1, max_n + 1)])
    shape_value_cols.extend(["ShapeWinner_recent_frac", "ShapeWinner_none_frac"])
    strength_value_cols = ["snapshot_name"]
    strength_value_cols.extend([f"Strength_{position}" for position in range(1, max_n + 1)])
    strength_value_cols.extend([f"PStrength_{position}" for position in range(1, max_n + 1)])
    strength_value_cols.extend([f"StrengthWinner_item{position}_frac" for position in range(1, max_n + 1)])
    strength_value_cols.extend(["StrengthWinner_recent_frac", "StrengthWinner_none_frac"])
    ping_value_cols = ["snapshot_name", *[f"PingProb_{position}" for position in range(1, max_n + 1)]]
    ping_value_cols.extend(["silent_prob", "other_first_prob", "ambiguous_member_prob"])
    ping_value_cols.extend([f"PingWinner_item{position}_frac" for position in range(1, max_n + 1)])
    ping_value_cols.extend(["PingWinner_recent_frac", "PingWinner_none_frac"])
    final_value_cols = ["silent_prob", "other_first_prob", "ambiguous_member_prob"]
    for prefix in ("ShapeWinner", "StrengthWinner", "PingWinner"):
        final_value_cols.extend([f"{prefix}_item{position}_frac" for position in range(1, max_n + 1)])
        final_value_cols.extend([f"{prefix}_recent_frac", f"{prefix}_none_frac"])
    return {
        "config": json_safe(asdict(cfg)),
        "device_requested": str(device_requested),
        "device_resolved": str(device_resolved),
        "sequence_count": int(sequence_count),
        "timecourse_shape_summary": {
            "by_n": build_nested_summary(
                df_shape,
                group_cols=["n_items", "snapshot_index"],
                value_cols=shape_value_cols,
            )
        },
        "timecourse_strength_summary": {
            "by_n": build_nested_summary(
                df_strength,
                group_cols=["n_items", "snapshot_index"],
                value_cols=strength_value_cols,
            )
        },
        "timecourse_ping_summary": {
            "by_n": build_nested_summary(
                df_ping,
                group_cols=["n_items", "snapshot_index"],
                value_cols=ping_value_cols,
            )
        },
        "final_winner_summary": {
            "by_n": build_nested_summary(
                df_final_winners,
                group_cols=["n_items"],
                value_cols=final_value_cols,
            )
        },
        "dissociation_summary": {
            "by_n": build_nested_summary(
                df_dissociation,
                group_cols=["n_items", "snapshot_index"],
                value_cols=[
                    "snapshot_name",
                    "ping_matches_shape",
                    "ping_matches_strength",
                    "shape_matches_strength",
                    "silent_prob",
                    "other_first_prob",
                ],
            )
        },
        "exported_files": json_safe(exported_files),
    }


def collect_example_state_npz(
    rollout: SequenceRollout,
    *,
    tag: str,
    max_trials: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for snapshot in rollout.ordered_snapshots:
        for state_name in ("u", "x", "g"):
            out[f"{tag}_{snapshot.snapshot_name}_{state_name}"] = snapshot.state[state_name][:max_trials].astype(
                np.float32,
                copy=False,
            )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = normalize_config(args)
    layout = prepare_result_layout(cfg.output_dir)
    log_lines: list[str] = []

    device, device_message = resolve_device_with_fallback(cfg.device)
    log_and_print(log_lines, device_message)
    save_run_config(json_safe(asdict(cfg)), layout.root)

    seed_everything(int(cfg.seed))
    log_and_print(log_lines, f"[Setup] seed={cfg.seed}")
    log_and_print(log_lines, f"[Setup] loading dataset split={cfg.split} from {cfg.dataset_root}")
    dataset = load_mnist_dataset(cfg.dataset_root, cfg.split)
    images, labels, _ = build_dataset_arrays(dataset)
    df_sequences = build_random_multiitem_sequences(labels, cfg)
    sequences_csv = save_tidy_csv(
        df_sequences,
        layout.data_file("random_multiitem_sequences.csv"),
        sort_by=["n_items", "trial_id"],
    )
    log_and_print(log_lines, f"[Data] sequences={len(df_sequences)} saved={sequences_csv}")

    net, encoder = load_model_and_encoder(
        cfg.model_path,
        device=device,
        dt=cfg.dt,
        max_duration_ms=cfg.max_duration_ms,
    )
    baseline_u = float(net.layer3.stsp_U)
    log_and_print(log_lines, f"[Model] loaded {cfg.model_path}")
    log_and_print(log_lines, f"[Model] layer3 baseline U={baseline_u:.6f}")

    max_n = int(cfg.max_n_items)
    shape_rows: list[dict[str, object]] = []
    strength_rows: list[dict[str, object]] = []
    ping_rows: list[dict[str, object]] = []
    example_state_payload: dict[str, np.ndarray] = {}

    batch_iter = prepare_multiitem_batches(df_sequences, images, encoder, cfg, device)
    for batch_counter, batch in enumerate(batch_iter, start=1):
        log_and_print(
            log_lines,
            f"[Run] batch={batch_counter} n_items={batch.n_items} batch_size={len(batch.df)}",
        )
        rollout = run_sequence_with_snapshots(net, batch, cfg)
        unique_synapse_sets = compute_item_unique_synapse_sets(
            rollout,
            n_items=batch.n_items,
            unique_threshold=cfg.unique_threshold,
        )
        shape_rows.extend(
            compute_shape_metrics_over_time(
                batch.df,
                rollout,
                n_items=batch.n_items,
                max_n=max_n,
                eps=cfg.epsilon,
            )
        )
        strength_rows.extend(
            compute_strength_metrics_over_time(
                batch.df,
                rollout,
                unique_synapse_sets,
                n_items=batch.n_items,
                max_n=max_n,
                baseline_u=baseline_u,
            )
        )
        ping_rows.extend(
            compute_ping_metrics_over_time(
                net,
                cfg,
                batch.df,
                rollout,
                n_items=batch.n_items,
                max_n=max_n,
            )
        )
        if cfg.save_example_states and batch.n_items == cfg.timecourse_n and not example_state_payload:
            example_state_payload.update(
                collect_example_state_npz(
                    rollout,
                    tag=f"n{batch.n_items}_batch{batch.batch_id}",
                    max_trials=min(len(batch.df), 2),
                )
            )

    shape_value_cols = [f"ShapeSim_{position}" for position in range(1, max_n + 1)]
    strength_value_cols = [f"Strength_{position}" for position in range(1, max_n + 1)]
    strength_value_cols.extend([f"PStrength_{position}" for position in range(1, max_n + 1)])
    strength_value_cols.extend([f"MaskCount_{position}" for position in range(1, max_n + 1)])
    strength_value_cols.extend([f"UniqueMaskCount_{position}" for position in range(1, max_n + 1)])
    ping_value_cols = [f"PingProb_{position}" for position in range(1, max_n + 1)]
    ping_value_cols.extend(["first_fire_pred", "first_fire_t", "silent_prob", "other_first_prob", "ambiguous_member_prob"])

    df_shape = summarize_metric_table(
        shape_rows,
        group_cols=["n_items", "snapshot_index", "snapshot_name"],
        value_cols=shape_value_cols,
        winner_col="ShapeWinner",
        winner_prefix="ShapeWinner",
        max_n=max_n,
    )
    df_strength = summarize_metric_table(
        strength_rows,
        group_cols=["n_items", "snapshot_index", "snapshot_name"],
        value_cols=strength_value_cols,
        winner_col="StrengthWinner",
        winner_prefix="StrengthWinner",
        max_n=max_n,
    )
    df_ping = summarize_metric_table(
        ping_rows,
        group_cols=["n_items", "snapshot_index", "snapshot_name"],
        value_cols=ping_value_cols,
        winner_col="PingWinner",
        winner_prefix="PingWinner",
        max_n=max_n,
    )
    df_final_winners = compute_final_winner_metrics(df_shape, df_strength, df_ping, max_n=max_n)
    df_dissociation = compute_shape_strength_ping_dissociation(df_shape, df_strength, df_ping, max_n=max_n)

    shape_csv = save_tidy_csv(
        df_shape,
        layout.data_file("multiitem_timecourse_shape_metrics.csv"),
        sort_by=["record_type", "n_items", "snapshot_index", "trial_id"],
    )
    strength_csv = save_tidy_csv(
        df_strength,
        layout.data_file("multiitem_timecourse_strength_metrics.csv"),
        sort_by=["record_type", "n_items", "snapshot_index", "trial_id"],
    )
    ping_csv = save_tidy_csv(
        df_ping,
        layout.data_file("multiitem_timecourse_ping_metrics.csv"),
        sort_by=["record_type", "n_items", "snapshot_index", "trial_id"],
    )
    final_winner_csv = save_tidy_csv(
        df_final_winners,
        layout.data_file("multiitem_final_winner_metrics.csv"),
        sort_by=["record_type", "n_items", "trial_id"],
    )
    dissociation_csv = save_tidy_csv(
        df_dissociation,
        layout.data_file("multiitem_dissociation_metrics.csv"),
        sort_by=["record_type", "n_items", "snapshot_index", "trial_id"],
    )

    exported_files: dict[str, object] = {
        "random_multiitem_sequences_csv": str(sequences_csv),
        "multiitem_timecourse_shape_metrics_csv": str(shape_csv),
        "multiitem_timecourse_strength_metrics_csv": str(strength_csv),
        "multiitem_timecourse_ping_metrics_csv": str(ping_csv),
        "multiitem_final_winner_metrics_csv": str(final_winner_csv),
        "multiitem_dissociation_metrics_csv": str(dissociation_csv),
    }

    if example_state_payload:
        example_npz_path = layout.data_file("example_layer3_states.npz")
        np.savez_compressed(example_npz_path, **example_state_payload)
        exported_files["example_layer3_states_npz"] = str(example_npz_path)
        log_and_print(log_lines, f"[Export] example_states={example_npz_path}")

    if not cfg.skip_figures:
        figure_paths = save_random_multiitem_figures(
            layout,
            cfg=cfg,
            df_shape=df_shape,
            df_strength=df_strength,
            df_ping=df_ping,
            df_final_winners=df_final_winners,
            df_dissociation=df_dissociation,
        )
        exported_files["figures"] = figure_paths
        log_and_print(log_lines, f"[Export] figures={len(figure_paths)} groups")

    summary_path = layout.root_file("summary.json")
    log_txt_path = layout.root_file("log.txt")
    exported_files["summary_json"] = str(summary_path)
    exported_files["log_txt"] = str(log_txt_path)
    summary_payload = build_summary_payload(
        cfg,
        device_requested=cfg.device,
        device_resolved=str(device),
        sequence_count=len(df_sequences),
        df_shape=df_shape,
        df_strength=df_strength,
        df_ping=df_ping,
        df_final_winners=df_final_winners,
        df_dissociation=df_dissociation,
        exported_files=exported_files,
    )
    log_and_print(log_lines, f"[Done] summary={summary_path}")
    save_summary_json(json_safe(summary_payload), layout.root)
    save_log_lines(log_lines, layout.root, filename="log.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
