from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy import stats
from tqdm import tqdm

from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.plotting.common.io import (
    COLOR_DYNAMIC,
    COLOR_STATIC,
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
)
from src.plotting.common.theme_tokens import (
    ALPHA_BAR,
    COLOR_ACCENT_BLUE,
    COLOR_ACCENT_GREEN,
    COLOR_ACCENT_PURPLE,
    COLOR_ACCENT_RED,
    COLOR_ACCENT_SKY,
    COLOR_OFFWHITE,
    FIGSIZE_SINGLE_PANEL_COMPACT,
    FIGSIZE_THREE_PANEL_HEATMAP,
    FIGSIZE_THREE_PANEL_WIDE,
    GRID_ALPHA_SOFT,
    LINE_WIDTH_GUIDE,
    LINE_WIDTH_PRIMARY,
    LINE_WIDTH_REFERENCE,
    LINE_WIDTH_SECONDARY,
    MARKER_CIRCLE,
    apply_standard_legend,
)

ms = 1.0

STSP_MODES: Tuple[str, ...] = ("dynamic", "static_frozen")
DEFAULT_MODEL_PATH = "results/sdnn_deep_final/net_final.pth"
DEFAULT_OUTPUT_DIR = "results/similarity_bias_experiment"
DEFAULT_DATASET_ROOT = "./MNIST"
DEFAULT_NUM_BINS = 4
DEFAULT_MAX_PAIRS = 5000
DEFAULT_BATCH_SIZE = 128
DEFAULT_DELAY_MS = 500.0
DEFAULT_SAMPLE_MS = 200.0
DEFAULT_PROBE_MS = 100.0
DEFAULT_REPEAT_COUNT = 1
DEFAULT_CANDIDATE_MULTIPLIER = 5


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


def mix_seed(base_seed: int, *parts: int) -> int:
    value = int(base_seed) & 0xFFFFFFFF
    for idx, part in enumerate(parts, start=1):
        value = (value * 1664525 + 1013904223 + int(part) * (374761393 + idx * 97)) & 0xFFFFFFFF
    return int(value)


def _bin_labels(num_bins: int) -> list[str]:
    return [f"bin_{idx + 1}" for idx in range(int(num_bins))]


def _sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        return None
    return scalar


def _to_json_ready(value):
    if isinstance(value, dict):
        return {str(key): _to_json_ready(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _safe_float(value)
    return value


def _save_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_dataset(dataset_root: str, split: str):
    train_loader, _, test_loader = build_mnist_skeleton_loader(
        root=dataset_root,
        batch_size=1,
        input_size=28,
        num_workers=0,
    )
    split_name = str(split).strip().lower()
    if split_name == "train":
        return train_loader.dataset
    if split_name == "test":
        return test_loader.dataset
    raise ValueError(f"Unsupported split: {split}")


def _sample_class_pool(class_index: Mapping[int, Sequence[int]], max_samples: int | None, seed: int) -> Dict[int, list[int]]:
    rng = random.Random(int(seed))
    pooled: Dict[int, list[int]] = {}
    for class_label in sorted(class_index):
        indices = [int(idx) for idx in class_index[int(class_label)]]
        if max_samples is not None and int(max_samples) > 0 and len(indices) > int(max_samples):
            pooled[int(class_label)] = sorted(rng.sample(indices, int(max_samples)))
        else:
            pooled[int(class_label)] = sorted(indices)
    return pooled


def _cosine_similarity_1d(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a_flat = a.reshape(-1).to(torch.float32)
    b_flat = b.reshape(-1).to(torch.float32)
    denom = float(torch.norm(a_flat) * torch.norm(b_flat))
    if denom <= eps:
        return 0.0
    return float(torch.dot(a_flat, b_flat) / denom)


def _get_image_from_cache(dataset, index: int, image_cache: Dict[int, torch.Tensor]) -> torch.Tensor:
    cached = image_cache.get(int(index))
    if cached is not None:
        return cached
    image = dataset[int(index)][0].detach().cpu().to(torch.float32)
    image_cache[int(index)] = image
    return image


def _sample_unique_pairs_for_combo(
    sample_ids: Sequence[int],
    probe_ids: Sequence[int],
    target_count: int,
    rng: random.Random,
    same_label: bool,
) -> list[tuple[int, int]]:
    if target_count <= 0:
        return []
    chosen: set[tuple[int, int]] = set()
    max_attempts = max(int(target_count) * 30, 200)
    attempts = 0
    sample_list = [int(idx) for idx in sample_ids]
    probe_list = [int(idx) for idx in probe_ids]
    while len(chosen) < int(target_count) and attempts < max_attempts:
        attempts += 1
        sample_id = int(rng.choice(sample_list))
        probe_id = int(rng.choice(probe_list))
        if same_label and sample_id == probe_id:
            continue
        chosen.add((sample_id, probe_id))
    return list(chosen)


def _assign_similarity_bins(df: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    if len(df) < int(num_bins):
        raise ValueError(f"Need at least {num_bins} pairs to assign similarity bins; got {len(df)}.")
    out = df.copy()
    labels = _bin_labels(num_bins)
    ranks = out["pixel_similarity"].rank(method="first")
    out["similarity_bin"] = pd.qcut(ranks, q=int(num_bins), labels=labels).astype("object")
    out["similarity_bin_index"] = out["similarity_bin"].map({label: idx for idx, label in enumerate(labels)}).astype(np.int64)
    return out


def _select_balanced_pairs_from_candidates(df_candidates: pd.DataFrame, max_pairs: int, num_bins: int, seed: int) -> pd.DataFrame:
    if len(df_candidates) <= int(max_pairs):
        selected = df_candidates.copy()
    else:
        shuffled = df_candidates.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
        per_bin_base = int(max_pairs) // int(num_bins)
        remainder = int(max_pairs) % int(num_bins)
        selected_parts: list[pd.DataFrame] = []
        leftover_parts: list[pd.DataFrame] = []
        for bin_idx, label in enumerate(_bin_labels(num_bins)):
            sub = shuffled[shuffled["similarity_bin"] == label].copy().reset_index(drop=True)
            target = per_bin_base + (1 if bin_idx < remainder else 0)
            take = min(target, len(sub))
            selected_parts.append(sub.iloc[:take].copy())
            if take < len(sub):
                leftover_parts.append(sub.iloc[take:].copy())
        selected = pd.concat(selected_parts, axis=0, ignore_index=True) if selected_parts else shuffled.iloc[:0].copy()
        if len(selected) < int(max_pairs) and leftover_parts:
            leftovers = pd.concat(leftover_parts, axis=0, ignore_index=True)
            fill = leftovers.iloc[: int(max_pairs) - len(selected)].copy()
            selected = pd.concat([selected, fill], axis=0, ignore_index=True)
    selected = selected.sample(frac=1.0, random_state=int(seed) + 13).reset_index(drop=True)
    selected = selected.iloc[: int(max_pairs)].copy()
    selected = _assign_similarity_bins(selected, num_bins=num_bins)
    selected = selected.sort_values(["similarity_bin_index", "sample_label", "probe_label", "sample_id", "probe_id"], kind="stable")
    selected = selected.reset_index(drop=True)
    selected["pair_id"] = np.arange(len(selected), dtype=np.int64)
    return selected


def build_pair_specs(
    dataset,
    class_index: Mapping[int, Sequence[int]],
    *,
    num_bins: int,
    max_pairs: int,
    max_samples: int | None,
    seed: int,
    candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
) -> pd.DataFrame:
    if int(max_pairs) <= 0:
        raise ValueError("--max-pairs must be positive.")
    image_cache: Dict[int, torch.Tensor] = {}
    class_pools = _sample_class_pool(class_index=class_index, max_samples=max_samples, seed=mix_seed(seed, 101))
    label_pairs = [(int(sample_label), int(probe_label)) for sample_label in sorted(class_pools) for probe_label in sorted(class_pools)]
    target_candidates = max(int(max_pairs), int(max_pairs) * int(candidate_multiplier))
    per_combo_target = max(1, int(math.ceil(target_candidates / max(len(label_pairs), 1))))
    rows: list[dict[str, object]] = []
    for combo_idx, (sample_label, probe_label) in enumerate(label_pairs):
        sampled_pairs = _sample_unique_pairs_for_combo(
            sample_ids=class_pools[int(sample_label)],
            probe_ids=class_pools[int(probe_label)],
            target_count=per_combo_target,
            rng=random.Random(mix_seed(seed, 211, combo_idx, sample_label, probe_label)),
            same_label=int(sample_label) == int(probe_label),
        )
        for sample_id, probe_id in sampled_pairs:
            sample_image = _get_image_from_cache(dataset, sample_id, image_cache=image_cache)
            probe_image = _get_image_from_cache(dataset, probe_id, image_cache=image_cache)
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "probe_id": int(probe_id),
                    "sample_label": int(sample_label),
                    "probe_label": int(probe_label),
                    "pixel_similarity": float(_cosine_similarity_1d(sample_image, probe_image)),
                }
            )
    if not rows:
        raise RuntimeError("No candidate pairs were generated.")
    df_candidates = pd.DataFrame(rows).drop_duplicates(subset=["sample_id", "probe_id"], keep="first").reset_index(drop=True)
    if len(df_candidates) < int(num_bins):
        raise RuntimeError(f"Candidate pool is too small for {num_bins} bins: {len(df_candidates)} rows.")
    df_candidates = _assign_similarity_bins(df_candidates, num_bins=num_bins)
    return _select_balanced_pairs_from_candidates(df_candidates, max_pairs=max_pairs, num_bins=num_bins, seed=seed)


def _build_bin_metadata(df_pairs: pd.DataFrame, num_bins: int) -> tuple[list[dict[str, object]], dict[str, dict[str, float | int | None]]]:
    labels = _bin_labels(num_bins)
    rows: list[dict[str, object]] = []
    mapping: dict[str, dict[str, float | int | None]] = {}
    for label in labels:
        sub = df_pairs[df_pairs["similarity_bin"] == label].copy()
        row = {
            "similarity_bin": str(label),
            "bin_index": int(labels.index(label)),
            "min_similarity": _safe_float(sub["pixel_similarity"].min()) if not sub.empty else None,
            "max_similarity": _safe_float(sub["pixel_similarity"].max()) if not sub.empty else None,
            "bin_center": _safe_float(sub["pixel_similarity"].mean()) if not sub.empty else None,
            "count": int(len(sub)),
        }
        rows.append(row)
        mapping[str(label)] = row
    return rows, mapping


def _prepare_batch_spikes(
    dataset,
    batch_df: pd.DataFrame,
    encoder,
    spec: ExperimentSpec,
    device: torch.device,
    image_cache: Dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_ids = batch_df["sample_id"].astype(int).tolist()
    probe_ids = batch_df["probe_id"].astype(int).tolist()
    unique_sample_ids = list(dict.fromkeys(sample_ids))
    unique_probe_ids = list(dict.fromkeys(probe_ids))
    sample_images = torch.stack(
        [_get_image_from_cache(dataset, idx, image_cache=image_cache) for idx in unique_sample_ids],
        dim=0,
    ).to(device)
    probe_images = torch.stack(
        [_get_image_from_cache(dataset, idx, image_cache=image_cache) for idx in unique_probe_ids],
        dim=0,
    ).to(device)
    with torch.no_grad():
        sample_encoded = encoder.forward(sample_images)[:, : int(spec.sample_steps), ...].contiguous()
        probe_encoded = encoder.forward(probe_images)[:, : int(spec.probe_steps), ...].contiguous()
    sample_lookup = {int(idx): pos for pos, idx in enumerate(unique_sample_ids)}
    probe_lookup = {int(idx): pos for pos, idx in enumerate(unique_probe_ids)}
    sample_select = torch.tensor([sample_lookup[int(idx)] for idx in sample_ids], dtype=torch.long, device=device)
    probe_select = torch.tensor([probe_lookup[int(idx)] for idx in probe_ids], dtype=torch.long, device=device)
    sample_spikes = sample_encoded.index_select(0, sample_select)
    probe_spikes = probe_encoded.index_select(0, probe_select)
    return sample_spikes, probe_spikes


def _run_batched_mode(
    net,
    sample_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    *,
    delay_steps: int,
    stsp_mode: str,
    readout_step: int,
    repeat_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    seed_everything(repeat_seed)
    with torch.no_grad():
        out = run_dms_snapshot_rollout(
            net=net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=int(delay_steps),
            stsp_mode=str(stsp_mode),
            phase_reset=True,
            intervention_plan=None,
            readout_step=int(readout_step),
            snapshot_state_names=("v_mem",),
            record_full_trace_state_names=(),
        )
    pred = out["predictions"]["prediction_probe"].detach().cpu().numpy().astype(np.int64, copy=False)
    voltage_snapshot = out["readout_snapshots"]["layer3"]["v_mem"]
    bundles = extract_class_voltage_scores(
        voltage_snapshot=voltage_snapshot,
        num_classes=int(net.layer3.num_classes),
        neurons_per_class=int(net.layer3.neurons_per_class),
        pooling="top_m_mean",
        m=1,
        backend="dms_voltage_wta",
        readout_step=int(out["readout_step"]),
    )
    voltage_vectors = np.stack([np.asarray(bundle.class_scores, dtype=np.float64) for bundle in bundles], axis=0)
    return pred, voltage_vectors


def _aggregate_prediction(predictions: Sequence[int], mean_voltage: np.ndarray) -> int:
    counts = Counter(int(pred) for pred in predictions)
    max_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == max_count]
    if len(tied) == 1:
        return int(tied[0])
    tie_scores = {
        int(label): float(mean_voltage[int(label)]) if 0 <= int(label) < len(mean_voltage) else -float("inf")
        for label in tied
    }
    return int(max(tie_scores, key=tie_scores.get))


def _compute_bvec(voltage_dynamic: np.ndarray, voltage_static: np.ndarray) -> float:
    v_dyn = np.asarray(voltage_dynamic, dtype=np.float64)
    v_sta = np.asarray(voltage_static, dtype=np.float64)
    v_dyn_centered = v_dyn - float(v_dyn.mean())
    v_sta_centered = v_sta - float(v_sta.mean())
    return float(np.linalg.norm(v_dyn_centered - v_sta_centered, ord=2))


def run_similarity_bias_trials(
    *,
    net,
    encoder,
    dataset,
    df_pairs: pd.DataFrame,
    spec: ExperimentSpec,
    delay_ms: float,
    batch_size: int,
    repeats: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    delay_steps = int(round((float(delay_ms) * ms) / spec.dt))
    readout_step = resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(spec.probe_steps),
        decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )
    image_cache: Dict[int, torch.Tensor] = {}
    repeat_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    voltage_dynamic_rows: list[np.ndarray] = []
    voltage_static_rows: list[np.ndarray] = []

    total_batches = math.ceil(len(df_pairs) / int(batch_size))
    progress = tqdm(total=total_batches, desc="SimilarityBias")
    for start in range(0, len(df_pairs), int(batch_size)):
        batch = df_pairs.iloc[start:start + int(batch_size)].copy().reset_index(drop=True)
        sample_spikes, probe_spikes = _prepare_batch_spikes(
            dataset=dataset,
            batch_df=batch,
            encoder=encoder,
            spec=spec,
            device=device,
            image_cache=image_cache,
        )
        mode_preds: Dict[str, list[np.ndarray]] = {mode: [] for mode in STSP_MODES}
        mode_voltages: Dict[str, list[np.ndarray]] = {mode: [] for mode in STSP_MODES}
        for repeat_idx in range(int(repeats)):
            repeat_seed = mix_seed(seed, 701, repeat_idx)
            for stsp_mode in STSP_MODES:
                pred, voltage_vectors = _run_batched_mode(
                    net=net,
                    sample_spikes=sample_spikes,
                    probe_spikes=probe_spikes,
                    delay_steps=delay_steps,
                    stsp_mode=str(stsp_mode),
                    readout_step=int(readout_step),
                    repeat_seed=repeat_seed,
                )
                mode_preds[str(stsp_mode)].append(pred)
                mode_voltages[str(stsp_mode)].append(voltage_vectors)
            for idx_in_batch, row in enumerate(batch.itertuples(index=False)):
                dyn_vector = np.asarray(mode_voltages["dynamic"][repeat_idx][idx_in_batch], dtype=np.float64)
                sta_vector = np.asarray(mode_voltages["static_frozen"][repeat_idx][idx_in_batch], dtype=np.float64)
                dyn_pred = int(mode_preds["dynamic"][repeat_idx][idx_in_batch])
                sta_pred = int(mode_preds["static_frozen"][repeat_idx][idx_in_batch])
                repeat_rows.append(
                    {
                        "pair_id": int(row.pair_id),
                        "repeat_index": int(repeat_idx),
                        "sample_id": int(row.sample_id),
                        "probe_id": int(row.probe_id),
                        "sample_label": int(row.sample_label),
                        "probe_label": int(row.probe_label),
                        "pixel_similarity": float(row.pixel_similarity),
                        "similarity_bin": str(row.similarity_bin),
                        "pred_label_dynamic": int(dyn_pred),
                        "pred_label_static": int(sta_pred),
                        "correct_dynamic": int(dyn_pred == int(row.probe_label)),
                        "correct_static": int(sta_pred == int(row.probe_label)),
                        "b_vec": float(_compute_bvec(dyn_vector, sta_vector)),
                    }
                )
        for idx_in_batch, row in enumerate(batch.itertuples(index=False)):
            dyn_stack = np.stack([arr[idx_in_batch] for arr in mode_voltages["dynamic"]], axis=0)
            sta_stack = np.stack([arr[idx_in_batch] for arr in mode_voltages["static_frozen"]], axis=0)
            dyn_mean = np.asarray(dyn_stack.mean(axis=0), dtype=np.float64)
            sta_mean = np.asarray(sta_stack.mean(axis=0), dtype=np.float64)
            dyn_pred = _aggregate_prediction([int(pred[idx_in_batch]) for pred in mode_preds["dynamic"]], dyn_mean)
            sta_pred = _aggregate_prediction([int(pred[idx_in_batch]) for pred in mode_preds["static_frozen"]], sta_mean)
            voltage_index = len(voltage_dynamic_rows)
            voltage_dynamic_rows.append(dyn_mean)
            voltage_static_rows.append(sta_mean)
            trial_rows.append(
                {
                    "pair_id": int(row.pair_id),
                    "sample_id": int(row.sample_id),
                    "probe_id": int(row.probe_id),
                    "sample_label": int(row.sample_label),
                    "probe_label": int(row.probe_label),
                    "pixel_similarity": float(row.pixel_similarity),
                    "similarity_bin": str(row.similarity_bin),
                    "repeat_count": int(repeats),
                    "pred_label_dynamic": int(dyn_pred),
                    "pred_label_static": int(sta_pred),
                    "correct_dynamic": int(dyn_pred == int(row.probe_label)),
                    "correct_static": int(sta_pred == int(row.probe_label)),
                    "b_vec": float(_compute_bvec(dyn_mean, sta_mean)),
                    "voltage_vector_index": int(voltage_index),
                }
            )
        progress.update(1)
    progress.close()

    df_trials = pd.DataFrame(trial_rows).sort_values(["pair_id"], kind="stable").reset_index(drop=True)
    df_repeat = pd.DataFrame(repeat_rows).sort_values(["pair_id", "repeat_index"], kind="stable").reset_index(drop=True)
    voltage_payload = {
        "pair_id": df_trials["pair_id"].to_numpy(dtype=np.int64, copy=False),
        "voltage_dynamic": np.stack(voltage_dynamic_rows, axis=0) if voltage_dynamic_rows else np.zeros((0, 0), dtype=np.float64),
        "voltage_static": np.stack(voltage_static_rows, axis=0) if voltage_static_rows else np.zeros((0, 0), dtype=np.float64),
    }
    return df_trials, df_repeat, voltage_payload


def compute_accuracy_summary(df_trials: pd.DataFrame, bin_metadata: Mapping[str, Mapping[str, float | int | None]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bin_label, sub in df_trials.groupby("similarity_bin", sort=False):
        dyn = sub["correct_dynamic"].to_numpy(dtype=np.float64, copy=False)
        sta = sub["correct_static"].to_numpy(dtype=np.float64, copy=False)
        diff = sta - dyn
        meta = dict(bin_metadata[str(bin_label)])
        rows.append(
            {
                "similarity_bin": str(bin_label),
                "bin_index": int(meta["bin_index"]),
                "sim_bin_left": meta["min_similarity"],
                "sim_bin_right": meta["max_similarity"],
                "bin_center": meta["bin_center"],
                "n_trials": int(len(sub)),
                "acc_dynamic": float(dyn.mean()) if len(dyn) > 0 else float("nan"),
                "acc_static": float(sta.mean()) if len(sta) > 0 else float("nan"),
                "acc_drop": float(diff.mean()) if len(diff) > 0 else float("nan"),
                "sem_dynamic": float(_sem(dyn)),
                "sem_static": float(_sem(sta)),
                "sem_acc_drop": float(_sem(diff)),
            }
        )
    return pd.DataFrame(rows).sort_values(["bin_index"], kind="stable").reset_index(drop=True)


def compute_bvec_summary(df_trials: pd.DataFrame, bin_metadata: Mapping[str, Mapping[str, float | int | None]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bin_label, sub in df_trials.groupby("similarity_bin", sort=False):
        values = sub["b_vec"].to_numpy(dtype=np.float64, copy=False)
        meta = dict(bin_metadata[str(bin_label)])
        rows.append(
            {
                "similarity_bin": str(bin_label),
                "bin_index": int(meta["bin_index"]),
                "bin_center": meta["bin_center"],
                "n_trials": int(len(values)),
                "mean_B_vec": float(values.mean()) if len(values) > 0 else float("nan"),
                "std_B_vec": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "sem_B_vec": float(_sem(values)),
            }
        )
    return pd.DataFrame(rows).sort_values(["bin_index"], kind="stable").reset_index(drop=True)


def _prediction_support(df_trials: pd.DataFrame, num_classes: int) -> list[int]:
    support = list(range(int(num_classes)))
    if ((df_trials["pred_label_dynamic"] == -1) | (df_trials["pred_label_static"] == -1)).any():
        return [-1] + support
    return support


def _distribution_over_support(values: np.ndarray, support: Sequence[int]) -> np.ndarray:
    if values.size == 0:
        return np.full(len(support), np.nan, dtype=np.float64)
    return np.array([np.mean(values == int(label)) for label in support], dtype=np.float64)


def compute_cti_summary(df_trials: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    eps = 1e-12
    support = _prediction_support(df_trials, num_classes=num_classes)
    bin_order = {label: idx for idx, label in enumerate(pd.unique(df_trials["similarity_bin"]).tolist())}
    rows: list[dict[str, object]] = []
    for bin_label in bin_order:
        for sample_label in range(int(num_classes)):
            for probe_label in range(int(num_classes)):
                sub = df_trials[
                    (df_trials["similarity_bin"] == str(bin_label))
                    & (df_trials["sample_label"] == int(sample_label))
                    & (df_trials["probe_label"] == int(probe_label))
                ].copy()
                n_trials = int(len(sub))
                if n_trials == 0:
                    cti = capture = capture_ratio = float("nan")
                else:
                    q_dyn = _distribution_over_support(sub["pred_label_dynamic"].to_numpy(dtype=np.int64, copy=False), support)
                    q_sta = _distribution_over_support(sub["pred_label_static"].to_numpy(dtype=np.int64, copy=False), support)
                    cti = 0.5 * float(np.abs(q_dyn - q_sta).sum())
                    sample_idx = support.index(int(sample_label))
                    capture = float(q_dyn[sample_idx] - q_sta[sample_idx])
                    capture_ratio = float(max(capture, 0.0) / (cti + eps))
                rows.append(
                    {
                        "similarity_bin": str(bin_label),
                        "bin_index": int(bin_order[str(bin_label)]),
                        "sample_label": int(sample_label),
                        "probe_label": int(probe_label),
                        "n_trials": int(n_trials),
                        "cti": float(cti),
                        "capture": float(capture),
                        "capture_ratio": float(capture_ratio),
                    }
                )
    return pd.DataFrame(rows).sort_values(["bin_index", "sample_label", "probe_label"], kind="stable").reset_index(drop=True)


def compute_stats_summary(df_trials: pd.DataFrame, df_bvec_summary: pd.DataFrame, df_accuracy_summary: pd.DataFrame) -> dict:
    similarity = df_trials["pixel_similarity"].to_numpy(dtype=np.float64, copy=False)
    bvec = df_trials["b_vec"].to_numpy(dtype=np.float64, copy=False)
    stats_summary: dict[str, object] = {
        "overall_accuracy": {
            "dynamic": float(df_trials["correct_dynamic"].mean()),
            "static": float(df_trials["correct_static"].mean()),
            "acc_drop": float((df_trials["correct_static"] - df_trials["correct_dynamic"]).mean()),
        },
        "bvec_bin_means": {str(row.similarity_bin): float(row.mean_B_vec) for row in df_bvec_summary.itertuples(index=False)},
        "accdrop_bin_means": {str(row.similarity_bin): float(row.acc_drop) for row in df_accuracy_summary.itertuples(index=False)},
    }
    if len(df_trials) >= 3 and np.unique(similarity).size >= 2 and np.unique(bvec).size >= 2:
        spearman_rho, spearman_p = stats.spearmanr(similarity, bvec)
        pearson_r, pearson_p = stats.pearsonr(similarity, bvec)
        stats_summary["spearman_similarity_vs_bvec"] = {
            "rho": float(spearman_rho),
            "p_value": float(spearman_p),
            "n_trials": int(len(df_trials)),
            "status": "ok",
        }
        stats_summary["pearson_similarity_vs_bvec"] = {
            "r": float(pearson_r),
            "p_value": float(pearson_p),
            "n_trials": int(len(df_trials)),
            "status": "ok",
        }
    else:
        stats_summary["spearman_similarity_vs_bvec"] = {"status": "insufficient_samples", "n_trials": int(len(df_trials))}
        stats_summary["pearson_similarity_vs_bvec"] = {"status": "insufficient_samples", "n_trials": int(len(df_trials))}

    low_label = str(df_bvec_summary.iloc[0]["similarity_bin"])
    high_label = str(df_bvec_summary.iloc[-1]["similarity_bin"])
    low_vals = df_trials.loc[df_trials["similarity_bin"] == low_label, "b_vec"].to_numpy(dtype=np.float64, copy=False)
    high_vals = df_trials.loc[df_trials["similarity_bin"] == high_label, "b_vec"].to_numpy(dtype=np.float64, copy=False)
    if len(low_vals) >= 2 and len(high_vals) >= 2:
        test = stats.mannwhitneyu(low_vals, high_vals, alternative="two-sided")
        stats_summary["low_vs_high_bvec"] = {
            "low_bin": low_label,
            "high_bin": high_label,
            "low_mean": float(low_vals.mean()),
            "high_mean": float(high_vals.mean()),
            "n_low": int(len(low_vals)),
            "n_high": int(len(high_vals)),
            "u_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "status": "ok",
        }
    else:
        stats_summary["low_vs_high_bvec"] = {
            "low_bin": low_label,
            "high_bin": high_label,
            "n_low": int(len(low_vals)),
            "n_high": int(len(high_vals)),
            "status": "insufficient_samples",
        }
    return stats_summary


def compute_pairwise_overlap_metrics(
    df_pairs: pd.DataFrame,
    dataset,
    *,
    eps: float = 1e-12,
) -> pd.DataFrame:
    image_cache: Dict[int, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for row in df_pairs.itertuples(index=False):
        sample_image = _get_image_from_cache(dataset, int(row.sample_id), image_cache=image_cache)
        probe_image = _get_image_from_cache(dataset, int(row.probe_id), image_cache=image_cache)
        sample_fg = sample_image > 0
        probe_fg = probe_image > 0
        sample_fg_area = int(torch.count_nonzero(sample_fg).item())
        probe_fg_area = int(torch.count_nonzero(probe_fg).item())
        overlap_area = int(torch.count_nonzero(sample_fg & probe_fg).item())
        union_area = int(torch.count_nonzero(sample_fg | probe_fg).item())
        dice_overlap = float((2.0 * overlap_area) / (sample_fg_area + probe_fg_area + float(eps)))
        rows.append(
            {
                "pair_id": int(row.pair_id),
                "sample_fg_area": int(sample_fg_area),
                "probe_fg_area": int(probe_fg_area),
                "overlap_area": int(overlap_area),
                "union_area": int(union_area),
                "dice_overlap": float(dice_overlap),
            }
        )
    return pd.DataFrame(rows).sort_values(["pair_id"], kind="stable").reset_index(drop=True)


def build_within_bin_overlap_matches(
    df_within_bin: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | None]]:
    grouped = df_within_bin.copy().sort_values(["pixel_similarity", "pair_id"], kind="stable").reset_index(drop=True)
    if grouped.empty:
        grouped["overlap_group"] = pd.Series(dtype="object")
        return grouped, pd.DataFrame(
            columns=[
                "matched_pair_index",
                "pair_id_high",
                "pair_id_low",
                "correct_dynamic_high",
                "correct_dynamic_low",
                "correct_static_high",
                "correct_static_low",
                "pixel_similarity_high",
                "pixel_similarity_low",
                "dice_overlap_high",
                "dice_overlap_low",
                "b_vec_high",
                "b_vec_low",
                "delta_b_vec",
                "acc_drop_high",
                "acc_drop_low",
            ]
        ), {
            "median_dice_overlap": None,
            "n_high_overlap": 0,
            "n_low_overlap": 0,
        }

    median_overlap = float(grouped["dice_overlap"].median())
    grouped["overlap_group"] = np.where(grouped["dice_overlap"] >= median_overlap, "high-overlap", "low-overlap")
    high_df = grouped[grouped["overlap_group"] == "high-overlap"].copy().reset_index(drop=True)
    low_df = grouped[grouped["overlap_group"] == "low-overlap"].copy().reset_index(drop=True)

    matched_rows: list[dict[str, object]] = []
    if not high_df.empty and not low_df.empty:
        high_df = high_df.sort_values(["pixel_similarity", "pair_id"], kind="stable").reset_index(drop=True)
        low_df = low_df.sort_values(["pixel_similarity", "pair_id"], kind="stable").reset_index(drop=True)
        if len(high_df) <= len(low_df):
            anchor_df = high_df
            pool_df = low_df.copy()
            anchor_is_high = True
        else:
            anchor_df = low_df
            pool_df = high_df.copy()
            anchor_is_high = False

        for anchor in anchor_df.itertuples(index=False):
            if pool_df.empty:
                break
            candidates = pool_df.copy()
            candidates["_similarity_gap"] = np.abs(
                candidates["pixel_similarity"].to_numpy(dtype=np.float64) - float(anchor.pixel_similarity)
            )
            best_idx = candidates.sort_values(["_similarity_gap", "pair_id"], kind="stable").index[0]
            matched = pool_df.loc[best_idx]
            pool_df = pool_df.drop(index=best_idx).copy()

            if anchor_is_high:
                high_row = anchor
                low_row = matched
            else:
                high_row = matched
                low_row = anchor

            matched_rows.append(
                {
                    "matched_pair_index": int(len(matched_rows)),
                    "pair_id_high": int(high_row.pair_id),
                    "pair_id_low": int(low_row.pair_id),
                    "correct_dynamic_high": int(high_row.correct_dynamic),
                    "correct_dynamic_low": int(low_row.correct_dynamic),
                    "correct_static_high": int(high_row.correct_static),
                    "correct_static_low": int(low_row.correct_static),
                    "pixel_similarity_high": float(high_row.pixel_similarity),
                    "pixel_similarity_low": float(low_row.pixel_similarity),
                    "dice_overlap_high": float(high_row.dice_overlap),
                    "dice_overlap_low": float(low_row.dice_overlap),
                    "b_vec_high": float(high_row.b_vec),
                    "b_vec_low": float(low_row.b_vec),
                    "delta_b_vec": float(high_row.b_vec - low_row.b_vec),
                    "acc_drop_high": float(high_row.acc_drop),
                    "acc_drop_low": float(low_row.acc_drop),
                }
            )

    matched_df = pd.DataFrame(matched_rows)
    if matched_df.empty:
        matched_df = pd.DataFrame(
            columns=[
                "matched_pair_index",
                "pair_id_high",
                "pair_id_low",
                "correct_dynamic_high",
                "correct_dynamic_low",
                "correct_static_high",
                "correct_static_low",
                "pixel_similarity_high",
                "pixel_similarity_low",
                "dice_overlap_high",
                "dice_overlap_low",
                "b_vec_high",
                "b_vec_low",
                "delta_b_vec",
                "acc_drop_high",
                "acc_drop_low",
            ]
        )
    else:
        matched_df = matched_df.sort_values(["matched_pair_index"], kind="stable").reset_index(drop=True)

    metadata = {
        "median_dice_overlap": float(median_overlap),
        "n_high_overlap": int(len(high_df)),
        "n_low_overlap": int(len(low_df)),
    }
    return grouped, matched_df, metadata


def compute_within_bin_overlap_summary(
    df_within_bin: pd.DataFrame,
    df_matched: pd.DataFrame,
    *,
    target_bin_label: str,
    target_bin_index: int,
    match_metadata: Mapping[str, float | int | None],
) -> dict[str, object]:
    n_pairs = int(len(df_matched))
    delta = df_matched["delta_b_vec"].to_numpy(dtype=np.float64, copy=False) if n_pairs > 0 else np.zeros(0, dtype=np.float64)
    acc_drop_high = (
        df_matched["acc_drop_high"].to_numpy(dtype=np.float64, copy=False)
        if n_pairs > 0
        else np.zeros(0, dtype=np.float64)
    )
    acc_drop_low = (
        df_matched["acc_drop_low"].to_numpy(dtype=np.float64, copy=False)
        if n_pairs > 0
        else np.zeros(0, dtype=np.float64)
    )
    if n_pairs == 0:
        wilcoxon_statistic = None
        wilcoxon_p_value = None
    elif np.allclose(delta, 0.0):
        wilcoxon_statistic = 0.0
        wilcoxon_p_value = 1.0
    else:
        test = stats.wilcoxon(
            df_matched["b_vec_high"].to_numpy(dtype=np.float64, copy=False),
            df_matched["b_vec_low"].to_numpy(dtype=np.float64, copy=False),
            alternative="two-sided",
        )
        wilcoxon_statistic = float(test.statistic)
        wilcoxon_p_value = float(test.pvalue)

    status = "ok" if n_pairs >= 8 else "low_sample_size"
    summary = {
        "target_bin_label": str(target_bin_label),
        "target_bin_index": int(target_bin_index),
        "n_total_in_bin": int(len(df_within_bin)),
        "n_high_overlap": int(match_metadata.get("n_high_overlap", 0) or 0),
        "n_low_overlap": int(match_metadata.get("n_low_overlap", 0) or 0),
        "n_matched_pairs": int(n_pairs),
        "mean_similarity_high": _safe_float(df_matched["pixel_similarity_high"].mean()) if n_pairs > 0 else None,
        "mean_similarity_low": _safe_float(df_matched["pixel_similarity_low"].mean()) if n_pairs > 0 else None,
        "mean_overlap_high": _safe_float(df_matched["dice_overlap_high"].mean()) if n_pairs > 0 else None,
        "mean_overlap_low": _safe_float(df_matched["dice_overlap_low"].mean()) if n_pairs > 0 else None,
        "mean_bvec_high": _safe_float(df_matched["b_vec_high"].mean()) if n_pairs > 0 else None,
        "mean_bvec_low": _safe_float(df_matched["b_vec_low"].mean()) if n_pairs > 0 else None,
        "mean_delta_bvec": _safe_float(delta.mean()) if n_pairs > 0 else None,
        "sem_delta_bvec": _safe_float(_sem(delta)) if n_pairs > 0 else None,
        "acc_dynamic_high": _safe_float(df_matched["correct_dynamic_high"].mean()) if n_pairs > 0 else None,
        "acc_dynamic_low": _safe_float(df_matched["correct_dynamic_low"].mean()) if n_pairs > 0 else None,
        "acc_static_high": _safe_float(df_matched["correct_static_high"].mean()) if n_pairs > 0 else None,
        "acc_static_low": _safe_float(df_matched["correct_static_low"].mean()) if n_pairs > 0 else None,
        "acc_drop_high": _safe_float(df_matched["correct_static_high"].mean() - df_matched["correct_dynamic_high"].mean()) if n_pairs > 0 else None,
        "acc_drop_low": _safe_float(df_matched["correct_static_low"].mean() - df_matched["correct_dynamic_low"].mean()) if n_pairs > 0 else None,
        "sem_acc_drop_high": _safe_float(_sem(acc_drop_high)) if n_pairs > 0 else None,
        "sem_acc_drop_low": _safe_float(_sem(acc_drop_low)) if n_pairs > 0 else None,
        "wilcoxon_statistic": _safe_float(wilcoxon_statistic),
        "wilcoxon_p_value": _safe_float(wilcoxon_p_value),
        "median_dice_overlap": match_metadata.get("median_dice_overlap"),
        "status": status,
        "interpretation": (
            "Within the highest similarity bin, pairs with larger sample-probe overlap still show stronger bias "
            "than similarity-matched low-overlap pairs, supporting overlap as the mechanistic bridge from "
            "similarity to decision bias."
        ),
    }
    return summary


def plot_within_bin_overlap_bridge(df_matched: pd.DataFrame, summary: Mapping[str, object]) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_COMPACT)

    low_value = summary.get("acc_drop_low")
    high_value = summary.get("acc_drop_high")
    if low_value is not None and high_value is not None:
        x = np.arange(2, dtype=np.float64)
        heights = np.array([float(low_value), float(high_value)], dtype=np.float64)
        yerr = np.array(
            [
                0.0 if summary.get("sem_acc_drop_low") is None else float(summary["sem_acc_drop_low"]),
                0.0 if summary.get("sem_acc_drop_high") is None else float(summary["sem_acc_drop_high"]),
            ],
            dtype=np.float64,
        )
        ax.bar(
            x,
            heights,
            yerr=yerr,
            width=0.58,
            color=[COLOR_ACCENT_RED, COLOR_ACCENT_GREEN],
            edgecolor="black",
            linewidth=LINE_WIDTH_REFERENCE,
            alpha=0.82,
            capsize=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["Low-overlap", "High-overlap"])
    else:
        ax.text(0.5, 0.5, "No matched pairs", ha="center", va="center", transform=ax.transAxes)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=LINE_WIDTH_REFERENCE)
    ax.set_ylabel("Accuracy drop (static - dynamic)")
    ax.grid(axis="y", alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def _heatmap_matrix(df: pd.DataFrame, value_column: str, num_classes: int, bin_label: str) -> np.ndarray:
    subset = df[df["similarity_bin"] == str(bin_label)].copy()
    matrix = np.full((int(num_classes), int(num_classes)), np.nan, dtype=np.float64)
    for row in subset.itertuples(index=False):
        matrix[int(row.sample_label), int(row.probe_label)] = float(getattr(row, value_column))
    return matrix


def plot_similarity_histogram(df_trials: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_COMPACT)
    values = df_trials["pixel_similarity"].to_numpy(dtype=np.float64, copy=False)
    ax.hist(values, bins=min(40, max(10, int(math.sqrt(len(values))))), color=COLOR_ACCENT_BLUE, edgecolor="white", alpha=ALPHA_BAR)
    ax.set_xlabel("Pixel cosine similarity")
    ax.set_ylabel("Trial count")
    ax.set_title("Sample-probe pixel similarity distribution")
    ax.grid(axis="y", alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def plot_accuracy_curves_vs_similarity(df_accuracy: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_COMPACT)
    x = np.arange(len(df_accuracy), dtype=np.float64)
    labels = df_accuracy["similarity_bin"].astype(str).tolist()
    ax.errorbar(
        x,
        df_accuracy["acc_dynamic"].to_numpy(dtype=np.float64),
        yerr=df_accuracy["sem_dynamic"].to_numpy(dtype=np.float64),
        marker=MARKER_CIRCLE,
        linewidth=LINE_WIDTH_PRIMARY,
        color=COLOR_DYNAMIC,
        label="Dynamic",
    )
    ax.errorbar(
        x,
        df_accuracy["acc_static"].to_numpy(dtype=np.float64),
        yerr=df_accuracy["sem_static"].to_numpy(dtype=np.float64),
        marker=MARKER_CIRCLE,
        linewidth=LINE_WIDTH_PRIMARY,
        color=COLOR_STATIC,
        label="Static",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Similarity bin")
    ax.set_ylabel("Probe accuracy")
    ax.grid(alpha=GRID_ALPHA_SOFT)
    apply_standard_legend(ax)
    fig.tight_layout()
    return fig


def plot_accuracy_drop_vs_similarity(df_accuracy: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_COMPACT)
    x = np.arange(len(df_accuracy), dtype=np.float64)
    labels = df_accuracy["similarity_bin"].astype(str).tolist()
    ax.errorbar(
        x,
        df_accuracy["acc_drop"].to_numpy(dtype=np.float64),
        yerr=df_accuracy["sem_acc_drop"].to_numpy(dtype=np.float64),
        marker=MARKER_CIRCLE,
        linewidth=LINE_WIDTH_PRIMARY,
        color=COLOR_ACCENT_SKY,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=LINE_WIDTH_REFERENCE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Similarity bin")
    ax.set_ylabel("AccDrop = static - dynamic")
    ax.grid(alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def plot_cti_heatmaps(df_cti: pd.DataFrame, num_classes: int) -> plt.Figure:
    apply_publication_style()
    low_label = str(df_cti["similarity_bin"].iloc[0])
    high_label = str(df_cti["similarity_bin"].iloc[-1])
    low_matrix = _heatmap_matrix(df_cti, "cti", num_classes, low_label)
    high_matrix = _heatmap_matrix(df_cti, "cti", num_classes, high_label)
    capture_matrix = _heatmap_matrix(df_cti, "capture_ratio", num_classes, high_label)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL_HEATMAP)

    cti_values = np.concatenate([low_matrix[np.isfinite(low_matrix)], high_matrix[np.isfinite(high_matrix)]])
    cti_max = float(np.max(cti_values)) if cti_values.size > 0 else 1.0
    cti_norm = Normalize(vmin=0.0, vmax=max(cti_max, 1e-6))
    for ax, matrix, title in [
        (axes[0], low_matrix, f"CTI heatmap ({low_label})"),
        (axes[1], high_matrix, f"CTI heatmap ({high_label})"),
    ]:
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(color=COLOR_OFFWHITE)
        im = ax.imshow(matrix, cmap=cmap, origin="upper", norm=cti_norm)
        ax.set_title(title)
        ax.set_xlabel("Probe label")
        ax.set_ylabel("Sample label")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=LINE_WIDTH_GUIDE)
        ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=axes[:2], fraction=0.03, pad=0.02)
    cbar.set_label("CTI")

    capture_cmap = plt.get_cmap("viridis").copy()
    capture_cmap.set_bad(color=COLOR_OFFWHITE)
    im_capture = axes[2].imshow(capture_matrix, cmap=capture_cmap, origin="upper", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"CaptureRatio heatmap ({high_label})")
    axes[2].set_xlabel("Probe label")
    axes[2].set_ylabel("Sample label")
    axes[2].set_xticks(range(num_classes))
    axes[2].set_yticks(range(num_classes))
    axes[2].set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
    axes[2].set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
    axes[2].grid(which="minor", color="white", linewidth=LINE_WIDTH_GUIDE)
    axes[2].tick_params(which="minor", bottom=False, left=False)
    cbar2 = fig.colorbar(im_capture, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label("CaptureRatio")
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.11, top=0.90, wspace=0.42)
    return fig


def plot_bvec_vs_similarity(df_trials: pd.DataFrame, df_bvec: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    x = df_trials["pixel_similarity"].to_numpy(dtype=np.float64, copy=False)
    y = df_trials["b_vec"].to_numpy(dtype=np.float64, copy=False)
    axes[0].scatter(x, y, s=18, alpha=0.22, color=COLOR_ACCENT_BLUE, edgecolor="none")
    if len(df_trials) >= 3 and np.unique(x).size >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(float(x.min()), float(x.max()), 200)
        axes[0].plot(x_line, slope * x_line + intercept, color=COLOR_ACCENT_RED, linewidth=2.2, label="Linear trend")
    bin_x = df_trials.groupby("similarity_bin", sort=False)["pixel_similarity"].mean().reindex(df_bvec["similarity_bin"]).to_numpy(dtype=np.float64)
    axes[0].errorbar(
        bin_x,
        df_bvec["mean_B_vec"].to_numpy(dtype=np.float64),
        yerr=df_bvec["sem_B_vec"].to_numpy(dtype=np.float64),
        marker=MARKER_CIRCLE,
        markersize=7,
        linewidth=LINE_WIDTH_SECONDARY,
        color="black",
        label="Bin mean ± SEM",
    )
    axes[0].set_xlabel("Pixel similarity")
    axes[0].set_ylabel("B_vec")
    axes[0].set_title("B_vec increases with pixel similarity")
    axes[0].grid(alpha=GRID_ALPHA_SOFT)
    apply_standard_legend(axes[0])

    x_bins = np.arange(len(df_bvec), dtype=np.float64)
    axes[1].errorbar(
        x_bins,
        df_bvec["mean_B_vec"].to_numpy(dtype=np.float64),
        yerr=df_bvec["sem_B_vec"].to_numpy(dtype=np.float64),
        marker=MARKER_CIRCLE,
        linewidth=LINE_WIDTH_PRIMARY,
        color=COLOR_ACCENT_RED,
    )
    axes[1].set_xticks(x_bins)
    axes[1].set_xticklabels(df_bvec["similarity_bin"].astype(str).tolist())
    axes[1].set_xlabel("Similarity bin")
    axes[1].set_ylabel("Mean B_vec")
    axes[1].set_title("Bin-averaged B_vec")
    axes[1].grid(alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def plot_metric_summary(df_accuracy: pd.DataFrame, df_cti: pd.DataFrame, df_bvec: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL_WIDE)
    x = np.arange(len(df_accuracy), dtype=np.float64)
    labels = df_accuracy["similarity_bin"].astype(str).tolist()
    mean_cti = (
        df_cti[df_cti["n_trials"] > 0]
        .groupby("similarity_bin", sort=False)["cti"]
        .mean()
        .reindex(labels)
        .to_numpy(dtype=np.float64)
    )
    axes[0].plot(x, df_accuracy["acc_drop"].to_numpy(dtype=np.float64), marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=COLOR_ACCENT_SKY)
    axes[0].set_title("AccDrop")
    axes[0].set_ylabel("Static - dynamic")
    axes[1].plot(x, mean_cti, marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=COLOR_ACCENT_PURPLE)
    axes[1].set_title("Mean CTI")
    axes[1].set_ylabel("CTI")
    axes[2].errorbar(
        x,
        df_bvec["mean_B_vec"].to_numpy(dtype=np.float64),
        yerr=df_bvec["sem_B_vec"].to_numpy(dtype=np.float64),
        marker=MARKER_CIRCLE,
        linewidth=LINE_WIDTH_PRIMARY,
        color=COLOR_ACCENT_RED,
    )
    axes[2].set_title("Mean B_vec")
    axes[2].set_ylabel("B_vec")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Similarity bin")
        ax.grid(alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample-probe similarity vs STSP-induced bias strength experiment.")
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--delay-ms", type=float, default=DEFAULT_DELAY_MS)
    parser.add_argument("--sample-ms", type=float, default=DEFAULT_SAMPLE_MS)
    parser.add_argument("--probe-ms", type=float, default=DEFAULT_PROBE_MS)
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEAT_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    global ms, torch, build_mnist_skeleton_loader, build_class_index
    global load_model_and_encoder, run_dms_snapshot_rollout
    global resolve_device, seed_everything, extract_class_voltage_scores, resolve_readout_step

    import torch

    from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
    from src.config.units import ms
    from src.experiments.common.dataset import build_class_index
    from src.experiments.common.model_io import load_model_and_encoder
    from src.experiments.common.monitored_dms import run_dms_snapshot_rollout
    from src.experiments.common.runtime import resolve_device, seed_everything
    from src.experiments.common.voltage_readout import (
        extract_class_voltage_scores,
        resolve_readout_step,
    )
    if args.num_bins <= 1:
        raise ValueError("--num-bins must be greater than 1.")
    if args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive.")
    if args.max_samples is not None and int(args.max_samples) <= 1:
        raise ValueError("--max-samples must be > 1 when provided.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    spec = ExperimentSpec(dt=1.0 * ms, sample_ms=float(args.sample_ms), probe_ms=float(args.probe_ms))
    if spec.sample_steps <= 0 or spec.probe_steps <= 0:
        raise ValueError("sample/probe duration must resolve to positive steps.")

    layout = prepare_result_layout(args.output_dir)
    result_root = layout.root
    metrics_dir = layout.data_dir
    figures_dir = layout.figure_dir
    logs_dir = layout.log_dir

    dataset = _load_dataset(dataset_root=args.dataset_root, split=args.split)
    num_classes = len({int(dataset[idx][1]) for idx in range(len(dataset))})
    class_index = build_class_index(dataset, num_classes=num_classes)
    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(float(args.sample_ms), float(args.probe_ms)),
    )
    df_pairs = build_pair_specs(
        dataset=dataset,
        class_index=class_index,
        num_bins=int(args.num_bins),
        max_pairs=int(args.max_pairs),
        max_samples=args.max_samples,
        seed=int(args.seed),
    )
    bin_rows, bin_mapping = _build_bin_metadata(df_pairs=df_pairs, num_bins=int(args.num_bins))

    df_trials, df_repeat, voltage_payload = run_similarity_bias_trials(
        net=net,
        encoder=encoder,
        dataset=dataset,
        df_pairs=df_pairs,
        spec=spec,
        delay_ms=float(args.delay_ms),
        batch_size=int(args.batch_size),
        repeats=int(args.repeats),
        device=device,
        seed=int(args.seed),
    )

    df_accuracy = compute_accuracy_summary(df_trials=df_trials, bin_metadata=bin_mapping)
    df_bvec = compute_bvec_summary(df_trials=df_trials, bin_metadata=bin_mapping)
    df_cti = compute_cti_summary(df_trials=df_trials, num_classes=int(num_classes))
    stats_summary = compute_stats_summary(df_trials=df_trials, df_bvec_summary=df_bvec, df_accuracy_summary=df_accuracy)

    target_bin_index = max(int(row["bin_index"]) for row in bin_rows)
    target_bin_label = next(str(row["similarity_bin"]) for row in bin_rows if int(row["bin_index"]) == target_bin_index)
    df_within_bin_trials = df_trials[df_trials["similarity_bin"] == target_bin_label].copy().reset_index(drop=True)
    df_within_bin_trials["acc_drop"] = (
        df_within_bin_trials["correct_static"].to_numpy(dtype=np.float64)
        - df_within_bin_trials["correct_dynamic"].to_numpy(dtype=np.float64)
    )
    df_within_bin_pairs = df_pairs[df_pairs["similarity_bin"] == target_bin_label].copy().reset_index(drop=True)
    df_overlap_metrics = compute_pairwise_overlap_metrics(df_pairs=df_within_bin_pairs, dataset=dataset)
    df_within_bin_trials = df_within_bin_trials.merge(df_overlap_metrics, on="pair_id", how="left", validate="one_to_one")
    df_within_bin_trials, df_overlap_matched, overlap_match_metadata = build_within_bin_overlap_matches(
        df_within_bin=df_within_bin_trials
    )
    overlap_summary = compute_within_bin_overlap_summary(
        df_within_bin=df_within_bin_trials,
        df_matched=df_overlap_matched,
        target_bin_label=target_bin_label,
        target_bin_index=target_bin_index,
        match_metadata=overlap_match_metadata,
    )
    stats_summary["within_bin_overlap_bridge"] = overlap_summary

    trial_csv = save_tidy_csv(df_trials, metrics_dir / "trial_results.csv", sort_by=["pair_id"])
    repeat_csv = None
    if int(args.repeats) > 1:
        repeat_csv = save_tidy_csv(df_repeat, metrics_dir / "repeat_level_results.csv", sort_by=["pair_id", "repeat_index"])
    accuracy_csv = save_tidy_csv(df_accuracy, metrics_dir / "bin_accuracy_summary.csv", sort_by=["bin_index"])
    cti_csv = save_tidy_csv(df_cti, metrics_dir / "cti_summary.csv", sort_by=["bin_index", "sample_label", "probe_label"])
    bvec_csv = save_tidy_csv(df_bvec, metrics_dir / "bvec_summary.csv", sort_by=["bin_index"])
    overlap_matched_csv = save_tidy_csv(
        df_overlap_matched,
        metrics_dir / "within_bin_overlap_matched_pairs.csv",
        sort_by=["matched_pair_index"],
    )

    voltage_npz = metrics_dir / "trial_voltage_vectors.npz"
    np.savez(
        voltage_npz,
        pair_id=voltage_payload["pair_id"],
        voltage_dynamic=voltage_payload["voltage_dynamic"],
        voltage_static=voltage_payload["voltage_static"],
    )
    bin_json = _save_json(
        {
            "num_bins": int(args.num_bins),
            "bin_labels": _bin_labels(int(args.num_bins)),
            "bins": bin_rows,
        },
        metrics_dir / "similarity_bin_edges.json",
    )
    stats_json = _save_json(stats_summary, logs_dir / "stats_summary.json")
    overlap_summary_json = _save_json(overlap_summary, logs_dir / "within_bin_overlap_summary.json")

    empty_paths = {"png": "", "pdf": "", "svg": ""}
    fig1_paths = empty_paths.copy()
    fig2_supp_paths = empty_paths.copy()
    fig2_paths = empty_paths.copy()
    fig3_paths = empty_paths.copy()
    fig4_paths = empty_paths.copy()
    fig5_paths = empty_paths.copy()
    fig6_paths = empty_paths.copy()
    if not bool(args.skip_figures):
        fig1 = plot_similarity_histogram(df_trials=df_trials)
        fig1_paths = save_figure_all_formats(fig1, figures_dir / "figure_1_similarity_histogram")
        plt.close(fig1)

        fig2_supp = plot_accuracy_curves_vs_similarity(df_accuracy=df_accuracy)
        fig2_supp_paths = save_figure_all_formats(fig2_supp, figures_dir / "supplementary_accuracy_vs_similarity")
        plt.close(fig2_supp)

        fig2 = plot_accuracy_drop_vs_similarity(df_accuracy=df_accuracy)
        fig2_paths = save_figure_all_formats(fig2, figures_dir / "figure_2_accuracy_vs_similarity")
        plt.close(fig2)

        fig3 = plot_cti_heatmaps(df_cti=df_cti, num_classes=int(num_classes))
        fig3_paths = save_figure_all_formats(fig3, figures_dir / "figure_3_cti_heatmaps")
        plt.close(fig3)

        fig4 = plot_bvec_vs_similarity(df_trials=df_trials, df_bvec=df_bvec)
        fig4_paths = save_figure_all_formats(fig4, figures_dir / "figure_4_bvec_vs_similarity")
        plt.close(fig4)

        fig5 = plot_metric_summary(df_accuracy=df_accuracy, df_cti=df_cti, df_bvec=df_bvec)
        fig5_paths = save_figure_all_formats(fig5, figures_dir / "figure_5_metric_summary")
        plt.close(fig5)

        fig6 = plot_within_bin_overlap_bridge(df_matched=df_overlap_matched, summary=overlap_summary)
        fig6_paths = save_figure_all_formats(fig6, figures_dir / "figure_6_within_bin_overlap_bridge")
        plt.close(fig6)

    run_config_path = save_run_config(
        {
            "model_path": str(Path(args.model_path).resolve()),
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "split": str(args.split),
            "output_dir": str(result_root.resolve()),
            "device": str(device),
            "seed": int(args.seed),
            "config_argument": args.config,
            "delay_ms": float(args.delay_ms),
            "sample_ms": float(args.sample_ms),
            "probe_ms": float(args.probe_ms),
            "num_bins": int(args.num_bins),
            "max_pairs": int(args.max_pairs),
            "max_samples_per_class": None if args.max_samples is None else int(args.max_samples),
            "batch_size": int(args.batch_size),
            "repeats": int(args.repeats),
            "skip_figures": bool(args.skip_figures),
            "num_classes": int(num_classes),
            "candidate_multiplier": int(DEFAULT_CANDIDATE_MULTIPLIER),
            "assumptions": {
                "config_handling": "--config is recorded only; no config system is introduced.",
                "dataset_loader": "MNIST skeleton loader with split in {train,test}.",
                "static_vs_dynamic": "Same checkpoint evaluated under static_frozen and dynamic STSP modes.",
                "voltage_readout": "Layer-3 pre-decision v_mem class-score vector using decision_offset and top_m_mean(m=1).",
                "cti_support": "CTI uses labels 0..C-1 and includes silent bucket -1 internally only when silent predictions occur.",
            },
            "outputs": {
                "trial_results_csv": str(Path(trial_csv).resolve()),
                "repeat_level_results_csv": None if repeat_csv is None else str(Path(repeat_csv).resolve()),
                "bin_accuracy_summary_csv": str(Path(accuracy_csv).resolve()),
                "cti_summary_csv": str(Path(cti_csv).resolve()),
                "bvec_summary_csv": str(Path(bvec_csv).resolve()),
                "within_bin_overlap_matched_pairs_csv": str(Path(overlap_matched_csv).resolve()),
                "trial_voltage_vectors_npz": str(voltage_npz.resolve()),
                "similarity_bin_edges_json": str(bin_json.resolve()),
                "stats_summary_json": str(stats_json.resolve()),
                "within_bin_overlap_summary_json": str(overlap_summary_json.resolve()),
                "supplementary_accuracy_vs_similarity_png": fig2_supp_paths["png"],
                "figure_1_png": fig1_paths["png"],
                "figure_2_png": fig2_paths["png"],
                "figure_3_png": fig3_paths["png"],
                "figure_4_png": fig4_paths["png"],
                "figure_5_png": fig5_paths["png"],
                "figure_6_png": fig6_paths["png"],
            },
        },
        result_root,
    )
    summary_path = save_summary_json(
        {
            "experiment": "similarity_bias_experiment",
            "overall_acc_dynamic": float(df_trials["correct_dynamic"].mean()),
            "overall_acc_static": float(df_trials["correct_static"].mean()),
            "overall_drop": float((df_trials["correct_static"] - df_trials["correct_dynamic"]).mean()),
            "artifact_stats_summary_json": str(stats_json.resolve()),
            "artifact_within_bin_overlap_summary_json": str(overlap_summary_json.resolve()),
            "run_config_json": str(Path(run_config_path).resolve()),
        },
        result_root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=similarity_bias_experiment",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"seed={int(args.seed)}",
            f"trials={len(df_trials)}",
            f"result_root={result_root.resolve()}",
            f"summary_json={Path(summary_path).resolve()}",
        ],
        logs_dir,
    )

    overall_acc_dyn = float(df_trials["correct_dynamic"].mean())
    overall_acc_sta = float(df_trials["correct_static"].mean())
    overall_drop = float((df_trials["correct_static"] - df_trials["correct_dynamic"]).mean())
    spearman_summary = stats_summary["spearman_similarity_vs_bvec"]
    low_high = stats_summary["low_vs_high_bvec"]
    print("\n=== Similarity Bias Experiment Summary ===")
    print(f"Dynamic accuracy: {overall_acc_dyn:.4f}")
    print(f"Static accuracy: {overall_acc_sta:.4f}")
    print(f"AccDrop (static - dynamic): {overall_drop:.4f}")
    if spearman_summary.get("status") == "ok":
        print(
            "Similarity vs B_vec Spearman: "
            f"rho={float(spearman_summary['rho']):.4f}, p={float(spearman_summary['p_value']):.4g}"
        )
    else:
        print("Similarity vs B_vec Spearman: insufficient samples")
    if low_high.get("status") == "ok":
        print(
            f"Low vs high B_vec means: {float(low_high['low_mean']):.4f} vs {float(low_high['high_mean']):.4f} "
            f"(p={float(low_high['p_value']):.4g})"
        )
    else:
        print("Low vs high B_vec comparison: insufficient samples")
    print(
        "Within-bin overlap bridge: "
        f"n={int(overlap_summary['n_matched_pairs'])}, "
        f"delta_B_vec={float(overlap_summary['mean_delta_bvec']) if overlap_summary['mean_delta_bvec'] is not None else float('nan'):.4f}, "
        f"p={float(overlap_summary['wilcoxon_p_value']) if overlap_summary['wilcoxon_p_value'] is not None else float('nan'):.4g}, "
        f"status={overlap_summary['status']}"
    )
    print(f"Saved outputs under: {result_root.resolve()}")
    print(f"Saved: {trial_csv}")
    print(f"Saved: {accuracy_csv}")
    print(f"Saved: {cti_csv}")
    print(f"Saved: {bvec_csv}")
    print(f"Saved: {overlap_matched_csv}")
    print(f"Saved: {voltage_npz}")
    print(f"Saved: {stats_json}")
    print(f"Saved: {overlap_summary_json}")
    print(f"Saved: {run_config_path}")


if __name__ == "__main__":
    main()
