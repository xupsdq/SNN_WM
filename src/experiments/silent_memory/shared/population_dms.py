from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.decoding import decode_accuracy_with_splits
from src.experiments.common.model_io import load_model_and_encoder

LAYER_KEYS = ["layer1", "layer2", "layer3"]
STSP_MODES = ["dynamic", "static_frozen"]


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay_ms: float
    probe_ms: float
    phase_reset: bool

    @property
    def sample_steps(self) -> int:
        return int((self.sample_ms * ms) / self.dt)

    @property
    def delay_steps(self) -> int:
        return int((self.delay_ms * ms) / self.dt)

    @property
    def probe_steps(self) -> int:
        return int((self.probe_ms * ms) / self.dt)

    @property
    def total_steps(self) -> int:
        return self.sample_steps + self.delay_steps + self.probe_steps


def generate_balanced_dms_trial_specs(
    class_index: Dict[int, List[int]],
    num_trials: int,
    num_classes: int,
    rng: random.Random,
) -> pd.DataFrame:
    sample_labels = [i % num_classes for i in range(num_trials)]
    rng.shuffle(sample_labels)
    rows: List[Dict[str, int]] = []
    all_classes = list(range(num_classes))
    for trial_id, sample_label in enumerate(sample_labels):
        probe_candidates = [c for c in all_classes if c != sample_label]
        probe_label = rng.choice(probe_candidates)
        sample_index = rng.choice(class_index[sample_label])
        probe_index = rng.choice(class_index[probe_label])
        rows.append(
            {
                "trial_id": int(trial_id),
                "sample_label": int(sample_label),
                "probe_label": int(probe_label),
                "sample_index": int(sample_index),
                "probe_index": int(probe_index),
            }
        )
    return pd.DataFrame(rows)


def validate_trial_specs(df_specs: pd.DataFrame, num_classes: int) -> None:
    if df_specs["trial_id"].nunique() != len(df_specs):
        raise ValueError("trial_id must be unique")

    for col in ["sample_label", "probe_label"]:
        vals = df_specs[col].to_numpy(dtype=np.int64)
        if (vals < 0).any() or (vals >= num_classes).any():
            raise ValueError(f"{col} out of range")

    if not np.all(df_specs["sample_label"].to_numpy() != df_specs["probe_label"].to_numpy()):
        raise ValueError("Population DMS trials must use mismatch sample/probe labels")


def build_stratified_splits(
    labels: np.ndarray,
    n_splits: int,
    test_ratio: float,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_splits):
        train_idx: List[int] = []
        test_idx: List[int] = []
        for cls in classes:
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) < 2:
                raise ValueError(f"Class {int(cls)} has <2 trials; increase --trials.")
            perm = rng.permutation(cls_idx)
            n_test = max(1, int(round(len(perm) * test_ratio)))
            n_test = min(n_test, len(perm) - 1)
            test_idx.extend(perm[:n_test].tolist())
            train_idx.extend(perm[n_test:].tolist())
        splits.append((np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)))
    return splits


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        raise ValueError("bootstrap_mean_ci received empty values")
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot, dtype=np.float64)
    n = vals.size
    for idx in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot[idx] = float(vals[sample_idx].mean())
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def bootstrap_decode_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    decode_splits: int,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    if len(features) != len(labels):
        raise ValueError("Feature/label length mismatch in bootstrap decode")

    by_class = {int(cls): np.where(labels == cls)[0] for cls in np.unique(labels)}
    rng = np.random.default_rng(seed)
    scores = np.zeros(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        idx_blocks = []
        for cls in sorted(by_class):
            cls_idx = by_class[cls]
            idx_blocks.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
        boot_idx = np.concatenate(idx_blocks, axis=0)
        boot_x = features[boot_idx]
        boot_y = labels[boot_idx]
        splits = build_stratified_splits(
            labels=boot_y,
            n_splits=decode_splits,
            test_ratio=0.3,
            seed=seed + 1000 + idx,
        )
        scores[idx] = decode_accuracy_with_splits(x=boot_x, y=boot_y, splits=splits, num_classes=num_classes)

    return {
        "ci95_lower": float(np.percentile(scores, 2.5)),
        "ci95_upper": float(np.percentile(scores, 97.5)),
        "p_one_sided_gt_chance": float((np.sum(scores <= (1.0 / float(num_classes))) + 1.0) / (len(scores) + 1.0)),
    }


def build_trial_phase_rate_table(
    spikes: torch.Tensor,
    phase_slices: Dict[str, List[int]],
    layer_name: str,
    batch_df: pd.DataFrame,
    stsp_mode: str,
) -> pd.DataFrame:
    t_steps, batch_size, channels, height, width = spikes.shape
    n_neurons = int(channels * height * width)
    eps = 1e-12

    phase_rates: Dict[str, np.ndarray] = {}
    phase_counts: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, float]] = []
    for phase in ["sample", "delay", "probe"]:
        start, end = phase_slices[phase]
        if not (0 <= start <= end <= t_steps):
            raise ValueError(f"Invalid phase slice {phase}: [{start}, {end})")
        duration = int(end - start)
        seg = spikes[start:end]
        counts = seg.sum(dim=(0, 2, 3, 4)).cpu().numpy().astype(np.int64, copy=False)
        rates = counts.astype(np.float64) / float(max(1, n_neurons * max(1, duration)))
        phase_counts[phase] = counts
        phase_rates[phase] = rates

    ratio_delay_sample = phase_rates["delay"] / np.maximum(phase_rates["sample"], eps)
    ratio_delay_probe = phase_rates["delay"] / np.maximum(phase_rates["probe"], eps)

    trial_ids = batch_df["trial_id"].to_numpy(dtype=np.int64)
    sample_labels = batch_df["sample_label"].to_numpy(dtype=np.int64)
    probe_labels = batch_df["probe_label"].to_numpy(dtype=np.int64)

    for phase in ["sample", "delay", "probe"]:
        start, end = phase_slices[phase]
        duration = int(end - start)
        for idx in range(batch_size):
            rows.append(
                {
                    "trial_id": int(trial_ids[idx]),
                    "stsp_mode": stsp_mode,
                    "sample_label": int(sample_labels[idx]),
                    "probe_label": int(probe_labels[idx]),
                    "layer": layer_name,
                    "phase": phase,
                    "start_step": int(start),
                    "end_step": int(end),
                    "duration_steps": duration,
                    "n_neurons": n_neurons,
                    "spike_count": int(phase_counts[phase][idx]),
                    "rate_spikes_per_neuron_step": float(phase_rates[phase][idx]),
                    "ratio_delay_over_sample": float(ratio_delay_sample[idx]),
                    "ratio_delay_over_probe": float(ratio_delay_probe[idx]),
                }
            )
    return pd.DataFrame(rows)


def extract_delay_features(spikes: torch.Tensor, phase_slices: Dict[str, List[int]]) -> np.ndarray:
    start, end = phase_slices["delay"]
    seg = spikes[start:end]
    delay_counts = seg.permute(1, 0, 2, 3, 4).reshape(seg.shape[1], end - start, -1).sum(dim=1)
    return delay_counts.cpu().numpy().astype(np.float32, copy=False)
