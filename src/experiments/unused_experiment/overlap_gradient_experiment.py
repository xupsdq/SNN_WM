from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.config.units import ms
from src.core.network import SDNN_Network
from src.data.encoding import DoGSpikeEncoder
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.runtime import resolve_device, seed_everything
from src.plotting.common.io import (
    COLOR_DYNAMIC,
    COLOR_STATIC,
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
    validate_required_columns,
)
from src.plotting.common.theme_tokens import (
    ALPHA_FILL,
    ALPHA_SCATTER_LIGHT,
    DELAY_SWEEP_COLORS,
    FIGSIZE_TWO_PANEL,
    GRID_ALPHA_SOFT,
    LINE_WIDTH_PRIMARY,
    LINE_WIDTH_REFERENCE,
    MARKER_CIRCLE,
    MODE_COLORS_DYNAMIC_STATIC,
    apply_standard_legend,
)

STSP_MODES: Tuple[str, ...] = ("dynamic", "static_frozen")
OVERLAP_BIN_ORDER: Tuple[str, ...] = ("low_overlap", "medium_overlap", "high_overlap")
MODE_TITLES = {
    "dynamic": "Dynamic STSP",
    "static_frozen": "Static Frozen STSP",
}
MODE_COLORS = dict(MODE_COLORS_DYNAMIC_STATIC)
DELAY_COLORS = list(DELAY_SWEEP_COLORS)


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


def parse_delay_list(delay_text: str) -> List[int]:
    values: List[int] = []
    for raw in str(delay_text).split(","):
        item = raw.strip()
        if not item:
            continue
        delay = int(float(item))
        if delay <= 0:
            raise ValueError("Delay values must be positive.")
        values.append(delay)
    if not values:
        raise ValueError("At least one delay is required.")
    return sorted(dict.fromkeys(values))


def _cosine_similarity_1d(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    denom = float(torch.norm(a_flat) * torch.norm(b_flat))
    if denom <= eps:
        return 0.0
    return float(torch.dot(a_flat, b_flat) / denom)


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    n = arr.size
    for idx in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot[idx] = arr[sample_idx].mean() * 100.0
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _bootstrap_independent_diff_summary(values_a: np.ndarray, values_b: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise ValueError("Independent bootstrap requires non-empty arrays.")

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        a_idx = rng.integers(0, a.size, size=a.size)
        b_idx = rng.integers(0, b.size, size=b.size)
        boot[idx] = (a[a_idx].mean() - b[b_idx].mean()) * 100.0

    observed = (a.mean() - b.mean()) * 100.0
    p_low = float(np.mean(boot <= 0.0))
    p_high = float(np.mean(boot >= 0.0))
    p_two = min(1.0, 2.0 * min(p_low, p_high))
    return {
        "observed_diff_pp": float(observed),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_two_sided": float(p_two),
        "n_boot": int(n_boot),
    }


def _bootstrap_paired_diff_summary(values_a: np.ndarray, values_b: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired bootstrap arrays must have the same shape.")
    if a.size == 0:
        raise ValueError("Paired bootstrap requires non-empty arrays.")

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    n = a.size
    for idx in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot[idx] = (a[sample_idx].mean() - b[sample_idx].mean()) * 100.0

    observed = (a.mean() - b.mean()) * 100.0
    p_low = float(np.mean(boot <= 0.0))
    p_high = float(np.mean(boot >= 0.0))
    p_two = min(1.0, 2.0 * min(p_low, p_high))
    return {
        "observed_diff_pp": float(observed),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_two_sided": float(p_two),
        "n_boot": int(n_boot),
    }


def _binary_metric_row(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean() * 100.0)
    ci_low, ci_high = _bootstrap_mean_ci(arr, n_boot=n_boot, seed=seed)
    return mean, ci_low, ci_high


def _get_image_from_cache(dataset, index: int, image_cache: Dict[int, torch.Tensor]) -> torch.Tensor:
    cached = image_cache.get(int(index))
    if cached is not None:
        return cached
    image = dataset[int(index)][0].detach().cpu()
    image_cache[int(index)] = image
    return image


def compute_overlap_score(
    sample_img: torch.Tensor,
    probe_img: torch.Tensor,
    metric: str = "pixel",
    encoder: DoGSpikeEncoder | None = None,
    spec: ExperimentSpec | None = None,
    device: torch.device | None = None,
    sample_encoded: torch.Tensor | None = None,
    probe_encoded: torch.Tensor | None = None,
) -> float:
    metric_name = str(metric).lower()
    if metric_name == "pixel":
        sample_active = sample_img > 0
        probe_active = probe_img > 0
        intersection = float((sample_active & probe_active).sum().item())
        union = float((sample_active | probe_active).sum().item())
        return 0.0 if union <= 0.0 else intersection / union

    if metric_name != "encoder":
        raise ValueError(f"Unsupported overlap metric: {metric}")
    if encoder is None or spec is None or device is None:
        raise ValueError("Encoder overlap requires encoder, spec, and device.")

    with torch.no_grad():
        if sample_encoded is None:
            sample_encoded = encoder.forward(sample_img.unsqueeze(0).to(device))
        if probe_encoded is None:
            probe_encoded = encoder.forward(probe_img.unsqueeze(0).to(device))
    return _cosine_similarity_1d(sample_encoded, probe_encoded)


def build_overlap_trials(
    dataset,
    num_trials: int,
    overlap_metric: str,
    encoder: DoGSpikeEncoder,
    spec: ExperimentSpec,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    if num_trials < 3:
        raise ValueError("--num-trials must be at least 3 so tercile overlap bins can be assigned.")

    rng = random.Random(seed)
    image_cache: Dict[int, torch.Tensor] = {}
    labels = np.array([int(dataset[idx][1]) for idx in range(len(dataset))], dtype=np.int64)
    seen_pairs: set[Tuple[int, int]] = set()
    rows: List[Dict[str, object]] = []
    attempts = 0
    max_attempts = max(num_trials * 50, 5000)

    while len(rows) < num_trials:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Unable to sample {num_trials} unique sample-probe pairs after {max_attempts} attempts. "
                "Reduce --num-trials or relax pair constraints."
            )

        sample_id = int(rng.randrange(len(dataset)))
        probe_id = int(rng.randrange(len(dataset)))
        if sample_id == probe_id:
            continue
        pair_key = (sample_id, probe_id)
        if pair_key in seen_pairs:
            continue

        sample_label = int(labels[sample_id])
        probe_label = int(labels[probe_id])
        sample_img = _get_image_from_cache(dataset, sample_id, image_cache=image_cache)
        probe_img = _get_image_from_cache(dataset, probe_id, image_cache=image_cache)
        overlap_score = compute_overlap_score(
            sample_img=sample_img,
            probe_img=probe_img,
            metric=overlap_metric,
            encoder=encoder,
            spec=spec,
            device=device,
        )
        rows.append(
            {
                "pair_id": int(len(rows)),
                "sample_id": int(sample_id),
                "sample_label": int(sample_label),
                "probe_id": int(probe_id),
                "probe_label": int(probe_label),
                "label_relation": "same_label" if sample_label == probe_label else "different_label",
                "overlap_metric": str(overlap_metric),
                "overlap_score": float(overlap_score),
            }
        )
        seen_pairs.add(pair_key)

    df_pairs = pd.DataFrame(rows)
    ranks = df_pairs["overlap_score"].rank(method="first")
    df_pairs["overlap_bin"] = pd.qcut(ranks, q=3, labels=list(OVERLAP_BIN_ORDER)).astype("object")
    return df_pairs.sort_values(["pair_id"], kind="stable").reset_index(drop=True)


def _prepare_batch_spikes(
    dataset,
    batch_df: pd.DataFrame,
    encoder: DoGSpikeEncoder,
    spec: ExperimentSpec,
    device: torch.device,
    image_cache: Dict[int, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        encoded_sample = encoder.forward(sample_images)[:, :spec.sample_steps, ...].contiguous()
        encoded_probe = encoder.forward(probe_images)[:, :spec.probe_steps, ...].contiguous()

    sample_lookup = {int(idx): pos for pos, idx in enumerate(unique_sample_ids)}
    probe_lookup = {int(idx): pos for pos, idx in enumerate(unique_probe_ids)}
    sample_select = torch.tensor([sample_lookup[int(idx)] for idx in sample_ids], device=device, dtype=torch.long)
    probe_select = torch.tensor([probe_lookup[int(idx)] for idx in probe_ids], device=device, dtype=torch.long)
    sample_spikes = encoded_sample.index_select(0, sample_select)
    probe_spikes = encoded_probe.index_select(0, probe_select)
    return sample_spikes, probe_spikes


def run_overlap_trials(
    net: SDNN_Network,
    encoder: DoGSpikeEncoder,
    dataset,
    df_pairs: pd.DataFrame,
    spec: ExperimentSpec,
    delay_values_ms: Sequence[int],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    validate_required_columns(
        df_pairs,
        [
            "pair_id",
            "sample_id",
            "sample_label",
            "probe_id",
            "probe_label",
            "label_relation",
            "overlap_score",
            "overlap_bin",
        ],
    )
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    image_cache: Dict[int, torch.Tensor] = {}
    records: List[Dict[str, object]] = []
    total_steps = math.ceil(len(df_pairs) / batch_size) * len(delay_values_ms) * len(STSP_MODES)
    pbar = tqdm(total=total_steps, desc="OverlapGradient")
    trial_id = 0

    for start in range(0, len(df_pairs), batch_size):
        batch = df_pairs.iloc[start:start + batch_size].copy().reset_index(drop=True)
        sample_spikes, probe_spikes = _prepare_batch_spikes(
            dataset=dataset,
            batch_df=batch,
            encoder=encoder,
            spec=spec,
            device=device,
            image_cache=image_cache,
        )

        for delay_ms in delay_values_ms:
            delay_steps = int(round((float(delay_ms) * ms) / spec.dt))
            for stsp_mode in STSP_MODES:
                with torch.no_grad():
                    out = net.forward_classify_session(
                        sample_spikes=sample_spikes,
                        test_spikes=probe_spikes,
                        delay_duration_steps=delay_steps,
                        stsp_mode=str(stsp_mode),
                    )
                pred = out["prediction"].detach().cpu().numpy().astype(np.int64, copy=False)
                for idx_in_batch, row in enumerate(batch.itertuples(index=False)):
                    predicted_label = int(pred[idx_in_batch])
                    sample_label = int(row.sample_label)
                    probe_label = int(row.probe_label)
                    records.append(
                        {
                            "seed": int(seed),
                            "trial_id": int(trial_id),
                            "pair_id": int(row.pair_id),
                            "sample_id": int(row.sample_id),
                            "sample_label": int(sample_label),
                            "probe_id": int(row.probe_id),
                            "probe_label": int(probe_label),
                            "label_relation": str(row.label_relation),
                            "overlap_metric": str(row.overlap_metric),
                            "overlap_score": float(row.overlap_score),
                            "overlap_bin": str(row.overlap_bin),
                            "delay_ms": int(delay_ms),
                            "stsp_mode": str(stsp_mode),
                            "predicted_label": int(predicted_label),
                            "is_correct": int(predicted_label == probe_label),
                            "pred_equals_sample": int(predicted_label == sample_label),
                            "pred_equals_probe": int(predicted_label == probe_label),
                            "is_silent": int(predicted_label == -1),
                        }
                    )
                    trial_id += 1
                pbar.update(1)

    pbar.close()
    df_trials = pd.DataFrame(records)
    return df_trials.sort_values(["pair_id", "delay_ms", "stsp_mode"], kind="stable").reset_index(drop=True)


def summarize_overlap_metrics(
    df_trials: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    validate_required_columns(
        df_trials,
        [
            "stsp_mode",
            "delay_ms",
            "overlap_bin",
            "label_relation",
            "is_correct",
            "pred_equals_sample",
            "pred_equals_probe",
        ],
    )

    rows: List[Dict[str, object]] = []
    congruency_rows: List[Dict[str, object]] = []
    delays = sorted(pd.unique(df_trials["delay_ms"]).tolist())

    for mode_idx, stsp_mode in enumerate(STSP_MODES):
        for delay_idx, delay_ms in enumerate(delays):
            for bin_idx, overlap_bin in enumerate(OVERLAP_BIN_ORDER):
                base_mask = (
                    (df_trials["stsp_mode"] == stsp_mode)
                    & (df_trials["delay_ms"] == int(delay_ms))
                    & (df_trials["overlap_bin"] == overlap_bin)
                )
                subset = df_trials[base_mask].copy()
                probe_acc, probe_acc_lo, probe_acc_hi = _binary_metric_row(
                    subset["is_correct"].to_numpy(dtype=np.float64),
                    n_boot=n_boot,
                    seed=seed + 11 + mode_idx * 1000 + delay_idx * 100 + bin_idx * 11,
                )
                sample_bias, sample_bias_lo, sample_bias_hi = _binary_metric_row(
                    subset["pred_equals_sample"].to_numpy(dtype=np.float64),
                    n_boot=n_boot,
                    seed=seed + 21 + mode_idx * 1000 + delay_idx * 100 + bin_idx * 11,
                )
                probe_match, probe_match_lo, probe_match_hi = _binary_metric_row(
                    subset["pred_equals_probe"].to_numpy(dtype=np.float64),
                    n_boot=n_boot,
                    seed=seed + 31 + mode_idx * 1000 + delay_idx * 100 + bin_idx * 11,
                )
                rows.append(
                    {
                        "summary_type": "overall",
                        "stsp_mode": str(stsp_mode),
                        "delay_ms": int(delay_ms),
                        "overlap_bin": str(overlap_bin),
                        "label_relation": "all",
                        "n_trials": int(len(subset)),
                        "probe_accuracy": float(probe_acc),
                        "probe_accuracy_ci95_lower": float(probe_acc_lo),
                        "probe_accuracy_ci95_upper": float(probe_acc_hi),
                        "sample_bias_rate": float(sample_bias),
                        "sample_bias_rate_ci95_lower": float(sample_bias_lo),
                        "sample_bias_rate_ci95_upper": float(sample_bias_hi),
                        "pred_equals_probe_rate": float(probe_match),
                        "pred_equals_probe_rate_ci95_lower": float(probe_match_lo),
                        "pred_equals_probe_rate_ci95_upper": float(probe_match_hi),
                    }
                )

                for rel_idx, label_relation in enumerate(["same_label", "different_label"]):
                    rel_sub = subset[subset["label_relation"] == label_relation].copy()
                    rel_probe_acc, rel_probe_lo, rel_probe_hi = _binary_metric_row(
                        rel_sub["is_correct"].to_numpy(dtype=np.float64),
                        n_boot=n_boot,
                        seed=seed + 101 + mode_idx * 1000 + delay_idx * 100 + bin_idx * 11 + rel_idx,
                    )
                    rel_sample_bias, rel_bias_lo, rel_bias_hi = _binary_metric_row(
                        rel_sub["pred_equals_sample"].to_numpy(dtype=np.float64),
                        n_boot=n_boot,
                        seed=seed + 201 + mode_idx * 1000 + delay_idx * 100 + bin_idx * 11 + rel_idx,
                    )
                    rows.append(
                        {
                            "summary_type": "by_label_relation",
                            "stsp_mode": str(stsp_mode),
                            "delay_ms": int(delay_ms),
                            "overlap_bin": str(overlap_bin),
                            "label_relation": str(label_relation),
                            "n_trials": int(len(rel_sub)),
                            "probe_accuracy": float(rel_probe_acc),
                            "probe_accuracy_ci95_lower": float(rel_probe_lo),
                            "probe_accuracy_ci95_upper": float(rel_probe_hi),
                            "sample_bias_rate": float(rel_sample_bias),
                            "sample_bias_rate_ci95_lower": float(rel_bias_lo),
                            "sample_bias_rate_ci95_upper": float(rel_bias_hi),
                            "pred_equals_probe_rate": float(rel_probe_acc),
                            "pred_equals_probe_rate_ci95_lower": float(rel_probe_lo),
                            "pred_equals_probe_rate_ci95_upper": float(rel_probe_hi),
                        }
                    )

                same_sub = subset[subset["label_relation"] == "same_label"].copy()
                diff_sub = subset[subset["label_relation"] == "different_label"].copy()
                same_acc = float(same_sub["is_correct"].mean() * 100.0) if len(same_sub) > 0 else float("nan")
                diff_acc = float(diff_sub["is_correct"].mean() * 100.0) if len(diff_sub) > 0 else float("nan")
                effect = same_acc - diff_acc if np.isfinite(same_acc) and np.isfinite(diff_acc) else float("nan")
                congruency_row = {
                    "summary_type": "congruency_effect",
                    "stsp_mode": str(stsp_mode),
                    "delay_ms": int(delay_ms),
                    "overlap_bin": str(overlap_bin),
                    "n_same_label": int(len(same_sub)),
                    "n_different_label": int(len(diff_sub)),
                    "same_label_accuracy": float(same_acc),
                    "different_label_accuracy": float(diff_acc),
                    "congruency_effect_pp": float(effect),
                }
                congruency_rows.append(congruency_row)
                rows.append(
                    {
                        "summary_type": "congruency_effect",
                        "stsp_mode": str(stsp_mode),
                        "delay_ms": int(delay_ms),
                        "overlap_bin": str(overlap_bin),
                        "label_relation": "all",
                        "n_trials": int(len(same_sub) + len(diff_sub)),
                        "probe_accuracy": float("nan"),
                        "probe_accuracy_ci95_lower": float("nan"),
                        "probe_accuracy_ci95_upper": float("nan"),
                        "sample_bias_rate": float("nan"),
                        "sample_bias_rate_ci95_lower": float("nan"),
                        "sample_bias_rate_ci95_upper": float("nan"),
                        "pred_equals_probe_rate": float("nan"),
                        "pred_equals_probe_rate_ci95_lower": float("nan"),
                        "pred_equals_probe_rate_ci95_upper": float("nan"),
                        "same_label_accuracy": float(same_acc),
                        "different_label_accuracy": float(diff_acc),
                        "congruency_effect_pp": float(effect),
                        "n_same_label": int(len(same_sub)),
                        "n_different_label": int(len(diff_sub)),
                    }
                )

    df_summary = pd.DataFrame(rows)
    df_congruency = pd.DataFrame(congruency_rows)
    return (
        df_summary.sort_values(["summary_type", "stsp_mode", "delay_ms", "overlap_bin", "label_relation"], kind="stable")
        .reset_index(drop=True),
        df_congruency.sort_values(["stsp_mode", "delay_ms", "overlap_bin"], kind="stable").reset_index(drop=True),
    )


def run_overlap_statistics(
    df_trials: pd.DataFrame,
    n_boot: int,
    seed: int,
    min_group_n: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    validate_required_columns(
        df_trials,
        [
            "pair_id",
            "delay_ms",
            "stsp_mode",
            "overlap_score",
            "overlap_bin",
            "label_relation",
            "is_correct",
            "pred_equals_sample",
        ],
    )

    stats_rows: List[Dict[str, object]] = []
    regression_rows: List[Dict[str, object]] = []
    delays = sorted(pd.unique(df_trials["delay_ms"]).tolist())

    dynamic_df = df_trials[df_trials["stsp_mode"] == "dynamic"].copy()
    for delay_idx, delay_ms in enumerate(delays):
        delay_df = dynamic_df[dynamic_df["delay_ms"] == int(delay_ms)].copy()
        for bin_a, bin_b in combinations(OVERLAP_BIN_ORDER, 2):
            sub_a = delay_df[delay_df["overlap_bin"] == bin_a]["is_correct"].to_numpy(dtype=np.float64)
            sub_b = delay_df[delay_df["overlap_bin"] == bin_b]["is_correct"].to_numpy(dtype=np.float64)
            if sub_a.size == 0 or sub_b.size == 0:
                continue
            res = _bootstrap_independent_diff_summary(sub_a, sub_b, n_boot=n_boot, seed=seed + 100 + delay_idx * 50)
            stats_rows.append(
                {
                    "test_name": "dynamic_bin_accuracy",
                    "metric": "probe_accuracy",
                    "stsp_mode": "dynamic",
                    "delay_ms": int(delay_ms),
                    "group_a": str(bin_a),
                    "group_b": str(bin_b),
                    "comparison_label": f"{bin_a} - {bin_b}",
                    "n_a": int(sub_a.size),
                    "n_b": int(sub_b.size),
                    "observed_diff_pp": float(res["observed_diff_pp"]),
                    "ci95_lower_pp": float(res["ci_low"]),
                    "ci95_upper_pp": float(res["ci_high"]),
                    "p_two_sided": float(res["p_two_sided"]),
                    "status": "ok",
                }
            )

    for delay_idx, delay_ms in enumerate(delays):
        for bin_idx, overlap_bin in enumerate(["low_overlap", "high_overlap"]):
            sub = df_trials[
                (df_trials["delay_ms"] == int(delay_ms))
                & (df_trials["overlap_bin"] == overlap_bin)
            ].copy()
            for metric_col, metric_name in [("is_correct", "probe_accuracy"), ("pred_equals_sample", "sample_bias_rate")]:
                pivot = sub.pivot_table(index="pair_id", columns="stsp_mode", values=metric_col, aggfunc="first")
                if not {"dynamic", "static_frozen"}.issubset(set(pivot.columns)):
                    continue
                pivot = pivot.dropna(subset=["dynamic", "static_frozen"])
                if len(pivot) == 0:
                    continue
                res = _bootstrap_paired_diff_summary(
                    pivot["dynamic"].to_numpy(dtype=np.float64),
                    pivot["static_frozen"].to_numpy(dtype=np.float64),
                    n_boot=n_boot,
                    seed=seed + 500 + delay_idx * 50 + bin_idx * 7,
                )
                stats_rows.append(
                    {
                        "test_name": "dynamic_vs_static",
                        "metric": str(metric_name),
                        "stsp_mode": "paired_modes",
                        "delay_ms": int(delay_ms),
                        "group_a": "dynamic",
                        "group_b": "static_frozen",
                        "comparison_label": str(overlap_bin),
                        "n_a": int(len(pivot)),
                        "n_b": int(len(pivot)),
                        "observed_diff_pp": float(res["observed_diff_pp"]),
                        "ci95_lower_pp": float(res["ci_low"]),
                        "ci95_upper_pp": float(res["ci_high"]),
                        "p_two_sided": float(res["p_two_sided"]),
                        "status": "ok",
                    }
                )

    for mode_idx, stsp_mode in enumerate(STSP_MODES):
        for delay_idx, delay_ms in enumerate(delays):
            for bin_idx, overlap_bin in enumerate(OVERLAP_BIN_ORDER):
                subset = df_trials[
                    (df_trials["stsp_mode"] == stsp_mode)
                    & (df_trials["delay_ms"] == int(delay_ms))
                    & (df_trials["overlap_bin"] == overlap_bin)
                ].copy()
                same_vals = subset[subset["label_relation"] == "same_label"]["is_correct"].to_numpy(dtype=np.float64)
                diff_vals = subset[subset["label_relation"] == "different_label"]["is_correct"].to_numpy(dtype=np.float64)
                if same_vals.size < min_group_n or diff_vals.size < min_group_n:
                    stats_rows.append(
                        {
                            "test_name": "same_vs_different_accuracy",
                            "metric": "probe_accuracy",
                            "stsp_mode": str(stsp_mode),
                            "delay_ms": int(delay_ms),
                            "group_a": "same_label",
                            "group_b": "different_label",
                            "comparison_label": str(overlap_bin),
                            "n_a": int(same_vals.size),
                            "n_b": int(diff_vals.size),
                            "observed_diff_pp": float("nan"),
                            "ci95_lower_pp": float("nan"),
                            "ci95_upper_pp": float("nan"),
                            "p_two_sided": float("nan"),
                            "status": "insufficient_samples",
                        }
                    )
                    continue
                res = _bootstrap_independent_diff_summary(
                    same_vals,
                    diff_vals,
                    n_boot=n_boot,
                    seed=seed + 900 + mode_idx * 100 + delay_idx * 30 + bin_idx,
                )
                stats_rows.append(
                    {
                        "test_name": "same_vs_different_accuracy",
                        "metric": "probe_accuracy",
                        "stsp_mode": str(stsp_mode),
                        "delay_ms": int(delay_ms),
                        "group_a": "same_label",
                        "group_b": "different_label",
                        "comparison_label": str(overlap_bin),
                        "n_a": int(same_vals.size),
                        "n_b": int(diff_vals.size),
                        "observed_diff_pp": float(res["observed_diff_pp"]),
                        "ci95_lower_pp": float(res["ci_low"]),
                        "ci95_upper_pp": float(res["ci_high"]),
                        "p_two_sided": float(res["p_two_sided"]),
                        "status": "ok",
                    }
                )

    for stsp_mode in STSP_MODES:
        for metric_col, metric_name in [("is_correct", "accuracy"), ("pred_equals_sample", "sample_bias_rate")]:
            for delay_key, delay_filter in [("all", None)] + [(str(delay), int(delay)) for delay in delays]:
                sub = df_trials[df_trials["stsp_mode"] == stsp_mode].copy()
                if delay_filter is not None:
                    sub = sub[sub["delay_ms"] == delay_filter].copy()
                x = sub["overlap_score"].to_numpy(dtype=np.float64)
                y = sub[metric_col].to_numpy(dtype=np.float64)
                enough = len(sub) >= 3 and np.unique(x).size >= 2
                if enough:
                    rho, p_value = spearmanr(x, y)
                    slope, intercept = np.polyfit(x, y, deg=1)
                    status = "ok"
                else:
                    rho = p_value = slope = intercept = float("nan")
                    status = "insufficient_samples"
                regression_rows.append(
                    {
                        "stsp_mode": str(stsp_mode),
                        "delay_ms": delay_key,
                        "metric": str(metric_name),
                        "n_trials": int(len(sub)),
                        "spearman_rho": float(rho),
                        "spearman_p": float(p_value),
                        "linear_slope": float(slope),
                        "linear_intercept": float(intercept),
                        "status": status,
                    }
                )

    df_stats = pd.DataFrame(stats_rows).sort_values(
        ["test_name", "metric", "stsp_mode", "delay_ms", "comparison_label"],
        kind="stable",
    ).reset_index(drop=True)
    df_regression = pd.DataFrame(regression_rows).sort_values(
        ["metric", "stsp_mode", "delay_ms"],
        kind="stable",
    ).reset_index(drop=True)

    text_lines: List[str] = []
    text_lines.append("Overlap Gradient Experiment Statistics")
    text_lines.append("")
    text_lines.append("1. Dynamic overlap-bin accuracy comparisons")
    dyn_rows = df_stats[df_stats["test_name"] == "dynamic_bin_accuracy"]
    for row in dyn_rows.itertuples(index=False):
        text_lines.append(
            f"delay={row.delay_ms} ms, {row.comparison_label}: diff={row.observed_diff_pp:.2f} pp "
            f"[{row.ci95_lower_pp:.2f}, {row.ci95_upper_pp:.2f}], p={row.p_two_sided:.4g}"
        )

    text_lines.append("")
    text_lines.append("2. Dynamic vs static_frozen at low/high overlap")
    mode_rows = df_stats[df_stats["test_name"] == "dynamic_vs_static"]
    for row in mode_rows.itertuples(index=False):
        text_lines.append(
            f"delay={row.delay_ms} ms, bin={row.comparison_label}, metric={row.metric}: "
            f"dynamic-static={row.observed_diff_pp:.2f} pp "
            f"[{row.ci95_lower_pp:.2f}, {row.ci95_upper_pp:.2f}], p={row.p_two_sided:.4g}"
        )

    text_lines.append("")
    text_lines.append("3. Continuous overlap correlations")
    for row in df_regression.itertuples(index=False):
        text_lines.append(
            f"mode={row.stsp_mode}, delay={row.delay_ms}, metric={row.metric}: "
            f"rho={row.spearman_rho:.4f}, p={row.spearman_p:.4g}, slope={row.linear_slope:.4f}, status={row.status}"
        )

    text_lines.append("")
    text_lines.append("4. same_label vs different_label within overlap bins")
    rel_rows = df_stats[df_stats["test_name"] == "same_vs_different_accuracy"]
    for row in rel_rows.itertuples(index=False):
        if row.status != "ok":
            text_lines.append(
                f"mode={row.stsp_mode}, delay={row.delay_ms}, bin={row.comparison_label}: insufficient samples "
                f"(same={row.n_a}, diff={row.n_b})"
            )
            continue
        text_lines.append(
            f"mode={row.stsp_mode}, delay={row.delay_ms}, bin={row.comparison_label}: "
            f"same-different={row.observed_diff_pp:.2f} pp "
            f"[{row.ci95_lower_pp:.2f}, {row.ci95_upper_pp:.2f}], p={row.p_two_sided:.4g}"
        )

    return df_stats, df_regression, "\n".join(text_lines) + "\n"


def make_figure_accuracy_vs_overlap(df_summary: pd.DataFrame) -> plt.Figure:
    overall = df_summary[df_summary["summary_type"] == "overall"].copy()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL, sharey=True)
    x = np.arange(len(OVERLAP_BIN_ORDER))
    delays = sorted(pd.unique(overall["delay_ms"]).tolist())

    for ax, stsp_mode in zip(axes, STSP_MODES):
        sub = overall[overall["stsp_mode"] == stsp_mode].copy()
        for delay_idx, delay_ms in enumerate(delays):
            delay_sub = sub[sub["delay_ms"] == int(delay_ms)].copy().set_index("overlap_bin").reindex(OVERLAP_BIN_ORDER).reset_index()
            ax.plot(
                x,
                delay_sub["probe_accuracy"].to_numpy(dtype=np.float64),
                marker=MARKER_CIRCLE,
                linewidth=LINE_WIDTH_PRIMARY,
                color=DELAY_COLORS[delay_idx % len(DELAY_COLORS)],
                label=f"{delay_ms} ms",
            )
        ax.set_title(MODE_TITLES[stsp_mode])
        ax.set_xticks(x)
        ax.set_xticklabels(["low", "medium", "high"])
        ax.set_xlabel("Overlap bin")
        ax.set_ylim(0, 100)
        ax.grid(alpha=GRID_ALPHA_SOFT)
        apply_standard_legend(ax)
    axes[0].set_ylabel("Accuracy (%)")
    fig.tight_layout()
    return fig


def make_figure_sample_bias_vs_overlap(df_summary: pd.DataFrame) -> plt.Figure:
    overall = df_summary[df_summary["summary_type"] == "overall"].copy()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL, sharey=True)
    x = np.arange(len(OVERLAP_BIN_ORDER))
    delays = sorted(pd.unique(overall["delay_ms"]).tolist())

    for ax, stsp_mode in zip(axes, STSP_MODES):
        sub = overall[overall["stsp_mode"] == stsp_mode].copy()
        for delay_idx, delay_ms in enumerate(delays):
            delay_sub = sub[sub["delay_ms"] == int(delay_ms)].copy().set_index("overlap_bin").reindex(OVERLAP_BIN_ORDER).reset_index()
            ax.plot(
                x,
                delay_sub["sample_bias_rate"].to_numpy(dtype=np.float64),
                marker=MARKER_CIRCLE,
                linewidth=LINE_WIDTH_PRIMARY,
                color=DELAY_COLORS[delay_idx % len(DELAY_COLORS)],
                label=f"{delay_ms} ms",
            )
        ax.set_title(MODE_TITLES[stsp_mode])
        ax.set_xticks(x)
        ax.set_xticklabels(["low", "medium", "high"])
        ax.set_xlabel("Overlap bin")
        ax.set_ylim(0, 100)
        ax.grid(alpha=GRID_ALPHA_SOFT)
        apply_standard_legend(ax)
    axes[0].set_ylabel("P(pred = sample label) (%)")
    fig.tight_layout()
    return fig


def make_figure_overlap_regression(df_trials: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL, sharey=True)
    delays = sorted(pd.unique(df_trials["delay_ms"]).tolist())

    for ax, stsp_mode in zip(axes, STSP_MODES):
        sub_mode = df_trials[df_trials["stsp_mode"] == stsp_mode].copy()
        for delay_idx, delay_ms in enumerate(delays):
            sub = sub_mode[sub_mode["delay_ms"] == int(delay_ms)].copy().sort_values("overlap_score")
            if len(sub) == 0:
                continue
            x = sub["overlap_score"].to_numpy(dtype=np.float64)
            y = sub["pred_equals_sample"].to_numpy(dtype=np.float64)
            jitter = np.random.default_rng(1234 + delay_idx).normal(0.0, 0.015, size=len(sub))
            color = DELAY_COLORS[delay_idx % len(DELAY_COLORS)]
            ax.scatter(x, y + jitter, s=12, alpha=ALPHA_SCATTER_LIGHT, color=color, label=f"{delay_ms} ms")
            if len(sub) >= 3 and np.unique(x).size >= 2:
                slope, intercept = np.polyfit(x, y, deg=1)
                x_line = np.linspace(float(x.min()), float(x.max()), 100)
                ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=LINE_WIDTH_PRIMARY)
        ax.set_title(MODE_TITLES[stsp_mode])
        ax.set_xlabel("Continuous overlap score")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=GRID_ALPHA_SOFT)
        apply_standard_legend(ax)
    axes[0].set_ylabel("pred == sample label")
    fig.tight_layout()
    return fig


def make_figure_congruency_vs_overlap(df_congruency: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL, sharey=True)
    x = np.arange(len(OVERLAP_BIN_ORDER))
    delays = sorted(pd.unique(df_congruency["delay_ms"]).tolist())

    for ax, stsp_mode in zip(axes, STSP_MODES):
        sub = df_congruency[df_congruency["stsp_mode"] == stsp_mode].copy()
        for delay_idx, delay_ms in enumerate(delays):
            delay_sub = sub[sub["delay_ms"] == int(delay_ms)].copy().set_index("overlap_bin").reindex(OVERLAP_BIN_ORDER).reset_index()
            ax.plot(
                x,
                delay_sub["congruency_effect_pp"].to_numpy(dtype=np.float64),
                marker=MARKER_CIRCLE,
                linewidth=LINE_WIDTH_PRIMARY,
                color=DELAY_COLORS[delay_idx % len(DELAY_COLORS)],
                label=f"{delay_ms} ms",
            )
        ax.axhline(0.0, color="black", linewidth=LINE_WIDTH_REFERENCE, linestyle="--")
        ax.set_title(MODE_TITLES[stsp_mode])
        ax.set_xticks(x)
        ax.set_xticklabels(["low", "medium", "high"])
        ax.set_xlabel("Overlap bin")
        ax.grid(alpha=GRID_ALPHA_SOFT)
        apply_standard_legend(ax)
    axes[0].set_ylabel("same_label - different_label accuracy (pp)")
    fig.tight_layout()
    return fig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlap-gradient sample-delay-probe experiment.")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--output-dir", type=str, default="results/overlap_gradient_experiment")
    parser.add_argument("--num-trials", type=int, default=1000)
    parser.add_argument("--delay-ms-list", type=str, default="300,600,1000")
    parser.add_argument("--overlap-metric", type=str, default="pixel", choices=["pixel", "encoder"])
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_boot <= 0:
        raise ValueError("--num-boot must be positive.")

    delay_values_ms = parse_delay_list(args.delay_ms_list)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    spec = ExperimentSpec(dt=1.0 * ms, sample_ms=args.sample_ms, probe_ms=args.probe_ms)
    if spec.sample_steps <= 0 or spec.probe_steps <= 0:
        raise ValueError("sample and probe durations must resolve to positive step counts.")

    layout = prepare_result_layout(args.output_dir)
    result_root = layout.root
    save_dir = layout.data_dir
    figure_dir = layout.figure_dir
    log_dir = layout.log_dir
    apply_publication_style()

    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(spec.sample_ms, spec.probe_ms),
    )
    _, _, test_loader = build_mnist_skeleton_loader(
        root=args.dataset_root,
        batch_size=1,
        input_size=28,
        num_workers=0,
    )
    dataset = test_loader.dataset

    df_pairs = build_overlap_trials(
        dataset=dataset,
        num_trials=args.num_trials,
        overlap_metric=args.overlap_metric,
        encoder=encoder,
        spec=spec,
        device=device,
        seed=args.seed,
    )
    df_trials = run_overlap_trials(
        net=net,
        encoder=encoder,
        dataset=dataset,
        df_pairs=df_pairs,
        spec=spec,
        delay_values_ms=delay_values_ms,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    df_summary, df_congruency = summarize_overlap_metrics(
        df_trials=df_trials,
        n_boot=args.num_boot,
        seed=args.seed,
    )
    df_stats, df_regression, stats_text = run_overlap_statistics(
        df_trials=df_trials,
        n_boot=args.num_boot,
        seed=args.seed,
    )

    trial_csv = save_tidy_csv(
        df_trials,
        save_dir / "trial_level_results.csv",
        sort_by=["pair_id", "delay_ms", "stsp_mode"],
    )
    summary_csv = save_tidy_csv(
        df_summary,
        save_dir / "overlap_summary.csv",
        sort_by=["summary_type", "stsp_mode", "delay_ms", "overlap_bin", "label_relation"],
    )
    regression_csv = save_tidy_csv(
        df_regression,
        save_dir / "overlap_regression_stats.csv",
        sort_by=["metric", "stsp_mode", "delay_ms"],
    )
    statistics_csv = save_tidy_csv(
        df_stats,
        save_dir / "overlap_statistics.csv",
        sort_by=["test_name", "metric", "stsp_mode", "delay_ms", "comparison_label"],
    )
    statistics_txt = save_dir / "overlap_statistics.txt"
    statistics_txt.write_text(stats_text, encoding="utf-8")

    fig1 = make_figure_accuracy_vs_overlap(df_summary=df_summary)
    fig1_paths = save_figure_all_formats(fig1, figure_dir / "figure_1_accuracy_vs_overlap")
    plt.close(fig1)

    fig2 = make_figure_sample_bias_vs_overlap(df_summary=df_summary)
    fig2_paths = save_figure_all_formats(fig2, figure_dir / "figure_2_sample_bias_vs_overlap")
    plt.close(fig2)

    fig3 = make_figure_overlap_regression(df_trials=df_trials)
    fig3_paths = save_figure_all_formats(fig3, figure_dir / "figure_3_overlap_regression")
    plt.close(fig3)

    fig4 = make_figure_congruency_vs_overlap(df_congruency=df_congruency)
    fig4_paths = save_figure_all_formats(fig4, figure_dir / "figure_4_congruency_vs_overlap")
    plt.close(fig4)

    run_config_path = save_run_config(
        {
            "model_path": args.model_path,
            "dataset_root": args.dataset_root,
            "output_dir": str(result_root),
            "device": str(device),
            "seed": int(args.seed),
            "num_trials": int(args.num_trials),
            "delay_ms_list": [int(v) for v in delay_values_ms],
            "overlap_metric": str(args.overlap_metric),
            "sample_ms": float(args.sample_ms),
            "probe_ms": float(args.probe_ms),
            "batch_size": int(args.batch_size),
            "num_boot": int(args.num_boot),
            "stsp_modes": list(STSP_MODES),
            "overlap_bin_order": list(OVERLAP_BIN_ORDER),
            "notes": {
                "pixel_overlap": "Binary pixel IoU after thresholding input image at > 0.",
                "encoder_overlap": "Cosine similarity over full encoded spike tensor flattened across time and space.",
                "trial_count_definition": "num_trials counts unique base sample-probe pairs before expansion across delay and STSP mode.",
            },
            "outputs": {
                "trial_level_results_csv": str(trial_csv),
                "overlap_summary_csv": str(summary_csv),
                "overlap_regression_stats_csv": str(regression_csv),
                "overlap_statistics_csv": str(statistics_csv),
                "overlap_statistics_txt": str(statistics_txt),
                "figure_1_png": fig1_paths["png"],
                "figure_1_pdf": fig1_paths["pdf"],
                "figure_2_png": fig2_paths["png"],
                "figure_2_pdf": fig2_paths["pdf"],
                "figure_3_png": fig3_paths["png"],
                "figure_3_pdf": fig3_paths["pdf"],
                "figure_4_png": fig4_paths["png"],
                "figure_4_pdf": fig4_paths["pdf"],
            },
        },
        result_root,
    )
    summary_path = save_summary_json(
        {
            "experiment": "overlap_gradient_experiment",
            "trial_count": int(len(df_trials)),
            "summary_rows": int(len(df_summary)),
            "delay_ms_list": [int(v) for v in delay_values_ms],
            "run_config_json": str(run_config_path.resolve()),
        },
        result_root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=overlap_gradient_experiment",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"seed={int(args.seed)}",
            f"device={device}",
            f"trials={len(df_trials)}",
            f"result_root={result_root.resolve()}",
            f"summary_json={summary_path.resolve()}",
        ],
        log_dir,
    )

    print(f"[Done] Saved: {trial_csv}")
    print(f"[Done] Saved: {summary_csv}")
    print(f"[Done] Saved: {regression_csv}")
    print(f"[Done] Saved: {statistics_csv}")
    print(f"[Done] Saved: {statistics_txt}")
    print(f"[Done] Saved: {run_config_path}")


if __name__ == "__main__":
    main()
