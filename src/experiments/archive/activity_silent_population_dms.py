import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.platform.legacy_adapters.encoding import DoGSpikeEncoder, build_mnist_skeleton_loader
from src.platform.legacy_adapters.network import SDNN_Network
from src.experiments.common.dataset import build_class_index as shared_build_class_index
from src.experiments.common.dataset import encode_images as shared_encode_images
from src.experiments.common.decoding import decode_accuracy_with_splits as shared_decode_accuracy_with_splits
from src.experiments.common.model_io import compensate_stsp_gain as shared_compensate_stsp_gain
from src.experiments.common.model_io import load_model_and_encoder as shared_load_model_and_encoder
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_run_config, save_summary_json
from src.experiments.common.runtime import seed_everything as shared_seed_everything
from src.plotting.common.io import apply_publication_style, save_figure_all_formats, save_tidy_csv
from src.plotting.common.theme_tokens import (
    ALPHA_ANNOTATION_BOX,
    ALPHA_BAR,
    FIGSIZE_THREE_PANEL_COMPACT,
    FIGSIZE_THREE_PANEL_MEDIUM,
    GRID_ALPHA,
    LINE_WIDTH_REFERENCE,
    SILENT_MEMORY_MODE_COLORS,
    apply_standard_figure_legend,
)
from src.platform.legacy_adapters.units import ms


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compensate_stsp_gain(net: SDNN_Network, scaling_factor: float) -> None:
    with torch.no_grad():
        if hasattr(net, "layer1"):
            net.layer1.kernels.data *= scaling_factor
        if hasattr(net, "layer2"):
            net.layer2.kernels.data *= scaling_factor
        if hasattr(net, "layer3"):
            net.layer3.kernels.data *= scaling_factor


def load_model_and_encoder(
    model_path: str,
    device: torch.device,
    spec: ExperimentSpec,
) -> Tuple[SDNN_Network, DoGSpikeEncoder]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    net = SDNN_Network(device=str(device)).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    compensate_stsp_gain(net, scaling_factor=1.0 / net.layer3.stsp_U)
    net.eval()

    max_duration_ms = max(spec.sample_ms, spec.probe_ms)
    encoder = DoGSpikeEncoder(dt=spec.dt, max_duration=max_duration_ms * ms, device=str(device))
    return net, encoder


def build_class_index(dataset, num_classes: int) -> Dict[int, List[int]]:
    class_index: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_index[int(label)].append(idx)

    for cls in range(num_classes):
        if len(class_index[cls]) == 0:
            raise ValueError(f"Class {cls} has no samples in dataset")
    return class_index


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


def encode_images(encoder: DoGSpikeEncoder, images: torch.Tensor, steps: int) -> torch.Tensor:
    with torch.no_grad():
        spikes = encoder.forward(images)
    return spikes[:, :steps, ...].contiguous()


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


def decode_accuracy_with_splits(
    x: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
) -> float:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape={x.shape}")
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)

    accs: List[float] = []
    for train_idx, test_idx in splits:
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        d = x.shape[1]
        centroids = np.zeros((num_classes, d), dtype=np.float32)
        valid = np.zeros(num_classes, dtype=np.bool_)
        for c in range(num_classes):
            mask = y_train == c
            if not np.any(mask):
                continue
            centroids[c] = x_train[mask].mean(axis=0)
            valid[c] = True

        dist = ((x_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        dist[:, ~valid] = np.inf
        pred = np.argmin(dist, axis=1).astype(np.int64)
        accs.append(float(np.mean(pred == y_test)))
    return float(np.mean(accs))


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        raise ValueError("bootstrap_mean_ci received empty values")
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot, dtype=np.float64)
    n = vals.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(vals[idx].mean())
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
    for b in range(n_boot):
        idx_blocks = []
        for cls in sorted(by_class):
            cls_idx = by_class[cls]
            idx_blocks.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
        idx = np.concatenate(idx_blocks, axis=0)
        boot_x = features[idx]
        boot_y = labels[idx]
        splits = build_stratified_splits(
            labels=boot_y,
            n_splits=decode_splits,
            test_ratio=0.3,
            seed=seed + 1000 + b,
        )
        scores[b] = decode_accuracy_with_splits(x=boot_x, y=boot_y, splits=splits, num_classes=num_classes)

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
        for i in range(batch_size):
            rows.append(
                {
                    "trial_id": int(trial_ids[i]),
                    "stsp_mode": stsp_mode,
                    "sample_label": int(sample_labels[i]),
                    "probe_label": int(probe_labels[i]),
                    "layer": layer_name,
                    "phase": phase,
                    "start_step": int(start),
                    "end_step": int(end),
                    "duration_steps": duration,
                    "n_neurons": n_neurons,
                    "spike_count": int(phase_counts[phase][i]),
                    "rate_spikes_per_neuron_step": float(phase_rates[phase][i]),
                    "ratio_delay_over_sample": float(ratio_delay_sample[i]),
                    "ratio_delay_over_probe": float(ratio_delay_probe[i]),
                }
            )
    return pd.DataFrame(rows)


def extract_delay_features(spikes: torch.Tensor, phase_slices: Dict[str, List[int]]) -> np.ndarray:
    start, end = phase_slices["delay"]
    seg = spikes[start:end]
    delay_counts = seg.permute(1, 0, 2, 3, 4).reshape(seg.shape[1], end - start, -1).sum(dim=1)
    return delay_counts.cpu().numpy().astype(np.float32, copy=False)


def summarize_delay_decode(
    feature_buf: Dict[str, Dict[str, List[np.ndarray]]],
    labels: np.ndarray,
    num_classes: int,
    decode_splits: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    chance = 1.0 / float(num_classes)
    for layer_idx, layer_name in enumerate(LAYER_KEYS):
        for mode_idx, stsp_mode in enumerate(STSP_MODES):
            feat = np.concatenate(feature_buf[stsp_mode][layer_name], axis=0)
            splits = build_stratified_splits(
                labels=labels,
                n_splits=decode_splits,
                test_ratio=0.3,
                seed=seed + 131 + layer_idx * 100 + mode_idx * 17,
            )
            acc = decode_accuracy_with_splits(x=feat, y=labels, splits=splits, num_classes=num_classes)
            boot = bootstrap_decode_accuracy(
                features=feat,
                labels=labels,
                num_classes=num_classes,
                decode_splits=decode_splits,
                n_boot=n_boot,
                seed=seed + 911 + layer_idx * 100 + mode_idx * 17,
            )
            rows.append(
                {
                    "layer": layer_name,
                    "stsp_mode": stsp_mode,
                    "n_trials": int(len(labels)),
                    "decode_delay_acc": float(acc),
                    "decode_delay_acc_ci95_lower": float(boot["ci95_lower"]),
                    "decode_delay_acc_ci95_upper": float(boot["ci95_upper"]),
                    "chance_level": chance,
                    "decode_delay_minus_chance": float(acc - chance),
                    "p_one_sided_gt_chance": float(boot["p_one_sided_gt_chance"]),
                    "n_boot": int(n_boot),
                }
            )
    return pd.DataFrame(rows)


def summarize_phase_ratios(df_phase_rates: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    key_cols = ["trial_id", "stsp_mode", "sample_label", "probe_label", "layer"]
    df_trial = df_phase_rates[key_cols + ["ratio_delay_over_sample", "ratio_delay_over_probe"]].drop_duplicates(key_cols)
    rows: List[Dict[str, float]] = []

    for layer_idx, layer_name in enumerate(LAYER_KEYS):
        for mode_idx, stsp_mode in enumerate(STSP_MODES):
            sub_trial = df_trial[(df_trial["layer"] == layer_name) & (df_trial["stsp_mode"] == stsp_mode)].copy()
            if len(sub_trial) == 0:
                continue

            ratio_sample = sub_trial["ratio_delay_over_sample"].to_numpy(dtype=np.float64)
            ratio_probe = sub_trial["ratio_delay_over_probe"].to_numpy(dtype=np.float64)
            ratio_sample_ci = bootstrap_mean_ci(ratio_sample, n_boot=n_boot, seed=seed + 211 + layer_idx * 100 + mode_idx * 17)
            ratio_probe_ci = bootstrap_mean_ci(ratio_probe, n_boot=n_boot, seed=seed + 311 + layer_idx * 100 + mode_idx * 17)

            rate_lookup = (
                df_phase_rates[
                    (df_phase_rates["layer"] == layer_name)
                    & (df_phase_rates["stsp_mode"] == stsp_mode)
                ]
                .groupby("phase")["rate_spikes_per_neuron_step"]
                .apply(lambda x: x.to_numpy(dtype=np.float64))
                .to_dict()
            )
            sample_rate = rate_lookup["sample"]
            delay_rate = rate_lookup["delay"]
            probe_rate = rate_lookup["probe"]
            sample_rate_ci = bootstrap_mean_ci(sample_rate, n_boot=n_boot, seed=seed + 411 + layer_idx * 100 + mode_idx * 17)
            delay_rate_ci = bootstrap_mean_ci(delay_rate, n_boot=n_boot, seed=seed + 511 + layer_idx * 100 + mode_idx * 17)
            probe_rate_ci = bootstrap_mean_ci(probe_rate, n_boot=n_boot, seed=seed + 611 + layer_idx * 100 + mode_idx * 17)

            rows.append(
                {
                    "layer": layer_name,
                    "stsp_mode": stsp_mode,
                    "n_trials": int(len(sub_trial)),
                    "ratio_delay_over_sample_mean": float(ratio_sample.mean()),
                    "ratio_delay_over_sample_ci95_lower": float(ratio_sample_ci[0]),
                    "ratio_delay_over_sample_ci95_upper": float(ratio_sample_ci[1]),
                    "ratio_delay_over_probe_mean": float(ratio_probe.mean()),
                    "ratio_delay_over_probe_ci95_lower": float(ratio_probe_ci[0]),
                    "ratio_delay_over_probe_ci95_upper": float(ratio_probe_ci[1]),
                    "sample_rate_mean": float(sample_rate.mean()),
                    "sample_rate_ci95_lower": float(sample_rate_ci[0]),
                    "sample_rate_ci95_upper": float(sample_rate_ci[1]),
                    "delay_rate_mean": float(delay_rate.mean()),
                    "delay_rate_ci95_lower": float(delay_rate_ci[0]),
                    "delay_rate_ci95_upper": float(delay_rate_ci[1]),
                    "probe_rate_mean": float(probe_rate.mean()),
                    "probe_rate_ci95_lower": float(probe_rate_ci[0]),
                    "probe_rate_ci95_upper": float(probe_rate_ci[1]),
                    "n_boot": int(n_boot),
                }
            )
    return pd.DataFrame(rows)


def plot_delay_decode_by_layer(df_decode: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL_COMPACT, sharey=True)
    color_map = dict(SILENT_MEMORY_MODE_COLORS)
    label_map = {"dynamic": "Dynamic", "static_frozen": "Static"}

    for ax, layer_name in zip(axes, LAYER_KEYS):
        sub = df_decode[df_decode["layer"] == layer_name].copy()
        sub["order"] = sub["stsp_mode"].map({"dynamic": 0, "static_frozen": 1})
        sub = sub.sort_values("order")
        x = np.arange(len(sub), dtype=np.float64)
        vals = sub["decode_delay_acc"].to_numpy(dtype=np.float64)
        lower = sub["decode_delay_acc_ci95_lower"].to_numpy(dtype=np.float64)
        upper = sub["decode_delay_acc_ci95_upper"].to_numpy(dtype=np.float64)
        yerr = np.vstack([vals - lower, upper - vals])
        colors = [color_map[str(m)] for m in sub["stsp_mode"]]

        bars = ax.bar(x, vals, yerr=yerr, color=colors, edgecolor="black", alpha=ALPHA_BAR, capsize=5)
        chance = float(sub["chance_level"].iloc[0])
        ax.axhline(chance, color="black", linestyle="--", linewidth=LINE_WIDTH_REFERENCE, alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
        ax.set_xticks(x, [label_map[str(m)] for m in sub["stsp_mode"]])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{layer_name.upper()} delay decode")
        ax.text(
            0.03,
            0.95,
            f"Chance={chance:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
                bbox=dict(facecolor="white", edgecolor="gray", alpha=ALPHA_ANNOTATION_BOX),
        )

    axes[0].set_ylabel("Sample-label decode accuracy")
    fig.suptitle("Panel C: Delay-period spike decoding stays near chance", y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def plot_delay_ratio_distribution(df_phase_rates: pd.DataFrame) -> plt.Figure:
    key_cols = ["trial_id", "stsp_mode", "sample_label", "probe_label", "layer"]
    df_plot = df_phase_rates[key_cols + ["ratio_delay_over_sample"]].drop_duplicates(key_cols)
    label_map = {"dynamic": "Dynamic", "static_frozen": "Static"}
    df_plot["mode_label"] = df_plot["stsp_mode"].map(label_map)

    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL_COMPACT, sharey=False)
    for ax, layer_name in zip(axes, LAYER_KEYS):
        sub = df_plot[df_plot["layer"] == layer_name].copy()
        sns.boxplot(
            data=sub,
            x="mode_label",
            y="ratio_delay_over_sample",
            order=["Dynamic", "Static"],
            palette=[SILENT_MEMORY_MODE_COLORS["dynamic"], SILENT_MEMORY_MODE_COLORS["static_frozen"]],
            ax=ax,
            width=0.55,
            fliersize=2.5,
        )
        ax.set_title(f"{layer_name.upper()} delay/sample ratio")
        ax.set_xlabel("")
        ax.set_ylabel("ratio_delay_over_sample")
    fig.suptitle("Supplementary: Delay/sample firing-rate ratio across trials", y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def plot_phase_firing_rates_by_layer(df_ratio: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL_MEDIUM, sharey=False)
    phase_order = ["sample", "delay", "probe"]
    phase_labels = ["Sample", "Delay", "Probe"]
    color_map = dict(SILENT_MEMORY_MODE_COLORS)
    label_map = {"dynamic": "Dynamic", "static_frozen": "Static"}
    width = 0.34

    for ax, layer_name in zip(axes, LAYER_KEYS):
        sub = df_ratio[df_ratio["layer"] == layer_name].copy()
        if len(sub) == 0:
            continue
        sub["order"] = sub["stsp_mode"].map({"dynamic": 0, "static_frozen": 1})
        sub = sub.sort_values("order")
        x = np.arange(len(phase_order), dtype=np.float64)

        for offset_idx, stsp_mode in enumerate(STSP_MODES):
            row = sub[sub["stsp_mode"] == stsp_mode]
            if len(row) != 1:
                continue
            row = row.iloc[0]
            vals = np.array(
                [
                    float(row["sample_rate_mean"]),
                    float(row["delay_rate_mean"]),
                    float(row["probe_rate_mean"]),
                ],
                dtype=np.float64,
            )
            lower = np.array(
                [
                    float(row["sample_rate_ci95_lower"]),
                    float(row["delay_rate_ci95_lower"]),
                    float(row["probe_rate_ci95_lower"]),
                ],
                dtype=np.float64,
            )
            upper = np.array(
                [
                    float(row["sample_rate_ci95_upper"]),
                    float(row["delay_rate_ci95_upper"]),
                    float(row["probe_rate_ci95_upper"]),
                ],
                dtype=np.float64,
            )
            yerr = np.vstack([vals - lower, upper - vals])
            pos = x + (offset_idx - 0.5) * width + width / 2.0
            ax.bar(
                pos,
                vals,
                width=width,
                yerr=yerr,
                color=color_map[stsp_mode],
                edgecolor="black",
                alpha=ALPHA_BAR,
                capsize=4,
                label=label_map[stsp_mode],
            )

        ax.set_xticks(x, phase_labels)
        ax.set_title(f"{layer_name.upper()} phase firing rate")
        ax.set_ylabel("Spike rate / neuron / step")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        apply_standard_figure_legend(fig, handles, labels, ncol=2, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle("Phase firing rates across sample, delay, and probe", y=1.14, fontsize=13)
    fig.tight_layout()
    return fig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Population-level DMS activity-silent quantification.")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--save-dir", type=str, default="results/activity_silent_population_dms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-splits", type=int, default=5)
    parser.add_argument("--num-boot", type=int, default=1000)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--delay-ms", type=float, default=400.0)
    parser.add_argument("--probe-ms", type=float, default=60.0)
    parser.add_argument("--no-phase-reset", action="store_true")
    return parser


seed_everything = shared_seed_everything
build_class_index = shared_build_class_index
encode_images = shared_encode_images


def compensate_stsp_gain(net: SDNN_Network, scaling_factor: float) -> None:
    shared_compensate_stsp_gain(net, scaling_factor=scaling_factor)


def load_model_and_encoder(
    model_path: str,
    device: torch.device,
    spec: ExperimentSpec,
) -> Tuple[SDNN_Network, DoGSpikeEncoder]:
    return shared_load_model_and_encoder(
        model_path=model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(spec.sample_ms, spec.probe_ms),
    )


def decode_accuracy_with_splits(
    x: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
) -> float:
    return shared_decode_accuracy_with_splits(
        x=x,
        y=y,
        splits=splits,
        num_classes=num_classes,
        device=None,
    )


def main() -> None:
    args = build_argparser().parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.decode_splits <= 0:
        raise ValueError("--decode-splits must be positive")
    if args.num_boot <= 0:
        raise ValueError("--num-boot must be positive")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = ExperimentSpec(
        dt=1.0 * ms,
        sample_ms=args.sample_ms,
        delay_ms=args.delay_ms,
        probe_ms=args.probe_ms,
        phase_reset=(not args.no_phase_reset),
    )
    for name, steps in [("sample", spec.sample_steps), ("delay", spec.delay_steps), ("probe", spec.probe_steps)]:
        if steps <= 0:
            raise ValueError(f"{name} steps must be positive")

    layout = prepare_result_layout(args.save_dir)

    print(f"[Init] Device: {device}")
    print(f"[Init] Save dir: {layout.root}")
    print(
        f"[Init] Timing steps: sample={spec.sample_steps}, delay={spec.delay_steps}, "
        f"probe={spec.probe_steps}, total={spec.total_steps}"
    )
    print(
        f"[Init] Trials={args.trials} | batch={args.batch_size} | decode_splits={args.decode_splits} | "
        f"boot={args.num_boot}"
    )

    net, encoder = load_model_and_encoder(args.model_path, device, spec)
    _, _, test_loader = build_mnist_skeleton_loader(batch_size=1)
    dataset = test_loader.dataset
    class_index = build_class_index(dataset, num_classes=args.num_classes)
    rng = random.Random(args.seed)
    df_specs = generate_balanced_dms_trial_specs(
        class_index=class_index,
        num_trials=args.trials,
        num_classes=args.num_classes,
        rng=rng,
    )
    validate_trial_specs(df_specs, num_classes=args.num_classes)

    feature_buf: Dict[str, Dict[str, List[np.ndarray]]] = {
        stsp_mode: {layer_name: [] for layer_name in LAYER_KEYS} for stsp_mode in STSP_MODES
    }
    phase_tables: List[pd.DataFrame] = []

    starts = range(0, len(df_specs), args.batch_size)
    for start in tqdm(starts, desc="Population DMS batches"):
        batch_df = df_specs.iloc[start:start + args.batch_size].copy()
        sample_imgs = torch.stack([dataset[int(i)][0] for i in batch_df["sample_index"].tolist()], dim=0).to(device)
        probe_imgs = torch.stack([dataset[int(i)][0] for i in batch_df["probe_index"].tolist()], dim=0).to(device)
        sample_spikes = encode_images(encoder, sample_imgs, spec.sample_steps)
        probe_spikes = encode_images(encoder, probe_imgs, spec.probe_steps)

        for stsp_mode in STSP_MODES:
            with torch.no_grad():
                trace = net.forward_dms_spike_trace_session(
                    sample_spikes=sample_spikes,
                    probe_spikes=probe_spikes,
                    delay_steps=spec.delay_steps,
                    stsp_mode=stsp_mode,
                    phase_reset=spec.phase_reset,
                )

            phase_slices = trace["phase_slices"]
            for phase in ["sample", "delay", "probe"]:
                if phase not in phase_slices:
                    raise ValueError(f"Missing phase slice: {phase}")

            for layer_name in LAYER_KEYS:
                spikes = trace[f"{layer_name}_spikes"]
                feature_buf[stsp_mode][layer_name].append(extract_delay_features(spikes, phase_slices=phase_slices))
                phase_tables.append(
                    build_trial_phase_rate_table(
                        spikes=spikes,
                        phase_slices=phase_slices,
                        layer_name=layer_name,
                        batch_df=batch_df,
                        stsp_mode=stsp_mode,
                    )
                )

    df_phase_rates = pd.concat(phase_tables, axis=0, ignore_index=True)
    labels = df_specs["sample_label"].to_numpy(dtype=np.int64)
    df_decode = summarize_delay_decode(
        feature_buf=feature_buf,
        labels=labels,
        num_classes=args.num_classes,
        decode_splits=args.decode_splits,
        n_boot=args.num_boot,
        seed=args.seed,
    )
    df_ratio = summarize_phase_ratios(df_phase_rates=df_phase_rates, n_boot=args.num_boot, seed=args.seed)

    trial_specs_csv = save_tidy_csv(df_specs, layout.data_file("trial_specs.csv"), sort_by=["trial_id"])
    phase_csv = save_tidy_csv(
        df_phase_rates,
        layout.data_file("trial_phase_rates.csv"),
        sort_by=["trial_id", "stsp_mode", "layer", "phase"],
    )
    decode_csv = save_tidy_csv(
        df_decode,
        layout.data_file("metrics_delay_decode_by_layer.csv"),
        sort_by=["layer", "stsp_mode"],
    )
    ratio_csv = save_tidy_csv(
        df_ratio,
        layout.data_file("metrics_phase_ratio_by_layer.csv"),
        sort_by=["layer", "stsp_mode"],
    )

    fig_decode = plot_delay_decode_by_layer(df_decode=df_decode)
    decode_paths = save_figure_all_formats(fig_decode, layout.figure_base("delay_decode_by_layer"))
    plt.close(fig_decode)
    fig_ratio = plot_delay_ratio_distribution(df_phase_rates=df_phase_rates)
    ratio_paths = save_figure_all_formats(fig_ratio, layout.figure_base("delay_ratio_distribution"))
    plt.close(fig_ratio)
    fig_phase = plot_phase_firing_rates_by_layer(df_ratio=df_ratio)
    phase_paths = save_figure_all_formats(fig_phase, layout.figure_base("phase_firing_rates_by_layer"))
    plt.close(fig_phase)

    config = {
        "model_path": args.model_path,
        "device": str(device),
        "seed": int(args.seed),
        "num_classes": int(args.num_classes),
        "trials": int(args.trials),
        "batch_size": int(args.batch_size),
        "decode_splits": int(args.decode_splits),
        "num_boot": int(args.num_boot),
        "phase_reset": bool(spec.phase_reset),
        "timing_ms": {
            "sample": float(args.sample_ms),
            "delay": float(args.delay_ms),
            "probe": float(args.probe_ms),
        },
        "timing_steps": {
            "sample": int(spec.sample_steps),
            "delay": int(spec.delay_steps),
            "probe": int(spec.probe_steps),
            "total": int(spec.total_steps),
        },
    }
    summary_path = save_summary_json(
        {
            "experiment": "activity_silent_population_dms",
            "key_metrics": {
                layer_name: {
                    "dynamic_delay_decode": float(
                        df_decode[(df_decode["layer"] == layer_name) & (df_decode["stsp_mode"] == "dynamic")]["decode_delay_acc"].iloc[0]
                    )
                    if not df_decode[(df_decode["layer"] == layer_name) & (df_decode["stsp_mode"] == "dynamic")].empty
                    else float("nan"),
                    "static_delay_decode": float(
                        df_decode[(df_decode["layer"] == layer_name) & (df_decode["stsp_mode"] == "static_frozen")]["decode_delay_acc"].iloc[0]
                    )
                    if not df_decode[(df_decode["layer"] == layer_name) & (df_decode["stsp_mode"] == "static_frozen")].empty
                    else float("nan"),
                }
                for layer_name in LAYER_KEYS
            },
            "outputs": {
                "trial_specs_csv": str(trial_specs_csv),
                "trial_phase_rates_csv": str(phase_csv),
                "metrics_delay_decode_by_layer_csv": str(decode_csv),
                "metrics_phase_ratio_by_layer_csv": str(ratio_csv),
                "figure_decode_png": decode_paths["png"],
                "figure_ratio_png": ratio_paths["png"],
                "figure_phase_png": phase_paths["png"],
            },
        },
        layout.root,
    )
    run_config_path = save_run_config(config, layout.root)
    run_log_path = save_log_lines(
        [
            "experiment=activity_silent_population_dms",
            f"save_dir={layout.root}",
            f"trial_specs_csv={trial_specs_csv}",
            f"trial_phase_rates_csv={phase_csv}",
            f"metrics_delay_decode_by_layer_csv={decode_csv}",
            f"metrics_phase_ratio_by_layer_csv={ratio_csv}",
            f"figure_decode_png={decode_paths['png']}",
            f"figure_ratio_png={ratio_paths['png']}",
            f"figure_phase_png={phase_paths['png']}",
            f"summary_json={summary_path}",
            f"run_config_json={run_config_path}",
        ],
        layout.log_dir,
    )

    print("\n=== Population DMS Activity-Silent Summary ===")
    for layer_name in LAYER_KEYS:
        sub = df_decode[df_decode["layer"] == layer_name].sort_values("stsp_mode")
        if len(sub) != 2:
            continue
        dyn = sub[sub["stsp_mode"] == "dynamic"].iloc[0]
        stat = sub[sub["stsp_mode"] == "static_frozen"].iloc[0]
        print(
            f"{layer_name}: delay decode dynamic/static = "
            f"{float(dyn['decode_delay_acc']):.4f} / {float(stat['decode_delay_acc']):.4f} "
            f"(chance={float(dyn['chance_level']):.4f})"
        )
    print(f"Saved: {trial_specs_csv}")
    print(f"Saved: {phase_csv}")
    print(f"Saved: {decode_csv}")
    print(f"Saved: {ratio_csv}")
    print(f"Saved: {decode_paths['png']}")
    print(f"Saved: {ratio_paths['png']}")
    print(f"Saved: {phase_paths['png']}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {run_config_path}")
    print(f"Saved: {run_log_path}")


if __name__ == "__main__":
    main()
