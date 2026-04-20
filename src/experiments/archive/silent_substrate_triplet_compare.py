from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.svm import LinearSVC
from tqdm import tqdm

from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.config.units import ms
from src.experiments.silent_memory.shared.population_dms import (
    build_class_index,
    encode_images,
    generate_balanced_dms_trial_specs,
    validate_trial_specs,
)
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import run_monitored_dms_rollout
from src.experiments.common.ping_common import LAYER_KEYS
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_run_config, save_summary_json
from src.experiments.common.runtime import resolve_device, seed_everything
from src.plotting.common.io import apply_publication_style, save_figure_all_formats, save_tidy_csv
from src.plotting.common.theme_tokens import (
    ALPHA_BAR,
    FIGSIZE_THREE_PANEL_COMPACT,
    LINE_WIDTH_SECONDARY,
    MARKER_CIRCLE,
    SUBSTRATE_COLORS,
    apply_standard_figure_legend,
)

SUBSTRATE_ORDER = ["spike", "membrane", "stsp"]
SUBSTRATE_LABELS = {
    "spike": "Spikes",
    "membrane": "Membrane",
    "stsp": "STSP",
}
DEFAULT_SAVE_DIR = "results/fig5_silent_substrate_triplet"


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def delay_steps(self) -> int:
        return int(round((self.delay_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare silent-memory substrates across spikes, membrane, and STSP.")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--train-trials", type=int, default=300)
    parser.add_argument("--test-trials", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--delay-ms", type=float, default=500.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--bin-ms", type=float, default=20.0)
    parser.add_argument("--stride-ms", type=float, default=10.0)
    parser.add_argument("--num-boot", type=int, default=1000)
    parser.add_argument("--num-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def _bootstrap_ci(binary_values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    values = np.asarray(binary_values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("bootstrap ci received empty values")
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample_idx = rng.integers(0, len(values), size=len(values))
        boot[idx] = float(values[sample_idx].mean())
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _permutation_test_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_perm: int, seed: int) -> float:
    observed = float(np.mean(y_true == y_pred))
    rng = np.random.default_rng(seed)
    perm_scores = np.zeros(n_perm, dtype=np.float64)
    for idx in range(n_perm):
        perm_scores[idx] = float(np.mean(rng.permutation(y_true) == y_pred))
    return float((np.sum(perm_scores >= observed) + 1.0) / (len(perm_scores) + 1.0))


def _build_delay_bins(delay_steps: int, bin_steps: int, stride_steps: int, dt_ms: float) -> List[Dict[str, float]]:
    bins: List[Dict[str, float]] = []
    for start in range(0, max(1, delay_steps - bin_steps + 1), stride_steps):
        end = start + bin_steps
        if end > delay_steps:
            break
        bins.append(
            {
                "start_step": int(start),
                "end_step": int(end),
                "start_ms": float(start * dt_ms),
                "end_ms": float(end * dt_ms),
            }
        )
    if not bins:
        raise ValueError("No delay bins were generated; check bin-ms/stride-ms/delay-ms.")
    return bins


def _find_anchor_bin_index(delay_bins: Sequence[Mapping[str, float]], anchor_step: int) -> int:
    for idx, bin_info in enumerate(delay_bins):
        if int(bin_info["start_step"]) <= int(anchor_step) < int(bin_info["end_step"]):
            return int(idx)
    return int(
        min(
            range(len(delay_bins)),
            key=lambda idx: abs(int(delay_bins[idx]["start_step"]) - int(anchor_step)),
        )
    )


def _extract_bin_feature(
    layer_traces: Mapping[str, torch.Tensor],
    substrate: str,
    delay_slice: slice,
    start_step: int,
    end_step: int,
) -> np.ndarray:
    local_slice = slice(delay_slice.start + start_step, delay_slice.start + end_step)
    if substrate == "spike":
        seg = layer_traces["spikes"][local_slice].to(torch.float32)
        feat = seg.permute(1, 0, 2, 3, 4).reshape(seg.shape[1], seg.shape[0], -1).sum(dim=1)
        return feat.numpy().astype(np.float32, copy=False)
    if substrate == "membrane":
        seg = layer_traces["v_raw"][local_slice]
        feat = seg.permute(1, 0, 2, 3, 4).reshape(seg.shape[1], seg.shape[0], -1).mean(dim=1)
        return feat.numpy().astype(np.float32, copy=False)
    if substrate == "stsp":
        u_seg = layer_traces["u"][local_slice]
        x_seg = layer_traces["x"][local_slice]
        u_feat = u_seg.permute(1, 0, 2, 3, 4).reshape(u_seg.shape[1], u_seg.shape[0], -1).mean(dim=1)
        x_feat = x_seg.permute(1, 0, 2, 3, 4).reshape(x_seg.shape[1], x_seg.shape[0], -1).mean(dim=1)
        feat = torch.cat([u_feat, x_feat], dim=1)
        return feat.numpy().astype(np.float32, copy=False)
    raise ValueError(f"Unsupported substrate: {substrate}")


def _extract_whole_delay_feature(
    layer_traces: Mapping[str, torch.Tensor],
    substrate: str,
    delay_slice: slice,
) -> np.ndarray:
    return _extract_bin_feature(
        layer_traces=layer_traces,
        substrate=substrate,
        delay_slice=delay_slice,
        start_step=0,
        end_step=int(delay_slice.stop - delay_slice.start),
    )


def _top1_top2_margin(clf: LinearSVC, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scores = clf.decision_function(features)
    if scores.ndim == 1:
        pred_idx = (scores > 0).astype(np.int64)
        pred = clf.classes_[pred_idx]
        margin = np.abs(scores).astype(np.float32, copy=False)
        return pred.astype(np.int64, copy=False), margin
    pred_idx = np.argmax(scores, axis=1)
    sorted_scores = np.sort(scores, axis=1)
    margin = (sorted_scores[:, -1] - sorted_scores[:, -2]).astype(np.float32, copy=False)
    pred = clf.classes_[pred_idx]
    return pred.astype(np.int64, copy=False), margin


def _encode_batch_specs(dataset, batch_df: pd.DataFrame, device: torch.device, encoder, spec: ExperimentSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    sample_imgs = torch.stack([dataset[int(i)][0] for i in batch_df["sample_index"].tolist()], dim=0).to(device)
    probe_imgs = torch.stack([dataset[int(i)][0] for i in batch_df["probe_index"].tolist()], dim=0).to(device)
    sample_spikes = encode_images(encoder, sample_imgs, spec.sample_steps)
    probe_spikes = encode_images(encoder, probe_imgs, spec.probe_steps)
    return sample_spikes, probe_spikes


def _compute_anchor_steps(
    net,
    encoder,
    dataset,
    df_specs: pd.DataFrame,
    spec: ExperimentSpec,
    batch_size: int,
    device: torch.device,
) -> Dict[str, int]:
    voltage_sum = {
        layer_key: np.zeros(spec.delay_steps, dtype=np.float64)
        for layer_key in LAYER_KEYS
    }
    total_trials = 0
    starts = range(0, len(df_specs), batch_size)
    for start in tqdm(starts, desc="Anchor pass"):
        batch_df = df_specs.iloc[start:start + batch_size].copy().reset_index(drop=True)
        sample_spikes, probe_spikes = _encode_batch_specs(dataset, batch_df, device, encoder, spec)
        with torch.no_grad():
            out = run_monitored_dms_rollout(
                net=net,
                sample_spikes=sample_spikes,
                probe_spikes=probe_spikes,
                delay_steps=spec.delay_steps,
                stsp_mode="dynamic",
                record_state_names=("v_raw",),
            )
        phase_slices = out["phase_slices"]
        delay_slice = slice(phase_slices["delay"][0], phase_slices["delay"][1])
        batch_trials = len(batch_df)
        total_trials += batch_trials
        for layer_key in LAYER_KEYS:
            v_trace = out["state_traces"][layer_key]["v_raw"][delay_slice]
            summed = v_trace.reshape(v_trace.shape[0], v_trace.shape[1], -1).sum(dim=2).mean(dim=1).numpy()
            voltage_sum[layer_key] += summed * float(batch_trials)
    if total_trials <= 0:
        raise ValueError("No trials were available to compute anchor steps.")
    return {
        layer_key: int(np.argmax(voltage_sum[layer_key] / float(total_trials)))
        for layer_key in LAYER_KEYS
    }


def _collect_training_features(
    net,
    encoder,
    dataset,
    df_specs: pd.DataFrame,
    spec: ExperimentSpec,
    batch_size: int,
    device: torch.device,
    delay_bins: Sequence[Mapping[str, float]],
    anchor_step_by_layer: Mapping[str, int],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], np.ndarray]:
    anchor_bin_idx = {
        layer_key: _find_anchor_bin_index(delay_bins, anchor_step_by_layer[layer_key])
        for layer_key in LAYER_KEYS
    }
    anchor_buf: Dict[str, Dict[str, List[np.ndarray]]] = {
        layer_key: {substrate: [] for substrate in SUBSTRATE_ORDER}
        for layer_key in LAYER_KEYS
    }
    summary_buf: Dict[str, Dict[str, List[np.ndarray]]] = {
        layer_key: {substrate: [] for substrate in SUBSTRATE_ORDER}
        for layer_key in LAYER_KEYS
    }
    labels_all: List[np.ndarray] = []

    starts = range(0, len(df_specs), batch_size)
    for start in tqdm(starts, desc="Train feature pass"):
        batch_df = df_specs.iloc[start:start + batch_size].copy().reset_index(drop=True)
        sample_spikes, probe_spikes = _encode_batch_specs(dataset, batch_df, device, encoder, spec)
        with torch.no_grad():
            out = run_monitored_dms_rollout(
                net=net,
                sample_spikes=sample_spikes,
                probe_spikes=probe_spikes,
                delay_steps=spec.delay_steps,
                stsp_mode="dynamic",
                record_state_names=("spikes", "v_raw", "u", "x"),
            )
        phase_slices = out["phase_slices"]
        delay_slice = slice(phase_slices["delay"][0], phase_slices["delay"][1])
        labels_all.append(batch_df["sample_label"].to_numpy(dtype=np.int64))
        for layer_key in LAYER_KEYS:
            layer_traces = out["state_traces"][layer_key]
            anchor_bin = delay_bins[anchor_bin_idx[layer_key]]
            for substrate in SUBSTRATE_ORDER:
                anchor_buf[layer_key][substrate].append(
                    _extract_bin_feature(
                        layer_traces=layer_traces,
                        substrate=substrate,
                        delay_slice=delay_slice,
                        start_step=int(anchor_bin["start_step"]),
                        end_step=int(anchor_bin["end_step"]),
                    )
                )
                summary_buf[layer_key][substrate].append(
                    _extract_whole_delay_feature(
                        layer_traces=layer_traces,
                        substrate=substrate,
                        delay_slice=delay_slice,
                    )
                )

    packed_anchor = {
        layer_key: {
            substrate: np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
            for substrate, chunks in layer_map.items()
        }
        for layer_key, layer_map in anchor_buf.items()
    }
    packed_summary = {
        layer_key: {
            substrate: np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
            for substrate, chunks in layer_map.items()
        }
        for layer_key, layer_map in summary_buf.items()
    }
    labels = np.concatenate(labels_all, axis=0)
    return packed_anchor, packed_summary, labels


def _fit_decoders(
    anchor_features: Mapping[str, Mapping[str, np.ndarray]],
    summary_features: Mapping[str, Mapping[str, np.ndarray]],
    labels: np.ndarray,
) -> Tuple[Dict[str, Dict[str, LinearSVC]], Dict[str, Dict[str, LinearSVC]]]:
    anchor_clf: Dict[str, Dict[str, LinearSVC]] = {layer_key: {} for layer_key in LAYER_KEYS}
    summary_clf: Dict[str, Dict[str, LinearSVC]] = {layer_key: {} for layer_key in LAYER_KEYS}
    for layer_key in LAYER_KEYS:
        for substrate in SUBSTRATE_ORDER:
            anchor_model = LinearSVC(dual=False, C=1.0, max_iter=3000)
            anchor_model.fit(anchor_features[layer_key][substrate], labels)
            anchor_clf[layer_key][substrate] = anchor_model

            summary_model = LinearSVC(dual=False, C=1.0, max_iter=3000)
            summary_model.fit(summary_features[layer_key][substrate], labels)
            summary_clf[layer_key][substrate] = summary_model
    return anchor_clf, summary_clf


def _evaluate_test_features(
    net,
    encoder,
    dataset,
    df_specs: pd.DataFrame,
    spec: ExperimentSpec,
    batch_size: int,
    device: torch.device,
    delay_bins: Sequence[Mapping[str, float]],
    anchor_step_by_layer: Mapping[str, int],
    anchor_clf: Mapping[str, Mapping[str, LinearSVC]],
    summary_clf: Mapping[str, Mapping[str, LinearSVC]],
    num_classes: int,
    num_boot: int,
    num_perm: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    correct_by_bin: Dict[Tuple[str, str, int], List[np.ndarray]] = {}
    pred_by_bin: Dict[Tuple[str, str, int], List[np.ndarray]] = {}
    correct_summary: Dict[Tuple[str, str], List[np.ndarray]] = {}
    pred_summary: Dict[Tuple[str, str], List[np.ndarray]] = {}
    confidence_rows: List[Dict[str, float]] = []
    chance_level = 1.0 / float(num_classes)

    anchor_bin_idx = {
        layer_key: _find_anchor_bin_index(delay_bins, anchor_step_by_layer[layer_key])
        for layer_key in LAYER_KEYS
    }

    starts = range(0, len(df_specs), batch_size)
    for start in tqdm(starts, desc="Test evaluation"):
        batch_df = df_specs.iloc[start:start + batch_size].copy().reset_index(drop=True)
        sample_spikes, probe_spikes = _encode_batch_specs(dataset, batch_df, device, encoder, spec)
        with torch.no_grad():
            out = run_monitored_dms_rollout(
                net=net,
                sample_spikes=sample_spikes,
                probe_spikes=probe_spikes,
                delay_steps=spec.delay_steps,
                stsp_mode="dynamic",
                record_state_names=("spikes", "v_raw", "u", "x"),
            )
        phase_slices = out["phase_slices"]
        delay_slice = slice(phase_slices["delay"][0], phase_slices["delay"][1])
        labels = batch_df["sample_label"].to_numpy(dtype=np.int64)
        trial_ids = batch_df["trial_id"].to_numpy(dtype=np.int64)

        for layer_key in LAYER_KEYS:
            layer_traces = out["state_traces"][layer_key]
            anchor_ms = float(delay_bins[anchor_bin_idx[layer_key]]["start_ms"])
            for substrate in SUBSTRATE_ORDER:
                summary_feat = _extract_whole_delay_feature(layer_traces, substrate, delay_slice)
                pred_sum, margin_sum = _top1_top2_margin(summary_clf[layer_key][substrate], summary_feat)
                correct_summary.setdefault((layer_key, substrate), []).append((pred_sum == labels).astype(np.float32))
                pred_summary.setdefault((layer_key, substrate), []).append(pred_sum.astype(np.int64))
                for trial_id, true_label, pred_label, margin in zip(trial_ids, labels, pred_sum, margin_sum):
                    confidence_rows.append(
                        {
                            "trial_id": int(trial_id),
                            "layer": layer_key,
                            "substrate": substrate,
                            "eval_bin_start_ms": -1.0,
                            "window_name": "whole_delay",
                            "anchor_time_ms": anchor_ms,
                            "pred_label": int(pred_label),
                            "true_label": int(true_label),
                            "confidence_margin": float(margin),
                        }
                    )

                for bin_idx, bin_info in enumerate(delay_bins):
                    feat = _extract_bin_feature(
                        layer_traces=layer_traces,
                        substrate=substrate,
                        delay_slice=delay_slice,
                        start_step=int(bin_info["start_step"]),
                        end_step=int(bin_info["end_step"]),
                    )
                    pred_bin, margin_bin = _top1_top2_margin(anchor_clf[layer_key][substrate], feat)
                    correct_by_bin.setdefault((layer_key, substrate, bin_idx), []).append((pred_bin == labels).astype(np.float32))
                    pred_by_bin.setdefault((layer_key, substrate, bin_idx), []).append(pred_bin.astype(np.int64))
                    for trial_id, true_label, pred_label, margin in zip(trial_ids, labels, pred_bin, margin_bin):
                        confidence_rows.append(
                            {
                                "trial_id": int(trial_id),
                                "layer": layer_key,
                                "substrate": substrate,
                                "eval_bin_start_ms": float(bin_info["start_ms"]),
                                "window_name": "time_bin",
                                "anchor_time_ms": anchor_ms,
                                "pred_label": int(pred_label),
                                "true_label": int(true_label),
                                "confidence_margin": float(margin),
                            }
                        )

    time_rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []
    perm_rows: List[Dict[str, float]] = []
    labels_all = df_specs["sample_label"].to_numpy(dtype=np.int64)

    for layer_idx, layer_key in enumerate(LAYER_KEYS):
        anchor_ms = float(delay_bins[anchor_bin_idx[layer_key]]["start_ms"])
        for substrate_idx, substrate in enumerate(SUBSTRATE_ORDER):
            correct = np.concatenate(correct_summary[(layer_key, substrate)], axis=0)
            pred = np.concatenate(pred_summary[(layer_key, substrate)], axis=0)
            acc = float(correct.mean())
            ci_low, ci_high = _bootstrap_ci(
                correct,
                n_boot=num_boot,
                seed=seed + 5000 + layer_idx * 100 + substrate_idx * 13,
            )
            perm_p = _permutation_test_accuracy(
                labels_all,
                pred,
                n_perm=num_perm,
                seed=seed + 7000 + layer_idx * 100 + substrate_idx * 13,
            )
            summary_rows.append(
                {
                    "layer": layer_key,
                    "substrate": substrate,
                    "window_name": "whole_delay",
                    "anchor_time_ms": anchor_ms,
                    "decode_acc": acc,
                    "ci95_lower": ci_low,
                    "ci95_upper": ci_high,
                    "perm_p": perm_p,
                    "chance_level": chance_level,
                    "n_trials": int(len(labels_all)),
                }
            )
            perm_rows.append(
                {
                    "analysis": "whole_delay",
                    "layer": layer_key,
                    "substrate": substrate,
                    "target": "whole_delay",
                    "anchor_time_ms": anchor_ms,
                    "observed_acc": acc,
                    "chance_level": chance_level,
                    "perm_p": perm_p,
                }
            )

            for bin_idx, bin_info in enumerate(delay_bins):
                correct_bin = np.concatenate(correct_by_bin[(layer_key, substrate, bin_idx)], axis=0)
                pred_bin = np.concatenate(pred_by_bin[(layer_key, substrate, bin_idx)], axis=0)
                acc_bin = float(correct_bin.mean())
                ci_low_bin, ci_high_bin = _bootstrap_ci(
                    correct_bin,
                    n_boot=num_boot,
                    seed=seed + 1000 + layer_idx * 1000 + substrate_idx * 100 + bin_idx * 7,
                )
                perm_p_bin = _permutation_test_accuracy(
                    labels_all,
                    pred_bin,
                    n_perm=num_perm,
                    seed=seed + 3000 + layer_idx * 1000 + substrate_idx * 100 + bin_idx * 7,
                )
                time_rows.append(
                    {
                        "layer": layer_key,
                        "substrate": substrate,
                        "time_bin_start_ms": float(bin_info["start_ms"]),
                        "time_bin_end_ms": float(bin_info["end_ms"]),
                        "anchor_time_ms": anchor_ms,
                        "decode_acc": acc_bin,
                        "ci95_lower": ci_low_bin,
                        "ci95_upper": ci_high_bin,
                        "perm_p": perm_p_bin,
                        "chance_level": chance_level,
                        "n_trials": int(len(labels_all)),
                    }
                )
                perm_rows.append(
                    {
                        "analysis": "time_bin",
                        "layer": layer_key,
                        "substrate": substrate,
                        "target": f"{float(bin_info['start_ms']):g}-{float(bin_info['end_ms']):g}ms",
                        "anchor_time_ms": anchor_ms,
                        "observed_acc": acc_bin,
                        "chance_level": chance_level,
                        "perm_p": perm_p_bin,
                    }
                )

    return (
        pd.DataFrame(time_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(confidence_rows),
        pd.DataFrame(perm_rows),
    )


def plot_decode_window_summary(df_summary: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, len(LAYER_KEYS), figsize=FIGSIZE_THREE_PANEL_COMPACT, sharey=True)
    if len(LAYER_KEYS) == 1:
        axes = [axes]
    color_map = dict(SUBSTRATE_COLORS)

    for ax, layer in zip(axes, LAYER_KEYS):
        sub = df_summary[df_summary["layer"] == layer].copy()
        window_order = list(dict.fromkeys(sub["window_name"].tolist()))
        x = np.arange(len(window_order), dtype=np.float64)
        width = 0.24
        for idx, substrate in enumerate(SUBSTRATE_ORDER):
            row = sub[sub["substrate"] == substrate].set_index("window_name").reindex(window_order).reset_index()
            vals = row["decode_acc"].to_numpy(dtype=np.float64)
            lower = row["ci95_lower"].to_numpy(dtype=np.float64)
            upper = row["ci95_upper"].to_numpy(dtype=np.float64)
            yerr = np.vstack([vals - lower, upper - vals])
            pos = x + (idx - 1) * width
            ax.bar(
                pos,
                vals,
                width=width,
                yerr=yerr,
                capsize=4,
                color=color_map[substrate],
                edgecolor="black",
                alpha=ALPHA_BAR,
                label=SUBSTRATE_LABELS[substrate],
            )
        ax.set_xticks(x, window_order, rotation=20)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(layer.upper())
        ax.set_ylabel("Decode accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        apply_standard_figure_legend(fig, handles, labels, ncol=3, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Silent substrate decoding across delay windows", y=1.08)
    fig.tight_layout()
    return fig


def plot_confidence_margin(df_confidence: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, len(LAYER_KEYS), figsize=FIGSIZE_THREE_PANEL_COMPACT, sharey=True)
    if len(LAYER_KEYS) == 1:
        axes = [axes]
    color_map = dict(SUBSTRATE_COLORS)
    grouped = (
        df_confidence.groupby(["layer", "substrate", "eval_bin_start_ms"], as_index=False)["confidence_margin"]
        .mean()
        .sort_values(["layer", "substrate", "eval_bin_start_ms"], kind="stable")
    )

    for ax, layer in zip(axes, LAYER_KEYS):
        sub = grouped[grouped["layer"] == layer].copy()
        for substrate in SUBSTRATE_ORDER:
            row = sub[sub["substrate"] == substrate]
            if row.empty:
                continue
            ax.plot(
                row["eval_bin_start_ms"].to_numpy(dtype=np.float64),
                row["confidence_margin"].to_numpy(dtype=np.float64),
                marker=MARKER_CIRCLE,
                linewidth=LINE_WIDTH_SECONDARY,
                markersize=4.0,
                color=color_map[substrate],
                label=SUBSTRATE_LABELS[substrate],
            )
        ax.set_title(layer.upper())
        ax.set_xlabel("Delay bin start (ms)")
        ax.set_ylabel("Mean confidence margin")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        apply_standard_figure_legend(fig, handles, labels, ncol=3, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Delay-time confidence by substrate", y=1.08)
    fig.tight_layout()
    return fig


def main() -> None:
    args = build_argparser().parse_args()
    if args.train_trials <= 0 or args.test_trials <= 0:
        raise ValueError("train-trials and test-trials must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_classes < 3:
        raise ValueError("num-classes must be >= 3")
    if args.bin_ms <= 0 or args.stride_ms <= 0:
        raise ValueError("bin-ms and stride-ms must be positive")
    if args.num_boot <= 0 or args.num_perm <= 0:
        raise ValueError("num-boot and num-perm must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    spec = ExperimentSpec(
        dt=float(args.dt_ms * ms),
        sample_ms=float(args.sample_ms),
        delay_ms=float(args.delay_ms),
        probe_ms=float(args.probe_ms),
    )
    for name, steps in [("sample", spec.sample_steps), ("delay", spec.delay_steps), ("probe", spec.probe_steps)]:
        if steps <= 0:
            raise ValueError(f"{name} steps must be positive")

    bin_steps = int(round((float(args.bin_ms) * ms) / spec.dt))
    stride_steps = int(round((float(args.stride_ms) * ms) / spec.dt))
    if bin_steps <= 0 or stride_steps <= 0:
        raise ValueError("bin-ms and stride-ms must map to positive step counts")
    if bin_steps > spec.delay_steps:
        raise ValueError("bin-ms cannot exceed the delay duration")

    layout = prepare_result_layout(args.save_dir)
    dt_ms = float(spec.dt / ms)
    delay_bins = _build_delay_bins(spec.delay_steps, bin_steps=bin_steps, stride_steps=stride_steps, dt_ms=dt_ms)

    print(f"[Init] Device: {device}")
    print(f"[Init] Save dir: {layout.root}")
    print(
        f"[Init] Timing | sample={spec.sample_steps} steps, delay={spec.delay_steps} steps, "
        f"probe={spec.probe_steps} steps | bin={bin_steps} | stride={stride_steps}"
    )

    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(spec.sample_ms, spec.probe_ms, spec.delay_ms),
    )
    train_loader, _, test_loader = build_mnist_skeleton_loader(
        root=args.dataset_root,
        batch_size=1,
        input_size=28,
    )
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    train_index = build_class_index(train_dataset, num_classes=args.num_classes)
    test_index = build_class_index(test_dataset, num_classes=args.num_classes)
    df_train_specs = generate_balanced_dms_trial_specs(
        class_index=train_index,
        num_trials=args.train_trials,
        num_classes=args.num_classes,
        rng=random.Random(args.seed),
    )
    df_test_specs = generate_balanced_dms_trial_specs(
        class_index=test_index,
        num_trials=args.test_trials,
        num_classes=args.num_classes,
        rng=random.Random(args.seed + 1),
    )
    validate_trial_specs(df_train_specs, num_classes=args.num_classes)
    validate_trial_specs(df_test_specs, num_classes=args.num_classes)

    anchor_step_by_layer = _compute_anchor_steps(
        net=net,
        encoder=encoder,
        dataset=train_dataset,
        df_specs=df_train_specs,
        spec=spec,
        batch_size=args.batch_size,
        device=device,
    )
    anchor_features, summary_features, train_labels = _collect_training_features(
        net=net,
        encoder=encoder,
        dataset=train_dataset,
        df_specs=df_train_specs,
        spec=spec,
        batch_size=args.batch_size,
        device=device,
        delay_bins=delay_bins,
        anchor_step_by_layer=anchor_step_by_layer,
    )
    anchor_clf, summary_clf = _fit_decoders(anchor_features, summary_features, train_labels)
    df_time, df_summary, df_confidence, df_perm = _evaluate_test_features(
        net=net,
        encoder=encoder,
        dataset=test_dataset,
        df_specs=df_test_specs,
        spec=spec,
        batch_size=args.batch_size,
        device=device,
        delay_bins=delay_bins,
        anchor_step_by_layer=anchor_step_by_layer,
        anchor_clf=anchor_clf,
        summary_clf=summary_clf,
        num_classes=args.num_classes,
        num_boot=args.num_boot,
        num_perm=args.num_perm,
        seed=args.seed,
    )

    trial_specs = pd.concat(
        [
            df_train_specs.assign(split="train"),
            df_test_specs.assign(split="test"),
        ],
        axis=0,
        ignore_index=True,
    )

    trial_specs_csv = save_tidy_csv(trial_specs, layout.data_file("trial_specs.csv"), sort_by=["split", "trial_id"])
    time_csv = save_tidy_csv(
        df_time,
        layout.data_file("metrics_decode_time_resolved.csv"),
        sort_by=["layer", "substrate", "time_bin_start_ms"],
    )
    summary_csv = save_tidy_csv(
        df_summary,
        layout.data_file("metrics_decode_window_summary.csv"),
        sort_by=["layer", "substrate", "window_name"],
    )
    confidence_csv = save_tidy_csv(
        df_confidence,
        layout.data_file("trial_level_decoder_confidence.csv"),
        sort_by=["trial_id", "layer", "substrate", "eval_bin_start_ms"],
    )
    perm_csv = save_tidy_csv(
        df_perm,
        layout.data_file("metrics_permutation_tests.csv"),
        sort_by=["analysis", "layer", "substrate", "target"],
    )
    fig_summary = plot_decode_window_summary(df_summary)
    fig_summary_paths = save_figure_all_formats(fig_summary, layout.figure_base("decode_window_summary"))
    plt.close(fig_summary)
    fig_conf = plot_confidence_margin(df_confidence)
    fig_conf_paths = save_figure_all_formats(fig_conf, layout.figure_base("confidence_margin"))
    plt.close(fig_conf)

    run_config = {
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "seed": int(args.seed),
        "device": str(device),
        "num_classes": int(args.num_classes),
        "train_trials": int(args.train_trials),
        "test_trials": int(args.test_trials),
        "batch_size": int(args.batch_size),
        "timing_ms": {
            "sample": float(args.sample_ms),
            "delay": float(args.delay_ms),
            "probe": float(args.probe_ms),
            "dt": float(args.dt_ms),
            "bin": float(args.bin_ms),
            "stride": float(args.stride_ms),
        },
        "anchor_step_by_layer": {layer: int(step) for layer, step in anchor_step_by_layer.items()},
        "output_files": {
            "trial_specs_csv": trial_specs_csv,
            "metrics_decode_time_resolved_csv": time_csv,
            "metrics_decode_window_summary_csv": summary_csv,
            "trial_level_decoder_confidence_csv": confidence_csv,
            "metrics_permutation_tests_csv": perm_csv,
        },
    }
    run_config_path = save_run_config(run_config, layout.root)
    summary_path = save_summary_json(
        {
            "experiment": "silent_substrate_triplet_compare",
            "outputs": {
                "trial_specs_csv": str(trial_specs_csv),
                "metrics_decode_time_resolved_csv": str(time_csv),
                "metrics_decode_window_summary_csv": str(summary_csv),
                "trial_level_decoder_confidence_csv": str(confidence_csv),
                "metrics_permutation_tests_csv": str(perm_csv),
                "figure_decode_window_summary_png": fig_summary_paths["png"],
                "figure_confidence_margin_png": fig_conf_paths["png"],
            },
            "summary_by_layer": {
                layer_key: {
                    substrate: float(
                        df_summary[
                            (df_summary["layer"] == layer_key)
                            & (df_summary["substrate"] == substrate)
                            & (df_summary["window_name"] == "whole_delay")
                        ]["decode_acc"].iloc[0]
                    )
                    for substrate in SUBSTRATE_ORDER
                    if not df_summary[
                        (df_summary["layer"] == layer_key)
                        & (df_summary["substrate"] == substrate)
                        & (df_summary["window_name"] == "whole_delay")
                    ].empty
                }
                for layer_key in LAYER_KEYS
            },
        },
        layout.root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=silent_substrate_triplet_compare",
            f"save_dir={layout.root}",
            f"trial_specs_csv={trial_specs_csv}",
            f"metrics_decode_time_resolved_csv={time_csv}",
            f"metrics_decode_window_summary_csv={summary_csv}",
            f"trial_level_decoder_confidence_csv={confidence_csv}",
            f"metrics_permutation_tests_csv={perm_csv}",
            f"figure_decode_window_summary_png={fig_summary_paths['png']}",
            f"figure_confidence_margin_png={fig_conf_paths['png']}",
            f"summary_json={summary_path}",
            f"run_config_json={run_config_path}",
        ],
        layout.log_dir,
    )

    print(f"[Done] Saved: {trial_specs_csv}")
    print(f"[Done] Saved: {time_csv}")
    print(f"[Done] Saved: {summary_csv}")
    print(f"[Done] Saved: {confidence_csv}")
    print(f"[Done] Saved: {perm_csv}")
    print(f"[Done] Saved: {fig_summary_paths['png']}")
    print(f"[Done] Saved: {fig_conf_paths['png']}")
    print(f"[Done] Saved: {summary_path}")
    print(f"[Done] Saved: {run_config_path}")
    print(f"[Done] Saved: {run_log_path}")


if __name__ == "__main__":
    main()
