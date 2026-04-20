from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from tqdm import tqdm

from input_function import build_mnist_skeleton_loader
from src.config.units import ms
from src.experiments.common.dataset import build_class_index
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.deterministic_discovery import (
    ModelRuntime,
    ProbeDataSource,
    ScanConfig,
    TracePolicy,
    run_deterministic_discovery,
)
from src.experiments.common.diagnostic_mask_utils import (
    apply_ablation,
    apply_preserve_only,
    build_mask_from_nonzero,
    build_random_matched_mask,
    connected_component_count,
)
from src.experiments.common.voltage_readout import (
    ProbeScoreBundle,
    compute_voltage_margin,
    compute_voltage_margin_fixed_competitor,
    get_group_voltage_scores,
)
from diagnostic_feature_overlap_morphology import summarize_mask_description
from diagnostic_feature_overlap_stability import stratified_probe_split
from diagnostic_feature_overlap_stats import summarize_discovery_metrics, summarize_necessity_metrics, summarize_sufficiency_metrics
from src.plotting.common.io import save_figure_all_formats, save_run_config, save_tidy_csv, validate_required_columns


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    probe_ms: float

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


@dataclass
class DiagnosticMapAccumulator:
    importance_sum: np.ndarray | None = None
    selected_mask_count: np.ndarray | None = None
    probe_count: int = 0

    def update(self, discriminative_map: np.ndarray, selected_mask: np.ndarray) -> None:
        disc = np.asarray(discriminative_map, dtype=np.float64)
        mask = np.asarray(selected_mask, dtype=np.float64)
        if self.importance_sum is None:
            self.importance_sum = np.zeros_like(disc, dtype=np.float64)
        if self.selected_mask_count is None:
            self.selected_mask_count = np.zeros_like(mask, dtype=np.float64)
        self.importance_sum += disc
        self.selected_mask_count += mask
        self.probe_count += 1

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.probe_count <= 0 or self.importance_sum is None or self.selected_mask_count is None:
            return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
        mean_map = self.importance_sum / float(self.probe_count)
        coverage_map = self.selected_mask_count / float(self.probe_count)
        return np.asarray(mean_map, dtype=np.float64), np.asarray(coverage_map, dtype=np.float64)


def mix_seed(base_seed: int, *parts: int) -> int:
    value = int(base_seed) & 0xFFFFFFFF
    for idx, part in enumerate(parts, start=1):
        value = (value * 1664525 + 1013904223 + int(part) * (374761393 + idx * 97)) & 0xFFFFFFFF
    return int(value)


def build_dataset_arrays(dataset) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    images = torch.stack([dataset[idx][0] for idx in range(len(dataset))], dim=0).cpu().to(torch.float32)
    labels = np.asarray([int(dataset[idx][1]) for idx in range(len(dataset))], dtype=np.int64)
    flat = images.view(len(dataset), -1).numpy().astype(np.float32, copy=False)
    return images, labels, flat


def build_balanced_probe_candidate_pool(dataset_labels: np.ndarray, trial_count: int, seed: int, probe_pool_limit: int, probe_pool_per_class: int) -> pd.DataFrame:
    rng = np.random.default_rng(mix_seed(seed, 91))
    rows: List[Dict[str, int]] = []
    candidate_rank = 0
    for probe_label in sorted({int(label) for label in dataset_labels.tolist()}):
        ids = np.flatnonzero(dataset_labels == int(probe_label)).astype(np.int64)
        ids = rng.permutation(ids)[: int(probe_pool_per_class)]
        for probe_id in ids.tolist():
            rows.append({"candidate_rank": int(candidate_rank), "probe_id": int(probe_id), "probe_label": int(probe_label)})
            candidate_rank += 1
            if len(rows) >= int(probe_pool_limit):
                break
        if len(rows) >= int(probe_pool_limit):
            break
    return pd.DataFrame(rows).sort_values(["candidate_rank"], kind="stable").reset_index(drop=True)


def select_balanced_probe_candidates(
    df_baseline: pd.DataFrame,
    trial_count: int,
    seed: int,
    probe_partition: str = "correct",
) -> pd.DataFrame:
    partition = str(probe_partition).strip().lower()
    if partition == "correct":
        baseline_subset = df_baseline[df_baseline["is_correct"] == 1].copy()
        empty_reason = "No baseline-correct probes were found for deterministic discovery."
    elif partition == "wrong":
        baseline_subset = df_baseline[df_baseline["is_correct"] == 0].copy()
        empty_reason = "No baseline-wrong probes were found for deterministic discovery."
    elif partition == "all":
        baseline_subset = df_baseline.copy()
        empty_reason = "No baseline probes were found for deterministic discovery."
    else:
        raise ValueError(f"Unsupported probe_partition: {probe_partition}")
    if baseline_subset.empty:
        raise ValueError(empty_reason)
    per_label: Dict[int, List[Dict[str, int]]] = {}
    rng = np.random.default_rng(int(seed))
    baseline_subset = baseline_subset.copy()
    baseline_subset["baseline_is_correct"] = baseline_subset["is_correct"].astype(np.int64, copy=False)
    for probe_label, group in baseline_subset.groupby("probe_label", sort=True):
        rows = group[
            ["probe_id", "probe_label", "predicted_label", "first_fire_t_probe", "is_correct", "baseline_is_correct"]
        ].to_dict("records")
        order = rng.permutation(len(rows))
        per_label[int(probe_label)] = [rows[idx] for idx in order.tolist()]
    cursors = {label: 0 for label in per_label}
    selected: List[Dict[str, int]] = []
    labels = sorted(per_label)
    while len(selected) < int(trial_count):
        made_progress = False
        for label in labels:
            if cursors[label] >= len(per_label[label]):
                continue
            selected.append(dict(per_label[label][cursors[label]]))
            cursors[label] += 1
            made_progress = True
            if len(selected) >= int(trial_count):
                break
        if not made_progress:
            break
    return pd.DataFrame(selected).reset_index(drop=True)


def _baseline_cache_path(save_dir: Path) -> Path:
    return save_dir / "baseline_probe_reference.csv"


def _parse_float_csv(text: str) -> List[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _score_single_probe(model_runtime: ModelRuntime, probe_image: torch.Tensor) -> ProbeScoreBundle:
    return get_group_voltage_scores(
        net=model_runtime.net,
        encoder=model_runtime.encoder,
        probe_images=probe_image.unsqueeze(0),
        spec=model_runtime.spec,
        device=model_runtime.device,
        readout_mode=model_runtime.readout_mode,
        readout_step=model_runtime.readout_step,
        pooling=model_runtime.voltage_pooling,
        m=model_runtime.top_m,
        stsp_mode="static_frozen",
    )[0]


def _run_baseline_probe_scan_voltage(model_runtime: ModelRuntime, raw_images: torch.Tensor, candidate_pool: pd.DataFrame, *, batch_size: int) -> pd.DataFrame:
    validate_required_columns(candidate_pool, ["candidate_rank", "probe_id", "probe_label"])
    records: List[Dict[str, object]] = []
    for start in tqdm(range(0, len(candidate_pool), batch_size), desc="BaselineProbeScanVoltage"):
        batch = candidate_pool.iloc[start:start + batch_size].copy()
        bundles = get_group_voltage_scores(
            net=model_runtime.net,
            encoder=model_runtime.encoder,
            probe_images=raw_images[batch["probe_id"].tolist()],
            spec=model_runtime.spec,
            device=model_runtime.device,
            readout_mode=model_runtime.readout_mode,
            readout_step=model_runtime.readout_step,
            pooling=model_runtime.voltage_pooling,
            m=model_runtime.top_m,
            stsp_mode="static_frozen",
        )
        for bundle, row in zip(bundles, batch.itertuples(index=False)):
            margin = compute_voltage_margin(bundle, true_label=int(row.probe_label))
            records.append({
                "candidate_rank": int(row.candidate_rank),
                "probe_id": int(row.probe_id),
                "probe_label": int(row.probe_label),
                "predicted_label": int(bundle.predicted_label),
                "first_fire_t_probe": int(bundle.first_fire_t_probe),
                "is_correct": int(bundle.predicted_label == int(row.probe_label)),
                "true_score": float(margin.true_score),
                "best_wrong_score": float(margin.best_wrong_score),
                "best_wrong_label": int(margin.best_wrong_label),
                "margin": float(margin.margin),
            })
    return pd.DataFrame(records).sort_values(["candidate_rank"], kind="stable").reset_index(drop=True)


def _low_importance_mask(discriminative_map: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    area = int(np.asarray(ref_mask, dtype=bool).sum())
    out = np.zeros_like(ref_mask, dtype=bool)
    if area <= 0:
        return out
    flat = np.asarray(discriminative_map, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="stable")
    out.reshape(-1)[order[:area]] = True
    return out


def _build_selection_mask(discriminative_map: np.ndarray, mask_policy: str) -> np.ndarray:
    if str(mask_policy) != "nonzero":
        raise ValueError(f"Unsupported mask policy: {mask_policy}")
    return np.asarray(build_mask_from_nonzero(discriminative_map), dtype=bool)


def _baseline_wrong0_from_row(row, probe_label: int) -> tuple[int, float, float]:
    baseline_is_correct = int(getattr(row, "baseline_is_correct", getattr(row, "is_correct", -1)))
    predicted_label = int(getattr(row, "predicted_label", -1))
    if baseline_is_correct != 0 or predicted_label < 0 or predicted_label == int(probe_label):
        return -1, float("nan"), float("nan")
    baseline_true_score = float(getattr(row, "true_score", float("nan")))
    baseline_wrong0_score = float(getattr(row, "best_wrong_score", float("nan")))
    baseline_margin_fixed_wrong0 = float(getattr(row, "margin", float("nan")))
    return predicted_label, baseline_wrong0_score, baseline_margin_fixed_wrong0


def evaluate_sufficiency_probe_only(*, probe_id: int, probe_label: int, probe_image: torch.Tensor, selected_mask: np.ndarray, discriminative_map: np.ndarray, model_runtime: ModelRuntime, num_random_controls: int, selection_policy: str) -> pd.DataFrame:
    component_hint = connected_component_count(selected_mask)
    low_mask = _low_importance_mask(discriminative_map, selected_mask)
    full_bundle = _score_single_probe(model_runtime, probe_image)
    full_margin = compute_voltage_margin(full_bundle, true_label=probe_label)
    mask_area = int(np.asarray(selected_mask, dtype=bool).sum())
    items: List[tuple[str, np.ndarray]] = [("selected_mask_preserve_only", np.asarray(selected_mask, dtype=bool))]
    rng = np.random.default_rng(mix_seed(int(probe_id), 701))
    for idx in range(int(num_random_controls)):
        items.append((f"random_area_preserve_only_{idx}", build_random_matched_mask(selected_mask, rng=rng, component_hint=None, exclude_mask=selected_mask)))
        items.append((f"random_components_preserve_only_{idx}", build_random_matched_mask(selected_mask, rng=rng, component_hint=component_hint, exclude_mask=selected_mask)))
    items.append(("low_importance_preserve_only", low_mask))
    rows: List[Dict[str, object]] = []
    random_scores: List[float] = []
    for condition, mask in items:
        bundle = _score_single_probe(model_runtime, apply_preserve_only(probe_image, mask, fill_value=0.0))
        margin = compute_voltage_margin(bundle, true_label=probe_label)
        row = {
            "probe_id": int(probe_id),
            "probe_label": int(probe_label),
            "condition": str(condition),
            "selection_policy": str(selection_policy),
            "mask_area": int(mask_area),
            "predicted_label": int(bundle.predicted_label),
            "is_correct": int(bundle.predicted_label == int(probe_label)),
            "margin_retention_ratio": float(margin.margin / max(abs(full_margin.margin), 1e-8)),
            "true_score_retention_ratio": float(margin.true_score / max(abs(full_margin.true_score), 1e-8)),
            "sufficiency_gain": float("nan"),
        }
        rows.append(row)
        if str(condition).startswith("random_area_preserve_only"):
            random_scores.append(float(row["is_correct"]))
    random_baseline = float(np.mean(random_scores)) if random_scores else 0.0
    for row in rows:
        row["sufficiency_gain"] = float(row["is_correct"]) - random_baseline
    return pd.DataFrame(rows)


def evaluate_necessity_probe_only(*, probe_id: int, probe_label: int, probe_image: torch.Tensor, selected_mask: np.ndarray, discriminative_map: np.ndarray, model_runtime: ModelRuntime, num_random_controls: int, selection_policy: str) -> pd.DataFrame:
    component_hint = connected_component_count(selected_mask)
    low_mask = _low_importance_mask(discriminative_map, selected_mask)
    full_bundle = _score_single_probe(model_runtime, probe_image)
    full_margin = compute_voltage_margin(full_bundle, true_label=probe_label)
    mask_area = int(np.asarray(selected_mask, dtype=bool).sum())
    items: List[tuple[str, np.ndarray]] = [("selected_mask_ablation", np.asarray(selected_mask, dtype=bool))]
    rng = np.random.default_rng(mix_seed(int(probe_id), 1701))
    for idx in range(int(num_random_controls)):
        items.append((f"random_area_ablation_{idx}", build_random_matched_mask(selected_mask, rng=rng, component_hint=None, exclude_mask=selected_mask)))
        items.append((f"random_components_ablation_{idx}", build_random_matched_mask(selected_mask, rng=rng, component_hint=component_hint, exclude_mask=selected_mask)))
    items.append(("low_importance_ablation", low_mask))
    rows: List[Dict[str, object]] = []
    random_deltas: List[float] = []
    for condition, mask in items:
        bundle = _score_single_probe(model_runtime, apply_ablation(probe_image, mask, fill_value=0.0))
        margin = compute_voltage_margin(bundle, true_label=probe_label)
        delta_margin = float(full_margin.margin - margin.margin)
        row = {
            "probe_id": int(probe_id),
            "probe_label": int(probe_label),
            "condition": str(condition),
            "selection_policy": str(selection_policy),
            "mask_area": int(mask_area),
            "predicted_label": int(bundle.predicted_label),
            "is_correct": int(bundle.predicted_label == int(probe_label)),
            "delta_margin": delta_margin,
            "accuracy_drop": float(int(full_bundle.predicted_label == int(probe_label)) - int(bundle.predicted_label == int(probe_label))),
            "necessity_gain": float("nan"),
        }
        rows.append(row)
        if str(condition).startswith("random_area_ablation"):
            random_deltas.append(delta_margin)
    random_baseline = float(np.mean(random_deltas)) if random_deltas else 0.0
    for row in rows:
        row["necessity_gain"] = float(row["delta_margin"] - random_baseline)
    return pd.DataFrame(rows)


def _objective_from_metrics(suff_df: pd.DataFrame, nec_df: pd.DataFrame) -> float:
    suff_gain = float(suff_df[suff_df["condition"] == "selected_mask_preserve_only"]["sufficiency_gain"].mean()) if not suff_df.empty else 0.0
    nec_gain = float(nec_df[nec_df["condition"] == "selected_mask_ablation"]["necessity_gain"].mean()) if not nec_df.empty else 0.0
    return suff_gain + nec_gain


def _validate_removed_args(args: argparse.Namespace) -> None:
    if str(args.model_paths).strip():
        raise ValueError("--model-paths is removed; use --model-path.")
    if str(args.model_path_glob).strip():
        raise ValueError("--model-path-glob is removed; use --model-path.")
    if str(args.label_randomized_model_paths).strip():
        raise ValueError("--label-randomized-model-paths is removed.")
    if str(args.importance_seeds).strip():
        raise ValueError("--importance-seeds is removed.")
    if float(args.noise_std) != 0.0 or int(args.translation_jitter_px) != 0:
        raise ValueError("Noise and jitter are removed from the deterministic runtime.")
    if bool(args.run_sanity_checks) or bool(args.legacy_earliest_fire_comparison):
        raise ValueError("Model comparison paths were removed from the deterministic runtime.")
    if str(args.patch_size_grid).strip() or str(args.stride_grid).strip():
        raise ValueError("--patch-size-grid and --stride-grid were replaced by --patch-size and --scan-stride.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic dense-scan discovery pipeline.")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--save-dir", type=str, default="results/diagnostic_feature_overlap_experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--baseline-batch-size", type=int, default=128)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)
    parser.add_argument("--trial-count", type=int, default=2000)
    parser.add_argument("--probe-pool-limit", type=int, default=5000)
    parser.add_argument("--probe-pool-per-class", type=int, default=300)    
    parser.add_argument("--num-boot", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--readout-mode", type=str, default="decision_offset", choices=["decision_offset", "probe_last_step", "explicit_step"])
    parser.add_argument("--readout-step", type=int, default=-1)
    parser.add_argument("--voltage-pooling", type=str, default="top_m_mean", choices=["max", "top_m_mean", "full_mean"])
    parser.add_argument("--top-m", type=int, default=1)
    parser.add_argument("--lambda-global-grid", type=str, default="1")
    parser.add_argument("--patch-size", type=int, default=3)
    parser.add_argument("--scan-stride", type=int, default=1)
    parser.add_argument("--mask-policy", type=str, default="nonzero", choices=["nonzero"])
    parser.add_argument("--probe_partition", type=str, default="all", choices=["correct", "wrong", "all"])
    parser.add_argument("--discovery-frac", type=float, default=0)
    parser.add_argument("--num-random-controls", type=int, default=4)
    parser.add_argument("--trace-mode", type=str, default="none", choices=["none", "boundary", "anomalous", "filtered", "debug"])
    parser.add_argument("--trace-margin-threshold", type=float, default=0.0)
    parser.add_argument("--trace-window-ids", type=str, default="")
    parser.add_argument("--trace-debug-sample-rate", type=float, default=0.0)
    parser.add_argument("--max-traces-per-probe", type=int, default=0)
    parser.add_argument("--debug-full-stride-override", action="store_true")
    parser.add_argument("--model-paths", type=str, default="")
    parser.add_argument("--model-path-glob", type=str, default="")
    parser.add_argument("--label-randomized-model-paths", type=str, default="")
    parser.add_argument("--importance-seeds", type=str, default="")
    parser.add_argument("--translation-jitter-px", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--run-sanity-checks", action="store_true")
    parser.add_argument("--legacy-earliest-fire-comparison", action="store_true")
    parser.add_argument("--patch-size-grid", type=str, default="")
    parser.add_argument("--stride-grid", type=str, default="")
    return parser


def _sanitize_window_id(window_id: str) -> str:
    return str(window_id).replace(":", "_")


def _save_trace_payloads(probe_dir: Path, saved_traces: Dict[str, dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for window_id, payload in saved_traces.items():
        trace_path = probe_dir / f"{_sanitize_window_id(window_id)}_trace.pt"
        torch.save(payload["state_traces"], trace_path)
        rows.append({"window_id": str(window_id), "trace_path": str(trace_path), "readout_step": int(payload["readout_step"])})
    return pd.DataFrame(rows)


def append_tidy_csv(df: pd.DataFrame, save_path: Path, sort_by: str | List[str] | None = None) -> str:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    out_df = df.copy()
    if sort_by is not None:
        sort_columns = [sort_by] if isinstance(sort_by, str) else list(sort_by)
        if sort_columns:
            validate_required_columns(out_df, sort_columns)
            out_df = out_df.sort_values(by=sort_columns, kind="stable").reset_index(drop=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    exists = save_path.exists()
    if out_df.empty:
        if not exists:
            out_df.to_csv(save_path, index=False, encoding="utf-8")
        return str(save_path)
    out_df.to_csv(save_path, mode="a", header=not exists, index=False, encoding="utf-8")
    return str(save_path)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _to_display_image(probe_image: torch.Tensor) -> np.ndarray:
    image = probe_image.detach().cpu().numpy().astype(np.float64, copy=False)
    if image.ndim != 3:
        raise ValueError(f"Expected [C, H, W] probe image, got {tuple(image.shape)}")
    return np.asarray(image[0], dtype=np.float64)


def plot_probe_diagnostic_selection(*, probe_image: torch.Tensor, discriminative_map: np.ndarray, selected_mask: np.ndarray, probe_id: int, probe_label: int, lambda_global: float, mask_policy: str, mask_area: int) -> Figure:
    image = _to_display_image(probe_image)
    importance = np.asarray(discriminative_map, dtype=np.float64)
    mask = np.asarray(selected_mask, dtype=bool)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    axes[0].imshow(image, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"Probe {probe_id} label={probe_label}")
    heat = axes[1].imshow(importance, cmap="magma", interpolation="nearest")
    axes[1].set_title("Signed direction score")
    axes[2].imshow(image, cmap="gray", interpolation="nearest")
    overlay = np.zeros((*mask.shape, 4), dtype=np.float64)
    overlay[..., 0] = 1.0
    overlay[..., 1] = 0.45
    overlay[..., 2] = 0.05
    overlay[..., 3] = mask.astype(np.float64) * 0.55
    axes[2].imshow(overlay, interpolation="nearest")
    axes[2].set_title(f"Legacy {mask_policy} mask area={mask_area}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(f"Diagnostic Raw Score probe={probe_id} label={probe_label}", fontsize=11)
    fig.tight_layout()
    return fig


def plot_diagnostic_maps_summary(mean_map: np.ndarray, coverage_map: np.ndarray) -> Figure:
    mean_arr = np.asarray(mean_map, dtype=np.float64)
    coverage_arr = np.asarray(coverage_map, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    mean_im = axes[0].imshow(mean_arr, cmap="magma", interpolation="nearest")
    axes[0].set_title("Mean signed direction")
    coverage_im = axes[1].imshow(coverage_arr, cmap="inferno", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[1].set_title("Legacy selected-mask coverage")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(mean_im, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(coverage_im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("Diagnostic Raw Score Summary", fontsize=11)
    fig.tight_layout()
    return fig


def _build_probe_summary_row(
    *,
    probe_id: int,
    probe_label: int,
    lambda_global: float,
    selection_policy: str,
    baseline_is_correct: int,
    window_df: pd.DataFrame,
    nonzero_window_df: pd.DataFrame,
    discriminative_map: np.ndarray,
    selected_mask: np.ndarray,
    saved_traces: Dict[str, dict[str, object]],
    description: str,
) -> Dict[str, object]:
    projected = np.asarray(discriminative_map, dtype=np.float64)
    positive = projected[np.isfinite(projected) & (projected > 0.0)]
    total_window_count = int(len(window_df))
    nonzero_window_count = int(len(nonzero_window_df))
    return {
        "probe_id": int(probe_id),
        "probe_label": int(probe_label),
        "baseline_is_correct": int(baseline_is_correct),
        "lambda_global": float(lambda_global),
        "selection_policy": str(selection_policy),
        "mask_area": int(np.asarray(selected_mask, dtype=bool).sum()),
        "total_window_count": int(total_window_count),
        "nonzero_window_count": int(nonzero_window_count),
        "nonzero_window_rate": float(nonzero_window_count / max(total_window_count, 1)),
        "projected_nonzero_area": int(np.asarray(selected_mask, dtype=bool).sum()),
        "positive_importance_mean": float(np.mean(positive)) if positive.size else 0.0,
        "positive_importance_max": float(np.max(positive)) if positive.size else 0.0,
        "trace_saved_count": int(len(saved_traces)),
        "description": str(description),
    }


def main() -> None:
    args = build_argparser().parse_args()
    _validate_removed_args(args)
    if args.batch_size <= 0 or args.baseline_batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.sample_ms <= 0 or args.probe_ms <= 0 or args.patch_size <= 0:
        raise ValueError("sample/probe duration and patch size must be positive")
    if args.trial_count <= 0 or args.probe_pool_limit <= 0 or args.probe_pool_per_class <= 0:
        raise ValueError("trial and probe-pool settings must be positive")
    if args.num_boot <= 0 or args.top_m <= 0 or args.num_random_controls <= 0:
        raise ValueError("num-boot, top-m, and num-random-controls must be positive")
    if args.scan_stride <= 0:
        raise ValueError("--scan-stride must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    spec = ExperimentSpec(dt=1.0 * ms, sample_ms=float(args.sample_ms), probe_ms=float(args.probe_ms))
    readout_step = None if int(args.readout_step) < 0 else int(args.readout_step)
    lambda_grid = _parse_float_csv(args.lambda_global_grid)
    if not lambda_grid:
        lambda_grid = [float(args.lambda_global_grid) if str(args.lambda_global_grid).strip() else 1.0]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(spec.sample_ms, spec.probe_ms),
    )
    model_runtime = ModelRuntime(
        net=net,
        encoder=encoder,
        model_path=str(args.model_path),
        spec=spec,
        device=device,
        readout_mode=str(args.readout_mode),
        readout_step=readout_step,
        voltage_pooling=str(args.voltage_pooling),
        top_m=int(args.top_m),
    )

    _, _, test_loader = build_mnist_skeleton_loader(root=args.dataset_root, batch_size=1, input_size=28, num_workers=0)
    dataset = test_loader.dataset
    _ = build_class_index(dataset, num_classes=10)
    raw_images, dataset_labels, _ = build_dataset_arrays(dataset)

    candidate_pool = build_balanced_probe_candidate_pool(dataset_labels=dataset_labels, trial_count=int(args.trial_count), seed=int(args.seed), probe_pool_limit=int(args.probe_pool_limit), probe_pool_per_class=int(args.probe_pool_per_class))
    df_baseline = _run_baseline_probe_scan_voltage(model_runtime=model_runtime, raw_images=raw_images, candidate_pool=candidate_pool, batch_size=int(args.baseline_batch_size))
    baseline_csv = save_tidy_csv(df_baseline, _baseline_cache_path(save_dir), sort_by=["candidate_rank"])
    selected_probes = select_balanced_probe_candidates(
        df_baseline=df_baseline,
        trial_count=int(args.trial_count),
        seed=mix_seed(int(args.seed), 101),
        probe_partition=str(args.probe_partition),
    )
    discovery_df = selected_probes.copy()
    heldout_df = selected_probes.copy()
    discovery_split_csv = save_tidy_csv(discovery_df.assign(split="discovery"), save_dir / "discovery_split.csv", sort_by=["probe_label", "probe_id"])
    heldout_split_csv = save_tidy_csv(heldout_df.assign(split="heldout"), save_dir / "heldout_split.csv", sort_by=["probe_label", "probe_id"])

    best_lambda = float(lambda_grid[0])
    discovery_objective_rows: List[Dict[str, object]] = [
        {
            "lambda_global": float(best_lambda),
            "objective": float("nan"),
            "n_discovery_probes": int(len(discovery_df)),
            "selection_semantics": "legacy_audit_only",
            "note": "lambda_global is retained for compatibility only; raw_importance is the primary direction score.",
        }
    ]
    discovery_objective_csv = save_tidy_csv(pd.DataFrame(discovery_objective_rows), save_dir / "discovery_objective_summary.csv", sort_by=["lambda_global"])

    trace_policy = TracePolicy(
        mode=str(args.trace_mode),
        margin_threshold=float(args.trace_margin_threshold),
        debug_sample_rate=float(args.trace_debug_sample_rate),
        include_window_ids=frozenset(item.strip() for item in str(args.trace_window_ids).split(",") if item.strip()),
        max_traces_per_probe=int(args.max_traces_per_probe),
    )

    probes_dir = save_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    accumulator = DiagnosticMapAccumulator()

    window_summary_path = save_dir / "window_summary.csv"
    window_summary_nonzero_path = save_dir / "window_summary_nonzero.csv"
    window_rank_path = save_dir / "window_importance_rank.csv"
    pixel_rank_path = save_dir / "pixel_importance_rank.csv"
    probe_rule_path = save_dir / "probe_rule_table.csv"
    suff_path = save_dir / "sufficiency_probe_results.csv"
    nec_path = save_dir / "necessity_probe_results.csv"
    wrong_fixed_audit_path = save_dir / "wrong_fixed_competitor_audit.csv"
    wrong_fixed_audit_rows: List[Dict[str, object]] = []

    for row in tqdm(list(heldout_df.itertuples(index=False)), desc="DeterministicDiscovery"):
        probe_id = int(row.probe_id)
        probe_label = int(row.probe_label)
        wrong0_label, wrong0_score, margin_fixed_wrong0 = _baseline_wrong0_from_row(row, probe_label=probe_label)
        foreground_mask = np.asarray(raw_images[probe_id][0].detach().cpu().numpy() > 0.0, dtype=bool)
        scan_config = ScanConfig(
            model_path=str(args.model_path),
            patch_size=int(args.patch_size),
            scan_stride=int(args.scan_stride),
            batch_size=int(args.batch_size),
            micro_batch_size=int(args.micro_batch_size),
            lambda_global=float(best_lambda),
            debug_full_stride_override=bool(args.debug_full_stride_override),
        )
        result = run_deterministic_discovery(
            scan_config=scan_config,
            data_source=ProbeDataSource(probe_id=probe_id, probe_label=probe_label, probe_image=raw_images[probe_id]),
            model_runtime=model_runtime,
            trace_policy=trace_policy,
            fixed_competitor_label=None if wrong0_label < 0 else int(wrong0_label),
        )
        selected_mask = _build_selection_mask(result.projected_discriminative_map, str(args.mask_policy))
        suff_df = evaluate_sufficiency_probe_only(probe_id=probe_id, probe_label=probe_label, probe_image=raw_images[probe_id], selected_mask=selected_mask, discriminative_map=result.projected_discriminative_map, model_runtime=model_runtime, num_random_controls=int(args.num_random_controls), selection_policy=str(args.mask_policy))
        nec_df = evaluate_necessity_probe_only(probe_id=probe_id, probe_label=probe_label, probe_image=raw_images[probe_id], selected_mask=selected_mask, discriminative_map=result.projected_discriminative_map, model_runtime=model_runtime, num_random_controls=int(args.num_random_controls), selection_policy=str(args.mask_policy))
        base_scores = np.asarray(result.base_scores, dtype=np.float64)
        competitor_scores = base_scores.copy()
        competitor_scores[probe_label] = -np.inf
        competitor_label = int(np.argmax(competitor_scores))
        morphology = summarize_mask_description(selected_mask, true_label=probe_label, competitor_label=competitor_label, mean_importance=result.projected_discriminative_map)

        probe_dir = probes_dir / f"probe_{probe_id:05d}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        np.save(probe_dir / "importance_discriminative.npy", result.projected_discriminative_map)
        np.save(probe_dir / "importance_signed.npy", result.importance_map_signed)
        np.save(probe_dir / "foreground_mask.npy", np.asarray(foreground_mask, dtype=np.uint8))
        np.save(probe_dir / "critical_mask_nonzero.npy", np.asarray(selected_mask, dtype=np.uint8))
        np.save(probe_dir / "positive_importance_mask.npy", np.asarray(result.positive_importance_mask, dtype=np.uint8))
        np.save(probe_dir / "negative_importance_mask.npy", np.asarray(result.negative_importance_mask, dtype=np.uint8))
        (probe_dir / "description.txt").write_text(str(morphology["summary_text"]), encoding="utf-8")
        pd.DataFrame([{"class_label": idx, "score": float(score)} for idx, score in enumerate(base_scores.tolist())]).to_csv(probe_dir / "group_scores_full.csv", index=False)
        (probe_dir / "probe_metadata.json").write_text(
            json.dumps(
                {
                        "probe_id": int(probe_id),
                        "probe_label": int(probe_label),
                        "baseline_is_correct": int(getattr(row, "baseline_is_correct", getattr(row, "is_correct", -1))),
                        "baseline_predicted_label": int(getattr(row, "predicted_label", -1)),
                        "baseline_first_fire_t_probe": int(getattr(row, "first_fire_t_probe", -1)),
                        "baseline_margin": float(getattr(row, "margin", float("nan"))),
                        "baseline_wrong0_label": int(wrong0_label),
                        "baseline_wrong0_score": float(wrong0_score),
                        "baseline_margin_fixed_wrong0": float(margin_fixed_wrong0),
                        "wrong0_label": int(wrong0_label),
                        "wrong0_score": float(wrong0_score),
                        "margin_fixed_wrong0": float(margin_fixed_wrong0),
                        "direction_score_semantics": "raw_margin_drop",
                        "dn_selection_stage": "causal_only",
                        "active_value_col": str(result.active_value_col),
                        "uses_fixed_competitor": bool(result.uses_fixed_competitor),
                        "foreground_area": int(foreground_mask.sum()),
                        "wrong_mask_semantics": "" if wrong0_label < 0 else "support_harm_fixed_competitor",
                        "fixed_competitor_discovery": bool(wrong0_label >= 0),
                    },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if wrong0_label >= 0:
            wrong_fixed_audit_rows.append(
                {
                    "probe_id": int(probe_id),
                    "probe_label": int(probe_label),
                    "wrong0_label": int(wrong0_label),
                    "baseline_prediction": int(getattr(row, "predicted_label", -1)),
                    "baseline_true_score": float(getattr(row, "true_score", float("nan"))),
                    "baseline_wrong0_score": float(wrong0_score),
                    "baseline_margin_fixed_wrong0": float(margin_fixed_wrong0),
                }
            )

        window_probe = result.window_results.assign(probe_label=probe_label)
        nonzero_probe = result.nonzero_window_results.assign(probe_label=probe_label)
        ranking_probe = result.window_ranking.assign(probe_label=probe_label)
        pixel_probe = result.pixel_ranking.assign(probe_id=probe_id, probe_label=probe_label)
        save_tidy_csv(window_probe, probe_dir / "window_summary.csv", sort_by=["scan_stage", "row_start", "col_start"])
        save_tidy_csv(nonzero_probe, probe_dir / "window_summary_nonzero.csv", sort_by=["scan_stage", "row_start", "col_start"])
        save_tidy_csv(ranking_probe, probe_dir / "window_importance_rank.csv", sort_by=["rank"])
        save_tidy_csv(pixel_probe, probe_dir / "pixel_importance_rank.csv", sort_by=["rank"])
        if result.saved_traces:
            save_tidy_csv(_save_trace_payloads(probe_dir, result.saved_traces), probe_dir / "trace_index.csv", sort_by=["window_id"])

        append_tidy_csv(window_probe, window_summary_path)
        append_tidy_csv(nonzero_probe, window_summary_nonzero_path)
        append_tidy_csv(ranking_probe, window_rank_path)
        append_tidy_csv(pixel_probe, pixel_rank_path)
        append_tidy_csv(suff_df, suff_path)
        append_tidy_csv(nec_df, nec_path)
        append_tidy_csv(
            pd.DataFrame([
                _build_probe_summary_row(
                    probe_id=probe_id,
                    probe_label=probe_label,
                    baseline_is_correct=int(getattr(row, "baseline_is_correct", getattr(row, "is_correct", -1))),
                    lambda_global=float(best_lambda),
                    selection_policy=str(args.mask_policy),
                    window_df=result.window_results,
                    nonzero_window_df=result.nonzero_window_results,
                    discriminative_map=result.projected_discriminative_map,
                    selected_mask=selected_mask,
                    saved_traces=result.saved_traces,
                    description=str(morphology["summary_text"]),
                )
            ]),
            probe_rule_path,
        )

        accumulator.update(result.projected_discriminative_map, selected_mask)
        probe_fig = plot_probe_diagnostic_selection(
            probe_image=raw_images[probe_id],
            discriminative_map=result.projected_discriminative_map,
            selected_mask=selected_mask,
            probe_id=probe_id,
            probe_label=probe_label,
            lambda_global=float(best_lambda),
            mask_policy=str(args.mask_policy),
            mask_area=int(np.asarray(selected_mask, dtype=bool).sum()),
        )
        save_figure_all_formats(probe_fig, probe_dir / "plot_probe_diagnostic_selection")
        plt.close(probe_fig)

    window_df = _read_csv_if_exists(window_summary_path)
    nonzero_window_df = _read_csv_if_exists(window_summary_nonzero_path)
    window_rank_df = _read_csv_if_exists(window_rank_path)
    pixel_rank_df = _read_csv_if_exists(pixel_rank_path)
    probe_rule_df = _read_csv_if_exists(probe_rule_path)
    suff_df = _read_csv_if_exists(suff_path)
    nec_df = _read_csv_if_exists(nec_path)
    wrong_fixed_audit_df = pd.DataFrame(wrong_fixed_audit_rows)

    window_summary_csv = save_tidy_csv(window_df, window_summary_path, sort_by=["probe_id", "scan_stage", "row_start", "col_start"])
    nonzero_window_csv = save_tidy_csv(nonzero_window_df, window_summary_nonzero_path, sort_by=["probe_id", "scan_stage", "row_start", "col_start"])
    window_rank_csv = save_tidy_csv(window_rank_df, window_rank_path, sort_by=["probe_id", "rank"])
    pixel_rank_csv = save_tidy_csv(pixel_rank_df, pixel_rank_path, sort_by=["probe_id", "rank"])
    probe_rule_csv = save_tidy_csv(probe_rule_df, probe_rule_path, sort_by=["probe_label", "probe_id"])
    suff_csv = save_tidy_csv(suff_df, suff_path, sort_by=["probe_id", "condition"])
    nec_csv = save_tidy_csv(nec_df, nec_path, sort_by=["probe_id", "condition"])
    wrong_fixed_audit_csv = save_tidy_csv(wrong_fixed_audit_df, wrong_fixed_audit_path, sort_by=["probe_id"])

    suff_summary_csv = save_tidy_csv(summarize_sufficiency_metrics(suff_df, n_boot=int(args.num_boot), seed=mix_seed(int(args.seed), 911)), save_dir / "sufficiency_summary.csv", sort_by=["condition"])
    nec_summary_csv = save_tidy_csv(summarize_necessity_metrics(nec_df, n_boot=int(args.num_boot), seed=mix_seed(int(args.seed), 913)), save_dir / "necessity_summary.csv", sort_by=["condition"])
    discovery_summary_csv = save_tidy_csv(summarize_discovery_metrics(window_df, probe_rule_df, n_boot=int(args.num_boot), seed=mix_seed(int(args.seed), 909)), save_dir / "discovery_summary.csv", sort_by=["metric"])

    diagnostic_map_mean, diagnostic_mask_coverage = accumulator.finalize()
    diagnostic_map_mean_path = save_dir / "diagnostic_map_mean.npy"
    diagnostic_mask_coverage_path = save_dir / "diagnostic_mask_coverage.npy"
    np.save(diagnostic_map_mean_path, diagnostic_map_mean)
    np.save(diagnostic_mask_coverage_path, diagnostic_mask_coverage)
    summary_fig = plot_diagnostic_maps_summary(diagnostic_map_mean, diagnostic_mask_coverage)
    summary_plot_paths = save_figure_all_formats(summary_fig, save_dir / "plot_diagnostic_maps_summary")
    plt.close(summary_fig)

    run_config = {
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "save_dir": str(save_dir),
        "seed": int(args.seed),
        "device": str(device),
        "trial_count": int(args.trial_count),
        "probe_partition": str(args.probe_partition),
        "readout_mode": str(args.readout_mode),
        "readout_step": None if readout_step is None else int(readout_step),
        "voltage_pooling": str(args.voltage_pooling),
        "top_m": int(args.top_m),
        "lambda_global_selected": float(best_lambda),
        "lambda_global_grid": lambda_grid,
        "lambda_global_note": "legacy / compatibility only; not used as the primary direction score",
        "patch_size": int(args.patch_size),
        "scan_stride": 1 if bool(args.debug_full_stride_override) else int(args.scan_stride),
        "mask_policy": str(args.mask_policy),
        "mask_policy_note": "legacy / compatibility only; does not determine the primary causal D/N regions",
        "discovery_frac": float(args.discovery_frac),
        "discovery_frac_note": "legacy / compatibility only; top-k D/N selection happens only in causal",
        "direction_score_semantics": "raw_margin_drop",
        "dn_selection_stage": "causal_only",
        "trace_policy": {"mode": str(trace_policy.mode), "margin_threshold": float(trace_policy.margin_threshold), "debug_sample_rate": float(trace_policy.debug_sample_rate), "include_window_ids": sorted(trace_policy.include_window_ids), "max_traces_per_probe": int(trace_policy.max_traces_per_probe)},
        "outputs": {
            "baseline_probe_reference": str(baseline_csv),
            "discovery_split": str(discovery_split_csv),
            "heldout_split": str(heldout_split_csv),
            "discovery_objective_summary": str(discovery_objective_csv),
            "probe_rule_table": str(probe_rule_csv),
            "window_summary": str(window_summary_csv),
            "window_summary_nonzero": str(nonzero_window_csv),
            "window_importance_rank": str(window_rank_csv),
            "pixel_importance_rank": str(pixel_rank_csv),
            "sufficiency_probe_results": str(suff_csv),
            "necessity_probe_results": str(nec_csv),
            "sufficiency_summary": str(suff_summary_csv),
            "necessity_summary": str(nec_summary_csv),
            "discovery_summary": str(discovery_summary_csv),
            "wrong_fixed_competitor_audit": str(wrong_fixed_audit_csv),
            "diagnostic_map_mean": str(diagnostic_map_mean_path),
            "diagnostic_mask_coverage": str(diagnostic_mask_coverage_path),
            "plot_diagnostic_maps_summary": summary_plot_paths,
        },
    }
    readme_path = save_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Deterministic Dense-Scan Discovery",
                "",
                "Default outputs are dense-scan summaries and full signed direction-score raw materials for downstream causal analysis.",
                "",
                "## Interpretation",
                "",
                "- Discovery stores the full signed direction score map and does not freeze D/N masks.",
                "- `importance_map_signed` is the primary projected score and is built from `raw_importance` or `raw_importance_fixed`.",
                "- `global_drop` and `discriminative_importance*` are audit-only compatibility fields.",
                "- Top-k `D` / `N` selection happens only in the causal stage within the probe foreground.",
                "",
                "## Legacy Outputs",
                "",
                "- `window_summary_nonzero.csv`, `critical_mask_nonzero.npy`, `positive_importance_mask.npy`, `negative_importance_mask.npy`, `sufficiency_probe_results.csv`, `necessity_probe_results.csv`, and `discovery_summary.csv` are retained as legacy / audit-only artifacts.",
                "- `lambda_global_selected`, `mask_policy`, and `discovery_frac` are recorded for compatibility but do not define the primary direction score or the causal D/N regions.",
                "",
                "## Run Config",
                "",
                "```json",
                json.dumps(run_config, indent=2, ensure_ascii=False),
                "```",
            ]
        ),
        encoding="utf-8",
    )
    run_config["outputs"]["README"] = str(readme_path)
    run_config_path = save_run_config(run_config, save_dir)

    print("\n=== Deterministic Dense-Scan Discovery Summary ===")
    print(f"Discovery probes: {int(discovery_df.shape[0])}")
    print(f"Held-out probes: {int(heldout_df.shape[0])}")
    print(f"Selected lambda_global={best_lambda:.2f}")
    print(f"Mask policy={str(args.mask_policy)}")
    print(f"Saved: {probe_rule_csv}")
    print(f"Saved: {window_summary_csv}")
    print(f"Saved: {nonzero_window_csv}")
    print(f"Saved: {window_rank_csv}")
    print(f"Saved: {pixel_rank_csv}")
    print(f"Saved: {suff_summary_csv}")
    print(f"Saved: {nec_summary_csv}")
    print(f"Saved: {discovery_summary_csv}")
    print(f"Saved: {diagnostic_map_mean_path}")
    print(f"Saved: {diagnostic_mask_coverage_path}")
    print(f"Saved: {readme_path}")
    print(f"Saved: {run_config_path}")


if __name__ == "__main__":
    main()
