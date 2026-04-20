from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.config.units import ms
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.sample_capture import generate_sample_capture_outputs
from src.experiments.common.specs import StepSpecMixin
from src.plotting.common.io import (
    PUBLICATION_SINGLE_COLUMN_FIGSIZE,
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
    validate_required_columns,
)
from src.plotting.common.theme_tokens import (
    COLOR_OFFWHITE,
    DESTINATION_OUTCOME_COLORS,
    FIGSIZE_SINGLE_PANEL_HEATMAP,
    GRID_ALPHA_SOFT,
    LINE_WIDTH_GUIDE,
    LINE_WIDTH_SECONDARY,
    MARKER_CIRCLE,
    MODE_COLORS_DYNAMIC_STATIC,
    apply_standard_legend,
)

MODEL_ORDER: Tuple[str, ...] = ("static", "dynamic")
MODEL_COLORS: Dict[str, str] = dict(MODE_COLORS_DYNAMIC_STATIC)
DESTINATION_COLORS: Dict[str, str] = dict(DESTINATION_OUTCOME_COLORS)
DESTINATION_LABELS: Dict[str, str] = {
    "probe": "Pred = probe",
    "sample": "Pred = sample",
    "other": "Pred = other",
}


@dataclass(frozen=True)
class ExperimentSpec(StepSpecMixin):
    dt: float
    sample_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return self.ms_to_steps(self.sample_ms)

    @property
    def probe_steps(self) -> int:
        return self.ms_to_steps(self.probe_ms)


@dataclass(frozen=True)
class ModelBundle:
    model_type: str
    model_path: Path
    stsp_mode: str
    net: torch.nn.Module


def mix_seed(base_seed: int, *parts: int) -> int:
    value = int(base_seed) & 0xFFFFFFFF
    for idx, part in enumerate(parts, start=1):
        value = (value * 1664525 + 1013904223 + int(part) * (374761393 + idx * 97)) & 0xFFFFFFFF
    return int(value)


def _sorted_int_list(values: Iterable[int]) -> List[int]:
    return [int(v) for v in sorted(int(item) for item in values)]


def _normalize_optional_path(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    stripped = str(raw_path).strip()
    if not stripped:
        return None
    return Path(stripped).resolve()


def _validate_positive_steps(spec: ExperimentSpec, delay_ms: float) -> None:
    if spec.sample_steps <= 0:
        raise ValueError("--sample-ms must resolve to at least one step.")
    if spec.probe_steps <= 0:
        raise ValueError("--probe-ms must resolve to at least one step.")
    if spec.ms_to_steps(delay_ms) <= 0:
        raise ValueError("--delay must resolve to at least one step.")


def _resolve_probe_classes(num_classes: int, probe_class: int | None) -> List[int]:
    if probe_class is not None:
        if not (0 <= int(probe_class) < int(num_classes)):
            raise ValueError(f"--probe-class must be in [0, {num_classes - 1}]")
        return [int(probe_class)]
    return list(range(int(num_classes)))


def _sample_unique_indices(
    class_index: Mapping[int, Sequence[int]],
    target_class: int,
    n_samples: int,
    seed: int,
) -> List[int]:
    pool = [int(idx) for idx in class_index[int(target_class)]]
    if len(pool) < int(n_samples):
        raise ValueError(
            f"Class {target_class} has only {len(pool)} samples, which is fewer than requested n_per_class={n_samples}."
        )
    rng = random.Random(int(seed))
    return [int(idx) for idx in rng.sample(pool, int(n_samples))]


def build_trial_specs(
    class_index: Mapping[int, Sequence[int]],
    probe_classes: Sequence[int],
    num_classes: int,
    n_per_class: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, int]] = []
    global_pair_id = 0
    for probe_class in _sorted_int_list(probe_classes):
        probe_ids = _sample_unique_indices(
            class_index=class_index,
            target_class=int(probe_class),
            n_samples=int(n_per_class),
            seed=mix_seed(seed, 1001, int(probe_class)),
        )
        for sample_class in range(int(num_classes)):
            if int(sample_class) == int(probe_class):
                continue
            sample_ids = _sample_unique_indices(
                class_index=class_index,
                target_class=int(sample_class),
                n_samples=int(n_per_class),
                seed=mix_seed(seed, 2001, int(probe_class), int(sample_class)),
            )
            for pair_index, (sample_id, probe_id) in enumerate(zip(sample_ids, probe_ids)):
                rows.append(
                    {
                        "pair_id": int(global_pair_id),
                        "pair_index": int(pair_index),
                        "probe_class": int(probe_class),
                        "sample_class": int(sample_class),
                        "probe_image_id": int(probe_id),
                        "sample_image_id": int(sample_id),
                    }
                )
                global_pair_id += 1
    df_specs = pd.DataFrame(rows)
    if df_specs.empty:
        raise ValueError("No trial specs were generated.")
    return df_specs.sort_values(
        ["probe_class", "sample_class", "pair_index"],
        kind="stable",
    ).reset_index(drop=True)


def _get_image_from_cache(dataset, index: int, image_cache: Dict[int, torch.Tensor]) -> torch.Tensor:
    cached = image_cache.get(int(index))
    if cached is not None:
        return cached
    image = dataset[int(index)][0].detach().cpu()
    image_cache[int(index)] = image
    return image


def prepare_batch_spikes(
    dataset,
    batch_df: pd.DataFrame,
    encoder,
    spec: ExperimentSpec,
    device: torch.device,
    image_cache: Dict[int, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    sample_ids = batch_df["sample_image_id"].astype(int).tolist()
    probe_ids = batch_df["probe_image_id"].astype(int).tolist()
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

    sample_encoded = encode_images(encoder, sample_images, spec.sample_steps)
    probe_encoded = encode_images(encoder, probe_images, spec.probe_steps)

    sample_lookup = {int(idx): pos for pos, idx in enumerate(unique_sample_ids)}
    probe_lookup = {int(idx): pos for pos, idx in enumerate(unique_probe_ids)}
    sample_select = torch.tensor([sample_lookup[int(idx)] for idx in sample_ids], device=device, dtype=torch.long)
    probe_select = torch.tensor([probe_lookup[int(idx)] for idx in probe_ids], device=device, dtype=torch.long)
    sample_spikes = sample_encoded.index_select(0, sample_select)
    probe_spikes = probe_encoded.index_select(0, probe_select)
    return sample_spikes, probe_spikes


def load_model_bundles(
    model_path: str,
    static_ckpt: str | None,
    dynamic_ckpt: str | None,
    device: torch.device,
    spec: ExperimentSpec,
) -> Tuple[Dict[str, ModelBundle], object]:
    base_model = Path(model_path).resolve()
    static_model = _normalize_optional_path(static_ckpt) or base_model
    dynamic_model = _normalize_optional_path(dynamic_ckpt) or base_model
    max_duration_ms = max(float(spec.sample_ms), float(spec.probe_ms))

    bundles: Dict[str, ModelBundle] = {}
    shared_encoder = None
    for model_type, ckpt_path, stsp_mode in [
        ("static", static_model, "static_frozen"),
        ("dynamic", dynamic_model, "dynamic"),
    ]:
        net, encoder = load_model_and_encoder(
            model_path=ckpt_path,
            device=device,
            dt=spec.dt,
            max_duration_ms=max_duration_ms,
        )
        bundles[model_type] = ModelBundle(
            model_type=str(model_type),
            model_path=Path(ckpt_path).resolve(),
            stsp_mode=str(stsp_mode),
            net=net,
        )
        if shared_encoder is None:
            shared_encoder = encoder

    if shared_encoder is None:
        raise RuntimeError("Failed to initialize encoder.")
    return bundles, shared_encoder


def run_trials(
    model_bundles: Mapping[str, ModelBundle],
    encoder,
    dataset,
    df_specs: pd.DataFrame,
    spec: ExperimentSpec,
    delay_ms: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    validate_required_columns(
        df_specs,
        ["pair_id", "pair_index", "probe_class", "sample_class", "probe_image_id", "sample_image_id"],
    )
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    delay_steps = spec.ms_to_steps(delay_ms)
    image_cache: Dict[int, torch.Tensor] = {}
    records: List[Dict[str, int | float | str]] = []
    grouped = list(df_specs.groupby(["probe_class", "sample_class"], sort=True))
    total_batches = sum(math.ceil(len(group) / batch_size) for _, group in grouped) * len(model_bundles)

    with tqdm(total=total_batches, desc="FixedProbeVariedSample") as pbar:
        for (_, _), group in grouped:
            group = group.sort_values(["pair_index", "pair_id"], kind="stable").reset_index(drop=True)
            for start in range(0, len(group), batch_size):
                batch = group.iloc[start:start + batch_size].copy().reset_index(drop=True)
                sample_spikes, probe_spikes = prepare_batch_spikes(
                    dataset=dataset,
                    batch_df=batch,
                    encoder=encoder,
                    spec=spec,
                    device=device,
                    image_cache=image_cache,
                )

                for model_type in MODEL_ORDER:
                    bundle = model_bundles[str(model_type)]
                    with torch.no_grad():
                        out = bundle.net.forward_classify_session(
                            sample_spikes=sample_spikes,
                            test_spikes=probe_spikes,
                            delay_duration_steps=delay_steps,
                            stsp_mode=bundle.stsp_mode,
                        )
                    pred = out["prediction"].detach().cpu().numpy().astype(np.int64, copy=False)
                    for idx_in_batch, row in enumerate(batch.itertuples(index=False)):
                        records.append(
                            {
                                "seed": int(seed),
                                "delay_ms": float(delay_ms),
                                "model_type": str(model_type),
                                "pair_id": int(row.pair_id),
                                "pair_index": int(row.pair_index),
                                "probe_class": int(row.probe_class),
                                "sample_class": int(row.sample_class),
                                "probe_image_id": int(row.probe_image_id),
                                "sample_image_id": int(row.sample_image_id),
                                "pred_class": int(pred[idx_in_batch]),
                            }
                        )
                    pbar.update(1)

    df_trials = pd.DataFrame(records)
    if df_trials.empty:
        raise RuntimeError("No trial results were generated.")
    return df_trials.sort_values(
        ["model_type", "probe_class", "sample_class", "pair_index"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_condition_metrics(df_trials: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    validate_required_columns(
        df_trials,
        ["model_type", "probe_class", "sample_class", "pred_class"],
    )

    long_rows: List[Dict[str, int | float | str]] = []
    for (probe_class, sample_class, model_type), subset in df_trials.groupby(
        ["probe_class", "sample_class", "model_type"],
        sort=True,
    ):
        pred = subset["pred_class"].to_numpy(dtype=np.int64, copy=False)
        probe_class_int = int(probe_class)
        sample_class_int = int(sample_class)
        acc = float(np.mean(pred == probe_class_int))
        dest_probe = float(np.mean(pred == probe_class_int))
        dest_sample = float(np.mean(pred == sample_class_int))
        dest_other = float(1.0 - dest_probe - dest_sample)
        silent_rate = float(np.mean(pred == -1))
        long_rows.append(
            {
                "probe_class": probe_class_int,
                "sample_class": sample_class_int,
                "model_type": str(model_type),
                "n_trials": int(len(subset)),
                "Acc": acc,
                "DestProbe": dest_probe,
                "DestSample": dest_sample,
                "DestOther": dest_other,
                "SilentRate": silent_rate,
            }
        )

    df_metrics_long = pd.DataFrame(long_rows).sort_values(
        ["probe_class", "sample_class", "model_type"],
        kind="stable",
    ).reset_index(drop=True)
    pivot = df_metrics_long.pivot(
        index=["probe_class", "sample_class"],
        columns="model_type",
        values=["n_trials", "Acc", "DestProbe", "DestSample", "DestOther", "SilentRate"],
    )
    pivot.columns = [f"{metric}_{model_type}" for metric, model_type in pivot.columns.to_flat_index()]
    df_summary = pivot.reset_index()
    df_summary["DeltaAcc"] = df_summary["Acc_static"] - df_summary["Acc_dynamic"]
    return df_metrics_long, df_summary.sort_values(["probe_class", "sample_class"], kind="stable").reset_index(drop=True)


def build_prediction_histogram(df_trials: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    validate_required_columns(
        df_trials,
        ["probe_class", "sample_class", "model_type", "pred_class"],
    )
    rows: List[Dict[str, int | float | str]] = []
    for (probe_class, sample_class, model_type), subset in df_trials.groupby(
        ["probe_class", "sample_class", "model_type"],
        sort=True,
    ):
        pred = subset["pred_class"].to_numpy(dtype=np.int64, copy=False)
        total = int(len(subset))
        for pred_class in range(int(num_classes)):
            count = int(np.sum(pred == int(pred_class)))
            rows.append(
                {
                    "probe_class": int(probe_class),
                    "sample_class": int(sample_class),
                    "model_type": str(model_type),
                    "pred_class": int(pred_class),
                    "count": int(count),
                    "rate": float(count / total) if total > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["probe_class", "sample_class", "model_type", "pred_class"],
        kind="stable",
    ).reset_index(drop=True)


def compute_probe_selection_scores(df_summary: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df_summary, ["probe_class", "Acc_static", "Acc_dynamic", "DeltaAcc"])
    rows: List[Dict[str, int | float | bool]] = []
    for probe_class, subset in df_summary.groupby("probe_class", sort=True):
        delta = subset["DeltaAcc"].to_numpy(dtype=np.float64, copy=False)
        rows.append(
            {
                "probe_class": int(probe_class),
                "static_mean_acc": float(subset["Acc_static"].mean()),
                "dynamic_mean_acc": float(subset["Acc_dynamic"].mean()),
                "mean_deltaacc": float(subset["DeltaAcc"].mean()),
                "deltaacc_std": float(np.std(delta, ddof=0)),
                "deltaacc_range": float(np.max(delta) - np.min(delta)),
                "n_conditions": int(len(subset)),
            }
        )
    df_scores = pd.DataFrame(rows)
    if df_scores.empty:
        raise ValueError("No probe selection scores could be computed.")

    in_band = df_scores["static_mean_acc"].between(0.60, 0.95, inclusive="both")
    df_scores["in_target_static_band"] = in_band.astype(bool)
    candidate_mask = in_band if bool(in_band.any()) else pd.Series(True, index=df_scores.index)
    df_scores["selection_candidate"] = candidate_mask.astype(bool)

    rank_source = df_scores[candidate_mask].copy()
    rank_source = rank_source.sort_values(
        ["deltaacc_std", "mean_deltaacc", "static_mean_acc", "probe_class"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    rank_source["selection_rank"] = np.arange(1, len(rank_source) + 1, dtype=np.int64)
    df_scores = df_scores.merge(
        rank_source[["probe_class", "selection_rank"]],
        on="probe_class",
        how="left",
        validate="one_to_one",
    )
    return df_scores.sort_values(["selection_rank", "probe_class"], kind="stable").reset_index(drop=True)


def select_primary_probe(df_summary: pd.DataFrame) -> Tuple[int, pd.DataFrame, str]:
    if df_summary["probe_class"].nunique() == 1:
        probe_class = int(df_summary["probe_class"].iloc[0])
        scores = compute_probe_selection_scores(df_summary)
        note = "Only one probe class was evaluated, so it was used as the primary display probe."
        return probe_class, scores, note

    scores = compute_probe_selection_scores(df_summary)
    selected_row = scores.loc[scores["selection_rank"].eq(1)].iloc[0]
    selected_probe = int(selected_row["probe_class"])
    if bool(scores["in_target_static_band"].any()):
        note = (
            "Selected from probes with static_mean_acc in [0.60, 0.95], then ranked by "
            "deltaacc_std, mean_deltaacc, static_mean_acc, and probe_class."
        )
    else:
        note = (
            "No probe satisfied the static_mean_acc band [0.60, 0.95], so all probes were ranked by "
            "deltaacc_std, mean_deltaacc, static_mean_acc, and probe_class."
        )
    return selected_probe, scores, note


def _errorbar_safe_upper(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return float(np.max(finite))


def plot_probe_accuracy(df_summary: pd.DataFrame, probe_class: int) -> plt.Figure:
    apply_publication_style()
    subset = (
        df_summary[df_summary["probe_class"] == int(probe_class)]
        .sort_values("sample_class", kind="stable")
        .reset_index(drop=True)
    )
    sample_classes = subset["sample_class"].to_numpy(dtype=np.int64, copy=False)
    static_acc = subset["Acc_static"].to_numpy(dtype=np.float64, copy=False) * 100.0
    dynamic_acc = subset["Acc_dynamic"].to_numpy(dtype=np.float64, copy=False) * 100.0

    fig, ax = plt.subplots(figsize=PUBLICATION_SINGLE_COLUMN_FIGSIZE)
    ax.plot(sample_classes, static_acc, marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_SECONDARY, color=MODEL_COLORS["static"], label="Static")
    ax.plot(sample_classes, dynamic_acc, marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_SECONDARY, color=MODEL_COLORS["dynamic"], label="Dynamic")
    ax.set_xticks(sample_classes)
    ax.set_xlabel("Sample class")
    ax.set_ylabel("Probe accuracy (%)")
    ax.set_ylim(0.0, max(100.0, _errorbar_safe_upper(np.concatenate([static_acc, dynamic_acc])) + 5.0))
    ax.set_title(f"Fixed probe = {int(probe_class)}")
    ax.grid(alpha=GRID_ALPHA_SOFT)
    apply_standard_legend(ax, title=None)
    fig.tight_layout()
    return fig


def plot_probe_destination_stacked(df_summary: pd.DataFrame, probe_class: int) -> plt.Figure:
    apply_publication_style()
    subset = (
        df_summary[df_summary["probe_class"] == int(probe_class)]
        .sort_values("sample_class", kind="stable")
        .reset_index(drop=True)
    )
    sample_classes = subset["sample_class"].to_numpy(dtype=np.int64, copy=False)
    x = np.arange(len(sample_classes), dtype=np.float64)
    width = 0.38

    fig, ax = plt.subplots(figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    for model_idx, model_type in enumerate(MODEL_ORDER):
        offset = (-0.5 if model_idx == 0 else 0.5) * width
        alpha = 0.55 if model_type == "static" else 0.95
        hatch = "//" if model_type == "static" else ""
        probe_rate = subset[f"DestProbe_{model_type}"].to_numpy(dtype=np.float64, copy=False) * 100.0
        sample_rate = subset[f"DestSample_{model_type}"].to_numpy(dtype=np.float64, copy=False) * 100.0
        other_rate = subset[f"DestOther_{model_type}"].to_numpy(dtype=np.float64, copy=False) * 100.0

        bottom = np.zeros_like(probe_rate)
        for key, values in [("probe", probe_rate), ("sample", sample_rate), ("other", other_rate)]:
            ax.bar(
                x + offset,
                values,
                width=width,
                bottom=bottom,
                color=DESTINATION_COLORS[key],
                alpha=alpha,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.5,
            )
            bottom = bottom + values

    destination_handles = [
        Patch(facecolor=DESTINATION_COLORS[key], edgecolor="black", label=DESTINATION_LABELS[key])
        for key in ("probe", "sample", "other")
    ]
    model_handles = [
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Static"),
        Patch(facecolor="white", edgecolor="black", label="Dynamic"),
    ]
    legend1 = apply_standard_legend(ax, handles=destination_handles, loc="upper left", title=None)
    ax.add_artist(legend1)
    apply_standard_legend(ax, handles=model_handles, loc="upper right", title=None)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in sample_classes])
    ax.set_xlabel("Sample class")
    ax.set_ylabel("Destination rate (%)")
    ax.set_ylim(0.0, 100.0)
    ax.set_title(f"Prediction destinations for fixed probe = {int(probe_class)}")
    ax.grid(axis="y", alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def _build_heatmap_matrix(df_summary: pd.DataFrame, value_column: str, num_classes: int) -> np.ndarray:
    validate_required_columns(df_summary, ["probe_class", "sample_class", value_column])
    matrix = np.full((int(num_classes), int(num_classes)), np.nan, dtype=np.float64)
    for row in df_summary.itertuples(index=False):
        matrix[int(row.sample_class), int(row.probe_class)] = float(getattr(row, value_column))
    for class_idx in range(int(num_classes)):
        matrix[class_idx, class_idx] = np.nan
    return matrix


def plot_heatmap(
    matrix: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap_name: str,
    symmetric: bool,
) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_HEATMAP)
    display = matrix * 100.0
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=COLOR_OFFWHITE)

    finite = display[np.isfinite(display)]
    if symmetric:
        vmax = float(np.max(np.abs(finite))) if finite.size > 0 else 1.0
        if vmax <= 0.0:
            vmax = 1.0
        vmin = -vmax
    else:
        vmin = 0.0
        vmax = float(np.max(finite)) if finite.size > 0 else 1.0
        if vmax <= 0.0:
            vmax = 1.0

    im = ax.imshow(display, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xlabel("Probe class")
    ax.set_ylabel("Sample class")
    ax.set_title(title)
    ax.set_xticklabels([str(i) for i in range(matrix.shape[1])])
    ax.set_yticklabels([str(i) for i in range(matrix.shape[0])])
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=LINE_WIDTH_GUIDE)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig


def validate_outputs(
    df_trials: pd.DataFrame,
    df_summary: pd.DataFrame,
    probe_classes: Sequence[int],
    n_per_class: int,
) -> None:
    if df_trials.empty:
        raise ValueError("Trial dataframe is empty.")
    if (df_trials["probe_class"] == df_trials["sample_class"]).any():
        raise ValueError("Found matched sample/probe rows, but only mismatched rows are allowed.")

    counts = (
        df_trials.groupby(["model_type", "probe_class", "sample_class"], as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
    )
    if not counts["n_rows"].eq(int(n_per_class)).all():
        bad_rows = counts.loc[~counts["n_rows"].eq(int(n_per_class))].head(10).to_dict("records")
        raise ValueError(f"Each (model_type, probe_class, sample_class) must have n_per_class rows: {bad_rows}")

    totals = df_summary["DestProbe_static"] + df_summary["DestSample_static"] + df_summary["DestOther_static"]
    if not np.allclose(totals.to_numpy(dtype=np.float64), 1.0, atol=1e-8):
        raise ValueError("Static destination rates do not sum to 1.")
    totals = df_summary["DestProbe_dynamic"] + df_summary["DestSample_dynamic"] + df_summary["DestOther_dynamic"]
    if not np.allclose(totals.to_numpy(dtype=np.float64), 1.0, atol=1e-8):
        raise ValueError("Dynamic destination rates do not sum to 1.")

    observed_probes = sorted(pd.unique(df_summary["probe_class"]).tolist())
    if observed_probes != _sorted_int_list(probe_classes):
        raise ValueError(f"Observed probe classes {observed_probes} do not match requested probe classes {probe_classes}.")


def build_summary_markdown(
    *,
    selected_probe: int,
    selection_note: str,
    df_selection_scores: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_trials: pd.DataFrame,
) -> str:
    probe_scores = df_selection_scores[df_selection_scores["probe_class"] == int(selected_probe)].iloc[0]
    probe_summary = (
        df_summary[df_summary["probe_class"] == int(selected_probe)]
        .sort_values("sample_class", kind="stable")
        .reset_index(drop=True)
    )
    probe_trials = df_trials[df_trials["probe_class"] == int(selected_probe)].copy()
    static_trials = probe_trials[probe_trials["model_type"] == "static"].copy()
    dynamic_trials = probe_trials[probe_trials["model_type"] == "dynamic"].copy()

    top_drops = probe_summary.sort_values(
        ["DeltaAcc", "sample_class"],
        ascending=[False, True],
        kind="stable",
    ).head(3)
    top_drop_text = ", ".join(
        [
            f"s={int(row.sample_class)} (DeltaAcc={float(row.DeltaAcc) * 100.0:.1f} pp)"
            for row in top_drops.itertuples(index=False)
        ]
    )

    def _error_to_sample_share(df_subset: pd.DataFrame) -> float:
        pred = df_subset["pred_class"].to_numpy(dtype=np.int64, copy=False)
        probe = df_subset["probe_class"].to_numpy(dtype=np.int64, copy=False)
        sample = df_subset["sample_class"].to_numpy(dtype=np.int64, copy=False)
        error_mask = pred != probe
        if not np.any(error_mask):
            return float("nan")
        return float(np.mean(pred[error_mask] == sample[error_mask]))

    sample_bias_wins = int((probe_summary["DestSample_dynamic"] > probe_summary["DestOther_dynamic"]).sum())
    dynamic_sample_dest_mean = float(dynamic_trials["pred_class"].eq(dynamic_trials["sample_class"]).mean())
    dynamic_other_dest_mean = float(
        (
            (dynamic_trials["pred_class"] != dynamic_trials["probe_class"])
            & (dynamic_trials["pred_class"] != dynamic_trials["sample_class"])
        ).mean()
    )
    static_sample_dest_mean = float(static_trials["pred_class"].eq(static_trials["sample_class"]).mean())
    static_other_dest_mean = float(
        (
            (static_trials["pred_class"] != static_trials["probe_class"])
            & (static_trials["pred_class"] != static_trials["sample_class"])
        ).mean()
    )
    dynamic_error_sample_share = _error_to_sample_share(dynamic_trials)
    static_error_sample_share = _error_to_sample_share(static_trials)

    lines = [
        "# Fixed Probe / Varied Sample Summary",
        "",
        f"- Selected primary probe: `{int(selected_probe)}`",
        f"- Selection note: {selection_note}",
        (
            f"- Probe score snapshot: static_mean_acc={float(probe_scores.static_mean_acc) * 100.0:.1f}%, "
            f"dynamic_mean_acc={float(probe_scores.dynamic_mean_acc) * 100.0:.1f}%, "
            f"mean_DeltaAcc={float(probe_scores.mean_deltaacc) * 100.0:.1f} pp, "
            f"deltaacc_std={float(probe_scores.deltaacc_std) * 100.0:.1f} pp"
        ),
        "",
        "## Primary Probe Findings",
        "",
        (
            f"- Across the 9 mismatched sample classes, static mean accuracy was "
            f"{float(probe_summary['Acc_static'].mean()) * 100.0:.1f}% and dynamic mean accuracy was "
            f"{float(probe_summary['Acc_dynamic'].mean()) * 100.0:.1f}%, for an average DeltaAcc of "
            f"{float(probe_summary['DeltaAcc'].mean()) * 100.0:.1f} pp."
        ),
        f"- Largest probe-accuracy drops: {top_drop_text}",
        (
            f"- Dynamic sample-destination exceeded dynamic other-destination in {sample_bias_wins}/9 sample conditions "
            f"for the selected probe."
        ),
        (
            f"- Aggregate destination rates for the selected probe: "
            f"static sample={static_sample_dest_mean * 100.0:.1f}%, static other={static_other_dest_mean * 100.0:.1f}%, "
            f"dynamic sample={dynamic_sample_dest_mean * 100.0:.1f}%, dynamic other={dynamic_other_dest_mean * 100.0:.1f}%."
        ),
        (
            f"- Conditional on making an error, the prediction landed on the current sample class "
            f"{static_error_sample_share * 100.0:.1f}% of the time in static and "
            f"{dynamic_error_sample_share * 100.0:.1f}% of the time in dynamic."
            if np.isfinite(static_error_sample_share) and np.isfinite(dynamic_error_sample_share)
            else "- Conditional error-to-sample shares were not available because one condition produced no errors."
        ),
        "",
        "## Interpretation",
        "",
        (
            "- Use the selected probe figures and the full-probe heatmaps together: the line plot shows whether different "
            "sample classes systematically pull probe accuracy down, and the destination bars / heatmap show whether those "
            "errors are redirected toward the sample class rather than diffusing to unrelated classes."
        ),
        (
            "- `DestOther` is defined as the residual non-probe, non-sample outcome rate, so it also absorbs silent (`-1`) "
            "predictions if they occur."
        ),
    ]
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-probe varied-sample DMS evaluation.")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--static-ckpt", type=str, default=None)
    parser.add_argument("--dynamic-ckpt", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--output-dir", type=str, default="results/fixed_probe_varied_sample")
    parser.add_argument("--delay", type=float, default=500.0)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)
    parser.add_argument("--n-per-class", type=int, default=50)
    parser.add_argument("--probe-class", type=int, default=None)
    parser.add_argument("--run-all-probes", dest="run_all_probes", action="store_true")
    parser.add_argument("--no-run-all-probes", dest="run_all_probes", action="store_false")
    parser.set_defaults(run_all_probes=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.n_per_class <= 0:
        raise ValueError("--n-per-class must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_classes < 2:
        raise ValueError("--num-classes must be at least 2.")
    if not bool(args.run_all_probes) and args.probe_class is None:
        raise ValueError("--no-run-all-probes requires --probe-class to specify a single probe.")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    spec = ExperimentSpec(dt=1.0 * ms, sample_ms=float(args.sample_ms), probe_ms=float(args.probe_ms))
    _validate_positive_steps(spec=spec, delay_ms=float(args.delay))
    probe_classes = _resolve_probe_classes(num_classes=int(args.num_classes), probe_class=args.probe_class)

    layout = prepare_result_layout(args.output_dir)
    result_root = layout.root
    metrics_dir = layout.data_dir
    figures_dir = layout.figure_dir
    logs_dir = layout.log_dir

    model_bundles, encoder = load_model_bundles(
        model_path=args.model_path,
        static_ckpt=args.static_ckpt,
        dynamic_ckpt=args.dynamic_ckpt,
        device=device,
        spec=spec,
    )
    _, _, test_loader = build_mnist_skeleton_loader(
        root=args.dataset_root,
        batch_size=1,
        input_size=28,
        num_workers=0,
    )
    dataset = test_loader.dataset
    class_index = build_class_index(dataset, num_classes=int(args.num_classes))

    df_specs = build_trial_specs(
        class_index=class_index,
        probe_classes=probe_classes,
        num_classes=int(args.num_classes),
        n_per_class=int(args.n_per_class),
        seed=int(args.seed),
    )
    df_trials = run_trials(
        model_bundles=model_bundles,
        encoder=encoder,
        dataset=dataset,
        df_specs=df_specs,
        spec=spec,
        delay_ms=float(args.delay),
        batch_size=int(args.batch_size),
        device=device,
        seed=int(args.seed),
    )
    df_metrics_long, df_summary = summarize_condition_metrics(df_trials=df_trials)
    df_hist = build_prediction_histogram(df_trials=df_trials, num_classes=int(args.num_classes))
    validate_outputs(
        df_trials=df_trials,
        df_summary=df_summary,
        probe_classes=probe_classes,
        n_per_class=int(args.n_per_class),
    )

    selected_probe, df_selection_scores, selection_note = select_primary_probe(df_summary=df_summary)
    if int(selected_probe) not in [int(v) for v in probe_classes]:
        raise ValueError(f"Selected probe {selected_probe} was not part of the executed probe set.")

    trial_csv = save_tidy_csv(
        df_trials[
            [
                "model_type",
                "probe_class",
                "sample_class",
                "probe_image_id",
                "sample_image_id",
                "pred_class",
                "delay_ms",
                "seed",
                "pair_index",
                "pair_id",
            ]
        ],
        metrics_dir / "trial_level_predictions.csv",
        sort_by=["model_type", "probe_class", "sample_class", "pair_index"],
    )
    summary_csv = save_tidy_csv(
        df_summary[
            [
                "probe_class",
                "sample_class",
                "Acc_static",
                "Acc_dynamic",
                "DeltaAcc",
                "DestSample_static",
                "DestSample_dynamic",
                "DestOther_static",
                "DestOther_dynamic",
                "DestProbe_static",
                "DestProbe_dynamic",
                "SilentRate_static",
                "SilentRate_dynamic",
                "n_trials_static",
                "n_trials_dynamic",
            ]
        ],
        metrics_dir / "condition_summary.csv",
        sort_by=["probe_class", "sample_class"],
    )
    metrics_long_csv = save_tidy_csv(
        df_metrics_long,
        metrics_dir / "condition_summary_long.csv",
        sort_by=["probe_class", "sample_class", "model_type"],
    )
    hist_csv = save_tidy_csv(
        df_hist,
        metrics_dir / "prediction_histogram_long.csv",
        sort_by=["probe_class", "sample_class", "model_type", "pred_class"],
    )
    selection_csv = save_tidy_csv(
        df_selection_scores,
        metrics_dir / "probe_selection_scores.csv",
        sort_by=["selection_rank", "probe_class"],
    )
    sample_capture_result = generate_sample_capture_outputs(
        df_trials=df_trials,
        output_dir=output_dir / "sample_capture",
        num_classes=int(args.num_classes),
        input_label=str(trial_csv),
    )

    fig1 = plot_probe_accuracy(df_summary=df_summary, probe_class=selected_probe)
    fig1_paths = save_figure_all_formats(fig1, figures_dir / "figure1_fixed_probe_accuracy")
    plt.close(fig1)

    fig2 = plot_probe_destination_stacked(df_summary=df_summary, probe_class=selected_probe)
    fig2_paths = save_figure_all_formats(fig2, figures_dir / "figure2_fixed_probe_destination")
    plt.close(fig2)

    deltaacc_matrix = _build_heatmap_matrix(df_summary=df_summary, value_column="DeltaAcc", num_classes=int(args.num_classes))
    fig3 = plot_heatmap(
        deltaacc_matrix,
        title="DeltaAcc(s->p)",
        cbar_label="DeltaAcc (pp)",
        cmap_name="RdBu_r",
        symmetric=True,
    )
    fig3_paths = save_figure_all_formats(fig3, figures_dir / "figure3_deltaacc_heatmap")
    plt.close(fig3)

    destsample_matrix = _build_heatmap_matrix(
        df_summary=df_summary,
        value_column="DestSample_dynamic",
        num_classes=int(args.num_classes),
    )
    fig4 = plot_heatmap(
        destsample_matrix,
        title="Dynamic sample-destination rate",
        cbar_label="DestSample dynamic (%)",
        cmap_name="viridis",
        symmetric=False,
    )
    fig4_paths = save_figure_all_formats(fig4, figures_dir / "figure4_destsample_dynamic_heatmap")
    plt.close(fig4)

    summary_md = build_summary_markdown(
        selected_probe=selected_probe,
        selection_note=selection_note,
        df_selection_scores=df_selection_scores,
        df_summary=df_summary,
        df_trials=df_trials,
    )
    summary_path = logs_dir / "summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    run_config_path = save_run_config(
        {
            "model_path": str(Path(args.model_path).resolve()),
            "static_ckpt": str(model_bundles["static"].model_path),
            "dynamic_ckpt": str(model_bundles["dynamic"].model_path),
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "output_dir": str(output_dir.resolve()),
            "device": str(device),
            "seed": int(args.seed),
            "delay_ms": float(args.delay),
            "sample_ms": float(args.sample_ms),
            "probe_ms": float(args.probe_ms),
            "n_per_class": int(args.n_per_class),
            "probe_classes": [int(v) for v in probe_classes],
            "run_all_probes": bool(args.run_all_probes),
            "num_classes": int(args.num_classes),
            "batch_size": int(args.batch_size),
            "selection_note": str(selection_note),
            "assumptions": {
                "static_vs_dynamic": "Single checkpoint can be evaluated under static_frozen and dynamic STSP modes.",
                "dest_other_definition": "Residual non-probe, non-sample destination rate; silent predictions are absorbed here.",
                "histogram_definition": "Prediction histogram contains explicit rows for classes 0-9 only.",
            },
            "outputs": {
                "trial_level_predictions_csv": str(trial_csv),
                "condition_summary_csv": str(summary_csv),
                "condition_summary_long_csv": str(metrics_long_csv),
                "prediction_histogram_long_csv": str(hist_csv),
                "probe_selection_scores_csv": str(selection_csv),
                "sample_capture_overall_csv": str(sample_capture_result["overall_csv"]),
                "sample_capture_by_probe_csv": str(sample_capture_result["probe_csv"]),
                "sample_capture_by_pair_csv": str(sample_capture_result["pair_csv"]),
                "sample_capture_summary_txt": str(sample_capture_result["summary_path"]),
                "summary_md": str(summary_path),
                "figure1_png": fig1_paths["png"],
                "figure2_png": fig2_paths["png"],
                "figure3_png": fig3_paths["png"],
                "figure4_png": fig4_paths["png"],
            },
        },
        result_root,
    )
    summary_json_path = save_summary_json(
        {
            "experiment": "fixed_probe_varied_sample",
            "selected_probe": int(selected_probe),
            "selection_note": str(selection_note),
            "trial_count": int(len(df_trials)),
            "summary_row_count": int(len(df_summary)),
            "run_config_json": str(run_config_path.resolve()),
        },
        result_root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=fixed_probe_varied_sample",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"seed={int(args.seed)}",
            f"device={device}",
            f"selected_probe={int(selected_probe)}",
            f"trials={len(df_trials)}",
            f"result_root={result_root.resolve()}",
            f"summary_json={summary_json_path.resolve()}",
        ],
        logs_dir,
    )

    print(f"[Done] Saved: {trial_csv}")
    print(f"[Done] Saved: {summary_csv}")
    print(f"[Done] Saved: {metrics_long_csv}")
    print(f"[Done] Saved: {hist_csv}")
    print(f"[Done] Saved: {selection_csv}")
    print(f"[Done] Saved: {sample_capture_result['overall_csv']}")
    print(f"[Done] Saved: {sample_capture_result['probe_csv']}")
    print(f"[Done] Saved: {sample_capture_result['pair_csv']}")
    print(f"[Done] Saved: {sample_capture_result['summary_path']}")
    print(f"[Done] Saved: {summary_path}")
    print(f"[Done] Saved: {run_config_path}")
    print(f"[Done] Saved: {summary_json_path}")
    print(f"[Done] Saved: {run_log_path}")


if __name__ == "__main__":
    main()
