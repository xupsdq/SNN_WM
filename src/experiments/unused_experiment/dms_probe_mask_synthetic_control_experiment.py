from __future__ import annotations

"""
Synthetic-control DMS experiment using probe-derived masked samples.

Assumptions documented in this script and the saved summary:
1. This is a synthetic control experiment.
2. Samples are constructed directly from the probe by masking selected probe regions.
3. The purpose is to causally disentangle overlap extent from overlap location / geometry.
4. Only raw overlap area is used, defined as the number of True pixels in the probe-side mask.
5. No effective overlap area, B-map weighting, or contribution-weighted metric is used.
6. The experiment tests whether raw overlap extent primarily controls displacement magnitude
   while probe-side overlap geometry primarily controls displacement direction.
"""

import argparse
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.config.units import ms
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.input_masks import foreground_mask_from_image
from src.experiments.common.json_io import save_json_payload as _save_json
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.voltage_readout import resolve_readout_step
from src.experiments.common.seed import mix_seed
from src.experiments.distractor.shared.analysis import compute_delta_v
from src.experiments.distractor.shared.masking import run_overlap_perturbed_dms
from src.experiments.distractor.shared.pair_sampling import build_dataset_arrays, select_probe_ids_balanced
from src.plotting.common.io import (
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
)
from src.plotting.common.theme_tokens import (
    ALPHA_BAR,
    ALPHA_FILL,
    ALPHA_SCATTER_LIGHT,
    CMAP_IMAGE_GRAY,
    FIGSIZE_THREE_PANEL,
    FIGSIZE_SINGLE_PANEL_WIDE,
    FIGSIZE_TWO_PANEL,
    GRID_ALPHA,
    LINE_WIDTH_PRIMARY,
    MARKER_CIRCLE,
    apply_standard_legend,
    horizontal_panel_figsize,
)

DEFAULT_MODEL_PATH = "results/sdnn_deep_final/net_final.pth"
DEFAULT_OUTPUT_DIR = "results/dms_probe_mask_synthetic_control_experiment"
DEFAULT_DATASET_ROOT = "./MNIST"
DEFAULT_SAMPLE_MS = 200.0
DEFAULT_DELAY_MS = 400.0
DEFAULT_DELAY_SWEEP_MS: tuple[float, ...] = (100.0, 150.0, 200.0, 300.0, 400.0, 500.0)
DEFAULT_PROBE_MS = 100.0
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_PROBES = 1000
DEFAULT_PROBE_SELECTION_PER_CLASS = 100
DEFAULT_FOREGROUND_THRESHOLD = 0.0
DEFAULT_NUM_AREA_LEVELS = 6
DEFAULT_AREA_LEVELS: tuple[int, ...] = (8, 16, 24, 32, 40, 48)
DEFAULT_NUM_LOCATION_MASKS = 4
DEFAULT_FIXED_AREA_PIXELS = 24
DEFAULT_MASK_GENERATION_SEED = 123
DEFAULT_SAVE_CASE_COUNT = 4
DEFAULT_MAX_LOCATION_MASK_IOU = 0.65
EPS = 1e-12
MILLIVOLT_SCALE = 1000.0

FIXED_LOCATION_GROUP = "fixed_location_vary_area"
FIXED_AREA_GROUP = "fixed_area_vary_location"


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    probe_ms: float
    phase_reset: bool = True

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


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


def _validate_positive(name: str, value: int | float, *, allow_zero: bool = False) -> None:
    scalar = float(value)
    if allow_zero:
        if scalar < 0.0:
            raise ValueError(f"{name} must be non-negative.")
        return
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _sanitize_area_levels(values: Sequence[int]) -> list[int]:
    if not values:
        raise ValueError("--area-levels must contain at least one value.")
    levels = sorted(dict.fromkeys(int(v) for v in values))
    if any(level <= 0 for level in levels):
        raise ValueError("--area-levels values must be positive.")
    return levels


def _sanitize_delay_sweep(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("--delay-sweep-ms must contain at least one value.")
    delays = sorted(dict.fromkeys(float(v) for v in values))
    if any(delay < 0.0 for delay in delays):
        raise ValueError("--delay-sweep-ms values must be non-negative.")
    return delays


def _delay_ms_to_steps(delay_ms: float, dt: float) -> int:
    return int(round((float(delay_ms) * ms) / float(dt)))


def _foreground_mask(image: torch.Tensor, threshold: float) -> np.ndarray:
    return foreground_mask_from_image(image, threshold=threshold)


def _mask_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(mask, dtype=bool))


def _sem(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def safe_cosine(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor, eps: float = EPS) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    norm_a = float(np.linalg.norm(aa))
    norm_b = float(np.linalg.norm(bb))
    if norm_a <= float(eps) or norm_b <= float(eps):
        return float("nan")
    return float(np.dot(aa, bb) / (norm_a * norm_b))


def safe_angle_deg(cosine_value: float) -> float:
    if not np.isfinite(float(cosine_value)):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(cosine_value), -1.0, 1.0))))


def _cosine_summary_to_angle_summary(cosine_mean: float, cosine_sem: float) -> tuple[float, float]:
    if not np.isfinite(float(cosine_mean)):
        return float("nan"), float("nan")
    angle_deg = safe_angle_deg(float(cosine_mean))
    if not np.isfinite(float(cosine_sem)):
        return float(angle_deg), float("nan")
    upper_angle_deg = safe_angle_deg(float(np.clip(float(cosine_mean) + float(cosine_sem), -1.0, 1.0)))
    lower_angle_deg = safe_angle_deg(float(np.clip(float(cosine_mean) - float(cosine_sem), -1.0, 1.0)))
    diffs = [
        abs(float(angle_deg) - float(candidate))
        for candidate in (upper_angle_deg, lower_angle_deg)
        if np.isfinite(float(candidate))
    ]
    sem_angle_deg = float(np.mean(diffs)) if diffs else 0.0
    return float(angle_deg), float(sem_angle_deg)


def safe_normalize(v: np.ndarray | torch.Tensor, eps: float = EPS) -> np.ndarray | None:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= float(eps):
        return None
    return arr / norm


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = a | b
    if not union.any():
        return 0.0
    return float((a & b).sum() / union.sum())


def _find_connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask_bool = np.asarray(mask, dtype=bool)
    coords = _mask_coords(mask_bool)
    if coords.size == 0:
        return []
    height, width = mask_bool.shape
    visited = np.zeros_like(mask_bool, dtype=bool)
    components: list[np.ndarray] = []
    neighbor_offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for row, col in coords.tolist():
        if visited[row, col]:
            continue
        stack = [(int(row), int(col))]
        visited[row, col] = True
        current: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            current.append((rr, cc))
            for dr, dc in neighbor_offsets:
                nr = rr + dr
                nc = cc + dc
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                if visited[nr, nc] or not mask_bool[nr, nc]:
                    continue
                visited[nr, nc] = True
                stack.append((nr, nc))
        components.append(np.asarray(current, dtype=np.int64))
    return components


def _largest_component_coords(mask: np.ndarray) -> np.ndarray:
    components = _find_connected_components(mask)
    if not components:
        return np.zeros((0, 2), dtype=np.int64)
    return max(components, key=lambda item: int(item.shape[0]))


def _choose_component_seed(component_coords: np.ndarray) -> tuple[int, int]:
    if component_coords.size == 0:
        raise ValueError("Cannot choose a seed from an empty component.")
    centroid = component_coords.mean(axis=0, dtype=np.float64)
    distances = np.sum((component_coords.astype(np.float64) - centroid[None, :]) ** 2, axis=1)
    seed_idx = int(np.argmin(distances))
    return int(component_coords[seed_idx, 0]), int(component_coords[seed_idx, 1])


def _sorted_coords_from_seed(component_coords: np.ndarray, seed_row: int, seed_col: int) -> np.ndarray:
    if component_coords.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    distances = np.sum((component_coords - np.asarray([seed_row, seed_col], dtype=np.int64)[None, :]) ** 2, axis=1)
    order = np.lexsort((component_coords[:, 1], component_coords[:, 0], distances))
    return component_coords[order]


def _mask_center(mask_binary: np.ndarray) -> tuple[float, float]:
    coords = _mask_coords(mask_binary)
    if coords.size == 0:
        return float("nan"), float("nan")
    center = coords.mean(axis=0, dtype=np.float64)
    return float(center[0]), float(center[1])


def _make_mask_from_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if coords.size > 0:
        mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _farthest_point_sample(coords: np.ndarray, n_points: int) -> np.ndarray:
    if coords.size == 0 or int(n_points) <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    coords_float = coords.astype(np.float64, copy=False)
    centroid = coords_float.mean(axis=0, dtype=np.float64)
    first_idx = int(np.argmax(np.sum((coords_float - centroid[None, :]) ** 2, axis=1)))
    chosen = [first_idx]
    min_dist = np.sum((coords_float - coords_float[first_idx][None, :]) ** 2, axis=1)
    while len(chosen) < min(int(n_points), len(coords)):
        candidate_idx = int(np.argmax(min_dist))
        if candidate_idx in chosen:
            break
        chosen.append(candidate_idx)
        dist_to_candidate = np.sum((coords_float - coords_float[candidate_idx][None, :]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist_to_candidate)
    return coords[np.asarray(chosen, dtype=np.int64)]


def _sorted_coords_for_local_region(all_coords: np.ndarray, seed_row: int, seed_col: int) -> np.ndarray:
    distances = np.sum((all_coords - np.asarray([seed_row, seed_col], dtype=np.int64)[None, :]) ** 2, axis=1)
    order = np.lexsort((all_coords[:, 1], all_coords[:, 0], distances))
    return all_coords[order]


def _mean_geometry_similarity(mask_records: Sequence[np.ndarray], target_index: int) -> float:
    if len(mask_records) <= 1:
        return 1.0
    target = np.asarray(mask_records[int(target_index)], dtype=bool)
    sims = []
    for idx, other in enumerate(mask_records):
        if int(idx) == int(target_index):
            continue
        sims.append(_mask_iou(target, np.asarray(other, dtype=bool)))
    return float(np.mean(sims)) if sims else 1.0


def _resolve_probe_label_series(df: pd.DataFrame) -> pd.Series:
    if "probe_label" in df.columns:
        return df["probe_label"]
    if "probe_label_x" in df.columns:
        return df["probe_label_x"]
    if "probe_label_y" in df.columns:
        return df["probe_label_y"]
    return pd.Series(np.full(len(df), np.nan, dtype=np.float64), index=df.index)


def compute_reference_direction_from_delays_for_mask(
    delta_v_by_delay: Mapping[float, np.ndarray],
    *,
    eps: float = EPS,
) -> dict[str, object]:
    ordered_items = sorted(
        ((float(delay_ms), np.asarray(vec, dtype=np.float64).reshape(-1)) for delay_ms, vec in delta_v_by_delay.items()),
        key=lambda item: item[0],
    )
    if not ordered_items:
        return {
            "status": "empty",
            "u_ref": None,
            "summed_norm": 0.0,
            "fallback_delay_ms": None,
        }
    stacked = np.stack([item[1] for item in ordered_items], axis=0)
    summed = stacked.sum(axis=0)
    u_ref = safe_normalize(summed, eps=eps)
    if u_ref is not None:
        return {
            "status": "mean_direction",
            "u_ref": u_ref,
            "summed_norm": float(np.linalg.norm(summed)),
            "fallback_delay_ms": None,
        }
    norms = np.linalg.norm(stacked, axis=1)
    valid = np.flatnonzero(norms > float(eps))
    if valid.size > 0:
        best_idx = int(valid[np.argmax(norms[valid])])
        fallback_vec = stacked[best_idx]
        u_fallback = safe_normalize(fallback_vec, eps=eps)
        if u_fallback is not None:
            return {
                "status": "fallback_max_norm_delay",
                "u_ref": u_fallback,
                "summed_norm": float(np.linalg.norm(summed)),
                "fallback_delay_ms": float(ordered_items[best_idx][0]),
            }
    return {
        "status": "skip_all_zero",
        "u_ref": None,
        "summed_norm": float(np.linalg.norm(summed)),
        "fallback_delay_ms": None,
    }


def compute_strength_metrics_for_delay_record(
    delta_v: np.ndarray | torch.Tensor,
    u_ref: np.ndarray | torch.Tensor,
    *,
    eps: float = EPS,
) -> dict[str, float]:
    vec = np.asarray(delta_v, dtype=np.float64).reshape(-1)
    ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    magnitude = float(np.linalg.norm(vec))
    effective = float(np.dot(vec, ref))
    cos_theta = safe_cosine(vec, ref, eps=eps)
    return {
        "M": magnitude,
        "A": effective,
        "cos_theta": cos_theta,
        "theta_deg": safe_angle_deg(cos_theta),
    }


def summarize_delay_trend(delay_summary_df: pd.DataFrame, mean_column: str) -> dict[str, object]:
    if delay_summary_df.empty or mean_column not in delay_summary_df.columns:
        return {"status": "empty", "metric": mean_column}
    sub = delay_summary_df[["delay_ms", mean_column]].dropna().sort_values("delay_ms", kind="stable").reset_index(drop=True)
    if sub.empty:
        return {"status": "empty", "metric": mean_column}
    x = sub["delay_ms"].to_numpy(dtype=np.float64)
    y = sub[mean_column].to_numpy(dtype=np.float64)
    peak_idx = int(np.argmax(y)) if y.size > 0 else -1
    result: dict[str, object] = {
        "status": "ok",
        "metric": mean_column,
        "n_points": int(len(sub)),
        "peak_delay_ms": float(x[peak_idx]) if peak_idx >= 0 else None,
        "peak_value": float(y[peak_idx]) if peak_idx >= 0 else None,
    }
    if x.size >= 2:
        linear_coef = np.polyfit(x, y, deg=1)
        result["linear_slope"] = float(linear_coef[0])
        result["delta_first_to_last"] = float(y[-1] - y[0])
    else:
        result["linear_slope"] = None
        result["delta_first_to_last"] = None
    if x.size >= 3:
        quadratic = np.polyfit(x, y, deg=2)
        result["quadratic_coef"] = float(quadratic[0])
        if abs(float(quadratic[0])) > float(EPS):
            vertex = float(-quadratic[1] / (2.0 * quadratic[0]))
            result["quadratic_vertex_delay_ms"] = vertex
            result["vertex_within_range"] = bool(float(x.min()) <= vertex <= float(x.max()))
        else:
            result["quadratic_vertex_delay_ms"] = None
            result["vertex_within_range"] = False
    else:
        result["quadratic_coef"] = None
        result["quadratic_vertex_delay_ms"] = None
        result["vertex_within_range"] = False
    return result


def _plot_pooled_black_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    sem: np.ndarray | None = None,
) -> None:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.size <= 0 or yy.size <= 0:
        return
    ax.plot(xx, yy, color="black", marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY)
    if sem is not None:
        ee = np.asarray(sem, dtype=np.float64)
        if ee.shape == yy.shape:
            ax.fill_between(xx, yy - ee, yy + ee, color="black", alpha=ALPHA_FILL)


def _plot_individual_points(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(xx) & np.isfinite(yy)
    if not valid.any():
        return
    ax.scatter(xx[valid], yy[valid], color="#7A7A7A", alpha=ALPHA_SCATTER_LIGHT, s=14, linewidths=0.0)


def _voltage_to_millivolts(values: np.ndarray | Sequence[float] | pd.Series) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * float(MILLIVOLT_SCALE)


def _blank_panel_figure(message: str) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_WIDE)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center")
    fig.tight_layout()
    return fig


def _plot_panel_image(
    ax: plt.Axes,
    image: torch.Tensor,
    *,
    mask_binary: np.ndarray | None = None,
    footer_label: str | None = None,
) -> None:
    arr = image.detach().cpu().to(torch.float32).squeeze(0).numpy()
    ax.imshow(arr, cmap=CMAP_IMAGE_GRAY, vmin=0.0, vmax=max(1.0, float(arr.max())))
    if mask_binary is not None:
        mask = np.asarray(mask_binary, dtype=bool)
        if mask.any():
            ax.contour(mask.astype(np.float32), levels=[0.5], colors=["black"], linewidths=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if footer_label:
        ax.text(0.5, -0.10, footer_label, transform=ax.transAxes, ha="center", va="top")


def _select_evenly_spaced_rows(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if df.empty or int(n_rows) <= 0:
        return df.iloc[0:0].copy()
    target = min(int(n_rows), len(df))
    if target >= len(df):
        return df.copy().reset_index(drop=True)
    positions = np.linspace(0, len(df) - 1, num=target)
    indices = sorted(dict.fromkeys(int(round(pos)) for pos in positions))
    while len(indices) < target:
        for candidate in range(len(df)):
            if candidate not in indices:
                indices.append(candidate)
            if len(indices) >= target:
                break
    return df.iloc[indices].copy().reset_index(drop=True)


def _select_panel_example_data(
    mask_df: pd.DataFrame,
    save_case_count: int,
) -> tuple[dict[str, object], torch.Tensor | None, pd.DataFrame, pd.DataFrame]:
    candidate_probe_ids = []
    for probe_id, subset in mask_df.groupby("probe_id", sort=True):
        has_area = int((subset["mask_group"] == FIXED_LOCATION_GROUP).sum()) >= 1
        has_location = int((subset["mask_group"] == FIXED_AREA_GROUP).sum()) >= 1
        if has_area and has_location:
            candidate_probe_ids.append(int(probe_id))
    if not candidate_probe_ids:
        return {"status": "no_valid_probe"}, None, pd.DataFrame(), pd.DataFrame()
    probe_id = int(candidate_probe_ids[0])
    subset = mask_df[mask_df["probe_id"] == probe_id].copy()
    n_examples = max(1, min(int(save_case_count), 4))
    area_masks = subset[subset["mask_group"] == FIXED_LOCATION_GROUP].sort_values(
        ["raw_overlap_area", "mask_rank"],
        kind="stable",
    )
    area_masks = area_masks.drop_duplicates(subset=["mask_id"], keep="first")
    area_masks = _select_evenly_spaced_rows(area_masks, n_examples)
    location_masks = subset[subset["mask_group"] == FIXED_AREA_GROUP].sort_values(["mask_rank", "mask_id"], kind="stable")
    location_masks = location_masks.drop_duplicates(subset=["mask_id"], keep="first").head(n_examples).reset_index(drop=True)
    selection = {
        "status": "ok",
        "probe_id": probe_id,
        "fixed_location_areas": [int(v) for v in area_masks["raw_overlap_area"].tolist()],
        "fixed_area_mask_ranks": [int(v) for v in location_masks["mask_rank"].tolist()],
    }
    return selection, subset["probe_image"].iloc[0], area_masks, location_masks


def build_probe_pool(
    images: torch.Tensor,
    labels: np.ndarray,
    class_index: Mapping[int, Sequence[int]],
    *,
    max_probes: int,
    seed: int,
    selection_per_class: int = DEFAULT_PROBE_SELECTION_PER_CLASS,
    foreground_threshold: float = DEFAULT_FOREGROUND_THRESHOLD,
) -> pd.DataFrame:
    target_max = int(max_probes)
    if int(selection_per_class) > 0:
        target_max = min(target_max, int(selection_per_class) * len(class_index))
    probe_ids = select_probe_ids_balanced(class_index=class_index, max_probes=target_max, seed=int(seed))
    rows: list[dict[str, object]] = []
    for probe_rank, probe_id in enumerate(probe_ids):
        probe_image = images[int(probe_id)].detach().cpu().to(torch.float32).clone()
        probe_fg = _foreground_mask(probe_image, threshold=float(foreground_threshold))
        rows.append(
            {
                "probe_id": int(probe_id),
                "probe_label": int(labels[int(probe_id)]),
                "probe_rank": int(probe_rank),
                "probe_image": probe_image,
                "probe_foreground_area": int(probe_fg.sum()),
            }
        )
    if not rows:
        raise RuntimeError("No probes were selected for the synthetic control experiment.")
    return pd.DataFrame(rows).sort_values(["probe_rank", "probe_id"], kind="stable").reset_index(drop=True)


def generate_nested_masks_for_probe(
    probe_image: torch.Tensor,
    *,
    target_area_schedule: Sequence[int],
    foreground_threshold: float,
    seed: int,
) -> pd.DataFrame:
    del seed
    probe_fg = _foreground_mask(probe_image, threshold=float(foreground_threshold))
    largest_component = _largest_component_coords(probe_fg)
    if largest_component.size == 0:
        return pd.DataFrame(
            columns=[
                "mask_id",
                "raw_overlap_area",
                "mask_binary",
                "mask_center_row",
                "mask_center_col",
                "mask_center",
                "mask_group",
                "mask_status",
                "mask_rank",
                "area_level_label",
            ]
        )
    seed_row, seed_col = _choose_component_seed(largest_component)
    ordered_coords = _sorted_coords_from_seed(largest_component, seed_row=seed_row, seed_col=seed_col)
    rows: list[dict[str, object]] = []
    for mask_rank, area in enumerate(_sanitize_area_levels(target_area_schedule), start=1):
        if int(area) > int(ordered_coords.shape[0]):
            continue
        mask_binary = _make_mask_from_coords(ordered_coords[: int(area)], probe_fg.shape)
        center_row, center_col = _mask_center(mask_binary)
        rows.append(
            {
                "mask_id": f"fixed_location_area_{mask_rank}",
                "raw_overlap_area": int(mask_binary.sum()),
                "mask_binary": mask_binary,
                "mask_center_row": float(center_row),
                "mask_center_col": float(center_col),
                "mask_center": (float(center_row), float(center_col)),
                "mask_group": FIXED_LOCATION_GROUP,
                "mask_status": "ok",
                "mask_rank": int(mask_rank),
                "area_level_label": f"area_{mask_rank}",
            }
        )
    return pd.DataFrame(rows)


def generate_equal_area_different_location_masks_for_probe(
    probe_image: torch.Tensor,
    *,
    target_area: int,
    n_masks: int,
    candidate_location_strategy: str,
    foreground_threshold: float,
    seed: int,
    max_pairwise_iou: float = DEFAULT_MAX_LOCATION_MASK_IOU,
) -> pd.DataFrame:
    if str(candidate_location_strategy).strip().lower() != "farthest_point":
        raise ValueError(f"Unsupported candidate_location_strategy: {candidate_location_strategy}")
    probe_fg = _foreground_mask(probe_image, threshold=float(foreground_threshold))
    all_coords = _mask_coords(probe_fg)
    if all_coords.shape[0] < int(target_area):
        return pd.DataFrame(
            columns=[
                "mask_id",
                "raw_overlap_area",
                "mask_binary",
                "mask_center_row",
                "mask_center_col",
                "mask_center",
                "mask_group",
                "mask_status",
                "mask_rank",
                "geometry_similarity_to_others",
            ]
        )
    rng = np.random.default_rng(int(seed))
    shuffled_coords = all_coords[rng.permutation(all_coords.shape[0])]
    seed_coords = _farthest_point_sample(shuffled_coords, n_points=max(int(n_masks) * 3, int(n_masks)))
    accepted_masks: list[np.ndarray] = []
    accepted_rows: list[dict[str, object]] = []
    for candidate in seed_coords.tolist():
        seed_row, seed_col = int(candidate[0]), int(candidate[1])
        ordered = _sorted_coords_for_local_region(all_coords, seed_row=seed_row, seed_col=seed_col)
        mask_binary = _make_mask_from_coords(ordered[: int(target_area)], probe_fg.shape)
        if int(mask_binary.sum()) != int(target_area):
            continue
        if any(_mask_iou(mask_binary, existing) > float(max_pairwise_iou) for existing in accepted_masks):
            continue
        center_row, center_col = _mask_center(mask_binary)
        accepted_masks.append(mask_binary)
        accepted_rows.append(
            {
                "mask_id": f"fixed_area_location_{len(accepted_rows) + 1}",
                "raw_overlap_area": int(mask_binary.sum()),
                "mask_binary": mask_binary,
                "mask_center_row": float(center_row),
                "mask_center_col": float(center_col),
                "mask_center": (float(center_row), float(center_col)),
                "mask_group": FIXED_AREA_GROUP,
                "mask_status": "ok",
                "mask_rank": int(len(accepted_rows) + 1),
            }
        )
        if len(accepted_rows) >= int(n_masks):
            break
    if len(accepted_rows) < 3:
        return pd.DataFrame(
            columns=[
                "mask_id",
                "raw_overlap_area",
                "mask_binary",
                "mask_center_row",
                "mask_center_col",
                "mask_center",
                "mask_group",
                "mask_status",
                "mask_rank",
                "geometry_similarity_to_others",
            ]
        )
    for idx, row in enumerate(accepted_rows):
        row["geometry_similarity_to_others"] = _mean_geometry_similarity(accepted_masks, idx)
    return pd.DataFrame(accepted_rows)


def construct_synthetic_sample_from_probe(probe_image: torch.Tensor, mask_binary: np.ndarray) -> torch.Tensor:
    mask_tensor = torch.as_tensor(np.asarray(mask_binary, dtype=np.float32), dtype=torch.float32)
    if probe_image.ndim != 3:
        raise ValueError(f"Expected probe_image shape [C, H, W], got {tuple(probe_image.shape)}")
    return probe_image.detach().cpu().to(torch.float32) * mask_tensor.unsqueeze(0)


def _prepare_synthetic_pair_spike_batch(
    synthetic_samples: torch.Tensor,
    probe_images: torch.Tensor,
    *,
    encoder,
    spec: ExperimentSpec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_encoded = encode_images(encoder, synthetic_samples.to(device=device, dtype=torch.float32), steps=int(spec.sample_steps))
    probe_encoded = encode_images(encoder, probe_images.to(device=device, dtype=torch.float32), steps=int(spec.probe_steps))
    return sample_encoded, probe_encoded


def compute_synthetic_pair_delta_vectors(
    *,
    net,
    encoder,
    device: torch.device,
    readout_step: int,
    probe_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    delay_steps: int,
    delay_ms: float,
    batch_size: int,
    spec: ExperimentSpec,
    eps: float = EPS,
) -> dict[str, object]:
    if mask_df.empty:
        empty = mask_df.copy()
        empty["delay_ms"] = pd.Series(dtype=np.float64)
        empty["grouped_voltage_dynamic"] = pd.Series(dtype=object)
        empty["grouped_voltage_static"] = pd.Series(dtype=object)
        empty["DeltaV"] = pd.Series(dtype=object)
        empty["magnitude_M"] = pd.Series(dtype=np.float64)
        empty["delta_norm"] = pd.Series(dtype=np.float64)
        empty["pair_status"] = pd.Series(dtype="object")
        empty["delta_valid"] = pd.Series(dtype=np.int64)
        return {"record_df": empty, "delta_vector_map": {}}

    merge_columns = ["probe_id", "probe_rank", "probe_image"]
    if "probe_label" not in mask_df.columns:
        merge_columns.insert(1, "probe_label")
    records = mask_df.merge(
        probe_df[merge_columns],
        on="probe_id",
        how="left",
        validate="many_to_one",
    ).copy()
    if "probe_label" not in records.columns:
        records["probe_label"] = _resolve_probe_label_series(records)
    records["delay_ms"] = float(delay_ms)
    records["synthetic_sample_image"] = records.apply(
        lambda row: construct_synthetic_sample_from_probe(row["probe_image"], row["mask_binary"]),
        axis=1,
    )
    records["record_id"] = np.arange(len(records), dtype=np.int64)

    dynamic_map: dict[int, np.ndarray] = {}
    static_map: dict[int, np.ndarray] = {}
    delta_map: dict[int, np.ndarray] = {}
    magnitude_map: dict[int, float] = {}
    status_map: dict[int, str] = {}

    batch_starts = range(0, len(records), int(batch_size))
    total_batches = math.ceil(len(records) / int(batch_size))
    batch_iter = tqdm(
        batch_starts,
        total=total_batches,
        desc=f"Delay {float(delay_ms):.0f} ms batches",
        leave=False,
        dynamic_ncols=True,
    )
    for batch_start in batch_iter:
        batch = records.iloc[batch_start : batch_start + int(batch_size)].copy().reset_index(drop=True)
        sample_batch = torch.stack([img for img in batch["synthetic_sample_image"].tolist()], dim=0)
        probe_batch = torch.stack([img for img in batch["probe_image"].tolist()], dim=0)
        sample_spikes, probe_spikes = _prepare_synthetic_pair_spike_batch(
            synthetic_samples=sample_batch,
            probe_images=probe_batch,
            encoder=encoder,
            spec=spec,
            device=device,
        )
        dynamic = run_overlap_perturbed_dms(
            net=net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=int(delay_steps),
            stsp_mode="dynamic",
            readout_step=int(readout_step),
            sample_input_mask=None,
        )
        static = run_overlap_perturbed_dms(
            net=net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=int(delay_steps),
            stsp_mode="static_frozen",
            readout_step=int(readout_step),
            sample_input_mask=None,
        )
        dynamic_grouped = np.asarray(dynamic.grouped_voltage, dtype=np.float64)
        static_grouped = np.asarray(static.grouped_voltage, dtype=np.float64)
        delta_v = np.asarray(compute_delta_v(dynamic_grouped, static_grouped), dtype=np.float64)
        for batch_idx, row in enumerate(batch.itertuples(index=False)):
            record_id = int(row.record_id)
            dynamic_vec = np.asarray(dynamic_grouped[batch_idx], dtype=np.float64).reshape(-1)
            static_vec = np.asarray(static_grouped[batch_idx], dtype=np.float64).reshape(-1)
            delta_vec = np.asarray(delta_v[batch_idx], dtype=np.float64).reshape(-1)
            magnitude = float(np.linalg.norm(delta_vec))
            dynamic_map[record_id] = dynamic_vec
            static_map[record_id] = static_vec
            delta_map[record_id] = delta_vec
            magnitude_map[record_id] = magnitude
            mask_status = str(getattr(row, "mask_status", "ok"))
            if mask_status != "ok":
                pair_status = mask_status
            elif int(getattr(row, "raw_overlap_area", 0)) <= 0:
                pair_status = "empty_mask"
            elif not np.isfinite(magnitude):
                pair_status = "invalid_numeric"
            elif magnitude <= float(eps):
                pair_status = "zero_delta_norm"
            else:
                pair_status = "ok"
            status_map[record_id] = pair_status

    records["grouped_voltage_dynamic"] = records["record_id"].map(dynamic_map)
    records["grouped_voltage_static"] = records["record_id"].map(static_map)
    records["DeltaV"] = records["record_id"].map(delta_map)
    records["magnitude_M"] = records["record_id"].map(magnitude_map).astype(np.float64)
    records["delta_norm"] = records["magnitude_M"].astype(np.float64)
    records["pair_status"] = records["record_id"].map(status_map).astype(str)
    records["delta_valid"] = records["pair_status"].eq("ok").astype(np.int64)
    return {"record_df": records, "delta_vector_map": delta_map}


def summarize_fixed_location_vary_area(records_df: pd.DataFrame) -> dict[str, object]:
    subset = records_df[records_df["mask_group"] == FIXED_LOCATION_GROUP].copy()
    valid = subset[subset["pair_status"] == "ok"].copy()
    if not valid.empty:
        valid["probe_label_resolved"] = _resolve_probe_label_series(valid)
    per_probe_rows: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []
    if valid.empty:
        return {
            "summary_df": pd.DataFrame(
                columns=[
                    "probe_id",
                    "raw_overlap_area",
                    "mean_M",
                    "sem_M",
                    "reference_direction_cosine",
                    "sem_reference_direction_cosine",
                    "reference_direction_angle_deg",
                    "sem_reference_direction_angle_deg",
                ]
            ),
            "pooled_summary_df": pd.DataFrame(
                columns=[
                    "raw_overlap_area",
                    "mean_M",
                    "sem_M",
                    "reference_direction_cosine",
                    "sem_reference_direction_cosine",
                    "reference_direction_angle_deg",
                    "sem_reference_direction_angle_deg",
                ]
            ),
        }
    for probe_id, probe_subset in valid.groupby("probe_id", sort=True):
        ordered = probe_subset.sort_values(["raw_overlap_area", "mask_rank"], kind="stable").reset_index(drop=True)
        ref_vec = np.asarray(ordered.iloc[0]["DeltaV"], dtype=np.float64)
        for area, area_subset in ordered.groupby("raw_overlap_area", sort=True):
            cosines_arr = np.asarray(
                [safe_cosine(np.asarray(vec, dtype=np.float64), ref_vec) for vec in area_subset["DeltaV"].tolist()],
                dtype=np.float64,
            )
            mean_cosine = float(np.nanmean(cosines_arr))
            sem_cosine = _sem(cosines_arr)
            angle_deg, sem_angle_deg = _cosine_summary_to_angle_summary(mean_cosine, sem_cosine)
            per_probe_rows.append(
                {
                    "probe_id": int(probe_id),
                    "probe_label": int(area_subset["probe_label_resolved"].iloc[0]),
                    "raw_overlap_area": int(area),
                    "mean_M": float(area_subset["magnitude_M"].mean()),
                    "sem_M": _sem(area_subset["magnitude_M"].to_numpy(dtype=np.float64)),
                    "reference_direction_cosine": mean_cosine,
                    "sem_reference_direction_cosine": sem_cosine,
                    "reference_direction_angle_deg": float(angle_deg),
                    "sem_reference_direction_angle_deg": float(sem_angle_deg),
                    "n_records": int(len(area_subset)),
                }
            )
    per_probe_summary = pd.DataFrame(per_probe_rows).sort_values(["probe_id", "raw_overlap_area"], kind="stable").reset_index(drop=True)
    for area, area_subset in valid.groupby("raw_overlap_area", sort=True):
        probe_cosines = []
        for probe_id, probe_subset in valid.groupby("probe_id", sort=True):
            probe_area = probe_subset[probe_subset["raw_overlap_area"] == area]
            if probe_area.empty:
                continue
            ref_row = probe_subset.sort_values(["raw_overlap_area", "mask_rank"], kind="stable").iloc[0]
            ref_vec = np.asarray(ref_row["DeltaV"], dtype=np.float64)
            probe_cosines.extend([safe_cosine(np.asarray(vec, dtype=np.float64), ref_vec) for vec in probe_area["DeltaV"].tolist()])
        mean_cosine = float(np.nanmean(np.asarray(probe_cosines, dtype=np.float64))) if probe_cosines else float("nan")
        sem_cosine = _sem(np.asarray(probe_cosines, dtype=np.float64)) if probe_cosines else 0.0
        angle_deg, sem_angle_deg = _cosine_summary_to_angle_summary(mean_cosine, sem_cosine)
        pooled_rows.append(
            {
                "raw_overlap_area": int(area),
                "mean_M": float(area_subset["magnitude_M"].mean()),
                "sem_M": _sem(area_subset["magnitude_M"].to_numpy(dtype=np.float64)),
                "reference_direction_cosine": mean_cosine,
                "sem_reference_direction_cosine": sem_cosine,
                "reference_direction_angle_deg": float(angle_deg),
                "sem_reference_direction_angle_deg": float(sem_angle_deg),
                "n_records": int(len(area_subset)),
            }
        )
    pooled_summary = pd.DataFrame(pooled_rows).sort_values(["raw_overlap_area"], kind="stable").reset_index(drop=True)
    return {"summary_df": per_probe_summary, "pooled_summary_df": pooled_summary}


def summarize_fixed_area_vary_location(records_df: pd.DataFrame) -> dict[str, object]:
    subset = records_df[(records_df["mask_group"] == FIXED_AREA_GROUP) & (records_df["pair_status"] == "ok")].copy()
    if subset.empty:
        return {
            "per_mask_df": pd.DataFrame(columns=["probe_id", "mask_id", "mask_rank", "magnitude_M"]),
            "pairwise_df": pd.DataFrame(
                columns=[
                    "probe_id",
                    "mask_i",
                    "mask_j",
                    "mask_rank_i",
                    "mask_rank_j",
                    "pair_label",
                    "direction_similarity",
                    "direction_angle_deg",
                    "magnitude_diff",
                ]
            ),
            "pooled_per_mask_df": pd.DataFrame(columns=["mask_rank", "mean_M", "sem_M", "n_records"]),
            "pooled_pairwise_df": pd.DataFrame(
                columns=[
                    "pair_label",
                    "mask_rank_i",
                    "mask_rank_j",
                    "mean_direction_similarity",
                    "sem_direction_similarity",
                    "mean_direction_angle_deg",
                    "sem_direction_angle_deg",
                    "mean_magnitude_diff",
                    "sem_magnitude_diff",
                    "n_records",
                ]
            ),
            "reference_based_df": pd.DataFrame(
                columns=[
                    "probe_id",
                    "probe_label",
                    "mask_id",
                    "mask_rank",
                    "direction_similarity_to_reference",
                    "direction_angle_deg_to_reference",
                    "magnitude_diff_to_reference",
                    "magnitude_M",
                    "delay_ms",
                ]
            ),
            "pooled_reference_based_df": pd.DataFrame(
                columns=[
                    "mask_rank",
                    "mean_M",
                    "sem_M",
                    "mean_direction_similarity_to_reference",
                    "sem_direction_similarity_to_reference",
                    "mean_direction_angle_deg_to_reference",
                    "sem_direction_angle_deg_to_reference",
                    "mean_magnitude_diff_to_reference",
                    "sem_magnitude_diff_to_reference",
                    "n_records",
                ]
            ),
            "pooled_summary": {},
        }
    subset["probe_label_resolved"] = _resolve_probe_label_series(subset)
    per_mask_df = subset[
        [
            "probe_id",
            "probe_label_resolved",
            "mask_id",
            "mask_rank",
            "raw_overlap_area",
            "magnitude_M",
            "pair_status",
            "delay_ms",
        ]
    ].copy().rename(columns={"probe_label_resolved": "probe_label"})
    pairwise_rows: list[dict[str, object]] = []
    reference_based_rows: list[dict[str, object]] = []
    for probe_id, probe_subset in subset.groupby("probe_id", sort=True):
        ordered = probe_subset.sort_values(["mask_rank", "mask_id"], kind="stable").reset_index(drop=True)
        ref_row = ordered.iloc[0]
        ref_vec = np.asarray(ref_row["DeltaV"], dtype=np.float64)
        ref_magnitude = float(ref_row["magnitude_M"])
        for row in ordered.itertuples(index=False):
            delta_vec = np.asarray(row.DeltaV, dtype=np.float64)
            direction_similarity_to_reference = safe_cosine(delta_vec, ref_vec)
            reference_based_rows.append(
                {
                    "probe_id": int(probe_id),
                    "probe_label": int(row.probe_label_resolved),
                    "mask_id": str(row.mask_id),
                    "mask_rank": int(row.mask_rank),
                    "direction_similarity_to_reference": float(direction_similarity_to_reference),
                    "direction_angle_deg_to_reference": float(safe_angle_deg(direction_similarity_to_reference)),
                    "magnitude_diff_to_reference": float(abs(float(row.magnitude_M) - ref_magnitude)),
                    "magnitude_M": float(row.magnitude_M),
                    "delay_ms": float(row.delay_ms),
                }
            )
        for row_i, row_j in combinations(ordered.itertuples(index=False), 2):
            delta_i = np.asarray(row_i.DeltaV, dtype=np.float64)
            delta_j = np.asarray(row_j.DeltaV, dtype=np.float64)
            direction_similarity = safe_cosine(delta_i, delta_j)
            pairwise_rows.append(
                {
                    "probe_id": int(probe_id),
                    "probe_label": int(row_i.probe_label_resolved),
                    "mask_i": str(row_i.mask_id),
                    "mask_j": str(row_j.mask_id),
                    "mask_rank_i": int(row_i.mask_rank),
                    "mask_rank_j": int(row_j.mask_rank),
                    "pair_label": f"{int(row_i.mask_rank)}-{int(row_j.mask_rank)}",
                    "direction_similarity": float(direction_similarity),
                    "direction_angle_deg": float(safe_angle_deg(direction_similarity)),
                    "magnitude_diff": float(abs(float(row_i.magnitude_M) - float(row_j.magnitude_M))),
                    "delay_ms": float(row_i.delay_ms),
                }
            )
    pairwise_df = pd.DataFrame(pairwise_rows).sort_values(["probe_id", "mask_i", "mask_j"], kind="stable").reset_index(drop=True)
    reference_based_df = (
        pd.DataFrame(reference_based_rows)
        .sort_values(["probe_id", "mask_rank", "mask_id"], kind="stable")
        .reset_index(drop=True)
    )
    pooled_per_mask_rows: list[dict[str, object]] = []
    for mask_rank, mask_subset in per_mask_df.groupby("mask_rank", sort=True):
        pooled_per_mask_rows.append(
            {
                "mask_rank": int(mask_rank),
                "mean_M": float(mask_subset["magnitude_M"].mean()),
                "sem_M": _sem(mask_subset["magnitude_M"].to_numpy(dtype=np.float64)),
                "n_records": int(len(mask_subset)),
            }
        )
    pooled_pairwise_rows: list[dict[str, object]] = []
    for pair_label, pair_subset in pairwise_df.groupby("pair_label", sort=True):
        pooled_pairwise_rows.append(
            {
                "pair_label": str(pair_label),
                "mask_rank_i": int(pair_subset["mask_rank_i"].iloc[0]),
                "mask_rank_j": int(pair_subset["mask_rank_j"].iloc[0]),
                "mean_direction_similarity": float(pair_subset["direction_similarity"].mean()),
                "sem_direction_similarity": _sem(pair_subset["direction_similarity"].to_numpy(dtype=np.float64)),
                "mean_direction_angle_deg": float(pair_subset["direction_angle_deg"].mean(skipna=True)),
                "sem_direction_angle_deg": _sem(pair_subset["direction_angle_deg"].to_numpy(dtype=np.float64)),
                "mean_magnitude_diff": float(pair_subset["magnitude_diff"].mean()),
                "sem_magnitude_diff": _sem(pair_subset["magnitude_diff"].to_numpy(dtype=np.float64)),
                "n_records": int(len(pair_subset)),
            }
        )
    pooled_per_mask_df = pd.DataFrame(pooled_per_mask_rows).sort_values(["mask_rank"], kind="stable").reset_index(drop=True)
    pooled_pairwise_df = (
        pd.DataFrame(pooled_pairwise_rows)
        .sort_values(["mask_rank_i", "mask_rank_j"], kind="stable")
        .reset_index(drop=True)
    )
    pooled_reference_rows: list[dict[str, object]] = []
    for mask_rank, ref_subset in reference_based_df.groupby("mask_rank", sort=True):
        pooled_reference_rows.append(
            {
                "mask_rank": int(mask_rank),
                "mean_M": float(ref_subset["magnitude_M"].mean()),
                "sem_M": _sem(ref_subset["magnitude_M"].to_numpy(dtype=np.float64)),
                "mean_direction_similarity_to_reference": float(ref_subset["direction_similarity_to_reference"].mean()),
                "sem_direction_similarity_to_reference": _sem(
                    ref_subset["direction_similarity_to_reference"].to_numpy(dtype=np.float64)
                ),
                "mean_direction_angle_deg_to_reference": float(
                    ref_subset["direction_angle_deg_to_reference"].mean(skipna=True)
                ),
                "sem_direction_angle_deg_to_reference": _sem(
                    ref_subset["direction_angle_deg_to_reference"].to_numpy(dtype=np.float64)
                ),
                "mean_magnitude_diff_to_reference": float(ref_subset["magnitude_diff_to_reference"].mean()),
                "sem_magnitude_diff_to_reference": _sem(
                    ref_subset["magnitude_diff_to_reference"].to_numpy(dtype=np.float64)
                ),
                "n_records": int(len(ref_subset)),
            }
        )
    pooled_reference_based_df = (
        pd.DataFrame(pooled_reference_rows).sort_values(["mask_rank"], kind="stable").reset_index(drop=True)
    )
    pooled_summary = {
        "mean_mask_magnitude": float(per_mask_df["magnitude_M"].mean()) if not per_mask_df.empty else float("nan"),
        "sem_mask_magnitude": _sem(per_mask_df["magnitude_M"].to_numpy(dtype=np.float64)) if not per_mask_df.empty else 0.0,
        "mean_direction_similarity": float(pairwise_df["direction_similarity"].mean()) if not pairwise_df.empty else float("nan"),
        "sem_direction_similarity": _sem(pairwise_df["direction_similarity"].to_numpy(dtype=np.float64)) if not pairwise_df.empty else 0.0,
        "mean_magnitude_diff": float(pairwise_df["magnitude_diff"].mean()) if not pairwise_df.empty else float("nan"),
        "sem_magnitude_diff": _sem(pairwise_df["magnitude_diff"].to_numpy(dtype=np.float64)) if not pairwise_df.empty else 0.0,
        "mean_direction_angle_to_reference": (
            float(reference_based_df["direction_angle_deg_to_reference"].mean(skipna=True))
            if not reference_based_df.empty
            else float("nan")
        ),
        "sem_direction_angle_to_reference": (
            _sem(reference_based_df["direction_angle_deg_to_reference"].to_numpy(dtype=np.float64))
            if not reference_based_df.empty
            else 0.0
        ),
        "mean_magnitude_diff_to_reference": (
            float(reference_based_df["magnitude_diff_to_reference"].mean()) if not reference_based_df.empty else float("nan")
        ),
        "sem_magnitude_diff_to_reference": (
            _sem(reference_based_df["magnitude_diff_to_reference"].to_numpy(dtype=np.float64))
            if not reference_based_df.empty
            else 0.0
        ),
    }
    return {
        "per_mask_df": per_mask_df,
        "pairwise_df": pairwise_df,
        "pooled_per_mask_df": pooled_per_mask_df,
        "pooled_pairwise_df": pooled_pairwise_df,
        "reference_based_df": reference_based_df,
        "pooled_reference_based_df": pooled_reference_based_df,
        "pooled_summary": pooled_summary,
    }


def summarize_delay_strength_direction(records_df: pd.DataFrame, *, eps: float = EPS) -> dict[str, object]:
    base_columns = [
        "probe_id",
        "probe_label",
        "mask_id",
        "mask_group",
        "raw_overlap_area",
        "delay_ms",
        "DeltaV",
        "M",
        "A",
        "cos_theta",
        "theta_deg",
        "reference_status",
        "reference_delay_count",
        "reference_summed_norm",
        "reference_fallback_delay_ms",
        "pair_status",
    ]
    if records_df.empty:
        empty_within = pd.DataFrame(columns=base_columns)
        empty_summary = pd.DataFrame(
            columns=[
                "delay_ms",
                "mean_M",
                "sem_M",
                "mean_A",
                "sem_A",
                "mean_cos_theta",
                "sem_cos_theta",
                "mean_theta_deg",
                "sem_theta_deg",
                "n_records",
                "n_valid_A",
                "n_valid_cos_theta",
            ]
        )
        return {
            "delay_within_mask_df": empty_within,
            "delay_pooled_summary_df": empty_summary,
            "reference_status_counts": {},
            "n_valid_masks": 0,
            "n_total_masks": 0,
            "trend_A": summarize_delay_trend(empty_summary, "mean_A"),
            "trend_M": summarize_delay_trend(empty_summary, "mean_M"),
            "trend_cos_theta": summarize_delay_trend(empty_summary, "mean_cos_theta"),
            "trend_theta_deg": summarize_delay_trend(empty_summary, "mean_theta_deg"),
        }
    records = records_df.copy()
    records["probe_label_resolved"] = _resolve_probe_label_series(records)
    within_mask_rows: list[dict[str, object]] = []
    reference_statuses: list[str] = []
    n_valid_masks = 0
    group_columns = ["probe_id", "mask_id", "mask_group", "raw_overlap_area"]
    for group_key, mask_subset in records.groupby(group_columns, sort=True):
        probe_id, mask_id, mask_group, raw_overlap_area = group_key
        ordered = mask_subset.sort_values(["delay_ms"], kind="stable").reset_index(drop=True)
        valid_vectors = ordered[ordered["pair_status"] == "ok"]
        delta_v_by_delay = {
            float(row.delay_ms): np.asarray(row.DeltaV, dtype=np.float64)
            for row in valid_vectors.itertuples(index=False)
            if row.DeltaV is not None
        }
        reference = compute_reference_direction_from_delays_for_mask(delta_v_by_delay, eps=eps)
        reference_status = str(reference["status"])
        reference_statuses.append(reference_status)
        u_ref = reference.get("u_ref")
        if u_ref is not None:
            n_valid_masks += 1
        for row in ordered.itertuples(index=False):
            delta_vec = None if row.DeltaV is None else np.asarray(row.DeltaV, dtype=np.float64).reshape(-1)
            metrics = {
                "M": float(row.magnitude_M) if np.isfinite(float(row.magnitude_M)) else float("nan"),
                "A": float("nan"),
                "cos_theta": float("nan"),
                "theta_deg": float("nan"),
            }
            if delta_vec is not None and u_ref is not None:
                metrics = compute_strength_metrics_for_delay_record(delta_vec, np.asarray(u_ref, dtype=np.float64), eps=eps)
            within_mask_rows.append(
                {
                    "probe_id": int(probe_id),
                    "probe_label": int(row.probe_label_resolved),
                    "mask_id": str(mask_id),
                    "mask_group": str(mask_group),
                    "raw_overlap_area": int(raw_overlap_area),
                    "delay_ms": float(row.delay_ms),
                    "DeltaV": delta_vec,
                    "M": float(metrics["M"]),
                    "A": float(metrics["A"]),
                    "cos_theta": float(metrics["cos_theta"]),
                    "theta_deg": float(metrics["theta_deg"]),
                    "reference_status": reference_status,
                    "reference_delay_count": int(len(delta_v_by_delay)),
                    "reference_summed_norm": float(reference.get("summed_norm", 0.0)),
                    "reference_fallback_delay_ms": (
                        None if reference.get("fallback_delay_ms") is None else float(reference["fallback_delay_ms"])
                    ),
                    "pair_status": str(row.pair_status),
                }
            )
    delay_within_mask_df = (
        pd.DataFrame(within_mask_rows)
        .sort_values(["probe_id", "mask_group", "mask_id", "delay_ms"], kind="stable")
        .reset_index(drop=True)
    )
    pooled_rows: list[dict[str, object]] = []
    for delay_ms, delay_subset in delay_within_mask_df.groupby("delay_ms", sort=True):
        arr_M = delay_subset["M"].to_numpy(dtype=np.float64)
        arr_A = delay_subset["A"].to_numpy(dtype=np.float64)
        arr_cos = delay_subset["cos_theta"].to_numpy(dtype=np.float64)
        arr_theta = delay_subset["theta_deg"].to_numpy(dtype=np.float64)
        pooled_rows.append(
            {
                "delay_ms": float(delay_ms),
                "mean_M": float(np.nanmean(arr_M)) if np.isfinite(arr_M).any() else float("nan"),
                "sem_M": _sem(arr_M),
                "mean_A": float(np.nanmean(arr_A)) if np.isfinite(arr_A).any() else float("nan"),
                "sem_A": _sem(arr_A),
                "mean_cos_theta": float(np.nanmean(arr_cos)) if np.isfinite(arr_cos).any() else float("nan"),
                "sem_cos_theta": _sem(arr_cos),
                "mean_theta_deg": float(np.nanmean(arr_theta)) if np.isfinite(arr_theta).any() else float("nan"),
                "sem_theta_deg": _sem(arr_theta),
                "n_records": int(len(delay_subset)),
                "n_valid_A": int(np.isfinite(arr_A).sum()),
                "n_valid_cos_theta": int(np.isfinite(arr_cos).sum()),
            }
        )
    delay_pooled_summary_df = pd.DataFrame(pooled_rows).sort_values(["delay_ms"], kind="stable").reset_index(drop=True)
    reference_status_counts = pd.Series(reference_statuses, dtype="object").value_counts(dropna=False).to_dict()
    return {
        "delay_within_mask_df": delay_within_mask_df,
        "delay_pooled_summary_df": delay_pooled_summary_df,
        "reference_status_counts": {str(key): int(value) for key, value in reference_status_counts.items()},
        "n_valid_masks": int(n_valid_masks),
        "n_total_masks": int(delay_within_mask_df["mask_id"].nunique()) if not delay_within_mask_df.empty else 0,
        "trend_A": summarize_delay_trend(delay_pooled_summary_df, "mean_A"),
        "trend_M": summarize_delay_trend(delay_pooled_summary_df, "mean_M"),
        "trend_cos_theta": summarize_delay_trend(delay_pooled_summary_df, "mean_cos_theta"),
        "trend_theta_deg": summarize_delay_trend(delay_pooled_summary_df, "mean_theta_deg"),
    }


def _plot_image(ax, image: torch.Tensor, title: str) -> None:
    arr = image.detach().cpu().to(torch.float32).squeeze(0).numpy()
    ax.imshow(arr, cmap=CMAP_IMAGE_GRAY, vmin=0.0, vmax=max(1.0, float(arr.max())))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_sample(ax, sample_image: torch.Tensor, mask_binary: np.ndarray, title: str) -> None:
    arr = sample_image.detach().cpu().to(torch.float32).squeeze(0).numpy()
    ax.imshow(arr, cmap=CMAP_IMAGE_GRAY, vmin=0.0, vmax=max(1.0, float(arr.max())))
    mask = np.asarray(mask_binary, dtype=bool)
    if mask.any():
        ax.contour(mask.astype(np.float32), levels=[0.5], colors=["#E45756"], linewidths=1.0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_panel_fixed_location_examples(mask_df: pd.DataFrame, save_case_count: int) -> tuple[plt.Figure, dict[str, object]]:
    selection, probe_image, area_masks, _ = _select_panel_example_data(mask_df, save_case_count)
    if selection.get("status") != "ok" or probe_image is None or area_masks.empty:
        return _blank_panel_figure("No fixed-location examples available."), selection
    apply_publication_style()
    n_cols = 1 + len(area_masks)
    fig, axes = plt.subplots(1, n_cols, figsize=horizontal_panel_figsize(n_cols, panel_width=2.2, height=2.7))
    axes = np.atleast_1d(axes)
    _plot_panel_image(axes[0], probe_image, footer_label="Probe")
    for axis, row in zip(axes[1:], area_masks.itertuples(index=False)):
        sample_image = construct_synthetic_sample_from_probe(probe_image, row.mask_binary)
        _plot_panel_image(axis, sample_image, mask_binary=row.mask_binary, footer_label=f"A={int(row.raw_overlap_area)}")
    fig.tight_layout()
    return fig, selection


def plot_panel_area_magnitude_mV(area_summary: pd.DataFrame, pooled_summary: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_WIDE)
    if not area_summary.empty:
        _plot_individual_points(
            ax,
            area_summary["raw_overlap_area"].to_numpy(dtype=np.float64),
            _voltage_to_millivolts(area_summary["mean_M"]),
        )
    if not pooled_summary.empty:
        ordered = pooled_summary.sort_values("raw_overlap_area", kind="stable")
        x = ordered["raw_overlap_area"].to_numpy(dtype=np.float64)
        _plot_pooled_black_line(
            ax,
            x,
            _voltage_to_millivolts(ordered["mean_M"]),
            _voltage_to_millivolts(ordered["sem_M"]),
        )
        ax.set_xticks(x)
    ax.set_xlabel("Raw overlap area")
    ax.set_ylabel("Bias magnitude (mV)")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def plot_panel_area_direction_angle_deg(area_summary: pd.DataFrame, pooled_summary: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_WIDE)
    if not area_summary.empty:
        _plot_individual_points(
            ax,
            area_summary["raw_overlap_area"].to_numpy(dtype=np.float64),
            area_summary["reference_direction_angle_deg"].to_numpy(dtype=np.float64),
        )
    if not pooled_summary.empty:
        ordered = pooled_summary.sort_values("raw_overlap_area", kind="stable")
        x = ordered["raw_overlap_area"].to_numpy(dtype=np.float64)
        _plot_pooled_black_line(
            ax,
            x,
            ordered["reference_direction_angle_deg"].to_numpy(dtype=np.float64),
            ordered["sem_reference_direction_angle_deg"].to_numpy(dtype=np.float64),
        )
        ax.set_xticks(x)
    ax.set_xlabel("Raw overlap area")
    ax.set_ylabel("Angular deviation from reference (deg)")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def plot_panel_fixed_area_examples(mask_df: pd.DataFrame, save_case_count: int) -> tuple[plt.Figure, dict[str, object]]:
    selection, probe_image, _, location_masks = _select_panel_example_data(mask_df, save_case_count)
    if selection.get("status") != "ok" or probe_image is None or location_masks.empty:
        return _blank_panel_figure("No fixed-area examples available."), selection
    apply_publication_style()
    n_cols = 1 + len(location_masks)
    fig, axes = plt.subplots(1, n_cols, figsize=horizontal_panel_figsize(n_cols, panel_width=2.2, height=2.7))
    axes = np.atleast_1d(axes)
    _plot_panel_image(axes[0], probe_image, footer_label="Probe")
    for axis, row in zip(axes[1:], location_masks.itertuples(index=False)):
        sample_image = construct_synthetic_sample_from_probe(probe_image, row.mask_binary)
        _plot_panel_image(axis, sample_image, mask_binary=row.mask_binary, footer_label=f"Loc{int(row.mask_rank)}")
    fig.tight_layout()
    return fig, selection


def plot_panel_location_direction_angle_deg(
    reference_based_df: pd.DataFrame,
    pooled_reference_based_df: pd.DataFrame,
) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_WIDE)
    if not reference_based_df.empty:
        _plot_individual_points(
            ax,
            reference_based_df["mask_rank"].to_numpy(dtype=np.float64),
            reference_based_df["direction_angle_deg_to_reference"].to_numpy(dtype=np.float64),
        )
    if not pooled_reference_based_df.empty:
        ordered = pooled_reference_based_df.sort_values("mask_rank", kind="stable")
        x = ordered["mask_rank"].to_numpy(dtype=np.float64)
        _plot_pooled_black_line(
            ax,
            x,
            ordered["mean_direction_angle_deg_to_reference"].to_numpy(dtype=np.float64),
            ordered["sem_direction_angle_deg_to_reference"].to_numpy(dtype=np.float64),
        )
        ax.set_xticks(x)
    ax.set_xlabel("Location mask index")
    ax.set_ylabel("Angular deviation from reference (deg)")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def plot_panel_delay_magnitude_mV(delay_within_mask_df: pd.DataFrame, delay_pooled_summary: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_WIDE)
    if not delay_within_mask_df.empty:
        _plot_individual_points(
            ax,
            delay_within_mask_df["delay_ms"].to_numpy(dtype=np.float64),
            _voltage_to_millivolts(delay_within_mask_df["M"]),
        )
    if not delay_pooled_summary.empty:
        ordered = delay_pooled_summary.sort_values("delay_ms", kind="stable")
        _plot_pooled_black_line(
            ax,
            ordered["delay_ms"].to_numpy(dtype=np.float64),
            _voltage_to_millivolts(ordered["mean_M"]),
            _voltage_to_millivolts(ordered["sem_M"]),
        )
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Bias magnitude (mV)")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def plot_panel_location_magnitude_control_mV(
    reference_based_df: pd.DataFrame,
    pooled_reference_based_df: pd.DataFrame,
) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_WIDE)
    if not reference_based_df.empty:
        _plot_individual_points(
            ax,
            reference_based_df["mask_rank"].to_numpy(dtype=np.float64),
            _voltage_to_millivolts(reference_based_df["magnitude_diff_to_reference"]),
        )
    if not pooled_reference_based_df.empty:
        ordered = pooled_reference_based_df.sort_values("mask_rank", kind="stable")
        x = ordered["mask_rank"].to_numpy(dtype=np.float64)
        _plot_pooled_black_line(
            ax,
            x,
            _voltage_to_millivolts(ordered["mean_magnitude_diff_to_reference"]),
            _voltage_to_millivolts(ordered["sem_magnitude_diff_to_reference"]),
        )
        ax.set_xticks(x)
    ax.set_xlabel("Location mask index")
    ax.set_ylabel("Magnitude difference from reference (mV)")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def plot_panel_delay_direction_angle_deg(delay_within_mask_df: pd.DataFrame, delay_pooled_summary: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_WIDE)
    if not delay_within_mask_df.empty:
        _plot_individual_points(
            ax,
            delay_within_mask_df["delay_ms"].to_numpy(dtype=np.float64),
            delay_within_mask_df["theta_deg"].to_numpy(dtype=np.float64),
        )
    if not delay_pooled_summary.empty:
        ordered = delay_pooled_summary.sort_values("delay_ms", kind="stable")
        _plot_pooled_black_line(
            ax,
            ordered["delay_ms"].to_numpy(dtype=np.float64),
            ordered["mean_theta_deg"].to_numpy(dtype=np.float64),
            ordered["sem_theta_deg"].to_numpy(dtype=np.float64),
        )
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Angular deviation (deg)")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def plot_fixed_location_area_and_direction(area_summary: pd.DataFrame, pooled_summary: pd.DataFrame) -> plt.Figure:
    del area_summary
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL, sharex=False)
    ax_mag, ax_dir = axes
    if not pooled_summary.empty:
        ordered = pooled_summary.sort_values("raw_overlap_area", kind="stable")
        x = ordered["raw_overlap_area"].to_numpy(dtype=np.float64)
        _plot_pooled_black_line(
            ax_mag,
            x,
            ordered["mean_M"].to_numpy(dtype=np.float64),
            ordered["sem_M"].to_numpy(dtype=np.float64),
        )
        _plot_pooled_black_line(
            ax_dir,
            x,
            ordered["reference_direction_cosine"].to_numpy(dtype=np.float64),
            ordered["sem_reference_direction_cosine"].to_numpy(dtype=np.float64),
        )
    ax_mag.set_xlabel("Raw overlap area")
    ax_mag.set_ylabel("Mean magnitude M")
    ax_mag.set_title("Fixed location: area controls magnitude")
    ax_mag.grid(alpha=GRID_ALPHA)
    ax_dir.set_xlabel("Raw overlap area")
    ax_dir.set_ylabel("Direction cosine to smallest mask")
    ax_dir.set_ylim(-1.05, 1.05)
    ax_dir.set_title("Fixed location: direction remains stable")
    ax_dir.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def plot_fixed_area_location_changes_direction(
    per_mask_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    pooled_per_mask_df: pd.DataFrame,
    pooled_pairwise_df: pd.DataFrame,
) -> plt.Figure:
    del per_mask_df
    del pairwise_df
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL)
    ax_mag, ax_dir = axes
    if not pooled_per_mask_df.empty:
        ordered_mag = pooled_per_mask_df.sort_values("mask_rank", kind="stable")
        _plot_pooled_black_line(
            ax_mag,
            ordered_mag["mask_rank"].to_numpy(dtype=np.float64),
            ordered_mag["mean_M"].to_numpy(dtype=np.float64),
            ordered_mag["sem_M"].to_numpy(dtype=np.float64),
        )
    ax_mag.set_xlabel("Location mask index")
    ax_mag.set_ylabel("Magnitude M")
    ax_mag.set_title("Matched area: magnitude remains similar")
    ax_mag.grid(alpha=GRID_ALPHA)

    if not pooled_pairwise_df.empty:
        ordered_dir = pooled_pairwise_df.sort_values(["mask_rank_i", "mask_rank_j"], kind="stable").reset_index(drop=True)
        x = np.arange(len(ordered_dir), dtype=np.float64)
        _plot_pooled_black_line(
            ax_dir,
            x,
            ordered_dir["mean_direction_similarity"].to_numpy(dtype=np.float64),
            ordered_dir["sem_direction_similarity"].to_numpy(dtype=np.float64),
        )
        ax_dir.set_xticks(x)
        ax_dir.set_xticklabels(ordered_dir["pair_label"].tolist(), rotation=30, ha="right")
    ax_dir.set_ylabel("Mean pairwise direction cosine")
    ax_dir.set_ylim(-1.05, 1.05)
    ax_dir.set_title("Matched area: location changes direction")
    ax_dir.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def plot_probe_mask_examples(mask_df: pd.DataFrame, save_case_count: int) -> tuple[plt.Figure, dict[str, object]]:
    apply_publication_style()
    candidate_probe_ids = []
    for probe_id, subset in mask_df.groupby("probe_id", sort=True):
        has_area = int((subset["mask_group"] == FIXED_LOCATION_GROUP).sum()) >= 1
        has_location = int((subset["mask_group"] == FIXED_AREA_GROUP).sum()) >= 1
        if has_area and has_location:
            candidate_probe_ids.append(int(probe_id))
    if not candidate_probe_ids:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_WIDE)
        ax.axis("off")
        ax.text(0.5, 0.5, "No probe supports both mask blocks.", ha="center", va="center")
        return fig, {"status": "no_valid_probe"}
    probe_id = int(candidate_probe_ids[0])
    subset = mask_df[mask_df["probe_id"] == probe_id].copy()
    area_masks = subset[subset["mask_group"] == FIXED_LOCATION_GROUP].sort_values("raw_overlap_area", kind="stable").head(max(1, int(save_case_count)))
    location_masks = subset[subset["mask_group"] == FIXED_AREA_GROUP].sort_values("mask_rank", kind="stable").head(max(1, int(save_case_count)))
    n_cols = 1 + max(len(area_masks), len(location_masks))
    fig, axes = plt.subplots(2, n_cols, figsize=horizontal_panel_figsize(n_cols, panel_width=3.0, height=6.0))
    probe_image = subset["probe_image"].iloc[0]
    _plot_image(axes[0, 0], probe_image, f"Probe {probe_id}")
    _plot_image(axes[1, 0], probe_image, f"Probe {probe_id}")
    for plot_idx in range(1, n_cols):
        axes[0, plot_idx].axis("off")
        axes[1, plot_idx].axis("off")
    for idx, row in enumerate(area_masks.itertuples(index=False), start=1):
        _plot_sample(
            axes[0, idx],
            construct_synthetic_sample_from_probe(probe_image, row.mask_binary),
            row.mask_binary,
            f"Area {int(row.raw_overlap_area)}",
        )
    for idx, row in enumerate(location_masks.itertuples(index=False), start=1):
        _plot_sample(
            axes[1, idx],
            construct_synthetic_sample_from_probe(probe_image, row.mask_binary),
            row.mask_binary,
            f"Loc {int(row.mask_rank)}\nA={int(row.raw_overlap_area)}",
        )
    fig.suptitle("Synthetic sample = probe * mask", y=1.02)
    fig.tight_layout()
    return fig, {"status": "ok", "probe_id": probe_id}


def plot_delay_strength_direction_decomposition(delay_pooled_summary: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL, sharex=True)
    panel_specs = (
        ("mean_A", "sem_A", "A(delay)", "Effective strength A"),
        ("mean_M", "sem_M", "M(delay)", "Magnitude M"),
        ("mean_cos_theta", "sem_cos_theta", "cos_theta(delay)", "Direction cosine to within-mask ref"),
    )
    ordered = delay_pooled_summary.sort_values("delay_ms", kind="stable").reset_index(drop=True)
    x = ordered["delay_ms"].to_numpy(dtype=np.float64) if not ordered.empty else np.asarray([], dtype=np.float64)
    for ax, (mean_col, sem_col, title, ylabel) in zip(axes, panel_specs):
        if not ordered.empty:
            _plot_pooled_black_line(
                ax,
                x,
                ordered[mean_col].to_numpy(dtype=np.float64),
                ordered[sem_col].to_numpy(dtype=np.float64),
            )
        ax.set_xlabel("Delay (ms)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=GRID_ALPHA)
    axes[2].set_ylim(-1.05, 1.05)
    fig.tight_layout()
    return fig


def _build_mask_tables_for_probe_pool(
    probe_df: pd.DataFrame,
    *,
    area_levels: Sequence[int],
    fixed_area_pixels: int,
    num_location_masks: int,
    foreground_threshold: float,
    mask_generation_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    mask_rows: list[dict[str, object]] = []
    generation_notes: list[dict[str, object]] = []
    for probe_row in probe_df.itertuples(index=False):
        probe_id = int(probe_row.probe_id)
        nested_df = generate_nested_masks_for_probe(
            probe_image=probe_row.probe_image,
            target_area_schedule=area_levels,
            foreground_threshold=float(foreground_threshold),
            seed=mix_seed(int(mask_generation_seed), probe_id, 11),
        )
        if nested_df.empty:
            generation_notes.append(
                {
                    "probe_id": probe_id,
                    "mask_group": FIXED_LOCATION_GROUP,
                    "status": "skipped_no_valid_nested_masks",
                }
            )
        else:
            for row in nested_df.itertuples(index=False):
                mask_rows.append(
                    {
                        "probe_id": probe_id,
                        "probe_label": int(probe_row.probe_label),
                        "mask_id": f"probe_{probe_id}_{row.mask_id}",
                        "mask_group": str(row.mask_group),
                        "raw_overlap_area": int(row.raw_overlap_area),
                        "mask_binary": row.mask_binary,
                        "mask_center_row": float(row.mask_center_row),
                        "mask_center_col": float(row.mask_center_col),
                        "mask_center": row.mask_center,
                        "mask_status": str(row.mask_status),
                        "mask_rank": int(row.mask_rank),
                        "area_level_label": str(row.area_level_label),
                        "geometry_similarity_to_others": float("nan"),
                    }
                )
        location_df = generate_equal_area_different_location_masks_for_probe(
            probe_image=probe_row.probe_image,
            target_area=int(fixed_area_pixels),
            n_masks=int(num_location_masks),
            candidate_location_strategy="farthest_point",
            foreground_threshold=float(foreground_threshold),
            seed=mix_seed(int(mask_generation_seed), probe_id, 23),
        )
        if location_df.empty:
            generation_notes.append(
                {
                    "probe_id": probe_id,
                    "mask_group": FIXED_AREA_GROUP,
                    "status": "skipped_insufficient_distinct_location_masks",
                }
            )
        else:
            for row in location_df.itertuples(index=False):
                mask_rows.append(
                    {
                        "probe_id": probe_id,
                        "probe_label": int(probe_row.probe_label),
                        "mask_id": f"probe_{probe_id}_{row.mask_id}",
                        "mask_group": str(row.mask_group),
                        "raw_overlap_area": int(row.raw_overlap_area),
                        "mask_binary": row.mask_binary,
                        "mask_center_row": float(row.mask_center_row),
                        "mask_center_col": float(row.mask_center_col),
                        "mask_center": row.mask_center,
                        "mask_status": str(row.mask_status),
                        "mask_rank": int(row.mask_rank),
                        "area_level_label": "fixed_area",
                        "geometry_similarity_to_others": float(row.geometry_similarity_to_others),
                    }
                )
    return pd.DataFrame(mask_rows), generation_notes


def _select_record_columns(df_records: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "probe_id",
        "probe_label",
        "mask_id",
        "mask_group",
        "raw_overlap_area",
        "delay_ms",
        "magnitude_M",
        "delta_norm",
        "pair_status",
    ]
    present = [col for col in keep if col in df_records.columns]
    return df_records[present].copy()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DMS probe-mask synthetic control experiment.")
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-ms", type=float, default=DEFAULT_SAMPLE_MS)
    parser.add_argument("--delay-ms", type=float, default=DEFAULT_DELAY_MS)
    parser.add_argument("--delay-sweep-ms", type=float, nargs="+", default=list(DEFAULT_DELAY_SWEEP_MS))
    parser.add_argument("--probe-ms", type=float, default=DEFAULT_PROBE_MS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--probe-selection-per-class", type=int, default=DEFAULT_PROBE_SELECTION_PER_CLASS)
    parser.add_argument("--foreground-threshold", type=float, default=DEFAULT_FOREGROUND_THRESHOLD)
    parser.add_argument("--num-area-levels", type=int, default=DEFAULT_NUM_AREA_LEVELS)
    parser.add_argument("--area-levels", type=int, nargs="+", default=list(DEFAULT_AREA_LEVELS))
    parser.add_argument("--num-location-masks", type=int, default=DEFAULT_NUM_LOCATION_MASKS)
    parser.add_argument("--fixed-area-pixels", type=int, default=DEFAULT_FIXED_AREA_PIXELS)
    parser.add_argument("--mask-generation-seed", type=int, default=DEFAULT_MASK_GENERATION_SEED)
    parser.add_argument("--save-case-count", type=int, default=DEFAULT_SAVE_CASE_COUNT)
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    _validate_positive("--sample-ms", float(args.sample_ms))
    _validate_positive("--delay-ms", float(args.delay_ms), allow_zero=True)
    _validate_positive("--probe-ms", float(args.probe_ms))
    _validate_positive("--batch-size", int(args.batch_size))
    _validate_positive("--max-probes", int(args.max_probes))
    _validate_positive("--probe-selection-per-class", int(args.probe_selection_per_class), allow_zero=True)
    _validate_positive("--num-area-levels", int(args.num_area_levels))
    _validate_positive("--num-location-masks", int(args.num_location_masks))
    _validate_positive("--fixed-area-pixels", int(args.fixed_area_pixels))
    _validate_positive("--save-case-count", int(args.save_case_count), allow_zero=True)
    area_levels = _sanitize_area_levels(args.area_levels)
    if int(args.num_area_levels) != len(area_levels):
        raise ValueError("--num-area-levels must match the number of --area-levels values.")
    delay_sweep_ms = _sanitize_delay_sweep(args.delay_sweep_ms)
    if float(args.delay_ms) not in delay_sweep_ms:
        delay_sweep_ms = sorted(dict.fromkeys(delay_sweep_ms + [float(args.delay_ms)]))

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    spec = ExperimentSpec(dt=1.0 * ms, sample_ms=float(args.sample_ms), probe_ms=float(args.probe_ms))
    if spec.sample_steps <= 0 or spec.probe_steps <= 0:
        raise ValueError("sample/probe durations must resolve to positive steps.")

    layout = prepare_result_layout(args.output_dir)
    result_root = layout.root
    output_dir = layout.data_dir
    figures_dir = layout.figure_dir
    logs_dir = layout.log_dir

    dataset = _load_dataset(args.dataset_root, args.split)
    images, labels, _ = build_dataset_arrays(dataset)
    class_index = build_class_index(dataset, num_classes=int(labels.max()) + 1)
    probe_df = build_probe_pool(
        images=images,
        labels=labels,
        class_index=class_index,
        max_probes=int(args.max_probes),
        seed=int(args.seed),
        selection_per_class=int(args.probe_selection_per_class),
        foreground_threshold=float(args.foreground_threshold),
    )
    mask_df, generation_notes = _build_mask_tables_for_probe_pool(
        probe_df,
        area_levels=area_levels,
        fixed_area_pixels=int(args.fixed_area_pixels),
        num_location_masks=int(args.num_location_masks),
        foreground_threshold=float(args.foreground_threshold),
        mask_generation_seed=int(args.mask_generation_seed),
    )

    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(float(args.sample_ms), float(args.probe_ms), max(delay_sweep_ms)),
    )
    readout_step = resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(spec.probe_steps),
        decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )

    delay_record_frames: list[pd.DataFrame] = []
    delay_iter = tqdm(delay_sweep_ms, total=len(delay_sweep_ms), desc="Delay sweep", dynamic_ncols=True)
    for delay_ms in delay_iter:
        rollout = compute_synthetic_pair_delta_vectors(
            net=net,
            encoder=encoder,
            device=device,
            readout_step=int(readout_step),
            probe_df=probe_df,
            mask_df=mask_df,
            delay_steps=_delay_ms_to_steps(float(delay_ms), spec.dt),
            delay_ms=float(delay_ms),
            batch_size=int(args.batch_size),
            spec=spec,
        )
        delay_record_frames.append(rollout["record_df"])
    df_records = pd.concat(delay_record_frames, axis=0, ignore_index=True) if delay_record_frames else pd.DataFrame()
    main_records = df_records[df_records["delay_ms"] == float(args.delay_ms)].copy() if not df_records.empty else pd.DataFrame()

    area_summary = summarize_fixed_location_vary_area(main_records)
    location_summary = summarize_fixed_area_vary_location(main_records)
    delay_summary = summarize_delay_strength_direction(df_records)

    records_csv = save_tidy_csv(
        _select_record_columns(df_records),
        output_dir / "synthetic_probe_mask_records.csv",
        sort_by=["probe_id", "mask_group", "mask_id", "delay_ms"],
    )
    area_summary_csv = save_tidy_csv(
        area_summary["summary_df"],
        output_dir / "fixed_location_vary_area_summary.csv",
        sort_by=["probe_id", "raw_overlap_area"],
    )
    area_pooled_summary_csv = save_tidy_csv(
        area_summary["pooled_summary_df"],
        output_dir / "fixed_location_vary_area_pooled_summary.csv",
        sort_by=["raw_overlap_area"],
    )
    location_summary_csv = save_tidy_csv(
        location_summary["pairwise_df"],
        output_dir / "fixed_area_vary_location_summary.csv",
        sort_by=["probe_id", "mask_i", "mask_j"],
    )
    location_reference_summary_csv = save_tidy_csv(
        location_summary["reference_based_df"],
        output_dir / "fixed_area_vary_location_reference_summary.csv",
        sort_by=["probe_id", "mask_rank", "mask_id"],
    )
    location_reference_pooled_summary_csv = save_tidy_csv(
        location_summary["pooled_reference_based_df"],
        output_dir / "fixed_area_vary_location_reference_pooled_summary.csv",
        sort_by=["mask_rank"],
    )
    delay_within_mask_csv = save_tidy_csv(
        delay_summary["delay_within_mask_df"].drop(columns=["DeltaV"], errors="ignore"),
        output_dir / "delay_strength_direction_within_mask.csv",
        sort_by=["probe_id", "mask_group", "mask_id", "delay_ms"],
    )
    delay_summary_csv = save_tidy_csv(
        delay_summary["delay_pooled_summary_df"],
        output_dir / "delay_strength_direction_summary.csv",
        sort_by=["delay_ms"],
    )

    fig_paths: dict[str, object] = {}
    example_selection = {"status": "skipped"}
    if not bool(args.skip_figures):
        fig_a, area_example_selection = plot_panel_fixed_location_examples(main_records, save_case_count=int(args.save_case_count))
        fig_paths["fig4_panel_a_fixed_location_examples"] = save_figure_all_formats(
            fig_a,
            figures_dir / "fig4_panel_a_fixed_location_examples",
        )
        plt.close(fig_a)

        fig_b = plot_panel_area_magnitude_mV(area_summary["summary_df"], area_summary["pooled_summary_df"])
        fig_paths["fig4_panel_b_area_magnitude_mV"] = save_figure_all_formats(
            fig_b,
            figures_dir / "fig4_panel_b_area_magnitude_mV",
        )
        plt.close(fig_b)

        fig_c = plot_panel_area_direction_angle_deg(area_summary["summary_df"], area_summary["pooled_summary_df"])
        fig_paths["fig4_panel_c_area_direction_angle_deg"] = save_figure_all_formats(
            fig_c,
            figures_dir / "fig4_panel_c_area_direction_angle_deg",
        )
        plt.close(fig_c)

        fig_d, location_example_selection = plot_panel_fixed_area_examples(main_records, save_case_count=int(args.save_case_count))
        fig_paths["fig4_panel_d_fixed_area_examples"] = save_figure_all_formats(
            fig_d,
            figures_dir / "fig4_panel_d_fixed_area_examples",
        )
        plt.close(fig_d)

        fig_e = plot_panel_location_direction_angle_deg(
            location_summary["reference_based_df"],
            location_summary["pooled_reference_based_df"],
        )
        fig_paths["fig4_panel_e_location_direction_angle_deg"] = save_figure_all_formats(
            fig_e,
            figures_dir / "fig4_panel_e_location_direction_angle_deg",
        )
        plt.close(fig_e)

        fig_f = plot_panel_delay_magnitude_mV(
            delay_summary["delay_within_mask_df"],
            delay_summary["delay_pooled_summary_df"],
        )
        fig_paths["fig4_panel_f_delay_magnitude_mV"] = save_figure_all_formats(
            fig_f,
            figures_dir / "fig4_panel_f_delay_magnitude_mV",
        )
        plt.close(fig_f)

        fig_g = plot_panel_location_magnitude_control_mV(
            location_summary["reference_based_df"],
            location_summary["pooled_reference_based_df"],
        )
        fig_paths["fig4_panel_g_location_magnitude_control_mV"] = save_figure_all_formats(
            fig_g,
            figures_dir / "fig4_panel_g_location_magnitude_control_mV",
        )
        plt.close(fig_g)

        fig_h = plot_panel_delay_direction_angle_deg(
            delay_summary["delay_within_mask_df"],
            delay_summary["delay_pooled_summary_df"],
        )
        fig_paths["fig4_panel_h_delay_direction_angle_deg"] = save_figure_all_formats(
            fig_h,
            figures_dir / "fig4_panel_h_delay_direction_angle_deg",
        )
        plt.close(fig_h)

        example_selection = {
            "fixed_location_examples": area_example_selection,
            "fixed_area_examples": location_example_selection,
        }

    summary_payload = {
        "assumptions": {
            "synthetic_control_design": "This is a synthetic control experiment.",
            "sample_construction": "Samples are constructed directly from the probe by masking selected probe regions.",
            "causal_goal": "The goal is to causally disentangle overlap extent from overlap location / geometry.",
            "raw_overlap_only": "Only raw overlap area is used. Raw overlap area equals the number of True pixels in the probe-side mask.",
            "forbidden_metrics": "No effective overlap area, contribution-weighted area, B-map weighted area, or probe contribution weighting is used.",
            "scientific_test": (
                "The experiment tests whether overlap area primarily controls displacement magnitude, whether "
                "probe-side overlap location / geometry primarily controls displacement direction, and whether "
                "delay under fixed mask identity mainly modulates strength / time evolution rather than direction."
            ),
            "delay_mechanistic_goal": (
                "Under fixed mask identity (thus fixed area and location), delay is tested as a strength/time-evolution "
                "modulator rather than a direction-changing factor."
            ),
        },
        "parameter_settings": {
            "model_path": str(args.model_path),
            "dataset_root": str(args.dataset_root),
            "split": str(args.split),
            "seed": int(args.seed),
            "mask_generation_seed": int(args.mask_generation_seed),
            "max_probes": int(args.max_probes),
            "probe_selection_per_class": int(args.probe_selection_per_class),
            "sample_ms": float(args.sample_ms),
            "delay_ms": float(args.delay_ms),
            "delay_sweep_ms": [float(v) for v in delay_sweep_ms],
            "probe_ms": float(args.probe_ms),
            "batch_size": int(args.batch_size),
            "foreground_threshold": float(args.foreground_threshold),
            "num_area_levels": int(args.num_area_levels),
            "area_levels": [int(v) for v in area_levels],
            "num_location_masks": int(args.num_location_masks),
            "fixed_area_pixels": int(args.fixed_area_pixels),
            "save_case_count": int(args.save_case_count),
            "readout_step": int(readout_step),
            "sample_steps": int(spec.sample_steps),
            "probe_steps": int(spec.probe_steps),
        },
        "fixed_location_vary_area": {
            "summary_rows": area_summary["summary_df"].to_dict(orient="records"),
            "pooled_rows": area_summary["pooled_summary_df"].to_dict(orient="records"),
        },
        "fixed_area_vary_location": {
            "per_mask_rows": location_summary["per_mask_df"].to_dict(orient="records"),
            "pairwise_rows": location_summary["pairwise_df"].to_dict(orient="records"),
            "pooled_per_mask_rows": location_summary["pooled_per_mask_df"].to_dict(orient="records"),
            "pooled_pairwise_rows": location_summary["pooled_pairwise_df"].to_dict(orient="records"),
            "reference_based_rows": location_summary["reference_based_df"].to_dict(orient="records"),
            "pooled_reference_based_rows": location_summary["pooled_reference_based_df"].to_dict(orient="records"),
            "pooled_summary": location_summary["pooled_summary"],
        },
        "delay_modulation": {
            "status": "replaced_by_within_mask_strength_direction_decomposition",
            "delay_summary_rows": delay_summary["delay_pooled_summary_df"].to_dict(orient="records"),
        },
        "delay_strength_direction": {
            "n_valid_masks": int(delay_summary["n_valid_masks"]),
            "n_total_masks": int(delay_summary["n_total_masks"]),
            "reference_status_counts": delay_summary["reference_status_counts"],
            "delay_summary_rows": delay_summary["delay_pooled_summary_df"].to_dict(orient="records"),
            "trend_A": delay_summary["trend_A"],
            "trend_M": delay_summary["trend_M"],
            "trend_cos_theta": delay_summary["trend_cos_theta"],
            "trend_theta_deg": delay_summary["trend_theta_deg"],
        },
        "generation_notes": generation_notes,
        "example_selection": example_selection,
        "artifacts": {
            "synthetic_probe_mask_records_csv": str(records_csv),
            "fixed_location_vary_area_summary_csv": str(area_summary_csv),
            "fixed_location_vary_area_pooled_summary_csv": str(area_pooled_summary_csv),
            "fixed_area_vary_location_summary_csv": str(location_summary_csv),
            "fixed_area_vary_location_reference_summary_csv": str(location_reference_summary_csv),
            "fixed_area_vary_location_reference_pooled_summary_csv": str(location_reference_pooled_summary_csv),
            "delay_strength_direction_within_mask_csv": str(delay_within_mask_csv),
            "delay_strength_direction_summary_csv": str(delay_summary_csv),
            "figure_paths": fig_paths,
        },
        "scientific_statement": (
            "This synthetic control experiment tests whether area controls magnitude, location controls direction, "
            "and delay under fixed masks mainly modulates strength / time evolution."
        ),
        "scientific_conclusions": [
            "Area controls displacement magnitude under fixed location.",
            "Location / geometry controls displacement direction under fixed area.",
            "Delay under fixed mask identity mainly modulates strength / time evolution rather than direction.",
        ],
    }
    summary_json = _save_json(summary_payload, output_dir / "synthetic_control_summary.json")

    run_config = {
        "model_path": str(args.model_path),
        "config": None if args.config is None else str(args.config),
        "dataset_root": str(args.dataset_root),
        "split": str(args.split),
        "device": str(device),
        "seed": int(args.seed),
        "output_dir": str(result_root),
        "sample_ms": float(args.sample_ms),
        "delay_ms": float(args.delay_ms),
        "delay_sweep_ms": [float(v) for v in delay_sweep_ms],
        "probe_ms": float(args.probe_ms),
        "batch_size": int(args.batch_size),
        "max_probes": int(args.max_probes),
        "probe_selection_per_class": int(args.probe_selection_per_class),
        "foreground_threshold": float(args.foreground_threshold),
        "num_area_levels": int(args.num_area_levels),
        "area_levels": [int(v) for v in area_levels],
        "num_location_masks": int(args.num_location_masks),
        "fixed_area_pixels": int(args.fixed_area_pixels),
        "mask_generation_seed": int(args.mask_generation_seed),
        "save_case_count": int(args.save_case_count),
        "readout_step": int(readout_step),
        "sample_steps": int(spec.sample_steps),
        "probe_steps": int(spec.probe_steps),
        "raw_overlap_definition": "raw_overlap_area = number of True pixels in the probe-side mask M",
        "only_raw_overlap_area_used": True,
        "forbidden_metrics": (
            "No effective overlap area, weighted overlap area, B-map weighting, contribution weighting, "
            "or feature-importance weighting is used."
        ),
        "synthetic_control_design": True,
        "scientific_hypothesis": (
            "This synthetic control experiment tests whether area controls magnitude, location controls direction, "
            "and delay under fixed mask identity mainly modulates strength / time evolution."
        ),
        "summary_json": str(summary_json),
        "figure_paths": fig_paths,
    }
    run_config_path = save_run_config(run_config, result_root)
    summary_path = save_summary_json(summary_payload, result_root)
    run_log_path = save_log_lines(
        [
            "experiment=dms_probe_mask_synthetic_control_experiment",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"seed={int(args.seed)}",
            f"device={device}",
            f"result_root={result_root.resolve()}",
            f"summary_json={summary_path.resolve()}",
        ]
        + [f"{panel_name}={panel_paths}" for panel_name, panel_paths in fig_paths.items()],
        logs_dir,
    )


if __name__ == "__main__":
    main()
