from __future__ import annotations

"""
Unified three-level DMS synthetic-control experiment.

This experiment no longer treats dose, area, location, and delay as
parallel support properties. Instead, it first establishes an
importance-specific local dose law, then interprets location and area as
compositions of local support effects, and finally tests whether
delay-dependent working-memory phenomena can be explained as the temporal
evolution of these local STSP dose effects.

This experiment is organized in three levels. First, it establishes
importance-specific local STSP dose laws under fixed patch geometry.
Second, it explains location and area effects as compositions of these
local dose effects. Third, it tests whether delay-dependent working-memory
phenomena can be explained as the temporal evolution of local STSP dose,
projected through the same local dose laws into decision space.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.common.dataset import build_class_index, build_dataset_arrays, encode_images
from src.experiments.common.input_masks import foreground_mask_from_image
from src.experiments.common.importance_support import (
    assign_importance_tiers,
    build_atomic_tier_patches,
    compute_latency_importance_payload,
    summarize_support_composition,
)
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import run_dms_snapshot_rollout
from src.experiments.common.pattern_metrics import extract_grouped_voltage_vector
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.support_bias import (
    annotate_signed_direction_records,
    build_signed_direction_reference,
    compute_signed_direction_deg,
    generate_nested_support_masks,
)
from src.experiments.common.voltage_readout import resolve_readout_step
from src.plotting.common.io import apply_publication_style, save_figure_all_formats, save_run_config, save_tidy_csv

MS = 1e-3

EXPERIMENT_NAME = "dms_probe_mask_synthetic_control_experiment"
DEFAULT_MODEL_PATH = "results/sdnn_deep_final/net_final.pth"
DEFAULT_OUTPUT_DIR = "results/dms_probe_mask_synthetic_control_fig5_ux_support_properties"
DEFAULT_DATASET_ROOT = "./MNIST"
DEFAULT_SAMPLE_MS = 200.0
DEFAULT_DELAY_MS = 400.0
DEFAULT_DELAY_SWEEP_MS: tuple[float, ...] = (100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 1000.0, 1500.0)
DEFAULT_PROBE_MS = 100.0
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_PROBES = 100
DEFAULT_PROBE_SELECTION_PER_CLASS = 10
DEFAULT_FOREGROUND_THRESHOLD = 0.0
DEFAULT_FIXED_SUPPORT_AREA_PIXELS = 24
DEFAULT_SUPPORT_AREA_LEVELS: tuple[int, ...] = (12, 18, 24, 30, 36, 48)
DEFAULT_NUM_LOCATION_MASKS = 10
DEFAULT_MASK_GENERATION_SEED = 123
DEFAULT_SAVE_CASE_COUNT = 4
DEFAULT_MAX_LOCATION_MASK_IOU = 0.65
DEFAULT_UX_DOSE_TARGETS: tuple[float, ...] = (
    0.20,
    0.225,
    0.25,
    0.26,
    0.27,
    0.28,
    0.29,
    0.30,
    0.31,
    0.32,
    0.33,
    0.34,
    0.35,
    0.375,
    0.40,
)
DEFAULT_UX_TARGET_SOLVE_MAX_SCALE = 3.0
DEFAULT_UX_TARGET_SOLVE_COARSE_STEPS = 121
DEFAULT_UX_TARGET_SOLVE_REFINE_STEPS = 17
DEFAULT_UX_TARGET_SOLVE_REFINE_ROUNDS = 3
DEFAULT_BRIDGE_MIN_DOSE_POINTS = 3
DEFAULT_BRIDGE_MIN_DELAY_POINTS = 3
DEFAULT_IMPORTANCE_TIERS = 5
DEFAULT_ATOMIC_PATCH_AREA = 8
DEFAULT_IMPORTANCE_SMOOTHING_ALPHA = 0.10
DEFAULT_IMPORTANCE_SMOOTHING_PASSES = 1

SUPPORT_ROLE_CANONICAL = "canonical_support"
SUPPORT_ROLE_AREA = "area_support"
SUPPORT_ROLE_LOCATION = "location_support"

PANEL_FILENAMES = {
    "panel_a": "figX_panel_a_importance_map_and_atomic_patches",
    "panel_b": "figX_panel_b_dose_direction_by_importance_tier",
    "panel_c": "figX_panel_c_dose_magnitude_by_importance_tier",
    "panel_d": "figX_panel_d_dose_law_fit_summary",
    "panel_e": "figX_panel_e_location_support_composition",
    "panel_f": "figX_panel_f_location_observed_vs_predicted_direction",
    "panel_g": "figX_panel_g_location_observed_vs_predicted_magnitude",
    "panel_h": "figX_panel_h_area_support_composition",
    "panel_i": "figX_panel_i_area_observed_vs_predicted_direction",
    "panel_j": "figX_panel_j_area_observed_vs_predicted_magnitude",
    "panel_k": "figX_panel_k_delay_local_dose_evolution",
    "panel_l": "figX_panel_l_delay_actual_vs_predicted_direction",
    "panel_m": "figX_panel_m_delay_actual_vs_predicted_magnitude",
}

COLOR_SUPPORT = "#D1495B"
COLOR_MAGNITUDE = "#243B53"
COLOR_DIRECTION = "#C05621"
COLOR_DIRECTION_PREDICTED = "#9B2C2C"
COLOR_LOCATION = ("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E")
COLOR_TIER = ("#355070", "#6D597A", "#B56576", "#E56B6F", "#EAAC8B")
GRID_ALPHA = 0.25


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    probe_ms: float
    phase_reset: bool = True

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * MS) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * MS) / self.dt))


def _validate_positive(name: str, value: int | float, *, allow_zero: bool = False) -> None:
    scalar = float(value)
    if allow_zero:
        if scalar < 0.0:
            raise ValueError(f"{name} must be non-negative.")
        return
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _sanitize_delay_sweep(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("--delay-sweep-ms must contain at least one value.")
    delays = sorted(dict.fromkeys(float(v) for v in values))
    if any(delay < 0.0 for delay in delays):
        raise ValueError("--delay-sweep-ms values must be non-negative.")
    return delays


def _sanitize_area_levels(values: Sequence[int]) -> list[int]:
    if not values:
        raise ValueError("--support-area-levels must contain at least one value.")
    levels = sorted(dict.fromkeys(int(v) for v in values))
    if any(level <= 0 for level in levels):
        raise ValueError("--support-area-levels values must be positive.")
    if len(levels) < 3:
        raise ValueError("At least three support area conditions are required.")
    return levels


def _format_target_mean_ux_label(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _sanitize_target_mean_ux_levels(values: Sequence[float]) -> list[dict[str, object]]:
    if not values:
        raise ValueError("--ux-dose-targets must contain at least one value.")
    targets = sorted(dict.fromkeys(float(v) for v in values))
    if len(targets) < 3:
        raise ValueError("At least three ux target conditions are required.")
    if any((not np.isfinite(v)) or v <= 0.0 for v in targets):
        raise ValueError("--ux-dose-targets values must be finite and positive.")
    levels: list[dict[str, object]] = []
    for order, target in enumerate(targets, start=1):
        levels.append(
            {
                "dose_label": _format_target_mean_ux_label(target),
                "dose_order": int(order),
                "target_mean_ux_on_support": float(target),
            }
        )
    return levels


def _delay_ms_to_steps(delay_ms: float, dt: float) -> int:
    return int(round((float(delay_ms) * MS) / float(dt)))


def _sem(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _mask_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(mask, dtype=bool))


def _mask_center(mask: np.ndarray) -> tuple[float, float]:
    coords = _mask_coords(mask)
    if coords.size <= 0:
        return float("nan"), float("nan")
    center = coords.mean(axis=0, dtype=np.float64)
    return float(center[0]), float(center[1])


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = a | b
    if not union.any():
        return 0.0
    return float((a & b).sum() / union.sum())


def _make_mask_from_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if coords.size > 0:
        out[coords[:, 0], coords[:, 1]] = True
    return out


def _find_connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask_bool = np.asarray(mask, dtype=bool)
    coords = _mask_coords(mask_bool)
    if coords.size <= 0:
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
    if component_coords.size <= 0:
        raise ValueError("Cannot choose a seed from an empty component.")
    centroid = component_coords.mean(axis=0, dtype=np.float64)
    distances = np.sum((component_coords.astype(np.float64) - centroid[None, :]) ** 2, axis=1)
    seed_idx = int(np.argmin(distances))
    return int(component_coords[seed_idx, 0]), int(component_coords[seed_idx, 1])


def _sorted_coords_from_seed(component_coords: np.ndarray, seed_row: int, seed_col: int) -> np.ndarray:
    if component_coords.size <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    distances = np.sum((component_coords - np.asarray([seed_row, seed_col], dtype=np.int64)[None, :]) ** 2, axis=1)
    order = np.lexsort((component_coords[:, 1], component_coords[:, 0], distances))
    return component_coords[order]


def _farthest_point_sample(coords: np.ndarray, n_points: int) -> np.ndarray:
    if coords.size <= 0 or int(n_points) <= 0:
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


def _foreground_mask(image: torch.Tensor, threshold: float) -> np.ndarray:
    return foreground_mask_from_image(image, threshold=threshold)


def _center_voltage(v: np.ndarray | torch.Tensor) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return arr - arr.mean(axis=-1, keepdims=True)


def compute_delta_v(v_dynamic: np.ndarray | torch.Tensor, v_static: np.ndarray | torch.Tensor) -> np.ndarray:
    return _center_voltage(v_dynamic) - _center_voltage(v_static)


def _mean_gain_map_from_boundary(layer_boundary_state: Mapping[str, torch.Tensor], batch_idx: int) -> np.ndarray:
    u = np.asarray(layer_boundary_state["u"][batch_idx], dtype=np.float64)
    x = np.asarray(layer_boundary_state["x"][batch_idx], dtype=np.float64)
    return (u * x).mean(axis=0)


def _summarize_support_metrics_from_boundary(
    layer_boundary_state: Mapping[str, torch.Tensor],
    batch_idx: int,
    support_mask: np.ndarray,
) -> dict[str, float]:
    mask_bool = np.asarray(support_mask, dtype=bool)
    gain_map = _mean_gain_map_from_boundary(layer_boundary_state, batch_idx=batch_idx)
    area = int(mask_bool.sum())
    if area <= 0:
        return {
            "support_area": 0.0,
            "mean_ux_on_support": float("nan"),
            "total_ux_support": 0.0,
        }
    values = gain_map[mask_bool]
    return {
        "support_area": float(area),
        "mean_ux_on_support": float(np.mean(values)),
        "total_ux_support": float(np.sum(values)),
    }


def _balanced_probe_ids(class_index: Mapping[int, Sequence[int]], max_probes: int, seed: int) -> list[int]:
    rng = np.random.default_rng(int(seed))
    per_class: dict[int, list[int]] = {}
    for class_label in sorted(class_index):
        ids = np.asarray([int(idx) for idx in class_index[int(class_label)]], dtype=np.int64)
        per_class[int(class_label)] = rng.permutation(ids).tolist()
    selected: list[int] = []
    while len(selected) < int(max_probes):
        made_progress = False
        for class_label in sorted(per_class):
            if not per_class[class_label]:
                continue
            selected.append(int(per_class[class_label].pop(0)))
            made_progress = True
            if len(selected) >= int(max_probes):
                break
        if not made_progress:
            break
    return selected


def build_probe_pool(
    images: torch.Tensor,
    labels: np.ndarray,
    class_index: Mapping[int, Sequence[int]],
    *,
    max_probes: int,
    seed: int,
    selection_per_class: int,
    foreground_threshold: float,
) -> pd.DataFrame:
    target_max = int(max_probes)
    if int(selection_per_class) > 0:
        target_max = min(target_max, int(selection_per_class) * len(class_index))
    probe_ids = _balanced_probe_ids(class_index=class_index, max_probes=target_max, seed=int(seed))
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
        raise RuntimeError("No probes were selected for the Fig5 support-property experiment.")
    return pd.DataFrame(rows).sort_values(["probe_rank", "probe_id"], kind="stable").reset_index(drop=True)


def generate_canonical_support_mask(
    probe_image: torch.Tensor,
    *,
    target_area: int,
    foreground_threshold: float,
) -> dict[str, object] | None:
    probe_fg = _foreground_mask(probe_image, threshold=float(foreground_threshold))
    largest_component = _largest_component_coords(probe_fg)
    if largest_component.size <= 0 or int(target_area) > int(largest_component.shape[0]):
        return None
    seed_row, seed_col = _choose_component_seed(largest_component)
    ordered_coords = _sorted_coords_from_seed(largest_component, seed_row=seed_row, seed_col=seed_col)
    support_mask = _make_mask_from_coords(ordered_coords[: int(target_area)], probe_fg.shape)
    center_row, center_col = _mask_center(support_mask)
    coords_json = json.dumps(_mask_coords(support_mask).astype(int).tolist(), ensure_ascii=False)
    return {
        "support_mask": support_mask,
        "support_area": int(support_mask.sum()),
        "support_center_row": float(center_row),
        "support_center_col": float(center_col),
        "support_coords_json": coords_json,
    }


def generate_fixed_area_location_masks(
    probe_image: torch.Tensor,
    *,
    target_area: int,
    n_masks: int,
    foreground_threshold: float,
    seed: int,
    max_pairwise_iou: float,
) -> pd.DataFrame:
    probe_fg = _foreground_mask(probe_image, threshold=float(foreground_threshold))
    all_coords = _mask_coords(probe_fg)
    if all_coords.shape[0] < int(target_area):
        return pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    shuffled_coords = all_coords[rng.permutation(all_coords.shape[0])]
    seed_coords = _farthest_point_sample(shuffled_coords, n_points=max(int(n_masks) * 3, int(n_masks)))
    accepted_masks: list[np.ndarray] = []
    accepted_rows: list[dict[str, object]] = []
    for candidate in seed_coords.tolist():
        seed_row, seed_col = int(candidate[0]), int(candidate[1])
        ordered = _sorted_coords_for_local_region(all_coords, seed_row=seed_row, seed_col=seed_col)
        support_mask = _make_mask_from_coords(ordered[: int(target_area)], probe_fg.shape)
        if int(support_mask.sum()) != int(target_area):
            continue
        if any(_mask_iou(support_mask, existing) > float(max_pairwise_iou) for existing in accepted_masks):
            continue
        center_row, center_col = _mask_center(support_mask)
        accepted_masks.append(support_mask)
        accepted_rows.append(
            {
                "support_mask": support_mask,
                "support_area": int(support_mask.sum()),
                "support_center_row": float(center_row),
                "support_center_col": float(center_col),
                "support_coords_json": json.dumps(_mask_coords(support_mask).astype(int).tolist(), ensure_ascii=False),
            }
        )
        if len(accepted_rows) >= int(n_masks):
            break
    if len(accepted_rows) < 3:
        return pd.DataFrame()
    ordered_rows = sorted(
        accepted_rows,
        key=lambda item: (float(item["support_center_col"]), float(item["support_center_row"])),
    )
    for idx, row in enumerate(ordered_rows, start=1):
        row["location_index"] = int(idx)
        row["location_label"] = f"Loc {int(idx)}"
    return pd.DataFrame(ordered_rows)


def construct_synthetic_sample_from_support(probe_image: torch.Tensor, support_mask: np.ndarray) -> torch.Tensor:
    if probe_image.ndim != 3:
        raise ValueError(f"Expected probe_image shape [C, H, W], got {tuple(probe_image.shape)}")
    mask_tensor = torch.as_tensor(np.asarray(support_mask, dtype=np.float32), dtype=torch.float32)
    return probe_image.detach().cpu().to(torch.float32) * mask_tensor.unsqueeze(0)


def build_support_metadata_table(
    probe_df: pd.DataFrame,
    *,
    fixed_support_area_pixels: int,
    support_area_levels: Sequence[int],
    num_location_masks: int,
    foreground_threshold: float,
    mask_generation_seed: int,
    max_pairwise_iou: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    support_rows: list[dict[str, object]] = []
    generation_notes: list[dict[str, object]] = []
    for probe_row in probe_df.itertuples(index=False):
        canonical = generate_canonical_support_mask(
            probe_row.probe_image,
            target_area=int(fixed_support_area_pixels),
            foreground_threshold=float(foreground_threshold),
        )
        if canonical is None:
            generation_notes.append(
                {
                    "probe_id": int(probe_row.probe_id),
                    "support_role": SUPPORT_ROLE_CANONICAL,
                    "status": "skipped_insufficient_foreground_for_canonical_support",
                }
            )
        else:
            support_rows.append(
                {
                    "probe_id": int(probe_row.probe_id),
                    "probe_label": int(probe_row.probe_label),
                    "support_id": f"probe_{int(probe_row.probe_id)}_canonical",
                    "support_role": SUPPORT_ROLE_CANONICAL,
                    "support_area": int(canonical["support_area"]),
                    "support_center_row": float(canonical["support_center_row"]),
                    "support_center_col": float(canonical["support_center_col"]),
                    "area_index": 1,
                    "area_label": f"Area {int(canonical['support_area'])}",
                    "location_index": 1,
                    "location_label": "Canonical",
                    "support_mask": canonical["support_mask"],
                    "support_coords_json": str(canonical["support_coords_json"]),
                }
            )
        area_df = generate_nested_support_masks(
            probe_row.probe_image,
            target_area_schedule=support_area_levels,
            foreground_threshold=float(foreground_threshold),
        )
        if area_df.empty:
            generation_notes.append(
                {
                    "probe_id": int(probe_row.probe_id),
                    "support_role": SUPPORT_ROLE_AREA,
                    "status": "skipped_insufficient_foreground_for_area_supports",
                }
            )
        else:
            for row in area_df.itertuples(index=False):
                support_rows.append(
                    {
                        "probe_id": int(probe_row.probe_id),
                        "probe_label": int(probe_row.probe_label),
                        "support_id": f"probe_{int(probe_row.probe_id)}_area_{int(row.area_index)}",
                        "support_role": SUPPORT_ROLE_AREA,
                        "support_area": int(row.support_area),
                        "support_center_row": float(row.support_center_row),
                        "support_center_col": float(row.support_center_col),
                        "area_index": int(row.area_index),
                        "area_label": str(row.area_label),
                        "location_index": 1,
                        "location_label": "Fixed location",
                        "support_mask": row.support_mask,
                        "support_coords_json": str(row.support_coords_json),
                    }
                )
        location_df = generate_fixed_area_location_masks(
            probe_row.probe_image,
            target_area=int(fixed_support_area_pixels),
            n_masks=int(num_location_masks),
            foreground_threshold=float(foreground_threshold),
            seed=int(mask_generation_seed) + int(probe_row.probe_id) * 97,
            max_pairwise_iou=float(max_pairwise_iou),
        )
        if location_df.empty:
            generation_notes.append(
                {
                    "probe_id": int(probe_row.probe_id),
                    "support_role": SUPPORT_ROLE_LOCATION,
                    "status": "skipped_insufficient_distinct_location_supports",
                }
            )
        else:
            for row in location_df.itertuples(index=False):
                support_rows.append(
                    {
                        "probe_id": int(probe_row.probe_id),
                        "probe_label": int(probe_row.probe_label),
                        "support_id": f"probe_{int(probe_row.probe_id)}_location_{int(row.location_index)}",
                        "support_role": SUPPORT_ROLE_LOCATION,
                        "support_area": int(row.support_area),
                        "support_center_row": float(row.support_center_row),
                        "support_center_col": float(row.support_center_col),
                        "area_index": 1,
                        "area_label": f"Area {int(row.support_area)}",
                        "location_index": int(row.location_index),
                        "location_label": str(row.location_label),
                        "support_mask": row.support_mask,
                        "support_coords_json": str(row.support_coords_json),
                    }
                )
    if not support_rows:
        raise RuntimeError("No valid support masks were generated.")
    support_df = pd.DataFrame(support_rows).sort_values(
        ["probe_id", "support_role", "area_index", "location_index", "support_id"],
        kind="stable",
    ).reset_index(drop=True)
    return support_df, generation_notes


def _prepare_sample_probe_spikes(
    sample_images: torch.Tensor,
    probe_images: torch.Tensor,
    *,
    encoder,
    spec: ExperimentSpec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_encoded = encode_images(encoder, sample_images.to(device=device, dtype=torch.float32), steps=int(spec.sample_steps))
    probe_encoded = encode_images(encoder, probe_images.to(device=device, dtype=torch.float32), steps=int(spec.probe_steps))
    return sample_encoded, probe_encoded


def _scaled_gain_mean(
    u_tensor: torch.Tensor,
    x_tensor: torch.Tensor,
    mask_bool: torch.Tensor,
    *,
    baseline_u: float,
    scale: float,
) -> float:
    if not bool(mask_bool.any()):
        return float("nan")
    scaled_u = baseline_u + float(scale) * (u_tensor - float(baseline_u))
    scaled_x = 1.0 - float(scale) * (1.0 - x_tensor)
    scaled_u = torch.clamp(scaled_u, 0.0, 1.0)
    scaled_x = torch.clamp(scaled_x, 0.0, 1.0)
    gain_map = (scaled_u * scaled_x).mean(dim=0)
    return float(gain_map[mask_bool].mean().detach().cpu().item())


def _solve_scale_for_target_mean_ux(
    u_tensor: torch.Tensor,
    x_tensor: torch.Tensor,
    mask_bool: torch.Tensor,
    *,
    baseline_u: float,
    target_mean_ux: float,
    max_scale: float = DEFAULT_UX_TARGET_SOLVE_MAX_SCALE,
    coarse_steps: int = DEFAULT_UX_TARGET_SOLVE_COARSE_STEPS,
    refine_steps: int = DEFAULT_UX_TARGET_SOLVE_REFINE_STEPS,
    refine_rounds: int = DEFAULT_UX_TARGET_SOLVE_REFINE_ROUNDS,
) -> float:
    if not bool(mask_bool.any()) or not np.isfinite(float(target_mean_ux)):
        return 1.0
    target = float(target_mean_ux)
    coarse = np.linspace(0.0, float(max_scale), max(int(coarse_steps), 3), dtype=np.float64)
    coarse_means = np.asarray(
        [_scaled_gain_mean(u_tensor, x_tensor, mask_bool, baseline_u=float(baseline_u), scale=float(scale)) for scale in coarse],
        dtype=np.float64,
    )
    finite = np.isfinite(coarse_means)
    if not finite.any():
        return 1.0
    coarse = coarse[finite]
    coarse_means = coarse_means[finite]
    best_idx = int(np.argmin(np.abs(coarse_means - target)))
    best_scale = float(coarse[best_idx])
    left_idx = max(best_idx - 1, 0)
    right_idx = min(best_idx + 1, len(coarse) - 1)
    lo = float(coarse[left_idx])
    hi = float(coarse[right_idx])
    if math.isclose(lo, hi):
        return best_scale
    for _ in range(max(int(refine_rounds), 0)):
        refine = np.linspace(lo, hi, max(int(refine_steps), 3), dtype=np.float64)
        refine_means = np.asarray(
            [_scaled_gain_mean(u_tensor, x_tensor, mask_bool, baseline_u=float(baseline_u), scale=float(scale)) for scale in refine],
            dtype=np.float64,
        )
        finite_refine = np.isfinite(refine_means)
        if not finite_refine.any():
            break
        refine = refine[finite_refine]
        refine_means = refine_means[finite_refine]
        best_idx = int(np.argmin(np.abs(refine_means - target)))
        best_scale = float(refine[best_idx])
        left_idx = max(best_idx - 1, 0)
        right_idx = min(best_idx + 1, len(refine) - 1)
        lo = float(refine[left_idx])
        hi = float(refine[right_idx])
        if math.isclose(lo, hi):
            break
    return float(best_scale)


def _apply_support_state_scale(
    u_tensor: torch.Tensor,
    x_tensor: torch.Tensor,
    *,
    mask_bool: torch.Tensor,
    baseline_u: float,
    scale: float,
) -> None:
    if not bool(mask_bool.any()):
        return
    mask3 = mask_bool.unsqueeze(0).expand_as(u_tensor)
    scaled_u = torch.clamp(float(baseline_u) + float(scale) * (u_tensor - float(baseline_u)), 0.0, 1.0)
    scaled_x = torch.clamp(1.0 - float(scale) * (1.0 - x_tensor), 0.0, 1.0)
    u_tensor[mask3] = scaled_u[mask3]
    x_tensor[mask3] = scaled_x[mask3]


def support_df_row_mask(df: pd.DataFrame, support_id: str) -> np.ndarray:
    row = df[df["support_id"] == str(support_id)]
    if row.empty:
        raise KeyError(f"Support ID not found: {support_id}")
    return np.asarray(row.iloc[0]["support_mask"], dtype=bool)


def _build_intervention_plan(batch_df: pd.DataFrame) -> Mapping[str, object] | None:
    if batch_df.empty or batch_df["intervention_type"].eq("none").all():
        return None
    batch_records = batch_df.copy().reset_index(drop=True)

    def before_probe_fn(net, ctx):
        del ctx
        layer = net.layer1
        baseline_u = float(layer.stsp_U)
        interventions: list[dict[str, object]] = []
        with torch.no_grad():
            for batch_idx, row in enumerate(batch_records.itertuples(index=False)):
                if str(row.intervention_type) == "none":
                    interventions.append(
                        {
                            "record_id": int(row.record_id),
                            "intervention_type": "none",
                            "applied_scale": 1.0,
                        }
                    )
                    continue
                support_mask = np.asarray(row.support_mask, dtype=bool)
                mask_bool = torch.as_tensor(support_mask, dtype=torch.bool, device=layer.u_pre.device)
                if str(row.intervention_type) == "scale":
                    applied_scale = float(row.dose_scale)
                elif str(row.intervention_type) in {"match", "target_mean_ux"}:
                    applied_scale = _solve_scale_for_target_mean_ux(
                        layer.u_pre[batch_idx],
                        layer.x_pre[batch_idx],
                        mask_bool,
                        baseline_u=baseline_u,
                        target_mean_ux=float(row.target_mean_ux_on_support),
                    )
                else:
                    raise ValueError(f"Unsupported intervention_type: {row.intervention_type}")
                _apply_support_state_scale(
                    layer.u_pre[batch_idx],
                    layer.x_pre[batch_idx],
                    mask_bool=mask_bool,
                    baseline_u=baseline_u,
                    scale=float(applied_scale),
                )
                interventions.append(
                    {
                        "record_id": int(row.record_id),
                        "intervention_type": str(row.intervention_type),
                        "applied_scale": float(applied_scale),
                        "target_mean_ux_on_support": (
                            None if pd.isna(row.target_mean_ux_on_support) else float(row.target_mean_ux_on_support)
                        ),
                    }
                )
        return {"support_interventions": interventions}

    return {"before_probe_fn": before_probe_fn}


def execute_support_plan(
    plan_df: pd.DataFrame,
    *,
    net,
    encoder,
    device: torch.device,
    readout_step: int,
    batch_size: int,
    spec: ExperimentSpec,
) -> pd.DataFrame:
    if plan_df.empty:
        return pd.DataFrame()
    result_rows: list[dict[str, object]] = []
    ordered_plan = plan_df.sort_values(
        ["delay_ms", "module_name", "probe_id", "support_id", "condition_order", "record_id"],
        kind="stable",
    ).reset_index(drop=True)
    for delay_ms, delay_subset in ordered_plan.groupby("delay_ms", sort=True):
        delay_steps = _delay_ms_to_steps(float(delay_ms), spec.dt)
        batch_starts = range(0, len(delay_subset), int(batch_size))
        total_batches = math.ceil(len(delay_subset) / int(batch_size))
        desc = f"{delay_ms:.0f} ms"
        for batch_start in tqdm(batch_starts, total=total_batches, desc=desc, leave=False, dynamic_ncols=True):
            batch = delay_subset.iloc[batch_start : batch_start + int(batch_size)].copy().reset_index(drop=True)
            sample_batch = torch.stack([img for img in batch["sample_image"].tolist()], dim=0)
            probe_batch = torch.stack([img for img in batch["probe_image"].tolist()], dim=0)
            sample_spikes, probe_spikes = _prepare_sample_probe_spikes(
                sample_images=sample_batch,
                probe_images=probe_batch,
                encoder=encoder,
                spec=spec,
                device=device,
            )
            dynamic_output = run_dms_snapshot_rollout(
                net=net,
                sample_spikes=sample_spikes,
                probe_spikes=probe_spikes,
                delay_steps=int(delay_steps),
                stsp_mode="dynamic",
                phase_reset=bool(spec.phase_reset),
                intervention_plan=_build_intervention_plan(batch),
                readout_step=int(readout_step),
                snapshot_state_names=("v_mem", "u", "x"),
                record_full_trace_state_names=(),
            )
            static_output = run_dms_snapshot_rollout(
                net=net,
                sample_spikes=sample_spikes,
                probe_spikes=probe_spikes,
                delay_steps=int(delay_steps),
                stsp_mode="static_frozen",
                phase_reset=bool(spec.phase_reset),
                intervention_plan=None,
                readout_step=int(readout_step),
                snapshot_state_names=("v_mem",),
                record_full_trace_state_names=(),
            )
            grouped_voltage_dynamic = extract_grouped_voltage_vector(
                net,
                dynamic_output["readout_snapshots"]["layer3"]["v_mem"],
            )
            grouped_voltage_static = extract_grouped_voltage_vector(
                net,
                static_output["readout_snapshots"]["layer3"]["v_mem"],
            )
            delta_v = np.asarray(compute_delta_v(grouped_voltage_dynamic, grouped_voltage_static), dtype=np.float64)
            boundary_pre = dynamic_output["boundary_states"]["pre_intervention"]["layer1"]
            boundary_post = dynamic_output["boundary_states"]["post_intervention"]["layer1"]
            intervention_lookup = {
                int(item["record_id"]): item
                for item in dynamic_output["intervention_record"].get("support_interventions", [])
            }
            for batch_idx, row in enumerate(batch.itertuples(index=False)):
                pre_metrics = _summarize_support_metrics_from_boundary(boundary_pre, batch_idx=batch_idx, support_mask=row.support_mask)
                post_metrics = _summarize_support_metrics_from_boundary(boundary_post, batch_idx=batch_idx, support_mask=row.support_mask)
                gain_map_pre = _mean_gain_map_from_boundary(boundary_pre, batch_idx=batch_idx)
                gain_map_post = _mean_gain_map_from_boundary(boundary_post, batch_idx=batch_idx)
                delta_vec = np.asarray(delta_v[batch_idx], dtype=np.float64).reshape(-1)
                magnitude = float(np.linalg.norm(delta_vec))
                support_area = int(row.support_area)
                bias_magnitude_per_pixel = float(magnitude / max(support_area, 1))
                pair_status = "ok"
                if support_area <= 0:
                    pair_status = "empty_support"
                elif not np.isfinite(magnitude):
                    pair_status = "invalid_numeric"
                result_rows.append(
                    {
                        "record_id": int(row.record_id),
                        "module_name": str(row.module_name),
                        "probe_id": int(row.probe_id),
                        "probe_label": int(row.probe_label),
                        "support_id": str(row.support_id),
                        "patch_id": str(getattr(row, "patch_id", row.support_id)),
                        "support_role": str(row.support_role),
                        "importance_tier": int(getattr(row, "importance_tier", 0)),
                        "mean_importance_on_patch": float(getattr(row, "mean_importance_on_patch", float("nan"))),
                        "support_area": int(row.support_area),
                        "support_center_row": float(row.support_center_row),
                        "support_center_col": float(row.support_center_col),
                        "area_index": int(row.area_index),
                        "area_label": str(row.area_label),
                        "location_index": int(row.location_index),
                        "location_label": str(row.location_label),
                        "dose_label": str(row.dose_label),
                        "dose_order": int(row.dose_order),
                        "dose_scale": float(row.dose_scale),
                        "delay_ms": float(row.delay_ms),
                        "condition_name": str(row.condition_name),
                        "condition_label": str(row.condition_label),
                        "condition_order": int(row.condition_order),
                        "intervention_type": str(row.intervention_type),
                        "target_mean_ux_on_support": (
                            float(row.target_mean_ux_on_support) if pd.notna(row.target_mean_ux_on_support) else float("nan")
                        ),
                        "applied_scale": float(intervention_lookup.get(int(row.record_id), {}).get("applied_scale", 1.0)),
                        "mean_ux_on_support_pre": float(pre_metrics["mean_ux_on_support"]),
                        "total_ux_support_pre": float(pre_metrics["total_ux_support"]),
                        "mean_ux_on_support": float(post_metrics["mean_ux_on_support"]),
                        "total_ux_support": float(post_metrics["total_ux_support"]),
                        "bias_magnitude": float(magnitude),
                        "bias_magnitude_per_pixel": float(bias_magnitude_per_pixel),
                        "pair_status": str(pair_status),
                        "support_mask": np.asarray(row.support_mask, dtype=bool),
                        "gain_map_pre": np.asarray(gain_map_pre, dtype=np.float64),
                        "gain_map_post": np.asarray(gain_map_post, dtype=np.float64),
                        "grouped_voltage_dynamic": np.asarray(grouped_voltage_dynamic[batch_idx], dtype=np.float64).reshape(-1),
                        "grouped_voltage_static": np.asarray(grouped_voltage_static[batch_idx], dtype=np.float64).reshape(-1),
                        "DeltaV": delta_vec,
                    }
                )
    result_df = pd.DataFrame(result_rows).sort_values(
        ["module_name", "probe_id", "support_id", "condition_order", "delay_ms"],
        kind="stable",
    ).reset_index(drop=True)
    return result_df


def build_ux_dose_plan(
    support_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    *,
    dose_levels: Sequence[Mapping[str, object]],
    delay_ms: float,
) -> pd.DataFrame:
    canonical = support_df[support_df["support_role"] == SUPPORT_ROLE_CANONICAL].copy()
    probe_lookup = {int(row.probe_id): row.probe_image for row in probe_df.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    record_id = 0
    for row in canonical.itertuples(index=False):
        probe_image = probe_lookup[int(row.probe_id)]
        sample_image = construct_synthetic_sample_from_support(probe_image, row.support_mask)
        for level in dose_levels:
            rows.append(
                {
                    "record_id": int(record_id),
                    "module_name": "ux_dose",
                    "probe_id": int(row.probe_id),
                    "probe_label": int(row.probe_label),
                    "support_id": str(row.support_id),
                    "support_role": str(row.support_role),
                    "support_area": int(row.support_area),
                    "support_center_row": float(row.support_center_row),
                    "support_center_col": float(row.support_center_col),
                    "area_index": int(row.area_index),
                    "area_label": str(row.area_label),
                    "location_index": 1,
                    "location_label": "Canonical",
                    "dose_label": str(level["dose_label"]),
                    "dose_order": int(level["dose_order"]),
                    "dose_scale": float("nan"),
                    "delay_ms": float(delay_ms),
                    "condition_name": f"ux_mean_{str(level['dose_label']).replace('.', 'p')}",
                    "condition_label": str(level["dose_label"]),
                    "condition_order": int(level["dose_order"]),
                    "intervention_type": "target_mean_ux",
                    "target_mean_ux_on_support": float(level["target_mean_ux_on_support"]),
                    "support_mask": row.support_mask,
                    "probe_image": probe_image,
                    "sample_image": sample_image,
                }
            )
            record_id += 1
    return pd.DataFrame(rows)


def build_area_plan(
    support_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    *,
    delay_ms: float,
) -> pd.DataFrame:
    area_df = support_df[support_df["support_role"] == SUPPORT_ROLE_AREA].copy()
    probe_lookup = {int(row.probe_id): row.probe_image for row in probe_df.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    record_id = 0
    for row in area_df.itertuples(index=False):
        probe_image = probe_lookup[int(row.probe_id)]
        rows.append(
            {
                "record_id": int(record_id),
                "module_name": "support_area",
                "probe_id": int(row.probe_id),
                "probe_label": int(row.probe_label),
                "support_id": str(row.support_id),
                "support_role": str(row.support_role),
                "support_area": int(row.support_area),
                "support_center_row": float(row.support_center_row),
                "support_center_col": float(row.support_center_col),
                "area_index": int(row.area_index),
                "area_label": str(row.area_label),
                "location_index": 1,
                "location_label": "Fixed location",
                "dose_label": "Fixed",
                "dose_order": 1,
                "dose_scale": 1.0,
                "delay_ms": float(delay_ms),
                "condition_name": f"support_area_{int(row.support_area)}",
                "condition_label": str(row.area_label),
                "condition_order": int(row.area_index),
                "intervention_type": "none",
                "target_mean_ux_on_support": np.nan,
                "support_mask": row.support_mask,
                "probe_image": probe_image,
                "sample_image": construct_synthetic_sample_from_support(probe_image, row.support_mask),
            }
        )
        record_id += 1
    return pd.DataFrame(rows)


def build_location_natural_plan(
    support_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    *,
    delay_ms: float,
) -> pd.DataFrame:
    location_df = support_df[support_df["support_role"] == SUPPORT_ROLE_LOCATION].copy()
    probe_lookup = {int(row.probe_id): row.probe_image for row in probe_df.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    record_id = 0
    for row in location_df.itertuples(index=False):
        probe_image = probe_lookup[int(row.probe_id)]
        rows.append(
            {
                "record_id": int(record_id),
                "module_name": "support_location_natural",
                "probe_id": int(row.probe_id),
                "probe_label": int(row.probe_label),
                "support_id": str(row.support_id),
                "support_role": str(row.support_role),
                "support_area": int(row.support_area),
                "support_center_row": float(row.support_center_row),
                "support_center_col": float(row.support_center_col),
                "area_index": int(row.area_index),
                "area_label": str(row.area_label),
                "location_index": int(row.location_index),
                "location_label": str(row.location_label),
                "dose_label": "Matched",
                "dose_order": 1,
                "dose_scale": 1.0,
                "delay_ms": float(delay_ms),
                "condition_name": f"location_natural_{int(row.location_index)}",
                "condition_label": str(row.location_label),
                "condition_order": int(row.location_index),
                "intervention_type": "none",
                "target_mean_ux_on_support": np.nan,
                "support_mask": row.support_mask,
                "probe_image": probe_image,
                "sample_image": construct_synthetic_sample_from_support(probe_image, row.support_mask),
            }
        )
        record_id += 1
    return pd.DataFrame(rows)


def build_location_matched_plan(
    location_natural_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    support_df: pd.DataFrame,
) -> pd.DataFrame:
    valid = location_natural_df[location_natural_df["pair_status"] == "ok"].copy()
    probe_lookup = {int(row.probe_id): row.probe_image for row in probe_df.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    record_id = 0
    if valid.empty:
        return pd.DataFrame()
    for probe_id, probe_subset in valid.groupby("probe_id", sort=True):
        target_mean = float(probe_subset["mean_ux_on_support"].min())
        for row in probe_subset.itertuples(index=False):
            probe_image = probe_lookup[int(row.probe_id)]
            support_mask = support_df_row_mask(support_df, row.support_id)
            rows.append(
                {
                    "record_id": int(record_id),
                    "module_name": "support_location",
                    "probe_id": int(row.probe_id),
                    "probe_label": int(row.probe_label),
                    "support_id": str(row.support_id),
                    "support_role": str(row.support_role),
                    "support_area": int(row.support_area),
                    "support_center_row": float(row.support_center_row),
                    "support_center_col": float(row.support_center_col),
                    "area_index": int(row.area_index),
                    "area_label": str(row.area_label),
                    "location_index": int(row.location_index),
                    "location_label": str(row.location_label),
                    "dose_label": "Matched",
                    "dose_order": 1,
                    "dose_scale": 1.0,
                    "delay_ms": float(row.delay_ms),
                    "condition_name": f"support_location_{int(row.location_index)}",
                    "condition_label": str(row.location_label),
                    "condition_order": int(row.location_index),
                    "intervention_type": "match",
                    "target_mean_ux_on_support": float(target_mean),
                    "support_mask": support_mask,
                    "probe_image": probe_image,
                    "sample_image": construct_synthetic_sample_from_support(probe_image, support_mask),
                }
            )
            record_id += 1
    return pd.DataFrame(rows)


def build_delay_natural_plan(
    support_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    *,
    delay_sweep_ms: Sequence[float],
) -> pd.DataFrame:
    canonical = support_df[support_df["support_role"] == SUPPORT_ROLE_CANONICAL].copy()
    probe_lookup = {int(row.probe_id): row.probe_image for row in probe_df.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    record_id = 0
    for row in canonical.itertuples(index=False):
        probe_image = probe_lookup[int(row.probe_id)]
        sample_image = construct_synthetic_sample_from_support(probe_image, row.support_mask)
        for order, delay_ms in enumerate(delay_sweep_ms, start=1):
            rows.append(
                {
                    "record_id": int(record_id),
                    "module_name": "support_delay",
                    "probe_id": int(row.probe_id),
                    "probe_label": int(row.probe_label),
                    "support_id": str(row.support_id),
                    "support_role": str(row.support_role),
                    "support_area": int(row.support_area),
                    "support_center_row": float(row.support_center_row),
                    "support_center_col": float(row.support_center_col),
                    "area_index": int(row.area_index),
                    "area_label": str(row.area_label),
                    "location_index": 1,
                    "location_label": "Canonical",
                    "dose_label": "Natural",
                    "dose_order": 1,
                    "dose_scale": 1.0,
                    "delay_ms": float(delay_ms),
                    "condition_name": f"support_delay_{int(round(float(delay_ms)))}",
                    "condition_label": f"{float(delay_ms):.0f}",
                    "condition_order": int(order),
                    "intervention_type": "none",
                    "target_mean_ux_on_support": np.nan,
                    "support_mask": row.support_mask,
                    "probe_image": probe_image,
                    "sample_image": sample_image,
                }
            )
            record_id += 1
    return pd.DataFrame(rows)


def build_probe_importance_metadata(
    probe_df: pd.DataFrame,
    *,
    encoder,
    foreground_threshold: float,
    atomic_patch_area: int,
    importance_tiers: int,
    importance_smoothing_alpha: float,
    importance_smoothing_passes: int,
) -> tuple[dict[int, dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    payloads: dict[int, dict[str, object]] = {}
    patch_rows: list[pd.DataFrame] = []
    notes: list[dict[str, object]] = []
    for probe_row in probe_df.itertuples(index=False):
        payload = compute_latency_importance_payload(
            probe_row.probe_image,
            encoder=encoder,
            foreground_threshold=float(foreground_threshold),
            importance_smoothing_alpha=float(importance_smoothing_alpha),
            importance_smoothing_passes=int(importance_smoothing_passes),
        )
        tier_payload = assign_importance_tiers(
            payload["importance_map"],
            payload["foreground_mask"],
            n_tiers=int(importance_tiers),
        )
        patch_df, patch_notes = build_atomic_tier_patches(
            probe_id=int(probe_row.probe_id),
            importance_map=payload["importance_map"],
            foreground_mask=payload["foreground_mask"],
            tier_map=tier_payload["tier_map"],
            patch_area=int(atomic_patch_area),
            n_tiers=int(importance_tiers),
        )
        notes.extend(patch_notes)
        payloads[int(probe_row.probe_id)] = {
            **payload,
            **tier_payload,
            "patch_df": patch_df,
        }
        if not patch_df.empty:
            patch_rows.append(patch_df)
    patch_metadata_df = (
        pd.concat(patch_rows, axis=0, ignore_index=True)
        if patch_rows
        else pd.DataFrame(
            columns=[
                "probe_id",
                "patch_id",
                "importance_tier",
                "support_mask",
                "support_area",
                "mean_importance",
                "tier_mean_importance",
                "support_center_row",
                "support_center_col",
                "support_coords_json",
                "target_tier_coverage_ratio",
            ]
        )
    )
    return payloads, patch_metadata_df, notes


def build_importance_patch_dose_plan(
    patch_metadata_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    *,
    dose_levels: Sequence[Mapping[str, object]],
    delay_ms: float,
) -> pd.DataFrame:
    probe_lookup = {int(row.probe_id): row for row in probe_df.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    record_id = 0
    for row in patch_metadata_df.itertuples(index=False):
        probe_row = probe_lookup.get(int(row.probe_id))
        if probe_row is None:
            continue
        sample_image = construct_synthetic_sample_from_support(probe_row.probe_image, row.support_mask)
        for level in dose_levels:
            rows.append(
                {
                    "record_id": int(record_id),
                    "module_name": "importance_dose_law",
                    "probe_id": int(row.probe_id),
                    "probe_label": int(probe_row.probe_label),
                    "support_id": str(row.patch_id),
                    "patch_id": str(row.patch_id),
                    "support_role": "importance_tier_patch",
                    "importance_tier": int(row.importance_tier),
                    "mean_importance_on_patch": float(row.mean_importance),
                    "support_area": int(row.support_area),
                    "support_center_row": float(row.support_center_row),
                    "support_center_col": float(row.support_center_col),
                    "area_index": 1,
                    "area_label": f"Atomic patch {int(row.support_area)}",
                    "location_index": int(row.importance_tier),
                    "location_label": f"Tier {int(row.importance_tier)}",
                    "dose_label": str(level["dose_label"]),
                    "dose_order": int(level["dose_order"]),
                    "dose_scale": float("nan"),
                    "delay_ms": float(delay_ms),
                    "condition_name": f"tier_{int(row.importance_tier)}_ux_{str(level['dose_label']).replace('.', 'p')}",
                    "condition_label": f"Tier {int(row.importance_tier)} / {str(level['dose_label'])}",
                    "condition_order": int(level["dose_order"]),
                    "intervention_type": "target_mean_ux",
                    "target_mean_ux_on_support": float(level["target_mean_ux_on_support"]),
                    "support_mask": np.asarray(row.support_mask, dtype=bool),
                    "probe_image": probe_row.probe_image,
                    "sample_image": sample_image,
                }
            )
            record_id += 1
    return pd.DataFrame(rows)


def _json_dumps_float_list(values: np.ndarray | Sequence[float]) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return json.dumps(arr.astype(float).tolist(), ensure_ascii=False)


def summarize_importance_dose_records(records_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid = records_df[
        (records_df["module_name"] == "importance_dose_law")
        & (records_df["pair_status"] == "ok")
        & (records_df["importance_tier"] > 0)
    ].copy()
    if valid.empty:
        return {
            "record_df": pd.DataFrame(
                columns=[
                    "probe_id",
                    "patch_id",
                    "importance_tier",
                    "dose_order",
                    "target_mean_ux_on_support",
                    "mean_ux_on_support",
                    "bias_magnitude",
                    "signed_direction_deg",
                ]
            ),
            "pooled_df": pd.DataFrame(
                columns=[
                    "importance_tier",
                    "dose_order",
                    "dose_label",
                    "target_mean_ux_on_support",
                    "mean_ux_on_support",
                    "sem_mean_ux_on_support",
                    "bias_magnitude",
                    "sem_bias_magnitude",
                    "signed_direction_deg",
                    "sem_signed_direction_deg",
                    "n_records",
                ]
            ),
        }
    pooled = (
        valid.groupby(["importance_tier", "dose_order", "dose_label"], sort=True)
        .agg(
            target_mean_ux_on_support=("target_mean_ux_on_support", "mean"),
            mean_ux_on_support=("mean_ux_on_support", "mean"),
            sem_mean_ux_on_support=("mean_ux_on_support", _sem),
            bias_magnitude=("bias_magnitude", "mean"),
            sem_bias_magnitude=("bias_magnitude", _sem),
            signed_direction_deg=("signed_direction_deg", "mean"),
            sem_signed_direction_deg=("signed_direction_deg", _sem),
            n_records=("record_id", "count"),
        )
        .reset_index()
        .sort_values(["importance_tier", "dose_order"], kind="stable")
        .reset_index(drop=True)
    )
    return {"record_df": valid, "pooled_df": pooled}


def fit_tierwise_local_dose_laws(
    dose_record_df: pd.DataFrame,
    *,
    direction_reference: Mapping[str, object],
    importance_tiers: int,
) -> dict[str, object]:
    summary_rows: list[dict[str, object]] = []
    fit_lookup: dict[int, dict[str, object]] = {}
    for tier in range(1, int(importance_tiers) + 1):
        tier_df = dose_record_df[
            (dose_record_df["importance_tier"] == int(tier))
            & np.isfinite(dose_record_df["mean_ux_on_support"])
        ].copy()
        if tier_df.empty:
            summary_rows.append(
                {
                    "importance_tier": int(tier),
                    "fit_method": "piecewise_linear_anchor_interpolation",
                    "fit_params": json.dumps({}, ensure_ascii=False),
                    "fit_quality": json.dumps({"status": "empty"}, ensure_ascii=False),
                }
            )
            continue
        anchors = (
            tier_df.groupby("dose_order", sort=True)
            .agg(
                dose=("mean_ux_on_support", "mean"),
                target_mean_ux_on_support=("target_mean_ux_on_support", "mean"),
                bias_magnitude=("bias_magnitude", "mean"),
                signed_direction_deg=("signed_direction_deg", "mean"),
                n_records=("record_id", "count"),
            )
            .reset_index()
            .sort_values("dose", kind="stable")
            .reset_index(drop=True)
        )
        delta_by_order = {
            int(order): np.stack(group["DeltaV"].to_list(), axis=0).mean(axis=0).astype(np.float64)
            for order, group in tier_df.groupby("dose_order", sort=True)
        }
        delta_anchor = np.stack(
            [delta_by_order[int(order)] for order in anchors["dose_order"].tolist()],
            axis=0,
        ).astype(np.float64)
        x_anchor = anchors["dose"].to_numpy(dtype=np.float64)
        pred_vectors: list[np.ndarray] = []
        actual_vectors: list[np.ndarray] = []
        pred_magnitudes: list[float] = []
        actual_magnitudes: list[float] = []
        pred_directions: list[float] = []
        actual_directions: list[float] = []
        for row in tier_df.itertuples(index=False):
            predicted_delta = np.asarray(
                [
                    np.interp(
                        float(row.mean_ux_on_support),
                        x_anchor,
                        delta_anchor[:, dim],
                        left=float(delta_anchor[0, dim]),
                        right=float(delta_anchor[-1, dim]),
                    )
                    for dim in range(delta_anchor.shape[1])
                ],
                dtype=np.float64,
            )
            pred_vectors.append(predicted_delta)
            actual_vectors.append(np.asarray(row.DeltaV, dtype=np.float64).reshape(-1))
            pred_magnitudes.append(float(np.linalg.norm(predicted_delta)))
            actual_magnitudes.append(float(row.bias_magnitude))
            pred_directions.append(
                compute_signed_direction_deg(
                    np.asarray(row.grouped_voltage_static, dtype=np.float64).reshape(-1) + predicted_delta,
                    row.grouped_voltage_static,
                    reference=direction_reference,
                )
            )
            actual_directions.append(float(row.signed_direction_deg))
        vector_rmse = float(
            np.sqrt(
                np.mean(
                    [
                        float(np.mean(np.square(actual - predicted)))
                        for actual, predicted in zip(actual_vectors, pred_vectors)
                    ]
                )
            )
        )
        magnitude_mae = float(np.mean(np.abs(np.asarray(actual_magnitudes) - np.asarray(pred_magnitudes))))
        direction_errors = np.asarray(actual_directions, dtype=np.float64) - np.asarray(pred_directions, dtype=np.float64)
        direction_errors = direction_errors[np.isfinite(direction_errors)]
        direction_mae = float(np.mean(np.abs(direction_errors))) if direction_errors.size > 0 else float("nan")
        slope_mag = float("nan")
        slope_dir = float("nan")
        if len(anchors) >= 2:
            dx = float(anchors["dose"].iloc[1] - anchors["dose"].iloc[0])
            if abs(dx) > 1e-12:
                slope_mag = float((anchors["bias_magnitude"].iloc[1] - anchors["bias_magnitude"].iloc[0]) / dx)
                slope_dir = float((anchors["signed_direction_deg"].iloc[1] - anchors["signed_direction_deg"].iloc[0]) / dx)
        peak_idx = int(np.nanargmax(anchors["bias_magnitude"].to_numpy(dtype=np.float64)))
        fit_lookup[int(tier)] = {
            "dose_anchor": x_anchor,
            "target_anchor": anchors["target_mean_ux_on_support"].to_numpy(dtype=np.float64),
            "delta_anchor": delta_anchor,
            "magnitude_anchor": anchors["bias_magnitude"].to_numpy(dtype=np.float64),
            "direction_anchor": anchors["signed_direction_deg"].to_numpy(dtype=np.float64),
        }
        summary_rows.append(
            {
                "importance_tier": int(tier),
                "fit_method": "piecewise_linear_anchor_interpolation",
                "fit_params": json.dumps(
                    {
                        "dose_anchor": x_anchor.astype(float).tolist(),
                        "target_mean_ux_anchor": anchors["target_mean_ux_on_support"].astype(float).tolist(),
                        "magnitude_anchor": anchors["bias_magnitude"].astype(float).tolist(),
                        "direction_anchor": anchors["signed_direction_deg"].astype(float).tolist(),
                        "delta_v_anchor_json": delta_anchor.astype(float).tolist(),
                    },
                    ensure_ascii=False,
                ),
                "fit_quality": json.dumps(
                    {
                        "status": "ok",
                        "n_anchor_points": int(len(anchors)),
                        "vector_rmse": float(vector_rmse),
                        "magnitude_mae": float(magnitude_mae),
                        "direction_mae_deg": float(direction_mae),
                        "initial_magnitude_slope": float(slope_mag),
                        "initial_direction_slope_deg_per_ux": float(slope_dir),
                        "peak_magnitude": float(anchors["bias_magnitude"].iloc[peak_idx]),
                        "peak_dose": float(anchors["dose"].iloc[peak_idx]),
                        "direction_span_deg": float(
                            np.nanmax(anchors["signed_direction_deg"].to_numpy(dtype=np.float64))
                            - np.nanmin(anchors["signed_direction_deg"].to_numpy(dtype=np.float64))
                        ),
                    },
                    ensure_ascii=False,
                ),
            }
        )
    fit_summary_df = pd.DataFrame(summary_rows).sort_values("importance_tier", kind="stable").reset_index(drop=True)
    return {"fit_summary_df": fit_summary_df, "fit_lookup": fit_lookup}


def predict_dose_law_vector(
    fit_lookup: Mapping[int, Mapping[str, object]],
    *,
    importance_tier: int,
    dose: float,
) -> np.ndarray:
    fit = fit_lookup.get(int(importance_tier))
    if fit is None:
        return np.zeros(0, dtype=np.float64)
    dose_anchor = np.asarray(fit["dose_anchor"], dtype=np.float64)
    delta_anchor = np.asarray(fit["delta_anchor"], dtype=np.float64)
    if delta_anchor.ndim != 2 or dose_anchor.size != delta_anchor.shape[0]:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(
        [
            np.interp(
                float(dose),
                dose_anchor,
                delta_anchor[:, dim],
                left=float(delta_anchor[0, dim]),
                right=float(delta_anchor[-1, dim]),
            )
            for dim in range(delta_anchor.shape[1])
        ],
        dtype=np.float64,
    )


def build_composition_prediction_records(
    records_df: pd.DataFrame,
    *,
    module_name: str,
    probe_importance_payloads: Mapping[int, Mapping[str, object]],
    fit_lookup: Mapping[int, Mapping[str, object]],
    direction_reference: Mapping[str, object],
    atomic_patch_area: int,
    importance_tiers: int,
) -> pd.DataFrame:
    valid = records_df[(records_df["module_name"] == str(module_name)) & (records_df["pair_status"] == "ok")].copy()
    base_columns = [
        "probe_id",
        "probe_label",
        "support_id",
        "module_name",
        "support_role",
        "area_index",
        "area_label",
        "location_index",
        "location_label",
        "delay_ms",
        "support_area",
        "mean_importance_on_support",
        "actual_bias_magnitude",
        "predicted_bias_magnitude",
        "actual_signed_direction_deg",
        "predicted_signed_direction_deg",
        "vector_error_l2",
        "actual_delta_v_json",
        "predicted_delta_v_json",
    ] + [f"{prefix}{tier}" for prefix in ("c", "p", "d") for tier in range(1, int(importance_tiers) + 1)]
    if valid.empty:
        return pd.DataFrame(columns=base_columns)
    rows: list[dict[str, object]] = []
    for row in valid.itertuples(index=False):
        payload = probe_importance_payloads.get(int(row.probe_id))
        if payload is None:
            continue
        support_mask = np.asarray(row.support_mask, dtype=bool)
        tier_map = np.asarray(payload["tier_map"], dtype=np.int64)
        importance_map = np.asarray(payload["importance_map"], dtype=np.float64)
        gain_map = np.asarray(row.gain_map_post, dtype=np.float64)
        composition = summarize_support_composition(
            support_mask=support_mask,
            tier_map=tier_map,
            importance_map=importance_map,
            atomic_patch_area=int(atomic_patch_area),
            n_tiers=int(importance_tiers),
        )
        predicted_delta: np.ndarray | None = None
        local_doses: dict[str, float] = {}
        for tier in range(1, int(importance_tiers) + 1):
            tier_intersection = support_mask & (tier_map == int(tier))
            local_dose = float(np.mean(gain_map[tier_intersection])) if tier_intersection.any() else 0.0
            local_doses[f"d{int(tier)}"] = float(local_dose)
            tier_delta = predict_dose_law_vector(fit_lookup, importance_tier=int(tier), dose=float(local_dose))
            if tier_delta.size <= 0:
                continue
            if predicted_delta is None:
                predicted_delta = np.zeros_like(tier_delta, dtype=np.float64)
            predicted_delta = predicted_delta + float(composition[f"c{int(tier)}"]) * tier_delta
        if predicted_delta is None:
            predicted_delta = np.zeros_like(np.asarray(row.DeltaV, dtype=np.float64).reshape(-1), dtype=np.float64)
        predicted_magnitude = float(np.linalg.norm(predicted_delta))
        predicted_dynamic = np.asarray(row.grouped_voltage_static, dtype=np.float64).reshape(-1) + predicted_delta
        predicted_direction = compute_signed_direction_deg(
            predicted_dynamic,
            row.grouped_voltage_static,
            reference=direction_reference,
        )
        out = {
            "probe_id": int(row.probe_id),
            "probe_label": int(row.probe_label),
            "support_id": str(row.support_id),
            "module_name": str(row.module_name),
            "support_role": str(row.support_role),
            "area_index": int(row.area_index),
            "area_label": str(row.area_label),
            "location_index": int(row.location_index),
            "location_label": str(row.location_label),
            "delay_ms": float(row.delay_ms),
            "support_area": int(row.support_area),
            "mean_importance_on_support": float(composition["mean_importance_on_support"]),
            "actual_bias_magnitude": float(row.bias_magnitude),
            "predicted_bias_magnitude": float(predicted_magnitude),
            "actual_signed_direction_deg": float(row.signed_direction_deg),
            "predicted_signed_direction_deg": float(predicted_direction),
            "vector_error_l2": float(np.linalg.norm(np.asarray(row.DeltaV, dtype=np.float64).reshape(-1) - predicted_delta)),
            "actual_delta_v_json": _json_dumps_float_list(row.DeltaV),
            "predicted_delta_v_json": _json_dumps_float_list(predicted_delta),
        }
        for tier in range(1, int(importance_tiers) + 1):
            out[f"c{int(tier)}"] = float(composition[f"c{int(tier)}"])
            out[f"p{int(tier)}"] = float(composition[f"p{int(tier)}"])
            out[f"d{int(tier)}"] = float(local_doses[f"d{int(tier)}"])
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=base_columns)
    return pd.DataFrame(rows, columns=base_columns).sort_values(
        ["probe_id", "area_index", "location_index", "delay_ms", "support_id"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_composition_by_condition(
    composition_df: pd.DataFrame,
    *,
    condition_columns: Sequence[str],
    importance_tiers: int,
) -> pd.DataFrame:
    if composition_df.empty:
        cols = list(condition_columns) + [f"c{tier}" for tier in range(1, int(importance_tiers) + 1)] + ["n_records"]
        return pd.DataFrame(columns=cols)
    agg_map: dict[str, tuple[str, str | callable]] = {"n_records": ("probe_id", "count")}
    for tier in range(1, int(importance_tiers) + 1):
        agg_map[f"c{tier}"] = (f"c{tier}", "mean")
    return (
        composition_df.groupby(list(condition_columns), sort=True)
        .agg(**agg_map)
        .reset_index()
        .sort_values(list(condition_columns), kind="stable")
        .reset_index(drop=True)
    )


def summarize_prediction_by_condition(
    composition_df: pd.DataFrame,
    *,
    condition_columns: Sequence[str],
) -> pd.DataFrame:
    if composition_df.empty:
        return pd.DataFrame(
            columns=list(condition_columns)
            + [
                "actual_signed_direction_deg",
                "predicted_signed_direction_deg",
                "sem_actual_signed_direction_deg",
                "sem_predicted_signed_direction_deg",
                "actual_bias_magnitude",
                "predicted_bias_magnitude",
                "sem_actual_bias_magnitude",
                "sem_predicted_bias_magnitude",
                "n_records",
            ]
        )
    return (
        composition_df.groupby(list(condition_columns), sort=True)
        .agg(
            actual_signed_direction_deg=("actual_signed_direction_deg", "mean"),
            predicted_signed_direction_deg=("predicted_signed_direction_deg", "mean"),
            sem_actual_signed_direction_deg=("actual_signed_direction_deg", _sem),
            sem_predicted_signed_direction_deg=("predicted_signed_direction_deg", _sem),
            actual_bias_magnitude=("actual_bias_magnitude", "mean"),
            predicted_bias_magnitude=("predicted_bias_magnitude", "mean"),
            sem_actual_bias_magnitude=("actual_bias_magnitude", _sem),
            sem_predicted_bias_magnitude=("predicted_bias_magnitude", _sem),
            n_records=("probe_id", "count"),
        )
        .reset_index()
        .sort_values(list(condition_columns), kind="stable")
        .reset_index(drop=True)
    )


def summarize_delay_local_dose_evolution(delay_composition_df: pd.DataFrame, *, importance_tiers: int) -> pd.DataFrame:
    if delay_composition_df.empty:
        return pd.DataFrame(columns=["delay_ms"] + [f"d{tier}" for tier in range(1, int(importance_tiers) + 1)])
    agg_map = {f"d{tier}": (f"d{tier}", "mean") for tier in range(1, int(importance_tiers) + 1)}
    agg_map.update({f"sem_d{tier}": (f"d{tier}", _sem) for tier in range(1, int(importance_tiers) + 1)})
    return (
        delay_composition_df.groupby("delay_ms", sort=True)
        .agg(**agg_map, n_records=("probe_id", "count"))
        .reset_index()
        .sort_values("delay_ms", kind="stable")
        .reset_index(drop=True)
    )


def plot_panel_a_importance_map_and_atomic_patches(
    probe_df: pd.DataFrame,
    patch_metadata_df: pd.DataFrame,
    probe_importance_payloads: Mapping[int, Mapping[str, object]],
) -> tuple[plt.Figure, dict[str, object]]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    if patch_metadata_df.empty:
        ax.axis("off")
        return fig, {"status": "no_atomic_patches"}
    probe_id = int(patch_metadata_df.sort_values(["probe_id", "importance_tier"], kind="stable").iloc[0]["probe_id"])
    payload = probe_importance_payloads.get(int(probe_id))
    probe_image = probe_df[probe_df["probe_id"] == int(probe_id)].iloc[0]["probe_image"]
    if payload is None:
        ax.axis("off")
        return fig, {"status": "missing_importance_payload", "probe_id": probe_id}
    image = probe_image.detach().cpu().to(torch.float32).squeeze(0).numpy()
    im = ax.imshow(payload["importance_map"], cmap="magma", vmin=0.0, vmax=1.0)
    ax.contour(image, levels=[0.05], colors=["#FFFFFF"], linewidths=0.8, alpha=0.8)
    subset = patch_metadata_df[patch_metadata_df["probe_id"] == int(probe_id)].copy().sort_values("importance_tier", kind="stable")
    for row in subset.itertuples(index=False):
        mask = np.asarray(row.support_mask, dtype=bool)
        color = COLOR_TIER[int(row.importance_tier) - 1]
        ax.contour(mask.astype(np.float32), levels=[0.5], colors=[color], linewidths=1.4)
        center_row, center_col = _mask_center(mask)
        ax.text(center_col, center_row, f"T{int(row.importance_tier)}", color=color, fontsize=8, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Latency importance map and atomic patches")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Importance")
    fig.tight_layout()
    return fig, {"status": "ok", "probe_id": int(probe_id), "patch_count": int(len(subset))}


def plot_panel_dose_curve_by_tier(pooled_df: pd.DataFrame, *, value_col: str, ylabel: str) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    if pooled_df.empty:
        ax.set_xlabel("Mean ux on atomic patch")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        return fig
    for tier in sorted([int(v) for v in pooled_df["importance_tier"].dropna().unique().tolist()]):
        tier_df = pooled_df[pooled_df["importance_tier"] == int(tier)].copy().sort_values("mean_ux_on_support", kind="stable")
        if tier_df.empty:
            continue
        err_col = "sem_signed_direction_deg" if value_col == "signed_direction_deg" else "sem_bias_magnitude"
        ax.plot(
            tier_df["mean_ux_on_support"].to_numpy(dtype=np.float64),
            tier_df[value_col].to_numpy(dtype=np.float64),
            color=COLOR_TIER[int(tier) - 1],
            marker="o",
            linewidth=1.8,
            label=f"Tier {int(tier)}",
        )
        ax.fill_between(
            tier_df["mean_ux_on_support"].to_numpy(dtype=np.float64),
            tier_df[value_col].to_numpy(dtype=np.float64) - tier_df[err_col].to_numpy(dtype=np.float64),
            tier_df[value_col].to_numpy(dtype=np.float64) + tier_df[err_col].to_numpy(dtype=np.float64),
            color=COLOR_TIER[int(tier) - 1],
            alpha=0.12,
        )
    ax.set_xlabel("Mean ux on atomic patch")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def plot_panel_dose_law_fit_summary(fit_summary_df: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.6), sharex=True)
    if fit_summary_df.empty:
        for ax in axes:
            ax.axis("off")
        return fig
    metrics = fit_summary_df.copy()
    metrics["vector_rmse"] = metrics["fit_quality"].apply(lambda s: float(json.loads(s).get("vector_rmse", np.nan)))
    metrics["initial_direction_slope"] = metrics["fit_quality"].apply(lambda s: float(json.loads(s).get("initial_direction_slope_deg_per_ux", np.nan)))
    metrics["peak_magnitude"] = metrics["fit_quality"].apply(lambda s: float(json.loads(s).get("peak_magnitude", np.nan)))
    x = metrics["importance_tier"].to_numpy(dtype=np.float64)
    axes[0].bar(x, metrics["initial_direction_slope"].to_numpy(dtype=np.float64), color=list(COLOR_TIER))
    axes[0].set_ylabel("Initial direction slope")
    axes[1].bar(x, metrics["peak_magnitude"].to_numpy(dtype=np.float64), color=list(COLOR_TIER))
    axes[1].set_ylabel("Peak magnitude")
    axes[2].bar(x, metrics["vector_rmse"].to_numpy(dtype=np.float64), color=list(COLOR_TIER))
    axes[2].set_ylabel("Vector RMSE")
    for ax in axes:
        ax.set_xlabel("Importance tier")
        ax.set_xticks(x)
        ax.grid(alpha=GRID_ALPHA, axis="y")
    fig.tight_layout()
    return fig


def plot_panel_support_composition(composition_summary_df: pd.DataFrame, *, x_column: str, x_label: str) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    if composition_summary_df.empty:
        ax.set_xlabel(x_label)
        ax.set_ylabel("Atomic-patch equivalents")
        fig.tight_layout()
        return fig
    x = np.arange(len(composition_summary_df), dtype=np.float64)
    bottom = np.zeros(len(composition_summary_df), dtype=np.float64)
    for tier in range(1, DEFAULT_IMPORTANCE_TIERS + 1):
        values = composition_summary_df[f"c{tier}"].to_numpy(dtype=np.float64)
        ax.bar(x, values, bottom=bottom, color=COLOR_TIER[tier - 1], label=f"Tier {tier}")
        bottom += values
    labels = [str(v) for v in composition_summary_df[x_column].tolist()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35 if len(labels) > 6 else 0, ha="right" if len(labels) > 6 else "center")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Atomic-patch equivalents")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=GRID_ALPHA, axis="y")
    fig.tight_layout()
    return fig


def plot_panel_observed_vs_predicted(
    summary_df: pd.DataFrame,
    *,
    x_column: str,
    x_label: str,
    actual_col: str,
    predicted_col: str,
    actual_sem_col: str,
    predicted_sem_col: str,
    y_label: str,
) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    if summary_df.empty:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.tight_layout()
        return fig
    x = np.arange(len(summary_df), dtype=np.float64)
    ax.plot(x, summary_df[actual_col].to_numpy(dtype=np.float64), color=COLOR_DIRECTION, marker="o", linewidth=1.8, label="Actual")
    ax.plot(x, summary_df[predicted_col].to_numpy(dtype=np.float64), color=COLOR_DIRECTION_PREDICTED, marker="s", linewidth=1.8, label="Predicted")
    ax.fill_between(
        x,
        summary_df[actual_col].to_numpy(dtype=np.float64) - summary_df[actual_sem_col].to_numpy(dtype=np.float64),
        summary_df[actual_col].to_numpy(dtype=np.float64) + summary_df[actual_sem_col].to_numpy(dtype=np.float64),
        color=COLOR_DIRECTION,
        alpha=0.12,
    )
    ax.fill_between(
        x,
        summary_df[predicted_col].to_numpy(dtype=np.float64) - summary_df[predicted_sem_col].to_numpy(dtype=np.float64),
        summary_df[predicted_col].to_numpy(dtype=np.float64) + summary_df[predicted_sem_col].to_numpy(dtype=np.float64),
        color=COLOR_DIRECTION_PREDICTED,
        alpha=0.10,
    )
    labels = [str(v) for v in summary_df[x_column].tolist()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35 if len(labels) > 6 else 0, ha="right" if len(labels) > 6 else "center")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def plot_panel_delay_local_dose_evolution(delay_local_dose_df: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if delay_local_dose_df.empty:
        ax.set_xlabel("Delay (ms)")
        ax.set_ylabel("Effective local dose / mean ux")
        fig.tight_layout()
        return fig
    x = delay_local_dose_df["delay_ms"].to_numpy(dtype=np.float64)
    for tier in range(1, DEFAULT_IMPORTANCE_TIERS + 1):
        ax.plot(x, delay_local_dose_df[f"d{tier}"].to_numpy(dtype=np.float64), color=COLOR_TIER[tier - 1], marker="o", linewidth=1.8, label=f"Tier {tier}")
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Effective local dose / mean ux")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def summarize_fixed_support_vary_ux_dose(records_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid = records_df[(records_df["module_name"] == "ux_dose") & (records_df["pair_status"] == "ok")].copy()
    if valid.empty:
        return {
            "record_df": pd.DataFrame(
                columns=[
                    "probe_id",
                    "dose_label",
                    "dose_order",
                    "target_mean_ux_on_support",
                    "mean_ux_on_support",
                    "bias_magnitude",
                    "signed_direction_deg",
                ]
            ),
            "pooled_df": pd.DataFrame(
                columns=[
                    "dose_order",
                    "dose_label",
                    "target_mean_ux_on_support",
                    "mean_ux_on_support",
                    "sem_mean_ux_on_support",
                    "bias_magnitude",
                    "sem_bias_magnitude",
                    "signed_direction_deg",
                    "sem_signed_direction_deg",
                    "total_ux_support",
                    "sem_total_ux_support",
                    "n_records",
                ]
            ),
        }
    pooled = (
        valid.groupby(["dose_order", "dose_label"], sort=True)
        .agg(
            target_mean_ux_on_support=("target_mean_ux_on_support", "mean"),
            mean_ux_on_support=("mean_ux_on_support", "mean"),
            sem_mean_ux_on_support=("mean_ux_on_support", _sem),
            bias_magnitude=("bias_magnitude", "mean"),
            sem_bias_magnitude=("bias_magnitude", _sem),
            signed_direction_deg=("signed_direction_deg", "mean"),
            sem_signed_direction_deg=("signed_direction_deg", _sem),
            total_ux_support=("total_ux_support", "mean"),
            sem_total_ux_support=("total_ux_support", _sem),
            n_records=("record_id", "count"),
        )
        .reset_index()
        .sort_values("dose_order", kind="stable")
        .reset_index(drop=True)
    )
    return {"record_df": valid, "pooled_df": pooled}


def summarize_fixed_location_vary_area(records_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid = records_df[(records_df["module_name"] == "support_area") & (records_df["pair_status"] == "ok")].copy()
    if valid.empty:
        return {
            "record_df": pd.DataFrame(
                columns=[
                    "probe_id",
                    "area_index",
                    "area_label",
                    "support_area",
                    "mean_ux_on_support",
                    "bias_magnitude",
                    "signed_direction_deg",
                ]
            ),
            "pooled_df": pd.DataFrame(
                columns=[
                    "area_index",
                    "area_label",
                    "support_area",
                    "mean_ux_on_support",
                    "sem_mean_ux_on_support",
                    "bias_magnitude",
                    "sem_bias_magnitude",
                    "signed_direction_deg",
                    "sem_signed_direction_deg",
                    "total_ux_support",
                    "sem_total_ux_support",
                    "n_records",
                ]
            ),
        }
    pooled = (
        valid.groupby(["area_index", "area_label", "support_area"], sort=True)
        .agg(
            mean_ux_on_support=("mean_ux_on_support", "mean"),
            sem_mean_ux_on_support=("mean_ux_on_support", _sem),
            bias_magnitude=("bias_magnitude", "mean"),
            sem_bias_magnitude=("bias_magnitude", _sem),
            signed_direction_deg=("signed_direction_deg", "mean"),
            sem_signed_direction_deg=("signed_direction_deg", _sem),
            total_ux_support=("total_ux_support", "mean"),
            sem_total_ux_support=("total_ux_support", _sem),
            n_records=("record_id", "count"),
        )
        .reset_index()
        .sort_values("area_index", kind="stable")
        .reset_index(drop=True)
    )
    return {"record_df": valid, "pooled_df": pooled}


def summarize_fixed_support_vary_location(records_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid = records_df[(records_df["module_name"] == "support_location") & (records_df["pair_status"] == "ok")].copy()
    if valid.empty:
        return {
            "record_df": pd.DataFrame(
                columns=[
                    "probe_id",
                    "location_index",
                    "location_label",
                    "bias_magnitude",
                    "mean_ux_on_support",
                    "signed_direction_deg",
                ]
            ),
            "pooled_df": pd.DataFrame(
                columns=[
                    "location_index",
                    "location_label",
                    "signed_direction_deg",
                    "sem_signed_direction_deg",
                    "bias_magnitude",
                    "sem_bias_magnitude",
                    "mean_ux_on_support",
                    "sem_mean_ux_on_support",
                    "n_records",
                ]
            ),
        }
    pooled = (
        valid.groupby(["location_index", "location_label"], sort=True)
        .agg(
            signed_direction_deg=("signed_direction_deg", "mean"),
            sem_signed_direction_deg=("signed_direction_deg", _sem),
            bias_magnitude=("bias_magnitude", "mean"),
            sem_bias_magnitude=("bias_magnitude", _sem),
            mean_ux_on_support=("mean_ux_on_support", "mean"),
            sem_mean_ux_on_support=("mean_ux_on_support", _sem),
            n_records=("record_id", "count"),
        )
        .reset_index()
        .sort_values("location_index", kind="stable")
        .reset_index(drop=True)
    )
    return {"record_df": valid, "pooled_df": pooled}


def summarize_fixed_support_vary_delay(records_df: pd.DataFrame, *, module_name: str) -> dict[str, pd.DataFrame]:
    valid = records_df[(records_df["module_name"] == module_name) & (records_df["pair_status"] == "ok")].copy()
    if valid.empty:
        return {
            "record_df": pd.DataFrame(
                columns=[
                    "probe_id",
                    "delay_ms",
                    "mean_ux_on_support",
                    "bias_magnitude",
                    "signed_direction_deg",
                ]
            ),
            "pooled_df": pd.DataFrame(
                columns=[
                    "delay_ms",
                    "mean_ux_on_support",
                    "sem_mean_ux_on_support",
                    "bias_magnitude",
                    "sem_bias_magnitude",
                    "signed_direction_deg",
                    "sem_signed_direction_deg",
                    "total_ux_support",
                    "sem_total_ux_support",
                    "n_records",
                ]
            ),
        }
    pooled = (
        valid.groupby("delay_ms", sort=True)
        .agg(
            mean_ux_on_support=("mean_ux_on_support", "mean"),
            sem_mean_ux_on_support=("mean_ux_on_support", _sem),
            bias_magnitude=("bias_magnitude", "mean"),
            sem_bias_magnitude=("bias_magnitude", _sem),
            signed_direction_deg=("signed_direction_deg", "mean"),
            sem_signed_direction_deg=("signed_direction_deg", _sem),
            total_ux_support=("total_ux_support", "mean"),
            sem_total_ux_support=("total_ux_support", _sem),
            n_records=("record_id", "count"),
        )
        .reset_index()
        .sort_values("delay_ms", kind="stable")
        .reset_index(drop=True)
    )
    return {"record_df": valid, "pooled_df": pooled}


def extract_probe_specific_dose_mapping_inputs(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame(
            columns=[
                "probe_id",
                "support_id",
                "support_role",
                "dose_label",
                "dose_order",
                "mean_ux_on_support",
                "signed_direction_deg",
                "bias_magnitude",
                "pair_status",
            ]
        )
    return (
        records_df[
            (records_df["module_name"] == "ux_dose")
            & (records_df["support_role"] == SUPPORT_ROLE_CANONICAL)
            & (records_df["pair_status"] == "ok")
        ]
        .copy()
        .sort_values(["probe_id", "dose_order"], kind="stable")
        .reset_index(drop=True)
    )


def extract_probe_specific_delay_bridge_records(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame(
            columns=[
                "probe_id",
                "support_id",
                "support_role",
                "delay_ms",
                "mean_ux_on_support",
                "signed_direction_deg",
                "bias_magnitude",
                "pair_status",
            ]
        )
    return (
        records_df[
            (records_df["module_name"] == "support_delay")
            & (records_df["support_role"] == SUPPORT_ROLE_CANONICAL)
        ]
        .copy()
        .sort_values(["probe_id", "delay_ms"], kind="stable")
        .reset_index(drop=True)
    )


def fit_probe_specific_dose_to_direction_mapping(
    dose_record_df: pd.DataFrame,
    *,
    min_anchor_points: int = DEFAULT_BRIDGE_MIN_DOSE_POINTS,
) -> dict[str, object]:
    mapping_columns = [
        "probe_id",
        "support_id",
        "mapping_method",
        "n_anchor_points",
        "ux_anchor_json",
        "direction_anchor_json",
        "mapping_status",
        "mapping_reason",
        "support_role",
        "dose_record_count",
    ]
    if dose_record_df.empty:
        return {
            "mapping_df": pd.DataFrame(columns=mapping_columns),
            "summary": {
                "mapping_method": "probe_specific_piecewise_linear_interpolation_with_boundary_clipping",
                "n_total_probes": 0,
                "n_valid_mappings": 0,
                "n_invalid_mappings": 0,
                "min_anchor_points": int(min_anchor_points),
                "invalid_reason_counts": {},
            },
        }
    rows: list[dict[str, object]] = []
    invalid_reason_counts: dict[str, int] = {}
    for (probe_id, support_id), probe_df in dose_record_df.groupby(["probe_id", "support_id"], sort=True):
        valid = probe_df[
            np.isfinite(probe_df["mean_ux_on_support"]) & np.isfinite(probe_df["signed_direction_deg"])
        ].copy()
        valid = valid.sort_values(["mean_ux_on_support", "dose_order"], kind="stable").reset_index(drop=True)
        mapping_reason = "ok"
        if valid.empty:
            mapping_reason = "no_finite_dose_points"
            anchor_df = pd.DataFrame(columns=["mean_ux_on_support", "signed_direction_deg"])
        else:
            anchor_df = (
                valid.groupby("mean_ux_on_support", sort=True, as_index=False)
                .agg(
                    signed_direction_deg=("signed_direction_deg", "mean"),
                )
                .sort_values("mean_ux_on_support", kind="stable")
                .reset_index(drop=True)
            )
            if len(anchor_df) < int(min_anchor_points):
                mapping_reason = "insufficient_dose_anchor_points"
        mapping_status = "ok" if mapping_reason == "ok" else "invalid"
        if mapping_status != "ok":
            invalid_reason_counts[mapping_reason] = invalid_reason_counts.get(mapping_reason, 0) + 1
        rows.append(
            {
                "probe_id": int(probe_id),
                "support_id": str(support_id),
                "mapping_method": "probe_specific_piecewise_linear_interpolation_with_boundary_clipping",
                "n_anchor_points": int(len(anchor_df)),
                "ux_anchor_json": json.dumps(anchor_df["mean_ux_on_support"].astype(float).tolist(), ensure_ascii=False),
                "direction_anchor_json": json.dumps(anchor_df["signed_direction_deg"].astype(float).tolist(), ensure_ascii=False),
                "mapping_status": str(mapping_status),
                "mapping_reason": str(mapping_reason),
                "support_role": str(probe_df["support_role"].iloc[0]),
                "dose_record_count": int(len(valid)),
            }
        )
    mapping_df = pd.DataFrame(rows, columns=mapping_columns).sort_values(["probe_id"], kind="stable").reset_index(drop=True)
    return {
        "mapping_df": mapping_df,
        "summary": {
            "mapping_method": "probe_specific_piecewise_linear_interpolation_with_boundary_clipping",
            "n_total_probes": int(mapping_df["probe_id"].nunique()),
            "n_valid_mappings": int((mapping_df["mapping_status"] == "ok").sum()),
            "n_invalid_mappings": int((mapping_df["mapping_status"] != "ok").sum()),
            "min_anchor_points": int(min_anchor_points),
            "invalid_reason_counts": invalid_reason_counts,
        },
    }


def predict_probe_specific_delay_direction(
    delay_record_df: pd.DataFrame,
    probe_specific_mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "probe_id",
        "support_id",
        "delay_ms",
        "mean_ux_on_support",
        "actual_signed_direction_deg",
        "predicted_signed_direction_deg",
        "direction_prediction_error_deg",
        "absolute_direction_prediction_error_deg",
        "bias_magnitude",
        "pair_status",
        "bridge_prediction_status",
    ]
    if delay_record_df.empty:
        return pd.DataFrame(columns=columns)
    mapping_lookup: dict[tuple[int, str], dict[str, object]] = {}
    if not probe_specific_mapping_df.empty:
        for row in probe_specific_mapping_df.itertuples(index=False):
            mapping_lookup[(int(row.probe_id), str(row.support_id))] = {
                "mapping_status": str(row.mapping_status),
                "mapping_reason": str(row.mapping_reason),
                "ux_anchors": np.asarray(json.loads(str(row.ux_anchor_json)), dtype=np.float64),
                "direction_anchors": np.asarray(json.loads(str(row.direction_anchor_json)), dtype=np.float64),
            }
    output_rows: list[dict[str, object]] = []
    for row in delay_record_df.itertuples(index=False):
        mapping = mapping_lookup.get((int(row.probe_id), str(row.support_id)))
        actual = float(row.signed_direction_deg) if np.isfinite(row.signed_direction_deg) else float("nan")
        predicted = float("nan")
        prediction_status = "missing_probe_mapping"
        if str(row.pair_status) != "ok":
            prediction_status = "invalid_delay_record"
        elif mapping is None:
            prediction_status = "missing_probe_mapping"
        elif str(mapping["mapping_status"]) != "ok":
            prediction_status = str(mapping["mapping_reason"])
        elif not np.isfinite(float(row.mean_ux_on_support)):
            prediction_status = "invalid_delay_mean_ux"
        else:
            ux_anchors = np.asarray(mapping["ux_anchors"], dtype=np.float64)
            direction_anchors = np.asarray(mapping["direction_anchors"], dtype=np.float64)
            if ux_anchors.size < 1 or direction_anchors.size != ux_anchors.size:
                prediction_status = "invalid_probe_mapping"
            elif ux_anchors.size == 1:
                predicted = float(direction_anchors[0])
                prediction_status = "ok"
            else:
                predicted = float(
                    np.interp(
                        float(row.mean_ux_on_support),
                        ux_anchors,
                        direction_anchors,
                        left=float(direction_anchors[0]),
                        right=float(direction_anchors[-1]),
                    )
                )
                prediction_status = "ok"
        error = actual - predicted if np.isfinite(actual) and np.isfinite(predicted) else float("nan")
        output_rows.append(
            {
                "probe_id": int(row.probe_id),
                "support_id": str(row.support_id),
                "delay_ms": float(row.delay_ms),
                "mean_ux_on_support": float(row.mean_ux_on_support),
                "actual_signed_direction_deg": float(actual),
                "predicted_signed_direction_deg": float(predicted),
                "direction_prediction_error_deg": float(error),
                "absolute_direction_prediction_error_deg": float(abs(error)) if np.isfinite(error) else float("nan"),
                "bias_magnitude": float(row.bias_magnitude),
                "pair_status": str(row.pair_status),
                "bridge_prediction_status": str(prediction_status),
            }
        )
    return pd.DataFrame(output_rows, columns=columns).sort_values(["probe_id", "delay_ms"], kind="stable").reset_index(drop=True)


def evaluate_within_probe_delay_bridge(
    bridge_record_df: pd.DataFrame,
    *,
    min_delay_points: int = DEFAULT_BRIDGE_MIN_DELAY_POINTS,
) -> dict[str, object]:
    metric_columns = [
        "probe_id",
        "support_id",
        "n_delay_points",
        "bridge_status",
        "bridge_reason",
        "mae_deg",
        "rmse_deg",
        "corr",
        "mean_actual_direction_deg",
        "mean_predicted_direction_deg",
    ]
    pooled_columns = [
        "n_total_probes",
        "n_valid_probes",
        "n_excluded_probes",
        "mean_mae_deg",
        "median_mae_deg",
        "mean_rmse_deg",
        "median_rmse_deg",
        "mean_corr",
        "median_corr",
        "min_delay_points",
        "excluded_reason_counts_json",
    ]
    if bridge_record_df.empty:
        return {
            "probe_fit_metrics_df": pd.DataFrame(columns=metric_columns),
            "pooled_summary_df": pd.DataFrame([{col: 0 if col.startswith("n_") else (int(min_delay_points) if col == "min_delay_points" else json.dumps({}) if col == "excluded_reason_counts_json" else float("nan")) for col in pooled_columns}]),
            "summary": {
                "n_total_probes": 0,
                "n_valid_probes": 0,
                "n_excluded_probes": 0,
                "min_delay_points": int(min_delay_points),
                "excluded_reason_counts": {},
            },
        }
    rows: list[dict[str, object]] = []
    excluded_reason_counts: dict[str, int] = {}
    for (probe_id, support_id), probe_df in bridge_record_df.groupby(["probe_id", "support_id"], sort=True):
        valid = probe_df[
            (probe_df["pair_status"] == "ok")
            & (probe_df["bridge_prediction_status"] == "ok")
            & np.isfinite(probe_df["actual_signed_direction_deg"])
            & np.isfinite(probe_df["predicted_signed_direction_deg"])
        ].copy()
        bridge_reason = "ok"
        if len(valid) < int(min_delay_points):
            bridge_reason = "insufficient_delay_points"
        bridge_status = "ok" if bridge_reason == "ok" else "invalid"
        actual = valid["actual_signed_direction_deg"].to_numpy(dtype=np.float64)
        predicted = valid["predicted_signed_direction_deg"].to_numpy(dtype=np.float64)
        corr = float("nan")
        if bridge_status == "ok" and len(valid) >= 2:
            if np.std(actual, ddof=1) > 0.0 and np.std(predicted, ddof=1) > 0.0:
                corr = float(np.corrcoef(actual, predicted)[0, 1])
        mae = float(np.mean(np.abs(actual - predicted))) if bridge_status == "ok" else float("nan")
        rmse = float(np.sqrt(np.mean(np.square(actual - predicted)))) if bridge_status == "ok" else float("nan")
        if bridge_status != "ok":
            excluded_reason_counts[bridge_reason] = excluded_reason_counts.get(bridge_reason, 0) + 1
        rows.append(
            {
                "probe_id": int(probe_id),
                "support_id": str(support_id),
                "n_delay_points": int(len(valid)),
                "bridge_status": str(bridge_status),
                "bridge_reason": str(bridge_reason),
                "mae_deg": float(mae),
                "rmse_deg": float(rmse),
                "corr": float(corr),
                "mean_actual_direction_deg": float(np.mean(actual)) if bridge_status == "ok" else float("nan"),
                "mean_predicted_direction_deg": float(np.mean(predicted)) if bridge_status == "ok" else float("nan"),
            }
        )
    probe_fit_metrics_df = pd.DataFrame(rows, columns=metric_columns).sort_values(["probe_id"], kind="stable").reset_index(drop=True)
    valid_metrics = probe_fit_metrics_df[probe_fit_metrics_df["bridge_status"] == "ok"].copy()
    pooled_row = {
        "n_total_probes": int(len(probe_fit_metrics_df)),
        "n_valid_probes": int(len(valid_metrics)),
        "n_excluded_probes": int(len(probe_fit_metrics_df) - len(valid_metrics)),
        "mean_mae_deg": float(valid_metrics["mae_deg"].mean()) if not valid_metrics.empty else float("nan"),
        "median_mae_deg": float(valid_metrics["mae_deg"].median()) if not valid_metrics.empty else float("nan"),
        "mean_rmse_deg": float(valid_metrics["rmse_deg"].mean()) if not valid_metrics.empty else float("nan"),
        "median_rmse_deg": float(valid_metrics["rmse_deg"].median()) if not valid_metrics.empty else float("nan"),
        "mean_corr": float(valid_metrics["corr"].dropna().mean()) if not valid_metrics["corr"].dropna().empty else float("nan"),
        "median_corr": float(valid_metrics["corr"].dropna().median()) if not valid_metrics["corr"].dropna().empty else float("nan"),
        "min_delay_points": int(min_delay_points),
        "excluded_reason_counts_json": json.dumps(excluded_reason_counts, ensure_ascii=False),
    }
    return {
        "probe_fit_metrics_df": probe_fit_metrics_df,
        "pooled_summary_df": pd.DataFrame([pooled_row], columns=pooled_columns),
        "summary": {
            "n_total_probes": int(pooled_row["n_total_probes"]),
            "n_valid_probes": int(pooled_row["n_valid_probes"]),
            "n_excluded_probes": int(pooled_row["n_excluded_probes"]),
            "min_delay_points": int(min_delay_points),
            "excluded_reason_counts": excluded_reason_counts,
            "mean_mae_deg": pooled_row["mean_mae_deg"],
            "median_mae_deg": pooled_row["median_mae_deg"],
            "mean_rmse_deg": pooled_row["mean_rmse_deg"],
            "median_rmse_deg": pooled_row["median_rmse_deg"],
            "mean_corr": pooled_row["mean_corr"],
            "median_corr": pooled_row["median_corr"],
        },
    }


def summarize_delay_dose_bridge(
    dose_record_df: pd.DataFrame,
    delay_record_df: pd.DataFrame,
    *,
    min_dose_points: int = DEFAULT_BRIDGE_MIN_DOSE_POINTS,
    min_delay_points: int = DEFAULT_BRIDGE_MIN_DELAY_POINTS,
) -> dict[str, object]:
    dose_bridge_records = extract_probe_specific_dose_mapping_inputs(dose_record_df)
    delay_bridge_inputs = extract_probe_specific_delay_bridge_records(delay_record_df)
    mapping_payload = fit_probe_specific_dose_to_direction_mapping(
        dose_bridge_records,
        min_anchor_points=int(min_dose_points),
    )
    bridge_records = predict_probe_specific_delay_direction(delay_bridge_inputs, mapping_payload["mapping_df"])
    evaluation = evaluate_within_probe_delay_bridge(
        bridge_records,
        min_delay_points=int(min_delay_points),
    )
    return {
        "dose_bridge_record_df": dose_bridge_records,
        "delay_bridge_input_df": delay_bridge_inputs,
        "mapping_df": mapping_payload["mapping_df"],
        "record_df": bridge_records,
        "probe_fit_metrics_df": evaluation["probe_fit_metrics_df"],
        "pooled_summary_df": evaluation["pooled_summary_df"],
        "fit_summary": {
            "bridge_scope": "within_probe_within_canonical_support_only",
            "mapping_method": "probe_specific_piecewise_linear_interpolation_with_boundary_clipping",
            "min_dose_points": int(min_dose_points),
            "min_delay_points": int(min_delay_points),
            "mapping_summary": mapping_payload["summary"],
            "evaluation_summary": evaluation["summary"],
        },
    }


def plot_panel_a_support_definition_examples(probe_df: pd.DataFrame, support_df: pd.DataFrame) -> tuple[plt.Figure, dict[str, object]]:
    candidate = support_df[support_df["support_role"] == SUPPORT_ROLE_CANONICAL].copy()
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    if candidate.empty:
        ax.axis("off")
        return fig, {"status": "no_canonical_support"}
    row = candidate.sort_values(["probe_id"], kind="stable").iloc[0]
    probe_image = probe_df[probe_df["probe_id"] == int(row["probe_id"])].iloc[0]["probe_image"]
    support_mask = np.asarray(row["support_mask"], dtype=bool)
    arr = probe_image.detach().cpu().to(torch.float32).squeeze(0).numpy()
    ax.imshow(arr, cmap="gray", vmin=0.0, vmax=max(1.0, float(arr.max())))
    ax.contour(support_mask.astype(np.float32), levels=[0.5], colors=[COLOR_SUPPORT], linewidths=1.6)
    center = _mask_center(support_mask)
    if np.isfinite(center[1]) and np.isfinite(center[0]):
        ax.scatter([center[1]], [center[0]], s=28, color=COLOR_SUPPORT, label="Support center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Support-defined synthetic sample")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    return fig, {"status": "ok", "probe_id": int(row["probe_id"]), "support_id": str(row["support_id"])}


def _dual_axis_condition_panel(
    pooled_df: pd.DataFrame,
    *,
    x_column: str,
    x_labels: Sequence[str],
    xlabel: str,
    xtick_rotation: float = 0.0,
    figsize: tuple[float, float] = (5.2, 4.0),
) -> plt.Figure:
    apply_publication_style()
    fig, ax_mag = plt.subplots(figsize=figsize)
    ax_dir = ax_mag.twinx()
    if pooled_df.empty:
        ax_mag.set_xlabel(xlabel)
        ax_mag.set_ylabel("Bias magnitude", color=COLOR_MAGNITUDE)
        ax_dir.set_ylabel("Signed direction (deg)", color=COLOR_DIRECTION)
        fig.tight_layout()
        return fig
    x = pooled_df[x_column].to_numpy(dtype=np.float64)
    ax_mag.plot(x, pooled_df["bias_magnitude"].to_numpy(dtype=np.float64), color=COLOR_MAGNITUDE, marker="o", linewidth=1.8)
    ax_mag.fill_between(
        x,
        pooled_df["bias_magnitude"].to_numpy(dtype=np.float64) - pooled_df["sem_bias_magnitude"].to_numpy(dtype=np.float64),
        pooled_df["bias_magnitude"].to_numpy(dtype=np.float64) + pooled_df["sem_bias_magnitude"].to_numpy(dtype=np.float64),
        color=COLOR_MAGNITUDE,
        alpha=0.14,
    )
    ax_dir.plot(
        x,
        pooled_df["signed_direction_deg"].to_numpy(dtype=np.float64),
        color=COLOR_DIRECTION,
        marker="o",
        linewidth=1.8,
    )
    ax_dir.fill_between(
        x,
        pooled_df["signed_direction_deg"].to_numpy(dtype=np.float64) - pooled_df["sem_signed_direction_deg"].to_numpy(dtype=np.float64),
        pooled_df["signed_direction_deg"].to_numpy(dtype=np.float64) + pooled_df["sem_signed_direction_deg"].to_numpy(dtype=np.float64),
        color=COLOR_DIRECTION,
        alpha=0.12,
    )
    ax_mag.set_xticks(x)
    ax_mag.set_xticklabels(
        list(x_labels),
        rotation=float(xtick_rotation),
        ha="right" if float(xtick_rotation) else "center",
    )
    ax_mag.set_xlabel(xlabel)
    ax_mag.set_ylabel("Bias magnitude", color=COLOR_MAGNITUDE)
    ax_dir.set_ylabel("Signed direction (deg)", color=COLOR_DIRECTION)
    ax_mag.tick_params(axis="y", colors=COLOR_MAGNITUDE)
    ax_dir.tick_params(axis="y", colors=COLOR_DIRECTION)
    ax_mag.set_ylim(bottom=0.0)
    ax_mag.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def plot_panel_b_ux_dose_manipulation_check(record_df: pd.DataFrame, pooled_df: pd.DataFrame) -> plt.Figure:
    del record_df
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    if pooled_df.empty:
        ax.set_xlabel("Target mean ux on support")
        ax.set_ylabel("Measured mean ux on support")
        fig.tight_layout()
        return fig
    x = pooled_df["target_mean_ux_on_support"].to_numpy(dtype=np.float64)
    ax.plot(x, pooled_df["mean_ux_on_support"].to_numpy(dtype=np.float64), color=COLOR_SUPPORT, marker="o", linewidth=1.8)
    ax.fill_between(
        x,
        pooled_df["mean_ux_on_support"].to_numpy(dtype=np.float64) - pooled_df["sem_mean_ux_on_support"].to_numpy(dtype=np.float64),
        pooled_df["mean_ux_on_support"].to_numpy(dtype=np.float64) + pooled_df["sem_mean_ux_on_support"].to_numpy(dtype=np.float64),
        color=COLOR_SUPPORT,
        alpha=0.14,
    )
    ax.set_xlabel("Target mean ux on support")
    ax.set_ylabel("Measured mean ux on support")
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def plot_panel_g_delay_ux_decay(record_df: pd.DataFrame, pooled_df: pd.DataFrame) -> plt.Figure:
    del record_df
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    x = pooled_df["delay_ms"].to_numpy(dtype=np.float64)
    ax.plot(x, pooled_df["mean_ux_on_support"].to_numpy(dtype=np.float64), color=COLOR_SUPPORT, marker="o", linewidth=1.8)
    ax.fill_between(
        x,
        pooled_df["mean_ux_on_support"].to_numpy(dtype=np.float64) - pooled_df["sem_mean_ux_on_support"].to_numpy(dtype=np.float64),
        pooled_df["mean_ux_on_support"].to_numpy(dtype=np.float64) + pooled_df["sem_mean_ux_on_support"].to_numpy(dtype=np.float64),
        color=COLOR_SUPPORT,
        alpha=0.14,
    )
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Mean ux on support")
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def plot_panel_dose_magnitude_and_direction(pooled_df: pd.DataFrame) -> plt.Figure:
    return _dual_axis_condition_panel(
        pooled_df,
        x_column="target_mean_ux_on_support",
        x_labels=pooled_df["dose_label"].tolist(),
        xlabel="Target mean ux on support",
        xtick_rotation=45.0,
        figsize=(6.2, 4.0),
    )


def plot_panel_area_magnitude_and_direction(pooled_df: pd.DataFrame) -> plt.Figure:
    return _dual_axis_condition_panel(
        pooled_df,
        x_column="support_area",
        x_labels=[f"{int(v)}" for v in pooled_df["support_area"].tolist()],
        xlabel="Support area",
    )


def plot_panel_location_magnitude_and_direction(pooled_df: pd.DataFrame) -> plt.Figure:
    return _dual_axis_condition_panel(
        pooled_df,
        x_column="location_index",
        x_labels=pooled_df["location_label"].tolist(),
        xlabel="Support location",
        xtick_rotation=35.0,
        figsize=(5.8, 4.0),
    )


def plot_panel_delay_magnitude_and_direction(pooled_df: pd.DataFrame) -> plt.Figure:
    labels = [f"{float(v):.0f}" for v in pooled_df["delay_ms"].tolist()]
    return _dual_axis_condition_panel(
        pooled_df,
        x_column="delay_ms",
        x_labels=labels,
        xlabel="Delay (ms)",
    )


def plot_panel_delay_to_dose_bridge_fit_quality(probe_fit_metrics_df: pd.DataFrame, pooled_summary_df: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if probe_fit_metrics_df.empty:
        ax.set_xlabel("Valid probes")
        ax.set_ylabel("Bridge MAE (deg)")
        fig.tight_layout()
        return fig
    valid = probe_fit_metrics_df[probe_fit_metrics_df["bridge_status"] == "ok"].copy()
    if valid.empty:
        ax.set_xlabel("Valid probes")
        ax.set_ylabel("Bridge MAE (deg)")
        fig.tight_layout()
        return fig
    valid = valid.sort_values(["mae_deg", "probe_id"], kind="stable").reset_index(drop=True)
    x = np.arange(len(valid), dtype=np.float64)
    ax.plot(x, valid["mae_deg"].to_numpy(dtype=np.float64), color=COLOR_DIRECTION, marker="o", linewidth=1.6)
    pooled_mean = float(pooled_summary_df["mean_mae_deg"].iloc[0]) if not pooled_summary_df.empty else float("nan")
    pooled_sem = _sem(valid["mae_deg"].to_numpy(dtype=np.float64))
    if np.isfinite(pooled_mean):
        ax.axhline(pooled_mean, color=COLOR_DIRECTION_PREDICTED, linewidth=1.6, linestyle="--")
        ax.fill_between(
            [x.min() if len(x) else 0.0, x.max() if len(x) else 0.0],
            pooled_mean - pooled_sem,
            pooled_mean + pooled_sem,
            color=COLOR_DIRECTION_PREDICTED,
            alpha=0.10,
        )
    xticklabels = [str(int(v)) for v in valid["probe_id"].tolist()]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=90 if len(xticklabels) > 8 else 45, ha="right")
    ax.set_xlabel("Valid probes")
    ax.set_ylabel("Bridge MAE (deg)")
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def _records_for_csv(records_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "record_id",
        "module_name",
        "probe_id",
        "probe_label",
        "support_id",
        "support_role",
        "support_area",
        "support_center_row",
        "support_center_col",
        "area_index",
        "area_label",
        "location_index",
        "location_label",
        "dose_label",
        "dose_order",
        "dose_scale",
        "delay_ms",
        "condition_name",
        "condition_label",
        "condition_order",
        "intervention_type",
        "target_mean_ux_on_support",
        "applied_scale",
        "mean_ux_on_support_pre",
        "mean_ux_on_support",
        "total_ux_support_pre",
        "total_ux_support",
        "bias_magnitude",
        "bias_magnitude_per_pixel",
        "signed_direction_deg",
        "pair_status",
    ]
    if records_df.empty:
        return pd.DataFrame(columns=columns)
    return records_df[columns].copy()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sample-ms", type=float, default=DEFAULT_SAMPLE_MS)
    parser.add_argument("--delay-ms", type=float, default=DEFAULT_DELAY_MS)
    parser.add_argument("--delay-sweep-ms", type=float, nargs="+", default=list(DEFAULT_DELAY_SWEEP_MS))
    parser.add_argument("--probe-ms", type=float, default=DEFAULT_PROBE_MS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--probe-selection-per-class", type=int, default=DEFAULT_PROBE_SELECTION_PER_CLASS)
    parser.add_argument("--foreground-threshold", type=float, default=DEFAULT_FOREGROUND_THRESHOLD)
    parser.add_argument("--fixed-support-area-pixels", type=int, default=DEFAULT_FIXED_SUPPORT_AREA_PIXELS)
    parser.add_argument("--support-area-levels", type=int, nargs="+", default=list(DEFAULT_SUPPORT_AREA_LEVELS))
    parser.add_argument("--num-location-masks", type=int, default=DEFAULT_NUM_LOCATION_MASKS)
    parser.add_argument("--importance-tiers", type=int, default=DEFAULT_IMPORTANCE_TIERS)
    parser.add_argument("--atomic-patch-area", type=int, default=DEFAULT_ATOMIC_PATCH_AREA)
    parser.add_argument("--importance-smoothing-alpha", type=float, default=DEFAULT_IMPORTANCE_SMOOTHING_ALPHA)
    parser.add_argument("--importance-smoothing-passes", type=int, default=DEFAULT_IMPORTANCE_SMOOTHING_PASSES)
    parser.add_argument("--mask-generation-seed", type=int, default=DEFAULT_MASK_GENERATION_SEED)
    parser.add_argument("--save-case-count", type=int, default=DEFAULT_SAVE_CASE_COUNT)
    parser.add_argument("--ux-dose-targets", nargs="+", type=float, default=list(DEFAULT_UX_DOSE_TARGETS))
    parser.add_argument("--bridge-min-dose-points", type=int, default=DEFAULT_BRIDGE_MIN_DOSE_POINTS)
    parser.add_argument("--bridge-min-delay-points", type=int, default=DEFAULT_BRIDGE_MIN_DELAY_POINTS)
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    _validate_positive("--sample-ms", float(args.sample_ms))
    _validate_positive("--delay-ms", float(args.delay_ms), allow_zero=True)
    _validate_positive("--probe-ms", float(args.probe_ms))
    _validate_positive("--batch-size", int(args.batch_size))
    _validate_positive("--max-probes", int(args.max_probes))
    _validate_positive("--fixed-support-area-pixels", int(args.fixed_support_area_pixels))
    _validate_positive("--num-location-masks", int(args.num_location_masks))
    _validate_positive("--importance-tiers", int(args.importance_tiers))
    _validate_positive("--atomic-patch-area", int(args.atomic_patch_area))
    _validate_positive("--importance-smoothing-passes", int(args.importance_smoothing_passes), allow_zero=True)
    _validate_positive("--bridge-min-dose-points", int(args.bridge_min_dose_points))
    _validate_positive("--bridge-min-delay-points", int(args.bridge_min_delay_points))
    support_area_levels = _sanitize_area_levels(args.support_area_levels)
    delay_sweep_ms = _sanitize_delay_sweep(args.delay_sweep_ms)
    if float(args.delay_ms) not in delay_sweep_ms:
        delay_sweep_ms = sorted(dict.fromkeys(delay_sweep_ms + [float(args.delay_ms)]))
    dose_levels = _sanitize_target_mean_ux_levels(args.ux_dose_targets)
    if int(args.importance_tiers) != int(DEFAULT_IMPORTANCE_TIERS):
        raise ValueError("This unified experiment requires exactly 5 importance tiers.")

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    spec = ExperimentSpec(dt=1.0 * MS, sample_ms=float(args.sample_ms), probe_ms=float(args.probe_ms))
    if spec.sample_steps <= 0 or spec.probe_steps <= 0:
        raise ValueError("sample/probe durations must resolve to positive steps.")

    layout = prepare_result_layout(args.output_dir)
    dataset = load_mnist_skeleton_dataset(args.dataset_root, args.split)
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
    support_df, generation_notes = build_support_metadata_table(
        probe_df,
        fixed_support_area_pixels=int(args.fixed_support_area_pixels),
        support_area_levels=support_area_levels,
        num_location_masks=int(args.num_location_masks),
        foreground_threshold=float(args.foreground_threshold),
        mask_generation_seed=int(args.mask_generation_seed),
        max_pairwise_iou=float(DEFAULT_MAX_LOCATION_MASK_IOU),
    )
    n_selected_probes_before_importance_filter = int(len(probe_df))

    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(float(args.sample_ms), float(args.probe_ms), max(delay_sweep_ms)),
    )
    probe_importance_payloads, patch_metadata_df, importance_notes = build_probe_importance_metadata(
        probe_df,
        encoder=encoder,
        foreground_threshold=float(args.foreground_threshold),
        atomic_patch_area=int(args.atomic_patch_area),
        importance_tiers=int(args.importance_tiers),
        importance_smoothing_alpha=float(args.importance_smoothing_alpha),
        importance_smoothing_passes=int(args.importance_smoothing_passes),
    )
    generation_notes.extend(importance_notes)
    valid_probe_counts = (
        patch_metadata_df.groupby("probe_id")["importance_tier"].nunique().reset_index(name="n_tiers")
        if not patch_metadata_df.empty
        else pd.DataFrame(columns=["probe_id", "n_tiers"])
    )
    valid_probe_ids = valid_probe_counts[valid_probe_counts["n_tiers"] == int(args.importance_tiers)]["probe_id"].astype(int).tolist()
    if not valid_probe_ids:
        raise RuntimeError("No probes produced all 5 importance tiers with fixed-area continuous atomic patches.")
    probe_df = probe_df[probe_df["probe_id"].isin(valid_probe_ids)].copy().reset_index(drop=True)
    support_df = support_df[support_df["probe_id"].isin(valid_probe_ids)].copy().reset_index(drop=True)
    patch_metadata_df = patch_metadata_df[patch_metadata_df["probe_id"].isin(valid_probe_ids)].copy().reset_index(drop=True)
    probe_importance_payloads = {int(probe_id): probe_importance_payloads[int(probe_id)] for probe_id in valid_probe_ids}
    generation_notes.append(
        {
            "stage": "importance_filter",
            "required_tiers": int(args.importance_tiers),
            "n_selected_probes_before_importance_filter": int(n_selected_probes_before_importance_filter),
            "n_valid_probes_after_importance_filter": int(len(valid_probe_ids)),
        }
    )
    readout_step = resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(spec.probe_steps),
        decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )

    importance_dose_plan = build_importance_patch_dose_plan(
        patch_metadata_df,
        probe_df,
        dose_levels=dose_levels,
        delay_ms=float(args.delay_ms),
    )
    area_plan = build_area_plan(support_df, probe_df, delay_ms=float(args.delay_ms))
    location_plan = build_location_natural_plan(support_df, probe_df, delay_ms=float(args.delay_ms))
    delay_plan = build_delay_natural_plan(support_df, probe_df, delay_sweep_ms=delay_sweep_ms)

    importance_dose_records = execute_support_plan(
        importance_dose_plan,
        net=net,
        encoder=encoder,
        device=device,
        readout_step=int(readout_step),
        batch_size=int(args.batch_size),
        spec=spec,
    )
    area_records = execute_support_plan(
        area_plan,
        net=net,
        encoder=encoder,
        device=device,
        readout_step=int(readout_step),
        batch_size=int(args.batch_size),
        spec=spec,
    )
    location_records = execute_support_plan(
        location_plan,
        net=net,
        encoder=encoder,
        device=device,
        readout_step=int(readout_step),
        batch_size=int(args.batch_size),
        spec=spec,
    )
    delay_records = execute_support_plan(
        delay_plan,
        net=net,
        encoder=encoder,
        device=device,
        readout_step=int(readout_step),
        batch_size=int(args.batch_size),
        spec=spec,
    )

    reference_source_frames = [
        frame
        for frame in (importance_dose_records, area_records, location_records, delay_records)
        if not frame.empty
    ]
    direction_reference = build_signed_direction_reference(
        pd.concat(reference_source_frames, axis=0, ignore_index=True)
        if reference_source_frames
        else pd.DataFrame()
    )
    importance_dose_records = annotate_signed_direction_records(importance_dose_records, reference=direction_reference)
    area_records = annotate_signed_direction_records(area_records, reference=direction_reference)
    location_records = annotate_signed_direction_records(location_records, reference=direction_reference)
    delay_records = annotate_signed_direction_records(delay_records, reference=direction_reference)

    dose_summary = summarize_importance_dose_records(importance_dose_records)
    dose_law_payload = fit_tierwise_local_dose_laws(
        dose_summary["record_df"],
        direction_reference=direction_reference,
        importance_tiers=int(args.importance_tiers),
    )
    location_composition_records = build_composition_prediction_records(
        location_records,
        module_name="support_location_natural",
        probe_importance_payloads=probe_importance_payloads,
        fit_lookup=dose_law_payload["fit_lookup"],
        direction_reference=direction_reference,
        atomic_patch_area=int(args.atomic_patch_area),
        importance_tiers=int(args.importance_tiers),
    )
    area_composition_records = build_composition_prediction_records(
        area_records,
        module_name="support_area",
        probe_importance_payloads=probe_importance_payloads,
        fit_lookup=dose_law_payload["fit_lookup"],
        direction_reference=direction_reference,
        atomic_patch_area=int(args.atomic_patch_area),
        importance_tiers=int(args.importance_tiers),
    )
    delay_composition_records = build_composition_prediction_records(
        delay_records,
        module_name="support_delay",
        probe_importance_payloads=probe_importance_payloads,
        fit_lookup=dose_law_payload["fit_lookup"],
        direction_reference=direction_reference,
        atomic_patch_area=int(args.atomic_patch_area),
        importance_tiers=int(args.importance_tiers),
    )
    location_composition_summary = summarize_composition_by_condition(
        location_composition_records,
        condition_columns=["location_index", "location_label"],
        importance_tiers=int(args.importance_tiers),
    )
    area_composition_summary = summarize_composition_by_condition(
        area_composition_records,
        condition_columns=["area_index", "area_label", "support_area"],
        importance_tiers=int(args.importance_tiers),
    )
    location_prediction_summary = summarize_prediction_by_condition(
        location_composition_records,
        condition_columns=["location_index", "location_label"],
    )
    area_prediction_summary = summarize_prediction_by_condition(
        area_composition_records,
        condition_columns=["area_index", "area_label", "support_area"],
    )
    delay_local_dose_summary = summarize_delay_local_dose_evolution(
        delay_composition_records,
        importance_tiers=int(args.importance_tiers),
    )
    delay_prediction_summary = summarize_prediction_by_condition(
        delay_composition_records,
        condition_columns=["delay_ms"],
    )

    importance_patch_metadata_csv = save_tidy_csv(
        patch_metadata_df[
            [
                "probe_id",
                "patch_id",
                "importance_tier",
                "support_area",
                "mean_importance",
                "support_center_row",
                "support_center_col",
                "support_coords_json",
            ]
        ].copy(),
        layout.data_file("importance_patch_metadata.csv"),
        sort_by=["probe_id", "importance_tier"],
    )
    dose_law_records_csv = save_tidy_csv(
        dose_summary["record_df"]
        .assign(
            dose=lambda df: df["target_mean_ux_on_support"],
            DeltaV_summary=lambda df: df["DeltaV"].apply(_json_dumps_float_list),
        )[
            [
                "probe_id",
                "patch_id",
                "importance_tier",
                "dose",
                "mean_ux_on_support",
                "bias_magnitude",
                "signed_direction_deg",
                "DeltaV_summary",
            ]
        ].copy(),
        layout.data_file("dose_law_records.csv"),
        sort_by=["probe_id", "importance_tier", "dose"],
    )
    dose_law_fit_summary_csv = save_tidy_csv(
        dose_law_payload["fit_summary_df"],
        layout.data_file("dose_law_fit_summary.csv"),
        sort_by=["importance_tier"],
    )
    location_composition_csv = save_tidy_csv(
        location_composition_records[
            [
                "probe_id",
                "support_id",
                "c1",
                "c2",
                "c3",
                "c4",
                "c5",
                "actual_bias_magnitude",
                "predicted_bias_magnitude",
                "actual_signed_direction_deg",
                "predicted_signed_direction_deg",
            ]
        ].copy(),
        layout.data_file("location_composition_records.csv"),
        sort_by=["probe_id", "support_id"],
    )
    area_composition_csv = save_tidy_csv(
        area_composition_records[
            [
                "probe_id",
                "support_id",
                "c1",
                "c2",
                "c3",
                "c4",
                "c5",
                "actual_bias_magnitude",
                "predicted_bias_magnitude",
                "actual_signed_direction_deg",
                "predicted_signed_direction_deg",
            ]
        ].copy(),
        layout.data_file("area_composition_records.csv"),
        sort_by=["probe_id", "support_id"],
    )
    delay_composition_csv = save_tidy_csv(
        delay_composition_records[
            [
                "probe_id",
                "delay_ms",
                "d1",
                "d2",
                "d3",
                "d4",
                "d5",
                "actual_bias_magnitude",
                "predicted_bias_magnitude",
                "actual_signed_direction_deg",
                "predicted_signed_direction_deg",
            ]
        ].copy(),
        layout.data_file("delay_composition_records.csv"),
        sort_by=["probe_id", "delay_ms"],
    )

    fig_paths: dict[str, object] = {}
    example_selection: dict[str, object] = {"status": "skipped"}
    if not bool(args.skip_figures):
        fig_a, panel_a_meta = plot_panel_a_importance_map_and_atomic_patches(
            probe_df,
            patch_metadata_df,
            probe_importance_payloads,
        )
        fig_paths[PANEL_FILENAMES["panel_a"]] = save_figure_all_formats(
            fig_a,
            layout.figure_base(PANEL_FILENAMES["panel_a"]),
        )
        plt.close(fig_a)
        fig_b = plot_panel_dose_curve_by_tier(
            dose_summary["pooled_df"],
            value_col="signed_direction_deg",
            ylabel="Signed direction (deg)",
        )
        fig_paths[PANEL_FILENAMES["panel_b"]] = save_figure_all_formats(
            fig_b,
            layout.figure_base(PANEL_FILENAMES["panel_b"]),
        )
        plt.close(fig_b)
        fig_c = plot_panel_dose_curve_by_tier(
            dose_summary["pooled_df"],
            value_col="bias_magnitude",
            ylabel="Bias magnitude",
        )
        fig_paths[PANEL_FILENAMES["panel_c"]] = save_figure_all_formats(
            fig_c,
            layout.figure_base(PANEL_FILENAMES["panel_c"]),
        )
        plt.close(fig_c)
        fig_d = plot_panel_dose_law_fit_summary(dose_law_payload["fit_summary_df"])
        fig_paths[PANEL_FILENAMES["panel_d"]] = save_figure_all_formats(
            fig_d,
            layout.figure_base(PANEL_FILENAMES["panel_d"]),
        )
        plt.close(fig_d)
        fig_e = plot_panel_support_composition(
            location_composition_summary,
            x_column="location_label",
            x_label="Location condition",
        )
        fig_paths[PANEL_FILENAMES["panel_e"]] = save_figure_all_formats(
            fig_e,
            layout.figure_base(PANEL_FILENAMES["panel_e"]),
        )
        plt.close(fig_e)
        fig_f = plot_panel_observed_vs_predicted(
            location_prediction_summary,
            x_column="location_label",
            x_label="Location condition",
            actual_col="actual_signed_direction_deg",
            predicted_col="predicted_signed_direction_deg",
            actual_sem_col="sem_actual_signed_direction_deg",
            predicted_sem_col="sem_predicted_signed_direction_deg",
            y_label="Signed direction (deg)",
        )
        fig_paths[PANEL_FILENAMES["panel_f"]] = save_figure_all_formats(
            fig_f,
            layout.figure_base(PANEL_FILENAMES["panel_f"]),
        )
        plt.close(fig_f)
        fig_g = plot_panel_observed_vs_predicted(
            location_prediction_summary,
            x_column="location_label",
            x_label="Location condition",
            actual_col="actual_bias_magnitude",
            predicted_col="predicted_bias_magnitude",
            actual_sem_col="sem_actual_bias_magnitude",
            predicted_sem_col="sem_predicted_bias_magnitude",
            y_label="Bias magnitude",
        )
        fig_paths[PANEL_FILENAMES["panel_g"]] = save_figure_all_formats(
            fig_g,
            layout.figure_base(PANEL_FILENAMES["panel_g"]),
        )
        plt.close(fig_g)
        fig_h = plot_panel_support_composition(
            area_composition_summary,
            x_column="support_area",
            x_label="Support area",
        )
        fig_paths[PANEL_FILENAMES["panel_h"]] = save_figure_all_formats(
            fig_h,
            layout.figure_base(PANEL_FILENAMES["panel_h"]),
        )
        plt.close(fig_h)
        fig_i = plot_panel_observed_vs_predicted(
            area_prediction_summary,
            x_column="support_area",
            x_label="Support area",
            actual_col="actual_signed_direction_deg",
            predicted_col="predicted_signed_direction_deg",
            actual_sem_col="sem_actual_signed_direction_deg",
            predicted_sem_col="sem_predicted_signed_direction_deg",
            y_label="Signed direction (deg)",
        )
        fig_paths[PANEL_FILENAMES["panel_i"]] = save_figure_all_formats(
            fig_i,
            layout.figure_base(PANEL_FILENAMES["panel_i"]),
        )
        plt.close(fig_i)
        fig_j = plot_panel_observed_vs_predicted(
            area_prediction_summary,
            x_column="support_area",
            x_label="Support area",
            actual_col="actual_bias_magnitude",
            predicted_col="predicted_bias_magnitude",
            actual_sem_col="sem_actual_bias_magnitude",
            predicted_sem_col="sem_predicted_bias_magnitude",
            y_label="Bias magnitude",
        )
        fig_paths[PANEL_FILENAMES["panel_j"]] = save_figure_all_formats(
            fig_j,
            layout.figure_base(PANEL_FILENAMES["panel_j"]),
        )
        plt.close(fig_j)
        fig_k = plot_panel_delay_local_dose_evolution(delay_local_dose_summary)
        fig_paths[PANEL_FILENAMES["panel_k"]] = save_figure_all_formats(
            fig_k,
            layout.figure_base(PANEL_FILENAMES["panel_k"]),
        )
        plt.close(fig_k)
        fig_l = plot_panel_observed_vs_predicted(
            delay_prediction_summary,
            x_column="delay_ms",
            x_label="Delay (ms)",
            actual_col="actual_signed_direction_deg",
            predicted_col="predicted_signed_direction_deg",
            actual_sem_col="sem_actual_signed_direction_deg",
            predicted_sem_col="sem_predicted_signed_direction_deg",
            y_label="Signed direction (deg)",
        )
        fig_paths[PANEL_FILENAMES["panel_l"]] = save_figure_all_formats(
            fig_l,
            layout.figure_base(PANEL_FILENAMES["panel_l"]),
        )
        plt.close(fig_l)
        fig_m = plot_panel_observed_vs_predicted(
            delay_prediction_summary,
            x_column="delay_ms",
            x_label="Delay (ms)",
            actual_col="actual_bias_magnitude",
            predicted_col="predicted_bias_magnitude",
            actual_sem_col="sem_actual_bias_magnitude",
            predicted_sem_col="sem_predicted_bias_magnitude",
            y_label="Bias magnitude",
        )
        fig_paths[PANEL_FILENAMES["panel_m"]] = save_figure_all_formats(
            fig_m,
            layout.figure_base(PANEL_FILENAMES["panel_m"]),
        )
        plt.close(fig_m)
        example_selection = {"panel_a": panel_a_meta}

    summary_payload = {
        "experiment": EXPERIMENT_NAME,
        "scientific_statement": (
            "This experiment no longer treats dose, area, location, and delay as parallel support properties. "
            "Instead, it first establishes an importance-specific local dose law, then interprets location and area "
            "as compositions of local support effects, and finally tests whether delay-dependent working-memory "
            "phenomena can be explained as the temporal evolution of these local STSP dose effects."
        ),
        "organizing_statement": (
            "This experiment is organized in three levels. First, it establishes importance-specific local STSP "
            "dose laws under fixed patch geometry. Second, it explains location and area effects as compositions of "
            "these local dose effects. Third, it tests whether delay-dependent working-memory phenomena can be "
            "explained as the temporal evolution of local STSP dose, projected through the same local dose laws into "
            "decision space."
        ),
        "summary_conclusion": (
            "STSP does not merely amplify readout. Instead, its effect on decision can be decomposed into "
            "importance-specific local dose laws, whose compositions explain location and area effects, and whose "
            "temporal evolution explains delay-dependent working-memory phenomena."
        ),
        "importance_definition": {
            "primary_definition": "importance = 1 - latency / 20, where latency is the earliest ON/OFF latency inside the encoding window",
            "latency_is_primary": True,
            "gamma_replication_is_not_recounted": True,
            "importance_tier_count": int(args.importance_tiers),
            "atomic_patch_area": int(args.atomic_patch_area),
        },
        "readout_definition": {
            "bias_vector": "DeltaV = centered_grouped_voltage(dynamic) - centered_grouped_voltage(static).",
            "bias_magnitude": "bias_magnitude = ||DeltaV||. This is the primary displacement-strength readout and it is not normalized by support area.",
            "bias_magnitude_per_pixel": "Retained only as an auxiliary CSV field; it is not used as a main panel or main summary readout.",
            "signed_direction_deg": (
                "Zero direction is the centered static grouped-voltage direction for the same record. Positive "
                "direction is defined by the pooled dynamic-minus-static sign axis after that global axis is "
                "projected onto the tangent plane orthogonal to the record-specific static baseline. The signed "
                "angle is computed as atan2(<dynamic_hat, tangent_hat>, <dynamic_hat, static_hat>) in that local "
                "static-centered basis, so the result distinguishes positive versus negative deflection away from "
                "the static condition."
            ),
            "direction_reference_status": str(direction_reference.get("status")),
            "direction_reference_record_count": int(direction_reference.get("n_records", 0)),
        },
        "manipulation_methods": {
            "local_dose_law": "Dose sweeps are run on fixed-area continuous atomic patches, one per importance tier, using target mean ux conditions.",
            "composition": "Location and area are analyzed as compositions of tier-specific local effects measured on the probe-synchronized importance map.",
            "temporal_validation": "Delay is evaluated as tier-specific local effective dose evolution on a fixed canonical support.",
        },
        "parameter_settings": {
            "model_path": str(args.model_path),
            "dataset_root": str(args.dataset_root),
            "split": str(args.split),
            "seed": int(args.seed),
            "device": str(device),
            "sample_ms": float(args.sample_ms),
            "delay_ms": float(args.delay_ms),
            "delay_sweep_ms": [float(v) for v in delay_sweep_ms],
            "probe_ms": float(args.probe_ms),
            "batch_size": int(args.batch_size),
            "max_probes": int(args.max_probes),
            "probe_selection_per_class": int(args.probe_selection_per_class),
            "foreground_threshold": float(args.foreground_threshold),
            "fixed_support_area_pixels": int(args.fixed_support_area_pixels),
            "atomic_patch_area": int(args.atomic_patch_area),
            "importance_tiers": int(args.importance_tiers),
            "importance_smoothing_alpha": float(args.importance_smoothing_alpha),
            "importance_smoothing_passes": int(args.importance_smoothing_passes),
            "support_area_levels": [int(v) for v in support_area_levels],
            "num_location_masks": int(args.num_location_masks),
            "mask_generation_seed": int(args.mask_generation_seed),
            "ux_dose_target_mean_ux_levels": [float(level["target_mean_ux_on_support"]) for level in dose_levels],
            "readout_step": int(readout_step),
        },
        "main_figure_panels": list(PANEL_FILENAMES.values()),
        "three_level_structure": {
            "local_dose_law": {
                "pooled_rows": dose_summary["pooled_df"].to_dict(orient="records"),
                "fit_rows": dose_law_payload["fit_summary_df"].to_dict(orient="records"),
            },
            "composition": {
                "location_prediction_rows": location_prediction_summary.to_dict(orient="records"),
                "area_prediction_rows": area_prediction_summary.to_dict(orient="records"),
            },
            "temporal_validation": {
                "delay_local_dose_rows": delay_local_dose_summary.to_dict(orient="records"),
                "delay_prediction_rows": delay_prediction_summary.to_dict(orient="records"),
            },
        },
        "generation_notes": generation_notes,
        "example_selection": example_selection,
        "artifacts": {
            "importance_patch_metadata_csv": str(importance_patch_metadata_csv),
            "dose_law_records_csv": str(dose_law_records_csv),
            "dose_law_fit_summary_csv": str(dose_law_fit_summary_csv),
            "location_composition_records_csv": str(location_composition_csv),
            "area_composition_records_csv": str(area_composition_csv),
            "delay_composition_records_csv": str(delay_composition_csv),
            "figure_paths": fig_paths,
        },
    }
    summary_path = save_summary_json(summary_payload, layout.root)

    run_config = {
        "experiment": EXPERIMENT_NAME,
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "split": str(args.split),
        "device": str(device),
        "seed": int(args.seed),
        "output_dir": str(layout.root),
        "sample_ms": float(args.sample_ms),
        "delay_ms": float(args.delay_ms),
        "delay_sweep_ms": [float(v) for v in delay_sweep_ms],
        "probe_ms": float(args.probe_ms),
        "batch_size": int(args.batch_size),
        "max_probes": int(args.max_probes),
        "probe_selection_per_class": int(args.probe_selection_per_class),
        "foreground_threshold": float(args.foreground_threshold),
        "fixed_support_area_pixels": int(args.fixed_support_area_pixels),
        "atomic_patch_area": int(args.atomic_patch_area),
        "importance_tiers": int(args.importance_tiers),
        "importance_smoothing_alpha": float(args.importance_smoothing_alpha),
        "importance_smoothing_passes": int(args.importance_smoothing_passes),
        "support_area_levels": [int(v) for v in support_area_levels],
        "num_location_masks": int(args.num_location_masks),
        "mask_generation_seed": int(args.mask_generation_seed),
        "ux_dose_target_mean_ux_levels": [float(level["target_mean_ux_on_support"]) for level in dose_levels],
        "scientific_hypothesis": (
            "STSP does not merely amplify readout. Instead, its effect on decision can be decomposed into "
            "importance-specific local dose laws, whose compositions explain location and area effects, and whose "
            "temporal evolution explains delay-dependent working-memory phenomena."
        ),
        "summary_json": str(summary_path),
        "figure_paths": fig_paths,
    }
    run_config_path = save_run_config(run_config, layout.root)
    save_log_lines(
        [
            f"experiment={EXPERIMENT_NAME}",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"seed={int(args.seed)}",
            f"device={device}",
            f"result_root={layout.root.resolve()}",
            f"summary_json={Path(summary_path).resolve()}",
            f"run_config_json={Path(run_config_path).resolve()}",
        ]
        + [f"{panel_name}={panel_paths}" for panel_name, panel_paths in fig_paths.items()],
        layout.log_dir,
    )


if __name__ == "__main__":
    main()
