from __future__ import annotations

import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.experiments.common.input_masks import foreground_mask_from_image

EPS = 1e-12


def _sanitize_area_levels(values: Sequence[int]) -> list[int]:
    if not values:
        raise ValueError("target_area_schedule must contain at least one value.")
    levels = sorted(dict.fromkeys(int(v) for v in values))
    if any(level <= 0 for level in levels):
        raise ValueError("target_area_schedule values must be positive.")
    return levels


def _mask_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(mask, dtype=bool))


def _mask_center(mask: np.ndarray) -> tuple[float, float]:
    coords = _mask_coords(mask)
    if coords.size <= 0:
        return float("nan"), float("nan")
    center = coords.mean(axis=0, dtype=np.float64)
    return float(center[0]), float(center[1])


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


def center_grouped_voltage(v: np.ndarray | torch.Tensor) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return arr - arr.mean(axis=-1, keepdims=True)


def safe_normalize(v: np.ndarray | torch.Tensor, eps: float = EPS) -> np.ndarray | None:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= float(eps):
        return None
    return arr / norm


def _fallback_tangent_axis(static_axis: np.ndarray, eps: float = EPS) -> np.ndarray | None:
    if static_axis.size <= 0:
        return None
    best_idx = int(np.argmin(np.abs(static_axis)))
    basis = np.zeros_like(static_axis, dtype=np.float64)
    basis[best_idx] = 1.0
    tangent = basis - float(np.dot(basis, static_axis)) * static_axis
    return safe_normalize(tangent, eps=eps)


def generate_nested_support_masks(
    probe_image: torch.Tensor,
    *,
    target_area_schedule: Sequence[int],
    foreground_threshold: float,
) -> pd.DataFrame:
    probe_fg = foreground_mask_from_image(probe_image, threshold=float(foreground_threshold))
    largest_component = _largest_component_coords(probe_fg)
    columns = [
        "support_mask",
        "support_area",
        "support_center_row",
        "support_center_col",
        "support_coords_json",
        "area_index",
        "area_label",
    ]
    if largest_component.size <= 0:
        return pd.DataFrame(columns=columns)
    seed_row, seed_col = _choose_component_seed(largest_component)
    ordered_coords = _sorted_coords_from_seed(largest_component, seed_row=seed_row, seed_col=seed_col)
    rows: list[dict[str, object]] = []
    for area_index, target_area in enumerate(_sanitize_area_levels(target_area_schedule), start=1):
        if int(target_area) > int(ordered_coords.shape[0]):
            continue
        support_mask = _make_mask_from_coords(ordered_coords[: int(target_area)], probe_fg.shape)
        center_row, center_col = _mask_center(support_mask)
        rows.append(
            {
                "support_mask": support_mask,
                "support_area": int(support_mask.sum()),
                "support_center_row": float(center_row),
                "support_center_col": float(center_col),
                "support_coords_json": json.dumps(_mask_coords(support_mask).astype(int).tolist(), ensure_ascii=False),
                "area_index": int(area_index),
                "area_label": f"Area {int(target_area)}",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_signed_direction_reference(
    records_df: pd.DataFrame,
    *,
    static_column: str = "grouped_voltage_static",
    delta_column: str = "DeltaV",
    eps: float = EPS,
) -> dict[str, object]:
    if records_df.empty:
        return {
            "status": "empty",
            "n_records": 0,
            "static_axis": None,
            "sign_axis": None,
            "tangent_axis": None,
        }
    static_vectors: list[np.ndarray] = []
    delta_vectors: list[np.ndarray] = []
    for row in records_df.itertuples(index=False):
        static_vec = center_grouped_voltage(getattr(row, static_column))
        delta_vec = np.asarray(getattr(row, delta_column), dtype=np.float64).reshape(-1)
        if static_vec.size <= 0 or delta_vec.size <= 0:
            continue
        if not np.all(np.isfinite(static_vec)) or not np.all(np.isfinite(delta_vec)):
            continue
        static_vectors.append(np.asarray(static_vec, dtype=np.float64).reshape(-1))
        delta_vectors.append(delta_vec)
    if not static_vectors:
        return {
            "status": "empty_after_filter",
            "n_records": 0,
            "static_axis": None,
            "sign_axis": None,
            "tangent_axis": None,
        }
    static_stack = np.stack(static_vectors, axis=0)
    delta_stack = np.stack(delta_vectors, axis=0)
    static_axis = safe_normalize(static_stack.mean(axis=0), eps=eps)
    if static_axis is None:
        for vec in static_stack:
            static_axis = safe_normalize(vec, eps=eps)
            if static_axis is not None:
                break
    if static_axis is None:
        return {
            "status": "degenerate_static_axis",
            "n_records": int(len(static_vectors)),
            "static_axis": None,
            "sign_axis": None,
            "tangent_axis": None,
        }
    mean_delta = delta_stack.mean(axis=0)
    sign_axis = safe_normalize(mean_delta, eps=eps)
    status = "mean_delta_axis"
    if sign_axis is None:
        delta_norms = np.linalg.norm(delta_stack, axis=1)
        valid = delta_stack[delta_norms > float(eps)]
        if valid.size > 0:
            _, _, vh = np.linalg.svd(valid, full_matrices=False)
            sign_axis = safe_normalize(vh[0], eps=eps)
            status = "delta_pca_axis"
    if sign_axis is None:
        for vec in delta_stack:
            sign_axis = safe_normalize(vec, eps=eps)
            if sign_axis is not None:
                status = "first_nonzero_delta_axis"
                break
    tangent_axis = None
    if sign_axis is None:
        tangent_axis = _fallback_tangent_axis(static_axis, eps=eps)
        status = "basis_fallback"
    return {
        "status": status if (sign_axis is not None or tangent_axis is not None) else "degenerate_sign_axis",
        "n_records": int(len(static_vectors)),
        "static_axis": static_axis,
        "sign_axis": sign_axis,
        # Keep the old key for compatibility with callers that still expect it.
        "tangent_axis": sign_axis if sign_axis is not None else tangent_axis,
    }


def compute_signed_direction_deg(
    dynamic_vector: np.ndarray | torch.Tensor,
    static_vector: np.ndarray | torch.Tensor,
    *,
    reference: Mapping[str, object],
    eps: float = EPS,
) -> float:
    sign_seed = reference.get("sign_axis")
    if sign_seed is None:
        sign_seed = reference.get("tangent_axis")
    if sign_seed is None:
        return float("nan")
    static_centered = center_grouped_voltage(static_vector).reshape(-1)
    dynamic_centered = center_grouped_voltage(dynamic_vector).reshape(-1)
    static_axis = safe_normalize(static_centered, eps=eps)
    if static_axis is None:
        static_axis = reference.get("static_axis")
    if static_axis is None:
        return float("nan")
    dynamic_axis = safe_normalize(dynamic_centered, eps=eps)
    if dynamic_axis is None:
        return float("nan")
    tangent_local = np.asarray(sign_seed, dtype=np.float64).reshape(-1)
    tangent_local = tangent_local - float(np.dot(tangent_local, static_axis)) * static_axis
    tangent_axis = safe_normalize(tangent_local, eps=eps)
    if tangent_axis is None:
        tangent_axis = _fallback_tangent_axis(np.asarray(static_axis, dtype=np.float64), eps=eps)
    if tangent_axis is None:
        return float("nan")
    if not np.all(np.isfinite(dynamic_axis)):
        return float("nan")
    x_coord = float(np.dot(dynamic_axis, static_axis))
    y_coord = float(np.dot(dynamic_axis, tangent_axis))
    return float(np.degrees(np.arctan2(y_coord, x_coord)))


def annotate_signed_direction_records(
    records_df: pd.DataFrame,
    *,
    reference: Mapping[str, object],
    dynamic_column: str = "grouped_voltage_dynamic",
    static_column: str = "grouped_voltage_static",
    output_column: str = "signed_direction_deg",
) -> pd.DataFrame:
    records = records_df.copy()
    if records.empty:
        records[output_column] = pd.Series(dtype=np.float64)
        return records
    records[output_column] = records.apply(
        lambda row: compute_signed_direction_deg(
            row[dynamic_column],
            row[static_column],
            reference=reference,
        ),
        axis=1,
    )
    return records


__all__ = [
    "annotate_signed_direction_records",
    "build_signed_direction_reference",
    "center_grouped_voltage",
    "compute_signed_direction_deg",
    "generate_nested_support_masks",
    "safe_normalize",
]
