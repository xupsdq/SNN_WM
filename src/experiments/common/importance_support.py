from __future__ import annotations

import heapq
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.experiments.common.input_masks import foreground_mask_from_image


def _mask_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(mask, dtype=bool))


def _mask_center(mask: np.ndarray) -> tuple[float, float]:
    coords = _mask_coords(mask)
    if coords.size <= 0:
        return float("nan"), float("nan")
    center = coords.mean(axis=0, dtype=np.float64)
    return float(center[0]), float(center[1])


def _make_mask_from_coords(coords: Sequence[tuple[int, int]] | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    arr = np.asarray(coords, dtype=np.int64)
    if arr.size > 0:
        out[arr[:, 0], arr[:, 1]] = True
    return out


def compute_latency_importance_payload(
    probe_image: torch.Tensor,
    *,
    encoder,
    foreground_threshold: float,
    importance_smoothing_alpha: float = 0.0,
    importance_smoothing_passes: int = 0,
) -> dict[str, object]:
    image = probe_image.detach().to(dtype=torch.float32)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected probe_image with shape [1, C, H, W] or [C, H, W], got {tuple(probe_image.shape)}")

    with torch.no_grad():
        dog_map = encoder.preprocessor(image.to(device=encoder.device)).detach().cpu().to(torch.float32)[0]

    channels, height, width = dog_map.shape
    if int(channels) != 2:
        raise ValueError(f"Expected ON/OFF DoG channels, got {channels}")

    encoding_window = int(getattr(encoder, "encoding_window", 20))
    latency_map = torch.full((channels, height, width), -1, dtype=torch.long)
    valid_mask = dog_map > 0.0
    valid_values = dog_map[valid_mask]
    if valid_values.numel() > 0:
        sort_indices = torch.argsort(valid_values)
        valid_ranks = torch.argsort(sort_indices)
        valid_latency = ((valid_values.numel() - 1 - valid_ranks).float() * float(encoding_window) / float(valid_values.numel())).long()
        valid_latency = torch.clamp(valid_latency, 0, encoding_window - 1)
        latency_map[valid_mask] = valid_latency

    foreground_mask = foreground_mask_from_image(probe_image, threshold=float(foreground_threshold))
    on_latency = latency_map[0].numpy().astype(np.int64, copy=False)
    off_latency = latency_map[1].numpy().astype(np.int64, copy=False)
    valid_any = (on_latency >= 0) | (off_latency >= 0)
    earliest = np.full((height, width), int(encoding_window), dtype=np.int64)
    if valid_any.any():
        on_fill = np.where(on_latency >= 0, on_latency, int(encoding_window))
        off_fill = np.where(off_latency >= 0, off_latency, int(encoding_window))
        earliest = np.minimum(on_fill, off_fill)
    importance_map = np.zeros((height, width), dtype=np.float64)
    importance_map[valid_any] = 1.0 - (earliest[valid_any].astype(np.float64) / float(encoding_window))
    importance_map = np.clip(importance_map, 0.0, 1.0)
    importance_map = np.where(foreground_mask, importance_map, 0.0)

    alpha = float(importance_smoothing_alpha)
    passes = max(int(importance_smoothing_passes), 0)
    if passes > 0 and alpha > 0.0:
        current = np.asarray(importance_map, dtype=np.float64)
        fg = np.asarray(foreground_mask, dtype=bool)
        for _ in range(passes):
            neighbor_sum = np.zeros_like(current)
            neighbor_count = np.zeros_like(current)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                shifted = np.roll(current, shift=(dr, dc), axis=(0, 1))
                shifted_fg = np.roll(fg, shift=(dr, dc), axis=(0, 1))
                if dr > 0:
                    shifted[:dr, :] = 0.0
                    shifted_fg[:dr, :] = False
                elif dr < 0:
                    shifted[dr:, :] = 0.0
                    shifted_fg[dr:, :] = False
                if dc > 0:
                    shifted[:, :dc] = 0.0
                    shifted_fg[:, :dc] = False
                elif dc < 0:
                    shifted[:, dc:] = 0.0
                    shifted_fg[:, dc:] = False
                neighbor_sum += shifted * shifted_fg
                neighbor_count += shifted_fg.astype(np.float64)
            neighbor_mean = np.divide(
                neighbor_sum,
                np.maximum(neighbor_count, 1.0),
                out=np.zeros_like(neighbor_sum),
                where=neighbor_count > 0.0,
            )
            current = np.where(fg, (1.0 - alpha) * current + alpha * neighbor_mean, 0.0)
        importance_map = current

    return {
        "foreground_mask": np.asarray(foreground_mask, dtype=bool),
        "dog_map": dog_map.numpy().astype(np.float64, copy=False),
        "latency_on": on_latency,
        "latency_off": off_latency,
        "earliest_latency": earliest,
        "importance_map": importance_map,
        "encoding_window": int(encoding_window),
    }


def assign_importance_tiers(
    importance_map: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    n_tiers: int = 5,
) -> dict[str, object]:
    fg = np.asarray(foreground_mask, dtype=bool)
    if int(n_tiers) <= 1:
        raise ValueError("n_tiers must be at least 2")
    coords = _mask_coords(fg)
    tier_map = np.zeros_like(fg, dtype=np.int64)
    if coords.size <= 0:
        return {
            "tier_map": tier_map,
            "tier_edges": [],
            "tier_value_summary": pd.DataFrame(columns=["importance_tier", "n_pixels", "min_importance", "max_importance", "mean_importance"]),
        }
    values = np.asarray(importance_map, dtype=np.float64)[fg]
    order = np.lexsort((coords[:, 1], coords[:, 0], values))
    sorted_coords = coords[order]
    sorted_values = values[order]
    splits = np.array_split(np.arange(sorted_coords.shape[0]), int(n_tiers))
    rows: list[dict[str, object]] = []
    edges: list[float] = []
    for tier_idx, split in enumerate(splits, start=1):
        if split.size <= 0:
            rows.append(
                {
                    "importance_tier": int(tier_idx),
                    "n_pixels": 0,
                    "min_importance": float("nan"),
                    "max_importance": float("nan"),
                    "mean_importance": float("nan"),
                }
            )
            edges.append(float("nan"))
            continue
        tier_coords = sorted_coords[split]
        tier_values = sorted_values[split]
        tier_map[tier_coords[:, 0], tier_coords[:, 1]] = int(tier_idx)
        rows.append(
            {
                "importance_tier": int(tier_idx),
                "n_pixels": int(split.size),
                "min_importance": float(np.min(tier_values)),
                "max_importance": float(np.max(tier_values)),
                "mean_importance": float(np.mean(tier_values)),
            }
        )
        edges.append(float(np.max(tier_values)))
    return {
        "tier_map": tier_map,
        "tier_edges": edges,
        "tier_value_summary": pd.DataFrame(rows),
    }


def _choose_seed(
    tier_coords: np.ndarray,
    blocked_mask: np.ndarray,
) -> tuple[int, int]:
    centroid = tier_coords.mean(axis=0, dtype=np.float64)
    distances = np.sum((tier_coords.astype(np.float64) - centroid[None, :]) ** 2, axis=1)
    blocked_penalty = np.asarray(blocked_mask[tier_coords[:, 0], tier_coords[:, 1]], dtype=np.float64) * 1e6
    best_idx = int(np.argmin(distances + blocked_penalty))
    return int(tier_coords[best_idx, 0]), int(tier_coords[best_idx, 1])


def _grow_contiguous_patch(
    *,
    foreground_mask: np.ndarray,
    tier_map: np.ndarray,
    importance_map: np.ndarray,
    target_tier: int,
    patch_area: int,
    blocked_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    fg = np.asarray(foreground_mask, dtype=bool)
    if int(patch_area) <= 0:
        raise ValueError("patch_area must be positive")
    tier_coords = _mask_coords(fg & (np.asarray(tier_map, dtype=np.int64) == int(target_tier)))
    if tier_coords.shape[0] <= 0:
        return None
    blocked = np.zeros_like(fg, dtype=bool) if blocked_mask is None else np.asarray(blocked_mask, dtype=bool)
    seed_row, seed_col = _choose_seed(tier_coords, blocked)
    target_mean = float(np.mean(np.asarray(importance_map, dtype=np.float64)[tier_coords[:, 0], tier_coords[:, 1]]))
    height, width = fg.shape
    selected: list[tuple[int, int]] = []
    selected_mask = np.zeros_like(fg, dtype=bool)
    visited = np.zeros_like(fg, dtype=bool)
    heap: list[tuple[float, int, int, int]] = []

    def push(row: int, col: int) -> None:
        if row < 0 or row >= height or col < 0 or col >= width:
            return
        if visited[row, col] or not fg[row, col]:
            return
        visited[row, col] = True
        tier_penalty = 0.0 if int(tier_map[row, col]) == int(target_tier) else 0.35
        overlap_penalty = 0.75 if blocked[row, col] else 0.0
        dist_penalty = 0.025 * float((row - seed_row) ** 2 + (col - seed_col) ** 2)
        importance_penalty = abs(float(importance_map[row, col]) - target_mean)
        heapq.heappush(heap, (importance_penalty + tier_penalty + overlap_penalty + dist_penalty, row, col, len(selected)))

    push(seed_row, seed_col)
    while heap and len(selected) < int(patch_area):
        _, row, col, _ = heapq.heappop(heap)
        if selected_mask[row, col]:
            continue
        selected_mask[row, col] = True
        selected.append((int(row), int(col)))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            push(row + dr, col + dc)
    if len(selected) < int(patch_area):
        return None
    return _make_mask_from_coords(selected, fg.shape)


def build_atomic_tier_patches(
    *,
    probe_id: int,
    importance_map: np.ndarray,
    foreground_mask: np.ndarray,
    tier_map: np.ndarray,
    patch_area: int,
    n_tiers: int = 5,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    notes: list[dict[str, object]] = []
    blocked = np.zeros_like(np.asarray(foreground_mask, dtype=bool), dtype=bool)
    for tier in range(int(n_tiers), 0, -1):
        support_mask = _grow_contiguous_patch(
            foreground_mask=foreground_mask,
            tier_map=tier_map,
            importance_map=importance_map,
            target_tier=int(tier),
            patch_area=int(patch_area),
            blocked_mask=blocked,
        )
        if support_mask is None or int(np.asarray(support_mask, dtype=bool).sum()) != int(patch_area):
            notes.append(
                {
                    "probe_id": int(probe_id),
                    "importance_tier": int(tier),
                    "status": "skipped_insufficient_contiguous_patch",
                }
            )
            continue
        blocked |= np.asarray(support_mask, dtype=bool)
        coords = _mask_coords(support_mask)
        center_row, center_col = _mask_center(support_mask)
        support_importance = np.asarray(importance_map, dtype=np.float64)[support_mask]
        tier_values = np.asarray(importance_map, dtype=np.float64)[np.asarray(tier_map, dtype=np.int64) == int(tier)]
        rows.append(
            {
                "probe_id": int(probe_id),
                "patch_id": f"probe_{int(probe_id)}_tier_{int(tier)}",
                "importance_tier": int(tier),
                "support_mask": np.asarray(support_mask, dtype=bool),
                "support_area": int(np.asarray(support_mask, dtype=bool).sum()),
                "mean_importance": float(np.mean(support_importance)) if support_importance.size > 0 else float("nan"),
                "tier_mean_importance": float(np.mean(tier_values)) if tier_values.size > 0 else float("nan"),
                "support_center_row": float(center_row),
                "support_center_col": float(center_col),
                "support_coords_json": json.dumps(coords.astype(int).tolist(), ensure_ascii=False),
                "target_tier_coverage_ratio": float(np.mean(np.asarray(tier_map, dtype=np.int64)[support_mask] == int(tier))),
            }
        )
    patch_df = pd.DataFrame(rows).sort_values(["probe_id", "importance_tier"], kind="stable").reset_index(drop=True)
    return patch_df, notes


def summarize_support_composition(
    *,
    support_mask: np.ndarray,
    tier_map: np.ndarray,
    importance_map: np.ndarray,
    atomic_patch_area: int,
    n_tiers: int = 5,
) -> dict[str, float]:
    support = np.asarray(support_mask, dtype=bool)
    tier_arr = np.asarray(tier_map, dtype=np.int64)
    importance_arr = np.asarray(importance_map, dtype=np.float64)
    rows: dict[str, float] = {
        "support_area": float(int(support.sum())),
        "mean_importance_on_support": float(np.mean(importance_arr[support])) if support.any() else float("nan"),
    }
    total_pixels = int(support.sum())
    for tier in range(1, int(n_tiers) + 1):
        count = int((support & (tier_arr == int(tier))).sum())
        rows[f"tier_{int(tier)}_pixel_count"] = float(count)
        rows[f"c{int(tier)}"] = float(count / max(int(atomic_patch_area), 1))
        rows[f"p{int(tier)}"] = float(count / max(total_pixels, 1))
    return rows


__all__ = [
    "assign_importance_tiers",
    "build_atomic_tier_patches",
    "compute_latency_importance_payload",
    "summarize_support_composition",
]
