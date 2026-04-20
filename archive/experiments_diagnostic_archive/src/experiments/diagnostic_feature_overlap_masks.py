from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Iterable, Iterator, List, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class PatchSpec:
    patch_id: int
    patch_row: int
    patch_col: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int


def build_patch_grid(height: int, width: int, patch_size: int, stride: int) -> List[PatchSpec]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    patches: List[PatchSpec] = []
    patch_id = 0
    row_positions = list(range(0, max(1, height - patch_size + 1), stride))
    col_positions = list(range(0, max(1, width - patch_size + 1), stride))
    if not row_positions or row_positions[-1] != max(0, height - patch_size):
        row_positions.append(max(0, height - patch_size))
    if not col_positions or col_positions[-1] != max(0, width - patch_size):
        col_positions.append(max(0, width - patch_size))
    row_positions = sorted(dict.fromkeys(row_positions))
    col_positions = sorted(dict.fromkeys(col_positions))
    for row_idx, row_start in enumerate(row_positions):
        row_end = min(height, row_start + patch_size)
        for col_idx, col_start in enumerate(col_positions):
            col_end = min(width, col_start + patch_size)
            patches.append(
                PatchSpec(
                    patch_id=patch_id,
                    patch_row=row_idx,
                    patch_col=col_idx,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                )
            )
            patch_id += 1
    return patches


def iter_patch_grid(height: int, width: int, patch_size: int, stride: int) -> Iterator[PatchSpec]:
    yield from iter_patch_grid_region(
        image_height=height,
        image_width=width,
        row_start=0,
        row_end=height,
        col_start=0,
        col_end=width,
        patch_size=patch_size,
        stride=stride,
    )


def count_patch_grid(height: int, width: int, patch_size: int, stride: int) -> int:
    return sum(1 for _ in iter_patch_grid(height=height, width=width, patch_size=patch_size, stride=stride))


def iter_patch_grid_region(
    *,
    image_height: int,
    image_width: int,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    patch_size: int,
    stride: int,
    patch_id_start: int = 0,
) -> Iterator[PatchSpec]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")
    row_start = max(0, min(int(row_start), image_height))
    row_end = max(row_start, min(int(row_end), image_height))
    col_start = max(0, min(int(col_start), image_width))
    col_end = max(col_start, min(int(col_end), image_width))
    region_height = max(1, row_end - row_start)
    region_width = max(1, col_end - col_start)

    row_positions = list(range(row_start, max(row_start + 1, row_end - patch_size + 1), stride))
    col_positions = list(range(col_start, max(col_start + 1, col_end - patch_size + 1), stride))
    final_row = max(row_start, row_end - patch_size)
    final_col = max(col_start, col_end - patch_size)
    if not row_positions or row_positions[-1] != final_row:
        row_positions.append(final_row)
    if not col_positions or col_positions[-1] != final_col:
        col_positions.append(final_col)
    row_positions = sorted(dict.fromkeys(row_positions))
    col_positions = sorted(dict.fromkeys(col_positions))

    patch_id = int(patch_id_start)
    for row_idx, patch_row_start in enumerate(row_positions):
        patch_row_end = min(image_height, patch_row_start + min(patch_size, region_height))
        for col_idx, patch_col_start in enumerate(col_positions):
            patch_col_end = min(image_width, patch_col_start + min(patch_size, region_width))
            yield PatchSpec(
                patch_id=patch_id,
                patch_row=row_idx,
                patch_col=col_idx,
                row_start=int(patch_row_start),
                row_end=int(patch_row_end),
                col_start=int(patch_col_start),
                col_end=int(patch_col_end),
            )
            patch_id += 1


def apply_ablation(image: "torch.Tensor", mask: np.ndarray, fill_value: float = 0.0) -> "torch.Tensor":
    import torch

    if image.ndim != 3:
        raise ValueError(f"Expected image shape [C, H, W], got {tuple(image.shape)}")
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != tuple(image.shape[-2:]):
        raise ValueError("Mask shape must match image spatial shape")
    out = image.clone()
    mask_t = torch.as_tensor(mask_bool, dtype=torch.bool, device=image.device).unsqueeze(0)
    out = torch.where(mask_t, torch.full_like(out, float(fill_value)), out)
    return out


def apply_preserve_only(image: "torch.Tensor", mask: np.ndarray, fill_value: float = 0.0) -> "torch.Tensor":
    import torch

    if image.ndim != 3:
        raise ValueError(f"Expected image shape [C, H, W], got {tuple(image.shape)}")
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != tuple(image.shape[-2:]):
        raise ValueError("Mask shape must match image spatial shape")
    out = torch.full_like(image, float(fill_value))
    mask_t = torch.as_tensor(mask_bool, dtype=torch.bool, device=image.device).unsqueeze(0)
    out = torch.where(mask_t, image, out)
    return out


def build_topk_mask(arr: np.ndarray, topq_percent: float) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    flat = values.reshape(-1)
    finite_mask = np.isfinite(flat)
    out = np.zeros_like(flat, dtype=bool)
    if not finite_mask.any():
        return out.reshape(values.shape)
    if topq_percent <= 0.0 or topq_percent > 100.0:
        raise ValueError("topq_percent must be in (0, 100]")
    finite_values = flat[finite_mask]
    count = max(1, int(np.ceil(finite_values.size * float(topq_percent) / 100.0)))
    threshold = np.partition(finite_values, -count)[-count]
    out[finite_mask] = flat[finite_mask] >= threshold
    return out.reshape(values.shape)


def build_nonzero_mask(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    return np.isfinite(values) & (values != 0.0)


def rank_nonzero_values(arr: np.ndarray) -> pd.DataFrame:
    values = np.asarray(arr, dtype=np.float64)
    mask = build_nonzero_mask(values)
    if values.ndim == 1:
        rows = [
            {"index": int(idx), "value": float(values[idx])}
            for idx in np.flatnonzero(mask).tolist()
        ]
        ranking = pd.DataFrame(rows)
        if ranking.empty:
            return pd.DataFrame(columns=["rank", "index", "value"])
        ranking = ranking.sort_values(by=["value", "index"], ascending=[False, True], kind="stable").reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=np.int64))
        return ranking
    if values.ndim == 2:
        coords = np.argwhere(mask)
        rows = [
            {"row": int(row), "col": int(col), "value": float(values[row, col])}
            for row, col in coords.tolist()
        ]
        ranking = pd.DataFrame(rows)
        if ranking.empty:
            return pd.DataFrame(columns=["rank", "row", "col", "value"])
        ranking = ranking.sort_values(by=["value", "row", "col"], ascending=[False, True, True], kind="stable").reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=np.int64))
        return ranking
    raise ValueError("rank_nonzero_values expects a 1D or 2D array")


def build_mask_from_rank(arr: np.ndarray, area: int | None = None, percent: float | None = None) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    if (area is None) == (percent is None):
        raise ValueError("Exactly one of area or percent must be provided")
    flat = values.reshape(-1)
    finite_idx = np.flatnonzero(np.isfinite(flat))
    out = np.zeros_like(flat, dtype=bool)
    if finite_idx.size == 0:
        return out.reshape(values.shape)
    if percent is not None:
        if float(percent) <= 0.0 or float(percent) > 100.0:
            raise ValueError("percent must be in (0, 100]")
        count = int(np.ceil(finite_idx.size * float(percent) / 100.0))
    else:
        if int(area) < 0:
            raise ValueError("area must be non-negative")
        count = int(area)
    count = min(finite_idx.size, int(count))
    if count <= 0:
        return out.reshape(values.shape)
    finite_values = flat[finite_idx]
    order = np.argsort(-finite_values, kind="stable")[:count]
    out[finite_idx[order]] = True
    return out.reshape(values.shape)


def build_mask_from_threshold(arr: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    return np.isfinite(values) & (values >= float(threshold))


def build_mask_from_nonzero(arr: np.ndarray) -> np.ndarray:
    return build_nonzero_mask(arr)


def project_patch_values_to_image(
    height: int,
    width: int,
    patches: Sequence[PatchSpec],
    values: Sequence[float],
) -> np.ndarray:
    accum = np.zeros((height, width), dtype=np.float64)
    coverage = np.zeros((height, width), dtype=np.float64)
    for patch, value in zip(patches, values):
        accum[patch.row_start:patch.row_end, patch.col_start:patch.col_end] += float(value)
        coverage[patch.row_start:patch.row_end, patch.col_start:patch.col_end] += 1.0
    coverage[coverage <= 0.0] = 1.0
    return accum / coverage


def connected_component_count(mask: np.ndarray) -> int:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2:
        raise ValueError("connected_component_count expects a 2D mask")
    visited = np.zeros_like(mask_bool, dtype=bool)
    total = 0
    height, width = mask_bool.shape
    for row in range(height):
        for col in range(width):
            if not mask_bool[row, col] or visited[row, col]:
                continue
            total += 1
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            while queue:
                cur_row, cur_col = queue.popleft()
                for nxt_row, nxt_col in (
                    (cur_row - 1, cur_col),
                    (cur_row + 1, cur_col),
                    (cur_row, cur_col - 1),
                    (cur_row, cur_col + 1),
                ):
                    if (
                        0 <= nxt_row < height
                        and 0 <= nxt_col < width
                        and mask_bool[nxt_row, nxt_col]
                        and not visited[nxt_row, nxt_col]
                    ):
                        visited[nxt_row, nxt_col] = True
                        queue.append((nxt_row, nxt_col))
    return int(total)


def build_random_matched_mask(
    ref_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    component_hint: int | None = None,
    exclude_mask: np.ndarray | None = None,
    max_tries: int = 128,
) -> np.ndarray:
    ref_bool = np.asarray(ref_mask, dtype=bool)
    area = int(ref_bool.sum())
    if area <= 0:
        return np.zeros_like(ref_bool, dtype=bool)
    exclude_bool = np.zeros_like(ref_bool, dtype=bool) if exclude_mask is None else np.asarray(exclude_mask, dtype=bool)
    available = np.flatnonzero(~exclude_bool.reshape(-1))
    if available.size < area:
        # Large stable masks can leave too few free pixels for a disjoint
        # area-matched control. In that case, fall back to whole-image sampling
        # so evaluation degrades gracefully instead of crashing.
        available = np.arange(ref_bool.size, dtype=np.int64)
    best_mask = None
    best_gap = None
    target_components = int(component_hint) if component_hint is not None else connected_component_count(ref_bool)
    for _ in range(max_tries):
        sample = rng.choice(available, size=area, replace=False)
        candidate = np.zeros(ref_bool.size, dtype=bool)
        candidate[sample] = True
        candidate = candidate.reshape(ref_bool.shape)
        gap = abs(connected_component_count(candidate) - target_components)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_mask = candidate
            if gap == 0:
                break
    if best_mask is None:
        raise RuntimeError("Failed to sample a matched random mask")
    return np.asarray(best_mask, dtype=bool)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any():
        return None
    rows = np.flatnonzero(mask_bool.any(axis=1))
    cols = np.flatnonzero(mask_bool.any(axis=0))
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
