from __future__ import annotations

import math

import numpy as np


def select_top_mask(
    values: np.ndarray,
    q: float,
    *,
    positive: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    eligible = np.isfinite(arr)
    if positive is not None:
        eligible &= np.asarray(positive, dtype=bool)
    indices = np.flatnonzero(eligible.reshape(-1))
    mask = np.zeros(arr.size, dtype=bool)
    if indices.size:
        count = max(1, int(math.ceil(float(q) * indices.size)))
        chosen = indices[np.argsort(arr.reshape(-1)[indices])[-count:]]
        mask[chosen] = True
    return mask.reshape(arr.shape)


def select_matched_nonpeak_mask(
    peak: np.ndarray,
    pool: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero((~peak) & pool)
    count = int(np.sum(peak))
    if candidates.size < count:
        candidates = np.flatnonzero(~peak)
    chosen = (
        rng.choice(candidates, size=min(count, candidates.size), replace=False)
        if candidates.size
        else np.asarray([], dtype=int)
    )
    out = np.zeros_like(peak, dtype=bool)
    out[chosen] = True
    return out


__all__ = ["select_matched_nonpeak_mask", "select_top_mask"]
