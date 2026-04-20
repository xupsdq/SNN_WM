from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def sem(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def bootstrap_mean_ci(values: Sequence[float] | np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("bootstrap_mean_ci received empty values")
    rng = np.random.default_rng(int(seed))
    boot = np.zeros(int(n_boot), dtype=np.float64)
    for index in range(int(n_boot)):
        sample_idx = rng.integers(0, arr.size, size=arr.size)
        boot[index] = float(arr[sample_idx].mean())
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a_flat = np.asarray(a, dtype=np.float64).reshape(-1)
    b_flat = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    if denom <= eps:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


__all__ = ["bootstrap_mean_ci", "cosine_similarity", "sem"]
