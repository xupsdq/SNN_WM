from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


def parse_delay_list(delay_text: str) -> List[int]:
    values: List[int] = []
    for raw in str(delay_text).split(","):
        item = raw.strip()
        if not item:
            continue
        delay = int(float(item))
        if delay <= 0:
            raise ValueError("Delay values must be positive.")
        values.append(delay)
    if not values:
        raise ValueError("At least one delay is required.")
    return sorted(dict.fromkeys(values))


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p_hat = float(successes) / float(total)
    denom = 1.0 + (z ** 2) / float(total)
    center = (p_hat + (z ** 2) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / total) + ((z ** 2) / (4.0 * (total ** 2))))
    return 100.0 * max(0.0, center - margin), 100.0 * min(1.0, center + margin)


def paired_bootstrap_diff_summary(values_a: np.ndarray, values_b: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have the same shape.")
    if a.size == 0:
        raise ValueError("paired_bootstrap_diff_summary received empty arrays.")

    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot, dtype=np.float64)
    n = a.size
    for idx in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot[idx] = (float(a[sample_idx].mean()) - float(b[sample_idx].mean())) * 100.0

    observed = (float(a.mean()) - float(b.mean())) * 100.0
    return {
        "observed_diff_pp": observed,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n_boot": int(n_boot),
    }


def sem(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def bootstrap_mean_ci(values: np.ndarray | Sequence[float], n_boot: int, seed: int) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        raise ValueError("bootstrap_mean_ci received empty values")
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot, dtype=np.float64)
    n = vals.size
    for idx in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot[idx] = float(vals[sample_idx].mean())
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
