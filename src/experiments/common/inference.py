from __future__ import annotations

import hashlib

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray, *, draws: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(draws), len(array)))
    samples = array[indices].mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def crossed_bootstrap_mean_ci(
    values: np.ndarray,
    family_ids: np.ndarray,
    anchor_ids: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan")
    family_ids = np.asarray(family_ids, dtype=np.int64)
    anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
    families = sorted(int(value) for value in np.unique(family_ids))
    anchors = sorted(int(value) for value in np.unique(anchor_ids))
    family_index = {value: index for index, value in enumerate(families)}
    anchor_index = {value: index for index, value in enumerate(anchors)}
    f_rows = np.asarray([family_index[int(value)] for value in family_ids], dtype=np.int64)
    a_rows = np.asarray([anchor_index[int(value)] for value in anchor_ids], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        f_sample = rng.integers(0, len(families), size=len(families))
        a_sample = rng.integers(0, len(anchors), size=len(anchors))
        f_weight = np.bincount(f_sample, minlength=len(families))[f_rows]
        a_weight = np.bincount(a_sample, minlength=len(anchors))[a_rows]
        weights = f_weight * a_weight
        denominator = float(weights.sum())
        means[draw] = (
            float(np.sum(values * weights) / denominator)
            if denominator > 0
            else float("nan")
        )
    finite = means[np.isfinite(means)]
    if not len(finite):
        return float("nan"), float("nan")
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def exact_sign_flip_p(values: np.ndarray, *, alternative: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    if len(array) > 24:
        raise ValueError("Exact sign-flip is bounded to 24 network values")
    sums = np.array([0.0], dtype=np.float64)
    observed = 0.0
    for value in array:
        # Match the enumeration order so the observed sign pattern is counted.
        observed += value
        sums = np.concatenate((sums + value, sums - value))
    tolerance = 1e-15
    if alternative == "greater":
        return float(np.mean(sums >= observed - tolerance))
    if alternative == "less":
        return float(np.mean(sums <= observed + tolerance))
    if alternative == "two-sided":
        return float(np.mean(np.abs(sums) >= abs(observed) - tolerance))
    raise ValueError(f"Unsupported alternative: {alternative}")


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = float((count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def stable_seed(*parts: object) -> int:
    payload = ":".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
