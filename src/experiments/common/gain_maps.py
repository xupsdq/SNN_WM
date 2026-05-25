from __future__ import annotations

import numpy as np


def compute_gain_ratio_map(
    g_final: np.ndarray,
    g_baseline: np.ndarray,
    eps: float = 1e-6,
    clip_quantiles: tuple[float, float] | None = (0.01, 0.99),
    use_log: bool = False,
) -> np.ndarray:
    final = np.asarray(g_final, dtype=np.float64)
    baseline = np.asarray(g_baseline, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = final / np.maximum(baseline, float(eps))
    finite = ratio[np.isfinite(ratio)]
    if finite.size and clip_quantiles is not None:
        q0, q1 = tuple(float(v) for v in clip_quantiles)
        lo, hi = np.nanquantile(finite, [q0, q1])
        ratio = np.clip(ratio, lo, hi)
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)
    if use_log:
        ratio = np.log(np.maximum(ratio, float(eps)))
    return ratio.astype(np.float32)


__all__ = ["compute_gain_ratio_map"]
