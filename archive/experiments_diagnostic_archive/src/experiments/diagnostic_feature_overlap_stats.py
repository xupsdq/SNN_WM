from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def bootstrap_mean_ci(values: Iterable[float], *, n_boot: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    boot = np.zeros(int(n_boot), dtype=np.float64)
    for idx in range(int(n_boot)):
        sample_idx = rng.integers(0, arr.size, size=arr.size)
        boot[idx] = float(arr[sample_idx].mean())
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize_discovery_metrics(
    window_df: pd.DataFrame,
    probe_summary_df: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "mean",
                "ci_low",
                "ci_high",
                "n",
            ]
        )
    metrics = {
        "cross_boundary_rate": window_df["cross_boundary_flag"].to_numpy(dtype=np.float64),
        "trace_saved_rate": window_df["trace_saved_flag"].to_numpy(dtype=np.float64),
        "boundary_margin_mean": window_df["boundary_margin"].to_numpy(dtype=np.float64),
        "nonzero_window_rate": probe_summary_df["nonzero_window_rate"].to_numpy(dtype=np.float64) if not probe_summary_df.empty else np.zeros(1, dtype=np.float64),
        "positive_importance_mean": probe_summary_df["positive_importance_mean"].to_numpy(dtype=np.float64) if not probe_summary_df.empty else np.zeros(1, dtype=np.float64),
        "positive_importance_max": probe_summary_df["positive_importance_max"].to_numpy(dtype=np.float64) if not probe_summary_df.empty else np.zeros(1, dtype=np.float64),
        "projected_nonzero_area": probe_summary_df["projected_nonzero_area"].to_numpy(dtype=np.float64) if not probe_summary_df.empty else np.zeros(1, dtype=np.float64),
    }
    rows = []
    for idx, (metric, values) in enumerate(metrics.items(), start=1):
        ci_low, ci_high = bootstrap_mean_ci(values, n_boot=n_boot, seed=seed + idx)
        rows.append(
            {
                "metric": str(metric),
                "mean": float(np.nanmean(values)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": int(len(values)),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric"], kind="stable").reset_index(drop=True)


def summarize_sufficiency_metrics(
    eval_df: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    rows = []
    for condition, group in eval_df.groupby("condition", sort=True):
        acc = group["is_correct"].to_numpy(dtype=np.float64)
        gain = group["sufficiency_gain"].to_numpy(dtype=np.float64)
        margin_ratio = group["margin_retention_ratio"].to_numpy(dtype=np.float64)
        score_ratio = group["true_score_retention_ratio"].to_numpy(dtype=np.float64)
        acc_ci = bootstrap_mean_ci(acc, n_boot=n_boot, seed=seed + len(rows) * 10 + 1)
        gain_ci = bootstrap_mean_ci(gain, n_boot=n_boot, seed=seed + len(rows) * 10 + 2)
        rows.append(
            {
                "condition": str(condition),
                "accuracy_mean": float(np.nanmean(acc)),
                "accuracy_ci_low": acc_ci[0],
                "accuracy_ci_high": acc_ci[1],
                "sufficiency_gain_mean": float(np.nanmean(gain)),
                "sufficiency_gain_ci_low": gain_ci[0],
                "sufficiency_gain_ci_high": gain_ci[1],
                "margin_retention_ratio_mean": float(np.nanmean(margin_ratio)),
                "true_score_retention_ratio_mean": float(np.nanmean(score_ratio)),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition"], kind="stable").reset_index(drop=True)


def summarize_necessity_metrics(
    eval_df: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    rows = []
    for condition, group in eval_df.groupby("condition", sort=True):
        delta_margin = group["delta_margin"].to_numpy(dtype=np.float64)
        gain = group["necessity_gain"].to_numpy(dtype=np.float64)
        acc_drop = group["accuracy_drop"].to_numpy(dtype=np.float64)
        delta_ci = bootstrap_mean_ci(delta_margin, n_boot=n_boot, seed=seed + len(rows) * 10 + 1)
        gain_ci = bootstrap_mean_ci(gain, n_boot=n_boot, seed=seed + len(rows) * 10 + 2)
        rows.append(
            {
                "condition": str(condition),
                "delta_margin_mean": float(np.nanmean(delta_margin)),
                "delta_margin_ci_low": delta_ci[0],
                "delta_margin_ci_high": delta_ci[1],
                "necessity_gain_mean": float(np.nanmean(gain)),
                "necessity_gain_ci_low": gain_ci[0],
                "necessity_gain_ci_high": gain_ci[1],
                "accuracy_drop_mean": float(np.nanmean(acc_drop)),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition"], kind="stable").reset_index(drop=True)
