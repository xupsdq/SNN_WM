from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.experiments.paper_figures.fig6.constants import (
    PANEL_F_HIGH_STSP_ABLATION_COLUMNS,
    PANEL_F_HIGH_STSP_ABLATION_SUMMARY_COLUMNS,
)
from src.experiments.paper_figures.fig6.subexperiments.helpers_1 import (
    _ablation_condition_metrics,
    _encode_sequence_cached,
    _ensure_probe_trials,
    _high_rho_site_mask,
    _matched_probe_removal_mask,
    _ms_to_steps,
    _probe_entry_mask,
    _progress,
    _remove_probe_sites_from_spikes,
    _removed_probe_energy,
    _run_real_probe_layer1_capture_batch,
    _save_csv,
    collapse_layer1_spikes_spatial,
    compute_entry_gated_stsp_score_map,
    compute_gain_ratio_map,
)
from src.experiments.paper_figures.fig6.subexperiments.helpers_2 import _nan_subtract, _sequence_index
from src.experiments.paper_figures.fig6.types import ExperimentContext, PeakAmplifiedReentryBank



def compute_high_stsp_overlap_ablation(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    _ensure_probe_trials(ctx, bank)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], Any] = {}
    primary_window = int(ctx.cfg.primary_score_early_window_ms)
    primary_steps = _ms_to_steps(primary_window, ctx.cfg.dt)
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 6006)
    for r in _progress(bank.probe_trials.itertuples(index=False), total=len(bank.probe_trials), desc="fig6 high-STSP overlap ablation", enabled=ctx.cfg.show_progress):
        seq_idx = _sequence_index(bank, int(r.sequence_id))
        rho = compute_gain_ratio_map(
            bank.g_final[seq_idx].reshape(28, 28),
            bank.g_baseline[seq_idx].reshape(28, 28),
            eps=float(ctx.cfg.score_eps),
            clip_quantiles=tuple(ctx.cfg.gain_ratio_clip_quantiles),
            use_log=bool(ctx.cfg.score_use_log_gain),
        )
        local_score, local_valid = compute_entry_gated_stsp_score_map(rho, np.isfinite(rho))
        valid_mask = np.asarray(local_valid, dtype=bool) & np.isfinite(local_score)
        entry_mask = _probe_entry_mask(ctx, int(r.probe_image_id), mode=str(ctx.cfg.real_probe_entry_mode), cache=encode_cache)
        high_rho_sites = _high_rho_site_mask(rho, float(ctx.cfg.stsp_group_quantile))
        remove_high = np.asarray(entry_mask, dtype=bool) & high_rho_sites
        matched = _matched_probe_removal_mask(entry_mask, high_rho_sites, int(remove_high.sum()), rng)
        probe_spikes = _encode_sequence_cached(ctx, [int(r.probe_image_id)], ctx.cfg.probe_steps, encode_cache)
        condition_masks = {
            "intact": np.zeros_like(remove_high, dtype=bool),
            "remove_high_stsp_overlap": remove_high,
            "matched_removal": matched,
        }
        condition_results: dict[str, dict[str, Any]] = {}
        if bool(getattr(ctx.cfg, "enable_high_stsp_ablation_batch", False)):
            condition_results = _ablation_condition_metrics_condition_batch(
                ctx,
                bank,
                r,
                condition_masks,
                probe_spikes,
                valid_mask,
                int(primary_steps),
            )
        else:
            for condition, remove_mask in condition_masks.items():
                manipulated = _remove_probe_sites_from_spikes(probe_spikes, remove_mask)
                condition_results[condition] = _ablation_condition_metrics(
                    ctx,
                    bank,
                    r,
                    manipulated,
                    valid_mask,
                    int(primary_steps),
                    remove_mask,
                    original_probe_spikes=probe_spikes,
                )
        for condition, result in condition_results.items():
            result.update(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": int(r.sequence_id),
                    "probe_id": int(r.probe_id),
                    "probe_label": int(r.probe_label),
                    "condition": condition,
                    "early_window_ms": primary_window,
                }
            )
            rows.append(result)
        intact = condition_results.get("intact", {})
        for loss_condition, condition in (("high_stsp_overlap", "remove_high_stsp_overlap"), ("matched_removal", "matched_removal")):
            current = condition_results.get(condition, {})
            summary_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": int(r.sequence_id),
                    "probe_id": int(r.probe_id),
                    "probe_label": int(r.probe_label),
                    "early_window_ms": primary_window,
                    "loss_condition": loss_condition,
                    "loss_delta_spike_probability": _nan_subtract(intact.get("delta_spike_probability"), current.get("delta_spike_probability")),
                    "loss_mean_delta_spike_count": _nan_subtract(intact.get("mean_delta_spike_count"), current.get("mean_delta_spike_count")),
                    "removed_active_area": current.get("removed_active_area", np.nan),
                    "removed_input_energy": current.get("removed_input_energy", np.nan),
                }
            )
    detail_df = pd.DataFrame(rows, columns=PANEL_F_HIGH_STSP_ABLATION_COLUMNS)
    summary_df = pd.DataFrame(summary_rows, columns=PANEL_F_HIGH_STSP_ABLATION_SUMMARY_COLUMNS)
    _save_csv(ctx, detail_df, ctx.metrics_dir / "panel_a_high_stsp_overlap_ablation.csv")
    _save_csv(ctx, summary_df, ctx.metrics_dir / "panel_a_high_stsp_overlap_ablation_summary.csv")
    # Legacy aliases keep older plotting bundles readable while Panel A is the active contract.
    _save_csv(ctx, detail_df, ctx.metrics_dir / "panel_f_high_stsp_overlap_ablation.csv")
    _save_csv(ctx, summary_df, ctx.metrics_dir / "panel_f_high_stsp_overlap_ablation_summary.csv")
    ctx.completed_modules["high_stsp_overlap_ablation"] = True


def _ablation_condition_metrics_condition_batch(
    ctx: ExperimentContext,
    bank: PeakAmplifiedReentryBank,
    trial: Any,
    condition_masks: Mapping[str, np.ndarray],
    probe_spikes: Any,
    valid_mask: np.ndarray,
    early_window_steps: int,
) -> dict[str, dict[str, Any]]:
    condition_names = list(condition_masks.keys())
    manipulated_spikes = [_remove_probe_sites_from_spikes(probe_spikes, condition_masks[name]) for name in condition_names]
    batched_spikes = torch.cat([spikes for spikes in manipulated_spikes for _ in range(2)], dim=0).contiguous()
    dynamic_boundary = bank.boundaries.get(int(trial.sequence_id))
    boundaries = [boundary for _name in condition_names for boundary in (dynamic_boundary, None)]
    traces = _run_real_probe_layer1_capture_batch(ctx, boundaries, batched_spikes)
    valid = np.asarray(valid_mask, dtype=bool)
    out: dict[str, dict[str, Any]] = {}
    for condition_index, condition in enumerate(condition_names):
        remove_mask = condition_masks[condition]
        dynamic_trace = traces[condition_index * 2]
        baseline_trace = traces[condition_index * 2 + 1]
        dynamic_count, dynamic_fired, _dynamic_latency = collapse_layer1_spikes_spatial(dynamic_trace, None, int(early_window_steps))
        baseline_count, baseline_fired, _baseline_latency = collapse_layer1_spikes_spatial(baseline_trace, None, int(early_window_steps))
        if not np.any(valid):
            out[condition] = {
                "removed_active_area": int(np.asarray(remove_mask, dtype=bool).sum()),
                "removed_input_energy": _removed_probe_energy(probe_spikes, remove_mask),
                "dynamic_spike_probability": np.nan,
                "baseline_spike_probability": np.nan,
                "delta_spike_probability": np.nan,
                "mean_delta_spike_count": np.nan,
            }
            continue
        dyn_prob = float(np.mean(np.asarray(dynamic_fired, dtype=bool)[valid]))
        base_prob = float(np.mean(np.asarray(baseline_fired, dtype=bool)[valid]))
        out[condition] = {
            "removed_active_area": int(np.asarray(remove_mask, dtype=bool).sum()),
            "removed_input_energy": _removed_probe_energy(probe_spikes, remove_mask),
            "dynamic_spike_probability": dyn_prob,
            "baseline_spike_probability": base_prob,
            "delta_spike_probability": float(dyn_prob - base_prob),
            "mean_delta_spike_count": float(np.nanmean(np.asarray(dynamic_count, dtype=float)[valid] - np.asarray(baseline_count, dtype=float)[valid])),
        }
    return out
