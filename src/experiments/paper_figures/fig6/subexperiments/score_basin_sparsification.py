from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.experiments.paper_figures.fig6.constants import PANEL_E_BASIN_COLUMNS
from src.experiments.paper_figures.fig6.subexperiments.helpers_1 import (
    _encode_sequence_cached,
    _ensure_probe_trials,
    _make_score_region_ping_masks,
    _ms_to_steps,
    _overlay_payload,
    _probe_entry_mask,
    _progress,
    _run_masked_ping_layer1_capture,
    _run_real_probe_layer1_capture,
    _save_csv,
    collapse_layer1_spikes_spatial,
    compute_basin_enrichment,
    compute_entry_gated_stsp_score_map,
    compute_gain_ratio_map,
)
from src.experiments.paper_figures.fig6.subexperiments.helpers_2 import _sequence_index
from src.experiments.paper_figures.fig6.types import ExperimentContext, PeakAmplifiedReentryBank



def compute_score_basin_sparsification(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    _ensure_probe_trials(ctx, bank)
    rows: list[dict[str, Any]] = []
    overlay_payload: dict[str, np.ndarray] = {}
    encode_cache: dict[tuple[Any, ...], Any] = {}
    radii = tuple(sorted({1, int(ctx.cfg.basin_radius), 3}))
    primary_steps = _ms_to_steps(int(ctx.cfg.primary_score_early_window_ms), ctx.cfg.dt)
    for seq_idx, meta in _progress(enumerate(bank.sequence_meta.itertuples(index=False)), total=len(bank.sequence_meta), desc="fig6 basin ping", enabled=ctx.cfg.show_progress):
        seq_id = int(meta.sequence_id)
        rho = compute_gain_ratio_map(bank.g_final[seq_idx].reshape(28, 28), bank.g_baseline[seq_idx].reshape(28, 28), eps=float(ctx.cfg.score_eps), clip_quantiles=tuple(ctx.cfg.gain_ratio_clip_quantiles), use_log=bool(ctx.cfg.score_use_log_gain))
        masks = _make_score_region_ping_masks(rho, float(ctx.cfg.basin_top_q), int(ctx.cfg.network_seed) + seq_id + 2900)
        for entry_condition, entry_mask in masks.items():
            score_map, valid_mask = compute_entry_gated_stsp_score_map(rho, entry_mask)
            _pred, _fire_ms, _total_current, _active_sites, trace = _run_masked_ping_layer1_capture(ctx, bank.boundaries.get(seq_id), entry_mask, float(ctx.cfg.ping_amp), int(ctx.cfg.ping_steps))
            spike_count, fired, _latency = collapse_layer1_spikes_spatial(trace, None, primary_steps)
            if not overlay_payload:
                overlay_payload.update(_overlay_payload("ping", seq_id, entry_condition, score_map, fired, entry_mask, rho))
            for radius in radii:
                row = compute_basin_enrichment(score_map, valid_mask, fired, radius=int(radius), top_q=float(ctx.cfg.basin_top_q))
                row.update({"network_seed": int(ctx.cfg.network_seed), "sequence_id": seq_id, "entry_type": "ping", "entry_condition": str(entry_condition), "basin_radius": int(radius), "top_score_quantile": float(ctx.cfg.basin_top_q)})
                rows.append(row)
    for r in _progress(bank.probe_trials.itertuples(index=False), total=len(bank.probe_trials), desc="fig6 basin real probe", enabled=ctx.cfg.show_progress):
        seq_idx = _sequence_index(bank, int(r.sequence_id))
        rho = compute_gain_ratio_map(bank.g_final[seq_idx].reshape(28, 28), bank.g_baseline[seq_idx].reshape(28, 28), eps=float(ctx.cfg.score_eps), clip_quantiles=tuple(ctx.cfg.gain_ratio_clip_quantiles), use_log=bool(ctx.cfg.score_use_log_gain))
        entry_mask = _probe_entry_mask(ctx, int(r.probe_image_id), mode=str(ctx.cfg.real_probe_entry_mode), cache=encode_cache)
        score_map, valid_mask = compute_entry_gated_stsp_score_map(rho, entry_mask)
        probe_spikes = _encode_sequence_cached(ctx, [int(r.probe_image_id)], ctx.cfg.probe_steps, encode_cache)
        trace = _run_real_probe_layer1_capture(ctx, int(r.probe_image_id), bank.boundaries.get(int(r.sequence_id)), probe_spikes=probe_spikes)
        spike_count, fired, _latency = collapse_layer1_spikes_spatial(trace, None, primary_steps)
        if not overlay_payload:
            overlay_payload.update(_overlay_payload("real_probe", int(r.sequence_id), str(ctx.cfg.real_probe_entry_mode), score_map, fired, entry_mask, rho))
        for radius in radii:
            row = compute_basin_enrichment(score_map, valid_mask, fired, radius=int(radius), top_q=float(ctx.cfg.basin_top_q))
            row.update({"network_seed": int(ctx.cfg.network_seed), "sequence_id": int(r.sequence_id), "entry_type": "real_probe", "entry_condition": str(ctx.cfg.real_probe_entry_mode), "basin_radius": int(radius), "top_score_quantile": float(ctx.cfg.basin_top_q)})
            rows.append(row)
    out = pd.DataFrame(rows, columns=PANEL_E_BASIN_COLUMNS)
    _save_csv(ctx, out, ctx.metrics_dir / "panel_e_score_basin_sparsification.csv")
    if not overlay_payload:
        overlay_payload = {"score_map": np.zeros((28, 28), dtype=np.float32), "fired_map": np.zeros((28, 28), dtype=np.uint8), "entry_mask": np.zeros((28, 28), dtype=np.uint8)}
    np.savez_compressed(ctx.raw_dir / "panel_e_example_score_spike_overlay.npz", **overlay_payload)
    ctx.output_files["panel_e_example_score_spike_overlay"] = "data/raw/panel_e_example_score_spike_overlay.npz"
    ctx.completed_modules["score_basin_sparsification"] = True
