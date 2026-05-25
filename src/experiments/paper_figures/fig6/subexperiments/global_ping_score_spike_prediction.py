from __future__ import annotations

from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as _legacy

# Keep module-level names identical while Fig.6 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_global_ping_score_spike_prediction(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    rows: list[dict[str, Any]] = []
    primary_window = int(ctx.cfg.primary_score_early_window_ms)
    primary_steps = _ms_to_steps(primary_window, ctx.cfg.dt)
    for seq_idx, meta in _progress(enumerate(bank.sequence_meta.itertuples(index=False)), total=len(bank.sequence_meta), desc="fig6 global ping score/spike", enabled=ctx.cfg.show_progress):
        seq_id = int(meta.sequence_id)
        rho = compute_gain_ratio_map(
            bank.g_final[seq_idx].reshape(28, 28),
            bank.g_baseline[seq_idx].reshape(28, 28),
            eps=float(ctx.cfg.score_eps),
            clip_quantiles=tuple(ctx.cfg.gain_ratio_clip_quantiles),
            use_log=bool(ctx.cfg.score_use_log_gain),
        )
        _record_gain_ratio_audit(ctx, _gain_ratio_audit_row(ctx, seq_id, rho, bank.g_final[seq_idx], bank.g_baseline[seq_idx]))
        global_ping_mask = np.isfinite(rho)
        score_map, valid_mask = compute_entry_gated_stsp_score_map(rho, global_ping_mask)
        _record_entry_score_audit(ctx, _entry_score_audit_row(ctx, seq_id, "global_ping", "global", score_map, valid_mask, global_ping_mask, None))
        _pred, _fire_ms, _total_current, _active_sites, layer1_trace = _run_masked_ping_layer1_capture(
            ctx,
            bank.boundaries.get(seq_id),
            global_ping_mask,
            float(ctx.cfg.global_ping_amp),
            int(ctx.cfg.global_ping_steps),
        )
        spike_count_map, fired_map, latency_map = collapse_layer1_spikes_spatial(layer1_trace, None, int(primary_steps))
        metric_rows = compute_score_quantile_metrics(
            score_map,
            valid_mask,
            spike_count_map,
            fired_map,
            n_bins=int(ctx.cfg.score_n_bins),
        )
        for row in metric_rows:
            row.update(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": seq_id,
                    "early_window_ms": int(primary_window),
                    "mean_first_spike_latency_ms": _mean_latency_ms(latency_map, valid_mask, ctx.cfg.dt),
                }
            )
            rows.append(row)
    out = pd.DataFrame(rows, columns=PANEL_C_GLOBAL_PING_SCORE_COLUMNS)
    _save_csv(ctx, out, ctx.metrics_dir / "panel_c_global_ping_score_spike_prediction.csv")
    ctx.completed_modules["global_ping_score_spike_prediction"] = True
