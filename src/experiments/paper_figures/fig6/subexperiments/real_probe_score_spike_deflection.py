from __future__ import annotations

from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as _legacy

# Keep module-level names identical while Fig.6 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_real_probe_score_spike_deflection(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    _ensure_probe_trials(ctx, bank)
    rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], Any] = {}
    for r in _progress(bank.probe_trials.itertuples(index=False), total=len(bank.probe_trials), desc="fig6 real probe score/spike", enabled=ctx.cfg.show_progress):
        seq_idx = _sequence_index(bank, int(r.sequence_id))
        rho = compute_gain_ratio_map(
            bank.g_final[seq_idx].reshape(28, 28),
            bank.g_baseline[seq_idx].reshape(28, 28),
            eps=float(ctx.cfg.score_eps),
            clip_quantiles=tuple(ctx.cfg.gain_ratio_clip_quantiles),
            use_log=bool(ctx.cfg.score_use_log_gain),
        )
        _record_gain_ratio_audit(ctx, _gain_ratio_audit_row(ctx, int(r.sequence_id), rho, bank.g_final[seq_idx], bank.g_baseline[seq_idx]))
        entry_mask = _probe_entry_mask(ctx, int(r.probe_image_id), mode=str(ctx.cfg.real_probe_entry_mode), cache=encode_cache)
        score_map, valid_mask = compute_entry_gated_stsp_score_map(rho, entry_mask)
        probe_spikes = _encode_sequence_cached(ctx, [int(r.probe_image_id)], ctx.cfg.probe_steps, encode_cache)
        dynamic_trace = _run_real_probe_layer1_capture(ctx, int(r.probe_image_id), bank.boundaries.get(int(r.sequence_id)), probe_spikes=probe_spikes)
        baseline_trace = _run_real_probe_layer1_capture(ctx, int(r.probe_image_id), None, probe_spikes=probe_spikes)
        _record_entry_score_audit(ctx, _entry_score_audit_row(ctx, int(r.sequence_id), "real_probe", "foreground", score_map, valid_mask, entry_mask, dynamic_trace))
        prior = bank.prior_updated_mask[seq_idx].reshape(28, 28).astype(bool)
        for window_ms, window_steps in zip(ctx.cfg.score_early_windows_ms, ctx.cfg.score_early_window_steps):
            dynamic_count, dynamic_fired, dynamic_latency = collapse_layer1_spikes_spatial(dynamic_trace, None, int(window_steps))
            baseline_count, baseline_fired, baseline_latency = collapse_layer1_spikes_spatial(baseline_trace, None, int(window_steps))
            metric_rows = compute_spike_deflection_metrics(
                score_map,
                valid_mask,
                dynamic_count,
                baseline_count,
                dynamic_fired=dynamic_fired,
                baseline_fired=baseline_fired,
                dynamic_latency_map=dynamic_latency,
                baseline_latency_map=baseline_latency,
                n_bins=int(ctx.cfg.score_n_bins),
            )
            for row in metric_rows:
                row.update(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": int(r.sequence_id),
                        "probe_id": int(r.probe_id),
                        "probe_label": int(r.probe_label),
                        "early_window_ms": int(window_ms),
                        "valid_site_count": int(np.asarray(valid_mask, dtype=bool).sum()),
                        "probe_active_area": int(np.asarray(entry_mask, dtype=bool).sum()),
                        "prior_updated_overlap_area": int(np.logical_and(entry_mask, prior).sum()),
                    }
                )
                rows.append(row)
    out = pd.DataFrame(rows, columns=PANEL_D_REAL_PROBE_SCORE_COLUMNS)
    _save_csv(ctx, out, ctx.metrics_dir / "panel_d_real_probe_score_spike_deflection.csv")
    ctx.completed_modules["real_probe_score_spike_deflection"] = True
