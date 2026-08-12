from __future__ import annotations

from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as _legacy

# Keep module-level names identical while Fig.6 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_overlap_gated_stsp_recruitment(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    _ensure_probe_trials(ctx, bank)
    rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], Any] = {}
    q = float(ctx.cfg.fig6e_stsp_group_quantile)
    requested_overlap_threshold = float(ctx.cfg.overlap_threshold)
    for r in _progress(bank.probe_trials.itertuples(index=False), total=len(bank.probe_trials), desc="fig6 overlap-gated STSP", enabled=ctx.cfg.show_progress):
        seq_idx = _sequence_index(bank, int(r.sequence_id))
        rho = compute_gain_ratio_map(
            bank.g_final[seq_idx].reshape(28, 28),
            bank.g_baseline[seq_idx].reshape(28, 28),
            eps=float(ctx.cfg.score_eps),
            clip_quantiles=tuple(ctx.cfg.gain_ratio_clip_quantiles),
            use_log=bool(ctx.cfg.score_use_log_gain),
        )
        _record_gain_ratio_audit(ctx, _gain_ratio_audit_row(ctx, int(r.sequence_id), rho, bank.g_final[seq_idx], bank.g_baseline[seq_idx]))
        local_score, local_valid = compute_entry_gated_stsp_score_map(rho, np.isfinite(rho))
        entry_mask = _probe_entry_mask(ctx, int(r.probe_image_id), mode=str(ctx.cfg.real_probe_entry_mode), cache=encode_cache)
        overlap_map, overlap_valid = compute_probe_overlap_map(entry_mask)
        valid_mask = np.asarray(local_valid, dtype=bool) & np.asarray(overlap_valid, dtype=bool) & np.isfinite(local_score)
        probe_spikes = _encode_sequence_cached(ctx, [int(r.probe_image_id)], ctx.cfg.probe_steps, encode_cache)
        dynamic_trace = _run_real_probe_layer1_capture(ctx, int(r.probe_image_id), bank.boundaries.get(int(r.sequence_id)), probe_spikes=probe_spikes)
        baseline_trace = _run_real_probe_layer1_capture(ctx, int(r.probe_image_id), None, probe_spikes=probe_spikes)
        for window_ms, window_steps in zip(ctx.cfg.score_early_windows_ms, ctx.cfg.score_early_window_steps):
            dynamic_count, dynamic_fired, _dynamic_latency = collapse_layer1_spikes_spatial(dynamic_trace, None, int(window_steps))
            baseline_count, baseline_fired, _baseline_latency = collapse_layer1_spikes_spatial(baseline_trace, None, int(window_steps))
            group_rows, group_lookup, overlap_threshold_used = _overlap_gated_group_metrics(
                local_score,
                overlap_map,
                valid_mask,
                dynamic_count,
                baseline_count,
                dynamic_fired,
                baseline_fired,
                stsp_group_quantile=q,
                overlap_threshold=requested_overlap_threshold,
            )
            for row in group_rows:
                row.update(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": int(r.sequence_id),
                        "probe_id": int(r.probe_id),
                        "probe_label": int(r.probe_label),
                        "early_window_ms": int(window_ms),
                        "stsp_group_quantile": q,
                        "overlap_threshold": float(overlap_threshold_used),
                    }
                )
                rows.append(row)
            interaction_rows.append(
                _overlap_gated_interaction_row(
                    ctx,
                    r,
                    int(window_ms),
                    q,
                    float(overlap_threshold_used),
                    group_lookup,
                )
            )
    _save_csv(ctx, pd.DataFrame(rows, columns=PANEL_E_OVERLAP_GATED_COLUMNS), ctx.metrics_dir / "panel_e_overlap_gated_stsp_recruitment.csv")
    _save_csv(ctx, pd.DataFrame(interaction_rows, columns=PANEL_E_OVERLAP_GATED_INTERACTION_COLUMNS), ctx.metrics_dir / "panel_e_overlap_gated_stsp_interaction.csv")
    ctx.completed_modules["overlap_gated_stsp_recruitment"] = True
