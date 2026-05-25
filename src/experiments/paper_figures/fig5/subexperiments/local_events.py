from __future__ import annotations

from src.experiments.paper_figures import fig5_local_support_competition_experiment as _legacy

# Keep module-level names identical while Fig.5 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_event_aligned_metrics(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> None:
    event_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    raw_trace_payload: dict[str, np.ndarray] = {}
    time_axis_steps = np.arange(-ctx.cfg.event_align_pre_steps, ctx.cfg.event_align_post_steps + 1, dtype=int)
    time_axis_ms = time_axis_steps.astype(float) * float(ctx.cfg.dt / ms)
    raw_trace_payload["time_axis_steps"] = time_axis_steps
    raw_trace_payload["time_axis_ms"] = time_axis_ms
    event_id = 0

    for trial in _progress(bank.trials.itertuples(index=False), total=len(bank.trials), desc="fig5 local events", enabled=ctx.cfg.show_progress):
        trial_id = int(trial.trial_id)
        groups = bank.unit_groups[bank.unit_groups["trial_id"].eq(trial_id)]
        dynamic = bank.branch_traces[trial_id]["dynamic_intact"]
        static = bank.branch_traces[trial_id]["static_frozen"]
        first_dyn = _first_spike_map(dynamic.spikes)
        first_sta = _first_spike_map(static.spikes)
        winners = groups[
            groups["unit_group"].eq("overlap_dominant")
            & groups["unit_id"].isin(_advanced_or_recruited_units(first_dyn, first_sta))
        ].copy()
        losers = groups[
            groups["unit_group"].isin(["probe_only_dominant", "balanced"])
            & groups["unit_id"].isin(_delayed_or_lost_units(first_dyn, first_sta))
        ].copy()
        if losers.empty:
            losers = groups[groups["unit_group"].eq("probe_only_dominant")].copy()
        for win in winners.sort_values("support_value", ascending=False).itertuples(index=False):
            loser = _nearest_loser(win, losers, ctx.cfg.local_kernel_radius)
            if loser is None:
                loser = _nearest_loser(win, losers, max(ctx.cfg.local_kernel_radius, 6))
            if loser is None:
                audit_rows.append(_event_audit_row(ctx, trial_id, event_id, "winner_loser_pair", False, "no_local_loser", win.unit_group, "", float(win.overlap_drive_score), float("nan")))
                continue
            t0 = int(first_dyn[int(win.row), int(win.col)])
            if t0 < 0:
                continue
            winner_delta_v = _aligned_delta(dynamic.v_effective[:, int(win.row), int(win.col)], static.v_effective[:, int(win.row), int(win.col)], t0, ctx)
            loser_delta_v = _aligned_delta(dynamic.v_effective[:, int(loser.row), int(loser.col)], static.v_effective[:, int(loser.row), int(loser.col)], t0, ctx)
            loser_inh = _aligned_delta(dynamic.inhibition[:, int(loser.row), int(loser.col)], static.inhibition[:, int(loser.row), int(loser.col)], t0, ctx)
            pre = slice(0, ctx.cfg.event_align_pre_steps)
            post = slice(ctx.cfg.event_align_pre_steps + 1, None)
            row = {
                "network_seed": int(ctx.cfg.network_seed),
                "trial_id": trial_id,
                "event_id": int(event_id),
                "winner_unit_idx": int(win.unit_id),
                "loser_unit_idx": int(loser.unit_id),
                "winner_group": str(win.unit_group),
                "loser_group": str(loser.unit_group),
                "winner_first_spike_dynamic": int(first_dyn[int(win.row), int(win.col)]),
                "winner_first_spike_static": int(first_sta[int(win.row), int(win.col)]),
                "loser_first_spike_dynamic": int(first_dyn[int(loser.row), int(loser.col)]),
                "loser_first_spike_static": int(first_sta[int(loser.row), int(loser.col)]),
                "winner_pre_spike_delta_v_mean": float(np.nanmean(winner_delta_v[pre])),
                "winner_pre_spike_boost": bool(np.nanmean(winner_delta_v[pre]) > 0.0),
                "winner_spikes_earlier": bool(_spikes_earlier(first_dyn[int(win.row), int(win.col)], first_sta[int(win.row), int(win.col)])),
                "loser_post_winner_delta_v_mean": float(np.nanmean(loser_delta_v[post])),
                "loser_post_winner_inh_rise": float(np.nanmean(loser_inh[post])),
                "loser_post_winner_suppressed": bool(np.nanmean(loser_delta_v[post]) < 0.0 or _is_loser_suppressed(first_dyn[int(loser.row), int(loser.col)], first_sta[int(loser.row), int(loser.col)])),
                "winner_loser_latency_gap": _latency_delta(first_dyn[int(loser.row), int(loser.col)], first_dyn[int(win.row), int(win.col)]),
                "neighborhood_radius": int(ctx.cfg.local_kernel_radius),
                "local_distance": float(abs(int(win.row) - int(loser.row)) + abs(int(win.col) - int(loser.col))),
            }
            event_rows.append(row)
            audit_rows.append(_event_audit_row(ctx, trial_id, event_id, "winner_loser_pair", True, "", win.unit_group, loser.unit_group, float(win.overlap_drive_score), float(loser.overlap_drive_score)))
            raw_trace_payload[f"event_{event_id}_winner_delta_v"] = winner_delta_v.astype(np.float32)
            raw_trace_payload[f"event_{event_id}_loser_delta_v"] = loser_delta_v.astype(np.float32)
            raw_trace_payload[f"event_{event_id}_loser_inhibition"] = loser_inh.astype(np.float32)
            for t_ms, value in zip(time_axis_ms, winner_delta_v):
                trace_rows.append(_trace_summary_row(ctx, t_ms, "winner_delta_v", value))
            for t_ms, value in zip(time_axis_ms, loser_delta_v):
                trace_rows.append(_trace_summary_row(ctx, t_ms, "loser_delta_v", value))
            for t_ms, value in zip(time_axis_ms, loser_inh):
                trace_rows.append(_trace_summary_row(ctx, t_ms, "loser_inhibition", value))
            event_id += 1
            if event_id >= max(1, len(bank.trials) * 3):
                break
    ctx.n_events = int(event_id)
    events = pd.DataFrame(event_rows, columns=PANEL_C_EVENT_COLUMNS)
    _save_csv(ctx, events, ctx.metrics_dir / "panel_c_winner_loser_event_metrics.csv")
    _save_csv(ctx, _event_trace_summary(ctx, trace_rows), ctx.metrics_dir / "panel_c_event_trace_summary.csv")
    _save_csv(ctx, pd.DataFrame(audit_rows, columns=SUPP_EVENT_AUDIT_COLUMNS), ctx.metrics_dir / "supp_event_selection_audit.csv")
    _save_csv(ctx, _neighborhood_radius_robustness(ctx, events), ctx.metrics_dir / "supp_neighborhood_radius_robustness.csv")
    if ctx.cfg.save_full_traces:
        np.savez_compressed(ctx.raw_dir / "panel_c_event_aligned_traces.npz", **raw_trace_payload)
        ctx.output_files["panel_c_event_aligned_traces"] = _rel(ctx.raw_dir / "panel_c_event_aligned_traces.npz", ctx.seed_dir)
    ctx.completed_modules["local_events"] = True
