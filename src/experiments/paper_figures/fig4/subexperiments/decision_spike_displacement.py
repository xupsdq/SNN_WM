from __future__ import annotations

from src.experiments.paper_figures import fig4_overlap_reentry_experiment as _legacy

# Keep module-level names identical while Fig.4 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_decision_spike_displacement(ctx: ExperimentContext, bank: OverlapReentryDMSBank) -> None:
    time_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for _, pair in bank.pair_trials.iterrows():
        pair_id = int(pair["pair_id"])
        dyn_row = _cond_row(bank.condition_metrics, pair_id, "full_dynamic")
        sta_row = _cond_row(bank.condition_metrics, pair_id, "full_static")
        for condition in CORE_CONDITIONS:
            dpi_t, s_dyn, s_sta = _dpi_timecourse(bank, pair_id, condition)
            for t, value in enumerate(dpi_t):
                time_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": pair_id,
                        "condition": condition,
                        "time_step": int(t),
                        "time_ms": float(t * ctx.cfg.dt / ms),
                        "S_dyn_L3": float(s_dyn[t]),
                        "S_sta_L3": float(s_sta[t]),
                        "DPI_L3_t": float(value),
                        "overlap_bin": str(pair["overlap_bin"]),
                        "similarity_bin": str(pair["similarity_bin"]),
                    }
                )
            summary_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "pair_id": pair_id,
                    "condition": condition,
                    "mean_DPI_L3": float(np.nanmean(dpi_t)) if len(dpi_t) else float("nan"),
                    "first_spike_time_dynamic": int(dyn_row["first_fire_time"]),
                    "first_spike_time_static": int(sta_row["first_fire_time"]),
                    "decision_spike_advance": int(sta_row["first_fire_time"]) - int(dyn_row["first_fire_time"]),
                    "overlap_bin": str(pair["overlap_bin"]),
                    "similarity_bin": str(pair["similarity_bin"]),
                }
            )
    _save_csv(ctx, pd.DataFrame(time_rows), ctx.metrics_dir / "panel_e_time_resolved_l3_displacement.csv")
    _save_csv(ctx, pd.DataFrame(summary_rows), ctx.metrics_dir / "panel_e_decision_spike_displacement.csv")
    ctx.completed_modules["decision_spike_displacement"] = True

def compute_probe_l3_trace_dpi_metrics(ctx: ExperimentContext, bank: OverlapPerturbationCompatibleBank) -> None:
    time_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for _, pair in bank.pair_trials.iterrows():
        pair_id = int(pair["pair_id"])
        dyn_row = _cond_row(bank.condition_metrics, pair_id, "full_dynamic")
        sta_row = _cond_row(bank.condition_metrics, pair_id, "full_static")
        dyn_trace = _trace(bank, pair_id, "full_dynamic")
        sta_trace = _trace(bank, pair_id, "full_static")
        for condition in CORE_CONDITIONS:
            cond_trace = _trace(bank, pair_id, condition)
            t_steps = min(int(cond_trace.shape[0]), int(dyn_trace.shape[0]), int(sta_trace.shape[0]))
            s_dyn_values: list[float] = []
            s_sta_values: list[float] = []
            dpi_values: list[float] = []
            for t in range(t_steps):
                cond_vec = normalize_pattern_vector(cond_trace[t])
                dyn_vec = normalize_pattern_vector(dyn_trace[t])
                sta_vec = normalize_pattern_vector(sta_trace[t])
                s_dyn = float(np.dot(cond_vec, dyn_vec))
                s_sta = float(np.dot(cond_vec, sta_vec))
                dpi = float(s_dyn - s_sta)
                s_dyn_values.append(s_dyn)
                s_sta_values.append(s_sta)
                dpi_values.append(dpi)
                time_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": pair_id,
                        "condition": condition,
                        "time_step": int(t),
                        "time_ms": float(t * ctx.cfg.dt / ms),
                        "S_dyn_L3": s_dyn,
                        "S_sta_L3": s_sta,
                        "DPI_L3_t": dpi,
                        "overlap_bin": str(pair["overlap_bin"]),
                        "similarity_bin": str(pair["similarity_bin"]),
                        "trace_object": "probe_l3_trace_s2p",
                        "pattern_normalization": "centered_l2",
                    }
                )
            summary_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "pair_id": pair_id,
                    "condition": condition,
                    "mean_DPI_L3": float(np.nanmean(dpi_values)) if dpi_values else float("nan"),
                    "mean_S_dyn_L3": float(np.nanmean(s_dyn_values)) if s_dyn_values else float("nan"),
                    "mean_S_sta_L3": float(np.nanmean(s_sta_values)) if s_sta_values else float("nan"),
                    "first_fire_time_dynamic": int(dyn_row["first_fire_time"]),
                    "first_fire_time_static": int(sta_row["first_fire_time"]),
                    "decision_spike_advance": int(sta_row["first_fire_time"]) - int(dyn_row["first_fire_time"]),
                    "overlap_bin": str(pair["overlap_bin"]),
                    "similarity_bin": str(pair["similarity_bin"]),
                    "trace_object": "probe_l3_trace_s2p",
                }
            )
    _save_csv(ctx, pd.DataFrame(time_rows), ctx.metrics_dir / "panel_e_time_resolved_l3_displacement.csv")
    _save_csv(ctx, pd.DataFrame(summary_rows), ctx.metrics_dir / "panel_e_decision_spike_displacement.csv")
    ctx.completed_modules["decision_spike_displacement"] = True
