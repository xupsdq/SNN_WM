from __future__ import annotations

from src.experiments.paper_figures import fig5_local_support_competition_experiment as _legacy

# Keep module-level names identical while Fig.5 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_early_firing_transition_metrics(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> None:
    rows: list[dict[str, Any]] = []
    for trial in _progress(bank.trials.itertuples(index=False), total=len(bank.trials), desc="fig5 early firing", enabled=ctx.cfg.show_progress):
        trial_id = int(trial.trial_id)
        groups = bank.unit_groups[bank.unit_groups["trial_id"].eq(trial_id)]
        dynamic = bank.branch_traces[trial_id]["dynamic_intact"]
        static = bank.branch_traces[trial_id]["static_frozen"]
        first_dyn = _first_spike_map(dynamic.spikes)
        first_sta = _first_spike_map(static.spikes)
        early_dyn = dynamic.spikes[: ctx.cfg.early_window_steps].sum(axis=0)
        early_sta = static.spikes[: ctx.cfg.early_window_steps].sum(axis=0)
        for row in groups.itertuples(index=False):
            r = int(row.row)
            c = int(row.col)
            fd = int(first_dyn[r, c])
            fs = int(first_sta[r, c])
            transition = _transition_type(fd, fs)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": trial_id,
                    "unit_id": int(row.unit_id),
                    "unit_group": str(row.unit_group),
                    "early_window_ms": int(ctx.cfg.early_window_ms),
                    "transition_type": transition,
                    "first_spike_dynamic": fd,
                    "first_spike_static": fs,
                    "delta_first_spike_latency": _latency_delta(fd, fs),
                    "early_spike_count_dynamic": float(early_dyn[r, c]),
                    "early_spike_count_static": float(early_sta[r, c]),
                    "delta_early_spike_count": float(early_dyn[r, c] - early_sta[r, c]),
                }
            )
    metrics = pd.DataFrame(rows, columns=PANEL_B_UNIT_COLUMNS)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_b_early_firing_transition_metrics.csv")
    _save_csv(ctx, _transition_summary(ctx.cfg.network_seed, metrics, ctx.cfg.early_window_ms), ctx.metrics_dir / "panel_b_transition_summary_by_group.csv")
    _save_csv(ctx, _early_window_robustness(ctx, bank), ctx.metrics_dir / "supp_early_window_robustness.csv")
    ctx.completed_modules["early_firing"] = True
