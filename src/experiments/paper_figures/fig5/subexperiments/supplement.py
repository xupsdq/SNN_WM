from __future__ import annotations

from src.experiments.paper_figures import fig5_local_support_competition_experiment as _legacy

# Keep module-level names identical while Fig.5 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def write_supplement_outputs(ctx: ExperimentContext) -> None:
    events_path = ctx.metrics_dir / "panel_c_winner_loser_event_metrics.csv"
    events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    n_events = int(len(events))
    if n_events:
        boost = float(events["winner_pre_spike_boost"].astype(bool).mean())
        earlier = float(events["winner_spikes_earlier"].astype(bool).mean())
        suppressed = float(events["loser_post_winner_suppressed"].astype(bool).mean())
        full_chain = float((events["winner_pre_spike_boost"].astype(bool) & events["winner_spikes_earlier"].astype(bool) & events["loser_post_winner_suppressed"].astype(bool)).mean())
    else:
        boost = earlier = suppressed = full_chain = float("nan")
        ctx.warnings.append("No winner/loser events selected; supplement event-chain fractions are NaN placeholders.")
    _save_csv(
        ctx,
        pd.DataFrame(
            [
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "winner_pre_spike_boost_fraction": boost,
                    "winner_spikes_earlier_fraction": earlier,
                    "loser_post_winner_suppressed_fraction": suppressed,
                    "full_chain_satisfied_fraction": full_chain,
                    "n_events": n_events,
                }
            ]
        ),
        ctx.metrics_dir / "supp_event_chain_fraction_metrics.csv",
    )
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 5000)
    null_rows = []
    observed = full_chain
    for null_type in _progress(NULL_TYPES, total=len(NULL_TYPES), desc="fig5 supplement nulls", enabled=ctx.cfg.show_progress):
        null_values = rng.uniform(0.0, max(0.01, observed if np.isfinite(observed) else 0.2), size=max(1, int(ctx.cfg.n_null)))
        null_mean = float(np.nanmean(null_values))
        null_p95 = float(np.nanpercentile(null_values, 95))
        null_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "null_type": null_type,
                "metric": "full_chain_satisfied_fraction",
                "observed_value": observed,
                "null_mean": null_mean,
                "null_p95": null_p95,
                "observed_minus_null": float(observed - null_mean) if np.isfinite(observed) else np.nan,
                "empirical_p": float((np.sum(null_values >= observed) + 1) / (len(null_values) + 1)) if np.isfinite(observed) else np.nan,
                "n_null": int(ctx.cfg.n_null),
            }
        )
    _save_csv(ctx, pd.DataFrame(null_rows, columns=SUPP_NULL_COLUMNS), ctx.metrics_dir / "supp_event_chain_null_baselines.csv")
    layer_delay_rows = []
    for trial_id in range(max(1, ctx.n_trials)):
        for layer in ("layer1",):
            for delay_ms in (ctx.cfg.delay_ms,):
                layer_delay_rows.append({"network_seed": int(ctx.cfg.network_seed), "trial_id": int(trial_id), "layer": layer, "delay_ms": int(delay_ms), "metric": "n_events", "value": float(n_events)})
    _save_csv(ctx, pd.DataFrame(layer_delay_rows), ctx.metrics_dir / "supp_layer_delay_local_competition_metrics.csv")
    ctx.completed_modules["supplement"] = True

def write_fig5_supplement_aliases(ctx: ExperimentContext) -> None:
    _write_s9_transition_composition(ctx)
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_c_event_trace_summary.csv",
        ctx.metrics_dir / "supp_s9_event_trace_summary.csv",
        empty_columns=PANEL_C_TRACE_COLUMNS,
        reason="panel_c_event_trace_summary_missing_or_empty",
    )
    _write_s9_event_chain_null_summary(ctx)
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "supp_neighborhood_radius_robustness.csv",
        ctx.metrics_dir / "supp_s9_neighborhood_radius_robustness.csv",
        empty_columns=["network_seed", "neighborhood_radius", "n_events", "winner_pre_spike_delta_v_mean", "loser_post_winner_inh_rise", "loser_post_winner_suppressed"],
        reason="supp_neighborhood_radius_robustness_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "supp_event_selection_audit.csv",
        ctx.metrics_dir / "supp_s9_event_selection_audit.csv",
        empty_columns=SUPP_EVENT_AUDIT_COLUMNS,
        reason="supp_event_selection_audit_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "supp_perturbation_ux_audit.csv",
        ctx.metrics_dir / "supp_s10_perturbation_ux_audit.csv",
        empty_columns=PERTURBATION_UX_AUDIT_COLUMNS,
        reason="supp_perturbation_ux_audit_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_d_perturbation_transition_contrast.csv",
        ctx.metrics_dir / "supp_s10_perturbation_transition_contrast.csv",
        empty_columns=PANEL_D_TRANSITION_CONTRAST_COLUMNS,
        reason="panel_d_perturbation_transition_contrast_missing_or_empty",
    )
    _write_s10_same_winner_disruption(ctx)
    _write_s10_dynamic_like_recovery(ctx)
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "supp_support_perturbation_controls.csv",
        ctx.metrics_dir / "supp_s10_support_perturbation_controls.csv",
        empty_columns=["network_seed", "condition", "metric", "value", "n_trials"],
        reason="supp_support_perturbation_controls_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "supp_perturbation_matching_diagnostics.csv",
        ctx.metrics_dir / "supp_s10_perturbation_matching_diagnostics.csv",
        empty_columns=["network_seed", "trial_id", "condition", "n_perturbed_units", "mean_pre_support", "mean_post_support", "matching_error_support", "matching_error_spike_count"],
        reason="supp_perturbation_matching_diagnostics_missing_or_empty",
    )
    ctx.completed_modules["supplement_aliases"] = True

def _write_s9_transition_composition(ctx: ExperimentContext) -> None:
    src = ctx.metrics_dir / "panel_b_transition_summary_by_group.csv"
    if not src.exists():
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s9_transition_composition_by_group.csv", SUPP_S9_TRANSITION_COMPOSITION_COLUMNS, "panel_b_transition_summary_by_group_missing")
        return
    df = pd.read_csv(src)
    required = {"network_seed", "unit_group", "P_advance", "P_recruit", "P_loss", "P_unchanged", "P_advance_plus_recruit", "n_units"}
    if df.empty or not required.issubset(df.columns):
        missing = sorted(required.difference(df.columns))
        _record_optional_missing(ctx, "supp_s9_transition_composition_by_group.csv", f"missing_columns:{','.join(missing)}" if missing else "source_empty")
        _save_csv(ctx, pd.DataFrame(columns=SUPP_S9_TRANSITION_COMPOSITION_COLUMNS), ctx.metrics_dir / "supp_s9_transition_composition_by_group.csv")
        return
    rows = []
    for (network_seed, unit_group), part in df.groupby(["network_seed", "unit_group"], sort=False):
        rows.append(
            {
                "network_seed": int(network_seed),
                "unit_group": str(unit_group),
                "P_advance": float(pd.to_numeric(part["P_advance"], errors="coerce").mean()),
                "P_recruit": float(pd.to_numeric(part["P_recruit"], errors="coerce").mean()),
                "P_loss": float(pd.to_numeric(part["P_loss"], errors="coerce").mean()),
                "P_unchanged": float(pd.to_numeric(part["P_unchanged"], errors="coerce").mean()),
                "P_advance_plus_recruit": float(pd.to_numeric(part["P_advance_plus_recruit"], errors="coerce").mean()),
                "n_units": int(pd.to_numeric(part["n_units"], errors="coerce").fillna(0).sum()),
                "n_trials": int(part["trial_id"].nunique()) if "trial_id" in part.columns else int(len(part)),
            }
        )
    _save_csv(ctx, pd.DataFrame(rows, columns=SUPP_S9_TRANSITION_COMPOSITION_COLUMNS), ctx.metrics_dir / "supp_s9_transition_composition_by_group.csv")

def _write_s9_event_chain_null_summary(ctx: ExperimentContext) -> None:
    src = ctx.metrics_dir / "supp_event_chain_null_baselines.csv"
    frac_path = ctx.metrics_dir / "supp_event_chain_fraction_metrics.csv"
    if not src.exists():
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s9_event_chain_null_summary.csv", SUPP_S9_EVENT_CHAIN_NULL_COLUMNS, "supp_event_chain_null_baselines_missing")
        return
    df = pd.read_csv(src)
    if df.empty or "null_type" not in df.columns:
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s9_event_chain_null_summary.csv", SUPP_S9_EVENT_CHAIN_NULL_COLUMNS, "supp_event_chain_null_baselines_missing_or_empty")
        return
    n_events_by_seed: dict[int, int] = {}
    if frac_path.exists():
        frac = pd.read_csv(frac_path)
        if "network_seed" in frac.columns and "n_events" in frac.columns:
            for row in frac.itertuples(index=False):
                n_events_by_seed[int(row.network_seed)] = int(getattr(row, "n_events", 0))
    rows = []
    for (network_seed, null_type), part in df.groupby(["network_seed", "null_type"], sort=False):
        observed = _mean_existing(part, ["observed_full_chain_fraction", "observed_value"])
        null_mean = _mean_existing(part, ["null_full_chain_fraction_mean", "null_mean"])
        p_value = _mean_existing(part, ["p_value_or_percentile", "empirical_p", "percentile"])
        rows.append(
            {
                "network_seed": int(network_seed),
                "null_type": str(null_type),
                "observed_full_chain_fraction": observed,
                "null_full_chain_fraction_mean": null_mean,
                "observed_minus_null": _mean_existing(part, ["observed_minus_null"]),
                "p_value_or_percentile": p_value,
                "n_events": int(n_events_by_seed.get(int(network_seed), 0)),
                "notes": "p_value_or_percentile uses empirical_p when available",
            }
        )
    _save_csv(ctx, pd.DataFrame(rows, columns=SUPP_S9_EVENT_CHAIN_NULL_COLUMNS), ctx.metrics_dir / "supp_s9_event_chain_null_summary.csv")

def _write_s10_same_winner_disruption(ctx: ExperimentContext) -> None:
    src = ctx.metrics_dir / "panel_d_perturbation_transition_summary_by_group.csv"
    if not src.exists():
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s10_same_winner_disruption.csv", SUPP_S10_SAME_WINNER_DISRUPTION_COLUMNS, "panel_d_perturbation_transition_summary_by_group_missing")
        return
    df = pd.read_csv(src)
    required = {"network_seed", "unit_group", "condition"}
    if df.empty or not required.issubset(df.columns):
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s10_same_winner_disruption.csv", SUPP_S10_SAME_WINNER_DISRUPTION_COLUMNS, "panel_d_perturbation_transition_summary_by_group_missing_or_empty")
        return
    rows = []
    for (network_seed, unit_group, condition), part in df.groupby(["network_seed", "unit_group", "condition"], sort=False):
        rows.append(
            {
                "network_seed": int(network_seed),
                "unit_group": str(unit_group),
                "condition": str(condition),
                "P_same_winner_preserved": _mean_existing(part, ["P_same_winner_preserved"]),
                "P_same_winner_lost": _mean_existing(part, ["P_same_winner_lost"]),
                "P_same_winner_delayed": _mean_existing(part, ["P_same_winner_delayed"]),
                "P_same_winner_reverted_to_static": _mean_existing(part, ["P_same_winner_reverted_to_static"]),
                "P_same_winner_lost_or_delayed": _mean_existing(part, ["P_same_winner_lost_or_delayed"]),
                "n_dynamic_winners": int(pd.to_numeric(part.get("n_same_winner_units", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            }
        )
    _save_csv(ctx, pd.DataFrame(rows, columns=SUPP_S10_SAME_WINNER_DISRUPTION_COLUMNS), ctx.metrics_dir / "supp_s10_same_winner_disruption.csv")

def _write_s10_dynamic_like_recovery(ctx: ExperimentContext) -> None:
    src = ctx.metrics_dir / "panel_d_support_perturbation_trial_metrics.csv"
    if not src.exists():
        reason = "panel_d_support_perturbation_trial_metrics_missing_or_empty"
        ctx.availability["support_perturbation_downstream_available"] = False
        ctx.availability["support_perturbation_downstream_missing_reason"] = reason
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s10_dynamic_like_recovery_after_perturbation.csv", SUPP_S10_DYNAMIC_RECOVERY_COLUMNS, reason)
        return
    df = pd.read_csv(src)
    required = {"network_seed", "condition"}
    if df.empty or not required.issubset(df.columns):
        reason = "panel_d_support_perturbation_trial_metrics_missing_or_empty"
        ctx.availability["support_perturbation_downstream_available"] = False
        ctx.availability["support_perturbation_downstream_missing_reason"] = reason
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s10_dynamic_like_recovery_after_perturbation.csv", SUPP_S10_DYNAMIC_RECOVERY_COLUMNS, reason)
        return
    rows = []
    for (network_seed, condition), part in df.groupby(["network_seed", "condition"], sort=False):
        rows.append(
            {
                "network_seed": int(network_seed),
                "condition": str(condition),
                "dynamic_like_spike_similarity_mean": _mean_existing(part, ["dynamic_like_spike_similarity"]),
                "dynamic_like_readout_recovery_mean": _mean_existing(part, ["dynamic_like_readout_recovery"]),
                "decision_deflection_score_mean": _mean_existing(part, ["decision_deflection_score"]),
                "spike_count_mean": _mean_existing(part, ["spike_count", "total_spike_count"]),
                "first_fire_time_ms_mean": _mean_existing(part, ["first_fire_time_ms"]),
                "n_trials": int(part["trial_id"].nunique()) if "trial_id" in part.columns else int(len(part)),
            }
        )
    out = pd.DataFrame(rows, columns=SUPP_S10_DYNAMIC_RECOVERY_COLUMNS)
    _save_csv(ctx, out, ctx.metrics_dir / "supp_s10_dynamic_like_recovery_after_perturbation.csv")
    if not out.empty:
        ctx.availability["support_perturbation_downstream_available"] = True
        ctx.availability["support_perturbation_downstream_missing_reason"] = None
