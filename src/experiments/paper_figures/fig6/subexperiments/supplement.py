from __future__ import annotations

from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as _legacy

# Keep module-level names identical while Fig.6 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_supplement_outputs(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    """Active Fig.6 S11 reanalysis for overlap-gated STSP recruitment controls."""
    _ = bank
    _save_csv(ctx, _active_s11_score_input_ping_audit(ctx), ctx.metrics_dir / "supp_s11a_score_input_ping_audit.csv")
    _save_csv(ctx, _active_s11_global_ping_count_endpoint(ctx), ctx.metrics_dir / "supp_s11b_global_ping_count_endpoint.csv")
    _save_csv(ctx, _active_s11_real_probe_window_robustness(ctx), ctx.metrics_dir / "supp_s11c_real_probe_window_robustness.csv")
    _save_csv(ctx, _active_s11_overlap_interaction_window_robustness(ctx), ctx.metrics_dir / "supp_s11d_overlap_interaction_window_robustness.csv")
    _save_csv(ctx, _active_s11_overlap_site_availability(ctx), ctx.metrics_dir / "supp_s11e_overlap_site_availability.csv")
    _save_csv(ctx, _active_s11_ablation_paired_difference(ctx), ctx.metrics_dir / "supp_s11f_high_stsp_ablation_paired_difference.csv")
    ctx.completed_modules["supplement"] = True


def compute_legacy_supplement_outputs(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    unit_df = pd.read_csv(ctx.metrics_dir / "panel_b_peak_update_history.csv")
    source_df = pd.read_csv(ctx.metrics_dir / "panel_a_peak_source_attribution.csv")
    overlap_df = pd.read_csv(ctx.metrics_dir / "panel_c_peak_input_overlap_similarity.csv")
    compute_supp_update_recency_support_model(ctx, unit_df)
    _save_csv(ctx, unit_df[["network_seed", "sequence_id", "unit_id", "update_count"]].assign(entry_mask_mode=ctx.cfg.real_probe_entry_mode, exposure_threshold=ctx.cfg.foreground_threshold, notes="Unit update count from item entry-mask exposure."), ctx.metrics_dir / "supp_update_count_definition.csv")
    _save_csv(ctx, unit_df[["network_seed", "sequence_id", "unit_id", "last_update_position", "time_since_last_update"]].assign(recent_window=int(ctx.cfg.recent_window), notes="Recency is measured backward from the final sequence position."), ctx.metrics_dir / "supp_recency_definition.csv")
    _save_csv(ctx, _leave_one_out_timing_controls(ctx, source_df), ctx.metrics_dir / "supp_leave_one_out_timing_controls.csv")
    _save_csv(ctx, _peak_source_old_vs_recent(ctx, source_df), ctx.metrics_dir / "supp_peak_source_attribution_old_vs_recent.csv")
    _save_csv(ctx, _recent_overlap_window_robustness(ctx, overlap_df), ctx.metrics_dir / "supp_recent_overlap_window_robustness.csv")
    _save_csv(ctx, _random_window_overlap_controls(ctx, bank), ctx.metrics_dir / "supp_peak_overlap_origin_random_window_controls.csv")
    _save_csv(ctx, _matched_peak_comparison(ctx, bank.reentry_metrics), ctx.metrics_dir / "supp_matched_raw_overlap_peak_comparison.csv")
    _save_csv(ctx, _visual_energy_controls(ctx, bank.reentry_metrics, bank.downstream_metrics), ctx.metrics_dir / "supp_visual_energy_classpair_controls.csv")
    _save_csv(ctx, _alternative_peak_definitions(ctx, bank), ctx.metrics_dir / "supp_alternative_peak_definitions.csv")
    _save_csv(ctx, _global_support_controls(ctx, bank.reentry_metrics), ctx.metrics_dir / "supp_global_support_spike_count_controls.csv")
    _save_csv(ctx, _real_reentry_control_s0_static(ctx, bank.reentry_metrics), ctx.metrics_dir / "supp_real_reentry_control_S0_static.csv")
    _save_csv(ctx, _real_downstream_metric_definitions(ctx), ctx.metrics_dir / "supp_real_downstream_metric_definitions.csv")
    _save_csv(ctx, _trial_condition_audit(ctx), ctx.metrics_dir / "supp_trial_condition_audit.csv")
    ctx.completed_modules["legacy_supplement"] = True

def write_fig6_supplement_aliases(ctx: ExperimentContext) -> None:
    if not getattr(ctx.cfg, "run_legacy_supplement", False):
        return
    write_legacy_fig6_supplement_aliases(ctx)


def _active_s11_score_input_ping_audit(ctx: ExperimentContext) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gain = _read_csv_if_exists(ctx.metrics_dir / "fig6_gain_ratio_audit.csv")
    entry = _read_csv_if_exists(ctx.metrics_dir / "fig6_entry_score_audit.csv")
    ping = _read_csv_if_exists(ctx.metrics_dir / "panel_b_region_ping_readout_bias.csv")
    if gain is not None and not gain.empty:
        rows.extend(
            [
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "metric": "nonfinite_raw_count",
                    "condition": "gain_ratio",
                    "value": float(pd.to_numeric(gain.get("nonfinite_raw_count"), errors="coerce").fillna(0).sum()),
                    "unit": "count",
                    "n_rows": int(len(gain)),
                },
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "metric": "baseline_floor_count",
                    "condition": "gain_ratio",
                    "value": float(pd.to_numeric(gain.get("baseline_floor_count"), errors="coerce").fillna(0).sum()),
                    "unit": "count",
                    "n_rows": int(len(gain)),
                },
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "metric": "clipped_ratio_max",
                    "condition": "gain_ratio",
                    "value": float(pd.to_numeric(gain.get("clipped_ratio_max"), errors="coerce").max()),
                    "unit": "ratio",
                    "n_rows": int(len(gain)),
                },
            ]
        )
    if entry is not None and not entry.empty:
        grouped = entry.groupby(["entry_type", "entry_condition"], dropna=False)
        for (entry_type, condition), part in grouped:
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "metric": "mean_valid_site_count",
                    "condition": f"{entry_type}:{condition}",
                    "value": float(pd.to_numeric(part.get("valid_site_count"), errors="coerce").mean()),
                    "unit": "sites",
                    "n_rows": int(len(part)),
                }
            )
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "metric": "mean_entry_area",
                    "condition": f"{entry_type}:{condition}",
                    "value": float(pd.to_numeric(part.get("entry_area"), errors="coerce").mean()),
                    "unit": "pixels",
                    "n_rows": int(len(part)),
                }
            )
    if ping is not None and not ping.empty:
        for condition, part in ping.groupby("entry_condition", dropna=False):
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "metric": "ping_active_sites",
                    "condition": str(condition),
                    "value": float(pd.to_numeric(part.get("ping_active_sites"), errors="coerce").mean()),
                    "unit": "sites",
                    "n_rows": int(len(part)),
                }
            )
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "metric": "total_ping_current",
                    "condition": str(condition),
                    "value": float(pd.to_numeric(part.get("total_ping_current"), errors="coerce").mean()),
                    "unit": "current",
                    "n_rows": int(len(part)),
                }
            )
    return pd.DataFrame(rows)


def _active_s11_global_ping_count_endpoint(ctx: ExperimentContext) -> pd.DataFrame:
    df = _read_csv_if_exists(ctx.metrics_dir / "panel_c_global_ping_score_spike_prediction.csv")
    if df is None or df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = df.groupby("score_quantile_bin", dropna=False)
    for quantile, part in grouped:
        for metric, unit in (("mean_early_spike_count", "spike count"), ("spike_probability", "probability")):
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "metric": metric,
                    "condition": str(quantile),
                    "score_quantile_bin": str(quantile),
                    "value": float(pd.to_numeric(part.get(metric), errors="coerce").mean()),
                    "unit": unit,
                    "n_rows": int(len(part)),
                }
            )
    return pd.DataFrame(rows)


def _active_s11_real_probe_window_robustness(ctx: ExperimentContext) -> pd.DataFrame:
    df = _read_csv_if_exists(ctx.metrics_dir / "panel_d_real_probe_score_spike_deflection.csv")
    if df is None or df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for window_ms, part in df.groupby("early_window_ms", dropna=False):
        means = part.groupby("score_quantile_bin")["delta_spike_probability"].mean()
        q1 = float(means.get("Q1", np.nan))
        q5 = float(means.get("Q5", np.nan))
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "metric": "q5_minus_q1_delta_spike_probability",
                "condition": str(window_ms),
                "early_window_ms": float(window_ms),
                "value": q5 - q1,
                "unit": "probability difference",
                "q1_mean": q1,
                "q5_mean": q5,
                "n_rows": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def _active_s11_overlap_interaction_window_robustness(ctx: ExperimentContext) -> pd.DataFrame:
    df = _read_csv_if_exists(ctx.metrics_dir / "panel_e_overlap_gated_stsp_interaction.csv")
    if df is None or df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for window_ms, part in df.groupby("early_window_ms", dropna=False):
        vals = pd.to_numeric(part.get("interaction_delta"), errors="coerce").dropna()
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "metric": "interaction_delta",
                "condition": str(window_ms),
                "early_window_ms": float(window_ms),
                "value": float(vals.mean()) if len(vals) else np.nan,
                "unit": "probability difference",
                "n_valid": int(len(vals)),
                "n_rows": int(len(part)),
                "fraction_positive": float((vals > 0).mean()) if len(vals) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _active_s11_overlap_site_availability(ctx: ExperimentContext) -> pd.DataFrame:
    df = _read_csv_if_exists(ctx.metrics_dir / "panel_e_overlap_gated_stsp_recruitment.csv")
    if df is None or df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (stsp_group, overlap_group), part in df.groupby(["stsp_group", "overlap_group"], dropna=False):
        sites = pd.to_numeric(part.get("n_sites"), errors="coerce")
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "metric": "mean_sites",
                "condition": f"{stsp_group}:{overlap_group}",
                "stsp_group": str(stsp_group),
                "overlap_group": str(overlap_group),
                "value": float(sites.mean()),
                "unit": "sites",
                "median_sites": float(sites.median()),
                "nonzero_fraction": float((sites > 0).mean()),
                "n_rows": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def _active_s11_ablation_paired_difference(ctx: ExperimentContext) -> pd.DataFrame:
    path = ctx.metrics_dir / "panel_a_high_stsp_overlap_ablation_summary.csv"
    df = _read_csv_if_exists(path)
    if df is None or df.empty:
        df = _read_csv_if_exists(ctx.metrics_dir / "panel_f_high_stsp_overlap_ablation_summary.csv")
    if df is None or df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index=["network_seed", "sequence_id", "probe_id"],
        columns="loss_condition",
        values="loss_delta_spike_probability",
        aggfunc="mean",
    )
    required = {"high_stsp_overlap", "matched_removal"}
    if not required.issubset(set(pivot.columns)):
        return pd.DataFrame()
    out = pivot.reset_index()
    out["value"] = out["high_stsp_overlap"] - out["matched_removal"]
    out["metric"] = "high_stsp_overlap_minus_matched_loss"
    out["condition"] = "paired_difference"
    out["unit"] = "probability difference"
    return out[["network_seed", "sequence_id", "probe_id", "metric", "condition", "value", "unit", "high_stsp_overlap", "matched_removal"]]


def compute_score_shuffle_null_extension(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    """Optional S11G: keep spike/entry maps fixed and shuffle the STSP score map."""
    _ensure_probe_trials(ctx, bank)
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 611)
    rows: list[dict[str, Any]] = []
    primary_window = int(ctx.cfg.primary_score_early_window_ms)
    primary_steps = _ms_to_steps(primary_window, ctx.cfg.dt)
    encode_cache: dict[tuple[Any, ...], Any] = {}
    for seq_idx, meta in _progress(enumerate(bank.sequence_meta.itertuples(index=False)), total=len(bank.sequence_meta), desc="fig6 S11G score-shuffle global", enabled=ctx.cfg.show_progress):
        seq_id = int(meta.sequence_id)
        rho = compute_gain_ratio_map(
            bank.g_final[seq_idx].reshape(28, 28),
            bank.g_baseline[seq_idx].reshape(28, 28),
            eps=float(ctx.cfg.score_eps),
            clip_quantiles=tuple(ctx.cfg.gain_ratio_clip_quantiles),
            use_log=bool(ctx.cfg.score_use_log_gain),
        )
        score, valid = compute_entry_gated_stsp_score_map(rho, np.isfinite(rho))
        null_score, null_valid = compute_entry_gated_stsp_score_map(_shuffle_score_map(rho, rng), np.isfinite(rho))
        _pred, _fire_ms, _total_current, _active_sites, trace = _run_masked_ping_layer1_capture(
            ctx,
            bank.boundaries.get(seq_id),
            np.isfinite(rho),
            float(ctx.cfg.global_ping_amp),
            int(ctx.cfg.global_ping_steps),
        )
        spike_count, fired, _latency = collapse_layer1_spikes_spatial(trace, None, int(primary_steps))
        observed_rows = compute_score_quantile_metrics(score, valid, spike_count, fired, n_bins=int(ctx.cfg.score_n_bins))
        null_rows = compute_score_quantile_metrics(null_score, null_valid, spike_count, fired, n_bins=int(ctx.cfg.score_n_bins))
        observed = _q5_minus_q1_from_rows(observed_rows, "mean_early_spike_count")
        null = _q5_minus_q1_from_rows(null_rows, "mean_early_spike_count")
        rows.append(_extension_row(ctx, "global_ping_count_q5_q1", "C", observed, null, sequence_id=seq_id, probe_id=""))
    for r in _progress(bank.probe_trials.itertuples(index=False), total=len(bank.probe_trials), desc="fig6 S11G score-shuffle probe", enabled=ctx.cfg.show_progress):
        seq_idx = _sequence_index(bank, int(r.sequence_id))
        rho = compute_gain_ratio_map(
            bank.g_final[seq_idx].reshape(28, 28),
            bank.g_baseline[seq_idx].reshape(28, 28),
            eps=float(ctx.cfg.score_eps),
            clip_quantiles=tuple(ctx.cfg.gain_ratio_clip_quantiles),
            use_log=bool(ctx.cfg.score_use_log_gain),
        )
        null_rho = _shuffle_score_map(rho, rng)
        entry_mask = _probe_entry_mask(ctx, int(r.probe_image_id), mode=str(ctx.cfg.real_probe_entry_mode), cache=encode_cache)
        score, valid = compute_entry_gated_stsp_score_map(rho, entry_mask)
        null_score, null_valid = compute_entry_gated_stsp_score_map(null_rho, entry_mask)
        local_score, local_valid = compute_entry_gated_stsp_score_map(rho, np.isfinite(rho))
        null_local_score, null_local_valid = compute_entry_gated_stsp_score_map(null_rho, np.isfinite(rho))
        overlap_map, overlap_valid = compute_probe_overlap_map(entry_mask)
        valid_overlap = np.asarray(local_valid, dtype=bool) & np.asarray(overlap_valid, dtype=bool) & np.isfinite(local_score)
        null_valid_overlap = np.asarray(null_local_valid, dtype=bool) & np.asarray(overlap_valid, dtype=bool) & np.isfinite(null_local_score)
        probe_spikes = _encode_sequence_cached(ctx, [int(r.probe_image_id)], ctx.cfg.probe_steps, encode_cache)
        dynamic_trace = _run_real_probe_layer1_capture(ctx, int(r.probe_image_id), bank.boundaries.get(int(r.sequence_id)), probe_spikes=probe_spikes)
        baseline_trace = _run_real_probe_layer1_capture(ctx, int(r.probe_image_id), None, probe_spikes=probe_spikes)
        dynamic_count, dynamic_fired, dynamic_latency = collapse_layer1_spikes_spatial(dynamic_trace, None, int(primary_steps))
        baseline_count, baseline_fired, baseline_latency = collapse_layer1_spikes_spatial(baseline_trace, None, int(primary_steps))
        observed_probe = compute_spike_deflection_metrics(score, valid, dynamic_count, baseline_count, dynamic_fired=dynamic_fired, baseline_fired=baseline_fired, dynamic_latency_map=dynamic_latency, baseline_latency_map=baseline_latency, n_bins=int(ctx.cfg.score_n_bins))
        null_probe = compute_spike_deflection_metrics(null_score, null_valid, dynamic_count, baseline_count, dynamic_fired=dynamic_fired, baseline_fired=baseline_fired, dynamic_latency_map=dynamic_latency, baseline_latency_map=baseline_latency, n_bins=int(ctx.cfg.score_n_bins))
        observed = _q5_minus_q1_from_rows(observed_probe, "delta_spike_probability")
        null = _q5_minus_q1_from_rows(null_probe, "delta_spike_probability")
        rows.append(_extension_row(ctx, "real_probe_deflection_q5_q1", "D", observed, null, sequence_id=int(r.sequence_id), probe_id=int(r.probe_id)))
        observed_groups, observed_lookup, observed_threshold = _overlap_gated_group_metrics(
            local_score,
            overlap_map,
            valid_overlap,
            dynamic_count,
            baseline_count,
            dynamic_fired,
            baseline_fired,
            stsp_group_quantile=float(ctx.cfg.stsp_group_quantile),
            overlap_threshold=float(ctx.cfg.overlap_threshold),
        )
        null_groups, null_lookup, null_threshold = _overlap_gated_group_metrics(
            null_local_score,
            overlap_map,
            null_valid_overlap,
            dynamic_count,
            baseline_count,
            dynamic_fired,
            baseline_fired,
            stsp_group_quantile=float(ctx.cfg.stsp_group_quantile),
            overlap_threshold=float(ctx.cfg.overlap_threshold),
        )
        _ = observed_groups, null_groups
        observed_interaction = _overlap_gated_interaction_row(ctx, r, primary_window, float(ctx.cfg.stsp_group_quantile), float(observed_threshold), observed_lookup).get("interaction_delta", np.nan)
        null_interaction = _overlap_gated_interaction_row(ctx, r, primary_window, float(ctx.cfg.stsp_group_quantile), float(null_threshold), null_lookup).get("interaction_delta", np.nan)
        rows.append(_extension_row(ctx, "overlap_interaction", "E", observed_interaction, null_interaction, sequence_id=int(r.sequence_id), probe_id=int(r.probe_id)))
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "supp_s11g_score_shuffle_null.csv")
    ctx.completed_modules["score_shuffle_null"] = True


def compute_overlap_threshold_sensitivity_extension(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    """Optional S11H: recompute the E interaction over STSP/overlap thresholds."""
    _ensure_probe_trials(ctx, bank)
    rows: list[dict[str, Any]] = []
    primary_window = int(ctx.cfg.primary_score_early_window_ms)
    primary_steps = _ms_to_steps(primary_window, ctx.cfg.dt)
    q_values = (0.10, 0.20, 0.30)
    threshold_values = (0.02, 0.05, 0.10)
    encode_cache: dict[tuple[Any, ...], Any] = {}
    for r in _progress(bank.probe_trials.itertuples(index=False), total=len(bank.probe_trials), desc="fig6 S11H threshold sweep", enabled=ctx.cfg.show_progress):
        seq_idx = _sequence_index(bank, int(r.sequence_id))
        rho = compute_gain_ratio_map(
            bank.g_final[seq_idx].reshape(28, 28),
            bank.g_baseline[seq_idx].reshape(28, 28),
            eps=float(ctx.cfg.score_eps),
            clip_quantiles=tuple(ctx.cfg.gain_ratio_clip_quantiles),
            use_log=bool(ctx.cfg.score_use_log_gain),
        )
        entry_mask = _probe_entry_mask(ctx, int(r.probe_image_id), mode=str(ctx.cfg.real_probe_entry_mode), cache=encode_cache)
        local_score, local_valid = compute_entry_gated_stsp_score_map(rho, np.isfinite(rho))
        overlap_map, overlap_valid = compute_probe_overlap_map(entry_mask)
        valid_mask = np.asarray(local_valid, dtype=bool) & np.asarray(overlap_valid, dtype=bool) & np.isfinite(local_score)
        probe_spikes = _encode_sequence_cached(ctx, [int(r.probe_image_id)], ctx.cfg.probe_steps, encode_cache)
        dynamic_trace = _run_real_probe_layer1_capture(ctx, int(r.probe_image_id), bank.boundaries.get(int(r.sequence_id)), probe_spikes=probe_spikes)
        baseline_trace = _run_real_probe_layer1_capture(ctx, int(r.probe_image_id), None, probe_spikes=probe_spikes)
        dynamic_count, dynamic_fired, _dynamic_latency = collapse_layer1_spikes_spatial(dynamic_trace, None, int(primary_steps))
        baseline_count, baseline_fired, _baseline_latency = collapse_layer1_spikes_spatial(baseline_trace, None, int(primary_steps))
        for q in q_values:
            for threshold in threshold_values:
                _group_rows, group_lookup, threshold_used = _overlap_gated_group_metrics(
                    local_score,
                    overlap_map,
                    valid_mask,
                    dynamic_count,
                    baseline_count,
                    dynamic_fired,
                    baseline_fired,
                    stsp_group_quantile=q,
                    overlap_threshold=threshold,
                )
                interaction = _overlap_gated_interaction_row(ctx, r, primary_window, q, float(threshold_used), group_lookup)
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": int(r.sequence_id),
                        "probe_id": int(r.probe_id),
                        "metric": "interaction_delta",
                        "condition": f"q={q:.2f};overlap={float(threshold_used):.2f}",
                        "stsp_group_quantile": float(q),
                        "overlap_threshold": float(threshold_used),
                        "early_window_ms": int(primary_window),
                        "value": float(interaction.get("interaction_delta", np.nan)),
                        "unit": "probability difference",
                    }
                )
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "supp_s11h_threshold_sensitivity.csv")
    ctx.completed_modules["overlap_threshold_sensitivity"] = True


def _shuffle_score_map(rho: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(rho, dtype=float).copy()
    flat = out.reshape(-1)
    valid = np.isfinite(flat)
    if valid.any():
        flat[valid] = rng.permutation(flat[valid])
    return out


def _q5_minus_q1_from_rows(rows: list[dict[str, Any]], value_col: str) -> float:
    if not rows:
        return np.nan
    df = pd.DataFrame(rows)
    if "score_quantile_bin" not in df.columns or value_col not in df.columns:
        return np.nan
    means = df.groupby("score_quantile_bin")[value_col].mean()
    q1 = float(means.get("Q1", np.nan))
    q5 = float(means.get("Q5", np.nan))
    return float(q5 - q1) if np.isfinite(q1) and np.isfinite(q5) else np.nan


def _extension_row(
    ctx: ExperimentContext,
    endpoint: str,
    panel: str,
    observed: Any,
    null: Any,
    *,
    sequence_id: int,
    probe_id: int | str,
) -> dict[str, Any]:
    obs = float(observed) if np.isfinite(observed) else np.nan
    nul = float(null) if np.isfinite(null) else np.nan
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "sequence_id": sequence_id,
        "probe_id": probe_id,
        "endpoint": endpoint,
        "metric": "observed_minus_null",
        "condition": panel,
        "observed_value": obs,
        "null_value": nul,
        "value": float(obs - nul) if np.isfinite(obs) and np.isfinite(nul) else np.nan,
        "unit": "effect",
    }


def write_legacy_fig6_supplement_aliases(ctx: ExperimentContext) -> None:
    alt_df = _read_csv_if_exists(ctx.metrics_dir / "supp_alternative_peak_definitions.csv")
    if alt_df is not None:
        _save_csv(ctx, _s11_alternative_peak_definitions(ctx, alt_df), ctx.metrics_dir / "supp_s11_alternative_peak_definitions.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S11 alternative peak-definition source missing: supp_alternative_peak_definitions.csv")

    unit_df = _read_csv_if_exists(ctx.metrics_dir / "panel_b_peak_update_history.csv")
    if unit_df is not None:
        _save_csv(ctx, _s11_peak_update_group_enrichment(ctx, unit_df), ctx.metrics_dir / "supp_s11_peak_update_group_enrichment.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S11 update-group enrichment source missing: panel_b_peak_update_history.csv")

    model_df = _read_csv_if_exists(ctx.metrics_dir / "supp_update_recency_support_model_metrics.csv")
    if model_df is not None:
        _save_csv(ctx, _s11_update_recency_model_comparison(ctx, model_df), ctx.metrics_dir / "supp_s11_update_recency_model_comparison.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S11 update-recency model source missing: supp_update_recency_support_model_metrics.csv")

    source_df = _read_csv_if_exists(ctx.metrics_dir / "panel_a_peak_source_attribution.csv")
    if source_df is not None:
        _save_csv(ctx, _s11_leave_one_out_source_details(ctx, source_df), ctx.metrics_dir / "supp_s11_leave_one_out_source_details.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S11 leave-one-out detail source missing: panel_a_peak_source_attribution.csv")

    overlap_df = _read_csv_if_exists(ctx.metrics_dir / "panel_c_peak_input_overlap_similarity.csv")
    if overlap_df is not None:
        _save_csv(ctx, _s11_recent_overlap_window_robustness(ctx, overlap_df), ctx.metrics_dir / "supp_s11_recent_overlap_window_robustness.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S11 overlap-window robustness source missing: panel_c_peak_input_overlap_similarity.csv")

    d_metrics = _read_csv_if_exists(ctx.metrics_dir / "panel_d_peak_weighted_reentry_metrics.csv")
    if d_metrics is not None:
        _save_csv(ctx, _panel_d_matched_contrast(ctx, d_metrics), ctx.metrics_dir / "supp_s12_raw_overlap_matched_peak_overlap_contrast.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S12 matched contrast source missing: panel_d_peak_weighted_reentry_metrics.csv")

    d_reg = _read_csv_if_exists(ctx.metrics_dir / "panel_d_peak_weighted_reentry_regression.csv")
    e_summary = _read_csv_if_exists(ctx.metrics_dir / "panel_e_peak_weighted_downstream_summary.csv")
    controls = _s12_peak_weighted_regression_controls(ctx, d_reg, e_summary)
    if controls is not None:
        _save_csv(ctx, controls, ctx.metrics_dir / "supp_s12_peak_weighted_regression_controls.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S12 regression-control sources missing: panel_d_peak_weighted_reentry_regression.csv and panel_e_peak_weighted_downstream_summary.csv")

    _copy_csv_if_exists(ctx.metrics_dir / "panel_e_downstream_metric_breakdown.csv", ctx.metrics_dir / "supp_s12_downstream_metric_breakdown.csv", ctx)
    e_metrics = _read_csv_if_exists(ctx.metrics_dir / "panel_e_peak_weighted_downstream_metrics.csv")
    if d_metrics is not None:
        _save_csv(ctx, _s11_visual_energy_classpair_controls(ctx, d_metrics), ctx.metrics_dir / "supp_s11_visual_energy_classpair_controls.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S11 visual-energy/class-pair control source missing: panel_d_peak_weighted_reentry_metrics.csv")
    if e_metrics is not None:
        _save_csv(ctx, _s12_global_support_controls(ctx, e_metrics), ctx.metrics_dir / "supp_s12_global_support_spike_count_controls.csv")
    elif ctx.cfg.run_supplement and not (ctx.metrics_dir / "supp_s12_global_support_spike_count_controls.csv").exists():
        ctx.warnings.append("S12 global-support control source missing: panel_e_peak_weighted_downstream_metrics.csv")

    if ctx.cfg.run_peak_perturbation:
        _write_standardized_peak_perturbation_outputs(ctx)

    audit = _real_rollout_scientific_use_audit(ctx)
    _save_csv(ctx, audit, ctx.metrics_dir / "panel_de_real_rollout_scientific_use_audit.csv")
    _save_csv(ctx, audit, ctx.metrics_dir / "supp_s12_real_rollout_scientific_use_audit.csv")

def _write_standardized_panel_d_outputs(ctx: ExperimentContext, df: pd.DataFrame) -> None:
    metrics = _standardize_panel_d_metrics(ctx, df)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_d_peak_weighted_reentry_metrics.csv")
    _save_csv(ctx, _panel_d_summary(ctx, metrics), ctx.metrics_dir / "panel_d_peak_weighted_reentry_summary.csv")
    _save_csv(ctx, _regression_long_table(ctx, metrics, ["reentry_strength", "dynamic_like_reentry", "decision_deflection_score"]), ctx.metrics_dir / "panel_d_peak_weighted_reentry_regression.csv")
    _save_csv(ctx, _panel_d_matched_contrast(ctx, metrics), ctx.metrics_dir / "panel_d_peak_overlap_matched_contrast.csv")

def _write_standardized_panel_e_outputs(ctx: ExperimentContext, df: pd.DataFrame) -> None:
    metrics = _standardize_panel_e_metrics(ctx, df)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_e_peak_weighted_downstream_metrics.csv")
    _save_csv(ctx, _panel_e_summary(ctx, metrics), ctx.metrics_dir / "panel_e_peak_weighted_downstream_summary.csv")
    _save_csv(ctx, _panel_e_breakdown(ctx, metrics), ctx.metrics_dir / "panel_e_downstream_metric_breakdown.csv")

def _standardize_panel_d_metrics(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "network_seed",
        "sequence_id",
        "probe_id",
        "target_label",
        "raw_overlap",
        "peak_weighted_overlap",
        "visual_similarity",
        "input_energy",
        "global_support",
        "nonpeak_support",
        "reentry_strength",
        "dynamic_like_reentry",
        "decision_deflection_score",
        "proxy_mode",
        "real_rollout",
        "final_scientific_use",
        "n_units",
        "n_probe_trials",
        "matched_set_id",
        "peak_overlap_group",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)
    out = df.copy()
    trial = _read_csv_if_exists(ctx.metrics_dir / "panel_d_later_probe_peak_overlap_definitions.csv")
    if trial is not None and not trial.empty:
        join_cols = [
            c
            for c in ("sequence_id", "probe_id", "probe_label", "peak_support_sum", "nonpeak_support_sum", "class_pair")
            if c in trial.columns
        ]
        out = out.merge(trial[join_cols].drop_duplicates(["sequence_id", "probe_id"]), on=["sequence_id", "probe_id"], how="left")
    out["target_label"] = pd.to_numeric(out.get("probe_label", np.nan), errors="coerce")
    out["global_support"] = pd.to_numeric(out.get("peak_support_sum", np.nan), errors="coerce") + pd.to_numeric(out.get("nonpeak_support_sum", np.nan), errors="coerce")
    out["nonpeak_support"] = pd.to_numeric(out.get("nonpeak_support_sum", np.nan), errors="coerce")
    out["reentry_strength"] = pd.to_numeric(out.get("reentry_strength_real", out.get("reentry_strength", np.nan)), errors="coerce")
    out["dynamic_like_reentry"] = pd.to_numeric(out.get("dynamic_like_recovery_real", out.get("dynamic_like_recovery", np.nan)), errors="coerce")
    out["decision_deflection_score"] = pd.to_numeric(out.get("decision_deflection_score_real", out.get("decision_deflection_score", np.nan)), errors="coerce")
    out["proxy_mode"] = _bool_col(out, "proxy_mode", default=_is_proxy_mode(ctx))
    out["real_rollout"] = ~out["proxy_mode"].astype(bool)
    out["final_scientific_use"] = out["real_rollout"].astype(bool)
    out["n_units"] = 28 * 28
    out["n_probe_trials"] = out.groupby("sequence_id")["probe_id"].transform("count") if "sequence_id" in out.columns else len(out)
    out["matched_set_id"] = out.get("matched_group_id", "")
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[cols]

def _standardize_panel_e_metrics(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "network_seed",
        "sequence_id",
        "probe_id",
        "raw_overlap",
        "peak_weighted_overlap",
        "visual_similarity",
        "input_energy",
        "global_support",
        "nonpeak_support",
        "total_spike_count",
        "downstream_metric",
        "metric_value",
        "proxy_mode",
        "real_rollout",
        "final_scientific_use",
        "matched_set_id",
        "peak_overlap_group",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)
    base = df.copy()
    trial = _read_csv_if_exists(ctx.metrics_dir / "panel_d_later_probe_peak_overlap_definitions.csv")
    if trial is not None and not trial.empty:
        join_cols = [c for c in ("sequence_id", "probe_id", "peak_support_sum", "nonpeak_support_sum") if c in trial.columns]
        base = base.merge(trial[join_cols].drop_duplicates(["sequence_id", "probe_id"]), on=["sequence_id", "probe_id"], how="left")
    base["global_support"] = pd.to_numeric(base.get("peak_support_sum", np.nan), errors="coerce") + pd.to_numeric(base.get("nonpeak_support_sum", np.nan), errors="coerce")
    base["nonpeak_support"] = pd.to_numeric(base.get("nonpeak_support_sum", np.nan), errors="coerce")
    base["total_spike_count"] = pd.to_numeric(base.get("total_spike_count", base.get("P_recruit_real", np.nan)), errors="coerce")
    base["proxy_mode"] = _bool_col(base, "proxy_mode", default=_is_proxy_mode(ctx))
    base["real_rollout"] = ~base["proxy_mode"].astype(bool)
    base["final_scientific_use"] = base["real_rollout"].astype(bool)
    base["matched_set_id"] = base.get("matched_group_id", "")
    rows = []
    for metric in DOWNSTREAM_METRICS:
        source_col = f"{metric}_real" if f"{metric}_real" in base.columns else metric
        if source_col not in base.columns:
            continue
        part = base.copy()
        part["downstream_metric"] = metric
        part["metric_value"] = pd.to_numeric(part[source_col], errors="coerce")
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.concat(rows, ignore_index=True)
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[cols]

def _panel_d_summary(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "network_seed",
        "metric",
        "beta_peak_weighted_overlap",
        "beta_raw_overlap",
        "beta_visual_similarity",
        "beta_input_energy",
        "beta_global_support",
        "r2",
        "cv_r2",
        "n_samples",
        "real_rollout",
        "final_scientific_use",
    ]
    return _summary_regression_rows(ctx, df, ["reentry_strength", "dynamic_like_reentry", "decision_deflection_score"], "metric", cols)

def _panel_e_summary(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "network_seed",
        "downstream_metric",
        "beta_peak_weighted_overlap",
        "beta_raw_overlap",
        "beta_visual_similarity",
        "beta_input_energy",
        "beta_global_support",
        "beta_total_spike_count",
        "r2",
        "cv_r2",
        "n_samples",
        "real_rollout",
        "final_scientific_use",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for metric, part in df.groupby("downstream_metric", sort=True):
        rows.extend(_summary_regression_rows(ctx, part, ["metric_value"], "downstream_metric", cols, label=str(metric)).to_dict("records"))
    return pd.DataFrame(rows, columns=cols)

def _summary_regression_rows(ctx: ExperimentContext, df: pd.DataFrame, metrics: Sequence[str], label_col: str, columns: Sequence[str], label: str | None = None) -> pd.DataFrame:
    rows = []
    predictors = ["peak_weighted_overlap", "raw_overlap", "visual_similarity", "input_energy", "global_support", "total_spike_count"]
    for metric in metrics:
        available = [p for p in predictors if p in df.columns]
        cols = available + [metric]
        use = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        fit = _fit_ols(use[available].to_numpy(dtype=float), use[metric].to_numpy(dtype=float)) if len(use) >= 4 and available else None
        beta = dict(zip(["intercept"] + available, np.asarray(fit["beta"], dtype=float))) if fit is not None else {}
        r2 = float(fit["r2"]) if fit is not None else np.nan
        cv = _cv_r2(use[available].to_numpy(dtype=float), use[metric].to_numpy(dtype=float), n_folds=min(5, max(2, len(use) // 2))) if len(use) >= 6 and available else np.nan
        row = {
            "network_seed": int(ctx.cfg.network_seed),
            label_col: label if label is not None else metric,
            "beta_peak_weighted_overlap": float(beta.get("peak_weighted_overlap", np.nan)),
            "beta_raw_overlap": float(beta.get("raw_overlap", np.nan)),
            "beta_visual_similarity": float(beta.get("visual_similarity", np.nan)),
            "beta_input_energy": float(beta.get("input_energy", np.nan)),
            "beta_global_support": float(beta.get("global_support", np.nan)),
            "beta_total_spike_count": float(beta.get("total_spike_count", np.nan)),
            "r2": r2,
            "cv_r2": float(cv),
            "n_samples": int(len(use)),
            "real_rollout": bool(_df_all_true(df, "real_rollout")),
            "final_scientific_use": bool(_df_all_true(df, "final_scientific_use")),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)

def _regression_long_table(ctx: ExperimentContext, df: pd.DataFrame, dependent_metrics: Sequence[str]) -> pd.DataFrame:
    columns = [
        "network_seed",
        "dependent_metric",
        "predictor",
        "beta",
        "std_beta",
        "se",
        "t_value",
        "p_value",
        "r2",
        "n_samples",
        "model_formula",
        "real_rollout",
        "final_scientific_use",
    ]
    predictors = ["peak_weighted_overlap", "raw_overlap", "visual_similarity", "input_energy", "global_support", "total_spike_count"]
    rows = []
    for metric in dependent_metrics:
        available = [p for p in predictors if p in df.columns]
        if metric not in df.columns or not available:
            continue
        use = df[available + [metric]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(use) >= 4:
            x = use[available].to_numpy(dtype=float)
            y = use[metric].to_numpy(dtype=float)
            fit = _fit_ols(x, y)
            beta = np.asarray(fit["beta"], dtype=float)
            se = np.asarray(fit["se"], dtype=float)
            p = np.asarray(fit["p"], dtype=float)
            r2 = float(fit["r2"])
        else:
            x = np.empty((0, len(available)))
            y = np.empty(0)
            beta = np.full(len(available) + 1, np.nan)
            se = np.full(len(available) + 1, np.nan)
            p = np.full(len(available) + 1, np.nan)
            r2 = np.nan
        for idx, pred in enumerate(["intercept"] + available):
            b = float(beta[idx])
            s = float(se[idx])
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "dependent_metric": metric,
                    "predictor": pred,
                    "beta": b,
                    "std_beta": _standardized_coef(b, x, y, pred, available) if len(y) else np.nan,
                    "se": s,
                    "t_value": float(b / s) if np.isfinite(s) and s > 1e-12 else np.nan,
                    "p_value": float(p[idx]),
                    "r2": r2,
                    "n_samples": int(len(use)),
                    "model_formula": f"{metric} ~ " + " + ".join(available),
                    "real_rollout": bool(_df_all_true(df, "real_rollout")),
                    "final_scientific_use": bool(_df_all_true(df, "final_scientific_use")),
                }
            )
    return pd.DataFrame(rows, columns=columns)

def _panel_d_matched_contrast(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "matched_set_id",
        "raw_overlap_low",
        "raw_overlap_high",
        "peak_weighted_overlap_low",
        "peak_weighted_overlap_high",
        "raw_overlap_difference",
        "peak_weighted_overlap_difference",
        "visual_similarity_difference",
        "input_energy_difference",
        "reentry_low",
        "reentry_high",
        "reentry_high_minus_low",
        "decision_deflection_low",
        "decision_deflection_high",
        "decision_deflection_high_minus_low",
        "n_pairs",
    ]
    if df.empty or "matched_set_id" not in df.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    matched = df[df["matched_set_id"].astype(str).str.len() > 0]
    for gid, part in matched.groupby("matched_set_id", sort=True):
        high = part[part.get("peak_overlap_group", "").astype(str).eq("high_peak_overlap")]
        low = part[part.get("peak_overlap_group", "").astype(str).eq("low_peak_overlap")]
        if high.empty or low.empty:
            continue
        h = high.iloc[0]
        l = low.iloc[0]
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "matched_set_id": str(gid),
                "raw_overlap_low": float(l["raw_overlap"]),
                "raw_overlap_high": float(h["raw_overlap"]),
                "peak_weighted_overlap_low": float(l["peak_weighted_overlap"]),
                "peak_weighted_overlap_high": float(h["peak_weighted_overlap"]),
                "raw_overlap_difference": float(h["raw_overlap"] - l["raw_overlap"]),
                "peak_weighted_overlap_difference": float(h["peak_weighted_overlap"] - l["peak_weighted_overlap"]),
                "visual_similarity_difference": float(h["visual_similarity"] - l["visual_similarity"]),
                "input_energy_difference": float(h["input_energy"] - l["input_energy"]),
                "reentry_low": float(l.get("reentry_strength", np.nan)),
                "reentry_high": float(h.get("reentry_strength", np.nan)),
                "reentry_high_minus_low": float(h.get("reentry_strength", np.nan) - l.get("reentry_strength", np.nan)),
                "decision_deflection_low": float(l.get("decision_deflection_score", np.nan)),
                "decision_deflection_high": float(h.get("decision_deflection_score", np.nan)),
                "decision_deflection_high_minus_low": float(h.get("decision_deflection_score", np.nan) - l.get("decision_deflection_score", np.nan)),
                "n_pairs": int(min(len(high), len(low))),
            }
        )
    return pd.DataFrame(rows, columns=columns)

def _panel_e_breakdown(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "downstream_metric",
        "high_peak_overlap_mean",
        "low_peak_overlap_mean",
        "high_minus_low",
        "beta_peak_weighted_overlap",
        "beta_raw_overlap",
        "beta_visual_similarity",
        "beta_input_energy",
        "n_samples",
        "real_rollout",
        "final_scientific_use",
    ]
    rows = []
    if not df.empty:
        summary = _panel_e_summary(ctx, df)
        for metric, part in df.groupby("downstream_metric", sort=True):
            high = pd.to_numeric(part[part["peak_overlap_group"].astype(str).eq("high_peak_overlap")]["metric_value"], errors="coerce")
            low = pd.to_numeric(part[part["peak_overlap_group"].astype(str).eq("low_peak_overlap")]["metric_value"], errors="coerce")
            s = summary[summary["downstream_metric"].eq(metric)]
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "downstream_metric": str(metric),
                    "high_peak_overlap_mean": float(high.mean()) if len(high) else np.nan,
                    "low_peak_overlap_mean": float(low.mean()) if len(low) else np.nan,
                    "high_minus_low": float(high.mean() - low.mean()) if len(high) and len(low) else np.nan,
                    "beta_peak_weighted_overlap": float(s["beta_peak_weighted_overlap"].iloc[0]) if not s.empty else np.nan,
                    "beta_raw_overlap": float(s["beta_raw_overlap"].iloc[0]) if not s.empty else np.nan,
                    "beta_visual_similarity": float(s["beta_visual_similarity"].iloc[0]) if not s.empty else np.nan,
                    "beta_input_energy": float(s["beta_input_energy"].iloc[0]) if not s.empty else np.nan,
                    "n_samples": int(len(part)),
                    "real_rollout": bool(_df_all_true(part, "real_rollout")),
                    "final_scientific_use": bool(_df_all_true(part, "final_scientific_use")),
                }
            )
    return pd.DataFrame(rows, columns=columns)

def _s11_peak_update_group_enrichment(ctx: ExperimentContext, unit_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "update_group", "region_type", "P_peak", "mean_final_support", "mean_delta_support", "mean_update_count", "mean_recent_update_count", "n_units"]
    rows = []
    if unit_df.empty:
        return pd.DataFrame(columns=columns)
    df = unit_df.copy()
    df["update_group"] = "single_old"
    for group in UPDATE_GROUPS:
        df.loc[_group_mask(df, int(ctx.cfg.recent_window), int(ctx.cfg.multi_update_threshold), group), "update_group"] = group
    regions = {
        "peak": df["is_peak"].astype(bool),
        "nonpeak_control": df["is_nonpeak_control"].astype(bool),
        "prior_updated_nonpeak": (~df["is_peak"].astype(bool)) & (pd.to_numeric(df["update_count"], errors="coerce") > 0),
    }
    for group in UPDATE_GROUPS:
        group_df = df[df["update_group"].eq(group)]
        for region, mask in regions.items():
            part = group_df[mask.reindex(group_df.index, fill_value=False)]
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "update_group": group,
                    "region_type": region,
                    "P_peak": _mean_col(part, "is_peak"),
                    "mean_final_support": _mean_col(part, "final_support"),
                    "mean_delta_support": _mean_col(part, "delta_support"),
                    "mean_update_count": _mean_col(part, "update_count"),
                    "mean_recent_update_count": _mean_col(part, f"is_multi_recent_w{ctx.cfg.recent_window}"),
                    "n_units": int(len(part)),
                }
            )
    return pd.DataFrame(rows, columns=columns)

def _s11_update_recency_model_comparison(ctx: ExperimentContext, model_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "dependent_metric", "model_name", "r2", "cv_r2", "delta_r2_vs_baseline", "n_samples", "model_formula"]
    if model_df.empty:
        return pd.DataFrame(columns=columns)
    df = model_df.copy()
    target_col = "target" if "target" in df.columns else "dependent_metric"
    rows = []
    for target, part in df.groupby(target_col, sort=True):
        baseline = part[part["model_name"].eq("baseline_only")]
        baseline_r2 = float(baseline["r2"].iloc[0]) if not baseline.empty and "r2" in baseline else np.nan
        for r in part.itertuples(index=False):
            model_name = str(getattr(r, "model_name"))
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "dependent_metric": str(target),
                    "model_name": model_name,
                    "r2": float(getattr(r, "r2", np.nan)),
                    "cv_r2": float(getattr(r, "cv_r2", np.nan)),
                    "delta_r2_vs_baseline": float(getattr(r, "r2", np.nan) - baseline_r2) if np.isfinite(baseline_r2) else np.nan,
                    "n_samples": int(getattr(r, "n_units", 0)),
                    "model_formula": _model_formula(model_name, str(target)),
                }
            )
    return pd.DataFrame(rows, columns=columns)

def _s11_leave_one_out_source_details(ctx: ExperimentContext, source_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "sequence_id", "removed_position", "relative_position_from_end", "peak_loss", "nonpeak_loss", "prior_updated_loss", "peak_loss_fraction", "nonpeak_loss_fraction", "peak_vs_nonpeak_loss_ratio", "support_loss_total", "n_peak_units", "n_nonpeak_units"]
    if source_df.empty:
        return pd.DataFrame(columns=columns)
    out = source_df.copy()
    out["relative_position_from_end"] = pd.to_numeric(out["seq_len"], errors="coerce") - pd.to_numeric(out["removed_position"], errors="coerce")
    out["n_peak_units"] = np.nan
    out["n_nonpeak_units"] = np.nan
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[columns]

def _s11_recent_overlap_window_robustness(ctx: ExperimentContext, overlap_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "window_type", "recent_k", "dice_peak_overlap", "jaccard_peak_overlap", "peak_coverage", "overlap_precision", "cosine_delta_support_overlap", "spearman_delta_support_overlap", "old_window_control", "n_sequences"]
    rows = []
    if not overlap_df.empty:
        df = overlap_df.copy()
        df["recent_k"] = df["overlap_window"].astype(str).str.extract(r"(\d+)").astype(float)
        for (window_type, recent_k), part in df.groupby(["overlap_type", "recent_k"], dropna=False, sort=True):
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "window_type": str(window_type),
                    "recent_k": int(recent_k) if np.isfinite(recent_k) else 0,
                    "dice_peak_overlap": _mean_col(part, "dice_peak_overlap"),
                    "jaccard_peak_overlap": _mean_col(part, "jaccard_peak_overlap"),
                    "peak_coverage": _mean_col(part, "peak_coverage"),
                    "overlap_precision": _mean_col(part, "overlap_precision"),
                    "cosine_delta_support_overlap": _mean_col(part, "cosine_delta_support_overlap_count"),
                    "spearman_delta_support_overlap": _mean_col(part, "spearman_delta_support_overlap_count"),
                    "old_window_control": bool(str(window_type) == "old"),
                    "n_sequences": int(part["sequence_id"].nunique()) if "sequence_id" in part.columns else int(len(part)),
                }
            )
    return pd.DataFrame(rows, columns=columns)

def _s11_alternative_peak_definitions(ctx: ExperimentContext, alt_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "peak_definition", "peak_quantile", "positive_delta_only", "n_peak_units", "multi_recent_enrichment", "peak_overlap_dice", "peak_overlap_coverage", "peak_weighted_overlap_effect", "n_sequences"]
    rows = []
    if not alt_df.empty:
        for name, part in alt_df.groupby("peak_definition", sort=True):
            metrics = {str(r.metric): float(r.value) for r in part.itertuples(index=False) if hasattr(r, "metric") and hasattr(r, "value")}
            q = 0.10 if "10" in str(name) else 0.30 if "30" in str(name) else 0.20 if "20" in str(name) else np.nan
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "peak_definition": str(name),
                    "peak_quantile": float(q),
                    "positive_delta_only": bool("positive" in str(name) or "top_" in str(name)),
                    "n_peak_units": int(metrics.get("n_peak_units", np.nan)) if np.isfinite(metrics.get("n_peak_units", np.nan)) else 0,
                    "multi_recent_enrichment": float(metrics.get("multi_recent_enrichment", np.nan)),
                    "peak_overlap_dice": float(metrics.get("peak_overlap_dice", np.nan)),
                    "peak_overlap_coverage": float(metrics.get("peak_overlap_coverage", np.nan)),
                    "peak_weighted_overlap_effect": float(metrics.get("peak_weighted_overlap_effect", np.nan)),
                    "n_sequences": int(ctx.n_sequences),
                }
            )
    return pd.DataFrame(rows, columns=columns)

def _s11_visual_energy_classpair_controls(ctx: ExperimentContext, d_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "comparison", "group", "mean_input_energy", "mean_foreground_area", "mean_visual_similarity", "class_pair_entropy", "class_pair_balance_stat", "n_samples", "energy_difference", "foreground_difference", "visual_similarity_difference"]
    rows = []
    if not d_metrics.empty:
        groups = {
            "high_peak_overlap": d_metrics[d_metrics.get("peak_overlap_group", "").astype(str).eq("high_peak_overlap")],
            "low_peak_overlap": d_metrics[d_metrics.get("peak_overlap_group", "").astype(str).eq("low_peak_overlap")],
            "all": d_metrics,
        }
        high_energy = _mean_col(groups["high_peak_overlap"], "input_energy")
        low_energy = _mean_col(groups["low_peak_overlap"], "input_energy")
        high_visual = _mean_col(groups["high_peak_overlap"], "visual_similarity")
        low_visual = _mean_col(groups["low_peak_overlap"], "visual_similarity")
        for group, part in groups.items():
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "comparison": "peak_overlap_group",
                    "group": group,
                    "mean_input_energy": _mean_col(part, "input_energy"),
                    "mean_foreground_area": _mean_col(part, "input_energy"),
                    "mean_visual_similarity": _mean_col(part, "visual_similarity"),
                    "class_pair_entropy": np.nan,
                    "class_pair_balance_stat": np.nan,
                    "n_samples": int(len(part)),
                    "energy_difference": float(high_energy - low_energy) if np.isfinite(high_energy) and np.isfinite(low_energy) else np.nan,
                    "foreground_difference": float(high_energy - low_energy) if np.isfinite(high_energy) and np.isfinite(low_energy) else np.nan,
                    "visual_similarity_difference": float(high_visual - low_visual) if np.isfinite(high_visual) and np.isfinite(low_visual) else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=columns)

def _s12_peak_weighted_regression_controls(ctx: ExperimentContext, d_reg: pd.DataFrame | None, e_summary: pd.DataFrame | None) -> pd.DataFrame | None:
    columns = ["network_seed", "dependent_metric", "predictor", "beta", "std_beta", "se", "t_value", "p_value", "r2", "n_samples", "model_formula", "controls_included", "real_rollout", "final_scientific_use"]
    rows = []
    if d_reg is not None and not d_reg.empty:
        for r in d_reg.itertuples(index=False):
            row = {col: getattr(r, col, np.nan) for col in columns if hasattr(r, col)}
            row["controls_included"] = "raw_overlap,visual_similarity,input_energy,global_support"
            rows.append(row)
    if e_summary is not None and not e_summary.empty:
        predictors = ["peak_weighted_overlap", "raw_overlap", "visual_similarity", "input_energy", "global_support", "total_spike_count"]
        for r in e_summary.itertuples(index=False):
            for pred in predictors:
                attr = f"beta_{pred}"
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "dependent_metric": str(getattr(r, "downstream_metric")),
                        "predictor": pred,
                        "beta": float(getattr(r, attr, np.nan)),
                        "std_beta": np.nan,
                        "se": np.nan,
                        "t_value": np.nan,
                        "p_value": np.nan,
                        "r2": float(getattr(r, "r2", np.nan)),
                        "n_samples": int(getattr(r, "n_samples", 0)),
                        "model_formula": "metric_value ~ " + " + ".join(predictors),
                        "controls_included": "raw_overlap,visual_similarity,input_energy,global_support,total_spike_count",
                        "real_rollout": bool(getattr(r, "real_rollout", False)),
                        "final_scientific_use": bool(getattr(r, "final_scientific_use", False)),
                    }
                )
    if not rows:
        return None
    return pd.DataFrame(rows, columns=columns)

def _s12_global_support_controls(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "dependent_metric", "beta_peak_weighted_overlap", "beta_global_support", "beta_total_spike_count", "beta_nonpeak_support", "r2", "n_samples", "controls_included", "real_rollout", "final_scientific_use"]
    rows = []
    if not df.empty:
        predictors = ["peak_weighted_overlap", "global_support", "total_spike_count", "nonpeak_support"]
        for metric, part in df.groupby("downstream_metric", sort=True):
            available = [p for p in predictors if p in part.columns]
            use = part[available + ["metric_value"]].apply(pd.to_numeric, errors="coerce").dropna()
            fit = _fit_ols(use[available].to_numpy(dtype=float), use["metric_value"].to_numpy(dtype=float)) if len(use) >= 4 and available else None
            beta = dict(zip(["intercept"] + available, np.asarray(fit["beta"], dtype=float))) if fit is not None else {}
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "dependent_metric": str(metric),
                    "beta_peak_weighted_overlap": float(beta.get("peak_weighted_overlap", np.nan)),
                    "beta_global_support": float(beta.get("global_support", np.nan)),
                    "beta_total_spike_count": float(beta.get("total_spike_count", np.nan)),
                    "beta_nonpeak_support": float(beta.get("nonpeak_support", np.nan)),
                    "r2": float(fit["r2"]) if fit is not None else np.nan,
                    "n_samples": int(len(use)),
                    "controls_included": ",".join(available),
                    "real_rollout": bool(_df_all_true(part, "real_rollout")),
                    "final_scientific_use": bool(_df_all_true(part, "final_scientific_use")),
                }
            )
    return pd.DataFrame(rows, columns=columns)

def _real_rollout_scientific_use_audit(ctx: ExperimentContext) -> pd.DataFrame:
    columns = ["network_seed", "module", "output_file", "proxy_mode", "real_rollout", "final_scientific_use", "n_rows", "n_real_rows", "n_proxy_rows", "main_claim_allowed", "claim_strength", "missing_reason"]
    rows = []
    for module, rel in (
        ("Fig6D_peak_weighted_reentry", "data/metrics/panel_d_peak_weighted_reentry_metrics.csv"),
        ("Fig6E_peak_weighted_downstream", "data/metrics/panel_e_peak_weighted_downstream_metrics.csv"),
    ):
        path = ctx.seed_dir / rel
        missing_reason = ""
        if not path.exists():
            df = pd.DataFrame()
            missing_reason = "missing_output"
        else:
            df = pd.read_csv(path)
            if df.empty:
                missing_reason = "empty_output"
        proxy = bool(_df_all_proxy(df)) if not df.empty else True
        real_rows = int(_bool_col(df, "real_rollout").sum()) if not df.empty and "real_rollout" in df.columns else 0
        proxy_rows = int(_bool_col(df, "proxy_mode").sum()) if not df.empty and "proxy_mode" in df.columns else int(len(df))
        final = bool(_df_all_true(df, "final_scientific_use")) if not df.empty else False
        real = bool(len(df) > 0 and real_rows == len(df))
        allowed = bool(real and final and len(df) > 0)
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "module": module,
                "output_file": rel,
                "proxy_mode": proxy,
                "real_rollout": real,
                "final_scientific_use": final,
                "n_rows": int(len(df)),
                "n_real_rows": real_rows,
                "n_proxy_rows": proxy_rows,
                "main_claim_allowed": allowed,
                "claim_strength": _claim_strength(ctx),
                "missing_reason": missing_reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)

def write_global_mechanism_metadata(ctx: ExperimentContext) -> None:
    payload = {
        "figure_chain": [
            "Fig.1 functional STSP substrate",
            "Fig.2 fused two-item state",
            "Fig.3 multi-item STSP landscape",
            "Fig.4 overlap re-entry route",
            "Fig.5 support-to-spike mechanism",
            "Fig.6 overlap-gated STSP recruitment",
        ],
        "mechanism_statement": "Multi-item STSP fields bias Layer 1 recruitment only where later input enters the high-gain field.",
        "key_statements": [
            "STSP gain provides local recruitment bias.",
            "Probe overlap gates whether high-STSP regions are expressed.",
            "High STSP without probe entry is not sufficient for increased firing.",
            "Final label prediction is not the primary endpoint.",
        ],
        "forbidden_claims": [
            "score predicts final label",
            "STSP alone determines firing",
            "high STSP automatically fires without entry",
            "connection weights define the score",
            "inhibition is part of the score",
        ],
        "score_name": "rho_stsp_gain_ratio",
        "score_definition": "rho(q) = G_final(q) / (G_baseline(q) + eps); H_p averages rho over RF(p); S_p(E) averages rho over RF(p) intersect E.",
        "probe_overlap_definition": "O_p = sum E_probe(q) over RF(p) divided by |RF(p)|.",
        "primary_endpoint": "Layer 1 spatial spike recruitment / spike deflection",
        "panel_f_status": "mechanism_metadata_only",
        "high_stsp_overlap_ablation_enabled": bool(ctx.cfg.run_high_stsp_overlap_ablation),
        "high_stsp_overlap_ablation_completed": bool(ctx.completed_modules.get("high_stsp_overlap_ablation")),
        "allowed_claim_strength": "overlap_gated_stsp_recruitment",
    }
    _write_json(payload, ctx.raw_dir / "panel_f_global_mechanism_metadata.json")
    ctx.output_files["panel_f_global_mechanism_metadata"] = "data/raw/panel_f_global_mechanism_metadata.json"
