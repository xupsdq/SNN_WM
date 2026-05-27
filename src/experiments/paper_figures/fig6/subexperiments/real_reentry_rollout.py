from __future__ import annotations

from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as _legacy
from src.experiments.paper_figures.fig6.subexperiments.helpers_1 import _probe_entry_mask

# Keep module-level names identical while Fig.6 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def build_later_probe_peak_overlap_trials(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    probe_trials = build_probe_candidate_trials(ctx, bank)
    rows: list[dict[str, Any]] = []
    for r in _progress(probe_trials.itertuples(index=False), total=len(probe_trials), desc="fig6 probe definitions", enabled=ctx.cfg.show_progress):
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": int(r.sequence_id),
                "probe_id": int(r.probe_id),
                "probe_image_id": int(r.probe_image_id),
                "probe_label": int(r.probe_label),
                "probe_source": str(r.probe_source),
                "entry_mask_mode": str(getattr(r, "entry_mask_mode", ctx.cfg.real_probe_entry_mode)),
                "raw_overlap": float(r.raw_overlap),
                "peak_weighted_overlap": float(r.peak_weighted_overlap),
                "peak_overlap_fraction": float(r.peak_overlap_fraction),
                "nonpeak_overlap_fraction": float(r.nonpeak_overlap_fraction),
                "visual_similarity": float(r.visual_similarity),
                "input_energy": float(r.input_energy),
                "peak_support_sum": float(r.peak_support_sum),
                "nonpeak_support_sum": float(r.nonpeak_support_sum),
                "class_pair": str(r.class_pair),
                "candidate_seed": int(r.candidate_seed),
            }
        )
    _save_csv(ctx, pd.DataFrame(rows, columns=PANEL_D_TRIAL_DEFINITION_COLUMNS), ctx.metrics_dir / "panel_d_later_probe_peak_overlap_definitions.csv")
    _save_panel_d_example(ctx, bank, probe_trials)
    bank.probe_trials = probe_trials
    ctx.n_probe_candidates = int(len(probe_trials))
    ctx.completed_modules["later_probe_peak_overlap_trials"] = True

def build_probe_candidate_trials(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> pd.DataFrame:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 6006)
    image_ids_by_label = {label: np.asarray(ids, dtype=np.int64) for label, ids in ctx.class_index.items()}
    rows: list[dict[str, Any]] = []
    entry_cache: dict[tuple[Any, ...], Any] = {}
    for seq_index, meta in _progress(enumerate(bank.sequence_meta.itertuples(index=False)), total=len(bank.sequence_meta), desc="fig6 probe candidates", enabled=ctx.cfg.show_progress):
        seq_id = int(meta.sequence_id)
        sequence_labels = [int(v) for v in str(meta.ordered_item_labels).split(";") if str(v) != ""]
        sequence_ids = [int(v) for v in str(meta.ordered_item_ids).split(";") if str(v) != ""]
        prior_updated = bank.prior_updated_mask[seq_index].reshape(28, 28)
        peak = bank.peak_mask[seq_index].reshape(28, 28)
        nonpeak = bank.nonpeak_mask[seq_index].reshape(28, 28)
        support = bank.g_final[seq_index].reshape(28, 28)
        for local_probe in _progress(range(int(ctx.cfg.num_probe_candidates_per_sequence)), total=int(ctx.cfg.num_probe_candidates_per_sequence), desc="fig6 probe per sequence", enabled=ctx.cfg.show_progress):
            if local_probe % 3 == 0:
                label = int(rng.choice(sequence_labels))
                source = "sequence_label"
            else:
                label = int(rng.integers(0, 10))
                source = "candidate_pool"
            probe_image_id = int(rng.choice(image_ids_by_label[label]))
            probe = _image_array(ctx.dataset, probe_image_id)
            probe_mask = _probe_entry_mask(ctx, probe_image_id, mode=str(ctx.cfg.real_probe_entry_mode), cache=entry_cache)
            route_mask = probe_mask & prior_updated
            peak_overlap_mask = route_mask & peak
            nonpeak_overlap_mask = route_mask & nonpeak
            raw_overlap = _safe_div(float(route_mask.sum()), float(max(1, probe_mask.sum())))
            peak_support_sum = float((support * peak_overlap_mask).sum())
            nonpeak_support_sum = float((support * nonpeak_overlap_mask).sum())
            peak_weighted_overlap = _safe_div(peak_support_sum, float(max(1, probe_mask.sum())))
            sim = float(np.mean([_centered_cosine(probe.reshape(-1), _image_array(ctx.dataset, sid).reshape(-1)) for sid in sequence_ids]))
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": seq_id,
                    "probe_id": int(seq_id * 1000 + local_probe),
                    "probe_image_id": int(probe_image_id),
                    "probe_label": int(label),
                    "probe_source": source,
                    "raw_overlap": float(raw_overlap),
                    "peak_weighted_overlap": float(peak_weighted_overlap),
                    "peak_overlap_fraction": _safe_div(float(peak_overlap_mask.sum()), float(max(1, route_mask.sum()))),
                    "nonpeak_overlap_fraction": _safe_div(float(nonpeak_overlap_mask.sum()), float(max(1, route_mask.sum()))),
                    "visual_similarity": float(sim),
                    "input_energy": float(probe.sum()),
                    "entry_mask_mode": str(ctx.cfg.real_probe_entry_mode),
                    "class_pair": f"{sequence_labels[-1] if sequence_labels else -1}->{label}",
                    "candidate_seed": int(rng.integers(0, 2**31 - 1)),
                    "peak_support_sum": peak_support_sum,
                    "nonpeak_support_sum": nonpeak_support_sum,
                }
            )
    df = pd.DataFrame(rows, columns=PROBE_TRIAL_COLUMNS)
    groups = _matched_raw_overlap_groups(ctx, df)
    _save_csv(ctx, df, ctx.trial_specs_dir / "probe_candidate_trials.csv")
    _save_csv(ctx, groups, ctx.trial_specs_dir / "matched_raw_overlap_groups.csv")
    bank.matched_groups = groups
    ctx.n_matched_groups = int(len(groups))
    return df

def run_probe_candidate_reentry_rollouts(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    ctx.warnings.append("Legacy run_probe_candidate_reentry_rollouts redirected to real restored-state rollout; formula proxy is not a main Fig.6 result.")
    run_real_probe_reentry_rollouts(ctx, bank)

def run_real_probe_reentry_rollouts(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    if ctx.net is None or ctx.encoder is None or torch is None:
        raise RuntimeError("Fig.6 real probe rollout requires a loaded real network and encoder.")
    if bank.probe_trials.empty:
        raise RuntimeError("Fig.6 real probe rollout requires non-empty probe trials.")
    rows: list[dict[str, Any]] = []
    downstream_rows: list[dict[str, Any]] = []
    trace_payload: dict[str, np.ndarray] = {}
    vector_payload: dict[str, np.ndarray] = {}
    matched_lookup = _matched_lookup(bank.matched_groups)
    encode_cache: dict[tuple[Any, ...], Any] = {}
    for r in _progress(bank.probe_trials.itertuples(index=False), total=len(bank.probe_trials), desc="fig6 real probes", enabled=ctx.cfg.show_progress):
        seq_idx = _sequence_index(bank, int(r.sequence_id))
        matched_group_id, peak_group = matched_lookup.get(int(r.probe_id), ("", "unmatched"))
        boundary = bank.boundaries.get(int(r.sequence_id))
        if not boundary:
            raise RuntimeError(f"Fig.6 real probe rollout missing S_final boundary for sequence={int(r.sequence_id)}.")
        try:
            probe_spikes = _encode_sequence_cached(ctx, [int(r.probe_image_id)], ctx.cfg.probe_steps, encode_cache)
            condition_results = _run_real_probe_conditions_batch(
                ctx,
                int(r.probe_image_id),
                (boundary, None),
                ("S_final", "S0"),
                probe_spikes=probe_spikes,
            )
            final_trace, final_pred, final_fire, final_vector = condition_results["S_final"]
            s0_trace, s0_pred, s0_fire, s0_vector = condition_results["S0"]
        except Exception as exc:
            raise RuntimeError(
                f"Real Fig.6 probe rollout failed for sequence={int(r.sequence_id)} probe={int(r.probe_id)}: {exc}"
            ) from exc
        l3_delta = float(np.linalg.norm(final_trace.reshape(-1) - s0_trace.reshape(-1)))
        evidence_final = _label_evidence(final_vector, int(r.probe_label))
        evidence_s0 = _label_evidence(s0_vector, int(r.probe_label))
        decision_deflection = float(evidence_final - evidence_s0)
        dynamic_recovery = _plain_cosine(final_trace.reshape(-1), s0_trace.reshape(-1))
        first_delta = _fire_delta(final_fire, s0_fire)
        early_gain = _early_spike_count(final_trace) - _early_spike_count(s0_trace)
        p_advance, p_recruit, spike_advance = _spike_timing_metrics(final_trace, s0_trace)
        displacement = float(np.linalg.norm(final_vector.reshape(-1) - s0_vector.reshape(-1)))
        key = f"sequence_{int(r.sequence_id)}_probe_{int(r.probe_id)}"
        if ctx.cfg.save_l3_trace:
            trace_payload[f"{key}_Sfinal_l3_trace"] = final_trace.astype(np.float32)
            trace_payload[f"{key}_S0_l3_trace"] = s0_trace.astype(np.float32)
        vector_payload[f"{key}_Sfinal_readout_vector"] = final_vector.astype(np.float32)
        vector_payload[f"{key}_S0_readout_vector"] = s0_vector.astype(np.float32)
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": int(r.sequence_id),
                "probe_id": int(r.probe_id),
                "matched_group_id": matched_group_id,
                "raw_overlap": float(r.raw_overlap),
                "peak_weighted_overlap": float(r.peak_weighted_overlap),
                "peak_overlap_group": peak_group,
                "visual_similarity": float(r.visual_similarity),
                "input_energy": float(r.input_energy),
                "prediction_Sfinal": int(final_pred),
                "prediction_S0": int(s0_pred),
                "correct_Sfinal": bool(int(final_pred) == int(r.probe_label)),
                "correct_S0": bool(int(s0_pred) == int(r.probe_label)),
                "first_fire_time_Sfinal": int(final_fire),
                "first_fire_time_S0": int(s0_fire),
                "first_fire_time_delta": first_delta,
                "l3_trace_delta_norm": l3_delta,
                "reentry_strength_real": l3_delta,
                "dynamic_like_recovery_real": dynamic_recovery,
                "decision_deflection_score_real": decision_deflection,
                "proxy_mode": False,
            }
        )
        downstream_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": int(r.sequence_id),
                "probe_id": int(r.probe_id),
                "matched_group_id": matched_group_id,
                "raw_overlap": float(r.raw_overlap),
                "peak_weighted_overlap": float(r.peak_weighted_overlap),
                "peak_overlap_group": peak_group,
                "visual_similarity": float(r.visual_similarity),
                "input_energy": float(r.input_energy),
                "early_recruitment_gain_real": float(early_gain),
                "P_advance_real": p_advance,
                "P_recruit_real": p_recruit,
                "spike_advance_real": spike_advance,
                "response_pattern_displacement_real": displacement,
                "decision_deflection_score_real": decision_deflection,
                "partial_cue_completion_gain_real": float("nan"),
                "proxy_mode": False,
            }
        )
    if not rows or not downstream_rows:
        raise RuntimeError("Fig.6 real probe rollout produced no rows.")
    bank.reentry_metrics = pd.DataFrame(rows, columns=PANEL_D_REAL_METRIC_COLUMNS)
    bank.downstream_metrics = pd.DataFrame(downstream_rows, columns=PANEL_E_REAL_METRIC_COLUMNS)
    np.savez_compressed(ctx.raw_dir / "reentry_trace_arrays_l3.npz", **trace_payload)
    np.savez_compressed(ctx.raw_dir / "downstream_dynamics_vectors.npz", **vector_payload)
    ctx.output_files["reentry_trace_arrays_l3"] = "data/raw/reentry_trace_arrays_l3.npz"
    ctx.output_files["downstream_dynamics_vectors"] = "data/raw/downstream_dynamics_vectors.npz"
    ctx.completed_modules["real_reentry_rollouts"] = True

def compute_real_peak_weighted_reentry_metrics(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    df = bank.reentry_metrics.copy()
    _save_csv(ctx, df, ctx.metrics_dir / "panel_d_real_reentry_metrics.csv")
    matched = df[df["matched_group_id"].astype(str).str.len() > 0].copy()
    _save_csv(ctx, matched, ctx.metrics_dir / "panel_d_raw_overlap_matched_peak_reentry.csv")
    reg = _regression_rows(ctx, df, metrics=("reentry_strength_real", "l3_trace_delta_norm", "dynamic_like_recovery_real", "decision_deflection_score_real"), n_name="n_trials")
    reg["proxy_mode"] = bool(_df_all_proxy(df))
    _save_csv(ctx, reg, ctx.metrics_dir / "panel_d_peak_overlap_reentry_regression.csv")
    _save_csv(ctx, reg, ctx.metrics_dir / "supp_raw_vs_peak_weighted_overlap_regression.csv")
    _write_standardized_panel_d_outputs(ctx, df)
    ctx.completed_modules["real_reentry_metrics"] = True

def compute_peak_weighted_reentry_metrics(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    df = bank.reentry_metrics.copy()
    _save_csv(ctx, df, ctx.metrics_dir / "supp_legacy_panel_d_peak_weighted_reentry_metrics.csv")
    matched = df[df["matched_group_id"].astype(str).str.len() > 0].copy()
    _save_csv(ctx, matched, ctx.metrics_dir / "supp_legacy_panel_d_matched_raw_overlap_comparison.csv")
    metrics = tuple(
        metric
        for metric in ("reentry_strength", "DPI_L3", "dynamic_like_recovery", "decision_deflection_score")
        if metric in df.columns
    )
    if not metrics:
        metrics = ("reentry_strength_real", "l3_trace_delta_norm", "dynamic_like_recovery_real", "decision_deflection_score_real")
    reg = _regression_rows(ctx, df, metrics=metrics, n_name="n_trials")
    _save_csv(ctx, reg, ctx.metrics_dir / "supp_legacy_panel_d_peak_weighted_overlap_regression.csv")
    _save_csv(ctx, reg, ctx.metrics_dir / "supp_raw_vs_peak_weighted_overlap_regression.csv")
    ctx.completed_modules["reentry_prediction"] = True
