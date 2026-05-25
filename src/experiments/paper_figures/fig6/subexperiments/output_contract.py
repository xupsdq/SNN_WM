from __future__ import annotations

from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as _legacy

# Keep module-level names identical while Fig.6 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def _write_config_files(ctx: ExperimentContext) -> None:
    cfg = ctx.cfg
    _write_json(asdict(cfg), ctx.config_dir / "run_config.json")
    _write_json(asdict(cfg), ctx.seed_dir / "run_config.json")
    _write_json(
        {
            "figure": FIGURE_ID,
            "fig6_design_version": FIG6_DESIGN_VERSION,
            "main_panels": MAIN_PANELS,
            "main_claim": MAIN_CLAIM,
            "mechanism_boundary": MECHANISM_BOUNDARY,
            "supplement_plan": SUPPLEMENT_PLAN,
            "main_required_outputs": MAIN_REQUIRED_OUTPUTS,
            "optional_main_outputs": OPTIONAL_MAIN_OUTPUTS,
            "supplementary_outputs": SUPPLEMENTARY_OUTPUTS,
            "optional_supplementary_outputs": OPTIONAL_SUPPLEMENTARY_OUTPUTS,
            "claim_boundary": "Main panels use rho-based STSP gain scores to predict Layer 1 spatial spike recruitment and real-probe deflection; final labels, route-peak perturbation, connection weights, inhibition, voltage, and WTA are not score endpoints.",
        },
        ctx.config_dir / "figure_requirements.json",
    )
    _write_json(
        {
            "conditions": list(UPDATE_GROUPS),
            "recent_window": cfg.recent_window,
            "multi_update_threshold": cfg.multi_update_threshold,
            "mechanism_summary": {
                "score": "rho_stsp_gain_ratio",
                "entry_gate": "global ping, region-gated ping mask, or real-probe active foreground",
                "primary_claim": MAIN_CLAIM,
            },
            "entry_gated_stsp_score_definition": {
                "formula": "mean(G_final / (G_baseline + eps)) over entry-active presynaptic sites in each Layer 1 receptive field",
                "excludes": MECHANISM_BOUNDARY["forbidden_claims"],
                "channel_policy": "STSP support is spatial; Layer 1 spikes are collapsed across output channels.",
            },
            "local_stsp_score_definition": {
                "formula": "H_p = mean rho(q) over q in RF(p)",
                "entry_independent": True,
            },
            "probe_overlap_score_definition": {
                "formula": "O_p = sum E_probe(q) over q in RF(p) divided by |RF(p)|",
            },
            "main_panels": MAIN_PANELS,
            "claim_strength_rules": {
                "main": "overlap_gated_stsp_recruitment",
                "route_peak_perturbation": "legacy_supplement_only",
                "forbidden": MECHANISM_BOUNDARY["forbidden_claims"],
            },
        },
        ctx.config_dir / "condition_spec.json",
    )
    _write_json(
        {
            "score_name": "rho_stsp_gain_ratio",
            "score_definition": "rho(q) = G_final(q) / (G_baseline(q) + eps); scores average rho across Layer 1 receptive fields",
            "score_eps": float(cfg.score_eps),
            "gain_ratio_clip_quantiles": list(cfg.gain_ratio_clip_quantiles),
            "score_use_log_gain": bool(cfg.score_use_log_gain),
            "layer1_receptive_field": {"kernel_size": 5, "stride": 1, "padding": 2},
            "score_excludes": ["connection_weights", "inhibition", "voltage", "threshold", "WTA", "final_label"],
            "primary_endpoint": "Layer 1 spatial spike recruitment / spike deflection",
            "global_ping": {"amp": float(cfg.global_ping_amp), "ms": int(cfg.global_ping_ms)},
            "overlap_gated_stsp": {"stsp_group_quantile": float(cfg.stsp_group_quantile), "overlap_threshold": float(cfg.overlap_threshold)},
        },
        ctx.config_dir / "entry_gated_stsp_score_spec.json",
    )
    _write_json(
        {
            "panels": list(SUPPLEMENT_PLAN.keys()),
            "active_outputs": SUPPLEMENTARY_OUTPUTS,
            "optional_outputs": OPTIONAL_SUPPLEMENTARY_OUTPUTS,
            "extensions": {
                "score_shuffle_null": "run by default with --run-supplement; explicit flag remains supported",
                "overlap_threshold_sensitivity": "run by default with --run-supplement; explicit flag remains supported",
            },
        },
        ctx.config_dir / "active_supplement_spec.json",
    )
    _write_json({"state_variable": "g = u * x", "peak_q": cfg.peak_q, "definition": "legacy top fraction of positive delta_support units", "main_status": "legacy_fig6"}, ctx.config_dir / "legacy_peak_definition_spec.json")
    _write_json({"models": list(MODEL_NAMES), "target": ["delta_support", "final_support"], "cv": "deterministic K-fold over units", "main_status": "legacy_fig6"}, ctx.config_dir / "legacy_update_recency_model_spec.json")
    _write_json(
        {
            "leave_one_out_mode": cfg.leave_one_out_mode,
            "blank_same_timing": cfg.leave_one_out_mode == "blank_same_timing",
            "support_loss_definition": "max(G_full - G_minus_i, 0)",
            "peak_loss_fraction_definition": "peak_loss_i / sum_j peak_loss_j",
            "real_network_required": True,
        },
        ctx.config_dir / "peak_source_attribution_spec.json",
    )
    _write_json(
        {
            "peak_definition": "top positive delta_support units by peak_q",
            "recent_windows": list(cfg.recent_overlap_windows),
            "multi_update_threshold": int(cfg.multi_update_threshold),
            "groups_summarized": ["peak", "nonpeak_control", "prior_updated_nonpeak"],
        },
        ctx.config_dir / "peak_update_history_spec.json",
    )
    _write_json(
        {
            "recent_overlap_windows": list(cfg.recent_overlap_windows),
            "high_overlap_mask_definition": "top n_peak positive-overlap units by overlap count",
            "similarity_metrics": ["dice", "jaccard", "peak_coverage", "overlap_precision", "cosine_delta_support_overlap_count"],
        },
        ctx.config_dir / "peak_input_overlap_origin_spec.json",
    )
    _write_json({"raw_overlap": "legacy later-probe overlap covariate", "peak_weighted_overlap": "legacy supplement-only covariate", "main_status": "legacy_fig6"}, ctx.config_dir / "legacy_peak_weighted_overlap_spec.json")
    _write_json(
        {
            "state_conditions": list(cfg.real_reentry_reference_conditions),
            "reference_state_definition": "S_final restores sequence boundary; S0 resets network to baseline if no explicit S0 boundary is available",
            "metrics": ["normalized_reentry_loss", "P_output_switch", "response_displacement_loss", "decision_deflection_loss"],
            "proxy_mode_not_final": True,
        },
        ctx.config_dir / "real_reentry_rollout_spec.json",
    )
    _write_json(
        {
            "implemented_by_default": False,
            "enabled_flag": "--run-peak-perturbation",
            "main_status": "legacy_supplement_only",
            "unit_sets": list(PERTURBATION_UNIT_SET_ORDER),
            "reset_variables": "u/x reset to S0 values; g_e zeroed when spatially compatible",
            "probe_input_modified": False,
            "failure_policy": "does not affect main Fig.6 score/spike outputs",
        },
        ctx.config_dir / "legacy_peak_perturbation_spec.json",
    )

def _write_summary(ctx: ExperimentContext) -> dict[str, Any]:
    required_main = [ctx.seed_dir / rel for rel in MAIN_REQUIRED_OUTPUTS]
    required_supp = [ctx.seed_dir / rel for rel in SUPPLEMENTARY_OUTPUTS] if ctx.cfg.run_supplement else []
    required_optional = []
    if ctx.cfg.run_score_shuffle_null:
        required_optional.append(ctx.seed_dir / OPTIONAL_SUPPLEMENTARY_OUTPUTS[0])
    if ctx.cfg.run_overlap_threshold_sensitivity:
        required_optional.append(ctx.seed_dir / OPTIONAL_SUPPLEMENTARY_OUTPUTS[1])
    required_optional_main = [ctx.seed_dir / rel for rel in OPTIONAL_MAIN_OUTPUTS] if ctx.cfg.run_high_stsp_overlap_ablation else []
    main_outputs_available = bool(all(path.exists() for path in required_main))
    summary = {
        "figure": FIGURE_ID,
        "fig6_design_version": FIG6_DESIGN_VERSION,
        "main_panels": MAIN_PANELS,
        "main_claim": MAIN_CLAIM,
        "supplement_plan": SUPPLEMENT_PLAN,
        "mechanism_boundary": MECHANISM_BOUNDARY,
        "score_name": "rho_stsp_gain_ratio",
        "score_definition": "rho(q) = G_final(q) / (G_baseline(q) + eps); H_p averages rho over RF(p); S_p(E) averages rho over RF(p) intersect E",
        "score_excludes": ["connection_weights", "inhibition", "voltage", "threshold", "WTA", "final_label"],
        "primary_endpoint": "Layer 1 spatial spike recruitment / spike deflection",
        "interpretation_boundary": "The score predicts spike recruitment and dynamic-baseline deflection in overlap-gated high-STSP regions, not deterministic one-to-one firing or final-label prediction.",
        "channel_policy": "Layer 1 STSP gain maps are spatial support maps averaged across input channels; Layer 1 spikes are collapsed across output channels.",
        "old_peak_origin_panels_demoted_from_main": True,
        "route_peak_perturbation_demoted_from_main": True,
        "panel_a_removed_from_main": False,
        "legacy_fig6": {
            "peak_origin": "available only through explicit legacy flags",
            "peak_weighted_overlap": "available only through explicit legacy flags",
            "route_peak_perturbation": "available only through explicit legacy flags",
            "downstream_final_label": "not part of active Fig.6 claim",
        },
        "active_supplement_plan": {
            "S7A": "score/input/ping audit",
            "S7B": "global-ping spike-count endpoint",
            "S7C": "real-probe window robustness",
            "S7D": "overlap interaction window robustness",
            "S7E": "site availability audit",
            "S7F": "paired ablation difference",
            "S7G": "score-shuffle null, run by default with --run-supplement",
            "S7H": "threshold sensitivity, run by default with --run-supplement",
        },
        "supplement_extensions_run_by_default": True,
        "optional_extensions_required_for_smoke": bool(ctx.cfg.run_supplement),
        "main_b_method": "rho_region_gated_ping_readout_bias",
        "main_c_method": "global_unbiased_ping_local_stsp_score_vs_layer1_spike_recruitment",
        "main_d_method": "entry_gated_score_vs_real_probe_layer1_spike_deflection",
        "main_e_method": "overlap_gated_local_stsp_recruitment_2x2",
        "main_f_method": "mechanism_metadata_only",
        "proxy_mode": False,
        "allow_proxy": False,
        "fig6_main_outputs_forced": bool(ctx.cfg.force_main_outputs),
        "main_a_method": "high_stsp_overlap_ablation_vs_matched_removal",
        "real_rollout_available": bool((ctx.metrics_dir / "panel_d_real_probe_score_spike_deflection.csv").exists()),
        "final_scientific_use": bool(main_outputs_available),
        "main_claim_allowed": bool(main_outputs_available),
        "claim_strength": "overlap_gated_stsp_recruitment",
        "peak_perturbation_status": _peak_perturbation_status(ctx),
        "peak_perturbation_claim_upgrade_allowed": False,
        "forbidden_claims": MECHANISM_BOUNDARY["forbidden_claims"],
        "network_seed": int(ctx.cfg.network_seed),
        "run_mode": "single_network",
        "smoke": bool(ctx.cfg.smoke),
        "completed_modules": ctx.completed_modules,
        "output_files": ctx.output_files,
        "n_sequences": int(ctx.n_sequences),
        "n_probe_candidates": int(ctx.n_probe_candidates),
        "n_matched_groups": int(ctx.n_matched_groups),
        "peak_definition": {"state_variable": "g", "peak_q": float(ctx.cfg.peak_q), "positive_delta_only": True},
        "entry_score_config": {
            "score_eps": float(ctx.cfg.score_eps),
            "score_early_windows_ms": list(ctx.cfg.score_early_windows_ms),
            "primary_score_early_window_ms": int(ctx.cfg.primary_score_early_window_ms),
            "score_n_bins": int(ctx.cfg.score_n_bins),
            "basin_radius": int(ctx.cfg.basin_radius),
            "basin_top_q": float(ctx.cfg.basin_top_q),
            "gain_ratio_clip_quantiles": list(ctx.cfg.gain_ratio_clip_quantiles),
            "real_probe_entry_mode": str(ctx.cfg.real_probe_entry_mode),
            "score_use_log_gain": bool(ctx.cfg.score_use_log_gain),
            "stsp_group_quantile": float(ctx.cfg.stsp_group_quantile),
            "overlap_threshold": float(ctx.cfg.overlap_threshold),
            "global_ping_amp": float(ctx.cfg.global_ping_amp),
            "global_ping_ms": int(ctx.cfg.global_ping_ms),
        },
        "update_recency_model": {"models": list(MODEL_NAMES), "cv": "K-fold over units", "main_status": "legacy_supplement_only"},
        "peak_weighted_overlap_definition": {"main_status": "legacy_supplement_only"},
        "peak_perturbation_implemented": bool(ctx.completed_modules.get("peak_perturbation")),
        "peak_perturbation_successful": bool(_peak_perturbation_claim_upgrade_allowed(ctx)),
        "fig6_route_peak_perturbation": _summary_route_peak_perturbation(ctx),
        "fig6d_route_peak_perturbation": _summary_route_peak_panel(ctx, panel="D"),
        "fig6e_route_peak_downstream": _summary_route_peak_panel(ctx, panel="E"),
        "allowed_claim_strength": "overlap_gated_stsp_recruitment",
        "conditions": list(UPDATE_GROUPS),
        "warnings": ctx.warnings,
        "main_claim_supported_fields_available": bool(main_outputs_available),
        "missing_for_main_figure": [_rel(path, ctx.seed_dir) for path in required_main if not path.exists()],
        "missing_for_optional_main": [_rel(path, ctx.seed_dir) for path in required_optional_main if not path.exists()],
        "missing_for_supplementary": [_rel(path, ctx.seed_dir) for path in required_supp if not path.exists()],
        "missing_for_optional_supplementary": [_rel(path, ctx.seed_dir) for path in required_optional if not path.exists()],
    }
    _write_json(summary, ctx.seed_dir / "summary.json")
    _write_json(_artifact_manifest(ctx), ctx.seed_dir / "artifact_manifest.json")
    ctx.output_files["summary"] = "summary.json"
    return summary

def _artifact_manifest(ctx: ExperimentContext) -> dict[str, Any]:
    files = []
    for path in ctx.seed_dir.rglob("*"):
        if path.is_file():
            files.append({"path": _rel(path, ctx.seed_dir), "size_bytes": int(path.stat().st_size)})
    return {"figure": FIGURE_ID, "files": sorted(files, key=lambda x: x["path"])}

def _summary_route_peak_perturbation(ctx: ExperimentContext) -> dict[str, Any]:
    audit_path = ctx.metrics_dir / "panel_de_route_peak_perturbation_scientific_use_audit.csv"
    contrast_path = ctx.metrics_dir / "panel_d_route_peak_reentry_loss_contrast.csv"
    summary_path = ctx.metrics_dir / "panel_d_route_peak_reentry_loss_summary.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    contrast = pd.read_csv(contrast_path) if contrast_path.exists() else pd.DataFrame()
    d_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    success = bool(not audit.empty and _bool_col(audit, "route_peak_perturbation_success").any())
    valid_trials = int(pd.to_numeric(audit.get("n_valid_trials", pd.Series([0])), errors="coerce").fillna(0).max()) if not audit.empty else 0
    contrasts: dict[str, float | None] = {}
    if not contrast.empty:
        for row in contrast.itertuples(index=False):
            contrasts[str(getattr(row, "contrast", ""))] = float(getattr(row, "route_peak_minus_control", np.nan)) if np.isfinite(float(getattr(row, "route_peak_minus_control", np.nan))) else None
    means: dict[str, float | None] = {}
    if not d_summary.empty:
        for row in d_summary.itertuples(index=False):
            value = float(getattr(row, "mean_normalized_reentry_loss", np.nan))
            means[str(getattr(row, "perturbation_unit_set", ""))] = value if np.isfinite(value) else None
    return {
        "enabled": bool(ctx.cfg.run_peak_perturbation),
        "success": success,
        "allowed_claim_strength": "causal_route_peak_gain" if success else "predictive_peak_amplified_only",
        "n_valid_trials": valid_trials,
        "all_unit_sets_valid": bool(success),
        "route_peak_minus_controls": contrasts,
        "mean_normalized_reentry_loss": means,
    }

def _summary_route_peak_panel(ctx: ExperimentContext, *, panel: str) -> dict[str, Any]:
    if panel == "D":
        path = ctx.metrics_dir / "panel_d_route_peak_reentry_loss_summary.csv"
        value_col = "mean_normalized_reentry_loss"
    elif panel == "E":
        path = ctx.metrics_dir / "panel_e_route_peak_downstream_summary.csv"
        value_col = "P_output_switch"
    else:
        raise ValueError(f"Unsupported Fig.6 route-peak panel: {panel}")
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    n_valid = int(pd.to_numeric(df.get("n_valid_trials", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not df.empty else 0
    present = set(df.get("perturbation_unit_set", pd.Series(dtype=str)).astype(str)) if not df.empty else set()
    values = pd.to_numeric(df.get(value_col, pd.Series(dtype=float)), errors="coerce") if not df.empty else pd.Series(dtype=float)
    missing_sets = [unit_set for unit_set in PERTURBATION_UNIT_SET_ORDER if unit_set not in present]
    invalid_sets = [
        unit_set
        for unit_set in PERTURBATION_UNIT_SET_ORDER
        if unit_set in present
        and int(pd.to_numeric(df.loc[df["perturbation_unit_set"].astype(str).eq(unit_set), "n_valid_trials"], errors="coerce").fillna(0).sum()) <= 0
    ] if not df.empty and "perturbation_unit_set" in df.columns and "n_valid_trials" in df.columns else list(PERTURBATION_UNIT_SET_ORDER)
    failure = None
    if not path.exists():
        failure = f"missing_source:{_rel(path, ctx.seed_dir)}"
    elif df.empty:
        failure = "empty_summary"
    elif not values.notna().any():
        failure = f"all_nan:{value_col}"
    elif missing_sets or invalid_sets:
        failure = f"invalid_unit_sets:missing={','.join(missing_sets) or 'none'}; invalid={','.join(invalid_sets) or 'none'}"
    return {
        "required_for_main": False,
        "output_exists": bool(path.exists()),
        "n_valid_trials": n_valid,
        "all_unit_sets_valid": bool(failure is None),
        "failure_reason": failure,
    }
