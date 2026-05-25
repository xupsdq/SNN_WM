from __future__ import annotations

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as _legacy

# Keep module-level names identical while Fig.3 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def _write_config_files(ctx: ExperimentContext) -> None:
    cfg = ctx.cfg
    _write_json(_json_safe(asdict(cfg)), ctx.config_dir / "run_config.json")
    _write_json(_json_safe(asdict(cfg)), ctx.seed_dir / "run_config.json")
    _write_json(
        {
            "main_panels": {
                "A": "multi-item sequence protocol",
                "B": "progressive state update / update weakening",
                "C": "peak-structured final STSP landscape",
                "D": "neutral ping serial-position readout",
                "E": "singleton-matched sequence-state weak-probe completion",
            },
            "primary_layer": PRIMARY_LAYER,
            "primary_state_variable": PRIMARY_STATE_VARIABLE,
            "state_bank_required": True,
            "structural_weak_cue_in_main": False,
            "main_required_outputs": [
                "data/metrics/panel_b_progressive_update_metrics.csv",
                "data/metrics/panel_c_example_landscape_summary.csv",
                "data/metrics/panel_d_ping_position_distribution.csv",
                "data/metrics/panel_d_ping_summary.csv",
                "data/raw/panel_e_weak_probe_trial_readout.csv",
                "data/metrics/panel_e_weak_probe_metrics.csv",
                "data/metrics/panel_e_weak_probe_memory_gain.csv",
                "data/metrics/panel_e_weak_probe_position_stratified_metrics.csv",
            ],
            "supplementary_outputs": {
                "S5": [
                    "data/metrics/supp_peak_valley_contrast.csv",
                    "data/metrics/supp_landscape_nonflatness.csv",
                    "data/metrics/supp_peak_valley_prevalence.csv",
                    "data/metrics/supp_network_peak_valley_summary.csv",
                    "data/metrics/supp_anchor_dynamics_metrics.csv",
                    "data/metrics/supp_recency_only_controls.csv",
                ],
                "S6": [
                    "data/metrics/supp_ping_recency_diagnostics.csv",
                    "data/metrics/supp_weak_probe_target_source_control.csv",
                    "data/metrics/supp_weak_probe_target_source_gain.csv",
                    "data/metrics/supp_structural_weak_cue_matching_diagnostics.csv",
                    "data/metrics/supp_structural_weak_cue_accuracy.csv",
                    "data/metrics/supp_structural_weak_cue_memory_gain.csv",
                    "data/metrics/supp_peak_cue_serial_position_metrics.csv",
                    "data/metrics/supp_peak_cue_serial_position_gain.csv",
                    "data/raw/supp_region_ping_amp_sweep_trial_readout.csv",
                    "data/metrics/supp_region_ping_amp_sweep_summary.csv",
                    "data/metrics/supp_region_ping_amp_sweep_latency.csv",
                ],
            },
        },
        ctx.config_dir / "figure_requirements.json",
    )
    _write_json(
        {
            "state_conditions": ["S0", "S_1..S_K", "S_final", "singleton_reference", "singleton_boundary", "decay_counterfactual"],
            "primary_representation": {"layer": PRIMARY_LAYER, "state_variable": PRIMARY_STATE_VARIABLE},
            "distance": "centered_cosine_distance",
            "fig3_boundary": "Exploratory multi-item STSP morphology and functional readout; mechanism proof is reserved for Fig.6.",
            "population_morphology_diagnostics": "supplementary",
            "panel_d_neutral_ping": "Configured functional restore of S0/S_final followed by neutral constant-drive ping; readout is serial-position distribution.",
            "panel_e_weak_probe": "Fig.2F-compatible encoded-spike dropout weak probe; same degraded spike probe across cue_only, slot-matched singleton, and final sequence STSP states.",
            "region_ping_scope": "Supplementary/legacy only in Fig.3; region-gated ping evidence is reserved for Fig.6 main-line support.",
            "panel_f_structural_weak_cue": "Legacy/supplement only; not a main Fig.3 source for this design version.",
            "region_ping": {
                "enabled": bool(cfg.run_region_ping),
                "support_metric": str(cfg.region_ping_support_metric),
                "region_q": float(cfg.region_ping_q),
                "conditions": list(cfg.region_ping_conditions),
                "repeats": int(cfg.region_ping_repeats),
                "s0_control": bool(cfg.run_region_ping_s0_control),
                "amp_sweep": bool(cfg.run_region_ping_amp_sweep),
                "amp_values": list(cfg.region_ping_amp_sweep),
            },
        },
        ctx.config_dir / "condition_spec.json",
    )
    _write_json(
        {
            "restore_mode": str(cfg.functional_restore_mode),
            "weak_probe_mask_space": str(cfg.weak_probe_mask_space),
            "weak_probe_use_same_mask_across_states": bool(cfg.weak_probe_use_same_mask_across_states),
            "weak_probe_scale": float(cfg.weak_probe_scale),
            "weak_probe_noise": float(cfg.weak_probe_noise),
            "weak_probe_metric_mode": str(cfg.weak_probe_metric_mode),
            "weak_probe_memory_scope": str(cfg.weak_probe_memory_scope),
            "weak_probe_target_source": str(cfg.weak_probe_target_source),
            "weak_probe_include_singleton": bool(cfg.weak_probe_include_singleton),
            "weak_probe_memory_conditions": list(MEMORY_CONDITIONS),
            "fig2F_compat_enabled": bool(cfg.weak_probe_mask_space == "encoded_spikes" and cfg.weak_probe_metric_mode == "fig2_compat"),
            "fig4_weak_probe_method_compat_enabled": bool(cfg.weak_probe_mask_space == "encoded_spikes"),
            "structural_weak_cue_in_main": False,
            "panel_e": "encoded-spike dropout weak-probe completion",
            "region_ping_main_status": "removed_from_fig3_main; reserved for Fig.6 or supplementary audit only",
            "region_ping": {
                "support_metric": str(cfg.region_ping_support_metric),
                "region_q": float(cfg.region_ping_q),
                "conditions": list(cfg.region_ping_conditions),
                "ping_amp": float(cfg.ping_amp),
                "ping_ms": int(cfg.ping_ms),
                "s0_control": bool(cfg.run_region_ping_s0_control),
                "amp_sweep_enabled": bool(cfg.run_region_ping_amp_sweep),
                "amp_sweep": list(cfg.region_ping_amp_sweep),
            },
        },
        ctx.config_dir / "functional_readout_spec.json",
    )
    _write_json(
        {
            "definition": "Masks are defined from final pre-cue STSP support landscape before weak-cue presentation.",
            "support": "gain_ratio_map = G_final / (G_baseline + eps) by default; delta_G remains available for compatibility",
            "peak_q": float(cfg.peak_q),
            "valley_q": float(cfg.valley_q),
            "cue_conditions": list(CUE_CONDITIONS),
            "mask_mode": str(cfg.weak_cue_mask_mode),
            "foreground_threshold": float(cfg.foreground_threshold),
            "pre_cue": True,
        },
        ctx.config_dir / "mask_definition_spec.json",
    )
    _write_json(
        {
            "purpose": "Exploratory unbiased neutral-ping readout distribution from final multi-item STSP state.",
            "main_state_condition": "S_final",
            "baseline_state_condition": "S0",
            "decoder": "decode_prediction_and_fire_time_from_layer3",
            "main_metric": "position_distribution",
            "not_main_claim": ["recent_bias", "latest_item_mass", "ping_COM"],
            "ping_ms": int(cfg.ping_ms),
            "ping_amp": float(cfg.ping_amp),
            "state_conditions": list(cfg.ping_main_state_conditions),
        },
        ctx.config_dir / "ping_readout_spec.json",
    )
    _write_json(
        {
            "purpose": "Test whether target-foreground locations with high, low, or random final STSP support differ in weak-cue classification efficacy.",
            "main_or_supplement": "supplementary_or_legacy_only",
            "target_source_main": "sequence_member_random",
            "mask_mode": str(cfg.weak_cue_mask_mode),
            "support_map_for_ranking": "structural weak cue remains delta_G-ranked; region ping is not part of Fig.3 main.",
            "cue_conditions": list(CUE_CONDITIONS),
            "memory_conditions": list(MEMORY_CONDITIONS),
            "primary_metric": "accuracy and memory_gain",
            "keep_fractions": list(cfg.weak_cue_keep_fractions),
            "main_keep_fraction": float(cfg.peak_cue_main_keep_fraction),
            "same_mask_used_across_memory_conditions": True,
            "main_panel_claim": "No Fig.3 main-panel claim; retained only for supplementary or legacy audit paths.",
        },
        ctx.config_dir / "structural_weak_cue_spec.json",
    )

def _write_summary(ctx: ExperimentContext) -> dict[str, Any]:
    required_main: list[Path] = []
    if ctx.cfg.run_progressive_update:
        required_main.append(ctx.metrics_dir / "panel_b_progressive_update_metrics.csv")
    if ctx.cfg.run_peak_valley_landscape:
        required_main.append(ctx.metrics_dir / "panel_c_example_landscape_summary.csv")
    if ctx.cfg.run_neutral_ping:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_d_ping_position_distribution.csv",
                ctx.metrics_dir / "panel_d_ping_summary.csv",
            ]
        )
    if ctx.cfg.run_weak_probe:
        required_main.extend(
            [
                ctx.raw_dir / "panel_e_weak_probe_trial_readout.csv",
                ctx.metrics_dir / "panel_e_weak_probe_metrics.csv",
                ctx.metrics_dir / "panel_e_weak_probe_auc_metrics.csv",
                ctx.metrics_dir / "panel_e_weak_probe_memory_gain.csv",
                ctx.metrics_dir / "panel_e_weak_probe_position_stratified_metrics.csv",
            ]
        )
    if ctx.cfg.run_peak_cue_main:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_f_peak_cue_accuracy.csv",
                ctx.metrics_dir / "panel_f_peak_cue_memory_gain.csv",
                ctx.metrics_dir / "panel_f_peak_cue_matching_diagnostics.csv",
            ]
        )
    required_supp: list[Path] = []
    if ctx.cfg.run_supplement or ctx.cfg.run_population_morphology_supplement:
        required_supp.extend(
            [
                ctx.metrics_dir / "supp_peak_valley_contrast.csv",
                ctx.metrics_dir / "supp_landscape_nonflatness.csv",
                ctx.metrics_dir / "supp_peak_valley_prevalence.csv",
                ctx.metrics_dir / "supp_network_peak_valley_summary.csv",
                ctx.metrics_dir / "supp_anchor_dynamics_metrics.csv",
                ctx.metrics_dir / "supp_recency_only_controls.csv",
            ]
        )
    if ctx.cfg.run_supplement or ctx.cfg.run_neutral_ping:
        required_supp.append(ctx.metrics_dir / "supp_ping_recency_diagnostics.csv")
    if ctx.cfg.run_supplement or ctx.cfg.run_weak_probe:
        required_supp.extend(
            [
                ctx.metrics_dir / "supp_weak_probe_target_source_control.csv",
                ctx.metrics_dir / "supp_weak_probe_target_source_gain.csv",
            ]
        )
    if ctx.cfg.run_supplement or ctx.cfg.run_peak_cue_main or ctx.cfg.run_structural_weak_cue_supplement:
        required_supp.extend(
            [
                ctx.metrics_dir / "supp_structural_weak_cue_accuracy.csv",
                ctx.metrics_dir / "supp_structural_weak_cue_memory_gain.csv",
                ctx.metrics_dir / "supp_structural_weak_cue_matching_diagnostics.csv",
                ctx.metrics_dir / "supp_peak_cue_serial_position_metrics.csv",
                ctx.metrics_dir / "supp_peak_cue_serial_position_gain.csv",
            ]
        )
    if ctx.cfg.run_region_ping_amp_sweep:
        required_supp.extend(
            [
                ctx.raw_dir / "supp_region_ping_amp_sweep_trial_readout.csv",
                ctx.metrics_dir / "supp_region_ping_amp_sweep_summary.csv",
                ctx.metrics_dir / "supp_region_ping_amp_sweep_latency.csv",
            ]
        )
    if ctx.cfg.run_region_ping:
        required_supp.extend(
            [
                ctx.raw_dir / "panel_f_region_ping_trial_readout.csv",
                ctx.metrics_dir / "panel_f_region_ping_position_distribution.csv",
                ctx.metrics_dir / "panel_f_region_ping_summary.csv",
                ctx.metrics_dir / "panel_f_region_ping_contrast.csv",
                ctx.metrics_dir / "panel_f_region_ping_current_matching.csv",
            ]
        )
    panel_f_optional = [
        "cue_fraction_actual",
        "cue_energy",
        "encoded_spike_count",
        "support_mean_selected",
        "support_mean_foreground",
        "support_quantile_mean",
    ]
    panel_f_raw_path = ctx.raw_dir / "panel_f_peak_cue_trial_readout.csv"
    missing_panel_f_optional = _missing_csv_columns(panel_f_raw_path, panel_f_optional)
    target_control_path = ctx.metrics_dir / "supp_weak_probe_target_source_control.csv"
    unseen_target_control_available = False
    if target_control_path.exists():
        try:
            target_control = pd.read_csv(target_control_path)
            unseen_target_control_available = bool("target_source" in target_control.columns and "unseen_random" in set(target_control["target_source"].astype(str)))
        except Exception:
            unseen_target_control_available = False
    region_matching = _read_csv_if_exists(ctx.metrics_dir / "panel_f_region_ping_current_matching.csv")
    region_contrast = _read_csv_if_exists(ctx.metrics_dir / "panel_f_region_ping_contrast.csv")
    region_raw = _read_csv_if_exists(ctx.raw_dir / "panel_f_region_ping_trial_readout.csv")
    weak_metrics = _read_csv_if_exists(ctx.metrics_dir / "panel_e_weak_probe_metrics.csv")
    region_current_status = _region_ping_current_matching_status(region_matching)
    legacy_peak_available = bool((ctx.metrics_dir / "panel_f_peak_cue_memory_gain.csv").exists())
    region_available = bool((ctx.metrics_dir / "panel_f_region_ping_summary.csv").exists())
    summary = {
        "figure": FIGURE_ID,
        "network_seed": int(ctx.cfg.network_seed),
        "run_mode": SINGLE_NETWORK_MODE,
        "fig3_design_version": FIG3_DESIGN_VERSION,
        "main_panels": {
            "A": "sequence protocol",
            "B": "progressive update / update weakening",
            "C": "peak-structured final STSP landscape",
            "D": "neutral ping serial-position readout",
            "E": "singleton-matched sequence-state weak-probe completion",
        },
        "removed_from_main": ["region-gated peak/valley/random ping"],
        "demoted_to_supplement": ["population peak-valley prevalence", "ping recency diagnostics", "target-source and serial-position controls"],
        "supplement_plan": {
            "S5": "morphology and anchor controls",
            "S6": "functional controls for multi-item and peak-guided access",
        },
        "weak_cue_mask_mode": str(ctx.cfg.weak_cue_mask_mode),
        "weak_cue_target_source_main": "sequence_member_random",
        "panel_d_restore_mode": str(ctx.cfg.functional_restore_mode),
        "panel_d_stsp_only_restore": str(ctx.cfg.functional_restore_mode) == "stsp_only",
        "panel_e_weak_probe_mask_space": str(ctx.cfg.weak_probe_mask_space),
        "panel_e_weak_probe_scale": float(ctx.cfg.weak_probe_scale),
        "panel_e_weak_probe_metric_mode": str(ctx.cfg.weak_probe_metric_mode),
        "panel_e_fig2F_compatible": bool(ctx.cfg.weak_probe_mask_space == "encoded_spikes" and ctx.cfg.weak_probe_metric_mode == "fig2_compat"),
        "fig2F_weak_probe_compatible": bool(ctx.cfg.weak_probe_mask_space == "encoded_spikes" and ctx.cfg.weak_probe_metric_mode == "fig2_compat"),
        "structural_weak_cue_in_main": False,
        "peak_cue_main_keep_fraction": float(ctx.cfg.peak_cue_main_keep_fraction),
        "panel_e_general_weak_probe_available": bool(ctx.completed_modules.get("weak_probe", False)),
        "fig3e_singleton_weak_probe": {
            "enabled": bool(ctx.cfg.run_weak_probe),
            "has_single_item_memory": bool("memory_condition" in weak_metrics.columns and "single_item_memory" in set(weak_metrics.get("memory_condition", pd.Series(dtype=str)).astype(str))),
            "n_memory_conditions": int(weak_metrics["memory_condition"].nunique()) if "memory_condition" in weak_metrics.columns else 0,
            "memory_conditions": sorted(weak_metrics["memory_condition"].dropna().astype(str).unique().tolist()) if "memory_condition" in weak_metrics.columns else [],
            "position_stratified_available": bool((ctx.metrics_dir / "panel_e_weak_probe_position_stratified_metrics.csv").exists()),
        },
        "legacy_region_ping": {
            "enabled": bool(ctx.cfg.run_region_ping),
            "main_status": "removed_from_fig3_main_reserved_for_fig6_or_supplement",
            "support_metric": str(ctx.cfg.region_ping_support_metric),
            "region_q": float(ctx.cfg.region_ping_q),
            "region_conditions": list(ctx.cfg.region_ping_conditions),
            "n_region_conditions": len(tuple(ctx.cfg.region_ping_conditions)),
            "s0_control_available": bool("state_condition" in region_raw.columns and "S0" in set(region_raw.get("state_condition", pd.Series(dtype=str)).astype(str))),
            "amp_sweep_available": bool((ctx.metrics_dir / "supp_region_ping_amp_sweep_summary.csv").exists()),
            "current_matching_status": region_current_status,
            "JS_peak_valley": _first_float(region_contrast, "JS_peak_valley"),
            "TV_peak_valley": _first_float(region_contrast, "TV_peak_valley"),
            "P_peak_label_differs_from_valley": _first_float(region_contrast, "P_peak_label_differs_from_valley"),
            "ambiguous_label_count": int(pd.to_numeric(region_raw.get("label_is_ambiguous", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not region_raw.empty else 0,
        },
        "legacy_peak_cue_outputs_available": legacy_peak_available,
        "region_ping_outputs_available": region_available,
        "main_fig3_region_ping_status": "removed_from_main_supplementary_legacy_available" if bool(ctx.cfg.run_region_ping) else "removed_from_main_not_required",
        "panel_f_peak_cue_available": bool(ctx.completed_modules.get("peak_cue_main", False)),
        "panel_f_peak_cue_missing_optional_fields": missing_panel_f_optional,
        "unseen_target_control_available": unseen_target_control_available,
        "structural_weak_cue_supplement_available": bool((ctx.metrics_dir / "supp_structural_weak_cue_accuracy.csv").exists()),
        "keep_fraction_sweep_is_real_rollout": True,
        "fixed_K_minus_1_target_used_for_main": False,
        "smoke": bool(ctx.cfg.smoke),
        "completed_modules": ctx.completed_modules,
        "output_files": ctx.output_files,
        "n_sequences": int(ctx.n_sequences),
        "sequence_lengths": list(ctx.cfg.sequence_lengths),
        "state_conditions": ["S0", "S_1..S_K", "S_final"],
        "cue_conditions": list(CUE_CONDITIONS),
        "memory_conditions": list(MEMORY_CONDITIONS),
        "mask_definition": {"peak_q": float(ctx.cfg.peak_q), "valley_q": float(ctx.cfg.valley_q), "pre_cue": True, "mode": str(ctx.cfg.weak_cue_mask_mode)},
        "warnings": ctx.warnings,
        "main_claim_supported_fields_available": all(path.exists() for path in required_main),
        "missing_for_main_figure": [_rel(path, ctx.seed_dir) for path in required_main if not path.exists()],
        "missing_for_supplementary": [_rel(path, ctx.seed_dir) for path in required_supp if not path.exists()],
    }
    _write_json(summary, ctx.seed_dir / "summary.json")
    ctx.output_files["summary"] = "summary.json"
    write_artifact_manifest(ctx, experiment_id=FIGURE_ID, title="Fig.3 multi-item peak landscape")
    return summary
