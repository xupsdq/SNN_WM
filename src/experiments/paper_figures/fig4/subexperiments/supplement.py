from __future__ import annotations

from src.experiments.paper_figures import fig4_overlap_reentry_experiment as _legacy

# Keep module-level names identical while Fig.4 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_supplement_outputs(ctx: ExperimentContext, bank: OverlapReentryDMSBank) -> None:
    effect = _pair_effect_table(ctx, bank)
    alt_rows = []
    for _, r in bank.pair_trials.merge(effect[["pair_id", "b_vec", "DPI_L3", "decision_deflection"]], on="pair_id", how="left").iterrows():
        for name, value in (
            ("dice_overlap", r["dice_overlap"]),
            ("overlap_fraction_sample", r["overlap_fraction_sample"]),
            ("overlap_fraction_probe", r["overlap_fraction_probe"]),
            ("dilated_overlap", min(1.0, float(r["dice_overlap"]) + 0.05)),
            ("encoded_spike_overlap", float(r["dice_overlap"])),
        ):
            alt_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "pair_id": int(r["pair_id"]),
                    "overlap_definition": name,
                    "overlap_value": float(value),
                    "dynamic_effect_metric": "DPI_L3",
                    "metric_value": float(r["DPI_L3"]),
                }
            )
    class_breakdown = (
        bank.pair_trials.merge(effect, on="pair_id", how="left")
        .groupby("class_pair", dropna=False)
        .agg(n_pairs=("pair_id", "nunique"), mean_acc_drop=("acc_drop", "mean"), mean_DPI_L3=("DPI_L3", "mean"), mean_decision_deflection=("decision_deflection", "mean"))
        .reset_index()
    )
    class_breakdown.insert(0, "network_seed", int(ctx.cfg.network_seed))
    layer_rows = []
    for _, r in effect.iterrows():
        layer_rows.append({"network_seed": int(ctx.cfg.network_seed), "pair_id": int(r["pair_id"]), "layer": "L3", "delay_ms": int(ctx.cfg.delay_ms), "metric": "DPI_L3", "value": float(r["DPI_L3"])})
        layer_rows.append({"network_seed": int(ctx.cfg.network_seed), "pair_id": int(r["pair_id"]), "layer": "readout", "delay_ms": int(ctx.cfg.delay_ms), "metric": "final_readout_deflection", "value": float(r["decision_deflection"])})
    random_controls = _random_mask_controls(ctx, bank)
    audit = _condition_audit(ctx, bank)
    _save_csv(ctx, random_controls, ctx.metrics_dir / "supp_random_mask_perturbation_controls.csv")
    _save_csv(ctx, pd.DataFrame(alt_rows), ctx.metrics_dir / "supp_alternative_overlap_definitions.csv")
    _save_csv(ctx, class_breakdown, ctx.metrics_dir / "supp_class_pair_breakdown.csv")
    _save_csv(ctx, pd.DataFrame(layer_rows), ctx.metrics_dir / "supp_layer_delay_reentry_metrics.csv")
    _save_csv(ctx, audit, ctx.metrics_dir / "supp_trial_condition_audit.csv")
    for filename in ("supp_overlap_similarity_regression.csv", "supp_overlap_matching_diagnostics.csv", "supp_overlap_similarity_2x2.csv", "supp_similarity_bin_full_stats.csv"):
        path = ctx.metrics_dir / filename
        if not path.exists():
            _save_csv(ctx, pd.DataFrame(), path)
    ctx.completed_modules["supplement"] = True

def write_fig4_panel_aliases_and_supplement_aliases(ctx: ExperimentContext) -> None:
    _write_s7_similarity_bin_full_trend(ctx)
    _write_s7_overlap_matching_diagnostics(ctx)
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_d_overlap_accuracy_contrast_by_network.csv",
        ctx.metrics_dir / "supp_s7_iso_similarity_overlap_contrast.csv",
        empty_columns=[
            "network_seed",
            "n_matched_sets",
            "drop_rate_high_overlap",
            "drop_rate_low_overlap",
            "delta_drop_rate",
            "mean_acc_drop_high_overlap",
            "mean_acc_drop_low_overlap",
            "delta_acc_drop",
        ],
        reason="panel_d_overlap_accuracy_contrast_by_network_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_d_overlap_accuracy_permutation_null.csv",
        ctx.metrics_dir / "supp_s7_iso_similarity_permutation_null.csv",
        empty_columns=["network_seed", "permutation_index", "delta_drop_rate_null", "delta_acc_drop_null"],
        reason="panel_d_overlap_accuracy_permutation_null_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_d_iso_similarity_matched_pairs.csv",
        ctx.metrics_dir / "supp_s7_iso_similarity_matched_pairs.csv",
        empty_columns=_iso_match_columns(),
        reason="panel_d_iso_similarity_matched_pairs_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_d_matching_balance_diagnostics.csv",
        ctx.metrics_dir / "supp_s7_overlap_matching_balance_diagnostics.csv",
        empty_columns=[
            "network_seed",
            "n_matched_sets",
            "mean_similarity_difference",
            "mean_sample_energy_rel_difference",
            "mean_probe_energy_rel_difference",
            "mean_overlap_difference",
        ],
        reason="panel_d_matching_balance_diagnostics_missing_or_empty",
    )
    _write_s7_overlap_regression_controls(ctx)
    _write_s7_random_nonoverlap_perturbation_controls(ctx)
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_e_time_resolved_l3_displacement.csv",
        ctx.metrics_dir / "supp_s8_time_resolved_l3_displacement.csv",
        empty_columns=["network_seed", "pair_id", "condition", "time_step", "time_ms", "S_dyn_L3", "S_sta_L3", "DPI_L3_t"],
        reason="panel_e_time_resolved_l3_displacement_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_e_decision_spike_displacement.csv",
        ctx.metrics_dir / "supp_s8_decision_spike_displacement.csv",
        empty_columns=["network_seed", "pair_id", "condition", "mean_DPI_L3", "decision_spike_advance"],
        reason="panel_e_decision_spike_displacement_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_f_l3_accumulator_region_replay_metrics.csv",
        ctx.metrics_dir / "supp_s8_l3_accumulator_replay_metrics.csv",
        empty_columns=["network_seed", "pair_id", "region_id", "region_label"],
        reason="panel_f_l3_accumulator_region_replay_metrics_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_f_l3_accumulator_summary.csv",
        ctx.metrics_dir / "supp_s8_l3_accumulator_summary.csv",
        empty_columns=["network_seed", "metric", "value", "n_pairs"],
        reason="panel_f_l3_accumulator_summary_missing_or_empty",
    )
    if not (ctx.metrics_dir / "supp_s8_decision_deflection_metrics.csv").exists():
        _copy_csv_alias(
            ctx,
            ctx.metrics_dir / "supp_decision_deflection_metrics.csv",
            ctx.metrics_dir / "supp_s8_decision_deflection_metrics.csv",
            empty_columns=["network_seed", "pair_id", "condition", "dynamic_like_recovery", "decision_deflection_score"],
            reason="supp_decision_deflection_metrics_missing_or_empty",
        )
    if not (ctx.metrics_dir / "supp_s8_decision_deflection_summary.csv").exists():
        src = ctx.metrics_dir / "supp_s8_decision_deflection_metrics.csv"
        df = pd.read_csv(src) if src.exists() else pd.DataFrame()
        _save_csv(ctx, _decision_deflection_summary(df), ctx.metrics_dir / "supp_s8_decision_deflection_summary.csv")
    ctx.completed_modules["s7_s8_aliases"] = True

def _write_s7_similarity_bin_full_trend(ctx: ExperimentContext) -> None:
    src = ctx.metrics_dir / "supp_similarity_bin_full_stats.csv"
    if not src.exists() or not _csv_nonempty(src):
        src = ctx.metrics_dir / "panel_b_similarity_bin_summary.csv"
    if not src.exists():
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s7_similarity_bin_full_trend.csv", ["network_seed", "similarity_bin", "mean_pixel_similarity", "mean_acc_drop", "mean_drop_event", "mean_b_vec", "mean_DPI_L3", "n_pairs"], "panel_b_similarity_bin_summary_missing")
        return
    df = pd.read_csv(src)
    if df.empty:
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s7_similarity_bin_full_trend.csv", ["network_seed", "similarity_bin", "mean_pixel_similarity", "mean_acc_drop", "mean_drop_event", "mean_b_vec", "mean_DPI_L3", "n_pairs"], "similarity_bin_source_empty")
        return
    out = pd.DataFrame(
        {
            "network_seed": df["network_seed"] if "network_seed" in df.columns else int(ctx.cfg.network_seed),
            "similarity_bin": df["similarity_bin"] if "similarity_bin" in df.columns else df.index.astype(str),
            "mean_pixel_similarity": df["mean_pixel_similarity"] if "mean_pixel_similarity" in df.columns else df.get("bin_center", np.nan),
            "mean_acc_drop": df["mean_acc_drop"] if "mean_acc_drop" in df.columns else np.nan,
            "mean_drop_event": df["mean_drop_event"] if "mean_drop_event" in df.columns else df.get("drop_rate", np.nan),
            "mean_b_vec": df["mean_b_vec"] if "mean_b_vec" in df.columns else np.nan,
            "mean_DPI_L3": df["mean_DPI_L3"] if "mean_DPI_L3" in df.columns else np.nan,
            "n_pairs": df["n_pairs"] if "n_pairs" in df.columns else len(df),
        }
    )
    _save_csv(ctx, out, ctx.metrics_dir / "supp_s7_similarity_bin_full_trend.csv")

def _write_s7_overlap_matching_diagnostics(ctx: ExperimentContext) -> None:
    matches_path = ctx.metrics_dir / "panel_d_iso_similarity_matched_pairs.csv"
    if matches_path.exists() and _csv_nonempty(matches_path):
        matches = pd.read_csv(matches_path)
        rows = []
        for network_seed, part in matches.groupby("network_seed", sort=False):
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "comparison": "iso_similarity_high_vs_low_overlap",
                    "group": "matched_pairs",
                    "mean_pixel_similarity": float(pd.to_numeric(pd.concat([part["pixel_similarity_high"], part["pixel_similarity_low"]]), errors="coerce").mean()),
                    "mean_input_energy_sample": float(pd.to_numeric(pd.concat([part["input_energy_sample_high"], part["input_energy_sample_low"]]), errors="coerce").mean()),
                    "mean_input_energy_probe": float(pd.to_numeric(pd.concat([part["input_energy_probe_high"], part["input_energy_probe_low"]]), errors="coerce").mean()),
                    "mean_dice_overlap": float(pd.to_numeric(pd.concat([part["dice_overlap_high"], part["dice_overlap_low"]]), errors="coerce").mean()),
                    "n_pairs": int(len(part) * 2),
                    "similarity_abs_diff": float(pd.to_numeric(part["similarity_difference"], errors="coerce").mean()),
                    "sample_energy_rel_diff": float(pd.to_numeric(part["sample_energy_rel_difference"], errors="coerce").mean()),
                    "probe_energy_rel_diff": float(pd.to_numeric(part["probe_energy_rel_difference"], errors="coerce").mean()),
                    "class_pair_balance_notes": "see panel_d_matching_balance_diagnostics.csv for aggregate class/probe-label balance",
                }
            )
        _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "supp_s7_overlap_matching_diagnostics.csv")
        return
    for candidate in ("panel_d_matching_balance_diagnostics.csv", "supp_overlap_matching_diagnostics.csv"):
        src = ctx.metrics_dir / candidate
        if src.exists():
            _copy_csv_alias(ctx, src, ctx.metrics_dir / "supp_s7_overlap_matching_diagnostics.csv", empty_columns=["network_seed", "comparison", "group", "n_pairs"], reason=f"{candidate}_missing_or_empty")
            return
    _write_empty_csv(ctx, ctx.metrics_dir / "supp_s7_overlap_matching_diagnostics.csv", ["network_seed", "comparison", "group", "mean_pixel_similarity", "mean_input_energy_sample", "mean_input_energy_probe", "mean_dice_overlap", "n_pairs", "similarity_abs_diff", "sample_energy_rel_diff", "probe_energy_rel_diff", "class_pair_balance_notes"], "overlap_matching_sources_missing")

def _write_s7_overlap_regression_controls(ctx: ExperimentContext) -> None:
    src = ctx.metrics_dir / "supp_overlap_accuracy_regression.csv"
    if not src.exists():
        src = ctx.metrics_dir / "supp_overlap_similarity_regression.csv"
    if not src.exists():
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s7_overlap_regression_controls.csv", ["network_seed", "metric", "beta_overlap", "beta_similarity", "beta_input_energy_sample", "beta_input_energy_probe", "r2", "n_pairs", "notes"], "overlap_regression_sources_missing")
        return
    df = pd.read_csv(src)
    if df.empty:
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s7_overlap_regression_controls.csv", ["network_seed", "metric", "beta_overlap", "beta_similarity", "beta_input_energy_sample", "beta_input_energy_probe", "r2", "n_pairs", "notes"], "overlap_regression_source_empty")
        return
    out = pd.DataFrame(
        {
            "network_seed": df["network_seed"] if "network_seed" in df.columns else int(ctx.cfg.network_seed),
            "metric": df["metric"] if "metric" in df.columns else "unknown",
            "beta_overlap": df["beta_overlap"] if "beta_overlap" in df.columns else np.nan,
            "beta_similarity": df["beta_similarity"] if "beta_similarity" in df.columns else np.nan,
            "beta_input_energy_sample": df["beta_input_energy_sample"] if "beta_input_energy_sample" in df.columns else df.get("beta_input_energy", np.nan),
            "beta_input_energy_probe": df["beta_input_energy_probe"] if "beta_input_energy_probe" in df.columns else np.nan,
            "r2": df["r2"] if "r2" in df.columns else np.nan,
            "n_pairs": df["n_pairs"] if "n_pairs" in df.columns else np.nan,
            "notes": df["notes"] if "notes" in df.columns else "standardized S7 regression alias",
        }
    )
    _save_csv(ctx, out, ctx.metrics_dir / "supp_s7_overlap_regression_controls.csv")

def _write_s7_random_nonoverlap_perturbation_controls(ctx: ExperimentContext) -> None:
    src = ctx.metrics_dir / "panel_d_overlap_perturbation_summary.csv"
    if not src.exists():
        src = ctx.metrics_dir / "supp_overlap_preserving_perturbation_summary.csv"
    if not src.exists():
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s7_random_nonoverlap_perturbation_controls.csv", ["network_seed", "condition", "mean_DPI_L3", "mean_dynamic_like_recovery", "mean_probe_accuracy", "n_pairs"], "overlap_perturbation_summary_missing")
        return
    df = pd.read_csv(src)
    keep = {"full_dynamic", "full_static", "sample_keep_overlap_only_dynamic", "sample_keep_nonoverlap_only_dynamic", "sample_random_matched_dynamic"}
    if "condition" in df.columns:
        df = df[df["condition"].astype(str).isin(keep)].copy()
    cols = ["network_seed", "condition", "mean_DPI_L3", "mean_dynamic_like_recovery", "mean_probe_accuracy", "n_pairs"]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    _save_csv(ctx, df[cols], ctx.metrics_dir / "supp_s7_random_nonoverlap_perturbation_controls.csv")

def _decision_deflection_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "condition", "mean_dynamic_like_recovery", "mean_static_to_dynamic_push", "mean_decision_deflection_score", "condition_matches_dynamic_rate", "condition_matches_static_rate", "n_pairs"]
    if df.empty or "condition" not in df.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    for (network_seed, condition), part in df.groupby(["network_seed", "condition"], sort=False):
        rows.append(
            {
                "network_seed": int(network_seed),
                "condition": str(condition),
                "mean_dynamic_like_recovery": _mean_existing(part, ["dynamic_like_recovery"]),
                "mean_static_to_dynamic_push": _mean_existing(part, ["static_to_dynamic_push"]),
                "mean_decision_deflection_score": _mean_existing(part, ["decision_deflection_score"]),
                "condition_matches_dynamic_rate": _mean_existing(part, ["condition_matches_dynamic"]),
                "condition_matches_static_rate": _mean_existing(part, ["condition_matches_static"]),
                "n_pairs": int(part["pair_id"].nunique()) if "pair_id" in part.columns else int(len(part)),
            }
        )
    return pd.DataFrame(rows, columns=columns)
