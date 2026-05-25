from __future__ import annotations

from src.experiments.paper_figures import fig4_overlap_reentry_experiment as _legacy

# Keep module-level names identical while Fig.4 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_overlap_accuracy_identification(ctx: ExperimentContext, bank: OverlapReentryDMSBank | SimilarityBiasCompatibleBank) -> None:
    pair_table = _accuracy_pair_table(ctx, bank)
    compute_high_similarity_overlap_accuracy_drop(ctx, pair_table)
    matches = _build_iso_similarity_overlap_matches(pair_table, ctx.cfg)
    if len(matches) < int(ctx.cfg.min_matches_per_network):
        ctx.warnings.append("Fig.4D iso-similarity matching found fewer than min_matches_per_network matched sets.")
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 404)
    null_df, perm_stats = _matched_overlap_permutation_test(matches, ctx.cfg, rng)
    contrast = _overlap_accuracy_contrast_by_network(matches, int(ctx.cfg.network_seed), perm_stats)
    balance = _matching_balance_diagnostics(matches, int(ctx.cfg.network_seed))
    excess = _compute_overlap_excess_accuracy(pair_table, ctx.cfg)
    regression = _overlap_accuracy_regression(pair_table, int(ctx.cfg.network_seed))
    _save_csv(ctx, pair_table, ctx.metrics_dir / "panel_d_overlap_accuracy_pair_table.csv")
    _save_csv(ctx, matches, ctx.metrics_dir / "panel_d_iso_similarity_matched_pairs.csv")
    _save_csv(ctx, null_df, ctx.metrics_dir / "panel_d_overlap_accuracy_permutation_null.csv")
    _save_csv(ctx, contrast, ctx.metrics_dir / "panel_d_overlap_accuracy_contrast_by_network.csv")
    _save_csv(ctx, balance, ctx.metrics_dir / "panel_d_matching_balance_diagnostics.csv")
    _save_csv(ctx, excess, ctx.metrics_dir / "supp_overlap_excess_accuracy_metrics.csv")
    _save_csv(ctx, regression, ctx.metrics_dir / "supp_overlap_accuracy_regression.csv")
    ctx.completed_modules["overlap_accuracy_identification"] = True

def compute_high_similarity_overlap_accuracy_drop(ctx: ExperimentContext, pair_table: pd.DataFrame) -> None:
    raw, summary, contrast = _high_similarity_overlap_accuracy_drop_tables(pair_table, ctx.cfg)
    _save_csv(ctx, raw, ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop.csv")
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop_summary.csv")
    _save_csv(ctx, contrast, ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop_contrast.csv")
    ctx.completed_modules["high_similarity_overlap_accuracy_drop"] = True
