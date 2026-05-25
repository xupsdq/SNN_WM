from __future__ import annotations

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as _legacy

# Keep module-level names identical while Fig.3 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_final_support_landscape(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    _save_csv(ctx, _example_landscape_summary(ctx, bank), ctx.metrics_dir / "panel_c_example_landscape_summary.csv")
    if not (ctx.cfg.run_population_morphology_supplement or ctx.cfg.run_supplement):
        ctx.completed_modules["peak_valley_landscape"] = True
        return
    contrast_rows: list[dict[str, Any]] = []
    nonflat_rows: list[dict[str, Any]] = []
    prevalence_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 505)
    for _, meta in _progress(bank.sequence_meta.iterrows(), total=len(bank.sequence_meta), desc="fig3 landscape sequences", enabled=ctx.cfg.show_progress):
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        landscape = bank.landscapes[seq_id]
        g_final = landscape["G_final"]
        peak = landscape["peak_mask"].astype(bool)
        valley = landscape["valley_mask"].astype(bool)
        random_mask = landscape["random_matched_mask"].astype(bool)
        peak_mean = float(g_final[peak].mean()) if np.any(peak) else 0.0
        valley_mean = float(g_final[valley].mean()) if np.any(valley) else 0.0
        random_mean = float(g_final[random_mask].mean()) if np.any(random_mask) else 0.0
        delta = peak_mean - valley_mean
        contrast_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": seq_id,
                "seq_len": seq_len,
                "layer": PRIMARY_LAYER,
                "state_variable": PRIMARY_STATE_VARIABLE,
                "peak_mean_support": peak_mean,
                "valley_mean_support": valley_mean,
                "random_mean_support": random_mean,
                "peak_valley_delta": delta,
                "peak_random_delta": float(peak_mean - random_mean),
                "random_valley_delta": float(random_mean - valley_mean),
                "peak_valley_ratio": float(peak_mean / max(valley_mean, 1e-12)),
            }
        )
        pos = np.clip(g_final - float(np.min(g_final)), 0.0, None)
        total = float(pos.sum())
        nonflat_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": seq_id,
                "seq_len": seq_len,
                "layer": PRIMARY_LAYER,
                "state_variable": PRIMARY_STATE_VARIABLE,
                "support_std": float(g_final.std()),
                "support_cv": float(g_final.std() / max(abs(g_final.mean()), 1e-12)),
                "support_gini": _gini(pos.reshape(-1)),
                "top_q_mass_fraction": float(pos[peak].sum() / max(total, 1e-12)) if np.any(peak) else 0.0,
                "positive_support_area": int((landscape["delta_gain_map"] > 0).sum()),
                "peak_area_fraction": float(peak.mean()),
            }
        )
        null_values = []
        flat = g_final.reshape(-1)
        peak_count = int(peak.sum())
        valley_count = int(valley.sum())
        for _ in _progress(range(int(ctx.cfg.n_null)), total=int(ctx.cfg.n_null), desc="fig3 landscape nulls", enabled=ctx.cfg.show_progress):
            perm = rng.permutation(flat.size)
            p_idx = perm[:peak_count]
            v_idx = perm[peak_count : peak_count + valley_count]
            if len(p_idx) and len(v_idx):
                null_values.append(float(flat[p_idx].mean() - flat[v_idx].mean()))
        null_arr = np.asarray(null_values, dtype=float)
        p95 = float(np.percentile(null_arr, 95)) if null_arr.size else 0.0
        prevalence_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": seq_id,
                "seq_len": seq_len,
                "layer": PRIMARY_LAYER,
                "state_variable": PRIMARY_STATE_VARIABLE,
                "observed_peak_valley_delta": delta,
                "null_peak_valley_delta_mean": float(null_arr.mean()) if null_arr.size else 0.0,
                "null_peak_valley_delta_p95": p95,
                "observed_minus_null": float(delta - (null_arr.mean() if null_arr.size else 0.0)),
                "is_structured": int(delta > p95),
                "n_null": int(ctx.cfg.n_null),
            }
        )
    contrast = pd.DataFrame(contrast_rows)
    nonflat = pd.DataFrame(nonflat_rows)
    prevalence = pd.DataFrame(prevalence_rows)
    _save_csv(ctx, contrast, ctx.metrics_dir / "supp_peak_valley_contrast.csv")
    _save_csv(ctx, nonflat, ctx.metrics_dir / "supp_landscape_nonflatness.csv")
    _save_csv(ctx, prevalence, ctx.metrics_dir / "supp_peak_valley_prevalence.csv")
    _save_csv(ctx, _network_peak_summary(ctx.cfg.network_seed, contrast, nonflat, prevalence), ctx.metrics_dir / "supp_network_peak_valley_summary.csv")
    ctx.completed_modules["peak_valley_landscape"] = True
