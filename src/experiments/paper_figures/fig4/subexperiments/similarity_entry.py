from __future__ import annotations

from src.experiments.paper_figures import fig4_overlap_reentry_experiment as _legacy

# Keep module-level names identical while Fig.4 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_similarity_entry_metrics(ctx: ExperimentContext, bank: OverlapReentryDMSBank) -> None:
    rows = []
    for _, pair in bank.pair_trials.iterrows():
        pair_id = int(pair["pair_id"])
        dyn = _cond_row(bank.condition_metrics, pair_id, "full_dynamic")
        sta = _cond_row(bank.condition_metrics, pair_id, "full_static")
        b_vec = _vec_distance(bank, pair_id, "full_dynamic", "full_static")
        dpi = _mean_dpi(bank, pair_id, "full_dynamic")
        defl = _decision_deflection(bank, pair_id, "full_dynamic")
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "pair_id": pair_id,
                "sample_image_id": int(pair["sample_image_id"]),
                "probe_image_id": int(pair["probe_image_id"]),
                "sample_label": int(pair["sample_label"]),
                "probe_label": int(pair["probe_label"]),
                "pixel_similarity": float(pair["pixel_similarity"]),
                "similarity_bin": str(pair["similarity_bin"]),
                "pred_dynamic": int(dyn["prediction"]),
                "pred_static": int(sta["prediction"]),
                "correct_dynamic": int(dyn["correctness"]),
                "correct_static": int(sta["correctness"]),
                "acc_drop": int(sta["correctness"]) - int(dyn["correctness"]),
                "b_vec": b_vec,
                "DPI_L3": dpi,
                "decision_deflection": defl,
            }
        )
    df = pd.DataFrame(rows)
    summary = _summary_by_bin(df, "similarity_bin", "pixel_similarity")
    _save_csv(ctx, df, ctx.metrics_dir / "panel_b_similarity_entry_metrics.csv")
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_b_similarity_bin_summary.csv")
    _save_csv(ctx, _panel_b_accuracy_drop_summary(df), ctx.metrics_dir / "panel_b_similarity_accuracy_drop_summary.csv")
    _save_csv(ctx, summary.copy(), ctx.metrics_dir / "supp_similarity_bin_full_stats.csv")
    ctx.completed_modules["similarity_entry"] = True
