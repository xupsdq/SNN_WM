from __future__ import annotations

from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as _legacy

# Keep module-level names identical while Fig.6 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def define_final_peaks_and_update_groups(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    seq_lookup = {int(r.sequence_id): idx for idx, r in enumerate(bank.sequence_meta.itertuples(index=False))}
    for meta in bank.sequence_meta.itertuples(index=False):
        seq_id = int(meta.sequence_id)
        idx = seq_lookup[seq_id]
        seq_len = int(meta.seq_len)
        for unit_id in range(bank.update_count.shape[1]):
            update_count = int(bank.update_count[idx, unit_id])
            last_pos = int(bank.last_update_position[idx, unit_id])
            recent = bool(last_pos > 0 and seq_len - last_pos < int(ctx.cfg.recent_window))
            recency_group = "recent" if recent else "old"
            multiplicity_group = "multi" if update_count >= int(ctx.cfg.multi_update_threshold) else "single"
            group = f"{multiplicity_group}_{recency_group}"
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": seq_id,
                    "seq_len": seq_len,
                    "layer": PRIMARY_LAYER,
                    "state_variable": STATE_VARIABLE,
                    "unit_id": int(unit_id),
                    "update_count": int(update_count),
                    "last_update_position": int(last_pos),
                    "time_since_last_update": int(bank.time_since_last_update[idx, unit_id]),
                    "recency_group": recency_group,
                    "multiplicity_group": multiplicity_group,
                    "update_history_group": group,
                    "is_peak": bool(bank.peak_mask[idx, unit_id]),
                    "final_support": float(bank.g_final[idx, unit_id]),
                    "baseline_support": float(bank.g_baseline[idx, unit_id]),
                    "delta_support": float(bank.delta_support[idx, unit_id]),
                }
            )
    df = pd.DataFrame(rows, columns=PANEL_A_UNIT_COLUMNS)
    overall_peak = float(df["is_peak"].mean()) if not df.empty else float("nan")
    for group in UPDATE_GROUPS:
        part = df[df["update_history_group"].eq(group)]
        summary_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "update_history_group": group,
                "P_peak": float(part["is_peak"].mean()) if len(part) else float("nan"),
                "mean_final_support": float(part["final_support"].mean()) if len(part) else float("nan"),
                "mean_delta_support": float(part["delta_support"].mean()) if len(part) else float("nan"),
                "peak_enrichment": _safe_div(float(part["is_peak"].mean()) if len(part) else float("nan"), overall_peak),
                "n_units": int(len(part)),
            }
        )
    _save_csv(ctx, df, ctx.metrics_dir / "supp_legacy_panel_a_multi_recent_peak_enrichment.csv")
    _save_csv(ctx, pd.DataFrame(summary_rows, columns=PANEL_A_SUMMARY_COLUMNS), ctx.metrics_dir / "supp_legacy_panel_a_peak_enrichment_summary.csv")
    ctx.completed_modules["peak_enrichment"] = True
