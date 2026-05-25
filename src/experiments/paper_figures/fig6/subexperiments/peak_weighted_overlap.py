from __future__ import annotations

from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as _legacy

# Keep module-level names identical while Fig.6 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_peak_weighted_overlap_definitions(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    probe_trials = build_probe_candidate_trials(ctx, bank)
    rows: list[dict[str, Any]] = []
    for r in probe_trials.itertuples(index=False):
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": int(r.sequence_id),
                "probe_id": int(r.probe_id),
                "probe_label": int(r.probe_label),
                "raw_overlap": float(r.raw_overlap),
                "peak_weighted_overlap": float(r.peak_weighted_overlap),
                "peak_overlap_fraction": float(r.peak_overlap_fraction),
                "nonpeak_overlap_fraction": float(r.nonpeak_overlap_fraction),
                "visual_similarity": float(r.visual_similarity),
                "input_energy": float(r.input_energy),
                "peak_support_sum": float(r.peak_support_sum),
                "nonpeak_support_sum": float(r.nonpeak_support_sum),
            }
        )
    _save_csv(ctx, pd.DataFrame(rows, columns=PANEL_C_COLUMNS), ctx.metrics_dir / "supp_legacy_panel_c_peak_weighted_overlap_definitions.csv")
    _save_panel_c_example(ctx, bank, probe_trials)
    bank.probe_trials = probe_trials
    ctx.n_probe_candidates = int(len(probe_trials))
    ctx.completed_modules["peak_weighted_overlap"] = True
