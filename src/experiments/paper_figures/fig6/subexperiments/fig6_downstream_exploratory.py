from __future__ import annotations

from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as _legacy

# Keep module-level names identical while Fig.6 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_fig6_downstream_exploratory(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    if bank.downstream_metrics.empty:
        _save_csv(ctx, pd.DataFrame(columns=["network_seed", "metric", "value", "notes"]), ctx.metrics_dir / "supp_fig6_downstream_exploratory.csv")
    else:
        rows = [{"network_seed": int(ctx.cfg.network_seed), "metric": col, "value": _mean_col(bank.downstream_metrics, col), "notes": "optional exploratory L3/readout metric; not required for main Fig.6"} for col in bank.downstream_metrics.columns if col.endswith("_real")]
        _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "supp_fig6_downstream_exploratory.csv")
    ctx.completed_modules["fig6_downstream_exploratory"] = True
