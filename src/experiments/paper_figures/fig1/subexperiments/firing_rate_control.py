from __future__ import annotations

from src.experiments.paper_figures import fig1_functional_stsp_substrate_experiment as _legacy

# During the first split, keep helper/global resolution identical to the legacy module.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def run_phase_firing_rate_control(ctx: ExperimentContext, rows: Sequence[Mapping[str, Any]]) -> None:
    df = pd.DataFrame(list(rows))
    if df.empty:
        df = pd.DataFrame(columns=["network_seed", "trial_id", "layer", "phase", "time_window_ms", "spike_count", "spike_rate_hz"])
    _save_csv(ctx, df, ctx.metrics_dir / "supp_phase_firing_rates.csv")
    ctx.completed_modules["firing_rate_control"] = True
