from __future__ import annotations

from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as _legacy

# Keep module-level names identical while Fig.6 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_ping_score_spike_prediction(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    compute_global_ping_score_spike_prediction(ctx, bank)
