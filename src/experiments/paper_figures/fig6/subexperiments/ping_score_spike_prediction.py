from __future__ import annotations

from src.experiments.paper_figures.fig6.subexperiments.global_ping_score_spike_prediction import (
    compute_global_ping_score_spike_prediction,
)
from src.experiments.paper_figures.fig6.types import ExperimentContext, PeakAmplifiedReentryBank



def compute_ping_score_spike_prediction(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    compute_global_ping_score_spike_prediction(ctx, bank)
