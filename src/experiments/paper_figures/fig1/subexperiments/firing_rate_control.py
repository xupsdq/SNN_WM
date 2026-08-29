from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig1.types import ExperimentContext

def run_phase_firing_rate_control(ctx: ExperimentContext, rows: Sequence[Mapping[str, Any]]) -> None:
    df = pd.DataFrame(list(rows))
    if df.empty:
        df = pd.DataFrame(columns=["network_seed", "trial_id", "layer", "phase", "time_window_ms", "spike_count", "spike_rate_hz"])
    _save_csv(ctx, df, ctx.metrics_dir / "supp_phase_firing_rates.csv")
    ctx.completed_modules["firing_rate_control"] = True


def run_phase_firing_rate_control_from_bank(ctx: ExperimentContext, boundary_bank: Any) -> None:
    run_phase_firing_rate_control(ctx, boundary_bank.phase_rate_rows())
