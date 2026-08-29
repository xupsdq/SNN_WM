from __future__ import annotations

from typing import Any

import pandas as pd

from src.experiments.paper_figures.fig6.constants import PANEL_C_COLUMNS
from src.experiments.paper_figures.fig6.subexperiments.helpers_1 import _save_csv
from src.experiments.paper_figures.fig6.subexperiments.helpers_2 import _save_panel_c_example
from src.experiments.paper_figures.fig6.subexperiments.real_reentry_rollout import build_probe_candidate_trials
from src.experiments.paper_figures.fig6.types import ExperimentContext, PeakAmplifiedReentryBank



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
