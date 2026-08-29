from __future__ import annotations

from src.experiments.paper_figures.fig6.constants import DOWNSTREAM_METRICS
from src.experiments.paper_figures.fig6.subexperiments.helpers_1 import _save_csv
from src.experiments.paper_figures.fig6.subexperiments.helpers_2 import _df_all_proxy, _regression_rows
from src.experiments.paper_figures.fig6.types import ExperimentContext, PeakAmplifiedReentryBank


def _write_standardized_panel_e_outputs(*args, **kwargs):
    from src.experiments.paper_figures.fig6.subexperiments.supplement import _write_standardized_panel_e_outputs as _impl

    return _impl(*args, **kwargs)



def compute_real_peak_overlap_downstream_metrics(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    df = bank.downstream_metrics.copy()
    _save_csv(ctx, df, ctx.metrics_dir / "panel_e_real_downstream_metrics.csv")
    reg = _regression_rows(
        ctx,
        df,
        metrics=(
            "early_recruitment_gain_real",
            "P_advance_real",
            "P_recruit_real",
            "spike_advance_real",
            "response_pattern_displacement_real",
            "decision_deflection_score_real",
            "partial_cue_completion_gain_real",
        ),
        n_name="n_trials",
    )
    reg["proxy_mode"] = bool(_df_all_proxy(df))
    _save_csv(ctx, reg, ctx.metrics_dir / "panel_e_peak_overlap_downstream_regression.csv")
    _write_standardized_panel_e_outputs(ctx, df)
    ctx.completed_modules["real_downstream_metrics"] = True

def compute_peak_weighted_downstream_metrics(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    df = bank.downstream_metrics.copy()
    _save_csv(ctx, df, ctx.metrics_dir / "supp_legacy_panel_e_peak_weighted_downstream_metrics.csv")
    metrics = tuple(metric for metric in DOWNSTREAM_METRICS if metric in df.columns)
    if not metrics:
        metrics = (
            "early_recruitment_gain_real",
            "P_advance_real",
            "P_recruit_real",
            "spike_advance_real",
            "response_pattern_displacement_real",
            "decision_deflection_score_real",
            "partial_cue_completion_gain_real",
        )
    reg = _regression_rows(ctx, df, metrics=metrics, n_name="n_trials")
    _save_csv(ctx, reg, ctx.metrics_dir / "supp_legacy_panel_e_downstream_regression.csv")
    ctx.completed_modules["downstream_prediction"] = True
