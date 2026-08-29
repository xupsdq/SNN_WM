from __future__ import annotations

import pandas as pd

from src.experiments.paper_figures.fig6.types import ExperimentContext
from src.plotting.common.io import apply_publication_style, save_figure_all_formats



def save_debug_figures(ctx: ExperimentContext) -> None:
    apply_publication_style()
    debug_specs = [
        ("fig6_debug_peak_source_attribution", ctx.metrics_dir / "panel_a_peak_source_attribution_summary.csv", "relative_position_from_end", "mean_peak_loss_fraction"),
        ("fig6_debug_peak_update_history", ctx.metrics_dir / "panel_b_peak_update_history_summary.csv", "group", "mean_update_count"),
        ("fig6_debug_peak_input_overlap_origin", ctx.metrics_dir / "panel_c_peak_input_overlap_summary.csv", "overlap_window", "mean_dice"),
        ("fig6_debug_real_reentry", ctx.metrics_dir / "panel_d_peak_weighted_reentry_metrics.csv", "peak_weighted_overlap", "reentry_strength"),
        ("fig6_debug_real_downstream", ctx.metrics_dir / "panel_e_peak_weighted_downstream_metrics.csv", "peak_weighted_overlap", "metric_value"),
        ("fig6_debug_s11_update_group_enrichment", ctx.metrics_dir / "supp_s11_peak_update_group_enrichment.csv", "update_group", "P_peak"),
        ("fig6_debug_s11_recent_overlap_window", ctx.metrics_dir / "supp_s11_recent_overlap_window_robustness.csv", "recent_k", "dice_peak_overlap"),
        ("fig6_debug_s12_matched_peak_overlap_contrast", ctx.metrics_dir / "supp_s12_raw_overlap_matched_peak_overlap_contrast.csv", "matched_set_id", "reentry_high_minus_low"),
        ("fig6_debug_s12_downstream_metric_breakdown", ctx.metrics_dir / "supp_s12_downstream_metric_breakdown.csv", "downstream_metric", "beta_peak_weighted_overlap"),
        ("fig6_debug_real_rollout_audit", ctx.metrics_dir / "panel_de_real_rollout_scientific_use_audit.csv", "module", "final_scientific_use"),
        ("fig6_debug_s12_peak_perturbation", ctx.metrics_dir / "supp_s12_peak_perturbation_summary.csv", "metric", "overlap_minus_control_reduction"),
    ]
    for name, path, x_col, y_col in debug_specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if x_col not in df.columns or y_col not in df.columns:
            continue
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.2, 2.2), dpi=160)
        x = df[x_col]
        y = pd.to_numeric(df[y_col], errors="coerce")
        if pd.api.types.is_numeric_dtype(x):
            ax.scatter(pd.to_numeric(x, errors="coerce"), y, s=10)
        else:
            order = list(dict.fromkeys(map(str, x.tolist())))
            ax.scatter([order.index(str(v)) for v in x], y, s=10)
            ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        save_figure_all_formats(fig, ctx.debug_dir / name)
        plt.close(fig)
    _save_global_debug_figure(ctx)

def _save_global_debug_figure(ctx: ExperimentContext) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 1.8), dpi=160)
    ax.axis("off")
    ax.text(0.05, 0.65, "Overlap provides route", fontsize=9, weight="bold")
    ax.text(0.55, 0.65, "Peaks provide gain", fontsize=9, weight="bold")
    ax.annotate("", xy=(0.50, 0.65), xytext=(0.35, 0.65), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.text(0.05, 0.25, "Predictive unless peak perturbation succeeds", fontsize=7)
    save_figure_all_formats(fig, ctx.debug_dir / "fig6_debug_global_mechanism")
    plt.close(fig)
