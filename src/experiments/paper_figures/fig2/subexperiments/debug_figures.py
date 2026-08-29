from __future__ import annotations

import pandas as pd

from src.experiments.paper_figures.fig2.types import ExperimentContext
from src.plotting.common.io import apply_publication_style, save_figure_all_formats


def save_debug_figures(ctx: ExperimentContext) -> None:
    apply_publication_style()
    jobs = [
        ("panel_b_dual_retention_metrics.csv", "fusion_dual_score", "fig2_debug_dual_retention"),
        ("panel_c_pair_specificity_metrics.csv", "true_minus_shuffled", "fig2_debug_pair_specificity"),
        ("panel_d_pair_level_organization_metrics.csv", "WPRI", "fig2_debug_wpri"),
        ("panel_d_linear_residual_pair_specificity_metrics.csv", "residual_pair_specificity", "fig2_debug_linear_residual"),
        ("panel_e_neutral_ping_metrics.csv", "P_pair", "fig2_debug_real_neutral_ping"),
        ("panel_f_partial_cue_metrics.csv", "P_target", "fig2_debug_real_partial_cue"),
    ]
    for filename, column, stem in jobs:
        path = ctx.metrics_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        part = df[(df.get("layer", "") == "layer3") & (df.get("state_variable", "") == "g")] if "layer" in df.columns else df
        if column not in part.columns or part.empty:
            continue
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=150)
        ax.hist(pd.to_numeric(part[column], errors="coerce").dropna(), bins=20, color="#4C78A8", alpha=0.8)
        ax.set_title(stem)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        save_figure_all_formats(fig, ctx.debug_dir / stem)
        plt.close(fig)
    ping_path = ctx.metrics_dir / "supp_ping_sweep_metrics.csv"
    if ping_path.exists():
        import matplotlib.pyplot as plt

        ping_df = pd.read_csv(ping_path)
        for sweep_type, x_col, stem in (
            ("amplitude", "ping_amp", "supp_ping_amp_sweep_pair_member_readout"),
            ("duration", "ping_ms", "supp_ping_ms_sweep_pair_member_readout"),
        ):
            part = ping_df[ping_df["sweep_type"].astype(str).eq(sweep_type)] if "sweep_type" in ping_df.columns else pd.DataFrame()
            if part.empty or not {x_col, "state_condition", "pair_member_readout_rate"}.issubset(part.columns):
                continue
            fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=150)
            for condition, cond_part in part.groupby("state_condition", sort=True):
                ordered = cond_part.sort_values(x_col)
                ax.plot(ordered[x_col], ordered["pair_member_readout_rate"], marker="o", label=str(condition))
            ax.set_xlabel(x_col)
            ax.set_ylabel("pair_member_readout_rate")
            ax.legend(frameon=False, fontsize=7)
            save_figure_all_formats(fig, ctx.debug_dir / stem)
            plt.close(fig)
    completion_path = ctx.metrics_dir / "supp_completion_delay_sweep_contrast.csv"
    if completion_path.exists():
        import matplotlib.pyplot as plt

        comp_df = pd.read_csv(completion_path)
        if not comp_df.empty and {"delay2_ms", "completion_gain_SAB_minus_SB"}.issubset(comp_df.columns):
            fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=150)
            ordered = comp_df.sort_values("delay2_ms")
            ax.plot(ordered["delay2_ms"], ordered["completion_gain_SAB_minus_SB"], marker="o")
            ax.axhline(0.0, color="0.5", linewidth=0.8)
            ax.set_xlabel("delay2_ms")
            ax.set_ylabel("completion_gain_SAB_minus_SB")
            save_figure_all_formats(fig, ctx.debug_dir / "supp_completion_delay_gain")
            plt.close(fig)
    ctx.completed_modules["debug_figures"] = True


__all__ = ["save_debug_figures"]
