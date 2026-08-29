from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.paper_figures.fig5.types import ExperimentContext
from src.plotting.common.io import apply_publication_style, save_figure_all_formats

def save_debug_figures(ctx: ExperimentContext) -> None:
    import matplotlib.pyplot as plt

    apply_publication_style()
    metric_files = [
        ("fig5_debug_preprobe_support", ctx.metrics_dir / "panel_a_preprobe_support_metrics.csv", "mean_support"),
        ("fig5_debug_early_firing", ctx.metrics_dir / "panel_b_transition_summary_by_group.csv", "P_advance_plus_recruit"),
        ("fig5_debug_perturbation_transition", ctx.metrics_dir / "panel_d_perturbation_transition_summary_by_group.csv", "P_advance_plus_recruit"),
        ("fig5_debug_same_winner_loss", ctx.metrics_dir / "panel_d_perturbation_transition_summary_by_group.csv", "P_same_winner_lost_or_delayed"),
        ("fig5_debug_chain_summary", ctx.metrics_dir / "supp_event_chain_fraction_metrics.csv", "full_chain_satisfied_fraction"),
    ]
    for stem, path, metric_col in metric_files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if metric_col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        values = pd.to_numeric(df[metric_col], errors="coerce").dropna().to_numpy(dtype=float)
        ax.plot(np.arange(len(values)), values, marker="o", linewidth=1.0)
        ax.set_title(stem)
        ax.set_ylabel(metric_col)
        ax.set_xlabel("row")
        fig.tight_layout()
        save_figure_all_formats(fig, ctx.debug_dir / stem)
        plt.close(fig)
    trace_path = ctx.metrics_dir / "panel_c_event_trace_summary.csv"
    if trace_path.exists():
        df = pd.read_csv(trace_path)
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        for trace_type, part in df.groupby("trace_type"):
            ax.plot(part["time_ms"], part["mean_value"], label=str(trace_type), linewidth=1.0)
        ax.axvline(0, color="0.2", linewidth=0.8)
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("fig5_debug_event_aligned_traces")
        fig.tight_layout()
        save_figure_all_formats(fig, ctx.debug_dir / "fig5_debug_event_aligned_traces")
        plt.close(fig)
    s9_transition = ctx.metrics_dir / "supp_s9_transition_composition_by_group.csv"
    if s9_transition.exists():
        df = pd.read_csv(s9_transition)
        if not df.empty and {"unit_group", "P_advance_plus_recruit"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(4.0, 2.5))
            values = df.groupby("unit_group", sort=False)["P_advance_plus_recruit"].mean(numeric_only=True)
            ax.bar(values.index.astype(str), values.to_numpy(dtype=float))
            ax.set_ylabel("P_advance_plus_recruit")
            ax.set_title("fig5_debug_s9_transition_composition")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            save_figure_all_formats(fig, ctx.debug_dir / "fig5_debug_s9_transition_composition")
            plt.close(fig)
    s9_null = ctx.metrics_dir / "supp_s9_event_chain_null_summary.csv"
    if s9_null.exists():
        df = pd.read_csv(s9_null)
        if not df.empty and {"null_type", "observed_minus_null"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(4.0, 2.5))
            values = df.groupby("null_type", sort=False)["observed_minus_null"].mean(numeric_only=True)
            ax.bar(values.index.astype(str), values.to_numpy(dtype=float))
            ax.set_ylabel("observed_minus_null")
            ax.set_title("fig5_debug_s9_event_chain_null")
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            save_figure_all_formats(fig, ctx.debug_dir / "fig5_debug_s9_event_chain_null")
            plt.close(fig)
    s10_transition = ctx.metrics_dir / "panel_d_perturbation_transition_contrast.csv"
    if s10_transition.exists():
        df = pd.read_csv(s10_transition)
        cols = {"unit_group", "attenuate_delta_P_advance_plus_recruit", "reset_delta_P_advance_plus_recruit"}
        if not df.empty and cols.issubset(df.columns):
            grouped = df.groupby("unit_group", sort=False)[["attenuate_delta_P_advance_plus_recruit", "reset_delta_P_advance_plus_recruit"]].mean(numeric_only=True)
            x = np.arange(len(grouped))
            fig, ax = plt.subplots(figsize=(4.2, 2.6))
            ax.bar(x - 0.18, grouped["attenuate_delta_P_advance_plus_recruit"].to_numpy(dtype=float), width=0.36, label="attenuate")
            ax.bar(x + 0.18, grouped["reset_delta_P_advance_plus_recruit"].to_numpy(dtype=float), width=0.36, label="reset")
            ax.set_xticks(x, grouped.index.astype(str), rotation=30)
            ax.set_ylabel("delta P_advance+recruit")
            ax.set_title("fig5_debug_s10_perturbation_transition")
            ax.legend(frameon=False, fontsize=7)
            fig.tight_layout()
            save_figure_all_formats(fig, ctx.debug_dir / "fig5_debug_s10_perturbation_transition")
            plt.close(fig)
    s10_recovery = ctx.metrics_dir / "supp_s10_dynamic_like_recovery_after_perturbation.csv"
    if s10_recovery.exists():
        df = pd.read_csv(s10_recovery)
        y_col = "dynamic_like_readout_recovery_mean" if "dynamic_like_readout_recovery_mean" in df.columns else "decision_deflection_score_mean"
        if not df.empty and {"condition", y_col}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(4.0, 2.5))
            values = df.groupby("condition", sort=False)[y_col].mean(numeric_only=True)
            ax.bar(values.index.astype(str), values.to_numpy(dtype=float))
            ax.set_ylabel(y_col)
            ax.set_title("fig5_debug_s10_dynamic_like_recovery")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            save_figure_all_formats(fig, ctx.debug_dir / "fig5_debug_s10_dynamic_like_recovery")
            plt.close(fig)
    ctx.completed_modules["debug_figures"] = True
