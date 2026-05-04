from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plotting.common.colors import get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL
from src.plotting.experiments._common import main_for, read_bundle_csv

EXPERIMENT_ID = "fig4_chunk_interaction_assay"


def _finish_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_NEUTRAL)
    ax.spines["bottom"].set_color(COLOR_NEUTRAL)
    ax.tick_params(labelsize=8, color=COLOR_NEUTRAL)


def _mean_sem(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_cols, sort=True)[value_col].agg(["mean", "count", "std"]).reset_index()
    grouped["sem"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    return grouped


def _metric_triptych(df: pd.DataFrame, metrics: list[tuple[str, str]], *, title: str) -> plt.Figure:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.1), squeeze=False)
    for ax, (metric, ylabel) in zip(axes[0], metrics):
        summary = _mean_sem(df, ["overlap_group"], metric)
        labels = summary["overlap_group"].astype(str).tolist()
        x = np.arange(len(labels), dtype=float)
        ax.bar(
            x,
            summary["mean"].to_numpy(dtype=float),
            yerr=summary["sem"].to_numpy(dtype=float),
            color=[get_plot_color("true_pair"), get_plot_color("anchor")][: len(labels)],
            edgecolor=COLOR_NEUTRAL,
            linewidth=0.7,
            alpha=0.78,
        )
        ax.axhline(0.0, color=COLOR_NEUTRAL, linestyle=":", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        _finish_axis(ax)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _stacked_state_bars(
    df: pd.DataFrame,
    *,
    states: list[str],
    value_cols: list[tuple[str, str]],
    title: str,
    ylabel: str,
) -> plt.Figure:
    sub = df[df["state_condition"].isin(states)].copy()
    summary = sub.groupby(["overlap_group", "state_condition"], sort=True)[[col for col, _ in value_cols]].mean().reset_index()
    overlaps = summary["overlap_group"].astype(str).drop_duplicates().tolist()
    fig, axes = plt.subplots(1, max(len(overlaps), 1), figsize=(max(5.0, 3.0 * max(len(overlaps), 1)), 3.4), squeeze=False)
    colors = [get_plot_color("true_pair"), get_plot_color("anchor"), get_plot_color("shuffled_pair"), get_plot_color("background_shade")]
    for ax, overlap in zip(axes[0], overlaps):
        panel = summary[summary["overlap_group"].astype(str) == overlap]
        x = np.arange(len(states), dtype=float)
        bottom = np.zeros(len(states), dtype=float)
        for idx, (col, label) in enumerate(value_cols):
            vals = []
            for state in states:
                row = panel[panel["state_condition"] == state]
                vals.append(float(row[col].iloc[0]) if not row.empty else 0.0)
            ax.bar(x, vals, bottom=bottom, label=label, color=colors[idx % len(colors)], edgecolor=COLOR_NEUTRAL, linewidth=0.5, alpha=0.82)
            bottom += np.asarray(vals, dtype=float)
        ax.set_title(str(overlap))
        ax.set_xticks(x)
        ax.set_xticklabels(states, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, max(1.0, float(np.nanmax(bottom)) + 0.05))
        _finish_axis(ax)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _causal_summary(member: pd.DataFrame, ping: pd.DataFrame, nonmember: pd.DataFrame) -> plt.Figure:
    states = ["S_AB", "S_AB_minus_chunk_peak", "S_AB_minus_random_peak", "S_AB_minus_nonshared_peak"]
    metrics = [
        ("Ping P_A", ping, "P_A"),
        ("Member probe P_A", member, "P_A_under_probe_A"),
        ("Nonmember excess pair error", nonmember, "excess_within_pair_error"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2), squeeze=False)
    for ax, (title, df, value_col) in zip(axes[0], metrics):
        sub = df[(df["overlap_group"] == "high") & (df["state_condition"].isin(states))].copy()
        summary = _mean_sem(sub, ["state_condition"], value_col)
        vals = []
        errs = []
        for state in states:
            row = summary[summary["state_condition"] == state]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["sem"].iloc[0]) if not row.empty else 0.0)
        x = np.arange(len(states), dtype=float)
        ax.bar(x, vals, yerr=errs, color=get_plot_color("anchor"), edgecolor=COLOR_NEUTRAL, linewidth=0.7, alpha=0.78)
        ax.axhline(0.0, color=COLOR_NEUTRAL, linestyle=":", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(states, rotation=35, ha="right")
        ax.set_title(title)
        _finish_axis(ax)
    fig.suptitle("Causal removal of Layer 3 chunk peaks")
    fig.tight_layout()
    return fig


def _nonmember_error_figure(df: pd.DataFrame) -> plt.Figure:
    states = ["baseline", "S_mean", "S_AB", "S_AB_minus_chunk_peak", "S_AB_minus_random_peak", "S_AB_minus_nonshared_peak"]
    sub = df[df["state_condition"].isin(states)].copy()
    value_cols = [
        ("excess_error_A", "excess A"),
        ("excess_error_B", "excess B"),
        ("excess_error_other", "excess other"),
    ]
    return _stacked_state_bars(
        sub,
        states=states,
        value_cols=value_cols,
        title="Baseline-corrected structured probe errors",
        ylabel="Excess error probability",
    )


def plot_bundle(input_dir: Path):
    peak = read_bundle_csv(
        input_dir,
        "layer3_peak_morphology_metrics.csv",
        ["overlap_group", "shared_peak_excess", "peak_enrichment", "peak_sharpness"],
    )
    ping = read_bundle_csv(input_dir, "ping_decomposed_summary.csv", ["overlap_group", "state_condition", "P_A", "P_B", "P_other", "P_silent"])
    member = read_bundle_csv(input_dir, "member_probe_A_summary.csv", ["overlap_group", "state_condition", "P_A_under_probe_A", "B_intrusion_rate", "P_other"])
    nonmember = read_bundle_csv(
        input_dir,
        "nonmember_probe_error_summary.csv",
        ["overlap_group", "state_condition", "excess_error_A", "excess_error_B", "excess_error_other", "excess_within_pair_error"],
    )
    readout_states = ["S_B", "S_mean", "S_AB", "S_AB_minus_chunk_peak"]
    return {
        "layer3_peak_morphology": _metric_triptych(
            peak,
            [
                ("shared_peak_excess", "Shared peak excess"),
                ("peak_enrichment", "Peak enrichment"),
                ("peak_sharpness", "Peak sharpness"),
            ],
            title="Layer 3 shared-feature peak integration",
        ),
        "ping_decomposed_readout": _stacked_state_bars(
            ping,
            states=readout_states,
            value_cols=[("P_A", "A"), ("P_B", "B"), ("P_other", "other"), ("P_silent", "silent")],
            title="Decomposed neutral-ping readout",
            ylabel="Ping probability",
        ),
        "member_probe_A_readout": _stacked_state_bars(
            member,
            states=readout_states,
            value_cols=[("P_A_under_probe_A", "A"), ("B_intrusion_rate", "B intrusion"), ("P_other", "other")],
            title="Weak member-probe readout",
            ylabel="Probe A probability",
        ),
        "nonmember_probe_structured_error": _nonmember_error_figure(nonmember),
        "causal_peak_removal_summary": _causal_summary(member, ping, nonmember),
    }


if __name__ == "__main__":
    raise SystemExit(main_for(EXPERIMENT_ID, plot_bundle, title="Fig4 Chunk Interaction Assay"))
