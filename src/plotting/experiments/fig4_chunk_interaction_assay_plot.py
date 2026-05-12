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


def _seed_first_mean_sem(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    seed_cols = ["seed", *group_cols]
    seed_level = df.groupby(seed_cols, sort=True)[value_col].mean().reset_index()
    return _mean_sem(seed_level, group_cols, value_col)


def _state_label(state: str) -> str:
    return "S0 / baseline" if str(state) == "baseline" else str(state)


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


def _ping_decomposed_main(df: pd.DataFrame) -> plt.Figure:
    states = ["baseline", "S_B", "S_AB"]
    value_cols = [("P_A", "A"), ("P_B", "B"), ("P_other", "other"), ("P_silent", "silent")]
    sub = df[df["state_condition"].isin(states)].copy()
    stats = {col: _seed_first_mean_sem(sub, ["state_condition"], col) for col, _ in value_cols}
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    x = np.arange(len(states), dtype=float)
    bottom = np.zeros(len(states), dtype=float)
    colors = [get_plot_color("true_pair"), get_plot_color("anchor"), get_plot_color("shuffled_pair"), get_plot_color("background_shade")]
    for idx, (col, label) in enumerate(value_cols):
        vals = []
        for state in states:
            row = stats[col][stats[col]["state_condition"] == state]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x, vals, bottom=bottom, label=label, color=colors[idx], edgecolor=COLOR_NEUTRAL, linewidth=0.6, alpha=0.86)
        bottom += np.asarray(vals, dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels([_state_label(s) for s in states], rotation=20, ha="right")
    ax.set_ylabel("Readout proportion")
    ax.set_ylim(0.0, max(1.0, float(np.nanmax(bottom)) + 0.05))
    ax.set_title("Neutral ping readout across memory states")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _finish_axis(ax)
    fig.tight_layout()
    return fig


def _weak_probe_completion_curve(df: pd.DataFrame) -> plt.Figure:
    states = ["baseline", "S_B", "S_AB"]
    sub = df[df["state_condition"].isin(states)].copy()
    summary = _seed_first_mean_sem(sub, ["state_condition", "keep_prob"], "P_A")
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    colors = {
        "baseline": get_plot_color("background_shade"),
        "S_B": get_plot_color("anchor"),
        "S_AB": get_plot_color("true_pair"),
    }
    markers = {"baseline": "o", "S_B": "s", "S_AB": "^"}
    for state in states:
        state_df = summary[summary["state_condition"] == state].sort_values("keep_prob")
        if state_df.empty:
            continue
        x = state_df["keep_prob"].to_numpy(dtype=float)
        y = state_df["mean"].to_numpy(dtype=float)
        err = state_df["sem"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, marker=markers[state], linewidth=1.8, markersize=4.5, color=colors[state], label=_state_label(state))
        ax.fill_between(x, np.clip(y - err, 0.0, 1.0), np.clip(y + err, 0.0, 1.0), color=colors[state], alpha=0.16, linewidth=0)
    ax.set_xlabel("Spike keep probability")
    ax.set_ylabel("P(pred = A)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Fused pair states support partial-cue completion")
    ax.legend(frameon=False, fontsize=8, loc="best")
    _finish_axis(ax)
    fig.tight_layout()
    return fig


def plot_bundle(input_dir: Path):
    ping = read_bundle_csv(input_dir, "ping_decomposed_summary.csv", ["seed", "state_condition", "P_A", "P_B", "P_other", "P_silent"])
    weak = read_bundle_csv(input_dir, "weak_probe_A_summary.csv", ["seed", "state_condition", "keep_prob", "P_A", "P_B", "P_other", "P_silent"])
    return {
        "fig4_ping_decomposed_readout": _ping_decomposed_main(ping),
        "fig4_weak_probe_A_completion_curve": _weak_probe_completion_curve(weak),
    }


if __name__ == "__main__":
    raise SystemExit(main_for(EXPERIMENT_ID, plot_bundle, title="Fig4 Chunk Interaction Assay"))
