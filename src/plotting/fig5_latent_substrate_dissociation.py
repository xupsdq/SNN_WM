from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figure_utils_common import (
    COLOR_DONOR_SHIFT,
    COLOR_DYNAMIC,
    COLOR_NOISE,
    COLOR_STATIC,
    PUBLICATION_ANNOTATION_FONT_SIZE,
    PUBLICATION_LINE_WIDTH,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
    validate_required_columns,
)
from paper_plot_style import DEFAULT_SUBPLOT_ADJUST, PANEL_LABEL_FONT_SIZE, apply_paper_style

DEFAULT_TRIPLET_DIR = Path("results/fig5_silent_substrate_triplet")
DEFAULT_CAUSAL_DIR = Path("results/fig5_causal_substrate_dissociation")
DEFAULT_SAVE_DIR = Path("results/fig5_latent_substrate_dissociation")

SUBSTRATE_COLORS = {
    "spike": COLOR_NOISE,
    "membrane": "#D98C2B",
    "stsp": COLOR_DYNAMIC,
}
CONDITION_COLORS = {
    "A_dynamic_base": COLOR_DYNAMIC,
    "B_trial_shuffle_ux": COLOR_DONOR_SHIFT,
    "B_pure_ux_only_shuffle": "#A85D7C",
    "D_spike_silencing": COLOR_NOISE,
    "E_membrane_reset": "#D98C2B",
    "F_stsp_baseline_reset": "#7A1E1E",
}
CONDITION_LABELS = {
    "A_dynamic_base": "Dynamic",
    "B_trial_shuffle_ux": "Shuffle u/x",
    "B_pure_ux_only_shuffle": "Shuffle u/x only",
    "D_spike_silencing": "Spike silence",
    "E_membrane_reset": "Membrane reset",
    "F_stsp_baseline_reset": "STSP reset",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Fig5 latent substrate dissociation from existing experiment outputs.")
    parser.add_argument("--triplet-dir", type=str, default=str(DEFAULT_TRIPLET_DIR))
    parser.add_argument("--causal-dir", type=str, default=str(DEFAULT_CAUSAL_DIR))
    parser.add_argument("--save-dir", type=str, default=str(DEFAULT_SAVE_DIR))
    return parser


def _p_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def _load_csv(path: Path, required_cols) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required result file: {path}")
    df = pd.read_csv(path)
    validate_required_columns(df, required_cols)
    return df


def _plot_triplet_timeseries(ax, df_time: pd.DataFrame, layer_key: str) -> None:
    sub = df_time[df_time["layer"] == layer_key].copy().sort_values(["substrate", "time_bin_start_ms"])
    for substrate, part in sub.groupby("substrate", sort=False):
        color = SUBSTRATE_COLORS[substrate]
        ax.plot(
            part["time_bin_start_ms"].to_numpy(dtype=float),
            part["decode_acc"].to_numpy(dtype=float),
            label=substrate,
            color=color,
            linewidth=PUBLICATION_LINE_WIDTH,
        )
        ax.fill_between(
            part["time_bin_start_ms"].to_numpy(dtype=float),
            part["ci95_lower"].to_numpy(dtype=float),
            part["ci95_upper"].to_numpy(dtype=float),
            color=color,
            alpha=0.18,
        )
    chance = float(sub["chance_level"].iloc[0]) if len(sub) else 0.1
    ax.axhline(chance, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(layer_key.replace("layer", "Layer "))
    ax.set_xlabel("Delay Time (ms)")
    ax.set_ylabel("Decode Accuracy")
    ax.set_ylim(0.0, 1.0)


def _plot_triplet_summary(ax, df_summary: pd.DataFrame) -> None:
    ordered_layers = ["layer1", "layer2", "layer3"]
    ordered_subs = ["spike", "membrane", "stsp"]
    width = 0.22
    x = np.arange(len(ordered_layers))
    for idx, substrate in enumerate(ordered_subs):
        sub = df_summary[df_summary["substrate"] == substrate].set_index("layer").reindex(ordered_layers).reset_index()
        xpos = x + (idx - 1) * width
        bars = ax.bar(
            xpos,
            sub["decode_acc"].to_numpy(dtype=float),
            width=width,
            color=SUBSTRATE_COLORS[substrate],
            alpha=0.92,
            edgecolor="black",
            linewidth=0.8,
            label=substrate,
        )
        for bar, (_, row) in zip(bars, sub.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.03,
                _p_to_stars(float(row["perm_p"])),
                ha="center",
                va="bottom",
                fontsize=PUBLICATION_ANNOTATION_FONT_SIZE,
            )
    ax.axhline(float(df_summary["chance_level"].iloc[0]), color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, ["L1", "L2", "L3"])
    ax.set_ylabel("Whole-Delay Accuracy")
    ax.set_title("Whole-Delay Summary")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=9)


def _plot_confidence(ax, df_confidence: pd.DataFrame, df_summary: pd.DataFrame) -> None:
    anchor_lookup: Dict[str, float] = (
        df_summary.groupby(["layer", "substrate"])["anchor_time_ms"].first().to_dict()
    )
    rows = []
    for _, row in df_confidence[df_confidence["window_name"] == "time_bin"].iterrows():
        key = (row["layer"], row["substrate"])
        if key not in anchor_lookup:
            continue
        if not np.isclose(float(row["eval_bin_start_ms"]), float(anchor_lookup[key]), atol=1e-9):
            continue
        rows.append(row)
    anchor_df = pd.DataFrame(rows)
    if anchor_df.empty:
        ax.text(0.5, 0.5, "No anchor-bin confidence rows", ha="center", va="center")
        ax.set_axis_off()
        return
    sns.boxenplot(
        data=anchor_df,
        x="substrate",
        y="confidence_margin",
        hue="substrate",
        palette=SUBSTRATE_COLORS,
        linewidth=0.8,
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Decision Margin")
    ax.set_title("Anchor-Bin Confidence")


def _plot_condition_summary(ax, df_condition: pd.DataFrame) -> None:
    ordered_conditions = list(CONDITION_LABELS)
    sub = df_condition.set_index("condition").reindex(ordered_conditions).reset_index()
    x = np.arange(len(ordered_conditions))
    width = 0.38
    ax.bar(
        x - width / 2.0,
        sub["acc_probe"].to_numpy(dtype=float),
        width=width,
        color=[CONDITION_COLORS[c] for c in ordered_conditions],
        edgecolor="black",
        linewidth=0.8,
        label="Probe accuracy",
    )
    ax.bar(
        x + width / 2.0,
        sub["sample_related_bias"].to_numpy(dtype=float),
        width=width,
        color="#F2C14E",
        edgecolor="black",
        linewidth=0.8,
        label="Sample-related bias",
    )
    ax.set_xticks(x, [CONDITION_LABELS[c] for c in ordered_conditions], rotation=20, ha="right")
    ax.set_ylabel("Percent")
    ax.set_title("Probe Accuracy and Bias")
    ax.legend(frameon=False, fontsize=9)


def _plot_error_destination(ax, df_error: pd.DataFrame) -> None:
    ordered_conditions = list(CONDITION_LABELS)
    ordered_destinations = ["original_sample", "donor_sample", "silent", "other"]
    pivot = (
        df_error[df_error["destination"].isin(ordered_destinations)]
        .pivot(index="condition", columns="destination", values="rate_percent")
        .reindex(ordered_conditions)
        .fillna(0.0)
    )
    colors = {
        "original_sample": "#E15759",
        "donor_sample": COLOR_DONOR_SHIFT,
        "silent": COLOR_NOISE,
        "other": COLOR_STATIC,
    }
    bottom = np.zeros(len(ordered_conditions), dtype=np.float64)
    x = np.arange(len(ordered_conditions))
    for destination in ordered_destinations:
        values = pivot[destination].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=colors[destination],
            edgecolor="black",
            linewidth=0.6,
            label=destination.replace("_", " "),
        )
        bottom += values
    ax.set_xticks(x, [CONDITION_LABELS[c] for c in ordered_conditions], rotation=20, ha="right")
    ax.set_ylabel("Error Destination (%)")
    ax.set_title("Error Destination Structure")
    ax.legend(frameon=False, fontsize=8, ncol=2)


def plot_triplet_summary_panel(ax: plt.Axes, df_summary: pd.DataFrame) -> None:
    _plot_triplet_summary(ax, df_summary)


def plot_confidence_panel(ax: plt.Axes, df_confidence: pd.DataFrame, df_summary: pd.DataFrame) -> None:
    _plot_confidence(ax, df_confidence, df_summary)


def plot_condition_summary_panel(ax: plt.Axes, df_condition: pd.DataFrame) -> None:
    _plot_condition_summary(ax, df_condition)


def plot_error_destination_panel(ax: plt.Axes, df_error: pd.DataFrame) -> None:
    _plot_error_destination(ax, df_error)


def create_figure(
    df_time: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_confidence: pd.DataFrame,
    df_condition: pd.DataFrame,
    df_error: pd.DataFrame,
) -> plt.Figure:
    fig = plt.figure(figsize=(15.0, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0, 1.0, 1.15])

    axes = {
        "a1_l1": fig.add_subplot(gs[0, 0]),
        "a1_l2": fig.add_subplot(gs[0, 1]),
        "a1_l3": fig.add_subplot(gs[0, 2]),
        "a2": fig.add_subplot(gs[0, 3]),
        "a3": fig.add_subplot(gs[1, 0]),
        "b1": fig.add_subplot(gs[1, 1:3]),
        "b2": fig.add_subplot(gs[1, 3]),
    }

    _plot_triplet_timeseries(axes["a1_l1"], df_time, "layer1")
    _plot_triplet_timeseries(axes["a1_l2"], df_time, "layer2")
    _plot_triplet_timeseries(axes["a1_l3"], df_time, "layer3")
    handles, labels = axes["a1_l3"].get_legend_handles_labels()
    axes["a1_l3"].legend(handles, [label.capitalize() for label in labels], frameon=False, fontsize=9)
    _plot_triplet_summary(axes["a2"], df_summary)
    _plot_confidence(axes["a3"], df_confidence, df_summary)
    _plot_condition_summary(axes["b1"], df_condition)
    _plot_error_destination(axes["b2"], df_error)

    panel_map = {
        "A1": axes["a1_l1"],
        "A2": axes["a2"],
        "A3": axes["a3"],
        "B1": axes["b1"],
        "B2": axes["b2"],
    }
    for label, ax in panel_map.items():
        ax.text(
            -0.18,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=PANEL_LABEL_FONT_SIZE,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    fig.subplots_adjust(**DEFAULT_SUBPLOT_ADJUST)
    return fig


def main() -> None:
    args = build_argparser().parse_args()
    apply_paper_style()

    triplet_dir = Path(args.triplet_dir)
    causal_dir = Path(args.causal_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df_time = _load_csv(
        triplet_dir / "metrics_decode_time_resolved.csv",
        ["layer", "substrate", "time_bin_start_ms", "time_bin_end_ms", "anchor_time_ms", "decode_acc", "perm_p"],
    )
    df_summary = _load_csv(
        triplet_dir / "metrics_decode_window_summary.csv",
        ["layer", "substrate", "window_name", "decode_acc", "ci95_lower", "ci95_upper", "perm_p"],
    )
    df_confidence = _load_csv(
        triplet_dir / "trial_level_decoder_confidence.csv",
        ["trial_id", "layer", "substrate", "eval_bin_start_ms", "pred_label", "true_label", "confidence_margin"],
    )
    df_condition = _load_csv(
        causal_dir / "metrics_condition_summary.csv",
        ["condition", "acc_probe", "sample_related_bias"],
    )
    df_error = _load_csv(
        causal_dir / "metrics_error_destination.csv",
        ["condition", "destination", "rate_percent"],
    )

    fig = create_figure(df_time, df_summary, df_confidence, df_condition, df_error)
    figure_paths = save_figure_all_formats(fig, save_dir / "figure_main")
    plt.close(fig)

    metrics_summary_rows = []
    for _, row in df_summary.iterrows():
        metrics_summary_rows.append(
            {
                "section": "triplet_summary",
                "group": f"{row['layer']}|{row['substrate']}",
                "metric": "decode_acc",
                "value": float(row["decode_acc"]),
            }
        )
    for _, row in df_condition.iterrows():
        for metric in ["acc_probe", "sample_related_bias", "donor_shift_bias"]:
            if metric in row:
                metrics_summary_rows.append(
                    {
                        "section": "causal_summary",
                        "group": row["condition"],
                        "metric": metric,
                        "value": float(row[metric]),
                    }
                )
    metrics_summary_csv = save_tidy_csv(
        pd.DataFrame(metrics_summary_rows),
        save_dir / "metrics_summary.csv",
        sort_by=["section", "group", "metric"],
    )

    save_run_config(
        {
            "triplet_dir": str(triplet_dir),
            "causal_dir": str(causal_dir),
            "output_files": {
                "metrics_summary_csv": metrics_summary_csv,
                "figure_main_png": figure_paths["png"],
                "figure_main_pdf": figure_paths["pdf"],
                "figure_main_svg": figure_paths["svg"],
            },
        },
        save_dir,
    )

    print(f"[Done] Saved: {metrics_summary_csv}")
    print(f"[Done] Saved: {save_dir / 'figure_main.png'}")


if __name__ == "__main__":
    main()
