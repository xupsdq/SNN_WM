from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plotting.common.io import (
    apply_publication_style,
    get_plot_color,
    save_figure_all_formats,
    validate_required_columns,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot-only QA replay for the reorganized manuscript P0 reanalysis."
    )
    parser.add_argument(
        "--input-dir",
        default="results/paper_figure_multi_seed/new_results_reanalysis",
        help="Existing reanalysis result bundle.",
    )
    return parser.parse_args(argv)


def _read(metrics_dir: Path, filename: str, required: Sequence[str]) -> pd.DataFrame:
    path = metrics_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    validate_required_columns(frame, required)
    return frame


def _mean_sem(frame: pd.DataFrame, group_columns: list[str], value: str) -> pd.DataFrame:
    return (
        frame.groupby(group_columns, as_index=False)
        .agg(
            mean=(value, "mean"),
            sem=(value, lambda values: float(np.std(values, ddof=1) / np.sqrt(len(values)))),
        )
    )


def _plot_fig1(metrics_dir: Path) -> plt.Figure:
    phase = _read(
        metrics_dir,
        "fig1_phase_firing_network_metrics.csv",
        ("network_seed", "layer", "phase", "mean_spike_rate_hz"),
    )
    summary = _mean_sem(phase, ["layer", "phase"], "mean_spike_rate_hz")
    order = ["stimulus", "early_delay", "late_delay", "probe"]
    x = np.arange(len(order), dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = {
        "layer1": get_plot_color("layer1"),
        "layer2": get_plot_color("layer2"),
        "layer3": get_plot_color("layer3"),
    }
    for layer in ("layer1", "layer2", "layer3"):
        part = summary.loc[summary["layer"].eq(layer)].set_index("phase").loc[order]
        ax.errorbar(
            x,
            np.log10(part["mean"].to_numpy(dtype=float) + 1.0),
            yerr=part["sem"].to_numpy(dtype=float)
            / ((part["mean"].to_numpy(dtype=float) + 1.0) * np.log(10.0)),
            marker="o",
            linewidth=1.8,
            label=layer.replace("layer", "Layer "),
            color=colors[layer],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Stimulus", "Early delay", "Late delay", "Probe"])
    ax.set_ylabel(r"$\log_{10}(\mathrm{rate}+1)$")
    ax.set_title("Population firing across episode phases")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return fig


def _plot_fig3(metrics_dir: Path) -> plt.Figure:
    event = _read(
        metrics_dir,
        "fig3_event_chain_network_metrics.csv",
        ("network_seed", "null_type", "observed_minus_null"),
    )
    event = event.loc[event["null_type"].eq("conservative_max_across_five_nulls")]
    writeback = _read(
        metrics_dir,
        "fig3_writeback_network_metrics.csv",
        ("network_seed", "conditional_difference_in_differences"),
    )
    path = _read(
        metrics_dir,
        "fig3_same_trial_path_network_metrics.csv",
        ("network_seed", "standardized_l1_to_l2_beta", "incremental_r2"),
    )
    endpoints = [
        ("Event chain\nminus null", event["observed_minus_null"].to_numpy(float)),
        (
            "Layer2\nwrite-back DID",
            writeback["conditional_difference_in_differences"].to_numpy(float),
        ),
        (
            "L1→L2\nstandardized β",
            path["standardized_l1_to_l2_beta"].to_numpy(float),
        ),
        ("L1→L2\nincremental $R^2$", path["incremental_r2"].to_numpy(float)),
    ]
    means = [float(values.mean()) for _, values in endpoints]
    sems = [float(values.std(ddof=1) / np.sqrt(len(values))) for _, values in endpoints]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(endpoints), dtype=float)
    ax.bar(
        x,
        means,
        yerr=sems,
        color=[
            get_plot_color("dynamic"),
            get_plot_color("layer2"),
            get_plot_color("layer1"),
            get_plot_color("whole_pair_representation"),
        ],
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in endpoints])
    ax.set_ylabel("Network-level effect")
    ax.set_title("Layer1 processing to Layer2 write-back evidence")
    ax.grid(axis="y", alpha=0.2)
    return fig


def _plot_fig4(metrics_dir: Path) -> plt.Figure:
    stage = _read(
        metrics_dir,
        "fig4_layer2_progressive_stage_metrics.csv",
        ("network_seed", "state_variable", "stage_k", "observed_minus_natural_decay"),
    )
    summary = _mean_sem(
        stage,
        ["state_variable", "stage_k"],
        "observed_minus_natural_decay",
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    styles = {
        "u": (get_plot_color("old_input"), "o"),
        "x": (get_plot_color("recent_input"), "s"),
        "ux_joint_mean": (get_plot_color("layer2"), "^"),
    }
    for variable in ("u", "x", "ux_joint_mean"):
        part = summary.loc[summary["state_variable"].eq(variable)].sort_values("stage_k")
        color, marker = styles[variable]
        ax.errorbar(
            part["stage_k"],
            part["mean"],
            yerr=part["sem"],
            marker=marker,
            color=color,
            linewidth=1.8,
            label={"u": "Layer2 u", "x": "Layer2 x", "ux_joint_mean": "u/x mean"}[variable],
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Sequence stage")
    ax.set_ylabel("Observed minus passive displacement")
    ax.set_title("Layer2 STSP updates recur with diminishing increments")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return fig


def _plot_fig6(metrics_dir: Path) -> plt.Figure:
    pair = _read(
        metrics_dir,
        "fig6_layer2_pair_network_metrics.csv",
        (
            "network_seed",
            "min_component_similarity",
            "true_minus_shuffled",
            "residual_pair_specificity",
        ),
    )
    multi = _read(
        metrics_dir,
        "fig6_layer2_multi_network_metrics.csv",
        ("network_seed", "seq_len", "n_eff"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
    pair_endpoints = [
        ("Dual\nsimilarity", pair["min_component_similarity"].to_numpy(float)),
        ("Pair\nspecificity", pair["true_minus_shuffled"].to_numpy(float)),
        (
            "Residual\nspecificity",
            pair["residual_pair_specificity"].to_numpy(float),
        ),
    ]
    x = np.arange(len(pair_endpoints), dtype=float)
    axes[0].bar(
        x,
        [values.mean() for _, values in pair_endpoints],
        yerr=[values.std(ddof=1) / np.sqrt(len(values)) for _, values in pair_endpoints],
        color=[
            get_plot_color("whole_pair_representation"),
            get_plot_color("true_pair"),
            get_plot_color("other_residual"),
        ],
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label for label, _ in pair_endpoints])
    axes[0].set_ylabel("Layer2 u/x metric")
    axes[0].set_title("Pair organization")
    axes[0].grid(axis="y", alpha=0.2)

    multi_summary = _mean_sem(multi, ["seq_len"], "n_eff").sort_values("seq_len")
    axes[1].errorbar(
        multi_summary["seq_len"],
        multi_summary["mean"],
        yerr=multi_summary["sem"],
        marker="o",
        linewidth=1.8,
        color=get_plot_color("layer2"),
    )
    axes[1].plot(
        multi_summary["seq_len"],
        multi_summary["seq_len"],
        linestyle=":",
        color=get_plot_color("other_residual"),
        label="Identity",
    )
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Sequence length K")
    axes[1].set_ylabel(r"Layer2 u/x $N_{\mathrm{eff}}$")
    axes[1].set_title("Multi-input constituent expression")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_dir = Path(args.input_dir).resolve()
    metrics_dir = input_dir / "metrics"
    figures_dir = input_dir / "figures"
    apply_publication_style()
    builders = {
        "qa_fig1_silent_state": _plot_fig1,
        "qa_fig3_processing_writeback": _plot_fig3,
        "qa_fig4_layer2_progressive": _plot_fig4,
        "qa_fig6_layer2_organization": _plot_fig6,
    }
    outputs: dict[str, dict[str, str]] = {}
    for stem, builder in builders.items():
        figure = builder(metrics_dir)
        outputs[stem] = save_figure_all_formats(figure, figures_dir / stem)
        plt.close(figure)
    manifest_path = input_dir / "plot_manifest.json"
    manifest_path.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_path = input_dir / "artifact_manifest.json"
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["files"] = {
            path.relative_to(input_dir).as_posix(): _sha256_file(path)
            for path in sorted(input_dir.rglob("*"))
            if path.is_file() and path != artifact_path
        }
        artifact_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
