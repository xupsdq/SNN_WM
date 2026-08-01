from __future__ import annotations

import argparse
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
from src.experiments.paper_figures.fig2.fixed_b_protocol import (
    FULL_COHORT_SEEDS,
)


ENDPOINTS = (
    "same_B_common_update_cosine",
    "processing_residual_gamma_energy_fraction",
    "full_trace_event_gamma_enrichment",
    "layer1_only_layer2_update_donor_transfer",
    "layer1_only_early_class_score_donor_transfer",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot-only fixed-B v4 confirmatory network inference."
    )
    parser.add_argument(
        "--input-dir",
        default=(
            "results/paper_figure_multi_seed/"
            "fig2_fixed_b_mechanism_confirmatory"
        ),
        help="Figure root containing aggregate fixed-B confirmatory tables.",
    )
    return parser.parse_args(argv)


def _read(path: Path, required: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    validate_required_columns(frame, required)
    return frame


def _plot(
    scalars: pd.DataFrame,
    inference: pd.DataFrame,
) -> plt.Figure:
    scalars = scalars.loc[scalars["endpoint"].isin(ENDPOINTS)].copy()
    inference = inference.loc[inference["endpoint"].isin(ENDPOINTS)].copy()
    observed_seeds = tuple(
        sorted(scalars["network_seed"].astype(int).unique())
    )
    if observed_seeds != FULL_COHORT_SEEDS:
        raise RuntimeError(
            "Fixed-B full-cohort plot requires exactly seeds 1000..1019"
        )

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4))
    colors = {
        1: get_plot_color("old_input"),
        5: get_plot_color("recent_input"),
    }
    panel_specs = (
        (
            axes[0],
            (
                "same_B_common_update_cosine",
                "processing_residual_gamma_energy_fraction",
            ),
            ("Common cosine", r"Residual $\Gamma$"),
            "Common exact-B update",
        ),
        (
            axes[1],
            ("full_trace_event_gamma_enrichment",),
            ("Event–Gamma\nenrichment",),
            "Layer2-presynaptic trace",
        ),
        (
            axes[2],
            (
                "layer1_only_layer2_update_donor_transfer",
                "layer1_only_early_class_score_donor_transfer",
            ),
            ("Layer2 u/x\nupdate", "Early class\nscore"),
            "Layer1-only causal transfer",
        ),
    )
    for axis, endpoints, labels, title in panel_specs:
        for endpoint_index, (endpoint, label) in enumerate(
            zip(endpoints, labels)
        ):
            endpoint_rows = scalars.loc[scalars["endpoint"].eq(endpoint)]
            for k_index, prefix_k in enumerate((1, 5)):
                part = endpoint_rows.loc[
                    endpoint_rows["prefix_k"].eq(prefix_k)
                ].sort_values("network_seed")
                x_center = endpoint_index + (k_index - 0.5) * 0.28
                jitter = np.linspace(-0.045, 0.045, len(part))
                axis.scatter(
                    np.full(len(part), x_center) + jitter,
                    part["value"],
                    s=13,
                    alpha=0.48,
                    color=colors[prefix_k],
                    linewidths=0,
                )
                summary = inference.loc[
                    inference["endpoint"].eq(endpoint)
                    & inference["prefix_k"].eq(prefix_k)
                ].iloc[0]
                mean = float(summary["mean"])
                low = float(summary["ci95_low"])
                high = float(summary["ci95_high"])
                axis.errorbar(
                    [x_center],
                    [mean],
                    yerr=[[mean - low], [high - mean]],
                    fmt="o",
                    color=colors[prefix_k],
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    capsize=3,
                    zorder=4,
                )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(len(labels), dtype=float))
        axis.set_xticklabels(labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].axhline(0.50, color="black", linestyle=":", linewidth=0.9)
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("Network-level metric")
    axes[1].set_ylabel("Changed vs matched-random enrichment")
    axes[2].set_ylabel("Donor-transfer index")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=colors[prefix_k],
            linestyle="none",
            label=f"K={prefix_k}",
        )
        for prefix_k in (1, 5)
    ]
    axes[2].legend(handles=handles, frameon=False, loc="best")
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_dir = Path(args.input_dir).resolve()
    aggregate_dir = input_dir / "aggregate"
    scalars = _read(
        aggregate_dir / "fixed_b_confirmatory_network_scalars.csv",
        ("network_seed", "endpoint", "prefix_k", "value"),
    )
    inference = _read(
        aggregate_dir / "fixed_b_confirmatory_inference.csv",
        (
            "endpoint",
            "prefix_k",
            "mean",
            "ci95_low",
            "ci95_high",
            "holm_adjusted_p",
        ),
    )
    apply_publication_style()
    figure = _plot(scalars, inference)
    outputs = save_figure_all_formats(
        figure,
        input_dir / "figures" / "fig2_fixed_b_confirmatory",
    )
    plt.close(figure)
    manifest = {
        "producer_task": "fixed_b_cohort_aggregate",
        "input_dir": str(input_dir),
        "network_seeds": list(FULL_COHORT_SEEDS),
        "n_networks": len(FULL_COHORT_SEEDS),
        "seed_1000_role": "development_protocol_alignment",
        "outputs": outputs,
    }
    path = input_dir / "fixed_b_confirmatory_plot_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
