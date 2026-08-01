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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot-only replay for the fixed-B v4 mechanism experiment."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Existing fixed-B seed bundle containing data/metrics.",
    )
    return parser.parse_args(argv)


def _read(
    metrics_dir: Path,
    filename: str,
    required: Sequence[str],
) -> pd.DataFrame:
    path = metrics_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    validate_required_columns(frame, required)
    return frame


def _plot_mechanism(metrics_dir: Path) -> plt.Figure:
    decomposition = _read(
        metrics_dir,
        "fixed_b_decomposition_summary.csv",
        (
            "prefix_k",
            "mean_same_B_common_update_cosine",
            "mean_processing_residual_gamma_energy_fraction",
            "mean_local_replay_fraction",
            "max_decomposition_relative_error",
        ),
    ).sort_values("prefix_k")
    events = _read(
        metrics_dir,
        "fixed_b_event_gamma_summary.csv",
        (
            "prefix_k",
            "mean_event_gamma_enrichment",
            "mean_changed_event_coordinate_fraction",
            "mean_changed_coordinate_gamma_energy_fraction",
        ),
    ).sort_values("prefix_k")
    swaps = _read(
        metrics_dir,
        "fixed_b_swap_summary.csv",
        (
            "prefix_k",
            "swap_scope",
            "endpoint",
            "mean_donor_transfer_index",
            "valid_coverage",
        ),
    )
    engineering = _read(
        metrics_dir,
        "fixed_b_engineering_gates.csv",
        ("gate", "passed"),
    )
    if not engineering["passed"].eq(1).all():
        failed = engineering.loc[engineering["passed"].ne(1), "gate"].tolist()
        raise RuntimeError(f"Cannot plot engineering-invalid fixed-B bundle: {failed}")

    k_values = decomposition["prefix_k"].to_numpy(dtype=int)
    x = np.arange(len(k_values), dtype=float)
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))

    axes[0].bar(
        x - width / 2,
        decomposition["mean_same_B_common_update_cosine"],
        width,
        label="Free A/C update cosine",
        color=get_plot_color("layer2"),
        edgecolor="black",
        linewidth=0.7,
    )
    axes[0].bar(
        x + width / 2,
        decomposition["mean_processing_residual_gamma_energy_fraction"],
        width,
        label=r"Processing residual $\Gamma$",
        color=get_plot_color("recent_input"),
        edgecolor="black",
        linewidth=0.7,
    )
    axes[0].axhline(0.50, color="black", linestyle=":", linewidth=0.9)
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"K={value}" for value in k_values])
    axes[0].set_ylabel("Network-aggregated metric")
    axes[0].set_title("Common update and residual")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(
        x,
        events["mean_event_gamma_enrichment"],
        width=0.48,
        color=get_plot_color("layer1"),
        edgecolor="black",
        linewidth=0.7,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"K={value}" for value in k_values])
    axes[1].set_ylabel("Changed vs matched-random\nGamma enrichment")
    axes[1].set_title("Full Layer2-presynaptic trace")
    axes[1].grid(axis="y", alpha=0.2)

    layer1 = swaps.loc[
        swaps["swap_scope"].eq("layer1_only")
        & swaps["endpoint"].isin(["layer2_update", "early_class_score"])
    ].copy()
    endpoint_order = ["layer2_update", "early_class_score"]
    colors = [
        get_plot_color("layer2"),
        get_plot_color("whole_pair_representation"),
    ]
    for offset, (endpoint, color) in enumerate(zip(endpoint_order, colors)):
        values = (
            layer1.loc[layer1["endpoint"].eq(endpoint)]
            .set_index("prefix_k")
            .loc[k_values, "mean_donor_transfer_index"]
            .to_numpy(dtype=float)
        )
        axes[2].bar(
            x + (offset - 0.5) * width,
            values,
            width,
            label={
                "layer2_update": "Layer2 u/x update",
                "early_class_score": "Early class score",
            }[endpoint],
            color=color,
            edgecolor="black",
            linewidth=0.7,
        )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"K={value}" for value in k_values])
    axes[2].set_ylabel("Layer1-only donor-transfer index")
    axes[2].set_title("Causal Layer1 STSP swap")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_dir = Path(args.input_dir).resolve()
    metrics_dir = input_dir / "data" / "metrics"
    figures_dir = input_dir / "figures"
    apply_publication_style()
    figure = _plot_mechanism(metrics_dir)
    outputs = save_figure_all_formats(
        figure,
        figures_dir / "fixed_b_v4_mechanism",
    )
    plt.close(figure)
    manifest = {
        "producer_task": "fixed_b_analysis",
        "input_dir": str(input_dir),
        "outputs": outputs,
    }
    manifest_path = input_dir / "fixed_b_plot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
