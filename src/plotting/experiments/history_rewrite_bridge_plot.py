from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_root = Path(args.input_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_root / "figures"
    )
    boundary_path = (
        input_root
        / "boundary_analysis"
        / "data"
        / "metrics"
        / "boundary_transition_inference.csv"
    )
    bridge_path = (
        input_root
        / "bridge"
        / "aggregate"
        / "data"
        / "metrics"
        / "bridge_cohort_inference.csv"
    )
    if not boundary_path.exists():
        raise FileNotFoundError(boundary_path)
    if not bridge_path.exists():
        raise FileNotFoundError(bridge_path)
    boundary = pd.read_csv(boundary_path)
    bridge = pd.read_csv(bridge_path)
    _validate_boundary(boundary)
    _validate_bridge(bridge)
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "simulation_executed": False,
                    "boundary_rows": int(len(boundary)),
                    "bridge_rows": int(len(bridge)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "history_rewrite_bridge_summary.png"
    _plot_summary(boundary, bridge, figure_path)
    manifest = {
        "schema_version": 1,
        "plot_only": True,
        "simulation_executed": False,
        "sources": [
            _file_record(boundary_path, input_root),
            _file_record(bridge_path, input_root),
        ],
        "figures": [_file_record(figure_path, input_root)],
    }
    manifest_path = output_dir / "plot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "simulation_executed": False,
                "figure": str(figure_path),
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate_boundary(frame: pd.DataFrame) -> None:
    required = {
        "endpoint",
        "prefix_k",
        "n_networks",
        "mean",
        "ci95_low",
        "ci95_high",
        "holm_adjusted_p",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Boundary inference is missing columns: {missing}")
    if set(frame["prefix_k"].astype(int)) != {1, 5}:
        raise ValueError("Boundary inference must contain K=1 and K=5")


def _validate_bridge(frame: pd.DataFrame) -> None:
    required = {
        "endpoint",
        "prefix_k",
        "n_networks",
        "mean",
        "ci95_low",
        "ci95_high",
        "holm_adjusted_p",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Bridge inference is missing columns: {missing}")
    expected = {
        (
            "layer1_to_layer2_update_donor_transfer",
            1,
        ),
        (
            "layer1_to_layer2_update_donor_transfer",
            5,
        ),
        (
            "layer1_to_early_class_score_donor_transfer",
            1,
        ),
        (
            "layer1_to_early_class_score_donor_transfer",
            5,
        ),
    }
    found = set(
        zip(
            frame["endpoint"].astype(str),
            frame["prefix_k"].astype(int),
        )
    )
    if found != expected:
        raise ValueError(
            f"Bridge endpoint set mismatch: expected={expected}, found={found}"
        )


def _plot_summary(
    boundary: pd.DataFrame,
    bridge: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    boundary_order = (
        "joint_ux_input_driven_boundary_displacement",
        "processing_residual_gamma_energy_fraction",
        "layer1_only_layer2_update_donor_transfer",
        "layer1_only_early_class_score_donor_transfer",
    )
    boundary_labels = {
        "joint_ux_input_driven_boundary_displacement": "Boundary u/x update",
        "processing_residual_gamma_energy_fraction": "History-dependent B rewrite",
        "layer1_only_layer2_update_donor_transfer": "Existing L1→L2 transfer",
        "layer1_only_early_class_score_donor_transfer": "Existing early-score transfer",
    }
    colors = {1: "#3366CC", 5: "#DC3912"}
    ax = axes[0]
    for prefix_k in (1, 5):
        part = boundary.loc[
            boundary["prefix_k"].eq(prefix_k)
        ].set_index("endpoint")
        means = np.asarray(
            [float(part.loc[name, "mean"]) for name in boundary_order]
        )
        lows = np.asarray(
            [float(part.loc[name, "ci95_low"]) for name in boundary_order]
        )
        highs = np.asarray(
            [float(part.loc[name, "ci95_high"]) for name in boundary_order]
        )
        x = np.arange(len(boundary_order)) + (-0.08 if prefix_k == 1 else 0.08)
        ax.errorbar(
            x,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            marker="o",
            capsize=3,
            linewidth=1.5,
            color=colors[prefix_k],
            label=f"K={prefix_k}",
        )
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xticks(
        np.arange(len(boundary_order)),
        [boundary_labels[name] for name in boundary_order],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Network-level mean effect")
    ax.set_title("Existing-result transition chain")
    ax.legend(frameon=False)

    ax = axes[1]
    bridge_order = (
        "layer1_to_layer2_update_donor_transfer",
        "layer1_to_early_class_score_donor_transfer",
    )
    bridge_labels = {
        bridge_order[0]: "C-induced Layer 2 u/x update",
        bridge_order[1]: "Early Layer 3 class score",
    }
    for prefix_k in (1, 5):
        part = bridge.loc[
            bridge["prefix_k"].eq(prefix_k)
        ].set_index("endpoint")
        means = np.asarray(
            [float(part.loc[name, "mean"]) for name in bridge_order]
        )
        lows = np.asarray(
            [float(part.loc[name, "ci95_low"]) for name in bridge_order]
        )
        highs = np.asarray(
            [float(part.loc[name, "ci95_high"]) for name in bridge_order]
        )
        x = np.arange(len(bridge_order)) + (-0.08 if prefix_k == 1 else 0.08)
        ax.errorbar(
            x,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            marker="o",
            capsize=3,
            linewidth=1.5,
            color=colors[prefix_k],
            label=f"K={prefix_k}",
        )
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xticks(
        np.arange(len(bridge_order)),
        [bridge_labels[name] for name in bridge_order],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Layer1-only donor-transfer coefficient")
    ax.set_title("Direct post-B → same-C bridge")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _file_record(path: Path, root: Path) -> dict[str, object]:
    path = Path(path)
    try:
        name = path.relative_to(root).as_posix()
    except ValueError:
        name = str(path)
    return {
        "path": name,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot saved history-rewrite bridge outputs without simulations."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input-dir",
        default="results/paper_figure_multi_seed/history_rewrite_bridge",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
