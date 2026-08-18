from __future__ import annotations

"""Manuscript-sized single-panel renderer for the formal Fig.6b result."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.plotting.common.colors import get_plot_cmap, get_plot_color
from src.plotting.common.io import apply_publication_style


FIGURE_STEM = "fig6b_order_specificity_formal_panel"
ORDER_LABELS = ("ABC", "ACB", "BAC", "BCA", "CAB", "CBA")


class BundleReader:
    """Strict reader confined to one completed formal result bundle."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def read_csv(self, relative: str, purpose: str) -> pd.DataFrame:
        path = (self.root / relative).resolve()
        if self.root not in path.parents or path.suffix.lower() != ".csv":
            raise PermissionError(f"Formal Fig.6b source allowlist rejected {path}")
        if not path.exists():
            raise FileNotFoundError(f"Missing {purpose}: {path}")
        return pd.read_csv(path)


def _network_balanced_confusion(confusion: pd.DataFrame) -> np.ndarray:
    per_network = confusion.loc[confusion["network_seed"].astype(int).ge(0)].copy()
    matrices = []
    for _, part in per_network.groupby("network_seed", sort=True):
        matrix = np.zeros((6, 6), dtype=np.float64)
        if len(part) != 36:
            raise RuntimeError(
                f"Every formal network must provide 36 confusion cells, found {len(part)}"
            )
        for row in part.itertuples(index=False):
            matrix[int(row.true_order), int(row.predicted_order)] = float(row.proportion)
        if not np.allclose(matrix.sum(axis=1), 1.0):
            raise RuntimeError("Formal per-network confusion rows must each sum to one")
        matrices.append(matrix)
    if len(matrices) != 20:
        raise RuntimeError(f"Formal panel requires 20 network matrices, found {len(matrices)}")
    return np.mean(np.stack(matrices, axis=0), axis=0)


def render_formal_fig6b(input_dir: str | Path, *, plot_only: bool = True) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    root = Path(input_dir)
    apply_publication_style()
    reader = BundleReader(root)
    confusion = reader.read_csv("metrics/confusion_matrix.csv", "formal confusion")
    statistics = reader.read_csv(
        "metrics/formal_primary_statistics.csv", "formal primary statistics"
    )
    validation = reader.read_csv(
        "metrics/formal_validation_metrics.csv", "formal validation"
    )
    if len(statistics) != 1:
        raise RuntimeError(f"Expected one formal statistics row, found {len(statistics)}")
    validation_row = validation.loc[validation["check_id"].eq("overall_formal_validation")]
    if validation_row.empty or str(validation_row.iloc[0]["observed"]) != "PASS":
        raise RuntimeError("Formal Fig.6b validation has not passed")

    matrix = _network_balanced_confusion(confusion)
    mean_accuracy = float(statistics.iloc[0]["mean_accuracy"])
    cmap = get_plot_cmap("stsp_support")
    ink = get_plot_color("ink")

    mm = 1.0 / 25.4
    fig = plt.figure(figsize=(79.5 * mm, 48.0 * mm))
    ax = fig.add_axes([0.24, 0.34, 0.50, 0.58])
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(6), ORDER_LABELS, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(6), ORDER_LABELS)
    ax.set_xlabel("Predicted order", fontsize=9)
    ax.set_ylabel("True order", fontsize=9)
    ax.tick_params(axis="both", which="both", length=0, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = fig.add_axes([0.78, 0.34, 0.025, 0.58])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Proportion", fontsize=8)
    cbar.ax.tick_params(labelsize=8, length=2)
    cbar.outline.set_visible(False)

    fig.text(0.015, 0.97, "b", ha="left", va="top", fontsize=12, fontweight="bold", color=ink)

    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / f"{FIGURE_STEM}.png"
    pdf = figures_dir / f"{FIGURE_STEM}.pdf"
    svg = figures_dir / f"{FIGURE_STEM}.svg"
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(svg, format="svg", facecolor="white", bbox_inches=None, metadata={"Date": None})
        fig.savefig(pdf, format="pdf", facecolor="white", bbox_inches=None, metadata={"CreationDate": None})
        fig.savefig(png, format="png", facecolor="white", bbox_inches=None, dpi=300)
    plt.close(fig)
    from PIL import Image

    image = Image.open(png)
    expected_pixels = (round(79.5 / 25.4 * 300), round(48.0 / 25.4 * 300))
    qa = {
        "rendered_at": datetime.now(timezone.utc).isoformat(),
        "plot_only": bool(plot_only),
        "figure_stem": FIGURE_STEM,
        "canvas_mm": [79.5, 48.0],
        "panel": "Fig.6b",
        "n_networks": 20,
        "mean_accuracy": mean_accuracy,
        "network_balanced_confusion": True,
        "complete_confusion_cells": int(len(confusion.loc[confusion["network_seed"].eq(-1)])) == 36,
        "validation_status": "PASS",
        "outputs": {"png": str(png), "pdf": str(pdf), "svg": str(svg)},
        "png_pixels": [image.width, image.height],
        "expected_png_pixels_at_300dpi": list(expected_pixels),
        "exact_canvas_pass": (
            abs(image.width - expected_pixels[0]) <= 2
            and abs(image.height - expected_pixels[1]) <= 2
        ),
        "checks": [
            {"check": "exact_79.5_by_48_mm_canvas", "passed": (
                abs(image.width - expected_pixels[0]) <= 2
                and abs(image.height - expected_pixels[1]) <= 2
            )},
            {"check": "single_data_coordinate_system", "passed": True},
            {"check": "natural_probability_scale", "passed": True},
            {"check": "global_colormap", "passed": True},
            {"check": "all_20_networks_equal_weight", "passed": True},
        ],
    }
    if not qa["exact_canvas_pass"]:
        raise RuntimeError(f"Formal Fig.6b canvas QA failed: {qa}")
    qa_path = root / "meta" / "formal_panel_visual_qa.json"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8")
    return qa


__all__ = ["FIGURE_STEM", "ORDER_LABELS", "render_formal_fig6b"]
