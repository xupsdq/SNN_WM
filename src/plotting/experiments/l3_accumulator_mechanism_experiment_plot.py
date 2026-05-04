from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plotting.common.colors import get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL, GRID_ALPHA_SOFT
from src.plotting.experiments._common import load_bundle_npz, main_for, read_bundle_csv


FIGURE_STEM = "figure_D_main_neural_decision_coupling"
REQUIRED_RESULT_COLUMNS = ("pair_id", "replacement_push_kstar", "replacement_pullback_kstar")
REQUIRED_VECTOR_KEYS = ("pair_id", "delta_V", "Delta_hat_plus", "Delta_hat_minus")


def _positive_robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return 1.0
    scale = float(np.nanpercentile(finite, 90))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _decision_projection(delta_hat: np.ndarray, delta_v: np.ndarray) -> np.ndarray:
    denom = np.sum(delta_v * delta_v, axis=1)
    numer = np.sum(delta_hat * delta_v, axis=1)
    out = np.zeros_like(numer, dtype=np.float64)
    valid = np.isfinite(denom) & (denom > 1e-16)
    out[valid] = numer[valid] / denom[valid]
    return out


def _normalized_shift(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64) / max(float(scale), 1e-12), 0.0, 1.0)


def _aligned_results_and_vectors(df: pd.DataFrame, vectors: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    for key in REQUIRED_VECTOR_KEYS:
        if key not in vectors:
            raise KeyError(f"pair_vectors.npz missing required array: {key}")
    vector_pair_ids = np.asarray(vectors["pair_id"], dtype=np.int64)
    vector_order = {int(pair_id): index for index, pair_id in enumerate(vector_pair_ids)}
    missing = sorted(set(df["pair_id"].astype(int)) - set(vector_order))
    if missing:
        preview = ", ".join(str(item) for item in missing[:8])
        raise ValueError(f"pair_vectors.npz missing pair_id values from pair_results.csv: {preview}")

    ordered_df = df.sort_values("pair_id", kind="stable").reset_index(drop=True)
    indices = np.asarray([vector_order[int(pair_id)] for pair_id in ordered_df["pair_id"]], dtype=np.int64)
    ordered_vectors = {key: np.asarray(value)[indices] for key, value in vectors.items() if key in REQUIRED_VECTOR_KEYS}
    return ordered_df, ordered_vectors


def _build_trajectory_table(df: pd.DataFrame, vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    df, vectors = _aligned_results_and_vectors(df, vectors)
    delta_v = np.asarray(vectors["delta_V"], dtype=np.float64)
    delta_hat_plus = np.asarray(vectors["Delta_hat_plus"], dtype=np.float64)
    delta_hat_minus = np.asarray(vectors["Delta_hat_minus"], dtype=np.float64)

    plus_fire = df["replacement_push_kstar"].to_numpy(dtype=np.float64)
    minus_fire = df["replacement_pullback_kstar"].to_numpy(dtype=np.float64)
    plus_decision = _decision_projection(delta_hat_plus, delta_v)
    minus_decision = _decision_projection(delta_hat_minus, delta_v)

    plus_fire_shift = _normalized_shift(plus_fire, _positive_robust_scale(plus_fire))
    minus_fire_shift = _normalized_shift(minus_fire, _positive_robust_scale(minus_fire))
    plus_decision_shift = _normalized_shift(plus_decision, _positive_robust_scale(plus_decision))
    minus_decision_shift = _normalized_shift(minus_decision, _positive_robust_scale(minus_decision))

    return pd.DataFrame(
        {
            "pair_id": df["pair_id"].astype(int),
            "plus_x0": -1.0,
            "plus_y0": -1.0,
            "plus_x1": -1.0 + 2.0 * plus_fire_shift,
            "plus_y1": -1.0 + 2.0 * plus_decision_shift,
            "minus_x0": 1.0,
            "minus_y0": 1.0,
            "minus_x1": 1.0 - 2.0 * minus_fire_shift,
            "minus_y1": 1.0 - 2.0 * minus_decision_shift,
        }
    )


def _draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    alpha: float,
    linewidth: float,
    mutation_scale: float,
    zorder: int,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        mutation_scale=mutation_scale,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _draw_group(
    ax: plt.Axes,
    table: pd.DataFrame,
    *,
    prefix: str,
    color: str,
    marker: str,
) -> None:
    x0 = table[f"{prefix}_x0"].to_numpy(dtype=np.float64)
    y0 = table[f"{prefix}_y0"].to_numpy(dtype=np.float64)
    x1 = table[f"{prefix}_x1"].to_numpy(dtype=np.float64)
    y1 = table[f"{prefix}_y1"].to_numpy(dtype=np.float64)

    for start_x, start_y, end_x, end_y in zip(x0, y0, x1, y1):
        ax.plot(
            [float(start_x), float(end_x)],
            [float(start_y), float(end_y)],
            color=color,
            alpha=0.07,
            linewidth=0.55,
            zorder=2,
        )

    ax.scatter(x0, y0, s=16, marker=marker, facecolors="white", edgecolors=color, linewidths=0.55, alpha=0.20, zorder=3)
    ax.scatter(x1, y1, s=18, marker=marker, facecolors=color, edgecolors="white", linewidths=0.4, alpha=0.20, zorder=4)

    mean_start = (float(np.nanmean(x0)), float(np.nanmean(y0)))
    mean_end = (float(np.nanmean(x1)), float(np.nanmean(y1)))
    _draw_arrow(
        ax,
        mean_start,
        mean_end,
        color=color,
        alpha=0.98,
        linewidth=3.0,
        mutation_scale=20,
        zorder=6,
    )
    ax.scatter([mean_start[0]], [mean_start[1]], s=86, marker=marker, facecolors="white", edgecolors=color, linewidths=1.8, zorder=7)
    ax.scatter([mean_end[0]], [mean_end[1]], s=96, marker=marker, facecolors=color, edgecolors="white", linewidths=1.0, zorder=8)


def _plot_d_main(table: pd.DataFrame) -> plt.Figure:
    plus_color = get_plot_color("dynamic")
    minus_color = get_plot_color("static_frozen")
    guide_color = get_plot_color("other_residual")

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    ax.plot([-1.1, 1.1], [-1.1, 1.1], color=guide_color, linewidth=1.4, linestyle="-", zorder=1)
    ax.axhline(0.0, color=guide_color, linewidth=0.9, linestyle=":", zorder=1)
    ax.axvline(0.0, color=guide_color, linewidth=0.9, linestyle=":", zorder=1)

    _draw_group(ax, table, prefix="plus", color=plus_color, marker="o")
    _draw_group(ax, table, prefix="minus", color=minus_color, marker="^")

    ax.annotate(
        "",
        xy=(0.86, -0.055),
        xytext=(0.14, -0.055),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": COLOR_NEUTRAL},
        annotation_clip=False,
    )
    ax.text(
        0.5,
        -0.095,
        "more dynamic-like firing pattern",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        color=COLOR_NEUTRAL,
    )
    ax.annotate(
        "",
        xy=(-0.055, 0.86),
        xytext=(-0.055, 0.14),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": COLOR_NEUTRAL},
        annotation_clip=False,
    )
    ax.text(
        -0.095,
        0.5,
        "more dynamic-like decision",
        transform=ax.transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=12,
        color=COLOR_NEUTRAL,
    )

    handles = [
        Line2D([0], [0], marker="o", color=plus_color, markerfacecolor=plus_color, linestyle="-", linewidth=2.2, markersize=7, label="plus: static -> dynamic"),
        Line2D([0], [0], marker="^", color=minus_color, markerfacecolor=minus_color, linestyle="-", linewidth=2.2, markersize=7, label="minus: dynamic -> static"),
        Line2D([0], [0], marker="o", color=COLOR_NEUTRAL, markerfacecolor="white", linestyle="None", markersize=6, label="before manipulation"),
        Line2D([0], [0], marker="o", color=COLOR_NEUTRAL, markerfacecolor=COLOR_NEUTRAL, linestyle="None", markersize=6, label="after manipulation"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=9)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=GRID_ALPHA_SOFT)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.22, top=0.98)
    return fig


def plot_bundle(input_dir: Path):
    df = read_bundle_csv(input_dir, "pair_results.csv", REQUIRED_RESULT_COLUMNS)
    vectors = load_bundle_npz(input_dir, "pair_vectors.npz")
    table = _build_trajectory_table(df, vectors)
    return {FIGURE_STEM: _plot_d_main(table)}


if __name__ == "__main__":
    raise SystemExit(main_for("l3_accumulator_mechanism_experiment", plot_bundle, title="L3 Accumulator Mechanism"))
