from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

from src.plotting.common.colors import NATURE_COMPATIBLE_PALETTE, get_plot_cmap, get_plot_color
from src.plotting.paper_fig.typography import (
    VECTOR_TEXT_RCPARAMS,
    apply_paper_figure_typography,
    mark_panel_label,
)
from src.plotting.paper_fig.utils import add_axes_mm


PALETTE = NATURE_COMPATIBLE_PALETTE
INK = PALETTE["ink"]
NEUTRAL_DARK = PALETTE["neutral_dark"]
NEUTRAL_MID = PALETTE["neutral_mid"]
NEUTRAL_LIGHT = PALETTE["neutral_light"]
NEUTRAL_PALE = PALETTE["neutral_pale"]
NAVY = PALETTE["primary_navy"]
CYAN = PALETTE["primary_cyan"]
PALE_BLUE = PALETTE["primary_pale"]
TEAL = PALETTE["mechanism_teal"]
MINT = PALETTE["mechanism_mint"]
CORAL = PALETTE["comparison_coral"]
PURPLE = PALETTE["fused_slate"]
WHITE = PALETTE["white"]


class BundleReader:
    """Allowlisted reader for frozen tables inside one supplementary bundle."""

    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = bundle_dir.resolve()
        if not self.bundle_dir.is_dir():
            raise FileNotFoundError(f"Supplementary bundle does not exist: {self.bundle_dir}")
        self.accesses: list[dict[str, Any]] = []

    def _resolve_internal(self, relative: str, purpose: str) -> Path:
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise ValueError(f"Absolute bundle path is forbidden: {relative}")
        path = (self.bundle_dir / relative_path).resolve()
        try:
            path.relative_to(self.bundle_dir)
        except ValueError as exc:
            raise PermissionError(f"Plot source escapes supplementary bundle: {path}") from exc
        if path.suffix.lower() not in {".csv", ".json"}:
            raise PermissionError(f"Unsupported supplementary plot source type: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Required supplementary plot source is missing: {path}")
        self.accesses.append(
            {
                "path": path.relative_to(self.bundle_dir).as_posix(),
                "purpose": purpose,
                "sha256": sha256_file(path),
            }
        )
        return path

    def read_csv(self, relative: str, purpose: str) -> pd.DataFrame:
        return pd.read_csv(self._resolve_internal(relative, purpose))

    def read_json(self, relative: str, purpose: str) -> dict[str, Any]:
        path = self._resolve_internal(relative, purpose)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected a JSON object in {path}")
        return payload


def figure_from_spec(spec: Mapping[str, Any], *, dpi: int = 300) -> Figure:
    width_mm, height_mm = (float(value) for value in spec["canvas_mm"])
    with mpl.rc_context(VECTOR_TEXT_RCPARAMS):
        fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi, facecolor=WHITE)
    return fig


def add_plot_axis(fig: Figure, spec: Mapping[str, Any], panel_id: str) -> Axes:
    width_mm, height_mm = (float(value) for value in spec["canvas_mm"])
    bbox = spec["panels"][panel_id]["plot_bbox_mm"]
    axis = add_axes_mm(fig, *bbox, canvas_h_mm=height_mm, canvas_w_mm=width_mm)
    axis.set_label(f"panel_{panel_id}_plot")
    return axis


def add_bbox_axis(fig: Figure, spec: Mapping[str, Any], bbox: Sequence[float]) -> Axes:
    width_mm, height_mm = (float(value) for value in spec["canvas_mm"])
    return add_axes_mm(fig, *bbox, canvas_h_mm=height_mm, canvas_w_mm=width_mm)


def add_panel_labels(fig: Figure, spec: Mapping[str, Any]) -> None:
    width_mm, height_mm = (float(value) for value in spec["canvas_mm"])
    for panel_id, panel in spec["panels"].items():
        left, top, _, _ = (float(value) for value in panel["slot_bbox_mm"])
        text = fig.text(
            (left + 0.3) / width_mm,
            1.0 - (top + 0.6) / height_mm,
            panel_id,
            ha="left",
            va="top",
            color=INK,
            clip_on=False,
        )
        mark_panel_label(text)


def style_axis(axis: Axes) -> None:
    axis.set_facecolor(WHITE)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_color(INK)
        axis.spines[side].set_linewidth(0.65)
    axis.tick_params(axis="both", which="major", length=2.5, width=0.6, color=INK, pad=1.5)
    axis.tick_params(axis="both", which="minor", length=0)


def apply_axis_spec(axis: Axes, panel: Mapping[str, Any]) -> None:
    axis.set_xlabel(str(panel.get("xlabel", "")))
    axis.set_ylabel(str(panel.get("ylabel", "")))
    if "xlim" in panel:
        axis.set_xlim(*[float(value) for value in panel["xlim"]])
    if "ylim" in panel:
        axis.set_ylim(*[float(value) for value in panel["ylim"]])
    if "xticks" in panel:
        axis.set_xticks([float(value) for value in panel["xticks"]])
    if "yticks" in panel:
        axis.set_yticks([float(value) for value in panel["yticks"]])


def draw_reference(axis: Axes, value: float, *, orientation: str = "horizontal", linestyle: str = "--") -> None:
    if orientation == "horizontal":
        axis.axhline(float(value), color=NEUTRAL_MID, linewidth=0.6, linestyle=linestyle, zorder=0)
    else:
        axis.axvline(float(value), color=NEUTRAL_MID, linewidth=0.6, linestyle=linestyle, zorder=0)


def deterministic_jitter(seeds: Iterable[int], *, width: float = 0.14, salt: int = 0) -> np.ndarray:
    offsets = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed) * 104729 + int(salt))
        offsets.append(float(rng.uniform(-width, width)))
    return np.asarray(offsets, dtype=float)


def statistic_row(
    statistics: pd.DataFrame,
    figure_id: str,
    panel_id: str,
    *,
    role: str | None = None,
    **filters: Any,
) -> pd.Series:
    mask = statistics["figure_id"].astype(str).eq(str(figure_id)) & statistics["panel_id"].astype(str).eq(str(panel_id))
    if role is not None:
        mask &= statistics["role"].astype(str).eq(str(role))
    for column, expected in filters.items():
        if column not in statistics.columns:
            raise KeyError(f"Statistics table has no column {column!r}")
        series = statistics[column]
        if isinstance(expected, (int, float, np.integer, np.floating)):
            numeric = pd.to_numeric(series, errors="coerce")
            mask &= np.isclose(numeric, float(expected), equal_nan=False)
        else:
            mask &= series.astype(str).eq(str(expected))
    rows = statistics.loc[mask]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one statistics row for {figure_id}{panel_id} role={role} filters={filters}, found {len(rows)}"
        )
    return rows.iloc[0]


def vertical_mean_ci(
    axis: Axes,
    x: float,
    row: pd.Series,
    *,
    color: str,
    marker: str = "o",
    markerfacecolor: str | None = None,
    markeredgecolor: str | None = None,
    zorder: int = 5,
) -> None:
    mean = float(row["mean"])
    low = float(row["ci95_low"])
    high = float(row["ci95_high"])
    axis.errorbar(
        [x],
        [mean],
        yerr=[[mean - low], [high - mean]],
        fmt=marker,
        markersize=4.5,
        color=color,
        markerfacecolor=markerfacecolor or color,
        markeredgecolor=markeredgecolor or color,
        markeredgewidth=0.8,
        elinewidth=0.85,
        capsize=2.2,
        capthick=0.85,
        zorder=zorder,
    )


def horizontal_mean_ci(
    axis: Axes,
    y: float,
    row: pd.Series,
    *,
    color: str,
    marker: str = "D",
    markerfacecolor: str | None = None,
    markeredgecolor: str | None = None,
    zorder: int = 5,
) -> None:
    mean = float(row["mean"])
    low = float(row["ci95_low"])
    high = float(row["ci95_high"])
    axis.errorbar(
        [mean],
        [y],
        xerr=[[mean - low], [high - mean]],
        fmt=marker,
        markersize=4.5,
        color=color,
        markerfacecolor=markerfacecolor or color,
        markeredgecolor=markeredgecolor or color,
        markeredgewidth=0.8,
        elinewidth=0.85,
        capsize=2.2,
        capthick=0.85,
        zorder=zorder,
    )


def add_top_colorbar(
    fig: Figure,
    spec: Mapping[str, Any],
    panel: Mapping[str, Any],
    image: Any,
    *,
    ticks: Sequence[float] | None = None,
) -> Axes:
    cax = add_bbox_axis(fig, spec, panel["colorbar_bbox_mm"])
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=ticks)
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_label(str(panel["colorbar_label"]), labelpad=1.0)
    colorbar.ax.tick_params(length=2.0, width=0.55, pad=1.0)
    colorbar.outline.set_edgecolor(INK)
    colorbar.outline.set_linewidth(0.5)
    return cax


def draw_matrix(
    axis: Axes,
    matrix: np.ndarray,
    *,
    cmap_role: str,
    vmin: float,
    vmax: float,
    xlabels: Sequence[Any],
    ylabels: Sequence[Any],
    annotate_decimals: int | None = None,
    unavailable_color: str = WHITE,
) -> Any:
    if str(cmap_role) == "residual":
        cmap = mpl.colors.LinearSegmentedColormap.from_list("supp_v5_residual", [WHITE, PURPLE])
    else:
        cmap = get_plot_cmap(cmap_role).copy()
    cmap.set_bad(unavailable_color)
    image = axis.imshow(
        np.ma.masked_invalid(np.asarray(matrix, dtype=float)),
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )
    axis.set_xticks(np.arange(len(xlabels)))
    axis.set_xticklabels([str(value) for value in xlabels])
    axis.set_yticks(np.arange(len(ylabels)))
    axis.set_yticklabels([str(value) for value in ylabels])
    style_axis(axis)
    if annotate_decimals is not None:
        threshold = float(vmin) + 0.58 * (float(vmax) - float(vmin))
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = float(matrix[row, column])
                if not np.isfinite(value):
                    continue
                axis.text(
                    column,
                    row,
                    f"{value:.{int(annotate_decimals)}f}",
                    ha="center",
                    va="center",
                    color=WHITE if value >= threshold else INK,
                )
    return image


def color_for_role(role: str) -> str:
    return get_plot_color(role, default=NEUTRAL_DARK)


def save_figure(fig: Figure, figure_dir: Path, stem: str, *, dpi: int = 300) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_figure_typography(fig)
    outputs: dict[str, str] = {}
    for extension in ("png", "pdf", "svg"):
        path = figure_dir / f"{stem}.{extension}"
        rcparams = dict(VECTOR_TEXT_RCPARAMS)
        metadata: dict[str, object] | None = None
        if extension == "pdf":
            metadata = {
                "CreationDate": None,
                "ModDate": None,
                "Creator": "Net_torch supplementary_v5 plot-only renderer",
                "Producer": "Matplotlib",
            }
        elif extension == "svg":
            rcparams["svg.hashsalt"] = "net_torch_supplementary_v5_v1"
            metadata = {
                "Date": None,
                "Creator": "Net_torch supplementary_v5 plot-only renderer",
            }
        with mpl.rc_context(rcparams):
            fig.savefig(path, dpi=dpi, facecolor=WHITE, metadata=metadata)
        outputs[extension] = str(path)
    grayscale_dir = figure_dir / "grayscale"
    grayscale_dir.mkdir(parents=True, exist_ok=True)
    grayscale_path = grayscale_dir / f"{stem}_grayscale.png"
    with Image.open(outputs["png"]) as image:
        image.convert("L").save(grayscale_path)
    outputs["grayscale"] = str(grayscale_path)
    return outputs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "BundleReader",
    "CORAL",
    "CYAN",
    "INK",
    "MINT",
    "NAVY",
    "NEUTRAL_DARK",
    "NEUTRAL_LIGHT",
    "NEUTRAL_MID",
    "NEUTRAL_PALE",
    "PALE_BLUE",
    "PURPLE",
    "TEAL",
    "WHITE",
    "add_bbox_axis",
    "add_panel_labels",
    "add_plot_axis",
    "add_top_colorbar",
    "apply_axis_spec",
    "color_for_role",
    "deterministic_jitter",
    "draw_matrix",
    "draw_reference",
    "figure_from_spec",
    "horizontal_mean_ci",
    "save_figure",
    "sha256_file",
    "statistic_row",
    "style_axis",
    "vertical_mean_ci",
    "write_json",
]
