from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyBboxPatch, Rectangle, Wedge
from PIL import Image
from pypdf import PdfReader, PdfWriter
from scipy import stats

from src.plotting.common.colors import (
    NATURE_COMPATIBLE_PALETTE,
    get_plot_cmap,
    get_plot_color,
    get_plot_distinction,
)
from src.plotting.paper_fig.layout_contract import validate_layout_contract
from src.plotting.paper_fig.typography import (
    VECTOR_TEXT_RCPARAMS,
    apply_paper_figure_typography,
    mark_panel_label,
    mark_relative_text_size,
)

from .specs import CANVAS_MM, get_figure_spec


RENDERER_VERSION = "final_six_csv_plotter_v1.12.0"
EXPECTED_SEEDS = tuple(range(1000, 1020))
ALLOWED_EXTERNAL_ASSETS = {
    "fig1": "results/paper_figures/outputs/structure-enhanced.svg",
    "fig2": "results/paper_figures/outputs/DMS-enhanced.svg",
}
MM_TO_INCH = 1.0 / 25.4
MM_TO_POINT = 72.0 / 25.4
INK = NATURE_COMPATIBLE_PALETTE["ink"]
NEUTRAL = NATURE_COMPATIBLE_PALETTE["neutral_mid"]
PALE = NATURE_COMPATIBLE_PALETTE["neutral_light"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass
class BundleReader:
    figure_id: str
    figure_dir: Path
    accesses: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.figure_dir = self.figure_dir.resolve()
        if self.figure_dir.name != self.figure_id:
            raise ValueError(
                f"{self.figure_id}: input dir must be its final bundle directory; "
                f"got {self.figure_dir}"
            )
        expected_root = (
            _repo_root() / "results" / "paper_figure_multi_seed" / "final_six_figures"
        ).resolve()
        if self.figure_dir.parent != expected_root:
            raise ValueError(
                f"{self.figure_id}: plotting only accepts the final-six bundle; "
                f"expected parent {expected_root}"
            )

    def _resolve_internal(self, relative: str, purpose: str) -> Path:
        if Path(relative).is_absolute():
            raise ValueError(f"absolute internal path is forbidden: {relative}")
        path = (self.figure_dir / relative).resolve()
        allowed = _inside(path, self.figure_dir)
        self.accesses.append(
            {
                "figure_id": self.figure_id,
                "path": str(path),
                "purpose": purpose,
                "external": False,
                "allowed": allowed,
                "sha256": _sha256(path) if allowed and path.is_file() else "",
            }
        )
        if not allowed:
            raise PermissionError(f"plot source escapes final bundle: {path}")
        if path.suffix.lower() not in {".csv", ".json"}:
            raise PermissionError(f"unsupported plot source type: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"required plot source is missing: {path}")
        return path

    def read_csv(self, relative: str, purpose: str) -> pd.DataFrame:
        path = self._resolve_internal(relative, purpose)
        return pd.read_csv(path)

    def read_json(self, relative: str, purpose: str) -> dict[str, Any]:
        path = self._resolve_internal(relative, purpose)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def read_registered_svg(self, manifest: pd.DataFrame) -> tuple[Path, bytes]:
        if self.figure_id not in ALLOWED_EXTERNAL_ASSETS:
            raise PermissionError(f"{self.figure_id}: no external SVG is allowed")
        if len(manifest) != 1:
            raise ValueError(f"{self.figure_id}: asset manifest must have one row")
        row = manifest.iloc[0]
        recorded = Path(str(row["asset_path"])).as_posix()
        expected = ALLOWED_EXTERNAL_ASSETS[self.figure_id]
        if recorded != expected:
            raise PermissionError(
                f"{self.figure_id}: unregistered SVG path {recorded!r}; expected {expected!r}"
            )
        path = (_repo_root() / recorded).resolve()
        expected_path = (_repo_root() / expected).resolve()
        if path != expected_path or path.suffix.lower() != ".svg":
            raise PermissionError(f"{self.figure_id}: external asset allowlist rejected {path}")
        if not path.is_file():
            raise FileNotFoundError(f"registered SVG is missing: {path}")
        actual_hash = _sha256(path)
        recorded_hash = str(row["asset_sha256"])
        allowed = actual_hash == recorded_hash
        self.accesses.append(
            {
                "figure_id": self.figure_id,
                "path": str(path),
                "purpose": "registered schematic SVG",
                "external": True,
                "allowed": allowed,
                "sha256": actual_hash,
            }
        )
        if not allowed:
            raise ValueError(
                f"{self.figure_id}: registered SVG hash mismatch "
                f"(expected {recorded_hash}, observed {actual_hash})"
            )
        return path, path.read_bytes()

    def write_access_log(self) -> None:
        path = self.figure_dir / "meta" / "plot_source_access.csv"
        frame = pd.DataFrame(self.accesses)
        frame.to_csv(path, index=False)


def _apply_filters(frame: pd.DataFrame, filters: Mapping[str, Any] | None) -> pd.DataFrame:
    out = frame.copy()
    for column, expected in (filters or {}).items():
        if column not in out:
            raise ValueError(f"plot filter column is missing: {column}")
        if isinstance(expected, list):
            out = out.loc[out[column].isin(expected)]
        else:
            out = out.loc[out[column].eq(expected)]
    if out.empty:
        raise ValueError(f"plot filters produced an empty table: {filters}")
    return out.copy()


def _validate_panel_data(
    frame: pd.DataFrame,
    *,
    figure_id: str,
    panel_id: str,
    schematic: bool = False,
) -> None:
    if frame.empty:
        raise ValueError(f"{figure_id}{panel_id}: empty plot source")
    if schematic:
        return
    required = {
        "figure_id",
        "panel_id",
        "network_seed",
        "record_type",
        "endpoint",
        "condition",
        "value",
        "unit",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{figure_id}{panel_id}: missing plot columns {sorted(missing)}")
    observed = set(pd.to_numeric(frame["network_seed"], errors="raise").astype(int))
    if observed != set(EXPECTED_SEEDS):
        raise ValueError(
            f"{figure_id}{panel_id}: seed set mismatch "
            f"(missing={sorted(set(EXPECTED_SEEDS) - observed)}, "
            f"extra={sorted(observed - set(EXPECTED_SEEDS))})"
        )
    values = pd.to_numeric(frame["value"], errors="coerce")
    if values.isna().all():
        raise ValueError(f"{figure_id}{panel_id}: all plotted values are missing")


def _ci95(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan, math.nan
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    sem = float(stats.sem(array))
    half = float(stats.t.ppf(0.975, len(array) - 1) * sem)
    return mean, mean - half, mean + half


def _summary_with_ci(
    rows: pd.DataFrame,
    spec: Mapping[str, Any],
) -> tuple[float, float, float]:
    values = pd.to_numeric(rows["value"], errors="coerce").dropna()
    if values.empty:
        raise ValueError("cannot summarize an empty plotted group")
    if not bool(spec.get("use_persisted_ci", False)):
        return _ci95(values)
    required = ("summary_mean", "summary_ci95_low", "summary_ci95_high")
    missing = [column for column in required if column not in rows]
    if missing:
        raise ValueError(f"persisted-CI plot rows are missing {missing}")
    summary: list[float] = []
    for column in required:
        unique = pd.to_numeric(rows[column], errors="raise").dropna().unique()
        if len(unique) != 1:
            raise ValueError(
                f"persisted-CI group requires one {column}, observed={unique.tolist()}"
            )
        summary.append(float(unique[0]))
    raw_mean = float(values.mean())
    if not np.isclose(raw_mean, summary[0], rtol=0.0, atol=1e-12):
        raise ValueError(
            "persisted-CI estimate disagrees with the plotted network mean: "
            f"raw={raw_mean}, persisted={summary[0]}"
        )
    if not summary[1] <= summary[0] <= summary[2]:
        raise ValueError(f"invalid persisted confidence interval: {summary}")
    return summary[0], summary[1], summary[2]


def _seed_jitter(seeds: Sequence[Any], scale: float = 0.12) -> np.ndarray:
    numeric = np.asarray([int(value) for value in seeds], dtype=float)
    centered = ((numeric - EXPECTED_SEEDS[0]) % len(EXPECTED_SEEDS)) - 9.5
    return centered / 9.5 * scale


def _tick_text(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _color(spec: Mapping[str, Any], key: Any) -> str:
    role = (spec.get("colors") or {}).get(str(key), key)
    return get_plot_color(role, context="final_six")


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(INK)
    axis.spines["bottom"].set_color(INK)
    axis.tick_params(axis="both", which="major", colors=INK, width=0.8, length=3)
    axis.minorticks_off()


def _legend(axis: plt.Axes, *, ncol: int | None = None) -> None:
    handles, labels = axis.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    if not unique:
        return
    axis.legend(
        list(unique.values()),
        list(unique),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol or min(3, len(unique)),
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.8,
        borderaxespad=0.1,
    )


def _references(axis: plt.Axes, spec: Mapping[str, Any], *, orientation: str) -> None:
    for reference in spec.get("references", []):
        value = float(reference["value"])
        label = str(reference.get("label") or "")
        if orientation == "vertical":
            axis.axvline(value, color=NEUTRAL, lw=0.8, ls="--", zorder=0)
            if label:
                axis.text(
                    value,
                    0.98,
                    label,
                    transform=axis.get_xaxis_transform(),
                    ha="right",
                    va="top",
                    color=NEUTRAL,
                )
        else:
            axis.axhline(value, color=NEUTRAL, lw=0.8, ls="--", zorder=0)
            if label:
                axis.text(
                    0.99,
                    value,
                    label,
                    transform=axis.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    color=NEUTRAL,
                )


def _plot_forest(axis: plt.Axes, frame: pd.DataFrame, spec: Mapping[str, Any]) -> None:
    category_field = str(spec["category_field"])
    order = list(spec["category_order"])
    labels = spec.get("category_labels") or {}
    for index, category in enumerate(order):
        rows = frame.loc[frame[category_field].eq(category)].copy()
        values = pd.to_numeric(rows["value"], errors="coerce")
        valid = values.notna()
        rows = rows.loc[valid]
        values = values.loc[valid]
        color = _color(spec, category)
        y = len(order) - 1 - index
        if bool(spec.get("show_raw_points", True)):
            jitter = _seed_jitter(rows["network_seed"].tolist(), scale=0.17)
            axis.scatter(
                values,
                y + jitter,
                s=9,
                facecolor=color,
                edgecolor="white",
                linewidth=0.25,
                alpha=0.48,
                zorder=2,
            )
        mean, low, high = _summary_with_ci(rows, spec)
        axis.errorbar(
            mean,
            y,
            xerr=np.array([[mean - low], [high - mean]]),
            fmt="D",
            color=color,
            markeredgecolor=INK,
            markeredgewidth=0.45,
            markersize=4.2,
            elinewidth=1.5,
            capsize=2.5,
            zorder=4,
        )
        if category in (spec.get("category_references") or {}):
            reference = float(spec["category_references"][category])
            axis.plot(
                [reference, reference],
                [y - 0.28, y + 0.28],
                color=NEUTRAL,
                ls="--",
                lw=0.9,
                zorder=1,
            )
    if bool(spec.get("show_category_axis", True)):
        axis.set_yticks(np.arange(len(order)))
        axis.set_yticklabels([labels.get(item, item) for item in reversed(order)])
    else:
        axis.set_yticks([])
        axis.tick_params(axis="y", left=False)
        axis.spines["left"].set_visible(False)
    axis.set_ylim(-0.65, len(order) - 0.35)
    axis.set_xlabel(str(spec.get("x_label") or ""))
    if spec.get("x_limits"):
        axis.set_xlim(*[float(value) for value in spec["x_limits"]])
    else:
        finite = pd.to_numeric(frame["value"], errors="coerce")
        finite = finite[np.isfinite(finite)]
        if len(finite) and float(finite.max() - finite.min()) < 1e-3:
            axis.ticklabel_format(
                axis="x", style="sci", scilimits=(0, 0), useMathText=True
            )
    if spec.get("x_ticks"):
        axis.set_xticks([float(value) for value in spec["x_ticks"]])
    _references(axis, spec, orientation="vertical")
    _style_axis(axis)


def _plot_grouped_bars(
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    x_field = str(spec["x_field"])
    x_order = list(spec["x_order"])
    x_labels = spec.get("x_labels") or {}
    hue_field = str(spec["hue_field"])
    hue_order = list(spec["hue_order"])
    hue_labels = spec.get("hue_labels") or {}
    bar_width = float(spec.get("bar_width", 0.32))
    offsets = (
        np.arange(len(hue_order), dtype=float) - (len(hue_order) - 1) / 2.0
    ) * bar_width
    summaries: dict[tuple[Any, Any], tuple[float, float, float]] = {}
    for hue_index, hue in enumerate(hue_order):
        hue_rows = frame.loc[frame[hue_field].eq(hue)].copy()
        color = _color(spec, hue)
        for x_index, category in enumerate(x_order):
            rows = hue_rows.loc[hue_rows[x_field].eq(category)]
            values = pd.to_numeric(rows["value"], errors="coerce").dropna()
            if values.empty:
                raise ValueError(
                    f"grouped bars missing values for {hue_field}={hue}, "
                    f"{x_field}={category}"
                )
            mean, low, high = _summary_with_ci(rows, spec)
            summaries[(hue, category)] = (mean, low, high)
            x = x_index + offsets[hue_index]
            axis.bar(
                x,
                mean,
                width=bar_width * 0.88,
                color=color,
                edgecolor=INK,
                linewidth=0.45,
                label=hue_labels.get(hue, str(hue)) if x_index == 0 else "",
                zorder=2,
            )
            axis.errorbar(
                x,
                mean,
                yerr=np.array([[mean - low], [high - mean]]),
                fmt="none",
                ecolor=INK,
                elinewidth=0.9,
                capsize=2.2,
                capthick=0.9,
                zorder=4,
            )
    for annotation in spec.get("contrast_annotations", []):
        hue = annotation["hue"]
        start = annotation["from"]
        end = annotation["to"]
        hue_index = hue_order.index(hue)
        end_index = x_order.index(end)
        start_mean = summaries[(hue, start)][0]
        end_mean, _, end_high = summaries[(hue, end)]
        difference = end_mean - start_mean
        decimals = int(annotation.get("decimals", 1))
        value_text = f"{difference:+.{decimals}f}".replace("-", "−")
        text = (
            f"{annotation.get('prefix', '')}"
            f"{value_text}"
            f"{annotation.get('suffix', '')}"
        )
        axis.annotate(
            text,
            xy=(end_index + offsets[hue_index], end_high),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=_color(spec, hue),
            clip_on=False,
            zorder=5,
        )
    axis.set_xticks(np.arange(len(x_order)))
    axis.set_xticklabels([x_labels.get(item, item) for item in x_order])
    axis.set_xlim(-0.55, len(x_order) - 0.45)
    axis.set_ylabel(str(spec.get("y_label") or ""))
    if spec.get("y_labelpad") is not None:
        axis.yaxis.labelpad = float(spec["y_labelpad"])
    if spec.get("y_limits"):
        axis.set_ylim(*[float(value) for value in spec["y_limits"]])
    if spec.get("y_ticks"):
        axis.set_yticks([float(value) for value in spec["y_ticks"]])
    _style_axis(axis)
    if spec.get("legend_owner") == "panel":
        _legend(axis, ncol=len(hue_order))


def _plot_bullet_gauges(
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    category_field = str(spec["category_field"])
    order = list(spec["category_order"])
    labels = spec.get("category_labels") or {}
    references = spec.get("category_references") or {}
    decimals = int(spec.get("annotation_decimals", 3))
    y_limits = [float(value) for value in spec.get("y_limits", [0.0, 1.0])]
    for x_index, category in enumerate(order):
        rows = frame.loc[frame[category_field].eq(category)]
        values = pd.to_numeric(rows["value"], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"bullet gauge missing category {category}")
        mean, low, high = _ci95(values)
        color = _color(spec, category)
        axis.plot(
            [x_index, x_index],
            y_limits,
            color=PALE,
            lw=5.0,
            solid_capstyle="round",
            zorder=1,
        )
        axis.plot(
            [x_index, x_index],
            [y_limits[0], mean],
            color=color,
            lw=5.0,
            solid_capstyle="round",
            zorder=2,
        )
        axis.errorbar(
            x_index,
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            fmt="D",
            color=color,
            markeredgecolor=INK,
            markeredgewidth=0.45,
            markersize=4.2,
            elinewidth=1.1,
            capsize=2.2,
            zorder=4,
        )
        if category in references:
            reference = float(references[category])
            axis.plot(
                [x_index - 0.17, x_index + 0.17],
                [reference, reference],
                color=NEUTRAL,
                lw=1.0,
                ls="--",
                solid_capstyle="butt",
                zorder=3,
            )
        axis.annotate(
            f"{mean:.{decimals}f}",
            xy=(x_index, mean),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=color,
            clip_on=False,
            zorder=5,
        )
    axis.set_xticks(np.arange(len(order)))
    axis.set_xticklabels([labels.get(item, item) for item in order])
    axis.set_xlim(-0.55, len(order) - 0.45)
    axis.set_ylabel(str(spec.get("y_label") or ""))
    axis.set_ylim(*y_limits)
    if spec.get("y_ticks"):
        axis.set_yticks([float(value) for value in spec["y_ticks"]])
    _style_axis(axis)


def _plot_joint_endpoint_plane(
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    x_endpoint = str(spec["x_endpoint"])
    y_endpoint = str(spec["y_endpoint"])
    subset = frame.loc[frame["endpoint"].isin([x_endpoint, y_endpoint])].copy()
    if subset.duplicated(["network_seed", "endpoint"]).any():
        raise ValueError("joint endpoint plane requires one value per seed and endpoint")
    subset["network_seed"] = pd.to_numeric(
        subset["network_seed"], errors="raise"
    ).astype(int)
    subset["value"] = pd.to_numeric(subset["value"], errors="raise")
    paired = (
        subset.pivot(index="network_seed", columns="endpoint", values="value")
        .reindex(index=EXPECTED_SEEDS, columns=[x_endpoint, y_endpoint])
    )
    if paired.isna().any(axis=None):
        missing = paired.loc[paired.isna().any(axis=1)].index.tolist()
        raise ValueError(
            "joint endpoint plane requires paired endpoints for every network; "
            f"incomplete seeds={missing}"
        )

    x_values = paired[x_endpoint].to_numpy(dtype=float)
    y_values = paired[y_endpoint].to_numpy(dtype=float)
    network_color = _color(spec, "network")
    mean_color = _color(spec, "mean")
    axis.axvline(
        float(spec["x_threshold"]),
        color=NEUTRAL,
        lw=0.75,
        ls=(0, (3.0, 2.4)),
        zorder=0,
    )
    axis.axhline(
        float(spec["y_threshold"]),
        color=NEUTRAL,
        lw=0.75,
        ls=(0, (3.0, 2.4)),
        zorder=0,
    )
    axis.scatter(
        x_values,
        y_values,
        s=8,
        facecolor="white",
        edgecolor=network_color,
        linewidth=0.55,
        alpha=0.78,
        zorder=4,
    )
    x_mean, x_low, x_high = _ci95(x_values)
    y_mean, y_low, y_high = _ci95(y_values)
    axis.errorbar(
        x_mean,
        y_mean,
        xerr=np.asarray([[x_mean - x_low], [x_high - x_mean]]),
        yerr=np.asarray([[y_mean - y_low], [y_high - y_mean]]),
        fmt="D",
        markersize=6.2,
        markerfacecolor="none",
        markeredgecolor=mean_color,
        markeredgewidth=1.0,
        ecolor=mean_color,
        elinewidth=1.05,
        capsize=2.1,
        capthick=0.8,
        zorder=2,
    )
    axis.set_xlabel(str(spec.get("x_label") or ""))
    axis.set_ylabel(str(spec.get("y_label") or ""))
    axis.set_xlim(*[float(value) for value in spec["x_limits"]])
    axis.set_ylim(*[float(value) for value in spec["y_limits"]])
    axis.set_xticks([float(value) for value in spec["x_ticks"]])
    axis.set_yticks([float(value) for value in spec["y_ticks"]])
    _style_axis(axis)


def _plot_threshold_margin_bars(
    axis: plt.Axes,
    frame: pd.DataFrame,
    statistics_frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    endpoint_order = [str(value) for value in spec["endpoint_order"]]
    endpoint_labels = spec.get("endpoint_labels") or {}
    subset = frame.loc[frame["endpoint"].isin(endpoint_order)].copy()
    if subset.duplicated(["network_seed", "endpoint"]).any():
        raise ValueError(
            "threshold-margin bars require one value per seed and endpoint"
        )
    subset["network_seed"] = pd.to_numeric(
        subset["network_seed"], errors="raise"
    ).astype(int)
    subset["value"] = pd.to_numeric(subset["value"], errors="raise")
    paired = (
        subset.pivot(index="network_seed", columns="endpoint", values="value")
        .reindex(index=EXPECTED_SEEDS, columns=endpoint_order)
    )
    if paired.isna().any(axis=None):
        missing = paired.loc[paired.isna().any(axis=1)].index.tolist()
        raise ValueError(
            "threshold-margin bars require every endpoint for every network; "
            f"incomplete seeds={missing}"
        )

    stats_rows = statistics_frame.loc[
        statistics_frame["endpoint"].isin(endpoint_order)
    ].copy()
    if stats_rows.duplicated(["endpoint"]).any():
        duplicates = sorted(
            stats_rows.loc[stats_rows.duplicated(["endpoint"], keep=False), "endpoint"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "threshold-margin bars require one statistics row per endpoint; "
            f"duplicates={duplicates}"
        )
    stats_rows = stats_rows.set_index("endpoint").reindex(endpoint_order)
    if stats_rows.isna().all(axis=1).any():
        missing = stats_rows.index[stats_rows.isna().all(axis=1)].tolist()
        raise ValueError(
            "threshold-margin bars are missing endpoint statistics: "
            f"{missing}"
        )
    thresholds = pd.to_numeric(stats_rows["null_value"], errors="raise")
    if not np.isfinite(thresholds.to_numpy(dtype=float)).all():
        raise ValueError("threshold-margin bars require finite predeclared thresholds")

    axis.axhline(
        0.0,
        color=NEUTRAL,
        lw=0.8,
        ls=(0, (3.0, 2.4)),
        zorder=0,
    )
    bar_width = float(spec.get("bar_width", 0.48))
    for x_index, endpoint in enumerate(endpoint_order):
        threshold = float(thresholds.loc[endpoint])
        endpoint_rows = subset.loc[subset["endpoint"].eq(endpoint)].copy()
        raw_mean, raw_low, raw_high = _summary_with_ci(endpoint_rows, spec)
        mean = raw_mean - threshold
        low = raw_low - threshold
        high = raw_high - threshold
        expected_mean = float(
            pd.to_numeric(stats_rows.loc[endpoint, "estimate"], errors="raise")
        ) - threshold
        expected_low = float(
            pd.to_numeric(stats_rows.loc[endpoint, "ci95_low"], errors="raise")
        ) - threshold
        expected_high = float(
            pd.to_numeric(stats_rows.loc[endpoint, "ci95_high"], errors="raise")
        ) - threshold
        if not np.allclose(
            [mean, low, high],
            [expected_mean, expected_low, expected_high],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "threshold-margin summary disagrees with persisted statistics for "
                f"{endpoint}"
            )
        color = _color(spec, endpoint)
        axis.bar(
            x_index,
            mean,
            width=bar_width,
            color=color,
            edgecolor=INK,
            linewidth=0.45,
            zorder=2,
        )
        axis.errorbar(
            x_index,
            mean,
            yerr=np.asarray([[mean - low], [high - mean]]),
            fmt="none",
            ecolor=INK,
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.9,
            zorder=4,
        )
    axis.set_xticks(np.arange(len(endpoint_order), dtype=float))
    axis.set_xticklabels(
        [endpoint_labels.get(endpoint, endpoint) for endpoint in endpoint_order]
    )
    axis.set_xlim(-0.55, len(endpoint_order) - 0.45)
    axis.set_ylabel(str(spec.get("y_label") or ""))
    if spec.get("y_labelpad") is not None:
        axis.yaxis.labelpad = float(spec["y_labelpad"])
    axis.set_ylim(*[float(value) for value in spec["y_limits"]])
    axis.set_yticks([float(value) for value in spec["y_ticks"]])
    _style_axis(axis)


def _plot_seed_paired_dumbbells(
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    seed_field = str(spec.get("seed_field") or "network_seed")
    condition_field = str(spec.get("condition_field") or "condition")
    matched_condition = str(spec["matched_condition"])
    changed_condition = str(spec["changed_condition"])
    subset = frame.loc[
        frame[condition_field].isin([matched_condition, changed_condition])
    ].copy()
    if subset.duplicated([seed_field, condition_field]).any():
        raise ValueError(
            "paired dumbbells require one value per seed and condition"
        )
    subset[seed_field] = pd.to_numeric(subset[seed_field], errors="raise").astype(int)
    subset["value"] = pd.to_numeric(subset["value"], errors="raise")
    paired = (
        subset.pivot(index=seed_field, columns=condition_field, values="value")
        .reindex(
            index=EXPECTED_SEEDS,
            columns=[matched_condition, changed_condition],
        )
    )
    if paired.isna().any(axis=None):
        missing = paired.loc[paired.isna().any(axis=1)].index.tolist()
        raise ValueError(
            "paired dumbbells require matched and changed values for every network; "
            f"incomplete seeds={missing}"
        )

    seeds = paired.index.to_numpy(dtype=float)
    matched = paired[matched_condition].to_numpy(dtype=float)
    changed = paired[changed_condition].to_numpy(dtype=float)
    matched_color = _color(spec, matched_condition)
    changed_color = _color(spec, changed_condition)
    axis.vlines(
        seeds,
        matched,
        changed,
        color=NEUTRAL,
        lw=0.55,
        alpha=0.62,
        zorder=1,
    )
    axis.scatter(
        seeds,
        matched,
        s=12,
        facecolor="white",
        edgecolor=matched_color,
        linewidth=0.75,
        zorder=2,
    )
    axis.scatter(
        seeds,
        changed,
        s=13,
        facecolor=changed_color,
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
    )
    x_label_position = float(spec["x_limits"][1]) - 0.02
    axis.text(
        x_label_position,
        float(spec["y_limits"][1]) - 0.001,
        "Changed events",
        ha="right",
        va="bottom",
        color=changed_color,
        clip_on=False,
        zorder=5,
    )
    axis.text(
        x_label_position,
        float(spec["y_limits"][0]) + 0.001,
        "Matched random",
        ha="right",
        va="top",
        color=matched_color,
        clip_on=False,
        zorder=5,
    )
    axis.set_xlabel(str(spec.get("x_label") or ""))
    axis.set_ylabel(str(spec.get("y_label") or ""))
    axis.yaxis.labelpad = float(spec.get("y_labelpad", 4.0))
    axis.set_xlim(*[float(value) for value in spec["x_limits"]])
    axis.set_ylim(*[float(value) for value in spec["y_limits"]])
    axis.set_xticks([float(value) for value in spec["x_ticks"]])
    axis.set_xticklabels([_tick_text(value) for value in spec["x_ticks"]])
    axis.set_yticks([float(value) for value in spec["y_ticks"]])
    _style_axis(axis)


def _plot_state_space_glyph(
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    common_endpoint = str(spec["common_endpoint"])
    history_endpoint = str(spec["history_endpoint"])
    common_rows = frame.loc[frame["endpoint"].eq(common_endpoint)]
    history_rows = frame.loc[frame["endpoint"].eq(history_endpoint)]
    if len(common_rows) != len(EXPECTED_SEEDS):
        raise ValueError(
            "state-space glyph requires one common-direction value per network; "
            f"observed={len(common_rows)}"
        )
    if len(history_rows) != len(EXPECTED_SEEDS):
        raise ValueError(
            "state-space glyph requires one history-imprint value per network; "
            f"observed={len(history_rows)}"
        )
    common = float(pd.to_numeric(common_rows["value"], errors="raise").mean())
    history = float(pd.to_numeric(history_rows["value"], errors="raise").mean())
    common_threshold = float(spec["common_threshold"])
    history_threshold = float(spec["history_threshold"])
    if not (-1.0 <= common <= 1.0):
        raise ValueError(f"common cosine lies outside [-1, 1]: {common}")
    if not (-1.0 <= common_threshold <= 1.0):
        raise ValueError(
            f"common cosine threshold lies outside [-1, 1]: {common_threshold}"
        )
    if history < 0.0:
        raise ValueError(f"history-imprint ratio must be nonnegative: {history}")

    layout = spec.get("glyph_layout") or {}
    x_limits = [float(value) for value in layout.get("x_limits", [0.0, 100.0])]
    y_limits = [float(value) for value in layout.get("y_limits", [0.0, 46.0])]
    origin = np.asarray(layout.get("origin", [10.0, 28.0]), dtype=float)
    vector_length = float(layout.get("vector_length", 42.0))
    threshold_radius = float(layout.get("threshold_radius", 30.0))
    observed_arc_radius = float(layout.get("observed_arc_radius", 18.0))
    ribbon_start = np.asarray(layout.get("ribbon_start", [60.0, 10.0]), dtype=float)
    ribbon_scale = float(layout.get("ribbon_scale", 31.0))
    ribbon_height = float(layout.get("ribbon_height", 2.2))
    common_color = _color(spec, common_endpoint)
    history_color = _color(spec, history_endpoint)

    observed_angle = math.degrees(math.acos(np.clip(common, -1.0, 1.0)))
    threshold_angle = math.degrees(
        math.acos(np.clip(common_threshold, -1.0, 1.0))
    )
    observed_half = observed_angle / 2.0
    threshold_half = threshold_angle / 2.0

    threshold_fan = Wedge(
        tuple(origin),
        threshold_radius,
        -threshold_half,
        threshold_half,
        facecolor=PALE,
        edgecolor="none",
        alpha=0.34,
        zorder=1,
    )
    axis.add_patch(threshold_fan)
    for angle in (-threshold_half, threshold_half):
        radians = math.radians(angle)
        endpoint = origin + threshold_radius * np.array(
            [math.cos(radians), math.sin(radians)]
        )
        axis.plot(
            [origin[0], endpoint[0]],
            [origin[1], endpoint[1]],
            color=NEUTRAL,
            lw=0.65,
            ls=(0, (2.0, 2.0)),
            alpha=0.72,
            zorder=2,
        )

    axis.annotate(
        "",
        xy=(origin[0] + vector_length + 2.0, origin[1]),
        xytext=tuple(origin),
        arrowprops={
            "arrowstyle": "-|>",
            "color": common_color,
            "lw": 2.2,
            "alpha": 0.22,
            "mutation_scale": 8,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=2,
    )
    observed_endpoints: dict[str, np.ndarray] = {}
    for label, angle in (("A", observed_half), ("C", -observed_half)):
        radians = math.radians(angle)
        endpoint = origin + vector_length * np.array(
            [math.cos(radians), math.sin(radians)]
        )
        observed_endpoints[label] = endpoint
        axis.annotate(
            "",
            xy=tuple(endpoint),
            xytext=tuple(origin),
            arrowprops={
                "arrowstyle": "-|>",
                "color": common_color,
                "lw": 1.25,
                "mutation_scale": 8,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=4,
        )
        label_offset = 1.8 if label == "A" else -1.8
        axis.text(
            endpoint[0] + 1.3,
            endpoint[1] + label_offset,
            label,
            ha="left",
            va="center",
            color=common_color,
            zorder=5,
        )
    axis.add_patch(
        Arc(
            tuple(origin),
            2.0 * observed_arc_radius,
            2.0 * observed_arc_radius,
            theta1=-observed_half,
            theta2=observed_half,
            color=common_color,
            lw=1.15,
            zorder=5,
        )
    )
    decimals = int(spec.get("annotation_decimals", 3))
    axis.text(
        origin[0] + observed_arc_radius + 2.0,
        origin[1] + 5.6,
        f"{common:.{decimals}f}",
        ha="center",
        va="bottom",
        color=common_color,
        zorder=6,
    )

    ribbon_end = ribbon_start[0] + min(history, 1.0) * ribbon_scale
    axis.plot(
        [ribbon_start[0], ribbon_start[0] + ribbon_scale],
        [ribbon_start[1], ribbon_start[1]],
        color=PALE,
        lw=0.9,
        solid_capstyle="round",
        zorder=1,
    )
    axis.plot(
        [ribbon_start[0], ribbon_end],
        [ribbon_start[1], ribbon_start[1]],
        color=history_color,
        lw=max(1.0, ribbon_height * 1.8),
        solid_capstyle="round",
        zorder=4,
    )
    threshold_x = ribbon_start[0] + min(history_threshold, 1.0) * ribbon_scale
    axis.plot(
        [threshold_x, threshold_x],
        [
            ribbon_start[1] - 1.8 * ribbon_height,
            ribbon_start[1] + 1.8 * ribbon_height,
        ],
        color=NEUTRAL,
        lw=0.8,
        zorder=5,
    )
    axis.annotate(
        "",
        xy=(ribbon_start[0] - 1.0, ribbon_start[1] + 0.4),
        xytext=(
            observed_endpoints["C"][0] - 1.0,
            observed_endpoints["C"][1] - 0.5,
        ),
        arrowprops={
            "arrowstyle": "-",
            "connectionstyle": "arc3,rad=-0.24",
            "color": history_color,
            "lw": 0.8,
            "alpha": 0.72,
        },
        zorder=3,
    )
    axis.text(
        ribbon_end,
        ribbon_start[1] + 4.1,
        f"{history:.{decimals}f}",
        ha="center",
        va="bottom",
        color=history_color,
        zorder=6,
    )
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def _plot_paired_slope(
    axis: plt.Axes,
    frame: pd.DataFrame,
    statistics_frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    x_field = str(spec["x_field"])
    x_order = list(spec["x_order"])
    x_labels = spec.get("x_labels") or {}
    pivot = frame.pivot(
        index="network_seed",
        columns=x_field,
        values="value",
    ).reindex(columns=x_order)
    if pivot.isna().any(axis=None):
        missing = pivot.loc[pivot.isna().any(axis=1)].index.tolist()
        raise ValueError(f"paired slope has incomplete network pairs: {missing}")
    pivot = pivot.sort_index()
    x_positions = np.arange(len(x_order), dtype=float)
    seed_offsets = np.linspace(-0.075, 0.075, len(pivot), dtype=float)
    for offset, (_, row) in zip(seed_offsets, pivot.iterrows()):
        values = row.to_numpy(dtype=float)
        axis.plot(
            x_positions + offset,
            values,
            color=PALE,
            lw=0.55,
            alpha=0.62,
            zorder=1,
        )
    summaries: dict[Any, tuple[float, float, float]] = {}
    for x_index, condition in enumerate(x_order):
        values = pivot[condition].to_numpy(dtype=float)
        mean, low, high = _summary_with_ci(rows, spec)
        summaries[condition] = (mean, low, high)
        color = _color(spec, condition)
        axis.scatter(
            np.full(len(values), x_index, dtype=float) + seed_offsets,
            values,
            s=8,
            facecolor=color,
            edgecolor="white",
            linewidth=0.25,
            alpha=0.5,
            zorder=2,
        )
        axis.errorbar(
            x_index,
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            fmt="D",
            color=color,
            markeredgecolor=INK,
            markeredgewidth=0.45,
            markersize=4.3,
            elinewidth=1.2,
            capsize=2.4,
            zorder=5,
        )
    axis.plot(
        x_positions,
        [summaries[condition][0] for condition in x_order],
        color=INK,
        lw=1.35,
        zorder=4,
    )
    annotation = spec.get("summary_annotation")
    if annotation:
        endpoint = str(annotation["statistics_endpoint"])
        rows = statistics_frame.loc[statistics_frame["endpoint"].eq(endpoint)]
        if len(rows) != 1:
            raise ValueError(
                f"paired slope expected one statistics row for {endpoint}, "
                f"observed={len(rows)}"
            )
        value = float(rows.iloc[0]["estimate"])
        decimals = int(annotation.get("decimals", 2))
        y_low, y_high = [
            float(value) for value in spec.get("y_limits", axis.get_ylim())
        ]
        span = y_high - y_low
        bracket_y = y_high - 0.055 * span
        cap = 0.025 * span
        axis.plot(
            [x_positions[0], x_positions[-1]],
            [bracket_y, bracket_y],
            color=INK,
            lw=0.8,
            zorder=4,
        )
        axis.plot(
            [x_positions[0], x_positions[0]],
            [bracket_y - cap, bracket_y],
            color=INK,
            lw=0.8,
            zorder=4,
        )
        axis.plot(
            [x_positions[-1], x_positions[-1]],
            [bracket_y - cap, bracket_y],
            color=INK,
            lw=0.8,
            zorder=4,
        )
        axis.annotate(
            f"{annotation.get('label', '')} {value:.{decimals}f}".strip(),
            xy=(float(x_positions.mean()), bracket_y),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=_color(spec, x_order[-1]),
            clip_on=False,
            zorder=5,
        )
    axis.set_xticks(x_positions)
    axis.set_xticklabels([x_labels.get(item, item) for item in x_order])
    axis.set_xlim(-0.35, len(x_order) - 0.65)
    axis.set_ylabel(str(spec.get("y_label") or ""))
    if spec.get("y_limits"):
        axis.set_ylim(*[float(value) for value in spec["y_limits"]])
    if spec.get("y_ticks"):
        axis.set_yticks([float(value) for value in spec["y_ticks"]])
    _style_axis(axis)


def _plot_boxplot(
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    x_field = str(spec["x_field"])
    x_order = list(spec["x_order"])
    x_labels = spec.get("x_labels") or {}
    whisker_iqr = float(spec.get("whisker_iqr", 1.5))
    show_fliers = bool(spec.get("show_fliers", False))
    for x_index, condition in enumerate(x_order):
        rows = frame.loc[frame[x_field].eq(condition)]
        values = pd.to_numeric(rows["value"], errors="coerce").dropna()
        if len(values) != len(EXPECTED_SEEDS):
            raise ValueError(
                f"boxplot requires one {condition} value per network; "
                f"observed={len(values)}"
            )
        color = _color(spec, condition)
        artists = axis.boxplot(
            [values.to_numpy(dtype=float)],
            positions=[float(x_index)],
            widths=0.46,
            whis=whisker_iqr,
            showfliers=show_fliers,
            patch_artist=True,
            manage_ticks=False,
            boxprops={
                "facecolor": color,
                "edgecolor": INK,
                "linewidth": 0.8,
            },
            whiskerprops={
                "color": color,
                "linewidth": 1.0,
            },
            capprops={
                "color": color,
                "linewidth": 1.0,
            },
            medianprops={
                "color": INK,
                "linewidth": 1.15,
            },
        )
        for box in artists["boxes"]:
            box.set_alpha(0.88)
    axis.set_xticks(np.arange(len(x_order), dtype=float))
    axis.set_xticklabels([x_labels.get(item, item) for item in x_order])
    axis.set_xlim(-0.55, len(x_order) - 0.45)
    axis.set_ylabel(str(spec.get("y_label") or ""))
    if spec.get("y_labelpad") is not None:
        axis.yaxis.labelpad = float(spec["y_labelpad"])
    if spec.get("y_limits"):
        axis.set_ylim(*[float(value) for value in spec["y_limits"]])
    if spec.get("y_ticks"):
        axis.set_yticks([float(value) for value in spec["y_ticks"]])
    _references(axis, spec, orientation="horizontal")
    _style_axis(axis)


def _plot_ordered_bars(
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    category_field = str(spec["category_field"])
    order = list(spec["category_order"])
    labels = spec.get("category_labels") or {}
    decimals = int(spec.get("annotation_decimals", 3))
    bar_width = float(spec.get("bar_width", 0.48))
    annotate_values = bool(spec.get("annotate_values", True))
    for x_index, category in enumerate(order):
        rows = frame.loc[frame[category_field].eq(category)]
        values = pd.to_numeric(rows["value"], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"ordered bars missing category {category}")
        mean, low, high = _summary_with_ci(rows, spec)
        color = _color(spec, category)
        axis.bar(
            x_index,
            mean,
            width=bar_width,
            color=color,
            edgecolor=INK,
            linewidth=0.45,
            zorder=2,
        )
        axis.errorbar(
            x_index,
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            fmt="none",
            ecolor=INK,
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.9,
            zorder=4,
        )
        if annotate_values:
            axis.annotate(
                f"{mean:.{decimals}f}",
                xy=(x_index, high),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=color,
                clip_on=False,
                zorder=5,
            )
    if spec.get("direction_arrow") and len(order) == 2:
        axis.annotate(
            "",
            xy=(0.65, 0.06),
            xytext=(0.35, 0.06),
            arrowprops={
                "arrowstyle": "-|>",
                "color": NEUTRAL,
                "lw": 0.8,
                "mutation_scale": 7,
            },
            zorder=4,
        )
    axis.set_xticks(np.arange(len(order)))
    axis.set_xticklabels([labels.get(item, item) for item in order])
    axis.set_xlim(-0.55, len(order) - 0.45)
    axis.set_ylabel(str(spec.get("y_label") or ""))
    if spec.get("y_labelpad") is not None:
        axis.yaxis.labelpad = float(spec["y_labelpad"])
    if spec.get("y_limits"):
        axis.set_ylim(*[float(value) for value in spec["y_limits"]])
    if spec.get("y_ticks"):
        axis.set_yticks([float(value) for value in spec["y_ticks"]])
    _references(axis, spec, orientation="horizontal")
    _style_axis(axis)


def _plot_category_points(
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    x_field = str(spec["x_field"])
    x_order = list(spec["x_order"])
    x_labels = spec.get("x_labels") or {}
    hue_field = spec.get("hue_field")
    hue_order = list(spec.get("hue_order") or ["single"])
    hue_labels = spec.get("hue_labels") or {}
    width = 0.50 if len(hue_order) > 1 else 0.0
    offsets = (
        np.linspace(-width / 2.0, width / 2.0, len(hue_order))
        if len(hue_order) > 1
        else np.array([0.0])
    )
    marker_cycle = ("o", "s", "^", "D")
    color_by_x = bool(spec.get("color_by_x", False))
    for hue_index, hue in enumerate(hue_order):
        subset = frame if hue_field is None else frame.loc[frame[str(hue_field)].eq(hue)]
        for x_index, category in enumerate(x_order):
            color_key = category if color_by_x else hue
            color = _color(spec, color_key)
            distinction = get_plot_distinction(
                (spec.get("colors") or {}).get(str(color_key), color_key)
            )
            rows = subset.loc[subset[x_field].eq(category)].copy()
            values = pd.to_numeric(rows["value"], errors="coerce")
            valid = values.notna()
            rows = rows.loc[valid]
            values = values.loc[valid]
            x = x_index + offsets[hue_index]
            jitter = _seed_jitter(rows["network_seed"].tolist(), scale=0.055)
            marker_by_key = spec.get("markers") or {}
            marker = str(
                marker_by_key.get(str(color_key))
                or marker_by_key.get(color_key)
                or spec.get("mean_marker")
                or marker_cycle[hue_index % len(marker_cycle)]
            )
            if bool(spec.get("show_raw_points", True)):
                axis.scatter(
                    x + jitter,
                    values,
                    s=8,
                    marker=marker,
                    facecolor=color if distinction.marker_fill == "filled" else "white",
                    edgecolor=color,
                    linewidth=0.45,
                    alpha=0.38,
                    zorder=2,
                )
            mean, low, high = _summary_with_ci(rows, spec)
            mean_filled = bool(
                spec.get(
                    "mean_marker_filled",
                    distinction.marker_fill == "filled",
                )
            )
            axis.errorbar(
                x,
                mean,
                yerr=np.array([[mean - low], [high - mean]]),
                fmt=marker,
                color=color,
                markerfacecolor=color if mean_filled else "white",
                markeredgecolor=INK,
                markeredgewidth=0.4,
                markersize=4.2,
                elinewidth=1.4,
                capsize=2.2,
                label=hue_labels.get(hue, str(hue)) if x_index == 0 and hue_field else "",
                zorder=4,
            )
    axis.set_xticks(np.arange(len(x_order)))
    axis.set_xticklabels([x_labels.get(item, item) for item in x_order])
    if len(x_order) >= 4:
        plt.setp(axis.get_xticklabels(), rotation=18, ha="right", rotation_mode="anchor")
    axis.set_xlim(-0.6, len(x_order) - 0.4)
    axis.set_xlabel(str(spec.get("x_label") or ""))
    axis.set_ylabel(
        str(spec.get("y_label") or ""),
        labelpad=float(spec.get("y_labelpad", plt.rcParams["axes.labelpad"])),
    )
    if spec.get("y_limits"):
        axis.set_ylim(*[float(value) for value in spec["y_limits"]])
    if spec.get("y_ticks"):
        axis.set_yticks([float(value) for value in spec["y_ticks"]])
    _references(axis, spec, orientation="horizontal")
    _style_axis(axis)
    if hue_field and spec.get("legend_owner") == "panel":
        _legend(axis, ncol=int(spec.get("legend_ncol") or min(3, len(hue_order))))


def _plot_ordered_lines(
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    frame = frame.copy()
    frame["value"] = (
        pd.to_numeric(frame["value"], errors="coerce")
        * float(spec.get("value_scale", 1.0))
    )
    x_field = str(spec["x_field"])
    x_order = list(spec["x_order"])
    hue_field = spec.get("hue_field")
    hue_order = list(spec.get("hue_order") or ["single"])
    hue_labels = spec.get("hue_labels") or {}
    numeric_x = bool(spec.get("numeric_x"))
    x_positions = (
        np.asarray(x_order, dtype=float)
        if numeric_x
        else np.arange(len(x_order), dtype=float)
    )
    marker_cycle = ("o", "s", "^", "D")
    linestyle_cycle = ("-", "--", "-.", ":")
    direct_label_payload: list[tuple[Any, float, float, str]] = []
    for hue_index, hue in enumerate(hue_order):
        subset = frame if hue_field is None else frame.loc[frame[str(hue_field)].eq(hue)]
        color = _color(spec, hue)
        distinction = get_plot_distinction((spec.get("colors") or {}).get(str(hue), hue))
        configured_linestyles = spec.get("linestyles") or {}
        linestyle = str(
            configured_linestyles.get(str(hue))
            or configured_linestyles.get(hue)
            or (
                distinction.linestyle
                if distinction.linestyle != "-" or len(hue_order) == 1
                else linestyle_cycle[hue_index % len(linestyle_cycle)]
            )
        )
        if bool(spec.get("show_individual_traces", True)):
            for _, network_rows in subset.groupby("network_seed", sort=True):
                series = (
                    network_rows.set_index(x_field)["value"]
                    .reindex(x_order)
                    .pipe(pd.to_numeric, errors="coerce")
                )
                axis.plot(
                    x_positions,
                    series.to_numpy(dtype=float),
                    color=color,
                    alpha=float(spec.get("individual_trace_alpha", 0.08)),
                    lw=float(spec.get("individual_trace_width", 0.45)),
                    ls=linestyle,
                    zorder=1,
                )
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for category in x_order:
            values = pd.to_numeric(
                subset.loc[subset[x_field].eq(category), "value"], errors="coerce"
            )
            rows = subset.loc[subset[x_field].eq(category)].copy()
            mean, low, high = _summary_with_ci(rows, spec)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        configured_markers = spec.get("markers") or {}
        marker = None
        if bool(spec.get("show_markers", True)):
            marker = str(
                configured_markers.get(str(hue))
                or configured_markers.get(hue)
                or marker_cycle[hue_index % len(marker_cycle)]
            )
        axis.plot(
            x_positions,
            means,
            color=color,
            lw=float(spec.get("line_width", 1.5)),
            ls=linestyle,
            marker=marker,
            markersize=float(spec.get("marker_size", 3.4)),
            markerfacecolor=color if distinction.marker_fill == "filled" else "white",
            markeredgecolor=color,
            label=hue_labels.get(hue, str(hue)) if hue_field else "",
            zorder=4,
        )
        if bool(spec.get("show_ci_band", True)):
            axis.fill_between(
                x_positions,
                lows,
                highs,
                color=color,
                alpha=float(spec.get("ci_alpha", 0.14)),
                linewidth=0,
                zorder=2,
            )
        direct_label_payload.append(
            (
                hue,
                float(x_positions[-1]),
                float(means[-1]),
                str(hue_labels.get(hue, str(hue))),
            )
        )
    if spec.get("identity_reference"):
        axis.plot(
            x_positions,
            np.asarray(x_order, dtype=float),
            color=NEUTRAL,
            lw=0.9,
            ls="--",
            label="N_eff = K",
            zorder=0,
        )
        axis.text(
            x_positions[-1],
            float(x_order[-1]),
            str(spec.get("identity_reference_label") or ""),
            color=NEUTRAL,
            ha="right",
            va="bottom",
        )
    if numeric_x:
        ticks = list(spec.get("x_ticks") or x_order)
        axis.set_xticks([float(item) for item in ticks])
        axis.set_xticklabels([_tick_text(item) for item in ticks])
    else:
        axis.set_xticks(x_positions)
        axis.set_xticklabels([_tick_text(item) for item in x_order])
    if x_field == "phase":
        axis.set_xticklabels(x_order)
        plt.setp(axis.get_xticklabels(), rotation=18, ha="right", rotation_mode="anchor")
    axis.set_xlabel(str(spec.get("x_label") or ""))
    axis.set_ylabel(
        str(spec.get("y_label") or ""),
        labelpad=float(spec.get("y_labelpad", plt.rcParams["axes.labelpad"])),
    )
    if spec.get("y_limits"):
        axis.set_ylim(*[float(value) for value in spec["y_limits"]])
    if spec.get("y_ticks"):
        axis.set_yticks([float(value) for value in spec["y_ticks"]])
    if spec.get("x_limits"):
        axis.set_xlim(*[float(value) for value in spec["x_limits"]])
    if spec.get("reference_x") is not None:
        axis.axvline(
            float(spec["reference_x"]),
            color=str(spec.get("reference_x_color") or NEUTRAL),
            lw=float(spec.get("reference_x_width", 0.8)),
            ls=str(spec.get("reference_x_style") or "--"),
            zorder=0,
        )
    if spec.get("reference_y") is not None:
        axis.axhline(
            float(spec["reference_y"]),
            color=str(spec.get("reference_y_color") or PALE),
            lw=float(spec.get("reference_y_width", 0.8)),
            ls=str(spec.get("reference_y_style") or "-"),
            zorder=0,
        )
    _references(axis, spec, orientation="horizontal")
    _style_axis(axis)
    if bool(spec.get("direct_labels", False)):
        offsets = spec.get("direct_label_offsets_pt") or {}
        horizontal_alignment = str(
            spec.get("direct_label_horizontal_alignment") or "left"
        )
        for hue, x_value, y_value, label in direct_label_payload:
            raw_offset = offsets.get(str(hue), offsets.get(hue, [4.0, 0.0]))
            x_offset, y_offset = [float(value) for value in raw_offset]
            axis.annotate(
                label,
                xy=(x_value, y_value),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=horizontal_alignment,
                va="bottom",
                color=_color(spec, hue),
                clip_on=False,
                zorder=6,
            )
    if hue_field and spec.get("legend_owner") == "panel":
        _legend(axis, ncol=int(spec.get("legend_ncol") or min(3, len(hue_order))))


def _plot_seed_trajectory(
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    x_field = str(spec.get("x_field") or "network_seed")
    ordered = frame.copy()
    ordered[x_field] = pd.to_numeric(ordered[x_field], errors="raise")
    ordered["value"] = pd.to_numeric(ordered["value"], errors="raise")
    ordered = ordered.sort_values(x_field, kind="mergesort")
    band = spec.get("emphasis_band")
    if band:
        axis.axhspan(
            float(band["lower"]),
            float(band["upper"]),
            facecolor=_color(spec, band.get("color", "sample_window")),
            edgecolor="none",
            zorder=0,
        )
    for boundary in spec.get("band_boundaries", []):
        axis.axhline(
            float(boundary),
            color=get_plot_color("layer1", context="final_six"),
            lw=0.8,
            ls="--",
            zorder=1,
        )
    color = _color(spec, str(ordered["endpoint"].iloc[0]))
    axis.plot(
        ordered[x_field].to_numpy(dtype=float),
        ordered["value"].to_numpy(dtype=float),
        color=color,
        lw=1.25,
        marker="o",
        markersize=3.6,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.35,
        zorder=3,
    )
    axis.set_xlabel(str(spec.get("x_label") or ""))
    axis.set_ylabel(
        str(spec.get("y_label") or ""),
        labelpad=float(spec.get("y_labelpad", plt.rcParams["axes.labelpad"])),
    )
    if spec.get("x_limits"):
        axis.set_xlim(*[float(value) for value in spec["x_limits"]])
    if spec.get("y_limits"):
        axis.set_ylim(*[float(value) for value in spec["y_limits"]])
    ticks = list(spec.get("x_ticks") or ordered[x_field].tolist())
    axis.set_xticks([float(value) for value in ticks])
    axis.set_xticklabels([_tick_text(value) for value in ticks])
    _style_axis(axis)


def _single_numeric_value(frame: pd.DataFrame, field: str) -> float:
    values = pd.to_numeric(frame[field], errors="raise").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"expected one {field} value, observed={values.tolist()}")
    return float(values[0])


def _plot_time_binned_lines(
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    x_field = str(spec["x_field"])
    hue_field = str(spec["hue_field"])
    hue_order = list(spec["hue_order"])
    hue_labels = spec.get("hue_labels") or {}
    start = _single_numeric_value(frame, str(spec["stimulus_start_field"]))
    end = _single_numeric_value(frame, str(spec["stimulus_end_field"]))
    axis.axvspan(
        start,
        end,
        facecolor=_color(spec, str(spec.get("stimulus_band_color") or "sample_window")),
        edgecolor="none",
        zorder=0,
    )
    for boundary in (start, end):
        axis.axvline(
            boundary,
            color=get_plot_color("layer1", context="final_six"),
            lw=0.8,
            ls="--",
            zorder=1,
        )
    marker_cycle = ("o", "s", "^")
    linestyle_cycle = ("-", "--", "-.")
    for hue_index, hue in enumerate(hue_order):
        subset = frame.loc[frame[hue_field].eq(hue)].copy()
        subset[x_field] = pd.to_numeric(subset[x_field], errors="raise")
        subset["value"] = pd.to_numeric(subset["value"], errors="raise")
        x_values = sorted(subset[x_field].unique().tolist())
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for x_value in x_values:
            values = subset.loc[subset[x_field].eq(x_value), "value"]
            mean, low, high = _ci95(values)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        color = _color(spec, hue)
        distinction = get_plot_distinction(
            (spec.get("colors") or {}).get(str(hue), hue)
        )
        linestyle = (
            distinction.linestyle
            if distinction.linestyle != "-"
            else linestyle_cycle[hue_index % len(linestyle_cycle)]
        )
        axis.fill_between(
            x_values,
            lows,
            highs,
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=2,
        )
        axis.plot(
            x_values,
            means,
            color=color,
            lw=1.45,
            ls=linestyle,
            marker=marker_cycle[hue_index % len(marker_cycle)],
            markersize=3.2,
            markerfacecolor=(
                color if distinction.marker_fill == "filled" else "white"
            ),
            markeredgecolor=color,
            label=hue_labels.get(hue, str(hue)),
            zorder=3,
        )
    axis.set_xlabel(str(spec.get("x_label") or ""))
    axis.set_ylabel(
        str(spec.get("y_label") or ""),
        labelpad=float(spec.get("y_labelpad", plt.rcParams["axes.labelpad"])),
    )
    if spec.get("x_limits"):
        axis.set_xlim(*[float(value) for value in spec["x_limits"]])
    if spec.get("x_ticks"):
        ticks = [float(value) for value in spec["x_ticks"]]
        axis.set_xticks(ticks)
        axis.set_xticklabels([_tick_text(value) for value in ticks])
    if spec.get("y_limits"):
        axis.set_ylim(*[float(value) for value in spec["y_limits"]])
    elif spec.get("y_min") is not None:
        axis.set_ylim(bottom=float(spec["y_min"]))
    if spec.get("y_sci_power") is not None:
        power = int(spec["y_sci_power"])
        axis.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(power, power),
            useMathText=True,
        )
    _style_axis(axis)
    if spec.get("legend_owner") == "panel":
        _legend(axis, ncol=len(hue_order))


def _plot_stacked_composition(
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    condition_field = str(spec["condition_field"])
    condition_order = list(spec["condition_order"])
    condition_labels = spec.get("condition_labels") or {}
    category_field = str(spec["category_field"])
    category_order = list(spec["category_order"])
    category_labels = spec.get("category_labels") or {}
    annotations = set(spec.get("annotate_categories") or [])
    decimals = int(spec.get("annotation_decimals", 1))
    per_network_sums = frame.groupby(
        ["network_seed", condition_field], as_index=False
    )["value"].sum()
    if bool(spec.get("require_sum_100", True)):
        if not np.allclose(
            pd.to_numeric(per_network_sums["value"], errors="raise"),
            100.0,
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("stacked composition rows must sum to 100% per network")
    x_positions = np.arange(len(condition_order), dtype=float)
    bottoms = np.zeros(len(condition_order), dtype=float)
    for category in category_order:
        means = np.asarray(
            [
                pd.to_numeric(
                    frame.loc[
                        frame[condition_field].eq(condition)
                        & frame[category_field].eq(category),
                        "value",
                    ],
                    errors="raise",
                ).mean()
                for condition in condition_order
            ],
            dtype=float,
        )
        color = _color(spec, category)
        axis.bar(
            x_positions,
            means,
            bottom=bottoms,
            width=0.56,
            color=color,
            edgecolor=INK,
            linewidth=0.55,
            label=category_labels.get(category, category),
            zorder=2,
        )
        if category in annotations:
            for x_value, bottom, height in zip(x_positions, bottoms, means):
                axis.text(
                    x_value,
                    bottom + height / 2.0,
                    f"{height:.{decimals}f}%",
                    ha="center",
                    va="center",
                    color="white",
                    zorder=3,
                )
        bottoms += means
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [condition_labels.get(value, value) for value in condition_order]
    )
    axis.set_xlim(-0.55, len(condition_order) - 0.45)
    axis.set_xlabel(str(spec.get("x_label") or ""))
    axis.set_ylabel(
        str(spec.get("y_label") or ""),
        labelpad=float(spec.get("y_labelpad", plt.rcParams["axes.labelpad"])),
    )
    if spec.get("y_limits"):
        axis.set_ylim(*[float(value) for value in spec["y_limits"]])
    if spec.get("y_ticks"):
        axis.set_yticks([float(value) for value in spec["y_ticks"]])
    _style_axis(axis)
    if spec.get("legend_owner") == "panel":
        _legend(axis, ncol=len(category_order))


def _plot_partial_cue_split(
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> list[plt.Axes]:
    if not bool(spec.get("approved_internal_split", False)):
        raise ValueError("partial-cue split requires explicit user approval in the figure spec")
    target_field = str(spec["target_field"])
    target_order = list(spec["target_order"])
    condition_field = str(spec["condition_field"])
    condition_order = list(spec["condition_order"])
    x_field = str(spec["x_field"])
    x_order = [float(value) for value in spec["x_order"]]
    condition_labels = spec.get("condition_labels") or {}
    axis.set_axis_off()
    parent_x, parent_y, parent_width, parent_height = [
        float(value) for value in spec["plot_bbox_mm"]
    ]
    child_bboxes = list(spec.get("child_plot_bboxes_mm") or [])
    if len(child_bboxes) != len(target_order):
        raise ValueError(
            "partial-cue split requires one declared child plot bbox per target"
        )
    inset_bounds: list[list[float]] = []
    for child_bbox in child_bboxes:
        child_x, child_y, child_width, child_height = [
            float(value) for value in child_bbox
        ]
        inset_bounds.append(
            [
                (child_x - parent_x) / parent_width,
                (parent_y + parent_height - child_y - child_height)
                / parent_height,
                child_width / parent_width,
                child_height / parent_height,
            ]
        )
    left = axis.inset_axes(inset_bounds[0])
    right = axis.inset_axes(inset_bounds[1], sharey=left)
    child_axes = [left, right]
    marker_cycle = ("o", "s", "^", "D")
    handles: list[Line2D] = []
    labels: list[str] = []
    for target_index, (target, child) in enumerate(zip(target_order, child_axes)):
        target_rows = frame.loc[frame[target_field].eq(target)].copy()
        for condition_index, condition in enumerate(condition_order):
            subset = target_rows.loc[target_rows[condition_field].eq(condition)].copy()
            if subset.empty:
                raise ValueError(f"partial-cue split missing {target}/{condition}")
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for x_value in x_order:
                rows = subset.loc[
                    np.isclose(
                        pd.to_numeric(subset[x_field], errors="coerce"),
                        x_value,
                        rtol=0.0,
                        atol=1e-12,
                    )
                ]
                mean, low, high = _summary_with_ci(rows, spec)
                means.append(mean)
                lows.append(low)
                highs.append(high)
            color = _color(spec, condition)
            marker = marker_cycle[condition_index % len(marker_cycle)]
            child.fill_between(
                x_order,
                lows,
                highs,
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )
            child.plot(
                x_order,
                means,
                color=color,
                lw=1.2,
                marker=marker,
                markersize=2.8,
                markerfacecolor=color,
                markeredgecolor=color,
                zorder=3,
            )
            if target_index == 0:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        lw=1.2,
                        marker=marker,
                        markersize=3.0,
                    )
                )
                labels.append(str(condition_labels.get(condition, condition)))
        child.set_xlim(0.0, 1.02)
        child.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
        child.set_ylim(*[float(value) for value in spec["y_limits"]])
        child.set_yticks([float(value) for value in spec["y_ticks"]])
        child.set_title(f"Target {target}", pad=2.0)
        child.set_xlabel(str(spec.get("x_label") or ""))
        _style_axis(child)
        if target_index == 0:
            child.set_ylabel(str(spec.get("y_label") or ""))
        else:
            child.tick_params(axis="y", labelleft=False)
            if not bool(spec.get("show_right_y_axis", True)):
                child.tick_params(axis="y", left=False)
                child.spines["left"].set_visible(False)
    legend_anchor = [float(value) for value in spec.get("legend_anchor", [0.5, 1.08])]
    axis.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=tuple(legend_anchor),
        ncol=int(spec.get("legend_ncol") or 2),
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    return child_axes


def _plot_heatmap(
    fig: Figure,
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    x_field = str(spec["x_field"])
    y_field = str(spec["y_field"])
    x_order = list(spec["x_order"])
    y_order = list(spec["y_order"])
    aggregate = str(spec.get("aggregate") or "mean")
    if aggregate == "identity":
        duplicates = frame.duplicated([x_field, y_field], keep=False)
        if duplicates.any():
            raise ValueError("identity heatmap contains duplicate cells")
        pivot = frame.pivot(index=y_field, columns=x_field, values="value")
    else:
        pivot = frame.pivot_table(
            index=y_field,
            columns=x_field,
            values="value",
            aggfunc="mean",
            dropna=False,
        )
    pivot = pivot.reindex(index=y_order, columns=x_order)
    data = pivot.to_numpy(dtype=float)
    cmap = get_plot_cmap(str(spec.get("cmap") or "stsp_support")).copy()
    cmap.set_bad(str(spec.get("unavailable_color") or PALE))
    norm = None
    vmin = spec.get("vmin")
    vmax = spec.get("vmax")
    if "center" in spec:
        center = float(spec["center"])
        finite = data[np.isfinite(data)]
        maximum = max(abs(float(finite.min())), abs(float(finite.max())), 1e-12)
        norm = TwoSlopeNorm(vmin=-maximum, vcenter=center, vmax=maximum)
        vmin = None
        vmax = None
    draw_edges = bool(spec.get("cell_edges", True))
    mesh = axis.pcolormesh(
        np.arange(len(x_order) + 1),
        np.arange(len(y_order) + 1),
        np.ma.masked_invalid(data),
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        edgecolors="white" if draw_edges else "none",
        linewidth=0.25 if draw_edges else 0.0,
        rasterized=False,
    )
    if bool(spec.get("annotate_cells", False)):
        finite = data[np.isfinite(data)]
        if finite.size:
            color_low = float(vmin) if vmin is not None else float(np.nanmin(finite))
            color_high = float(vmax) if vmax is not None else float(np.nanmax(finite))
            threshold = color_low + 0.55 * (color_high - color_low)
            decimals = int(spec.get("annotation_decimals", 2))
            for y_index in range(len(y_order)):
                for x_index in range(len(x_order)):
                    value = data[y_index, x_index]
                    if not np.isfinite(value):
                        continue
                    label = axis.text(
                        x_index + 0.5,
                        y_index + 0.5,
                        f"{value:.{decimals}f}",
                        ha="center",
                        va="center",
                        color="white" if value >= threshold else INK,
                        zorder=3,
                    )
                    mark_relative_text_size(label, 0.78)
    axis.set_xticks(np.arange(len(x_order)) + 0.5)
    axis.set_xticklabels([str(value) for value in x_order])
    y_tick_values = list(spec.get("y_tick_values") or y_order)
    y_tick_positions = [y_order.index(value) + 0.5 for value in y_tick_values]
    axis.set_yticks(y_tick_positions)
    axis.set_yticklabels([str(value) for value in y_tick_values])
    axis.set_xlim(0, len(x_order))
    axis.set_ylim(0, len(y_order))
    axis.invert_yaxis()
    axis.set_xlabel(str(spec.get("x_label") or ""))
    axis.set_ylabel(str(spec.get("y_label") or ""))
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)
    orientation = str(spec.get("colorbar_orientation") or "vertical")
    if orientation == "horizontal_top":
        canvas_height_mm = float(fig.get_figheight()) * 25.4
        bar_height_mm = float(spec.get("colorbar_height_mm", 1.4))
        bar_gap_mm = float(spec.get("colorbar_gap_mm", 3.8))
        width_fraction = float(spec.get("colorbar_width_fraction", 1.0))
        axis_bbox = axis.get_position()
        bar_width = axis_bbox.width * width_fraction
        bar_left = axis_bbox.x0 + (axis_bbox.width - bar_width) / 2.0
        colorbar_axis = fig.add_axes(
            [
                bar_left,
                axis_bbox.y1 + bar_gap_mm / canvas_height_mm,
                bar_width,
                bar_height_mm / canvas_height_mm,
            ]
        )
        colorbar = fig.colorbar(
            mesh,
            cax=colorbar_axis,
            orientation="horizontal",
        )
        tick_position = str(spec.get("colorbar_ticks_position") or "bottom")
        label_position = str(spec.get("colorbar_label_position") or "title")
        if tick_position not in {"top", "bottom"}:
            raise ValueError(f"Unsupported horizontal colorbar tick position: {tick_position}")
        colorbar.ax.xaxis.set_ticks_position(tick_position)
        colorbar.ax.tick_params(axis="x", pad=1.0)
        colorbar_label = str(spec.get("colorbar_label") or "")
        if label_position in {"top", "bottom"}:
            colorbar.ax.xaxis.set_label_position(label_position)
            colorbar.ax.set_xlabel(
                colorbar_label,
                labelpad=float(spec.get("colorbar_label_pad_pt", 1.0)),
            )
        elif label_position == "title":
            colorbar.ax.set_title(colorbar_label, pad=2.0)
        else:
            raise ValueError(
                f"Unsupported horizontal colorbar label position: {label_position}"
            )
    else:
        colorbar = fig.colorbar(mesh, ax=axis, fraction=0.055, pad=0.035)
        colorbar.set_label(str(spec.get("colorbar_label") or ""))
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(width=0.6, length=2.5)


def _subaxes_from_axis(
    fig: Figure,
    axis: plt.Axes,
    widths: Sequence[float],
    *,
    gap: float,
) -> list[plt.Axes]:
    bbox = axis.get_position()
    axis.set_visible(False)
    available = bbox.width - gap * (len(widths) - 1)
    total = float(sum(widths))
    output: list[plt.Axes] = []
    cursor = bbox.x0
    for width in widths:
        physical = available * float(width) / total
        output.append(fig.add_axes([cursor, bbox.y0, physical, bbox.height]))
        cursor += physical + gap
    return output


def _stacked_axes_from_axis(
    fig: Figure,
    axis: plt.Axes,
    count: int,
    *,
    gap: float,
) -> list[plt.Axes]:
    bbox = axis.get_position()
    axis.set_visible(False)
    height = (bbox.height - gap * (count - 1)) / count
    return [
        fig.add_axes([bbox.x0, bbox.y0 + (count - 1 - index) * (height + gap), bbox.width, height])
        for index in range(count)
    ]


def _plot_split_conditions(
    fig: Figure,
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> list[plt.Axes]:
    facets = list(spec["facet_order"])
    axes = _stacked_axes_from_axis(fig, axis, len(facets), gap=0.025)
    for index, (facet, facet_axis) in enumerate(zip(facets, axes)):
        subset = frame.loc[frame[str(spec["facet_field"])].eq(facet)].copy()
        x_order = list(spec["x_order"])
        for _, rows in subset.groupby("network_seed", sort=True):
            series = (
                rows.set_index(str(spec["x_field"]))["value"]
                .reindex(x_order)
                .pipe(pd.to_numeric, errors="coerce")
            )
            facet_axis.plot(
                np.arange(len(x_order)),
                series.to_numpy(dtype=float),
                color=PALE,
                lw=0.55,
                alpha=0.65,
                zorder=1,
            )
        for x_index, condition in enumerate(x_order):
            rows = subset.loc[subset[str(spec["x_field"])].eq(condition)]
            values = pd.to_numeric(rows["value"], errors="coerce")
            color = _color(spec, condition)
            jitter = _seed_jitter(rows["network_seed"].tolist(), scale=0.055)
            facet_axis.scatter(
                x_index + jitter,
                values,
                s=7,
                color=color,
                alpha=0.38,
                edgecolor="none",
                zorder=2,
            )
            mean, low, high = _ci95(values)
            facet_axis.errorbar(
                x_index,
                mean,
                yerr=np.array([[mean - low], [high - mean]]),
                fmt="D",
                color=color,
                markeredgecolor=INK,
                markeredgewidth=0.4,
                markersize=3.7,
                elinewidth=1.2,
                capsize=2,
                zorder=3,
            )
        facet_axis.set_xticks(np.arange(len(x_order)))
        facet_axis.set_xticklabels(
            [(spec.get("x_labels") or {}).get(item, item) for item in x_order]
        )
        if index < len(facets) - 1:
            facet_axis.tick_params(axis="x", labelbottom=False)
        facet_axis.set_xlim(-0.45, len(x_order) - 0.55)
        facet_axis.set_ylim(*[float(value) for value in spec["y_limits"]])
        facet_axis.text(
            0.01,
            0.96,
            str((spec.get("facet_labels") or {}).get(facet, facet)),
            transform=facet_axis.transAxes,
            ha="left",
            va="top",
        )
        if index == 0:
            facet_axis.set_ylabel(str(spec.get("y_label") or ""))
        _style_axis(facet_axis)
    return axes


def _plot_line_with_contrast(
    fig: Figure,
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> list[plt.Axes]:
    line_axis, contrast_axis = _subaxes_from_axis(fig, axis, [0.72, 0.28], gap=0.028)
    line_frame = _apply_filters(frame, spec.get("line_filter"))
    line_spec = dict(spec)
    line_spec["chart"] = "ordered_lines"
    values = sorted(
        pd.to_numeric(line_frame[str(spec["x_field"])], errors="coerce")
        .dropna()
        .unique()
        .tolist()
    )
    line_spec["x_order"] = values
    line_spec["legend_owner"] = spec.get("legend_owner")
    _plot_ordered_lines(line_axis, line_frame, line_spec)
    if spec.get("reference_x") is not None:
        x_value = float(spec["reference_x"])
        if x_value in values:
            reference_position = (
                x_value if spec.get("numeric_x") else values.index(x_value)
            )
            line_axis.axvline(reference_position, color=NEUTRAL, ls=":", lw=0.8)
    contrast = _apply_filters(frame, spec.get("contrast_filter"))
    contrast = contrast.copy()
    contrast["value"] = pd.to_numeric(contrast["value"], errors="coerce") * float(
        spec.get("contrast_display_scale", 1.0)
    )
    contrast_field = str(spec.get("contrast_field") or "endpoint")
    order = list(spec.get("contrast_order") or contrast[contrast_field].drop_duplicates())
    contrast_spec = {
        "category_field": contrast_field,
        "category_order": order,
        "category_labels": spec.get("contrast_labels") or {
            item: str(item).replace("_", " ") for item in order
        },
        "x_label": spec.get("contrast_label") or "",
        "references": [{"value": 0.0}],
        "colors": {item: "fused_state" for item in order},
    }
    _plot_forest(contrast_axis, contrast, contrast_spec)
    contrast_axis.tick_params(axis="y", labelsize=7)
    return [line_axis, contrast_axis]


def _map_two_by_two_cells(frame: pd.DataFrame, spec: Mapping[str, Any]) -> pd.DataFrame:
    if "cell_mapping" not in spec:
        return frame
    field = str(spec["cell_field"])
    mapping = spec["cell_mapping"]
    out = frame.copy()
    out[str(spec["x_field"])] = out[field].map(
        lambda value: mapping.get(value, [None, None])[0]
    )
    out[str(spec["hue_field"])] = out[field].map(
        lambda value: mapping.get(value, [None, None])[1]
    )
    if out[[str(spec["x_field"]), str(spec["hue_field"])]].isna().any().any():
        raise ValueError("2x2 cell mapping is incomplete")
    return out


def _plot_two_by_two(
    fig: Figure,
    axis: plt.Axes,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> list[plt.Axes]:
    show_contrast_panel = bool(spec.get("show_contrast_panel", True))
    if show_contrast_panel:
        cell_axis, contrast_axis = _subaxes_from_axis(
            fig, axis, [0.70, 0.30], gap=0.030
        )
    else:
        cell_axis = axis
        contrast_axis = None
    cells = _apply_filters(frame, spec.get("cell_filter"))
    cells = _map_two_by_two_cells(cells, spec)
    cell_spec = dict(spec)
    cell_spec["legend_owner"] = spec.get("legend_owner")
    _plot_category_points(cell_axis, cells, cell_spec)
    x_field = str(spec["x_field"])
    hue_field = str(spec["hue_field"])
    x_order = list(spec["x_order"])
    hue_order = list(spec["hue_order"])
    for hue in hue_order:
        subset = cells.loc[cells[hue_field].eq(hue)]
        means = [
            pd.to_numeric(
                subset.loc[subset[x_field].eq(x_value), "value"], errors="coerce"
            ).mean()
            for x_value in x_order
        ]
        offsets = np.linspace(-0.25, 0.25, len(hue_order))
        axis_offset = offsets[hue_order.index(hue)]
        cell_axis.plot(
            np.arange(len(x_order)) + axis_offset,
            means,
            color=_color(spec, hue),
            lw=1.0,
            zorder=3,
        )
    if contrast_axis is None:
        return [cell_axis]
    contrast = _apply_filters(frame, spec.get("contrast_filter"))
    contrast_order = contrast["endpoint"].drop_duplicates().tolist()
    contrast_spec = {
        "category_field": "endpoint",
        "category_order": contrast_order,
        "category_labels": {
            item: str(
                spec.get("contrast_tick_label")
                or spec.get("contrast_label")
                or item
            )
            for item in contrast_order
        },
        "x_label": str(spec.get("contrast_label") or ""),
        "references": [{"value": 0.0}],
        "colors": {item: "fused_state" for item in contrast_order},
    }
    _plot_forest(contrast_axis, contrast, contrast_spec)
    contrast_axis.tick_params(axis="y", labelsize=7)
    return [cell_axis, contrast_axis]


def _plot_protocol(
    axis: plt.Axes,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    node_lookup = nodes.set_index("node_id").to_dict("index")
    role_colors = {
        "parent_state": NATURE_COMPATIBLE_PALETTE["fused_tint"],
        "observed_input": NATURE_COMPATIBLE_PALETTE["primary_tint"],
        "observed_state": NATURE_COMPATIBLE_PALETTE["primary_pale"],
        "passive_branch": NATURE_COMPATIBLE_PALETTE["neutral_pale"],
        "passive_state": NATURE_COMPATIBLE_PALETTE["neutral_light"],
        "contrast": NATURE_COMPATIBLE_PALETTE["fused_tint"],
        "repeat_rule": NATURE_COMPATIBLE_PALETTE["mechanism_tint"],
    }
    branch_colors = {
        "observed": get_plot_color("dynamic"),
        "matched_passive": NEUTRAL,
        "comparison": get_plot_color("fused_state"),
        "repeat": NATURE_COMPATIBLE_PALETTE["mechanism_teal"],
    }
    for _, edge in edges.iterrows():
        source = node_lookup[str(edge["source_node"])]
        target = node_lookup[str(edge["target_node"])]
        color = branch_colors.get(str(edge["branch"]), NEUTRAL)
        axis.annotate(
            "",
            xy=(float(target["x_mm"]), float(target["y_mm"])),
            xytext=(float(source["x_mm"]), float(source["y_mm"])),
            arrowprops={"arrowstyle": "-|>", "lw": 1.25, "color": color, "shrinkA": 13, "shrinkB": 13},
            zorder=1,
        )
    for _, node in nodes.iterrows():
        x = float(node["x_mm"])
        y = float(node["y_mm"])
        role = str(node["role"])
        width_by_role = {
            "parent_state": 26.0,
            "observed_input": 24.0,
            "observed_state": 24.0,
            "passive_branch": 24.0,
            "passive_state": 24.0,
            "contrast": 27.0,
            "repeat_rule": 20.0,
        }
        width = width_by_role.get(role, 22.0)
        box = FancyBboxPatch(
            (x - width / 2.0, y - 6.0),
            width,
            12.0,
            boxstyle="round,pad=0.45,rounding_size=1.5",
            facecolor=role_colors.get(role, "white"),
            edgecolor=INK,
            linewidth=0.8,
            zorder=2,
        )
        axis.add_patch(box)
        node_label = (spec.get("node_label_overrides") or {}).get(
            str(node["node_id"]), str(node["label"])
        )
        axis.text(x, y + 1.7, node_label, ha="center", va="center", zorder=3)
        math_label = str(node["math_label"])
        math_label = math_label.replace("Delta ", r"\Delta ")
        math_label = math_label.replace("...", r"\ldots")
        math_label = math_label.replace("^obs", "^{obs}")
        math_label = math_label.replace("^passive", "^{passive}")
        axis.text(
            x,
            y - 2.6,
            f"${math_label}$",
            ha="center",
            va="center",
            color=NEUTRAL,
            zorder=3,
        )
    handles = [
        Line2D([0], [0], color=branch_colors["observed"], lw=1.5, label="Observed input"),
        Line2D([0], [0], color=branch_colors["matched_passive"], lw=1.5, label="Equal-time passive"),
        Line2D([0], [0], color=branch_colors["comparison"], lw=1.5, label="Stage contrast"),
    ]
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
    )
    axis.set_xlim(2, 163)
    axis.set_ylim(2, 46)
    axis.axis("off")


def _extract_rightmost_embedded_image(asset_bytes: bytes) -> Image.Image:
    parser = etree.XMLParser(resolve_entities=False)
    root = etree.fromstring(asset_bytes, parser)
    candidates: list[tuple[bool, float, float, str]] = []
    for element in root.xpath(".//*[local-name()='image']"):
        href = (
            element.get("{http://www.w3.org/1999/xlink}href")
            or element.get("href")
            or ""
        )
        if not href.startswith("data:image/") or ";base64," not in href:
            continue
        try:
            x_position = float(element.get("x") or 0.0)
        except ValueError:
            x_position = 0.0
        try:
            width = float(element.get("width") or 0.0)
            height = float(element.get("height") or 0.0)
        except ValueError:
            width = 0.0
            height = 0.0
        aspect = width / height if height > 0 else math.inf
        is_square_stimulus = (
            min(width, height) >= 50.0 and 0.85 <= aspect <= 1.15
        )
        candidates.append(
            (is_square_stimulus, min(width, height), x_position, href)
        )
    if not candidates:
        raise ValueError("registered DMS SVG contains no embedded stimulus image")
    _, _, _, href = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    encoded = href.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        return image.convert("RGBA")


def _plot_fig2_transition_schematic(
    axis: plt.Axes,
    asset_bytes: bytes,
    spec: Mapping[str, Any],
) -> None:
    layer1 = get_plot_color("layer1", context="final_six")
    current_input = get_plot_color("donor_trace", context="final_six")
    probe_image = np.asarray(_extract_rightmost_embedded_image(asset_bytes))
    layout = spec.get("schematic_layout") or {}

    def bbox(name: str, default: Sequence[float]) -> tuple[float, float, float, float]:
        values = layout.get(name, default)
        if len(values) != 4:
            raise ValueError(f"fig2a {name} must contain four coordinates")
        return tuple(float(value) for value in values)

    history_centers = {
        key: np.asarray(value, dtype=float)
        for key, value in (
            layout.get("history_centers")
            or {"A": [42.0, 28.0], "C": [42.0, 12.0]}
        ).items()
    }
    history_size = np.asarray(
        layout.get("history_node_size", [30.0, 8.0]), dtype=float
    )
    b_bbox = bbox("b_bbox", [108.0, 12.0, 16.0, 16.0])

    def rounded_box(
        bounds: Sequence[float],
        *,
        facecolor: str,
        edgecolor: str,
        linewidth: float = 0.75,
        radius: float = 1.2,
        zorder: int = 1,
    ) -> FancyBboxPatch:
        x, y, width, height = [float(value) for value in bounds]
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.08,rounding_size={radius}",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )
        axis.add_patch(patch)
        return patch

    def arrow(
        start: Sequence[float],
        end: Sequence[float],
        *,
        color: str = NEUTRAL,
        linewidth: float = 0.72,
        mutation_scale: float = 6.0,
        zorder: int = 5,
    ) -> None:
        axis.annotate(
            "",
            xy=tuple(float(value) for value in end),
            xytext=tuple(float(value) for value in start),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": linewidth,
                "mutation_scale": mutation_scale,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=zorder,
        )

    for condition in ("A", "C"):
        center = history_centers[condition]
        rounded_box(
            [
                center[0] - history_size[0] / 2.0,
                center[1] - history_size[1] / 2.0,
                history_size[0],
                history_size[1],
            ],
            facecolor="white",
            edgecolor=layer1,
            linewidth=1.05,
            radius=1.0,
            zorder=3,
        )
        axis.text(
            center[0],
            center[1],
            f"History {condition}",
            ha="center",
            va="center",
            color=INK,
            zorder=6,
        )

    b_x, b_y, b_width, b_height = b_bbox
    rounded_box(
        b_bbox,
        facecolor="#111111",
        edgecolor=current_input,
        linewidth=1.15,
        radius=1.0,
        zorder=3,
    )
    axis.imshow(
        probe_image,
        extent=(
            b_x + 0.45,
            b_x + b_width - 0.45,
            b_y + 0.45,
            b_y + b_height - 0.45,
        ),
        origin="upper",
        aspect="auto",
        zorder=4,
    )
    axis.text(
        b_x + b_width / 2.0,
        b_y + b_height + 2.0,
        "B",
        ha="center",
        va="bottom",
        color=current_input,
        zorder=6,
    )

    merge = np.asarray([86.0, 20.0], dtype=float)
    for condition in ("A", "C"):
        center = history_centers[condition]
        start = np.asarray(
            [center[0] + history_size[0] / 2.0, center[1]], dtype=float
        )
        axis.plot(
            [start[0], merge[0]],
            [start[1], merge[1]],
            color=NEUTRAL,
            lw=0.8,
            solid_capstyle="round",
            zorder=4,
        )
    arrow(
        merge,
        (b_x, b_y + b_height / 2.0),
        color=NEUTRAL,
        linewidth=0.85,
        mutation_scale=6.2,
        zorder=6,
    )
    axis.set_xlim(0.0, 152.0)
    axis.set_ylim(0.0, 40.0)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def _plot_bbox_mm(
    slot: Sequence[float],
    chart: str,
    panel_spec: Mapping[str, Any] | None = None,
) -> tuple[float, float, float, float]:
    x, y, width, height = [float(value) for value in slot]
    if panel_spec and panel_spec.get("plot_bbox_mm") is not None:
        explicit = tuple(
            float(value) for value in panel_spec["plot_bbox_mm"]
        )
        if len(explicit) != 4:
            raise ValueError(f"plot_bbox_mm must contain four values: {explicit}")
        px, py, pwidth, pheight = explicit
        if pwidth <= 0 or pheight <= 0:
            raise ValueError(f"non-positive explicit plot area: {explicit}")
        tolerance = 1e-9
        if (
            px < x - tolerance
            or py < y - tolerance
            or px + pwidth > x + width + tolerance
            or py + pheight > y + height + tolerance
        ):
            raise ValueError(
                f"explicit plot area {explicit} escapes slot {tuple(slot)}"
            )
        return explicit
    if chart in {"svg_asset", "protocol"}:
        left, right, top, bottom = 5.0, 4.0, 5.0, 3.0
    elif chart in {"forest", "estimate_strip"}:
        left, right, top, bottom = 27.0, 3.0, 7.0, 9.0
    elif chart == "heatmap":
        left, right, top, bottom = 11.0, 11.0, 7.0, 9.0
    else:
        left, right, top, bottom = 11.0, 3.0, 8.0, 10.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    if plot_width <= 0 or plot_height <= 0:
        raise ValueError(f"non-positive plot area for slot {slot} and chart {chart}")
    return x + left, y + top, plot_width, plot_height


def _as_figure_axes(
    bbox_mm: Sequence[float],
    canvas_mm: Sequence[float],
) -> list[float]:
    x, y, width, height = [float(value) for value in bbox_mm]
    canvas_width, canvas_height = [float(value) for value in canvas_mm]
    return [
        x / canvas_width,
        (canvas_height - y - height) / canvas_height,
        width / canvas_width,
        height / canvas_height,
    ]


def _wireframe(spec: Mapping[str, Any], output: Path) -> None:
    canvas_width, canvas_height = [
        float(value) for value in spec.get("canvas_mm", CANVAS_MM)
    ]
    fig, axis = plt.subplots(
        figsize=(canvas_width * MM_TO_INCH, canvas_height * MM_TO_INCH),
        dpi=160,
    )
    axis.set_xlim(0, canvas_width)
    axis.set_ylim(canvas_height, 0)
    axis.axis("off")
    for panel_id, panel in spec["panels"].items():
        x, y, width, height = spec["slots"][panel_id]
        rectangle = plt.Rectangle(
            (x, y),
            width,
            height,
            facecolor="white",
            edgecolor=INK,
            linewidth=0.8,
        )
        axis.add_patch(rectangle)
        plot = _plot_bbox_mm(
            spec["slots"][panel_id], panel["chart"], panel
        )
        plot_rectangle = plt.Rectangle(
            (plot[0], plot[1]),
            plot[2],
            plot[3],
            facecolor=NATURE_COMPATIBLE_PALETTE["neutral_pale"],
            edgecolor=NEUTRAL,
            linewidth=0.6,
            linestyle="--",
        )
        axis.add_patch(plot_rectangle)
        axis.text(x + 1, y + 4, panel_id, weight="bold", va="top")
        axis.text(
            plot[0] + plot[2] / 2,
            plot[1] + plot[3] / 2,
            panel["chart"],
            ha="center",
            va="center",
            color=NEUTRAL,
        )
    fig.savefig(output, dpi=160, facecolor="white", bbox_inches=None)
    plt.close(fig)


def _asset_injection_bbox(
    slot: Sequence[float],
    *,
    top_padding_mm: float,
) -> tuple[float, float, float, float]:
    x, y, width, height = [float(value) for value in slot]
    return x + 6.0, y + top_padding_mm, width - 11.0, height - top_padding_mm - 2.0


def _inject_svg_asset(
    base_svg: Path,
    final_svg: Path,
    *,
    asset_bytes: bytes,
    asset_viewbox: str,
    slot: Sequence[float],
    embedding_mode: str,
    top_padding_mm: float,
) -> None:
    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
    base_tree = etree.parse(str(base_svg), parser)
    base_root = base_tree.getroot()
    namespace = "http://www.w3.org/2000/svg"
    x_mm, y_mm, width_mm, height_mm = _asset_injection_bbox(
        slot, top_padding_mm=top_padding_mm
    )
    if embedding_mode == "svg_image":
        nested = etree.Element(f"{{{namespace}}}image")
        nested.set("id", "registered-schematic-svg-image")
        nested.set("x", f"{x_mm * MM_TO_POINT:.6f}")
        nested.set("y", f"{y_mm * MM_TO_POINT:.6f}")
        nested.set("width", f"{width_mm * MM_TO_POINT:.6f}")
        nested.set("height", f"{height_mm * MM_TO_POINT:.6f}")
        nested.set("preserveAspectRatio", "xMidYMid meet")
        encoded = base64.b64encode(asset_bytes).decode("ascii")
        nested.set(
            "{http://www.w3.org/1999/xlink}href",
            f"data:image/svg+xml;base64,{encoded}",
        )
        base_root.append(nested)
        base_tree.write(
            str(final_svg),
            encoding="utf-8",
            xml_declaration=True,
            pretty_print=False,
        )
        return
    if embedding_mode != "inline":
        raise ValueError(f"unsupported SVG embedding mode: {embedding_mode}")
    asset_root = etree.fromstring(asset_bytes, parser)
    nested = etree.Element(f"{{{namespace}}}svg")
    nested.set("id", "registered-schematic-asset")
    nested.set("x", f"{x_mm * MM_TO_POINT:.6f}")
    nested.set("y", f"{y_mm * MM_TO_POINT:.6f}")
    nested.set("width", f"{width_mm * MM_TO_POINT:.6f}")
    nested.set("height", f"{height_mm * MM_TO_POINT:.6f}")
    nested.set("viewBox", asset_viewbox)
    nested.set("preserveAspectRatio", "xMidYMid meet")
    style = etree.Element(f"{{{namespace}}}style")
    style.text = "text { font-family: Arial, 'DejaVu Sans', sans-serif; }"
    nested.append(style)
    for child in list(asset_root):
        nested.append(child)
    # Append after Matplotlib's opaque figure background so the registered
    # asset is visible. Its bbox stays inside panel a and does not cover the
    # panel label or any quantitative panel.
    base_root.append(nested)
    base_tree.write(
        str(final_svg),
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=False,
    )


def _find_chrome() -> Path:
    candidates = (
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Chrome/Edge is required for vector SVG-to-PDF export")


def _export_pdf_and_png(
    svg_path: Path,
    pdf_path: Path,
    png_path: Path,
    canvas_mm: Sequence[float],
) -> None:
    canvas_width, canvas_height = [float(value) for value in canvas_mm]
    chrome = _find_chrome()
    # The project-level .codex directory is read-only to normal plot-only
    # processes in the desktop sandbox. The persisted QA directory is the
    # project-defined conversion workspace for this bundle; TemporaryDirectory
    # removes the conversion files before returning.
    temp_parent = svg_path.parent / "qa"
    with tempfile.TemporaryDirectory(
        prefix=f"{svg_path.stem}_plot_", dir=str(temp_parent)
    ) as temp_name:
        temp_dir = Path(temp_name)
        html_path = temp_dir / f"{svg_path.stem}.html"
        user_data = temp_dir / "chrome-profile"
        svg_markup = svg_path.read_text(encoding="utf-8")
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>@page{{size:{canvas_width:g}mm {canvas_height:g}mm;margin:0}}"
            f"html,body{{margin:0;padding:0;width:{canvas_width:g}mm;"
            f"height:{canvas_height:g}mm;overflow:hidden}}"
            f"svg{{display:block;width:{canvas_width:g}mm;"
            f"height:{canvas_height:g}mm}}</style></head><body>"
            f"{svg_markup}</body></html>"
        )
        html_path.write_text(html, encoding="utf-8")
        command = [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-extensions",
            f"--user-data-dir={user_data}",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        if result.returncode != 0 or not pdf_path.is_file():
            raise RuntimeError(
                "SVG-to-PDF export failed: "
                f"exit={result.returncode}; stderr={result.stderr[-2000:]}"
            )
        # Chromium records wall-clock CreationDate/ModDate values in otherwise
        # deterministic PDFs. Rewriting only the container metadata keeps the
        # vector page content intact while making plot-only replay byte-stable.
        normalized_pdf = temp_dir / f"{svg_path.stem}.normalized.pdf"
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        writer.clone_document_from_reader(reader)
        writer.metadata = {
            "/Title": svg_path.stem,
            "/Creator": "Net_torch final-six CSV plotter",
            "/Producer": "pypdf deterministic normalization",
        }
        with normalized_pdf.open("wb") as handle:
            writer.write(handle)
        normalized_pdf.replace(pdf_path)
        screenshot = temp_dir / f"{svg_path.stem}.png"
        css_width = int(math.ceil(canvas_width * 96.0 / 25.4))
        css_height = int(math.ceil(canvas_height * 96.0 / 25.4))
        screenshot_command = [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-extensions",
            f"--user-data-dir={user_data}-png",
            "--hide-scrollbars",
            "--force-device-scale-factor=3.125",
            f"--window-size={css_width},{css_height}",
            f"--screenshot={screenshot}",
            html_path.resolve().as_uri(),
        ]
        screenshot_result = subprocess.run(
            screenshot_command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        if screenshot_result.returncode != 0 or not screenshot.is_file():
            raise RuntimeError(
                "SVG-to-PNG export failed: "
                f"exit={screenshot_result.returncode}; "
                f"stderr={screenshot_result.stderr[-2000:]}"
            )
        expected_size = (
            round(canvas_width * MM_TO_INCH * 300),
            round(canvas_height * MM_TO_INCH * 300),
        )
        with Image.open(screenshot) as image:
            if image.size != expected_size:
                image = image.resize(expected_size, Image.Resampling.LANCZOS)
            image.save(png_path, dpi=(300, 300))


def _write_panel_qa(
    final_svg: Path,
    final_png: Path,
    panels_dir: Path,
    spec: Mapping[str, Any],
) -> None:
    full_image = Image.open(final_png)
    canvas_width, canvas_height = [
        float(value) for value in spec.get("canvas_mm", CANVAS_MM)
    ]
    svg_parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
    for panel_id, slot in spec["slots"].items():
        x, y, width, height = [float(value) for value in slot]
        left = int(round(x / canvas_width * full_image.width))
        upper = int(round(y / canvas_height * full_image.height))
        right = int(round((x + width) / canvas_width * full_image.width))
        lower = int(round((y + height) / canvas_height * full_image.height))
        crop = full_image.crop((left, upper, right, lower))
        crop.save(panels_dir / f"{spec['figure_id']}{panel_id}.png", dpi=(300, 300))
        tree = etree.parse(str(final_svg), svg_parser)
        root = tree.getroot()
        root.set(
            "viewBox",
            " ".join(
                f"{value * MM_TO_POINT:.6f}"
                for value in (x, y, width, height)
            ),
        )
        root.set("width", f"{width}mm")
        root.set("height", f"{height}mm")
        tree.write(
            str(panels_dir / f"{spec['figure_id']}{panel_id}.svg"),
            encoding="utf-8",
            xml_declaration=True,
            pretty_print=False,
        )


def _write_qa_report(
    figure_dir: Path,
    spec: Mapping[str, Any],
    final_png: Path,
    final_pdf: Path,
    final_svg: Path,
    plot_bboxes: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    image = Image.open(final_png)
    canvas_width, canvas_height = [
        float(value) for value in spec.get("canvas_mm", CANVAS_MM)
    ]
    expected_pixels = (
        round(canvas_width * MM_TO_INCH * 300),
        round(canvas_height * MM_TO_INCH * 300),
    )
    parser = etree.XMLParser(resolve_entities=False)
    svg_root = etree.parse(str(final_svg), parser).getroot()
    namespace = {"s": "http://www.w3.org/2000/svg"}
    text_count = len(svg_root.xpath(".//s:text", namespaces=namespace))
    image_count = len(svg_root.xpath(".//s:image", namespaces=namespace))
    rows = []
    for panel_id, slot in spec["slots"].items():
        plot = plot_bboxes[panel_id]
        sx, sy, sw, sh = [float(value) for value in slot]
        px, py, pw, ph = [float(value) for value in plot]
        inside = (
            px >= sx
            and py >= sy
            and px + pw <= sx + sw + 1e-9
            and py + ph <= sy + sh + 1e-9
        )
        rows.append(
            {
                "figure_id": spec["figure_id"],
                "panel_id": panel_id,
                "slot_x_mm": sx,
                "slot_y_mm": sy,
                "slot_w_mm": sw,
                "slot_h_mm": sh,
                "plot_x_mm": px,
                "plot_y_mm": py,
                "plot_w_mm": pw,
                "plot_h_mm": ph,
                "plot_inside_slot": inside,
            }
        )
    pd.DataFrame(rows).to_csv(figure_dir / "meta" / "layout_measurements.csv", index=False)
    report = {
        "figure_id": spec["figure_id"],
        "renderer_version": RENDERER_VERSION,
        "canvas_mm": [canvas_width, canvas_height],
        "png_pixels": [image.width, image.height],
        "expected_png_pixels_at_300dpi": list(expected_pixels),
        "png_size_tolerance_pass": (
            abs(image.width - expected_pixels[0]) <= 3
            and abs(image.height - expected_pixels[1]) <= 3
        ),
        "pdf_bytes": final_pdf.stat().st_size,
        "svg_bytes": final_svg.stat().st_size,
        "svg_text_elements": text_count,
        "svg_image_elements": image_count,
        "editable_text_pass": text_count > 0,
        "all_plot_areas_inside_slots": all(row["plot_inside_slot"] for row in rows),
        "panel_order": list(spec["panels"]),
        "status": "passed",
    }
    if not (
        report["png_size_tolerance_pass"]
        and report["editable_text_pass"]
        and report["all_plot_areas_inside_slots"]
    ):
        report["status"] = "failed"
        raise ValueError(f"{spec['figure_id']}: export QA failed: {report}")
    with (figure_dir / "meta" / "visual_qa.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def _write_artifact_manifest(figure_dir: Path) -> None:
    rows = []
    for path in sorted(figure_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(figure_dir).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    payload = {
        "schema": "final_six_artifact_manifest_v1",
        "figure_id": figure_dir.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": rows,
    }
    with (figure_dir / "artifact_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_spec(figure_dir: Path, spec: Mapping[str, Any]) -> None:
    with (figure_dir / "meta" / "final_plot_spec.json").open("w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2, sort_keys=True)
        handle.write("\n")
    panel_rows = []
    for panel_order, (panel_id, panel) in enumerate(spec["panels"].items(), start=1):
        x_mm, y_mm, width_mm, height_mm = [
            float(value) for value in spec["slots"][panel_id]
        ]
        panel_rows.append(
            {
                "figure_id": spec["figure_id"],
                "panel_id": panel_id,
                "panel_order": panel_order,
                "chart": panel["chart"],
                "claim": panel["claim"],
                "role": panel["role"],
                "source": panel["source"],
                "slot_x_mm": x_mm,
                "slot_y_mm": y_mm,
                "slot_w_mm": width_mm,
                "slot_h_mm": height_mm,
            }
        )
    pd.DataFrame(panel_rows).to_csv(
        figure_dir / "meta" / "main_figure_panel_index.csv",
        index=False,
    )


def _prepare_panel_frame(
    reader: BundleReader,
    figure_id: str,
    panel_id: str,
    panel_spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    chart = str(panel_spec["chart"])
    if chart == "svg_asset":
        manifest = reader.read_csv(str(panel_spec["source"]), f"{figure_id}{panel_id} asset manifest")
        _validate_panel_data(
            manifest, figure_id=figure_id, panel_id=panel_id, schematic=True
        )
        return manifest, None
    if chart == "protocol":
        nodes = reader.read_csv(str(panel_spec["source"]), f"{figure_id}{panel_id} protocol nodes")
        edges = reader.read_csv(
            str(panel_spec["edge_source"]), f"{figure_id}{panel_id} protocol edges"
        )
        _validate_panel_data(nodes, figure_id=figure_id, panel_id=panel_id, schematic=True)
        if edges.empty:
            raise ValueError(f"{figure_id}{panel_id}: protocol edge table is empty")
        return nodes, edges
    frame = reader.read_csv(str(panel_spec["source"]), f"{figure_id}{panel_id} plot data")
    _validate_panel_data(frame, figure_id=figure_id, panel_id=panel_id)
    frame = _apply_filters(frame, panel_spec.get("filters"))
    statistics_source = f"metrics/panel_{panel_id}_statistics.csv"
    statistics_frame = reader.read_csv(
        statistics_source, f"{figure_id}{panel_id} statistics transparency"
    )
    if statistics_frame.empty:
        raise ValueError(f"{figure_id}{panel_id}: statistics CSV is empty")
    return frame, statistics_frame


def render_figure(
    figure_id: str,
    input_dir: str | os.PathLike[str],
    *,
    check_only: bool = False,
) -> dict[str, Any]:
    spec = get_figure_spec(figure_id)
    canvas_mm = tuple(float(value) for value in spec.get("canvas_mm", CANVAS_MM))
    canvas_width, canvas_height = canvas_mm
    layout_report = validate_layout_contract(spec)
    if not layout_report.ok:
        raise ValueError(
            f"{figure_id}: layout contract failed: {layout_report.failures}"
        )
    figure_dir = Path(input_dir).resolve()
    reader = BundleReader(figure_id=figure_id, figure_dir=figure_dir)
    loaded: dict[str, tuple[pd.DataFrame, pd.DataFrame | None]] = {}
    asset_payload: tuple[bytes, str, str] | None = None
    custom_asset_payloads: dict[str, bytes] = {}
    for panel_id, panel_spec in spec["panels"].items():
        loaded[panel_id] = _prepare_panel_frame(
            reader, figure_id, panel_id, panel_spec
        )
        if panel_spec["chart"] == "svg_asset":
            manifest = loaded[panel_id][0]
            _, asset_bytes = reader.read_registered_svg(manifest)
            if panel_spec.get("custom_renderer"):
                custom_asset_payloads[panel_id] = asset_bytes
            else:
                asset_payload = (
                    asset_bytes,
                    str(
                        panel_spec.get("asset_viewbox_override")
                        or manifest.iloc[0]["viewBox"]
                    ),
                    str(panel_spec.get("asset_embedding") or "inline"),
                )
    if check_only:
        reader.write_access_log()
        return {
            "figure_id": figure_id,
            "status": "check_passed",
            "layout_passes": layout_report.passes,
            "panel_count": len(spec["panels"]),
        }

    figures_dir = figure_dir / "figures"
    panels_dir = figures_dir / "panels"
    qa_dir = figures_dir / "qa"
    panels_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    _write_spec(figure_dir, spec)
    _wireframe(spec, qa_dir / f"{figure_id}_wireframe.png")

    plot_bboxes: dict[str, tuple[float, float, float, float]] = {}
    with plt.rc_context(
        {
            **VECTOR_TEXT_RCPARAMS,
            "svg.hashsalt": "net_torch_final_six_figures_v1",
        }
    ):
        fig = plt.figure(
            figsize=(canvas_width * MM_TO_INCH, canvas_height * MM_TO_INCH),
            dpi=300,
            facecolor="white",
        )
        for panel_id, panel_spec in spec["panels"].items():
            slot = spec["slots"][panel_id]
            chart = str(panel_spec["chart"])
            plot_bbox = _plot_bbox_mm(slot, chart, panel_spec)
            plot_bboxes[panel_id] = plot_bbox
            axis = fig.add_axes(_as_figure_axes(plot_bbox, canvas_mm))
            frame, auxiliary = loaded[panel_id]
            if chart == "svg_asset":
                custom_renderer = str(panel_spec.get("custom_renderer") or "")
                if custom_renderer == "fig2_transition":
                    _plot_fig2_transition_schematic(
                        axis,
                        custom_asset_payloads[panel_id],
                        panel_spec,
                    )
                else:
                    axis.axis("off")
                if panel_spec.get("asset_annotation") and not custom_renderer:
                    annotation = fig.text(
                        (float(slot[0]) + float(slot[2]) / 2.0) / canvas_width,
                        1.0 - (float(slot[1]) + 1.5) / canvas_height,
                        str(panel_spec["asset_annotation"]),
                        ha="center",
                        va="top",
                        color=INK,
                    )
                    mark_relative_text_size(annotation, 0.9)
            elif chart == "protocol":
                if auxiliary is None:
                    raise AssertionError("protocol edges were not loaded")
                _plot_protocol(axis, frame, auxiliary, panel_spec)
            elif chart in {"forest", "estimate_strip"}:
                _plot_forest(axis, frame, panel_spec)
            elif chart == "grouped_bars":
                _plot_grouped_bars(axis, frame, panel_spec)
            elif chart == "joint_endpoint_plane":
                _plot_joint_endpoint_plane(axis, frame, panel_spec)
            elif chart == "threshold_margin_bars":
                if auxiliary is None:
                    raise AssertionError(
                        "threshold-margin statistics were not loaded"
                    )
                _plot_threshold_margin_bars(
                    axis, frame, auxiliary, panel_spec
                )
            elif chart == "seed_paired_dumbbells":
                _plot_seed_paired_dumbbells(axis, frame, panel_spec)
            elif chart == "state_space_glyph":
                _plot_state_space_glyph(axis, frame, panel_spec)
            elif chart == "boxplot":
                _plot_boxplot(axis, frame, panel_spec)
            elif chart == "bullet_gauges":
                _plot_bullet_gauges(axis, frame, panel_spec)
            elif chart == "paired_slope":
                if auxiliary is None:
                    raise AssertionError("paired-slope statistics were not loaded")
                _plot_paired_slope(axis, frame, auxiliary, panel_spec)
            elif chart == "ordered_bars":
                _plot_ordered_bars(axis, frame, panel_spec)
            elif chart == "category_points":
                _plot_category_points(axis, frame, panel_spec)
            elif chart == "ordered_lines":
                _plot_ordered_lines(axis, frame, panel_spec)
            elif chart == "seed_trajectory":
                _plot_seed_trajectory(axis, frame, panel_spec)
            elif chart == "time_binned_lines":
                _plot_time_binned_lines(axis, frame, panel_spec)
            elif chart == "stacked_composition":
                _plot_stacked_composition(axis, frame, panel_spec)
            elif chart == "partial_cue_split":
                _plot_partial_cue_split(axis, frame, panel_spec)
            elif chart == "heatmap":
                _plot_heatmap(fig, axis, frame, panel_spec)
            elif chart == "split_conditions":
                _plot_split_conditions(fig, axis, frame, panel_spec)
            elif chart == "line_with_contrast":
                _plot_line_with_contrast(fig, axis, frame, panel_spec)
            elif chart == "two_by_two":
                _plot_two_by_two(fig, axis, frame, panel_spec)
            else:
                raise ValueError(f"{figure_id}{panel_id}: unknown chart {chart}")
            x, y, _, _ = [float(value) for value in slot]
            panel_label = fig.text(
                (x + 0.3) / canvas_width,
                1.0 - (y + 0.6) / canvas_height,
                panel_id,
                ha="left",
                va="top",
                color=INK,
                zorder=100,
            )
            mark_panel_label(panel_label)
        apply_paper_figure_typography(fig)
        base_svg = qa_dir / f"{figure_id}_base.svg"
        fig.savefig(
            base_svg,
            format="svg",
            facecolor="white",
            bbox_inches=None,
            metadata={"Date": None},
        )
        plt.close(fig)

    final_svg = figures_dir / f"{figure_id}.svg"
    if asset_payload is not None:
        _inject_svg_asset(
            base_svg,
            final_svg,
            asset_bytes=asset_payload[0],
            asset_viewbox=asset_payload[1],
            slot=spec["slots"]["a"],
            embedding_mode=asset_payload[2],
            top_padding_mm=float(
                spec["panels"]["a"].get("asset_top_padding_mm", 5.0)
            ),
        )
    else:
        shutil.copyfile(base_svg, final_svg)
    final_pdf = figures_dir / f"{figure_id}.pdf"
    final_png = figures_dir / f"{figure_id}.png"
    _export_pdf_and_png(final_svg, final_pdf, final_png, canvas_mm)
    Image.open(final_png).convert("L").save(
        qa_dir / f"{figure_id}_grayscale.png", dpi=(300, 300)
    )
    _write_panel_qa(final_svg, final_png, panels_dir, spec)
    qa_report = _write_qa_report(
        figure_dir,
        spec,
        final_png,
        final_pdf,
        final_svg,
        plot_bboxes,
    )
    reader.write_access_log()
    summary = reader.read_json("summary.json", f"{figure_id} bundle summary update")
    summary["plotting"] = {
        "renderer_version": RENDERER_VERSION,
        "status": "ready",
        "outputs": [
            f"figures/{figure_id}.png",
            f"figures/{figure_id}.pdf",
            f"figures/{figure_id}.svg",
        ],
        "visual_qa": qa_report,
    }
    with (figure_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    log_line = (
        f"{datetime.now(timezone.utc).isoformat()} {figure_id} "
        f"plot-only ready renderer={RENDERER_VERSION}\n"
    )
    (figure_dir / "logs" / "plot.log").write_text(log_line, encoding="utf-8")
    with (figure_dir.parent / "logs" / "plot_build.log").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(log_line)
    _write_artifact_manifest(figure_dir)
    return {
        "figure_id": figure_id,
        "status": "plot_ready",
        "png": str(final_png),
        "pdf": str(final_pdf),
        "svg": str(final_svg),
        "panel_qa_count": len(spec["panels"]),
        "layout_passes": layout_report.passes,
    }


__all__ = ["RENDERER_VERSION", "render_figure"]
