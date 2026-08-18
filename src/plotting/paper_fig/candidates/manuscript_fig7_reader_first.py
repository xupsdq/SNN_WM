from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from PIL import Image
from pypdf import PdfReader
from scipy import stats

from src.plotting.common.colors import get_plot_color
from src.plotting.paper_fig.layout_contract import validate_layout_contract
from src.plotting.paper_fig.typography import (
    VECTOR_TEXT_RCPARAMS,
    apply_paper_figure_typography,
    mark_panel_label,
    mark_relative_text_size,
)

CANDIDATE_VERSION = "manuscript_fig7_reader_first_v1"
DISPLAY_NAME = "Fig.7"
EXPECTED_SEEDS = tuple(range(1000, 1020))
MM_TO_INCH = 1.0 / 25.4
MM_TO_POINT = 72.0 / 25.4
SPEC_PATH = (
    Path(__file__).resolve().parent / "specs" / "manuscript_fig7_reader_first_v1.json"
)

INK = get_plot_color("ink", context="manuscript_fig7")
NEUTRAL_MID = get_plot_color("neutral_mid", context="manuscript_fig7")
NEUTRAL_LIGHT = get_plot_color("neutral_light", context="manuscript_fig7")
PALE = get_plot_color("neutral_pale", context="manuscript_fig7")

PARENT_DATA_FILES = {
    "data/panel_a_plot_data.csv",
    "data/panel_a_auc_contrasts.csv",
    "data/panel_b_absolute_access.csv",
    "data/panel_b_plot_data.csv",
    "data/panel_c_position_profiles.csv",
    "data/panel_d_plot_data.csv",
    "data/panel_e_plot_data.csv",
    "data/panel_f_plot_data.csv",
    "data/panel_f_window_robustness.csv",
    "metrics/panel_a_statistics.csv",
    "metrics/panel_b_absolute_access_statistics.csv",
    "metrics/panel_b_statistics.csv",
    "metrics/panel_c_position_statistics.csv",
    "metrics/panel_c_statistics.csv",
    "metrics/panel_d_statistics.csv",
    "metrics/panel_e_statistics.csv",
    "metrics/panel_f_statistics.csv",
    "meta/panel_c_source_manifest.csv",
    "meta/source_manifest.csv",
    "meta/parent_hashes_before.csv",
    "meta/parent_hashes_after.csv",
    "meta/final_plot_spec.json",
    "artifact_manifest.json",
    "summary.json",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_spec() -> dict[str, Any]:
    with SPEC_PATH.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("candidate_version") != CANDIDATE_VERSION:
        raise ValueError("candidate spec version mismatch")
    if spec.get("display_name") != DISPLAY_NAME:
        raise ValueError("candidate display name must be Fig.7")
    return spec


def _snapshot_tree(root: Path, source_scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "source_scope": source_scope,
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows, columns=["source_scope", "path", "bytes", "sha256"])


def _snapshot_digest(frame: pd.DataFrame) -> str:
    columns = ["source_scope", "path", "bytes", "sha256"]
    normalized = frame.sort_values(columns).loc[:, columns].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


@dataclass
class BundleReader:
    root: Path
    expected_root: Path
    allowed_files: set[str]
    accesses: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.expected_root = self.expected_root.resolve()
        if self.root != self.expected_root:
            raise ValueError("plotting accepts only the parent root pinned by the spec")
        if not self.root.is_dir():
            raise FileNotFoundError(f"pinned parent root is missing: {self.root}")

    def _resolve(self, relative: str, purpose: str) -> Path:
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise ValueError(f"absolute source path is forbidden: {relative}")
        normalized = relative_path.as_posix()
        if normalized not in self.allowed_files:
            raise PermissionError(f"unregistered parent source: {relative}")
        path = (self.root / relative_path).resolve()
        if not _inside(path, self.root) or not path.is_file():
            raise FileNotFoundError(f"required parent source is missing: {path}")
        if path.suffix.lower() not in {".csv", ".json"}:
            raise PermissionError(f"unsupported parent source type: {path}")
        self.accesses.append(
            {
                "candidate_figure": DISPLAY_NAME,
                "source_scope": "frozen_parent_bundle",
                "relative_path": normalized,
                "source_path": str(path),
                "purpose": purpose,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
        return path

    def read_csv(self, relative: str, purpose: str) -> pd.DataFrame:
        return pd.read_csv(self._resolve(relative, purpose))

    def read_json(self, relative: str, purpose: str) -> dict[str, Any]:
        with self._resolve(relative, purpose).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def access_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.accesses)


def _as_axes_bbox(bbox_mm: Sequence[float], canvas_mm: Sequence[float]) -> list[float]:
    left, top, width, height = [float(value) for value in bbox_mm]
    canvas_width, canvas_height = [float(value) for value in canvas_mm]
    return [
        left / canvas_width,
        (canvas_height - top - height) / canvas_height,
        width / canvas_width,
        height / canvas_height,
    ]


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_color(INK)
        axis.spines[side].set_linewidth(0.6)
    axis.tick_params(axis="both", which="major", colors=INK, width=0.6, length=2.5, pad=2.0)
    axis.tick_params(axis="both", which="minor", length=0)
    axis.minorticks_off()


def _numeric_tick(value: float, _position: int) -> str:
    if abs(float(value) - round(float(value))) < 1e-10:
        return str(int(round(float(value))))
    return f"{float(value):g}"


def _require_seed_set(frame: pd.DataFrame, label: str) -> None:
    if "network_seed" not in frame:
        raise ValueError(f"{label}: network_seed is missing")
    seeds = set(pd.to_numeric(frame["network_seed"], errors="raise").astype(int))
    if seeds != set(EXPECTED_SEEDS):
        raise ValueError(
            f"{label}: expected seeds 1000-1019; "
            f"missing={sorted(set(EXPECTED_SEEDS) - seeds)}, "
            f"extra={sorted(seeds - set(EXPECTED_SEEDS))}"
        )


def _one_statistic(
    statistics: pd.DataFrame,
    *,
    endpoint: str,
    contrast: str = "",
    group: str,
) -> pd.Series:
    rows = statistics.loc[
        statistics["endpoint"].astype(str).eq(endpoint)
        & statistics["contrast"].fillna("").astype(str).eq(contrast)
        & statistics["group"].fillna("").astype(str).eq(group)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one frozen statistic for endpoint={endpoint!r}, "
            f"contrast={contrast!r}, group={group!r}; observed {len(rows)}"
        )
    row = rows.iloc[0]
    values = pd.to_numeric(row[["estimate", "ci95_low", "ci95_high"]], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not values[1] <= values[0] <= values[2]:
        raise ValueError(f"invalid frozen estimate or confidence interval: {values}")
    return row


def _t_ci(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        raise ValueError("a Student t 95% CI requires at least two network values")
    mean = float(array.mean())
    sem = float(stats.sem(array))
    half = float(stats.t.ppf(0.975, len(array) - 1) * sem)
    return mean, mean - half, mean + half


def _validate_frozen_mean(values: pd.Series, statistic: pd.Series, label: str) -> None:
    observed = float(pd.to_numeric(values, errors="raise").mean())
    expected = float(statistic["estimate"])
    if not np.isclose(observed, expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"{label}: network mean {observed} disagrees with frozen estimate {expected}")


def _validate_t_ci(values: pd.Series, statistic: pd.Series, label: str) -> None:
    mean, low, high = _t_ci(values)
    expected = [float(statistic["ci95_low"]), float(statistic["ci95_high"])]
    if not np.isclose(mean, float(statistic["estimate"]), rtol=0.0, atol=1e-9):
        raise ValueError(f"{label}: materialized mean {mean} disagrees with frozen estimate {statistic['estimate']}")
    if not np.isclose(low, expected[0], rtol=0.0, atol=1e-9) or not np.isclose(high, expected[1], rtol=0.0, atol=1e-9):
        raise ValueError(f"{label}: materialized CI ({low}, {high}) disagrees with frozen CI ({expected[0]}, {expected[1]})")


def _relabel(frame: pd.DataFrame, panel_id: str) -> pd.DataFrame:
    output = frame.copy()
    if "figure_id" in output.columns:
        output["figure_id"] = DISPLAY_NAME
    if "panel_id" in output.columns:
        output["panel_id"] = panel_id
    return output


def _seed_jitter(seeds: Sequence[Any], width: float) -> np.ndarray:
    numeric = np.asarray([int(value) for value in seeds], dtype=float)
    centered = ((numeric - float(EXPECTED_SEEDS[0])) % len(EXPECTED_SEEDS)) - 9.5
    return centered / 9.5 * float(width)


# ---------------------------------------------------------------- panel a


def _validate_panel_a(
    raw: pd.DataFrame,
    contrasts_raw: pd.DataFrame,
    statistics: pd.DataFrame,
) -> dict[str, Any]:
    label = "Fig.7a"
    _require_seed_set(raw, label)
    _require_seed_set(contrasts_raw, label)
    required = {"network_seed", "condition", "target_item", "keep_prob", "value"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    data = raw.copy()
    data["network_seed"] = pd.to_numeric(data["network_seed"], errors="raise").astype(int)
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    data["keep_prob"] = pd.to_numeric(data["keep_prob"], errors="raise")
    if not np.isfinite(data["value"].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: non-finite network value")
    condition_order = ["S0", "S_A", "S_B", "S_AB"]
    target_order = ["A", "B"]
    keep_order = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    if set(data["condition"].astype(str)) != set(condition_order):
        raise ValueError(f"{label}: condition set mismatch")
    if set(data["target_item"].astype(str)) != set(target_order):
        raise ValueError(f"{label}: target set mismatch")
    if set(np.round(data["keep_prob"], 12)) != set(keep_order):
        raise ValueError(f"{label}: keep-probability grid mismatch")
    counts = data.groupby(["network_seed", "condition", "target_item", "keep_prob"]).size()
    if len(counts) != 20 * 4 * 2 * 7 or not (counts == 1).all():
        raise ValueError(f"{label}: expected exactly one row per network-condition-target-keep cell")
    curves: list[dict[str, Any]] = []
    stats_cross_checks: dict[str, bool] = {}
    for target in target_order:
        for condition in condition_order:
            for keep in keep_order:
                group = f"P_target|{target}|{condition}|{keep}"
                stat = _one_statistic(statistics, endpoint="P_target", group=group)
                subset = data.loc[
                    data["target_item"].astype(str).eq(target)
                    & data["condition"].astype(str).eq(condition)
                    & np.isclose(data["keep_prob"], keep, rtol=0.0, atol=1e-12)
                ]
                values = pd.to_numeric(subset["value"], errors="raise")
                if len(values) != 20:
                    raise ValueError(f"{label}: incomplete cell {group}")
                _validate_frozen_mean(values, stat, f"{label} {group}")
                mean, low, high = _t_ci(values)
                if not (
                    np.isclose(low, float(stat["ci95_low"]), rtol=0.0, atol=1e-9)
                    and np.isclose(high, float(stat["ci95_high"]), rtol=0.0, atol=1e-9)
                ):
                    raise ValueError(f"{label}: materialized CI disagrees with frozen CI at {group}")
                stats_cross_checks[group] = True
                curves.append(
                    {
                        "figure_id": DISPLAY_NAME,
                        "panel_id": "a",
                        "target_item": target,
                        "condition": condition,
                        "keep_prob": keep,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_networks": 20,
                        "frozen_estimate": float(stat["estimate"]),
                    }
                )
    curve_frame = pd.DataFrame(curves)
    contrast_rows: list[dict[str, Any]] = []
    contrast_stats: dict[str, dict[str, Any]] = {}
    for target in target_order:
        for endpoint in ("SAB_vs_S0_auc_gain", "SAB_vs_relevant_single_auc_gain"):
            group = f"{endpoint}|{target}"
            stat = _one_statistic(
                statistics,
                endpoint=endpoint,
                contrast=endpoint,
                group=group,
            )
            values = pd.to_numeric(
                contrasts_raw.loc[
                    contrasts_raw["target_item"].astype(str).eq(target)
                    & contrasts_raw["endpoint"].astype(str).eq(endpoint),
                    "value",
                ],
                errors="raise",
            )
            if len(values) != 20:
                raise ValueError(f"{label}: incomplete AUC contrast {group}")
            _validate_frozen_mean(values, stat, f"{label} {group}")
            mean, low, high = _t_ci(values)
            if not (
                np.isclose(low, float(stat["ci95_low"]), rtol=0.0, atol=1e-9)
                and np.isclose(high, float(stat["ci95_high"]), rtol=0.0, atol=1e-9)
            ):
                raise ValueError(f"{label}: AUC contrast CI disagrees at {group}")
            if mean <= 0.0:
                raise ValueError(f"{label}: AUC contrast must be positive at {group}")
            contrast_stats[group] = {
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "p_adjusted": float(stat["p_adjusted"]),
                "n_networks": int(stat["n_networks"]),
            }
            for seed in EXPECTED_SEEDS:
                seed_rows = contrasts_raw.loc[
                    contrasts_raw["network_seed"].astype(int).eq(seed)
                    & contrasts_raw["target_item"].astype(str).eq(target)
                    & contrasts_raw["endpoint"].astype(str).eq(endpoint)
                ]
                if len(seed_rows) != 1:
                    raise ValueError(f"{label}: AUC contrast row missing for seed {seed} {group}")
                contrast_rows.append(
                    {
                        "figure_id": DISPLAY_NAME,
                        "panel_id": "a",
                        "network_seed": seed,
                        "record_type": "paired_network_auc_contrast",
                        "endpoint": endpoint,
                        "condition": endpoint,
                        "target_item": target,
                        "value": float(seed_rows["value"].iloc[0]),
                        "unit": "normalized_auc_gain",
                    }
                )
    contrast_frame = pd.DataFrame(contrast_rows)
    checks = {
        "expected": float(contrast_stats["SAB_vs_S0_auc_gain|A"]["mean"]),
        "observed": float(contrast_stats["SAB_vs_S0_auc_gain|A"]["mean"]),
        "match": True,
    }
    return {
        "curves": curve_frame,
        "contrasts": contrast_frame,
        "contrast_stats": contrast_stats,
        "cells_cross_checked": len(stats_cross_checks),
        "auc_checks": {
            key: {"mean": value["mean"], "ci95": [value["ci95_low"], value["ci95_high"]]}
            for key, value in contrast_stats.items()
        },
        "checks": checks,
    }


def _draw_panel_a(
    axis: plt.Axes,
    payload: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
) -> dict[str, Any]:
    curves = payload["curves"]
    target_order = [str(value) for value in panel_spec["target_order"]]
    condition_order = [str(value) for value in panel_spec["condition_order"]]
    condition_labels = panel_spec["condition_labels"]
    keep_order = [float(value) for value in panel_spec["x_order"]]
    colors = {
        condition: get_plot_color(str(panel_spec["colors"][condition]), context="manuscript_fig7")
        for condition in condition_order
    }
    line_widths = {condition: float(panel_spec["line_width"][condition]) for condition in condition_order}
    ci_alpha = {condition: float(panel_spec["ci_band_alpha"][condition]) for condition in condition_order}
    marker_sizes = {condition: float(panel_spec["marker_size"][condition]) for condition in condition_order}
    marker_cycle = ("o", "s", "^", "D")
    foreground = str(panel_spec["foreground_condition"])
    axis.set_axis_off()
    parent_x, parent_y, parent_width, parent_height = [
        float(value) for value in panel_spec["plot_bbox_mm"]
    ]
    child_bboxes = [list(map(float, value)) for value in panel_spec["child_plot_bboxes_mm"]]
    inset_bounds: list[list[float]] = []
    for child_bbox in child_bboxes:
        child_x, child_y, child_width, child_height = child_bbox
        inset_bounds.append(
            [
                (child_x - parent_x) / parent_width,
                (parent_y + parent_height - child_y - child_height) / parent_height,
                child_width / parent_width,
                child_height / parent_height,
            ]
        )
    left = axis.inset_axes(inset_bounds[0])
    right = axis.inset_axes(inset_bounds[1], sharey=left)
    child_axes = [left, right]
    handles: list[Line2D] = []
    labels: list[str] = []
    per_condition_rows = {condition: 0 for condition in condition_order}
    for target_index, (target, child) in enumerate(zip(target_order, child_axes)):
        for condition_index, condition in enumerate(condition_order):
            subset = curves.loc[
                curves["target_item"].astype(str).eq(target)
                & curves["condition"].astype(str).eq(condition)
            ].sort_values("keep_prob")
            x_values = subset["keep_prob"].to_numpy(dtype=float)
            means = subset["mean"].to_numpy(dtype=float)
            lows = subset["ci95_low"].to_numpy(dtype=float)
            highs = subset["ci95_high"].to_numpy(dtype=float)
            zorder = 4 if condition == foreground else 3
            child.fill_between(
                x_values,
                lows,
                highs,
                color=colors[condition],
                alpha=ci_alpha[condition],
                linewidth=0,
                zorder=1,
            )
            child.plot(
                x_values,
                means,
                color=colors[condition],
                lw=line_widths[condition],
                marker=marker_cycle[condition_index % len(marker_cycle)],
                markersize=marker_sizes[condition],
                markerfacecolor=colors[condition],
                markeredgecolor=colors[condition],
                zorder=zorder,
            )
            per_condition_rows[condition] += len(subset)
            if target_index == 0:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=colors[condition],
                        lw=1.2,
                        marker=marker_cycle[condition_index % len(marker_cycle)],
                        markersize=3.0,
                    )
                )
                labels.append(str(condition_labels[condition]))
        child.set_xlim(0.0, 1.02)
        child.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
        child.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
        child.set_yticks([float(value) for value in panel_spec["y_ticks"]])
        target_label = child.text(
            0.5,
            1.02,
            f"Target {target}",
            transform=child.transAxes,
            ha="center",
            va="bottom",
        )
        mark_relative_text_size(target_label, 0.86)
        child.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
        _style_axis(child)
        if target_index == 0:
            child.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
        else:
            child.tick_params(axis="y", labelleft=False)
            if not bool(panel_spec.get("show_right_y_axis", True)):
                child.tick_params(axis="y", left=False)
    legend_anchor = [float(value) for value in panel_spec["legend_anchor"]]
    axis.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=tuple(legend_anchor),
        ncol=int(panel_spec["legend_ncol"]),
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
        labelspacing=0.3,
    )
    return {
        "child_axes": [str(child) for child in child_axes],
        "conditions": condition_order,
        "targets": target_order,
        "curve_points_per_condition": per_condition_rows,
        "legend_order": labels,
        "foreground_condition": foreground,
    }


# ---------------------------------------------------------------- panel b


def _validate_panel_b(
    raw: pd.DataFrame,
    statistics: pd.DataFrame,
    gain_statistics: pd.DataFrame,
) -> dict[str, Any]:
    label = "Fig.7b"
    _require_seed_set(raw, label)
    required = {"network_seed", "endpoint", "value", "target_position"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    data = raw.copy()
    data["network_seed"] = pd.to_numeric(data["network_seed"], errors="raise").astype(int)
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    data["target_position"] = pd.to_numeric(data["target_position"], errors="raise").astype(int)
    endpoints = ["P_target_cue_only", "P_target_sequence_state", "P_target_single_item_memory"]
    positions = list(range(1, 11))
    if set(data["endpoint"].astype(str)) != set(endpoints):
        raise ValueError(f"{label}: endpoint set mismatch")
    if set(data["target_position"]) != set(positions):
        raise ValueError(f"{label}: serial positions must be exactly 1-10")
    counts = data.groupby(["network_seed", "endpoint", "target_position"]).size()
    if len(counts) != 20 * 3 * 10 or not (counts == 1).all():
        raise ValueError(f"{label}: expected exactly one row per network-endpoint-position")
    rows: list[dict[str, Any]] = []
    for endpoint in endpoints:
        for position in positions:
            group = f"{endpoint}|{position}"
            stat = _one_statistic(statistics, endpoint=endpoint, group=group)
            subset = data.loc[
                data["endpoint"].astype(str).eq(endpoint)
                & data["target_position"].eq(position)
            ]
            values = pd.to_numeric(subset["value"], errors="raise")
            _validate_frozen_mean(values, stat, f"{label} {group}")
            mean, low, high = _t_ci(values)
            if not (
                np.isclose(low, float(stat["ci95_low"]), rtol=0.0, atol=1e-9)
                and np.isclose(high, float(stat["ci95_high"]), rtol=0.0, atol=1e-9)
            ):
                raise ValueError(f"{label}: CI disagrees at {group}")
            rows.append(
                {
                    "figure_id": DISPLAY_NAME,
                    "panel_id": "b",
                    "endpoint": endpoint,
                    "target_position": position,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_networks": 20,
                }
            )
    curve_frame = pd.DataFrame(rows)
    piv = data.pivot_table(index="network_seed", columns="endpoint", values="value")
    network_gain = piv["P_target_sequence_state"] - piv["P_target_single_item_memory"]
    mean, low, high = _t_ci(network_gain)
    stat = _one_statistic(
        gain_statistics,
        endpoint="sequence_minus_singleton_access_gain",
        contrast="mean_sequence_minus_singleton_access_gain_vs_zero",
        group="mean_sequence_minus_singleton_access_gain_vs_zero",
    )
    _validate_frozen_mean(network_gain, stat, f"{label} network-level mean gain")
    if not (
        np.isclose(low, float(stat["ci95_low"]), rtol=0.0, atol=1e-9)
        and np.isclose(high, float(stat["ci95_high"]), rtol=0.0, atol=1e-9)
    ):
        raise ValueError(f"{label}: network-level gain CI disagrees with frozen row")
    if not bool((network_gain > 0.0).all()):
        raise ValueError(f"{label}: expected all 20 network gains positive")
    per_seed = pd.DataFrame(
        {
            "figure_id": DISPLAY_NAME,
            "panel_id": "b",
            "network_seed": list(EXPECTED_SEEDS),
            "record_type": "network_level_cross_position_gain",
            "endpoint": "mean_sequence_minus_singleton_access_gain",
            "value": network_gain.to_numpy(dtype=float),
            "unit": "percent",
            "frozen_estimate": float(stat["estimate"]),
        }
    )
    return {
        "curves": curve_frame,
        "network_gain": per_seed,
        "network_gain_mean": mean,
        "network_gain_ci95": [low, high],
        "gain_frozen": {
            "estimate": float(stat["estimate"]),
            "ci95_low": float(stat["ci95_low"]),
            "ci95_high": float(stat["ci95_high"]),
            "p_adjusted": float(stat["p_adjusted"]),
        },
    }


def _draw_panel_b(
    axis: plt.Axes,
    payload: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
) -> dict[str, Any]:
    curves = payload["curves"]
    hue_order = [str(value) for value in panel_spec["hue_order"]]
    hue_labels = panel_spec["hue_labels"]
    colors = {
        endpoint: get_plot_color(str(panel_spec["colors"][endpoint]), context="manuscript_fig7")
        for endpoint in hue_order
    }
    line_widths = {endpoint: float(panel_spec["line_width"][endpoint]) for endpoint in hue_order}
    foreground = str(panel_spec["foreground_endpoint"])
    zorder = {endpoint: 4 if endpoint == foreground else 3 for endpoint in hue_order}
    for endpoint in hue_order:
        subset = curves.loc[curves["endpoint"].astype(str).eq(endpoint)].sort_values("target_position")
        x_values = subset["target_position"].to_numpy(dtype=float)
        means = subset["mean"].to_numpy(dtype=float)
        lows = subset["ci95_low"].to_numpy(dtype=float)
        highs = subset["ci95_high"].to_numpy(dtype=float)
        axis.fill_between(
            x_values,
            lows,
            highs,
            color=colors[endpoint],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x_values,
            means,
            color=colors[endpoint],
            lw=line_widths[endpoint],
            marker="o",
            markersize=2.8,
            markerfacecolor=colors[endpoint],
            markeredgecolor=colors[endpoint],
            zorder=zorder[endpoint],
        )
    axis.set_xlim(*[float(value) for value in panel_spec["x_limits"]])
    axis.set_xticks([float(value) for value in panel_spec["x_ticks"]])
    axis.xaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.yaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    handles = [
        Line2D(
            [0],
            [0],
            color=colors[endpoint],
            lw=1.2,
            marker="o",
            markersize=3.0,
            label=str(hue_labels[endpoint]),
        )
        for endpoint in hue_order
    ]
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=int(panel_spec["legend_ncol"]),
        handlelength=1.25,
        handletextpad=0.45,
        columnspacing=1.0,
        borderaxespad=0.0,
        labelspacing=0.3,
    )
    return {
        "hue_order": hue_order,
        "legend_order": [str(hue_labels[endpoint]) for endpoint in hue_order],
        "positions": list(range(1, 11)),
        "network_gain_mean": payload["network_gain_mean"],
        "network_gain_ci95": payload["network_gain_ci95"],
        "foreground_endpoint": foreground,
    }


# ---------------------------------------------------------------- panel c


def _validate_panel_c(
    raw: pd.DataFrame,
    statistics: pd.DataFrame,
    position_statistics: pd.DataFrame,
) -> dict[str, Any]:
    label = "Fig.7c"
    _require_seed_set(raw, label)
    required = {"network_seed", "condition", "value", "target_position"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    data = raw.copy()
    data["network_seed"] = pd.to_numeric(data["network_seed"], errors="raise").astype(int)
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    data["target_position"] = pd.to_numeric(data["target_position"], errors="raise").astype(int)
    conditions = ["matched", "same_label_novel", "unseen"]
    if set(data["condition"].astype(str)) != set(conditions):
        raise ValueError(f"{label}: condition set must be matched/same_label_novel/unseen")
    if set(data["target_position"]) != set(range(1, 8)):
        raise ValueError(f"{label}: serial positions must be exactly 1-7")
    counts = data.groupby(["network_seed", "condition", "target_position"]).size()
    if len(counts) != 20 * 3 * 7 or not (counts == 1).all():
        raise ValueError(f"{label}: expected exactly one row per network-condition-position")
    if data.duplicated(["network_seed", "condition", "target_position"]).any():
        raise ValueError(f"{label}: duplicate network-condition-position row")
    position_cross_checks: dict[str, bool] = {}
    for condition in conditions:
        for position in range(1, 8):
            group = f"target_probability|{condition}|{position}"
            stat = _one_statistic(position_statistics, endpoint="target_probability", group=group)
            values = pd.to_numeric(
                data.loc[
                    data["condition"].astype(str).eq(condition)
                    & data["target_position"].eq(position),
                    "value",
                ],
                errors="raise",
            )
            _validate_frozen_mean(values, stat, f"{label} {group}")
            position_cross_checks[group] = True
    per_network_condition = data.pivot_table(
        index="network_seed",
        columns="condition",
        values="value",
        aggfunc="mean",
    )
    if per_network_condition.shape != (20, 3):
        raise ValueError(f"{label}: position aggregation must yield 20 networks x 3 conditions")
    absolute_rows: list[dict[str, Any]] = []
    for condition in conditions:
        values = pd.to_numeric(per_network_condition[condition], errors="raise")
        mean, low, high = _t_ci(values)
        absolute_rows.append(
            {
                "figure_id": DISPLAY_NAME,
                "panel_id": "c",
                "condition": condition,
                "record_type": "network_level_absolute_position_mean",
                "endpoint": "target_probability_position_mean",
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "n_networks": 20,
                "unit": "percent",
            }
        )
    contrasts: dict[str, pd.Series] = {
        "matched_minus_same_label_novel": per_network_condition["matched"] - per_network_condition["same_label_novel"],
        "matched_minus_unseen": per_network_condition["matched"] - per_network_condition["unseen"],
    }
    contrast_rows: list[dict[str, Any]] = []
    contrast_stats: dict[str, dict[str, Any]] = {}
    for contrast_name, values in contrasts.items():
        stat = _one_statistic(
            statistics,
            endpoint=contrast_name,
            contrast=contrast_name,
            group=f"{contrast_name}|cue_specificity",
        )
        values = pd.to_numeric(values, errors="raise")
        _validate_frozen_mean(values, stat, f"{label} {contrast_name}")
        mean, low, high = _t_ci(values)
        if not (
            np.isclose(low, float(stat["ci95_low"]), rtol=0.0, atol=1e-9)
            and np.isclose(high, float(stat["ci95_high"]), rtol=0.0, atol=1e-9)
        ):
            raise ValueError(f"{label}: CI disagrees for {contrast_name}")
        if not bool((values > 0.0).all()):
            raise ValueError(f"{label}: expected all 20 networks positive for {contrast_name}")
        if not np.isclose(float(values.min()), float(stat["min"]), rtol=0.0, atol=1e-9):
            raise ValueError(f"{label}: materialized min disagrees with frozen min for {contrast_name}")
        if not np.isclose(float(values.max()), float(stat["max"]), rtol=0.0, atol=1e-9):
            raise ValueError(f"{label}: materialized max disagrees with frozen max for {contrast_name}")
        contrast_stats[contrast_name] = {
            "mean": mean,
            "ci95_low": low,
            "ci95_high": high,
            "positive_networks": int((values > 0.0).sum()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p_adjusted": float(stat["p_adjusted"]),
            "n_networks": int(stat["n_networks"]),
        }
        for seed in EXPECTED_SEEDS:
            contrast_rows.append(
                {
                    "figure_id": DISPLAY_NAME,
                    "panel_id": "c",
                    "network_seed": seed,
                    "record_type": "network_level_paired_contrast",
                    "endpoint": contrast_name,
                    "condition": contrast_name,
                    "value": float(values.loc[seed]),
                    "unit": "percent",
                    "matched_position_mean": float(per_network_condition.loc[seed, "matched"]),
                    "same_label_novel_position_mean": float(per_network_condition.loc[seed, "same_label_novel"]),
                    "unseen_position_mean": float(per_network_condition.loc[seed, "unseen"]),
                }
            )
    contrast_frame = pd.DataFrame(contrast_rows)
    if contrast_frame.duplicated(["network_seed", "endpoint"]).any():
        raise ValueError(f"{label}: duplicate network-contrast row")
    validation = {
        "raw_rows": int(len(data)),
        "network_count": 20,
        "condition_count": 3,
        "position_count": 7,
        "rows_per_network_condition_position": 1,
        "aggregation": "equal-weight mean over the 7 serial positions per network before the paired difference",
        "contrast_rows": int(len(contrast_frame)),
        "contrast_stats": contrast_stats,
        "absolute_position_means": {
            condition: next(
                row["mean"]
                for row in absolute_rows
                if row["condition"] == condition
            )
            for condition in conditions
        },
        "position_cross_checks": len(position_cross_checks),
        "status": "passed",
    }
    return {
        "contrasts": contrast_frame,
        "absolute_means": pd.DataFrame(absolute_rows),
        "contrast_stats": contrast_stats,
        "validation": validation,
    }


def _draw_panel_c(
    axis: plt.Axes,
    payload: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
) -> dict[str, Any]:
    contrasts = payload["contrasts"]
    x_order = [str(value) for value in panel_spec["x_order"]]
    x_labels = panel_spec["x_labels"]
    colors = {
        condition: get_plot_color(str(panel_spec["colors"][condition]), context="manuscript_fig7")
        for condition in x_order
    }
    bar_width = float(panel_spec["bar_width"])
    decimals = int(panel_spec["value_decimals"])
    pad_units = float(panel_spec["label_pad_units"])
    label_positions: dict[str, float] = {}
    label_offsets: dict[str, float] = {}
    for index, condition in enumerate(x_order):
        contrast_name = f"matched_minus_{condition}"
        stat = payload["contrast_stats"][contrast_name]
        mean = float(stat["mean"])
        low = float(stat["ci95_low"])
        high = float(stat["ci95_high"])
        axis.bar(
            index,
            mean,
            width=bar_width,
            color=colors[condition],
            edgecolor="none",
            zorder=2,
        )
        axis.errorbar(
            [index],
            [mean],
            yerr=[[mean - low], [high - mean]],
            fmt="none",
            ecolor=INK,
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.9,
            zorder=5,
        )
        label_y = high + pad_units
        label_positions[condition] = label_y
        label_offsets[condition] = label_y - high
        axis.text(
            float(index),
            label_y,
            f"{mean:.{decimals}f}",
            ha="center",
            va="bottom",
            color=INK,
            zorder=6,
            clip_on=False,
        )
    if bool(panel_spec.get("show_network_points", True)):
        raise ValueError("panel c artwork must not draw network points (author-approved removal)")
    axis.set_xlim(-0.55, len(x_order) - 0.45)
    axis.set_xticks(np.arange(len(x_order), dtype=float))
    axis.set_xticklabels([str(x_labels[condition]) for condition in x_order])
    axis.set_xlabel("", labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.yaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    return {
        "network_points_drawn": 0,
        "label_positions": label_positions,
        "label_offsets": label_offsets,
        "y_limits": list(panel_spec["y_limits"]),
        "numeric_labels": {
            condition: f"{payload['contrast_stats'][f'matched_minus_{condition}']['mean']:.{decimals}f}"
            for condition in x_order
        },
    }


# ---------------------------------------------------------------- panel d


def _validate_panel_d(
    raw: pd.DataFrame,
    statistics: pd.DataFrame,
) -> dict[str, Any]:
    label = "Fig.7d"
    _require_seed_set(raw, label)
    required = {"network_seed", "value", "seq_len", "delay_ms"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    data = raw.copy()
    data["network_seed"] = pd.to_numeric(data["network_seed"], errors="raise").astype(int)
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    data["seq_len"] = pd.to_numeric(data["seq_len"], errors="raise").astype(int)
    data["delay_ms"] = pd.to_numeric(data["delay_ms"], errors="raise").astype(int)
    seq_lens = [3, 5, 7, 10]
    delays = [100, 200, 400, 800]
    if set(data["seq_len"]) != set(seq_lens) or set(data["delay_ms"]) != set(delays):
        raise ValueError(f"{label}: K x delay grid must be 3/5/7/10 x 100/200/400/800")
    counts = data.groupby(["network_seed", "seq_len", "delay_ms"]).size()
    if len(counts) != 20 * 16 or not (counts == 1).all():
        raise ValueError(f"{label}: expected exactly one row per network-cell")
    if not data["value"].between(0.0, 1.0).all():
        raise ValueError(f"{label}: rescued fraction must lie in [0, 1]")
    cell_rows: list[dict[str, Any]] = []
    for seq_len in seq_lens:
        for delay in delays:
            group = f"rescued_fraction|{seq_len}|{delay}"
            stat = _one_statistic(statistics, endpoint="rescued_fraction", group=group)
            subset = data.loc[data["seq_len"].eq(seq_len) & data["delay_ms"].eq(delay)]
            values = pd.to_numeric(subset["value"], errors="raise")
            _validate_frozen_mean(values, stat, f"{label} {group}")
            mean, low, high = _t_ci(values)
            if not (
                np.isclose(low, float(stat["ci95_low"]), rtol=0.0, atol=1e-9)
                and np.isclose(high, float(stat["ci95_high"]), rtol=0.0, atol=1e-9)
            ):
                raise ValueError(f"{label}: CI disagrees at {group}")
            cell_rows.append(
                {
                    "figure_id": DISPLAY_NAME,
                    "panel_id": "d",
                    "seq_len": seq_len,
                    "delay_ms": delay,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_networks": 20,
                }
            )
    cell_frame = pd.DataFrame(cell_rows)
    stat = _one_statistic(
        statistics,
        endpoint="rescued_fraction",
        contrast="standardized_seq_len_x_delay_interaction",
        group="standardized_seq_len_x_delay_interaction",
    )
    interaction = {
        "estimate": float(stat["estimate"]),
        "ci95_low": float(stat["ci95_low"]),
        "ci95_high": float(stat["ci95_high"]),
        "p_value": float(stat["p_value"]),
        "p_adjusted": float(stat["p_adjusted"]),
        "n_networks": int(stat["n_networks"]),
    }
    return {"cells": cell_frame, "interaction": interaction}


def _draw_panel_d(
    fig: plt.Figure,
    axis: plt.Axes,
    payload: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
) -> dict[str, Any]:
    from matplotlib import colormaps

    cells = payload["cells"]
    x_order = [int(value) for value in panel_spec["x_order"]]
    y_order = [int(value) for value in panel_spec["y_order"]]
    pivot = cells.pivot(index="delay_ms", columns="seq_len", values="mean")
    pivot = pivot.reindex(index=y_order, columns=x_order)
    data = pivot.to_numpy(dtype=float)
    cmap = colormaps["Blues"].copy()
    cmap.set_bad("#FFFFFF")
    vmin = float(panel_spec["vmin"])
    vmax = float(panel_spec["vmax"])
    mesh = axis.pcolormesh(
        np.arange(len(x_order) + 1),
        np.arange(len(y_order) + 1),
        np.ma.masked_invalid(data),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        edgecolors="none",
        linewidth=0.0,
        rasterized=False,
    )
    if bool(panel_spec.get("annotate_cells", False)):
        threshold = vmin + 0.55 * (vmax - vmin)
        decimals = int(panel_spec.get("annotation_decimals", 2))
        for y_index in range(len(y_order)):
            for x_index in range(len(x_order)):
                value = data[y_index, x_index]
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
    axis.set_yticks(np.arange(len(y_order)) + 0.5)
    axis.set_yticklabels([str(value) for value in y_order])
    axis.set_xlim(0, len(x_order))
    axis.set_ylim(0, len(y_order))
    axis.invert_yaxis()
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.6)
    axis.spines["bottom"].set_linewidth(0.6)
    axis.tick_params(length=0)
    canvas_height_mm = float(fig.get_figheight()) * 25.4
    bar_height_mm = float(panel_spec["colorbar_height_mm"])
    bar_gap_mm = float(panel_spec["colorbar_gap_mm"])
    axis_bbox = axis.get_position()
    bar_width = axis_bbox.width
    colorbar_axis = fig.add_axes(
        [
            axis_bbox.x0,
            axis_bbox.y1 + bar_gap_mm / canvas_height_mm,
            bar_width,
            bar_height_mm / canvas_height_mm,
        ]
    )
    colorbar = fig.colorbar(mesh, cax=colorbar_axis, orientation="horizontal")
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.tick_params(axis="x", pad=1.0)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.set_xlabel(str(panel_spec["colorbar_label"]), labelpad=float(panel_spec["colorbar_label_pad_pt"]))
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(width=0.6, length=2.5)
    return {
        "cell_values": {
            f"K{x_order[xi]}_D{y_order[yi]}": float(data[yi, xi])
            for yi in range(len(y_order))
            for xi in range(len(x_order))
        },
        "interaction": payload["interaction"],
        "grid": {"seq_len": x_order, "delay_ms": y_order},
    }


# ---------------------------------------------------------------- panel e


def _validate_panel_e(
    raw: pd.DataFrame,
    statistics: pd.DataFrame,
) -> dict[str, Any]:
    label = "Fig.7e"
    _require_seed_set(raw, label)
    required = {"network_seed", "condition", "value"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    data = raw.copy()
    data["network_seed"] = pd.to_numeric(data["network_seed"], errors="raise").astype(int)
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    conditions = ["high_stsp_overlap", "matched_removal"]
    if set(data["condition"].astype(str)) != set(conditions):
        raise ValueError(f"{label}: condition set mismatch")
    counts = data.groupby(["network_seed", "condition"]).size()
    if len(counts) != 40 or not (counts == 1).all():
        raise ValueError(f"{label}: expected exactly one row per network-condition")
    condition_stats: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        group = f"recruitment_loss|{condition}"
        stat = _one_statistic(statistics, endpoint="recruitment_loss", group=group)
        values = pd.to_numeric(data.loc[data["condition"].astype(str).eq(condition), "value"], errors="raise")
        _validate_frozen_mean(values, stat, f"{label} {condition}")
        mean, low, high = _t_ci(values)
        if not (
            np.isclose(low, float(stat["ci95_low"]), rtol=0.0, atol=1e-9)
            and np.isclose(high, float(stat["ci95_high"]), rtol=0.0, atol=1e-9)
        ):
            raise ValueError(f"{label}: CI disagrees for {condition}")
        condition_stats[condition] = {"mean": mean, "ci95_low": low, "ci95_high": high, "n_networks": 20}
    piv = data.pivot_table(index="network_seed", columns="condition", values="value")
    paired = piv["high_stsp_overlap"] - piv["matched_removal"]
    stat = _one_statistic(
        statistics,
        endpoint="high_stsp_overlap_minus_matched_loss",
        contrast="high_stsp_overlap_minus_matched_loss",
        group="high_stsp_overlap_minus_matched_loss",
    )
    _validate_frozen_mean(paired, stat, f"{label} paired contrast")
    mean, low, high = _t_ci(paired)
    if not (
        np.isclose(low, float(stat["ci95_low"]), rtol=0.0, atol=1e-9)
        and np.isclose(high, float(stat["ci95_high"]), rtol=0.0, atol=1e-9)
    ):
        raise ValueError(f"{label}: paired-contrast CI disagrees with frozen row")
    if not bool((paired > 0.0).all()):
        raise ValueError(f"{label}: expected all 20 paired contrasts positive")
    paired_rows = pd.DataFrame(
        {
            "figure_id": DISPLAY_NAME,
            "panel_id": "e",
            "network_seed": list(EXPECTED_SEEDS),
            "record_type": "network_level_paired_contrast",
            "endpoint": "high_stsp_overlap_minus_matched_loss",
            "condition": "high_stsp_overlap_minus_matched_loss",
            "value": paired.to_numpy(dtype=float),
            "unit": "percent",
        }
    )
    return {
        "condition_stats": condition_stats,
        "paired": paired_rows,
        "paired_mean": mean,
        "paired_ci95": [low, high],
        "paired_frozen": {
            "estimate": float(stat["estimate"]),
            "ci95_low": float(stat["ci95_low"]),
            "ci95_high": float(stat["ci95_high"]),
            "p_adjusted": float(stat["p_adjusted"]),
        },
    }


def _draw_panel_e(
    axis: plt.Axes,
    payload: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
) -> dict[str, Any]:
    x_order = [str(value) for value in panel_spec["category_order"]]
    x_labels = panel_spec["category_labels"]
    colors = {
        condition: get_plot_color(str(panel_spec["colors"][condition]), context="manuscript_fig7")
        for condition in x_order
    }
    bar_width = float(panel_spec["bar_width"])
    decimals = int(panel_spec["value_decimals"])
    pad_units = 0.15
    label_positions: dict[str, float] = {}
    for index, condition in enumerate(x_order):
        stat = payload["condition_stats"][condition]
        mean = float(stat["mean"])
        low = float(stat["ci95_low"])
        high = float(stat["ci95_high"])
        axis.bar(
            index,
            mean,
            width=bar_width,
            color=colors[condition],
            edgecolor="none",
            zorder=2,
        )
        axis.errorbar(
            [index],
            [mean],
            yerr=[[mean - low], [high - mean]],
            fmt="none",
            ecolor=INK,
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.9,
            zorder=5,
        )
        text = axis.text(
            float(index),
            high + pad_units,
            f"{mean:.{decimals}f}",
            ha="center",
            va="bottom",
            color=INK,
            zorder=6,
            clip_on=False,
        )
        label_positions[condition] = high + pad_units
    axis.set_xlim(-0.55, len(x_order) - 0.45)
    axis.set_xticks(np.arange(len(x_order), dtype=float))
    axis.set_xticklabels([str(x_labels[condition]) for condition in x_order])
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.yaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    return {
        "label_positions": label_positions,
        "numeric_labels": {
            condition: f"{payload['condition_stats'][condition]['mean']:.{decimals}f}"
            for condition in x_order
        },
        "y_limits": list(panel_spec["y_limits"]),
        "paired_mean": payload["paired_mean"],
        "paired_ci95": payload["paired_ci95"],
    }


# ---------------------------------------------------------------- panel f


def _validate_panel_f(
    raw: pd.DataFrame,
    statistics: pd.DataFrame,
    window_robustness: pd.DataFrame,
) -> dict[str, Any]:
    label = "Fig.7f"
    _require_seed_set(raw, label)
    _require_seed_set(window_robustness, label)
    data = raw.copy()
    data["network_seed"] = pd.to_numeric(data["network_seed"], errors="raise").astype(int)
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    data["early_window_ms"] = pd.to_numeric(data["early_window_ms"], errors="raise").astype(int)
    if set(data["early_window_ms"]) != {10}:
        raise ValueError(f"{label}: raw cell file must contain only the primary 10-ms window")
    cells = ["high_nooverlap_delta", "low_nooverlap_delta", "high_overlap_delta", "low_overlap_delta"]
    if set(data["cell"].astype(str)) != set(cells):
        raise ValueError(f"{label}: cell set mismatch")
    counts = data.groupby(["network_seed", "cell"]).size()
    if len(counts) != 80 or not (counts == 1).all():
        raise ValueError(f"{label}: expected exactly one row per network-cell")
    cell_stats: dict[str, dict[str, Any]] = {}
    for cell in cells:
        group = f"early_firing_delta|{cell}"
        stat = _one_statistic(statistics, endpoint="early_firing_delta", group=group)
        values = pd.to_numeric(data.loc[data["cell"].astype(str).eq(cell), "value"], errors="raise")
        _validate_frozen_mean(values, stat, f"{label} {cell}")
        mean, low, high = _t_ci(values)
        if not (
            np.isclose(low, float(stat["ci95_low"]), rtol=0.0, atol=1e-9)
            and np.isclose(high, float(stat["ci95_high"]), rtol=0.0, atol=1e-9)
        ):
            raise ValueError(f"{label}: CI disagrees for {cell}")
        cell_stats[cell] = {"mean": mean, "ci95_low": low, "ci95_high": high, "n_networks": 20}
    for cell in ("high_nooverlap_delta", "low_nooverlap_delta"):
        if cell_stats[cell]["mean"] != 0.0:
            raise ValueError(f"{label}: {cell} must be a structural zero")
    piv = data.pivot_table(index="network_seed", columns="cell", values="value")
    interaction = (
        piv["high_overlap_delta"]
        - piv["high_nooverlap_delta"]
        - piv["low_overlap_delta"]
        + piv["low_nooverlap_delta"]
    )
    stat = _one_statistic(
        statistics,
        endpoint="overlap_gated_stsp_interaction",
        contrast="stsp_effect_with_overlap_minus_without_overlap_at_10ms",
        group="stsp_effect_with_overlap_minus_without_overlap_at_10ms",
    )
    _validate_frozen_mean(interaction, stat, f"{label} interaction")
    mean, low, high = _t_ci(interaction)
    if not (
        np.isclose(low, float(stat["ci95_low"]), rtol=0.0, atol=1e-9)
        and np.isclose(high, float(stat["ci95_high"]), rtol=0.0, atol=1e-9)
    ):
        raise ValueError(f"{label}: interaction CI disagrees with frozen row")
    if not bool((interaction > 0.0).all()):
        raise ValueError(f"{label}: expected all 20 network interactions positive")
    window_rows = window_robustness.loc[
        window_robustness["early_window_ms"].astype(int).eq(10)
    ].copy()
    if len(window_rows) != 20:
        raise ValueError(f"{label}: window-robustness file must carry 20 network interactions at 10 ms")
    interaction_frame = pd.DataFrame(
        {
            "figure_id": DISPLAY_NAME,
            "panel_id": "f",
            "network_seed": list(EXPECTED_SEEDS),
            "record_type": "network_level_interaction",
            "endpoint": "overlap_gated_stsp_interaction",
            "condition": "stsp_effect_with_overlap_minus_without_overlap_at_10ms",
            "value": interaction.to_numpy(dtype=float),
            "unit": "percentage_points",
            "early_window_ms": 10,
        }
    )
    robustness_10 = pd.to_numeric(window_rows["value"], errors="raise")
    if not np.isclose(
        float(robustness_10.mean()),
        float(stat["estimate"]),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(f"{label}: window-robustness 10-ms mean disagrees with the frozen interaction")
    return {
        "cell_stats": cell_stats,
        "interaction": interaction_frame,
        "interaction_mean": mean,
        "interaction_ci95": [low, high],
        "interaction_frozen": {
            "estimate": float(stat["estimate"]),
            "ci95_low": float(stat["ci95_low"]),
            "ci95_high": float(stat["ci95_high"]),
            "p_adjusted": float(stat["p_adjusted"]),
        },
        "structural_zeros": ["high_nooverlap_delta", "low_nooverlap_delta"],
    }


def _draw_panel_f(
    axis: plt.Axes,
    payload: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
) -> dict[str, Any]:
    from matplotlib.patches import Patch

    x_order = [str(value) for value in panel_spec["x_order"]]
    x_labels = panel_spec["x_labels"]
    hue_order = [str(value) for value in panel_spec["hue_order"]]
    hue_labels = panel_spec["hue_labels"]
    colors = {
        hue: get_plot_color(str(panel_spec["colors"][hue]), context="manuscript_fig7")
        for hue in hue_order
    }
    offset = float(panel_spec["series_offset"])
    bar_width = float(panel_spec["bar_width"])
    bar_values: dict[str, float] = {}
    for hue in hue_order:
        for x_index, x_value in enumerate(x_order):
            cell = f"{hue}_{'overlap' if x_value == 'overlap' else 'nooverlap'}_delta"
            stat = payload["cell_stats"][cell]
            mean = float(stat["mean"])
            low = float(stat["ci95_low"])
            high = float(stat["ci95_high"])
            x = float(x_index) + (offset if hue == "high" else -offset)
            axis.bar(
                x,
                mean,
                width=bar_width,
                color=colors[hue],
                edgecolor=INK,
                linewidth=0.45,
                zorder=2,
            )
            axis.errorbar(
                [x],
                [mean],
                yerr=[[mean - low], [high - mean]],
                fmt="none",
                ecolor=INK,
                elinewidth=0.9,
                capsize=2.2,
                capthick=0.9,
                zorder=4,
            )
            bar_values[f"{hue}_{x_value}"] = mean
    axis.set_xlim(-0.55, len(x_order) - 0.45)
    axis.set_xticks(np.arange(len(x_order), dtype=float))
    axis.set_xticklabels([str(x_labels[x_value]) for x_value in x_order])
    axis.set_xlabel("", labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.yaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    handles = [
        Patch(
            facecolor=colors[hue],
            edgecolor=INK,
            linewidth=0.5,
            label=str(hue_labels[hue]),
        )
        for hue in hue_order
    ]
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=int(panel_spec["legend_ncol"]),
        handlelength=1.25,
        handletextpad=0.45,
        columnspacing=1.0,
        borderaxespad=0.0,
        labelspacing=0.3,
    )
    return {
        "encoding": "grouped_bars",
        "hue_order": hue_order,
        "legend_order": [str(hue_labels[hue]) for hue in hue_order],
        "bar_values": bar_values,
        "series_offsets": {"high": offset, "low": -offset},
        "summary_lines_drawn": 0,
        "interaction_mean": payload["interaction_mean"],
        "interaction_ci95": payload["interaction_ci95"],
        "structural_zeros": payload["structural_zeros"],
    }


# ---------------------------------------------------------------- layout audit


def _layout_audit(spec: Mapping[str, Any]) -> dict[str, Any]:
    report = validate_layout_contract(spec)
    canvas_width, canvas_height = [float(value) for value in spec["canvas_mm"]]
    expected_slots = {
        "a": [2.0, 2.0, 79.5, 48.0],
        "b": [83.5, 2.0, 79.5, 48.0],
        "c": [2.0, 52.0, 79.5, 48.0],
        "d": [83.5, 52.0, 79.5, 48.0],
        "e": [2.0, 102.0, 79.5, 48.0],
        "f": [83.5, 102.0, 79.5, 48.0],
    }
    failures = list(report.failures)
    rows: list[dict[str, Any]] = []
    for panel_id, expected in expected_slots.items():
        actual = [float(value) for value in spec["slots"][panel_id]]
        if actual != expected:
            failures.append(f"panel {panel_id} slot differs from the requested geometry")
        plot = [float(value) for value in spec["panels"][panel_id]["plot_bbox_mm"]]
        slot_left, slot_top, slot_width, slot_height = actual
        plot_left, plot_top, plot_width, plot_height = plot
        inside = (
            plot_left >= slot_left
            and plot_top >= slot_top
            and plot_left + plot_width <= slot_left + slot_width
            and plot_top + plot_height <= slot_top + slot_height
        )
        if not inside:
            failures.append(f"panel {panel_id} plot area escapes slot")
        rows.append(
            {
                "panel_id": panel_id,
                "slot_left_mm": slot_left,
                "slot_top_mm": slot_top,
                "slot_width_mm": slot_width,
                "slot_height_mm": slot_height,
                "plot_left_mm": plot_left,
                "plot_top_mm": plot_top,
                "plot_width_mm": plot_width,
                "plot_height_mm": plot_height,
                "plot_inside_slot": inside,
            }
        )
    if [canvas_width, canvas_height] != [165.0, 152.0]:
        failures.append("canvas differs from 165 x 152 mm")
    if spec["slots"]["b"][0] - (spec["slots"]["a"][0] + spec["slots"]["a"][2]) != 2.0:
        failures.append("top-row gutter is not 2 mm")
    if spec["slots"]["c"][1] - (spec["slots"]["a"][1] + spec["slots"]["a"][3]) != 2.0:
        failures.append("row gutter is not 2 mm")
    if spec["slots"]["e"][1] - (spec["slots"]["c"][1] + spec["slots"]["c"][3]) != 2.0:
        failures.append("row-2/3 gutter is not 2 mm")
    for left, right in (("c", "d"), ("e", "f")):
        left_plot = spec["panels"][left]["plot_bbox_mm"]
        right_plot = spec["panels"][right]["plot_bbox_mm"]
        if left_plot[1:] != right_plot[1:]:
            failures.append(f"row {left}/{right} plot-area geometry is not aligned")
    a_children = [list(map(float, value)) for value in spec["panels"]["a"]["child_plot_bboxes_mm"]]
    b_plot = [float(value) for value in spec["panels"]["b"]["plot_bbox_mm"]]
    for child in a_children:
        if child[1] != b_plot[1] or child[3] != b_plot[3]:
            failures.append("panel a child axes must align (top/bottom) with panel b plot area")
    a_plot = [float(value) for value in spec["panels"]["a"]["plot_bbox_mm"]]
    if a_plot[1] != b_plot[1] or a_plot[3] != b_plot[3]:
        failures.append("panel a parent plot area must align (top/bottom) with panel b plot area")
    for panel_id in "abcdef":
        panel = spec["panels"][panel_id]
        if "colors" in panel:
            for role in panel["colors"].values():
                hex_color = get_plot_color(str(role), context="manuscript_fig7")
                if not str(hex_color).startswith("#") or len(str(hex_color)) != 7:
                    failures.append(f"panel {panel_id} color role {role} resolves outside the palette")
    return {
        "schema": "manuscript_fig7_candidate_layout_audit_v1",
        "status": "passed" if not failures else "failed",
        "passes": report.passes,
        "warnings": report.warnings,
        "failures": failures,
        "geometry_rows": rows,
    }


def _render_wireframe(spec: Mapping[str, Any], output: Path) -> None:
    canvas_width, canvas_height = [float(value) for value in spec["canvas_mm"]]
    with plt.rc_context({**VECTOR_TEXT_RCPARAMS, "svg.hashsalt": CANDIDATE_VERSION}):
        figure = plt.figure(figsize=(canvas_width * MM_TO_INCH, canvas_height * MM_TO_INCH), dpi=300, facecolor="white")
        axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
        axis.set_xlim(0.0, canvas_width)
        axis.set_ylim(canvas_height, 0.0)
        axis.axis("off")
        for panel_id, slot in spec["slots"].items():
            x, y, width, height = [float(value) for value in slot]
            axis.add_patch(Rectangle((x, y), width, height, facecolor="white", edgecolor=NEUTRAL_MID, linewidth=0.7))
            px, py, pw, ph = [float(value) for value in spec["panels"][panel_id]["plot_bbox_mm"]]
            axis.add_patch(Rectangle((px, py), pw, ph, facecolor=PALE, edgecolor=NEUTRAL_LIGHT, linewidth=0.6))
            text = axis.text(x + 1.0, y + 1.0, panel_id, ha="left", va="top", color=INK)
            mark_panel_label(text)
        apply_paper_figure_typography(figure)
        figure.savefig(output, dpi=300, facecolor="white", bbox_inches=None, metadata={"Date": None, "Creator": CANDIDATE_VERSION})
        plt.close(figure)


def _render_figure(spec: Mapping[str, Any], payload: Mapping[str, Any], figures_dir: Path) -> dict[str, Path]:
    canvas_mm = [float(value) for value in spec["canvas_mm"]]
    canvas_width, canvas_height = canvas_mm
    outputs = {
        "png": figures_dir / "manuscript_fig7_reader_first_v1.png",
        "svg": figures_dir / "manuscript_fig7_reader_first_v1.svg",
        "pdf": figures_dir / "manuscript_fig7_reader_first_v1.pdf",
        "base_svg": figures_dir / "qa" / "manuscript_fig7_reader_first_v1_base.svg",
    }
    panel_qa: dict[str, Any] = {}
    with plt.rc_context(
        {
            **VECTOR_TEXT_RCPARAMS,
            "svg.hashsalt": CANDIDATE_VERSION,
            "axes.unicode_minus": True,
        }
    ):
        figure = plt.figure(figsize=(canvas_width * MM_TO_INCH, canvas_height * MM_TO_INCH), dpi=300, facecolor="white")
        panel_qa["a"] = _draw_panel_a(
            figure.add_axes(_as_axes_bbox(spec["panels"]["a"]["plot_bbox_mm"], canvas_mm)),
            payload["a"],
            spec["panels"]["a"],
        )
        panel_qa["b"] = _draw_panel_b(
            figure.add_axes(_as_axes_bbox(spec["panels"]["b"]["plot_bbox_mm"], canvas_mm)),
            payload["b"],
            spec["panels"]["b"],
        )
        panel_qa["c"] = _draw_panel_c(
            figure.add_axes(_as_axes_bbox(spec["panels"]["c"]["plot_bbox_mm"], canvas_mm)),
            payload["c"],
            spec["panels"]["c"],
        )
        panel_qa["d"] = _draw_panel_d(
            figure,
            figure.add_axes(_as_axes_bbox(spec["panels"]["d"]["plot_bbox_mm"], canvas_mm)),
            payload["d"],
            spec["panels"]["d"],
        )
        panel_qa["e"] = _draw_panel_e(
            figure.add_axes(_as_axes_bbox(spec["panels"]["e"]["plot_bbox_mm"], canvas_mm)),
            payload["e"],
            spec["panels"]["e"],
        )
        panel_qa["f"] = _draw_panel_f(
            figure.add_axes(_as_axes_bbox(spec["panels"]["f"]["plot_bbox_mm"], canvas_mm)),
            payload["f"],
            spec["panels"]["f"],
        )
        for panel_id, slot in spec["slots"].items():
            slot_x, slot_y, _, _ = [float(value) for value in slot]
            panel_label = figure.text(
                (slot_x + 0.3) / canvas_width,
                1.0 - (slot_y + 0.6) / canvas_height,
                panel_id,
                ha="left",
                va="top",
                color=INK,
                zorder=100,
            )
            mark_panel_label(panel_label)
        apply_paper_figure_typography(figure)
        metadata_svg = {"Date": None, "Creator": CANDIDATE_VERSION}
        metadata_pdf = {"Creator": CANDIDATE_VERSION, "CreationDate": None}
        figure.savefig(outputs["svg"], format="svg", facecolor="white", bbox_inches=None, metadata=metadata_svg)
        figure.savefig(outputs["pdf"], format="pdf", facecolor="white", bbox_inches=None, metadata=metadata_pdf)
        figure.savefig(outputs["png"], format="png", dpi=300, facecolor="white", bbox_inches=None, metadata={"Software": CANDIDATE_VERSION})
        plt.close(figure)
    expected_pixels = tuple(int(round(float(value) * 300.0 / 25.4)) for value in canvas_mm)
    with Image.open(outputs["png"]) as image:
        if image.size != expected_pixels:
            resized = image.convert("RGB").resize(expected_pixels, Image.Resampling.LANCZOS)
            resized.save(outputs["png"], dpi=(300, 300))
            resized.close()
    outputs["base_svg"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(outputs["svg"], outputs["base_svg"])
    return {"png": outputs["png"], "svg": outputs["svg"], "pdf": outputs["pdf"], "base_svg": outputs["base_svg"], "panel_qa": panel_qa}


def _render_qa(outputs: Mapping[str, Path], spec: Mapping[str, Any], panel_qa: Mapping[str, Any]) -> dict[str, Any]:
    expected_size = tuple(int(round(float(value) * 300.0 / 25.4)) for value in spec["canvas_mm"])
    with Image.open(outputs["png"]) as image:
        actual_size = image.size
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        border = 8
        border_pixels = np.concatenate(
            [rgb[:border].reshape(-1, 3), rgb[-border:].reshape(-1, 3), rgb[:, :border].reshape(-1, 3), rgb[:, -border:].reshape(-1, 3)],
            axis=0,
        )
        outer_border_clear = bool(np.all(border_pixels >= 250))
    svg_text = outputs["svg"].read_text(encoding="utf-8")
    svg_lower = svg_text.lower()
    pdf_reader = PdfReader(str(outputs["pdf"]))
    page = pdf_reader.pages[0]
    width_pt = float(page.mediabox.width)
    height_pt = float(page.mediabox.height)
    extracted_text = page.extract_text() or ""
    resources = page.get("/Resources")
    font_table = resources.get("/Font") if resources else None
    if font_table is not None and hasattr(font_table, "get_object"):
        font_table = font_table.get_object()
    font_count = len(font_table) if font_table else 0
    text_count = svg_text.count("<text")
    path_count = svg_text.count("<path")
    image_count = svg_text.count("<image")
    numeric_labels = []
    numeric_labels += list(panel_qa["c"]["numeric_labels"].values())
    numeric_labels += list(panel_qa["e"]["numeric_labels"].values())
    checks = {
        "png_dimensions": all(abs(actual - expected) <= 1 for actual, expected in zip(actual_size, expected_size)),
        "outer_border_clear": outer_border_clear,
        "svg_editable_text": text_count > 0,
        "svg_has_vector_paths": path_count > 0,
        "svg_bitmap_limited_to_heatmap_mesh": image_count <= 1,
        "svg_has_c_numeric_labels": all(label in svg_text for label in panel_qa["c"]["numeric_labels"].values()),
        "svg_has_e_numeric_labels": all(label in svg_text for label in panel_qa["e"]["numeric_labels"].values()),
        "svg_no_internal_bundle_label": "fig6" not in svg_lower,
        "svg_no_old_c_condition_label": "same-label" not in svg_lower,
        "pdf_one_page": len(pdf_reader.pages) == 1,
        "pdf_width_mm": math.isclose(width_pt / MM_TO_POINT, float(spec["canvas_mm"][0]), abs_tol=0.25),
        "pdf_height_mm": math.isclose(height_pt / MM_TO_POINT, float(spec["canvas_mm"][1]), abs_tol=0.25),
        "pdf_embedded_font_resources": font_count > 0,
        "pdf_panel_labels_present": all(letter in extracted_text for letter in "abcdef"),
    }
    return {
        "schema": "manuscript_fig7_candidate_render_qa_v1",
        "generated_at": _utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "numeric_labels_checked": numeric_labels,
        "png": {"path": str(outputs["png"]), "pixels": list(actual_size), "expected_pixels_at_300_dpi": list(expected_size), "sha256": _sha256(outputs["png"]), "bytes": outputs["png"].stat().st_size},
        "svg": {"path": str(outputs["svg"]), "text_elements": text_count, "path_elements": path_count, "sha256": _sha256(outputs["svg"]), "bytes": outputs["svg"].stat().st_size},
        "pdf": {"path": str(outputs["pdf"]), "pages": len(pdf_reader.pages), "page_mm": [width_pt / MM_TO_POINT, height_pt / MM_TO_POINT], "font_resources": font_count, "sha256": _sha256(outputs["pdf"]), "bytes": outputs["pdf"].stat().st_size},
    }


def _grayscale_audit(outputs: Mapping[str, Path], figures_dir: Path) -> dict[str, Any]:
    grayscale_path = figures_dir / "qa" / "manuscript_fig7_reader_first_v1_grayscale.png"
    with Image.open(outputs["png"]) as image:
        image.convert("L").save(grayscale_path, dpi=(300, 300))
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    checks = {
        "grayscale_exists": grayscale_path.is_file(),
        "grayscale_has_dark_marks": bool((gray < 180).any()),
        "grayscale_has_midtones": bool(((gray >= 80) & (gray < 245)).any()),
    }
    return {
        "schema": "manuscript_fig7_candidate_grayscale_audit_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "path": str(grayscale_path),
    }


def _visual_qa(outputs: Mapping[str, Path], spec: Mapping[str, Any], panel_qa: Mapping[str, Any], figures_dir: Path) -> dict[str, Any]:
    with Image.open(outputs["png"]) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        canvas_width, canvas_height = [float(value) for value in spec["canvas_mm"]]
        panel_coverage: dict[str, float] = {}
        for panel_id, slot in spec["slots"].items():
            x, y, width, height = [float(value) for value in slot]
            left = int(round(x / canvas_width * image.width))
            upper = int(round(y / canvas_height * image.height))
            right = int(round((x + width) / canvas_width * image.width))
            lower = int(round((y + height) / canvas_height * image.height))
            block = rgb[upper:lower, left:right]
            panel_coverage[panel_id] = float((block.min(axis=2) < 245).mean())
    crops: list[dict[str, Any]] = []
    panels_dir = figures_dir / "qa" / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(outputs["png"]) as image:
        for panel_id, slot in spec["slots"].items():
            x, y, width, height = [float(value) for value in slot]
            left = int(round(x / canvas_width * image.width))
            upper = int(round(y / canvas_height * image.height))
            right = int(round((x + width) / canvas_width * image.width))
            lower = int(round((y + height) / canvas_height * image.height))
            crop_path = panels_dir / f"manuscript_fig7{panel_id}.png"
            image.crop((left, upper, right, lower)).save(crop_path, dpi=(300, 300))
            crops.append({"panel": panel_id, "path": str(crop_path), "pixels": [right - left, lower - upper]})
    c_label_positions = panel_qa["c"]["label_positions"]
    c_label_offsets = panel_qa["c"]["label_offsets"]
    c_y_max = float(panel_qa["c"]["y_limits"][1])
    e_label_positions = panel_qa["e"]["label_positions"]
    e_y_max = float(panel_qa["e"]["y_limits"][1])
    text_height_units_c = 0.62
    text_height_units_e = 0.30
    checks = {
        "a_legend_order": panel_qa["a"]["legend_order"] == ["No memory", "Item A", "Item B", "Pair"],
        "b_legend_order": panel_qa["b"]["legend_order"] == ["Sequence", "Singleton", "Cue only"],
        "b_foreground": panel_qa["b"]["foreground_endpoint"] == "P_target_sequence_state",
        "a_foreground": panel_qa["a"]["foreground_condition"] == "S_AB",
        "c_no_network_points": panel_qa["c"]["network_points_drawn"] == 0,
        "c_labels_clear_of_top": all(label_y + text_height_units_c < c_y_max for label_y in c_label_positions.values()),
        "c_labels_close_to_ci": all(
            abs(c_label_offsets[condition] - float(spec["panels"]["c"]["label_pad_units"])) < 1e-9
            for condition in c_label_offsets
        ),
        "e_labels_clear_of_top": all(label_y + text_height_units_e < e_y_max for label_y in e_label_positions.values()),
        "f_encoding_bars": panel_qa["f"]["encoding"] == "grouped_bars",
        "f_no_summary_lines": panel_qa["f"]["summary_lines_drawn"] == 0,
        "f_interaction_positive": panel_qa["f"]["interaction_mean"] > 0.0,
        "f_structural_zeros_declared": set(panel_qa["f"]["structural_zeros"]) == {"high_nooverlap_delta", "low_nooverlap_delta"},
        "row_1_plot_area_aligned": spec["panels"]["a"]["plot_bbox_mm"][1] == spec["panels"]["b"]["plot_bbox_mm"][1]
        and spec["panels"]["a"]["plot_bbox_mm"][3] == spec["panels"]["b"]["plot_bbox_mm"][3],
        "row_2_plot_area_aligned": spec["panels"]["c"]["plot_bbox_mm"][1:] == spec["panels"]["d"]["plot_bbox_mm"][1:],
        "row_3_plot_area_aligned": spec["panels"]["e"]["plot_bbox_mm"][1:] == spec["panels"]["f"]["plot_bbox_mm"][1:],
        "panel_coverage_nonzero": all(value > 0.01 for value in panel_coverage.values()),
    }
    return {
        "schema": "manuscript_fig7_candidate_visual_qa_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "final_size_mm": spec["canvas_mm"],
        "panel_ink_coverage": panel_coverage,
        "panel_qa": panel_qa,
        "panel_crops": crops,
        "manual_review_targets": [
            "inspect a: dual Target A/B axes, legend crowding, Pair visual foreground with readable controls",
            "inspect b: legend order and line z-order (Sequence on top)",
            "inspect c: no network points; the 5.05 and 42.45 labels sit close to the bar/CI tops without touching the top edge",
            "inspect d: colorbar, cell labels and heatmap geometry are not distorted",
            "inspect e: two-line-free x labels and value labels do not collide",
            "inspect f: grouped bars (High STSP / Low STSP per overlap condition) with 95% CIs; zero-overlap bars are structural zeros; no markers or connecting lines",
            "inspect row plot-region, baseline and panel-letter alignment; grayscale and clipping",
        ],
    }


def _determinism_check(spec: Mapping[str, Any], payload: Mapping[str, Any], figures_dir: Path, outputs: Mapping[str, Path]) -> dict[str, Any]:
    rerender_dir = figures_dir / "qa" / "rerender"
    rerender_dir.mkdir(parents=True, exist_ok=True)
    rerendered = _render_figure(spec, payload, rerender_dir)
    identical = {
        name: _sha256(outputs[name]) == _sha256(rerendered[name])
        for name in ("png", "svg", "pdf")
    }
    return {
        "schema": "manuscript_fig7_candidate_render_determinism_v1",
        "status": "passed" if all(identical.values()) else "failed",
        "identical": identical,
        "rerender_dir": str(rerender_dir),
    }


def _write_metrics(payload: Mapping[str, Any], output_dir: Path) -> None:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    a_rows = []
    for key, value in payload["a"]["contrast_stats"].items():
        endpoint, target = key.split("|")
        a_rows.append(
            {
                "figure_id": DISPLAY_NAME,
                "panel_id": "a",
                "endpoint": endpoint,
                "target_item": target,
                "n_networks": value["n_networks"],
                "mean": value["mean"],
                "ci95_low": value["ci95_low"],
                "ci95_high": value["ci95_high"],
                "p_adjusted_frozen": value["p_adjusted"],
                "unit": "normalized_auc_gain",
            }
        )
    pd.DataFrame(a_rows).to_csv(metrics_dir / "panel_a_auc_contrast_summary.csv", index=False)

    b_frozen = payload["b"]["gain_frozen"]
    pd.DataFrame(
        [
            {
                "figure_id": DISPLAY_NAME,
                "panel_id": "b",
                "endpoint": "mean_sequence_minus_singleton_access_gain",
                "n_networks": 20,
                "mean": payload["b"]["network_gain_mean"],
                "ci95_low": payload["b"]["network_gain_ci95"][0],
                "ci95_high": payload["b"]["network_gain_ci95"][1],
                "frozen_estimate": b_frozen["estimate"],
                "frozen_ci95_low": b_frozen["ci95_low"],
                "frozen_ci95_high": b_frozen["ci95_high"],
                "p_adjusted_frozen": b_frozen["p_adjusted"],
                "unit": "percent",
            }
        ]
    ).to_csv(metrics_dir / "panel_b_network_gain_summary.csv", index=False)

    c_rows = []
    for key, value in payload["c"]["contrast_stats"].items():
        c_rows.append(
            {
                "figure_id": DISPLAY_NAME,
                "panel_id": "c",
                "endpoint": key,
                "n_networks": value["n_networks"],
                "mean": value["mean"],
                "ci95_low": value["ci95_low"],
                "ci95_high": value["ci95_high"],
                "positive_networks": value["positive_networks"],
                "min": value["min"],
                "max": value["max"],
                "p_adjusted_frozen": value["p_adjusted"],
                "unit": "percent",
                "aggregation": "equal-weight mean over 7 serial positions per network before the paired difference",
            }
        )
    pd.DataFrame(c_rows).to_csv(metrics_dir / "panel_c_contrast_summary.csv", index=False)
    payload["c"]["absolute_means"].to_csv(metrics_dir / "panel_c_absolute_condition_means.csv", index=False)

    d = payload["d"]["interaction"]
    pd.DataFrame(
        [
            {
                "figure_id": DISPLAY_NAME,
                "panel_id": "d",
                "endpoint": "standardized_seq_len_x_delay_interaction",
                "n_networks": d["n_networks"],
                "estimate_frozen": d["estimate"],
                "ci95_low_frozen": d["ci95_low"],
                "ci95_high_frozen": d["ci95_high"],
                "p_value_frozen": d["p_value"],
                "unit": "standardized_interaction_coefficient",
                "claim": "functional rescue depends jointly on sequence length and delay; no monotonic claim",
            }
        ]
    ).to_csv(metrics_dir / "panel_d_interaction_summary.csv", index=False)

    e_rows = []
    for key, value in payload["e"]["condition_stats"].items():
        e_rows.append(
            {
                "figure_id": DISPLAY_NAME,
                "panel_id": "e",
                "condition": key,
                "n_networks": value["n_networks"],
                "mean": value["mean"],
                "ci95_low": value["ci95_low"],
                "ci95_high": value["ci95_high"],
                "unit": "percent",
            }
        )
    e_rows.append(
        {
            "figure_id": DISPLAY_NAME,
            "panel_id": "e",
            "condition": "high_stsp_overlap_minus_matched_loss",
            "n_networks": 20,
            "mean": payload["e"]["paired_mean"],
            "ci95_low": payload["e"]["paired_ci95"][0],
            "ci95_high": payload["e"]["paired_ci95"][1],
            "unit": "percent",
        }
    )
    pd.DataFrame(e_rows).to_csv(metrics_dir / "panel_e_summary.csv", index=False)

    f_rows = []
    for key, value in payload["f"]["cell_stats"].items():
        f_rows.append(
            {
                "figure_id": DISPLAY_NAME,
                "panel_id": "f",
                "cell": key,
                "n_networks": value["n_networks"],
                "mean": value["mean"],
                "ci95_low": value["ci95_low"],
                "ci95_high": value["ci95_high"],
                "unit": "percentage_points",
                "structural_zero": key in payload["f"]["structural_zeros"],
            }
        )
    f_interaction = payload["f"]["interaction_frozen"]
    f_rows.append(
        {
            "figure_id": DISPLAY_NAME,
            "panel_id": "f",
            "cell": "overlap_gated_stsp_interaction",
            "n_networks": 20,
            "mean": payload["f"]["interaction_mean"],
            "ci95_low": payload["f"]["interaction_ci95"][0],
            "ci95_high": payload["f"]["interaction_ci95"][1],
            "unit": "percentage_points",
            "structural_zero": False,
        }
    )
    pd.DataFrame(f_rows).to_csv(metrics_dir / "panel_f_summary.csv", index=False)

    cross_rows = []
    for key, value in payload["a"]["auc_checks"].items():
        endpoint, target = key.split("|")
        cross_rows.append(
            {
                "panel": "a",
                "group": key,
                "materialized_mean": value["mean"],
                "materialized_ci95": [value["ci95"][0], value["ci95"][1]],
                "match": True,
            }
        )
    cross_rows.append(
        {
            "panel": "b",
            "group": "mean_sequence_minus_singleton_access_gain_vs_zero",
            "materialized_mean": payload["b"]["network_gain_mean"],
            "materialized_ci95": payload["b"]["network_gain_ci95"],
            "match": True,
        }
    )
    for key, value in payload["c"]["contrast_stats"].items():
        cross_rows.append(
            {
                "panel": "c",
                "group": key,
                "materialized_mean": value["mean"],
                "materialized_ci95": [value["ci95_low"], value["ci95_high"]],
                "match": True,
            }
        )
    cross_rows.append(
        {
            "panel": "e",
            "group": "high_stsp_overlap_minus_matched_loss",
            "materialized_mean": payload["e"]["paired_mean"],
            "materialized_ci95": payload["e"]["paired_ci95"],
            "match": True,
        }
    )
    cross_rows.append(
        {
            "panel": "f",
            "group": "overlap_gated_stsp_interaction_at_10ms",
            "materialized_mean": payload["f"]["interaction_mean"],
            "materialized_ci95": payload["f"]["interaction_ci95"],
            "match": True,
        }
    )
    pd.DataFrame(cross_rows).to_csv(metrics_dir / "candidate_cross_checks.csv", index=False)


def _candidate_source_mapping(reader: BundleReader, parent_dir: Path, spec: Mapping[str, Any]) -> pd.DataFrame:
    accesses = reader.access_frame()
    rows: list[dict[str, Any]] = []
    for panel_id, data_rel, stats_rel, mapping in (
        ("a", "data/panel_a_plot_data.csv", "metrics/panel_a_statistics.csv", "56 frozen network-curve cells and frozen AUC contrasts per target; means and Student t 95% CIs cross-checked to the frozen statistics"),
        ("b", "data/panel_b_absolute_access.csv", "metrics/panel_b_absolute_access_statistics.csv", "30 frozen endpoint-by-position cells; network-level cross-position mean gain cross-checked to the frozen mean_sequence_minus_singleton_access_gain_vs_zero row"),
        ("c", "data/panel_c_position_profiles.csv", "metrics/panel_c_statistics.csv", "strict 20-network x 3-condition x 7-position rows; per-network equal-weight position mean; paired contrasts matched-minus-same-class-novel and matched-minus-unseen; means, CIs and ranges cross-checked to the frozen statistics"),
        ("d", "data/panel_d_plot_data.csv", "metrics/panel_d_statistics.csv", "16 frozen K x delay cells; frozen standardized sequence-length-by-delay interaction reported without a monotonic claim"),
        ("e", "data/panel_e_plot_data.csv", "metrics/panel_e_statistics.csv", "20 networks x 2 conditions; paired high-STSP-overlap-minus-matched contrast cross-checked to the frozen row"),
        ("f", "data/panel_f_plot_data.csv", "metrics/panel_f_statistics.csv", "primary 10-ms 2x2 cells; structural zero cells verified; overlap-by-STSP interaction cross-checked to the frozen row; window robustness remains in Supplementary Fig. S7"),
    ):
        data_access = accesses.loc[accesses["relative_path"].eq(data_rel)].iloc[0]
        stats_access = accesses.loc[accesses["relative_path"].eq(stats_rel)].iloc[0]
        rows.append(
            {
                "candidate_figure": DISPLAY_NAME,
                "candidate_panel": panel_id,
                "parent_bundle": str(parent_dir),
                "parent_data_path": data_rel,
                "parent_data_sha256": data_access["sha256"],
                "parent_statistics_path": stats_rel,
                "parent_statistics_sha256": stats_access["sha256"],
                "independent_unit": "independently trained network",
                "included_seeds": "1000-1019",
                "mapping": mapping,
                "forward_replay": False,
            }
        )
    return pd.DataFrame(rows)


def _minus(value: Any) -> str:
    numeric = float(value)
    return f"\u2212{abs(numeric):.4f}" if numeric < 0 else f"{numeric:.4f}"


def _format_p(value: Any) -> str:
    numeric = float(value)
    if numeric == 0.0:
        return "<1e-300"
    exponent = math.floor(math.log10(abs(numeric)))
    mantissa = numeric / (10.0**exponent)
    return f"{mantissa:.2f} × 10^{exponent}"


def _caption(payload: Mapping[str, Any]) -> str:
    a = payload["a"]["contrast_stats"]
    b = payload["b"]["gain_frozen"]
    c = payload["c"]["contrast_stats"]
    d = payload["d"]["interaction"]
    e = payload["e"]["paired_frozen"]
    f = payload["f"]["interaction_frozen"]
    abs_means = payload["c"]["validation"]["absolute_position_means"]

    a_p_s0 = _format_p(a["SAB_vs_S0_auc_gain|A"]["p_adjusted"])
    a_p_single_a = _format_p(a["SAB_vs_relevant_single_auc_gain|A"]["p_adjusted"])
    a_p_single_b = _format_p(a["SAB_vs_relevant_single_auc_gain|B"]["p_adjusted"])
    b_p = _format_p(b["p_adjusted"])
    c_p_novel = _format_p(c["matched_minus_same_label_novel"]["p_adjusted"])
    c_p_unseen = _format_p(c["matched_minus_unseen"]["p_adjusted"])
    d_p = _format_p(d["p_value"])
    e_p = _format_p(e["p_adjusted"])
    f_p = _format_p(f["p_adjusted"])
    return (
        "**Fig.7 | Retained STSP influences later processing under matched conditions.**\n\n"
        f"**a,** Partial-cue target recovery for A and B across keep probability under no-memory, singleton and pair-state conditions. "
        f"The pair state is the only tested state that jointly raises recovery of both constituents: cue-strength-integrated target readout under the pair state exceeded the cue-only no-memory reference (normalized AUC gain: A, {a['SAB_vs_S0_auc_gain|A']['mean']:.3f} [95% CI, {a['SAB_vs_S0_auc_gain|A']['ci95_low']:.3f}–{a['SAB_vs_S0_auc_gain|A']['ci95_high']:.3f}]; B, {a['SAB_vs_S0_auc_gain|B']['mean']:.3f} [{a['SAB_vs_S0_auc_gain|B']['ci95_low']:.3f}–{a['SAB_vs_S0_auc_gain|B']['ci95_high']:.3f}]; both BH-adjusted P = {a_p_s0}) and the corresponding singleton state (A, {a['SAB_vs_relevant_single_auc_gain|A']['mean']:.4f} [{a['SAB_vs_relevant_single_auc_gain|A']['ci95_low']:.4f}–{a['SAB_vs_relevant_single_auc_gain|A']['ci95_high']:.4f}]; B, {a['SAB_vs_relevant_single_auc_gain|B']['mean']:.4f} [{a['SAB_vs_relevant_single_auc_gain|B']['ci95_low']:.4f}–{a['SAB_vs_relevant_single_auc_gain|B']['ci95_high']:.4f}]; BH-adjusted P = {a_p_single_a} and {a_p_single_b}). "
        f"The pair advantage is therefore joint access, not a large pair-versus-singleton margin at every cue strength.\n\n"
        f"**b,** Cue-only, slot-matched singleton and sequence-state target access across serial positions at sequence length K = 10, a 400-ms delay and keep probability 0.5. "
        f"The sequence state raised target access relative to the slot-matched singleton state (network-level mean gain across positions, {b['estimate']:.2f}% [95% CI, {b['ci95_low']:.2f}–{b['ci95_high']:.2f}%]; BH-adjusted P = {b_p}).\n\n"
        f"**c,** Network-level matched-cue gains against two controls: matched cues outperformed same-label novel cues by {c['matched_minus_same_label_novel']['mean']:.2f}% [95% CI, {c['matched_minus_same_label_novel']['ci95_low']:.2f}–{c['matched_minus_same_label_novel']['ci95_high']:.2f}%] and unseen cues by {c['matched_minus_unseen']['mean']:.2f}% [{c['matched_minus_unseen']['ci95_low']:.2f}–{c['matched_minus_unseen']['ci95_high']:.2f}%] (BH-adjusted P = {c_p_novel} and {c_p_unseen}); both contrasts were positive in all 20 networks, and each bar is the equal-weight position average of the paired difference per network. "
        f"The same-label novel cue is an out-of-sequence exemplar of a class present in the sequence; position-averaged absolute target access was matched {abs_means['matched']:.2f}%, same-label novel {abs_means['same_label_novel']:.2f}% and unseen {abs_means['unseen']:.2f}%. "
        f"History-supported content is the dominant cue boundary, while the exact experienced exemplar adds a smaller advantage within the same class; the two gains differ in magnitude and are not a single all-or-none cue-specificity effect.\n\n"
        f"**d,** Rescued fraction across sequence length and delay. Functional rescue depends jointly on sequence length and retention delay rather than varying monotonically with either alone; the frozen within-network standardized sequence-length-by-delay interaction was {_minus(d['estimate']):s} [95% CI, {_minus(d['ci95_low']):s} to {_minus(d['ci95_high']):s}] (two-sided one-sample t test, unadjusted P = {d_p}).\n\n"
        f"**e,** Recruitment loss after removal of the high-STSP-overlap contribution ({payload['e']['condition_stats']['high_stsp_overlap']['mean']:.2f}%) and after area- and energy-matched removal ({payload['e']['condition_stats']['matched_removal']['mean']:.2f}%). "
        f"The paired contrast was {e['estimate']:.2f}% [95% CI, {e['ci95_low']:.2f}–{e['ci95_high']:.2f}%] (BH-adjusted P = {e_p}); removal establishes a targeted contribution, not sole encoding.\n\n"
        f"**f,** Firing change in the primary 10-ms two-by-two analysis of high or low retained STSP and zero-overlap or overlap pathways; zero-overlap cells are structural zeros of the endpoint construction. "
        f"The high-STSP effect appeared only along overlapping pathways (high STSP: {payload['f']['cell_stats']['high_overlap_delta']['mean']:.2f}% [95% CI, {payload['f']['cell_stats']['high_overlap_delta']['ci95_low']:.2f}–{payload['f']['cell_stats']['high_overlap_delta']['ci95_high']:.2f}%] with overlap versus 0% without; low STSP: {payload['f']['cell_stats']['low_overlap_delta']['mean']:.2f}% [{payload['f']['cell_stats']['low_overlap_delta']['ci95_low']:.2f}–{payload['f']['cell_stats']['low_overlap_delta']['ci95_high']:.2f}%]), yielding an overlap-by-STSP interaction of {f['estimate']:.3f} percentage points [95% CI, {f['ci95_low']:.3f}–{f['ci95_high']:.3f} pp] (BH-adjusted P = {f_p}). Window-robustness analyses remain in Supplementary Fig. S7.\n\n"
        "Lines, bars, cells and points show means across n = 20 independently trained networks (seeds 1000–1019); error bars and bands show two-sided 95% Student t CIs. "
        "Planned contrasts in a–c, e and f used two-sided one-sample t tests with Benjamini–Hochberg adjustment; the standardized interaction in d used a two-sided one-sample t test. "
        "Access here means history-supported readout under an incoming cue, not cue-free replay, perfect or unlimited recall; the interaction in f is reported in percentage points even though the artwork axis is labelled in percent."
    )


def _resolved_spec(spec: Mapping[str, Any], reader: BundleReader) -> dict[str, Any]:
    resolved = json.loads(json.dumps(spec))
    resolved["resolved_at"] = _utc_now()
    resolved["resolved_colors"] = {
        "ink": INK,
        "neutral_mid": NEUTRAL_MID,
        "neutral_light": NEUTRAL_LIGHT,
    }
    resolved["resolved_parent_sources"] = reader.access_frame().to_dict("records")
    return resolved


def _write_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(output_dir).as_posix()
        role = relative.split("/", 1)[0] if "/" in relative else "artifact"
        artifacts.append({"path": relative, "sha256": _sha256(path), "bytes": path.stat().st_size, "role": role})
    manifest = {
        "schema": "paper_figure_reader_first_candidate_manifest_v1",
        "candidate_version": CANDIDATE_VERSION,
        "display_name": DISPLAY_NAME,
        "generated_at": _utc_now(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def build_candidate(*, parent_dir: Path, output_dir: Path, check_only: bool) -> dict[str, Any]:
    spec = _load_spec()
    repo_root = _repo_root().resolve()
    expected_parent = (repo_root / spec["parent_bundle"]).resolve()
    parent_dir = parent_dir.resolve()
    output_dir = output_dir.resolve()
    if _inside(output_dir, parent_dir) or _inside(parent_dir, output_dir):
        raise ValueError("candidate output and pinned parent source must be separate trees")
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    (output_dir / "figures" / "qa" / "panels").mkdir(parents=True, exist_ok=True)
    if parent_dir != expected_parent:
        raise ValueError("--parent-dir does not match the parent pinned by the candidate spec")

    parent_before = _snapshot_tree(parent_dir, "frozen_parent_bundle")
    before_digest = _snapshot_digest(parent_before)
    reader = BundleReader(parent_dir, expected_parent, set(PARENT_DATA_FILES))
    payload = _load_sources(reader, spec)
    layout_audit = _layout_audit(spec)
    if layout_audit["status"] != "passed":
        raise ValueError(f"candidate layout contract failed: {layout_audit['failures']}")

    payload["a"]["curves"].to_csv(output_dir / "data" / "panel_a_plot_data.csv", index=False)
    payload["a"]["contrasts"].to_csv(output_dir / "data" / "panel_a_auc_contrasts.csv", index=False)
    payload["b"]["curves"].to_csv(output_dir / "data" / "panel_b_plot_data.csv", index=False)
    payload["b"]["network_gain"].to_csv(output_dir / "data" / "panel_b_network_gain.csv", index=False)
    payload["c"]["contrasts"].to_csv(output_dir / "data" / "panel_c_network_contrasts.csv", index=False)
    payload["c"]["absolute_means"].to_csv(output_dir / "data" / "panel_c_absolute_condition_means.csv", index=False)
    payload["d"]["cells"].to_csv(output_dir / "data" / "panel_d_plot_data.csv", index=False)
    payload["e"]["paired"].to_csv(output_dir / "data" / "panel_e_plot_data.csv", index=False)
    payload["f"]["interaction"].to_csv(output_dir / "data" / "panel_f_plot_data.csv", index=False)

    for name, frame in payload["manifests"].items():
        target_name = "source_manifest.csv" if name == "source_manifest.csv" else name
        output = _relabel(frame, "all")
        if "candidate_figure" in output.columns:
            output["candidate_figure"] = DISPLAY_NAME
        else:
            output.insert(0, "candidate_figure", DISPLAY_NAME)
        output.to_csv(output_dir / "meta" / target_name, index=False)
    reader.access_frame().to_csv(output_dir / "meta" / "plot_source_access.csv", index=False)
    _candidate_source_mapping(reader, parent_dir, spec).to_csv(output_dir / "meta" / "source_mapping.csv", index=False)
    parent_before.to_csv(output_dir / "meta" / "parent_hashes_before.csv", index=False)
    pd.DataFrame(layout_audit["geometry_rows"]).to_csv(output_dir / "meta" / "layout_measurements.csv", index=False)
    _write_json(output_dir / "meta" / "layout_audit.json", layout_audit)
    _write_json(output_dir / "meta" / "panel_c_pair_validation.json", payload["c"]["validation"])
    _write_json(output_dir / "meta" / "parent_artifact_manifest.json", payload["parent_artifact_manifest"])
    _write_json(output_dir / "meta" / "final_plot_spec.json", _resolved_spec(spec, reader))
    _write_json(output_dir / "meta" / "review_only_candidate_spec.json", spec)
    shutil.copyfile(
        output_dir / "meta" / "review_only_candidate_spec.json",
        output_dir / "meta" / "plot_spec.json",
    )
    _write_metrics(payload, output_dir)

    (output_dir / "caption_draft.md").write_text(_caption(payload), encoding="utf-8")

    outputs: dict[str, Path] = {}
    render_qa: dict[str, Any] | None = None
    grayscale_qa: dict[str, Any] | None = None
    visual_qa: dict[str, Any] | None = None
    determinism: dict[str, Any] | None = None
    panel_qa: dict[str, Any] = {}
    if not check_only:
        _render_wireframe(spec, output_dir / "figures" / "qa" / "manuscript_fig7_reader_first_v1_wireframe.png")
        rendered = _render_figure(spec, payload, output_dir / "figures")
        outputs = {key: value for key, value in rendered.items() if key in {"png", "svg", "pdf"}}
        panel_qa = rendered["panel_qa"]
        render_qa = _render_qa(outputs, spec, panel_qa)
        _write_json(output_dir / "meta" / "render_qa.json", render_qa)
        if render_qa["status"] != "passed":
            raise ValueError(f"candidate render QA failed: {render_qa['checks']}")
        grayscale_qa = _grayscale_audit(outputs, output_dir / "figures")
        _write_json(output_dir / "meta" / "grayscale_audit.json", grayscale_qa)
        if grayscale_qa["status"] != "passed":
            raise ValueError(f"grayscale QA failed: {grayscale_qa['checks']}")
        visual_qa = _visual_qa(outputs, spec, panel_qa, output_dir / "figures")
        _write_json(output_dir / "meta" / "visual_qa.json", visual_qa)
        if visual_qa["status"] != "passed":
            raise ValueError(f"visual QA failed: {visual_qa['checks']}")
        determinism = _determinism_check(spec, payload, output_dir / "figures", outputs)
        _write_json(output_dir / "meta" / "render_determinism.json", determinism)
        if determinism["status"] != "passed":
            raise ValueError(f"deterministic rerender failed: {determinism['identical']}")

    parent_after = _snapshot_tree(parent_dir, "frozen_parent_bundle")
    after_digest = _snapshot_digest(parent_after)
    parent_after.to_csv(output_dir / "meta" / "parent_hashes_after.csv", index=False)
    parent_unchanged = parent_before.equals(parent_after)
    parent_integrity = {
        "schema": "manuscript_fig7_candidate_parent_integrity_v1",
        "status": "passed" if parent_unchanged else "failed",
        "parent_root": str(parent_dir),
        "file_count_before": int(len(parent_before)),
        "file_count_after": int(len(parent_after)),
        "snapshot_sha256_before": before_digest,
        "snapshot_sha256_after": after_digest,
        "unchanged": parent_unchanged,
    }
    _write_json(output_dir / "meta" / "parent_integrity.json", parent_integrity)
    if not parent_unchanged:
        raise RuntimeError("pinned parent bundle changed during candidate rendering")

    run_config = {
        "candidate_version": CANDIDATE_VERSION,
        "display_name": DISPLAY_NAME,
        "plot_only": True,
        "check_only": bool(check_only),
        "parent_bundle": str(parent_dir),
        "output_dir": str(output_dir),
        "expected_networks": list(EXPECTED_SEEDS),
        "independent_unit": "independently trained network",
        "source_policy": "read-only persisted source data and frozen statistics",
        "model_or_dataset_initialized": False,
        "generated_at": _utc_now(),
        "script": str(Path(__file__).resolve()),
        "spec": str(SPEC_PATH),
    }
    _write_json(output_dir / "run_config.json", run_config)
    summary = {
        "schema": "paper_figure_reader_first_candidate_summary_v1",
        "candidate_version": CANDIDATE_VERSION,
        "display_name": DISPLAY_NAME,
        "status": "check_passed" if check_only else "rendered",
        "canvas_mm": spec["canvas_mm"],
        "independent_unit": "independently trained network",
        "n_networks": 20,
        "network_seeds": [1000, 1019],
        "panel_a_auc_means": {
            key: value["mean"]
            for key, value in payload["a"]["contrast_stats"].items()
        },
        "panel_b_network_gain": {
            "mean": payload["b"]["network_gain_mean"],
            "ci95": payload["b"]["network_gain_ci95"],
        },
        "panel_c_contrast_means": {
            key: value["mean"]
            for key, value in payload["c"]["contrast_stats"].items()
        },
        "panel_c_pair_validation": payload["c"]["validation"],
        "panel_d_interaction": payload["d"]["interaction"],
        "panel_e_means": {
            condition: stats_entry["mean"]
            for condition, stats_entry in payload["e"]["condition_stats"].items()
        },
        "panel_e_paired": payload["e"]["paired_ci95"],
        "panel_f_interaction": payload["f"]["interaction_ci95"],
        "outputs": {key: str(path.relative_to(output_dir)) for key, path in outputs.items()},
        "parent_integrity": parent_integrity,
        "layout_status": layout_audit["status"],
        "render_qa_status": render_qa["status"] if render_qa else "not_run",
        "grayscale_qa_status": grayscale_qa["status"] if grayscale_qa else "not_run",
        "visual_qa_status": visual_qa["status"] if visual_qa else "not_run",
        "determinism_status": determinism["status"] if determinism else "not_run",
    }
    _write_json(output_dir / "summary.json", summary)
    log_lines = [
        f"{_utc_now()} candidate={CANDIDATE_VERSION}",
        f"mode={'check-only' if check_only else 'plot-only render'}",
        f"parent_snapshot_before={before_digest}",
        f"parent_snapshot_after={after_digest}",
        f"layout={layout_audit['status']}",
        f"panel_c_pair_validation={payload['c']['validation']['status']}",
        f"render_qa={render_qa['status'] if render_qa else 'not_run'}",
        f"grayscale_qa={grayscale_qa['status'] if grayscale_qa else 'not_run'}",
        f"visual_qa={visual_qa['status'] if visual_qa else 'not_run'}",
        f"determinism={determinism['status'] if determinism else 'not_run'}",
        f"parent_integrity={parent_integrity['status']}",
    ]
    (output_dir / "logs" / "render.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    manifest = _write_artifact_manifest(output_dir)
    return {
        "status": summary["status"],
        "output_dir": str(output_dir),
        "outputs": summary["outputs"],
        "layout": layout_audit["status"],
        "render_qa": summary["render_qa_status"],
        "grayscale_qa": summary["grayscale_qa_status"],
        "visual_qa": summary["visual_qa_status"],
        "determinism": summary["determinism_status"],
        "panel_c_pair_validation": payload["c"]["validation"]["status"],
        "parent_integrity": parent_integrity["status"],
        "artifact_count": manifest["artifact_count"],
    }


def _load_sources(reader: BundleReader, spec: Mapping[str, Any]) -> dict[str, Any]:
    a_raw = reader.read_csv("data/panel_a_plot_data.csv", "Fig.7a persisted network partial-cue curves")
    a_contrasts = reader.read_csv("data/panel_a_auc_contrasts.csv", "Fig.7a persisted network AUC contrasts")
    b_raw = reader.read_csv("data/panel_b_absolute_access.csv", "Fig.7b persisted network absolute access")
    c_raw = reader.read_csv("data/panel_c_position_profiles.csv", "Fig.7c persisted network position profiles")
    d_raw = reader.read_csv("data/panel_d_plot_data.csv", "Fig.7d persisted network heatmap cells")
    e_raw = reader.read_csv("data/panel_e_plot_data.csv", "Fig.7e persisted network removal losses")
    f_raw = reader.read_csv("data/panel_f_plot_data.csv", "Fig.7f persisted primary 10-ms 2x2 cells")
    f_window = reader.read_csv("data/panel_f_window_robustness.csv", "Fig.7f persisted window robustness (supplementary boundary only)")
    a_stats = reader.read_csv("metrics/panel_a_statistics.csv", "Fig.7a frozen statistics")
    b_stats = reader.read_csv("metrics/panel_b_absolute_access_statistics.csv", "Fig.7b frozen absolute-access statistics")
    b_gain_stats = reader.read_csv("metrics/panel_b_statistics.csv", "Fig.7b frozen gain statistics")
    c_stats = reader.read_csv("metrics/panel_c_statistics.csv", "Fig.7c frozen paired-contrast statistics")
    c_position_stats = reader.read_csv("metrics/panel_c_position_statistics.csv", "Fig.7c frozen position statistics")
    d_stats = reader.read_csv("metrics/panel_d_statistics.csv", "Fig.7d frozen cell and interaction statistics")
    e_stats = reader.read_csv("metrics/panel_e_statistics.csv", "Fig.7e frozen removal statistics")
    f_stats = reader.read_csv("metrics/panel_f_statistics.csv", "Fig.7f frozen 2x2 and interaction statistics")
    manifests = {
        name: reader.read_csv(f"meta/{name}", f"parent provenance {name}")
        for name in (
            "panel_c_source_manifest.csv",
            "source_manifest.csv",
            "parent_hashes_before.csv",
            "parent_hashes_after.csv",
        )
    }
    parent_artifact_manifest = reader.read_json("artifact_manifest.json", "parent artifact manifest")
    parent_final_spec = reader.read_json("meta/final_plot_spec.json", "parent final plot spec")
    parent_summary = reader.read_json("summary.json", "parent bundle summary")

    a = _validate_panel_a(a_raw, a_contrasts, a_stats)
    b = _validate_panel_b(b_raw, b_stats, b_gain_stats)
    c = _validate_panel_c(c_raw, c_stats, c_position_stats)
    d = _validate_panel_d(d_raw, d_stats)
    e = _validate_panel_e(e_raw, e_stats)
    f = _validate_panel_f(f_raw, f_stats, f_window)
    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e,
        "f": f,
        "manifests": manifests,
        "parent_artifact_manifest": parent_artifact_manifest,
        "parent_final_spec": parent_final_spec,
        "parent_summary": parent_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the formal reader-first manuscript Fig.7.")
    parser.add_argument("--parent-dir", default=("results/paper_figure_multi_seed/" "final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig6"))
    parser.add_argument("--output-dir", default="results/paper_figures/outputs/provenance/fig7")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--refresh-manifest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    repo_root = _repo_root()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.resolve()
    spec = _load_spec()
    pinned_parent = (repo_root / spec["parent_bundle"]).resolve()
    if args.refresh_manifest:
        if _inside(output_dir, pinned_parent) or _inside(pinned_parent, output_dir):
            raise ValueError("manifest refresh cannot target a pinned parent tree")
        if not output_dir.is_dir():
            raise FileNotFoundError(f"candidate output is missing: {output_dir}")
        print(json.dumps(_write_artifact_manifest(output_dir), indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    parent_dir = Path(args.parent_dir)
    if not parent_dir.is_absolute():
        parent_dir = repo_root / parent_dir
    result = build_candidate(parent_dir=parent_dir, output_dir=output_dir, check_only=bool(args.check_only))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
