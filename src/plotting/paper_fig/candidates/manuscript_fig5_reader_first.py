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
from PIL import Image
from pypdf import PdfReader
from matplotlib.ticker import FuncFormatter

from src.plotting.common.colors import get_plot_color
from src.plotting.paper_fig.layout_contract import validate_layout_contract
from src.plotting.paper_fig.typography import (
    VECTOR_TEXT_RCPARAMS,
    apply_paper_figure_typography,
    mark_panel_label,
)


CANDIDATE_VERSION = "manuscript_fig5_reader_first_v1"
DISPLAY_NAME = "Fig.5"
EXPECTED_SEEDS = tuple(range(1000, 1020))
EXPECTED_STAGES = tuple(range(2, 11))
MM_TO_INCH = 1.0 / 25.4
MM_TO_POINT = 72.0 / 25.4
SPEC_PATH = (
    Path(__file__).resolve().parent
    / "specs"
    / "manuscript_fig5_reader_first_v1.json"
)
INK = get_plot_color("ink", context="manuscript_fig5")
NAVY = get_plot_color("dynamic", context="manuscript_fig5")
DONOR = get_plot_color("donor_trace", context="manuscript_fig5")
NEUTRAL_MID = get_plot_color("neutral_mid", context="manuscript_fig5")
NEUTRAL_LIGHT = get_plot_color("neutral_light", context="manuscript_fig5")


PARENT_DATA_FILES = {
    "data/panel_a_plot_data.csv",
    "data/panel_b_plot_data.csv",
    "data/panel_c_plot_data.csv",
    "data/panel_d_plot_data.csv",
    "metrics/panel_a_statistics.csv",
    "metrics/panel_b_statistics.csv",
    "metrics/panel_c_recurrence_inference.csv",
    "metrics/panel_d_statistics.csv",
    "metrics/panel_d_depth_inference.csv",
    "meta/panel_a_source_manifest.csv",
    "meta/panel_b_source_manifest.csv",
    "meta/panel_c_source_manifest.csv",
    "meta/panel_d_source_manifest.csv",
    "meta/source_manifest.csv",
    "meta/parent_hashes_before.csv",
    "meta/parent_hashes_after.csv",
    "artifact_manifest.json",
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
        raise ValueError("candidate display name must be Fig.5")
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


def _snapshot_selected(root: Path, relative_paths: Sequence[str], source_scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = (root / relative).resolve()
        if not _inside(path, root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"registered source is missing: {path}")
        rows.append(
            {
                "source_scope": source_scope,
                "path": Path(relative).as_posix(),
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
    axis.tick_params(
        axis="both",
        which="major",
        colors=INK,
        width=0.6,
        length=2.5,
        pad=2.0,
    )
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
    contrast: str,
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


def _validate_frozen_mean(values: pd.Series, statistic: pd.Series, label: str) -> None:
    observed = float(pd.to_numeric(values, errors="raise").mean())
    expected = float(statistic["estimate"])
    if not np.isclose(observed, expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"{label}: network mean {observed} disagrees with frozen estimate {expected}")


def _relabel(frame: pd.DataFrame, panel_id: str) -> pd.DataFrame:
    output = frame.copy()
    if "figure_id" in output.columns:
        output["figure_id"] = DISPLAY_NAME
    if "panel_id" in output.columns:
        output["panel_id"] = panel_id
    return output


def _validate_category_panel(
    frame: pd.DataFrame,
    statistics: pd.DataFrame,
    *,
    panel_id: str,
    endpoint: str,
    condition_order: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    label = f"Fig.5{panel_id}"
    _require_seed_set(frame, label)
    required = {"network_seed", "condition", "value"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    data = frame.copy()
    data["network_seed"] = pd.to_numeric(data["network_seed"], errors="raise").astype(int)
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    if not np.isfinite(data["value"].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: non-finite network value")
    if set(data["condition"].astype(str)) != set(condition_order):
        raise ValueError(f"{label}: condition set does not match {condition_order}")
    if data.duplicated(["network_seed", "condition"]).any():
        raise ValueError(f"{label}: duplicate network-condition rows")
    if len(data) != len(EXPECTED_SEEDS) * len(condition_order):
        raise ValueError(f"{label}: expected exactly 40 network-condition rows, observed {len(data)}")
    stats_by_condition: dict[str, pd.Series] = {}
    for condition in condition_order:
        group = f"{endpoint}|{condition}"
        contrast = f"{endpoint}_{condition}_vs_zero"
        stat = _one_statistic(statistics, endpoint=endpoint, contrast=contrast, group=group)
        values = data.loc[data["condition"].astype(str).eq(condition), "value"]
        if len(values) != len(EXPECTED_SEEDS):
            raise ValueError(f"{label}: incomplete condition {condition}")
        _validate_frozen_mean(values, stat, f"{label} {condition}")
        if float(stat["null_value"]) != 0.0:
            raise ValueError(f"{label} {condition}: frozen null is not zero")
        if not bool((values > 0.0).all()):
            raise ValueError(f"{label} {condition}: expected all 20 network values above zero")
        stats_by_condition[condition] = stat
    return data.sort_values(["condition", "network_seed"]).reset_index(drop=True), stats_by_condition


def _materialize_paired_displacement(
    raw: pd.DataFrame,
    statistics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, pd.Series], dict[str, Any]]:
    label = "Fig.5c"
    _require_seed_set(raw, label)
    required = {"network_seed", "stage_k", "condition", "value", "unit"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    data = raw.copy()
    data["network_seed"] = pd.to_numeric(data["network_seed"], errors="raise").astype(int)
    data["stage_k"] = pd.to_numeric(data["stage_k"], errors="raise").astype(int)
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    if set(data["stage_k"]) != set(EXPECTED_STAGES):
        raise ValueError(f"{label}: stages must be exactly 2-10")
    if set(data["condition"].astype(str)) != {"observed", "passive"}:
        raise ValueError(f"{label}: raw conditions must be observed and passive")
    if len(data) != len(EXPECTED_SEEDS) * len(EXPECTED_STAGES) * 2:
        raise ValueError(f"{label}: expected 360 raw network-stage-condition rows, observed {len(data)}")
    keys = ["network_seed", "stage_k"]
    counts = data.groupby(keys, sort=True).size()
    if len(counts) != len(EXPECTED_SEEDS) * len(EXPECTED_STAGES) or not (counts == 2).all():
        raise ValueError(f"{label}: every network-stage pair must have exactly two raw rows")
    duplicate_conditions = data.duplicated(keys + ["condition"])
    if bool(duplicate_conditions.any()):
        raise ValueError(f"{label}: duplicate observed/passive condition row")
    pair_rows: list[dict[str, Any]] = []
    frozen_by_stage: dict[int, pd.Series] = {}
    for stage in EXPECTED_STAGES:
        endpoint = f"joint_observed_minus_passive_stage_{stage}"
        stat = _one_statistic(
            statistics,
            endpoint=endpoint,
            contrast=endpoint,
            group=endpoint,
        )
        frozen_by_stage[stage] = stat
        stage_rows = data.loc[data["stage_k"].eq(stage)]
        values: list[float] = []
        for seed in EXPECTED_SEEDS:
            cell = stage_rows.loc[stage_rows["network_seed"].eq(seed)]
            if len(cell) != 2 or set(cell["condition"].astype(str)) != {"observed", "passive"}:
                raise ValueError(f"{label}: malformed pair at network {seed}, stage {stage}")
            observed = float(cell.loc[cell["condition"].astype(str).eq("observed"), "value"].iloc[0])
            passive = float(cell.loc[cell["condition"].astype(str).eq("passive"), "value"].iloc[0])
            displacement = observed - passive
            values.append(displacement)
            pair_rows.append(
                {
                    "figure_id": DISPLAY_NAME,
                    "panel_id": "c",
                    "network_seed": seed,
                    "stage_k": stage,
                    "condition": "input_driven",
                    "record_type": "paired_network_stage",
                    "endpoint": "joint_observed_minus_passive_displacement",
                    "observed_value": observed,
                    "passive_value": passive,
                    "value": displacement,
                    "unit": "cosine_distance",
                    "frozen_mean": float(stat["estimate"]),
                    "frozen_ci95_low": float(stat["ci95_low"]),
                    "frozen_ci95_high": float(stat["ci95_high"]),
                }
            )
        materialized_mean = float(np.mean(np.asarray(values, dtype=float)))
        if not np.isclose(materialized_mean, float(stat["estimate"]), rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{label}: materialized stage {stage} mean {materialized_mean} "
                f"disagrees with frozen estimate {stat['estimate']}"
            )
    paired = pd.DataFrame(pair_rows).sort_values(["stage_k", "network_seed"]).reset_index(drop=True)
    if len(paired) != 180:
        raise ValueError(f"{label}: expected 180 paired values, observed {len(paired)}")
    validation = {
        "raw_rows": int(len(data)),
        "paired_rows": int(len(paired)),
        "network_count": len(EXPECTED_SEEDS),
        "stage_count": len(EXPECTED_STAGES),
        "conditions_per_network_stage": 2,
        "raw_condition_rows": {"observed": 180, "passive": 180},
        "materialization": "observed - passive",
        "frozen_statistic_rows": [
            f"joint_observed_minus_passive_stage_{stage}" for stage in EXPECTED_STAGES
        ],
        "mean_cross_checks": {
            str(stage): {
                "materialized_mean": float(paired.loc[paired["stage_k"].eq(stage), "value"].mean()),
                "frozen_estimate": float(frozen_by_stage[stage]["estimate"]),
                "match": True,
            }
            for stage in EXPECTED_STAGES
        },
        "status": "passed",
    }
    return paired, frozen_by_stage, validation


def _load_sources(reader: BundleReader, spec: Mapping[str, Any]) -> dict[str, Any]:
    a_raw = reader.read_csv("data/panel_a_plot_data.csv", "Fig.5a persisted network points")
    b_raw = reader.read_csv("data/panel_b_plot_data.csv", "Fig.5b persisted network points")
    c_raw = reader.read_csv("data/panel_c_plot_data.csv", "Fig.5c persisted observed/passive network-stage rows")
    d_raw = reader.read_csv("data/panel_d_plot_data.csv", "Fig.5d persisted network rates")
    a_stats = reader.read_csv("metrics/panel_a_statistics.csv", "Fig.5a frozen endpoint-by-depth statistics")
    b_stats = reader.read_csv("metrics/panel_b_statistics.csv", "Fig.5b frozen endpoint-by-depth statistics")
    c_stats = reader.read_csv("metrics/panel_c_recurrence_inference.csv", "Fig.5c frozen paired stage statistics")
    d_stats = reader.read_csv("metrics/panel_d_statistics.csv", "Fig.5d frozen level statistics")
    d_depth_stats = reader.read_csv("metrics/panel_d_depth_inference.csv", "Fig.5d frozen depth contrasts")
    a, a_frozen = _validate_category_panel(
        a_raw,
        a_stats,
        panel_id="a",
        endpoint="early_layer2_event_map_donor_transfer",
        condition_order=("K1", "K5"),
    )
    b, b_frozen = _validate_category_panel(
        b_raw,
        b_stats,
        panel_id="b",
        endpoint="layer3_successor_ux_donor_transfer",
        condition_order=("K1", "K5"),
    )
    c, c_frozen, c_validation = _materialize_paired_displacement(c_raw, c_stats)
    _require_seed_set(d_raw, "Fig.5d")
    d = d_raw.copy()
    d["network_seed"] = pd.to_numeric(d["network_seed"], errors="raise").astype(int)
    d["value"] = pd.to_numeric(d["value"], errors="raise")
    d["prefix_k"] = d["prefix_k"].astype(str)
    d["outcome_type"] = d["outcome_type"].astype(str)
    if len(d) != 80 or set(d["prefix_k"]) != {"K1", "K5"} or set(d["outcome_type"]) != {"rescue", "loss"}:
        raise ValueError("Fig.5d: expected 80 rows across K1/K5 and Rescue/Loss")
    if d.duplicated(["network_seed", "prefix_k", "outcome_type"]).any():
        raise ValueError("Fig.5d: duplicate network-depth-outcome row")
    d_frozen: dict[tuple[str, str], pd.Series] = {}
    for prefix in ("K1", "K5"):
        for outcome in ("rescue", "loss"):
            endpoint = "rescue_relation_balanced_rate" if outcome == "rescue" else "loss_relation_balanced_rate"
            group = f"{endpoint}|{prefix}"
            rows = d_stats.loc[
                d_stats["endpoint"].astype(str).eq(endpoint)
                & d_stats["group"].fillna("").astype(str).eq(group)
            ]
            if len(rows) != 1:
                raise ValueError(f"Fig.5d: missing frozen statistic {group}")
            stat = rows.iloc[0]
            values = d.loc[d["prefix_k"].eq(prefix) & d["outcome_type"].eq(outcome), "value"]
            if len(values) != 20:
                raise ValueError(f"Fig.5d: incomplete network values for {prefix}/{outcome}")
            _validate_frozen_mean(values, stat, f"Fig.5d {prefix}/{outcome}")
            d_frozen[(prefix, outcome)] = stat
    manifests = {
        name: reader.read_csv(f"meta/{name}", f"parent provenance {name}")
        for name in (
            "panel_a_source_manifest.csv",
            "panel_b_source_manifest.csv",
            "panel_c_source_manifest.csv",
            "panel_d_source_manifest.csv",
            "source_manifest.csv",
            "parent_hashes_before.csv",
            "parent_hashes_after.csv",
        )
    }
    parent_artifact_manifest = reader.read_json("artifact_manifest.json", "parent artifact manifest")
    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d.sort_values(["prefix_k", "outcome_type", "network_seed"]).reset_index(drop=True),
        "a_frozen": a_frozen,
        "b_frozen": b_frozen,
        "c_frozen": c_frozen,
        "d_frozen": d_frozen,
        "a_stats": a_stats,
        "b_stats": b_stats,
        "c_stats": c_stats,
        "d_stats": d_stats,
        "d_depth_stats": d_depth_stats,
        "c_validation": c_validation,
        "raw_c": c_raw,
        "manifests": manifests,
        "parent_artifact_manifest": parent_artifact_manifest,
    }


# ---------------------------------------------------------------- plotting


def _seed_jitter(seeds: Sequence[Any], width: float) -> np.ndarray:
    numeric = np.asarray([int(value) for value in seeds], dtype=float)
    centered = ((numeric - float(EXPECTED_SEEDS[0])) % len(EXPECTED_SEEDS)) - 9.5
    return centered / 9.5 * float(width)


def _draw_category_points(
    axis: plt.Axes,
    frame: pd.DataFrame,
    frozen: Mapping[str, pd.Series],
    panel_spec: Mapping[str, Any],
) -> dict[str, Any]:
    condition_order = [str(value) for value in panel_spec["x_order"]]
    color = get_plot_color(str(panel_spec["color"]), context="manuscript_fig5")
    labels: dict[str, str] = {}
    raw_point_counts: dict[str, int] = {}
    label_clearance: dict[str, float] = {}
    for index, condition in enumerate(condition_order):
        subset = frame.loc[frame["condition"].astype(str).eq(condition)].sort_values("network_seed")
        values = subset["value"].to_numpy(dtype=float)
        seeds = subset["network_seed"].to_numpy(dtype=int)
        jitter = _seed_jitter(seeds, float(panel_spec["jitter_width"]))
        axis.scatter(
            np.full(len(subset), float(index)) + jitter,
            values,
            s=float(panel_spec["raw_point_size"]),
            marker="o",
            facecolor=color,
            edgecolor="none",
            alpha=float(panel_spec["raw_point_alpha"]),
            zorder=2,
        )
        stat = frozen[condition]
        mean = float(stat["estimate"])
        low = float(stat["ci95_low"])
        high = float(stat["ci95_high"])
        axis.errorbar(
            [float(index)],
            [mean],
            yerr=[[mean - low], [high - mean]],
            fmt=str(panel_spec["mean_marker"]),
            color=INK,
            markerfacecolor=color,
            markeredgecolor=INK,
            markeredgewidth=0.75,
            markersize=float(panel_spec["mean_marker_size"]),
            ecolor=INK,
            elinewidth=1.0,
            capsize=2.8,
            capthick=0.9,
            zorder=5,
        )
        label_y = high + 0.012
        y_max = float(panel_spec["y_limits"][1])
        if label_y > y_max - 0.012:
            label_y = y_max - 0.012
        text = axis.text(
            float(index),
            label_y,
            f"{mean:.{int(panel_spec['numeric_precision'])}f}",
            ha="center",
            va="bottom",
            color=INK,
            zorder=6,
            clip_on=False,
        )
        labels[condition] = str(text.get_text())
        raw_point_counts[condition] = int(len(subset))
        label_clearance[condition] = float(label_y - max(float(high), float(values.max())))
    axis.set_xlim(-0.42, len(condition_order) - 0.58)
    axis.set_xticks(np.arange(len(condition_order), dtype=float))
    axis.set_xticklabels([str(panel_spec["x_labels"][condition]) for condition in condition_order])
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.yaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    return {
        "raw_point_counts": raw_point_counts,
        "numeric_labels": labels,
        "label_clearance": label_clearance,
        "shared_y_scale": list(panel_spec["y_limits"]),
    }


def _draw_paired_trajectory(
    axis: plt.Axes,
    paired: pd.DataFrame,
    frozen: Mapping[int, pd.Series],
    panel_spec: Mapping[str, Any],
) -> dict[str, Any]:
    stages = [int(value) for value in panel_spec["x_order"]]
    means = np.asarray([float(frozen[stage]["estimate"]) for stage in stages], dtype=float)
    lows = np.asarray([float(frozen[stage]["ci95_low"]) for stage in stages], dtype=float)
    highs = np.asarray([float(frozen[stage]["ci95_high"]) for stage in stages], dtype=float)
    color = get_plot_color(str(panel_spec["color"]), context="manuscript_fig5")
    axis.plot(
        stages,
        means,
        color=color,
        linewidth=float(panel_spec["line_width"]),
        linestyle="-",
        marker=str(panel_spec["marker"]),
        markersize=float(panel_spec["marker_size"]),
        markerfacecolor=color,
        markeredgecolor=color,
        zorder=4,
    )
    for stage, mean, low, high in zip(stages, means, lows, highs):
        axis.errorbar(
            [stage],
            [mean],
            yerr=[[mean - low], [high - mean]],
            fmt="none",
            ecolor=INK,
            elinewidth=float(panel_spec["ci_line_width"]),
            capsize=float(panel_spec["ci_capsize"]),
            capthick=float(panel_spec["ci_line_width"]),
            zorder=3,
        )
    axis.set_xlim(*[float(value) for value in panel_spec["x_limits"]])
    axis.set_xticks(stages)
    axis.set_xticklabels([str(stage) for stage in stages])
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.yaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    return {
        "stages": stages,
        "mean_values": means.tolist(),
        "ci_low": lows.tolist(),
        "ci_high": highs.tolist(),
        "network_values_per_stage": {
            str(stage): int((paired["stage_k"].eq(stage)).sum()) for stage in stages
        },
        "passive_artwork": False,
    }


def _draw_grouped_bars(
    axis: plt.Axes,
    frame: pd.DataFrame,
    frozen: Mapping[tuple[str, str], pd.Series],
    panel_spec: Mapping[str, Any],
) -> dict[str, Any]:
    prefixes = [str(value) for value in panel_spec["x_order"]]
    series = list(panel_spec["series"])
    positions = np.arange(len(prefixes), dtype=float)
    width = 0.30
    offsets = np.linspace(-width / 2.0, width / 2.0, len(series))
    bar_values: dict[str, float] = {}
    for series_index, series_item in enumerate(series):
        outcome = str(series_item["key"])
        color = get_plot_color(str(series_item["color"]), context="manuscript_fig5")
        for prefix_index, prefix in enumerate(prefixes):
            stat = frozen[(prefix, outcome)]
            mean = float(stat["estimate"])
            low = float(stat["ci95_low"])
            high = float(stat["ci95_high"])
            x = float(positions[prefix_index] + offsets[series_index])
            axis.bar(
                x,
                mean,
                width=width * 0.88,
                color=color,
                edgecolor="none",
                zorder=3,
            )
            axis.errorbar(
                [x],
                [mean],
                yerr=[[mean - low], [high - mean]],
                fmt="none",
                ecolor=INK,
                elinewidth=0.8,
                capsize=2.0,
                capthick=0.8,
                zorder=4,
            )
            bar_values[f"{prefix}_{outcome}"] = mean
    axis.set_xlim(-0.60, len(prefixes) - 0.40)
    axis.set_xticks(positions)
    axis.set_xticklabels([str(panel_spec["x_labels"][prefix]) for prefix in prefixes])
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.yaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=get_plot_color(str(item["color"]), context="manuscript_fig5"),
            linewidth=5.0,
            solid_capstyle="butt",
            label=str(item["label"]),
        )
        for item in series
    ]
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        frameon=False,
        ncol=len(handles),
        handlelength=1.25,
        handletextpad=0.45,
        columnspacing=1.0,
        borderaxespad=0.0,
        labelspacing=0.3,
    )
    return {"bar_values": bar_values, "legend": [str(item["label"]) for item in series]}


def _layout_audit(spec: Mapping[str, Any]) -> dict[str, Any]:
    report = validate_layout_contract(spec)
    canvas_width, canvas_height = [float(value) for value in spec["canvas_mm"]]
    expected_slots = {
        "a": [2.0, 2.0, 79.5, 48.0],
        "b": [83.5, 2.0, 79.5, 48.0],
        "c": [2.0, 52.0, 79.5, 48.0],
        "d": [83.5, 52.0, 79.5, 48.0],
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
    if [canvas_width, canvas_height] != [165.0, 102.0]:
        failures.append("canvas differs from 165 x 102 mm")
    if spec["slots"]["b"][0] - (spec["slots"]["a"][0] + spec["slots"]["a"][2]) != 2.0:
        failures.append("top-row gutter is not 2 mm")
    if spec["slots"]["c"][1] - (spec["slots"]["a"][1] + spec["slots"]["a"][3]) != 2.0:
        failures.append("row gutter is not 2 mm")
    for left, right in (("a", "b"), ("c", "d")):
        left_plot = spec["panels"][left]["plot_bbox_mm"]
        right_plot = spec["panels"][right]["plot_bbox_mm"]
        if left_plot[1:] != right_plot[1:]:
            failures.append(f"row {left}/{right} plot-area geometry is not aligned")
    return {
        "schema": "manuscript_fig5_candidate_layout_audit_v1",
        "status": "passed" if not failures else "failed",
        "passes": report.passes,
        "warnings": report.warnings,
        "failures": failures,
        "geometry_rows": rows,
    }


def _render_wireframe(spec: Mapping[str, Any], output: Path) -> None:
    from matplotlib.patches import Rectangle

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
            axis.add_patch(Rectangle((px, py), pw, ph, facecolor=get_plot_color("neutral_pale"), edgecolor=NEUTRAL_LIGHT, linewidth=0.6))
            text = axis.text(x + 1.0, y + 1.0, panel_id, ha="left", va="top", color=INK)
            mark_panel_label(text)
        apply_paper_figure_typography(figure)
        figure.savefig(output, dpi=300, facecolor="white", bbox_inches=None, metadata={"Date": None, "Creator": CANDIDATE_VERSION})
        plt.close(figure)


def _render_figure(spec: Mapping[str, Any], payload: Mapping[str, Any], figures_dir: Path) -> dict[str, Path]:
    canvas_mm = [float(value) for value in spec["canvas_mm"]]
    canvas_width, canvas_height = canvas_mm
    outputs = {
        "png": figures_dir / "manuscript_fig5.png",
        "svg": figures_dir / "manuscript_fig5.svg",
        "pdf": figures_dir / "manuscript_fig5.pdf",
        "base_svg": figures_dir / "qa" / "manuscript_fig5_base.svg",
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
        panel_qa["a"] = _draw_category_points(figure.add_axes(_as_axes_bbox(spec["panels"]["a"]["plot_bbox_mm"], canvas_mm)), payload["a"], payload["a_frozen"], spec["panels"]["a"])
        panel_qa["b"] = _draw_category_points(figure.add_axes(_as_axes_bbox(spec["panels"]["b"]["plot_bbox_mm"], canvas_mm)), payload["b"], payload["b_frozen"], spec["panels"]["b"])
        panel_qa["c"] = _draw_paired_trajectory(figure.add_axes(_as_axes_bbox(spec["panels"]["c"]["plot_bbox_mm"], canvas_mm)), payload["c"], payload["c_frozen"], spec["panels"]["c"])
        panel_qa["d"] = _draw_grouped_bars(figure.add_axes(_as_axes_bbox(spec["panels"]["d"]["plot_bbox_mm"], canvas_mm)), payload["d"], payload["d_frozen"], spec["panels"]["d"])
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
    shutil.copyfile(outputs["svg"], outputs["base_svg"])
    return {"png": outputs["png"], "svg": outputs["svg"], "pdf": outputs["pdf"], "base_svg": outputs["base_svg"], "panel_qa": panel_qa}


def _tag_name(element: Any) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def _render_qa(outputs: Mapping[str, Path], spec: Mapping[str, Any], panel_qa: Mapping[str, Any]) -> dict[str, Any]:
    expected_size = tuple(int(round(float(value) * 300.0 / 25.4)) for value in spec["canvas_mm"])
    with Image.open(outputs["png"]) as image:
        actual_size = image.size
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        border = 8
        border_pixels = np.concatenate([rgb[:border].reshape(-1, 3), rgb[-border:].reshape(-1, 3), rgb[:, :border].reshape(-1, 3), rgb[:, -border:].reshape(-1, 3)], axis=0)
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
    checks = {
        "png_dimensions": all(abs(actual - expected) <= 1 for actual, expected in zip(actual_size, expected_size)),
        "outer_border_clear": outer_border_clear,
        "svg_editable_text": text_count > 0,
        "svg_has_vector_paths": path_count > 0,
        "svg_no_bitmap_images": "<image" not in svg_lower,
        "svg_has_numeric_labels": all(
            label in svg_text
            for panel_id in ("a", "b")
            for label in panel_qa[panel_id]["numeric_labels"].values()
        ),
        "svg_no_passive_artwork": "passive" not in svg_lower,
        "svg_no_internal_bundle_label": "fig4" not in svg_lower,
        "pdf_one_page": len(pdf_reader.pages) == 1,
        "pdf_width_mm": math.isclose(width_pt / MM_TO_POINT, float(spec["canvas_mm"][0]), abs_tol=0.25),
        "pdf_height_mm": math.isclose(height_pt / MM_TO_POINT, float(spec["canvas_mm"][1]), abs_tol=0.25),
        "pdf_embedded_font_resources": font_count > 0,
        "pdf_panel_labels_present": all(letter in extracted_text for letter in "abcd"),
    }
    return {
        "schema": "manuscript_fig5_candidate_render_qa_v1",
        "generated_at": _utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "png": {"path": str(outputs["png"]), "pixels": list(actual_size), "expected_pixels_at_300_dpi": list(expected_size), "sha256": _sha256(outputs["png"]), "bytes": outputs["png"].stat().st_size},
        "svg": {"path": str(outputs["svg"]), "text_elements": text_count, "path_elements": path_count, "sha256": _sha256(outputs["svg"]), "bytes": outputs["svg"].stat().st_size},
        "pdf": {"path": str(outputs["pdf"]), "pages": len(pdf_reader.pages), "page_mm": [width_pt / MM_TO_POINT, height_pt / MM_TO_POINT], "font_resources": font_count, "sha256": _sha256(outputs["pdf"]), "bytes": outputs["pdf"].stat().st_size},
    }


def _grayscale_audit(outputs: Mapping[str, Path], figures_dir: Path) -> dict[str, Any]:
    grayscale_path = figures_dir / "qa" / "manuscript_fig5_grayscale.png"
    with Image.open(outputs["png"]) as image:
        image.convert("L").save(grayscale_path, dpi=(300, 300))
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    checks = {
        "grayscale_exists": grayscale_path.is_file(),
        "grayscale_has_dark_marks": bool((gray < 180).any()),
        "grayscale_has_midtones": bool(((gray >= 80) & (gray < 245)).any()),
    }
    return {
        "schema": "manuscript_fig5_candidate_grayscale_audit_v1",
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
            crop_path = panels_dir / f"manuscript_fig5{panel_id}.png"
            image.crop((left, upper, right, lower)).save(crop_path, dpi=(300, 300))
            crops.append({"panel": panel_id, "path": str(crop_path), "pixels": [right - left, lower - upper]})
    checks = {
        "a_has_20_points_per_cell": all(value == 20 for value in panel_qa["a"]["raw_point_counts"].values()),
        "b_has_20_points_per_cell": all(value == 20 for value in panel_qa["b"]["raw_point_counts"].values()),
        "a_numeric_labels_clear": all(value > 0.0 for value in panel_qa["a"]["label_clearance"].values()),
        "b_numeric_labels_clear": all(value > 0.0 for value in panel_qa["b"]["label_clearance"].values()),
        "c_has_nine_stages": len(panel_qa["c"]["stages"]) == 9,
        "c_passive_artwork_removed": panel_qa["c"]["passive_artwork"] is False,
        "row_1_plot_area_aligned": spec["panels"]["a"]["plot_bbox_mm"][1:] == spec["panels"]["b"]["plot_bbox_mm"][1:],
        "row_2_plot_area_aligned": spec["panels"]["c"]["plot_bbox_mm"][1:] == spec["panels"]["d"]["plot_bbox_mm"][1:],
        "panel_coverage_nonzero": all(value > 0.01 for value in panel_coverage.values()),
    }
    return {
        "schema": "manuscript_fig5_candidate_visual_qa_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "final_size_mm": spec["canvas_mm"],
        "panel_ink_coverage": panel_coverage,
        "panel_qa": panel_qa,
        "panel_crops": crops,
        "manual_review_targets": [
            "inspect a/b numeric labels against CI caps and raw point clouds",
            "inspect grayscale legibility and c/d row alignment",
            "inspect font rendering and clipping in the final composite",
        ],
    }


def _candidate_source_mapping(reader: BundleReader, parent_dir: Path, spec: Mapping[str, Any]) -> pd.DataFrame:
    accesses = reader.access_frame()
    rows: list[dict[str, Any]] = []
    for panel_id, data_rel, stats_rel, mapping in (
        ("a", "data/panel_a_plot_data.csv", "metrics/panel_a_statistics.csv", "20 network points; frozen endpoint-by-depth means and CIs; each cell tested against zero"),
        ("b", "data/panel_b_plot_data.csv", "metrics/panel_b_statistics.csv", "20 network points; frozen endpoint-by-depth means and CIs; each cell tested against zero"),
        ("c", "data/panel_c_plot_data.csv", "metrics/panel_c_recurrence_inference.csv", "strict observed/passive pairing by network and stage; materialized observed minus passive; frozen stage CIs"),
        ("d", "data/panel_d_plot_data.csv", "metrics/panel_d_statistics.csv", "20 network-level Rescue/Loss rates at K=1 and K=5; distinct opportunity denominators"),
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


def _caption(payload: Mapping[str, Any]) -> str:
    a = payload["a_frozen"]
    b = payload["b_frozen"]
    c = payload["c_frozen"]
    d = payload["d_frozen"]
    c_stages = [int(value) for value in EXPECTED_STAGES]
    a_p = _format_p(next(iter(a.values()))["p_adjusted"])
    b_p = _format_p(next(iter(b.values()))["p_adjusted"])
    c_p = _format_p(c[c_stages[0]]["p_adjusted"])
    d_p = _format_p(payload["d_depth_stats"].iloc[0]["p_adjusted"])
    a_values = ", ".join(f"{key} {float(row['estimate']):.3f}" for key, row in a.items())
    b_values = ", ".join(f"{key} {float(row['estimate']):.3f}" for key, row in b.items())
    return (
        "**Fig.5 | Successor reuse and iterative updating across successive inputs.**\n\n"
        f"**a,** Selective transfer of the post-B Layer-2 successor between matched donor and receiver histories, with the identical next input C and the other retained and fast states held fixed, redirected early Layer-2 processing at both tested history depths. Network means were K=1 {float(a['K1']['estimate']):.3f} and K=5 {float(a['K5']['estimate']):.3f}. **b,** The same tested successor transfer also redirected the post-C Layer-3 successor at both history depths (network means: K=1 {float(b['K1']['estimate']):.3f} and K=5 {float(b['K5']['estimate']):.3f}). Across a and b, all four endpoint-by-depth cells were positive in all 20 networks; each cell was tested separately against zero, not as a K=1-versus-K=5 contrast. In a and b, points are 20 independently trained networks, direct labels are network means, and intervals are persisted two-sided 95% network-bootstrap CIs; one-sided exact sign-flip tests against zero with Holm adjustment gave adjusted P values of {a_p} and {b_p}, respectively. The donor-transfer index is not a percentage or a mediation fraction. **c,** Each point and interval shows the observed next-input joint-state displacement minus the equal-time passive continuation from the same boundary. Passive displacement was close to zero at all stages and is therefore not shown as a second artwork trajectory. Stage-wise inference used 20 independent network paired contrasts with persisted two-sided 95% network-bootstrap CIs and one-sided exact sign-flip tests with Holm adjustment (adjusted P = {c_p}); transition stage is not history depth K, and stage-wise recurrence does not mean that each stage repeated the complete transplant protocol. **d,** Network-level means and persisted two-sided 95% network-bootstrap CIs for behavioral Rescue and Loss at K=1 and K=5. Rescue and Loss use different opportunity denominators; the depth contrasts used two-sided exact sign-flip tests with Holm adjustment (the supplied depth-family adjusted P value is {d_p}).\n\n"
        "The independent replication unit throughout is the independently trained network (n = 20; seeds 1000-1019); lower-level trial, cell, event, coordinate and stage rows were aggregated within network, and stage rows were paired only for panel c. The donor-transfer results establish bounded causal sufficiency under the tested intervention only; they do not establish necessity, complete mediation or uniqueness. The progressive recurrence panel is an independent protocol and does not imply that every transition stage executed the complete transplant procedure."
    )


def _format_p(value: Any) -> str:
    return f"{float(value):.2g}"


def _resolved_spec(spec: Mapping[str, Any], reader: BundleReader) -> dict[str, Any]:
    resolved = json.loads(json.dumps(spec))
    resolved["resolved_at"] = _utc_now()
    resolved["resolved_colors"] = {
        "ink": INK,
        "donor_trace": DONOR,
        "dynamic": NAVY,
        "neutral_mid": NEUTRAL_MID,
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

    _relabel(payload["a"], "a").to_csv(output_dir / "data" / "panel_a_plot_data.csv", index=False)
    _relabel(payload["b"], "b").to_csv(output_dir / "data" / "panel_b_plot_data.csv", index=False)
    payload["c"].to_csv(output_dir / "data" / "panel_c_input_driven_displacement.csv", index=False)
    _relabel(payload["d"], "d").to_csv(output_dir / "data" / "panel_d_plot_data.csv", index=False)
    _relabel(payload["a_stats"], "a").to_csv(output_dir / "metrics" / "panel_a_statistics.csv", index=False)
    _relabel(payload["b_stats"], "b").to_csv(output_dir / "metrics" / "panel_b_statistics.csv", index=False)
    _relabel(payload["c_stats"], "c").to_csv(output_dir / "metrics" / "panel_c_recurrence_inference.csv", index=False)
    _relabel(payload["d_stats"], "d").to_csv(output_dir / "metrics" / "panel_d_statistics.csv", index=False)
    _relabel(payload["d_depth_stats"], "d").to_csv(output_dir / "metrics" / "panel_d_depth_inference.csv", index=False)

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
    _write_json(output_dir / "meta" / "panel_c_pair_validation.json", payload["c_validation"])
    _write_json(output_dir / "meta" / "parent_artifact_manifest.json", payload["parent_artifact_manifest"])
    _write_json(output_dir / "meta" / "final_plot_spec.json", _resolved_spec(spec, reader))
    _write_json(output_dir / "meta" / "review_only_candidate_spec.json", spec)
    (output_dir / "caption_draft.md").write_text(_caption(payload), encoding="utf-8")

    outputs: dict[str, Path] = {}
    render_qa: dict[str, Any] | None = None
    grayscale_qa: dict[str, Any] | None = None
    visual_qa: dict[str, Any] | None = None
    panel_qa: dict[str, Any] = {}
    if not check_only:
        _render_wireframe(spec, output_dir / "figures" / "qa" / "manuscript_fig5_wireframe.png")
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

    parent_after = _snapshot_tree(parent_dir, "frozen_parent_bundle")
    after_digest = _snapshot_digest(parent_after)
    parent_after.to_csv(output_dir / "meta" / "parent_hashes_after.csv", index=False)
    parent_unchanged = parent_before.equals(parent_after)
    parent_integrity = {
        "schema": "manuscript_fig5_candidate_parent_integrity_v1",
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
        "panel_a_means": {key: float(row["estimate"]) for key, row in payload["a_frozen"].items()},
        "panel_b_means": {key: float(row["estimate"]) for key, row in payload["b_frozen"].items()},
        "panel_c_pair_validation": payload["c_validation"],
        "panel_c_stage_means": {str(stage): float(payload["c_frozen"][stage]["estimate"]) for stage in EXPECTED_STAGES},
        "panel_d_means": {f"{prefix}_{outcome}": float(stat["estimate"]) for (prefix, outcome), stat in payload["d_frozen"].items()},
        "outputs": {key: str(path.relative_to(output_dir)) for key, path in outputs.items()},
        "parent_integrity": parent_integrity,
        "layout_status": layout_audit["status"],
        "render_qa_status": render_qa["status"] if render_qa else "not_run",
        "grayscale_qa_status": grayscale_qa["status"] if grayscale_qa else "not_run",
        "visual_qa_status": visual_qa["status"] if visual_qa else "not_run",
    }
    _write_json(output_dir / "summary.json", summary)
    log_lines = [
        f"{_utc_now()} candidate={CANDIDATE_VERSION}",
        f"mode={'check-only' if check_only else 'plot-only render'}",
        f"parent_snapshot_before={before_digest}",
        f"parent_snapshot_after={after_digest}",
        f"layout={layout_audit['status']}",
        f"paired_materialization={payload['c_validation']['status']}",
        f"render_qa={render_qa['status'] if render_qa else 'not_run'}",
        f"grayscale_qa={grayscale_qa['status'] if grayscale_qa else 'not_run'}",
        f"visual_qa={visual_qa['status'] if visual_qa else 'not_run'}",
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
        "paired_materialization": payload["c_validation"]["status"],
        "parent_integrity": parent_integrity["status"],
        "artifact_count": manifest["artifact_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the formal reader-first manuscript Fig.5.")
    parser.add_argument("--parent-dir", default=("results/paper_figure_multi_seed/" "final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig4"))
    parser.add_argument("--output-dir", default="results/paper_figures/outputs/provenance/fig5")
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
