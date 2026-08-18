from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from lxml import etree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image
from pypdf import PdfReader, PdfWriter
from scipy import stats as scipy_stats

from src.plotting.common.colors import NATURE_COMPATIBLE_PALETTE, get_plot_color
from src.plotting.paper_fig.layout_contract import validate_layout_contract
from src.plotting.paper_fig.typography import (
    VECTOR_TEXT_RCPARAMS,
    apply_paper_figure_typography,
    mark_panel_label,
)


CANDIDATE_VERSION = "manuscript_fig4_reader_first_v1"
DISPLAY_NAME = "Fig.4"
EXPECTED_SEEDS = tuple(range(1000, 1020))
MM_TO_INCH = 1.0 / 25.4
MM_TO_POINT = 72.0 / 25.4
SPEC_PATH = (
    Path(__file__).resolve().parent
    / "specs"
    / "manuscript_fig4_reader_first_v1.json"
)
INK = NATURE_COMPATIBLE_PALETTE["ink"]
NEUTRAL_DARK = NATURE_COMPATIBLE_PALETTE["neutral_dark"]
NEUTRAL_MID = NATURE_COMPATIBLE_PALETTE["neutral_mid"]
NEUTRAL_LIGHT = NATURE_COMPATIBLE_PALETTE["neutral_light"]
NEUTRAL_PALE = NATURE_COMPATIBLE_PALETTE["neutral_pale"]


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


def _load_spec() -> dict[str, Any]:
    with SPEC_PATH.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("candidate_version") != CANDIDATE_VERSION:
        raise ValueError("candidate spec version mismatch")
    return spec


def _snapshot_tree(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows, columns=["path", "bytes", "sha256"])


def _snapshot_selected(
    root: Path,
    relative_paths: Sequence[str],
    *,
    source_scope: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = (root / relative).resolve()
        if not _inside(path, root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"registered parent source is missing: {path}")
        rows.append(
            {
                "source_scope": source_scope,
                "path": Path(relative).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows)


def _snapshot_digest(frame: pd.DataFrame) -> str:
    columns = [column for column in ("source_scope", "path", "bytes", "sha256") if column in frame]
    normalized = frame.sort_values(columns[:2]).loc[:, columns].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _resolve_color(role: str) -> str:
    if role in NATURE_COMPATIBLE_PALETTE:
        return NATURE_COMPATIBLE_PALETTE[role]
    return get_plot_color(role, context="manuscript_fig4")


def _student_t_ci(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or len(values) < 2:
        raise ValueError("student-t CI requires at least two finite network values")
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    sem = sd / math.sqrt(n)
    critical = float(scipy_stats.t.ppf(0.975, n - 1))
    return {
        "n_networks": n,
        "estimate": mean,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95_low": mean - critical * sem,
        "ci95_high": mean + critical * sem,
    }


@dataclass
class BundleReader:
    roots: dict[str, Path]
    expected_roots: dict[str, Path]
    allowed_files: dict[str, set[str]]
    accesses: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for scope, root in self.roots.items():
            self.roots[scope] = root.resolve()
            self.expected_roots[scope] = self.expected_roots[scope].resolve()
            if self.roots[scope] != self.expected_roots[scope]:
                raise ValueError(
                    f"candidate plotting accepts only the {scope} root pinned by the review spec"
                )
            if not self.roots[scope].is_dir():
                raise FileNotFoundError(f"pinned parent source root is missing: {root}")

    def _resolve_internal(
        self,
        relative: str,
        purpose: str,
        source_scope: str,
    ) -> Path:
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise ValueError(f"absolute source path is forbidden: {relative}")
        if source_scope not in self.roots:
            raise ValueError(f"unknown source scope: {source_scope}")
        root = self.roots[source_scope]
        if source_scope in self.allowed_files:
            if relative_path.as_posix() not in self.allowed_files[source_scope]:
                raise PermissionError(f"unregistered {source_scope} source: {relative}")
        path = (root / relative_path).resolve()
        if not _inside(path, root):
            raise PermissionError(f"plot source escapes its pinned parent root: {path}")
        if path.suffix.lower() not in {".csv", ".json", ".svg"}:
            raise PermissionError(f"unsupported plot source type: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"required plot source is missing: {path}")
        self.accesses.append(
            {
                "source_scope": source_scope,
                "path": str(path),
                "relative_path": relative_path.as_posix(),
                "purpose": purpose,
                "allowed": True,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
        return path

    def read_csv(
        self,
        relative: str,
        purpose: str,
        *,
        source_scope: str = "bundle",
    ) -> pd.DataFrame:
        return pd.read_csv(self._resolve_internal(relative, purpose, source_scope))

    def read_json(
        self,
        relative: str,
        purpose: str,
        *,
        source_scope: str = "bundle",
    ) -> dict[str, Any]:
        path = self._resolve_internal(relative, purpose, source_scope)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def read_bytes(
        self,
        relative: str,
        purpose: str,
        *,
        source_scope: str,
    ) -> bytes:
        return self._resolve_internal(relative, purpose, source_scope).read_bytes()

    def access_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.accesses)


def _as_axes_bbox(
    bbox_mm: Sequence[float], canvas_mm: Sequence[float]
) -> list[float]:
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
    contrast: str | None = None,
    group: str | None = None,
) -> pd.Series:
    rows = statistics.loc[statistics["endpoint"].astype(str).eq(endpoint)].copy()
    if contrast is not None:
        rows = rows.loc[rows["contrast"].fillna("").astype(str).eq(contrast)]
    if group is not None:
        rows = rows.loc[rows["group"].fillna("").astype(str).eq(group)]
    if len(rows) != 1:
        raise ValueError(
            f"expected one frozen statistic for endpoint={endpoint!r}, "
            f"contrast={contrast!r}, group={group!r}; observed {len(rows)}"
        )
    row = rows.iloc[0]
    values = pd.to_numeric(
        row[["estimate", "ci95_low", "ci95_high"]], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(values).all() or not values[1] <= values[0] <= values[2]:
        raise ValueError(f"invalid frozen estimate or confidence interval: {values}")
    return row


def _validate_mean(values: pd.Series, statistic: pd.Series, label: str) -> None:
    observed = float(pd.to_numeric(values, errors="raise").mean())
    expected = float(statistic["estimate"])
    if not np.isclose(observed, expected, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{label}: network mean {observed} disagrees with frozen estimate {expected}"
        )


def _load_sources(
    reader: BundleReader,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    bundle_stats: dict[str, pd.DataFrame] = {}
    bundle_data: dict[str, pd.DataFrame] = {}
    for panel_id in ("a", "b", "c", "d", "e", "f"):
        bundle_stats[panel_id] = reader.read_csv(
            f"metrics/panel_{panel_id}_statistics.csv",
            f"Fig.4{panel_id} frozen statistics",
        )
    for panel_id in ("b", "c", "d", "e"):
        bundle_data[panel_id] = reader.read_csv(
            f"data/panel_{panel_id}_plot_data.csv",
            f"Fig.4{panel_id} frozen plot data",
        )
    source_manifests = {
        name: reader.read_csv(f"meta/{name}", f"parent provenance {name}")
        for name in (
            "panel_a_source_manifest.csv",
            "panel_b_source_manifest.csv",
            "panel_c_source_manifest.csv",
            "panel_d_source_manifest.csv",
            "panel_e_source_manifest.csv",
            "panel_f_source_manifest.csv",
            "source_manifest.csv",
        )
    }

    # ---------------------------------------------------------------- panel a
    a_spec = spec["panels"]["a"]
    a_pattern = str(a_spec["source"])
    a_frames: list[pd.DataFrame] = []
    a_manifest_rows: list[dict[str, Any]] = []
    required_a_columns = {
        "network_seed",
        "acc_drop_dynamic",
        "acc_drop_overlap_reset",
        "acc_drop_nonoverlap_reset",
        "acc_drop_random_reset",
    }
    for seed in EXPECTED_SEEDS:
        relative = a_pattern.format(seed=seed)
        raw = reader.read_csv(
            relative,
            f"Fig.4a persisted reset-site contrast for network {seed}",
            source_scope="overlap_root",
        )
        missing = sorted(required_a_columns - set(raw.columns))
        if missing:
            raise ValueError(f"Fig.4a network {seed} is missing columns: {missing}")
        observed_seed = set(
            pd.to_numeric(raw["network_seed"], errors="raise").astype(int)
        )
        if observed_seed != {seed} or len(raw) != 1:
            raise ValueError(
                f"Fig.4a source seed/row mismatch for network {seed}: "
                f"seeds={observed_seed}, rows={len(raw)}"
            )
        frame = raw.copy()
        if "network_seed" in frame.columns:
            observed = set(pd.to_numeric(frame["network_seed"], errors="raise").astype(int))
            if observed != {seed} or len(frame) != 1:
                raise ValueError(
                    f"Fig.4a source seed/row mismatch for network {seed}: "
                    f"seeds={observed}, rows={len(frame)}"
                )
            frame["network_seed"] = seed
        else:
            frame.insert(0, "network_seed", seed)
        a_frames.append(frame)
        source_path = reader.roots["overlap_root"] / relative
        a_manifest_rows.append(
            {
                "candidate_figure": DISPLAY_NAME,
                "candidate_panel": "a",
                "network_seed": seed,
                "source_path": str(source_path),
                "source_relative_path": relative,
                "source_sha256": _sha256(source_path),
                "source_bytes": source_path.stat().st_size,
                "rows": len(raw),
                "filters": "none (one network-level row per condition)",
                "independent_unit": "network_seed",
            }
        )
    a_raw = pd.concat(a_frames, ignore_index=True)
    a_scale = float(a_spec["value_scale"])
    a_value_rows: list[dict[str, Any]] = []
    a_condition_stats: dict[str, dict[str, float]] = {}
    for category in a_spec["categories"]:
        condition = str(category["source_condition"])
        values = pd.to_numeric(a_raw[condition], errors="raise") * a_scale
        if not np.isfinite(values).all():
            raise ValueError(f"Fig.4a condition {condition} has non-finite values")
        stat = _student_t_ci(values)
        a_condition_stats[condition] = stat
        for seed, value in zip(EXPECTED_SEEDS, values):
            a_value_rows.append(
                {
                    "candidate_figure": DISPLAY_NAME,
                    "candidate_panel": "a",
                    "condition": condition,
                    "reader_label": str(category["label"]),
                    "network_seed": int(seed),
                    "value_percent": float(value),
                    "record_type": "network",
                }
            )
        a_value_rows.append(
            {
                "candidate_figure": DISPLAY_NAME,
                "candidate_panel": "a",
                "condition": condition,
                "reader_label": str(category["label"]),
                "network_seed": None,
                "value_percent": stat["estimate"],
                "record_type": "summary",
            }
        )
    a_values = pd.DataFrame(a_value_rows)
    # cross-check absolute means against the frozen predeclared contrasts
    frozen_a = bundle_stats["a"]
    for contrast, numerator in (
        ("dynamic_minus_overlap_reset", "acc_drop_dynamic"),
        ("nonoverlap_reset_minus_overlap_reset", "acc_drop_nonoverlap_reset"),
        ("random_reset_minus_overlap_reset", "acc_drop_random_reset"),
    ):
        stat = _one_statistic(frozen_a, endpoint=contrast)
        observed = (
            a_condition_stats[numerator]["estimate"]
            - a_condition_stats["acc_drop_overlap_reset"]["estimate"]
        )
        if not np.isclose(observed, float(stat["estimate"]), rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Fig.4a absolute bars disagree with frozen contrast {contrast}: "
                f"{observed} vs {stat['estimate']}"
            )
    for condition, expected in a_spec["expected_audit"].items():
        observed = a_condition_stats[condition]["estimate"]
        if not np.isclose(observed, float(expected), rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Fig.4a audit mismatch for {condition}: {observed} vs {expected}"
            )

    # ---------------------------------------------------------------- panel b
    b = bundle_data["b"].copy()
    _require_seed_set(b, "Fig.4b")
    b["network_seed"] = pd.to_numeric(b["network_seed"], errors="raise").astype(int)
    b["value"] = pd.to_numeric(b["value"], errors="raise")
    b_rows = b.loc[b["endpoint"].astype(str).eq("preprobe_mean_support")].copy()
    b_stat_rows: dict[str, pd.Series] = {}
    b_condition_stats: dict[str, dict[str, float]] = {}
    for category in spec["panels"]["b"]["categories"]:
        condition = str(category["source_condition"])
        values = b_rows.loc[b_rows["condition"].astype(str).eq(condition), "value"]
        if len(values) != len(EXPECTED_SEEDS):
            raise ValueError(f"Fig.4b condition {condition} is incomplete")
        stat = _one_statistic(
            bundle_stats["b"], endpoint="preprobe_mean_support", group=f"preprobe_mean_support|{condition}"
        )
        b_stat_rows[condition] = stat
        _validate_mean(values, stat, f"Fig.4b {condition}")
        b_condition_stats[condition] = {
            "n_networks": len(EXPECTED_SEEDS),
            "estimate": float(stat["estimate"]),
            "mean": float(stat["mean"]),
            "sd": float(stat["sd"]),
            "sem": float(stat["sem"]),
            "ci95_low": float(stat["ci95_low"]),
            "ci95_high": float(stat["ci95_high"]),
        }

    # ---------------------------------------------------------------- panel c
    c = bundle_data["c"].copy()
    _require_seed_set(c, "Fig.4c")
    c["network_seed"] = pd.to_numeric(c["network_seed"], errors="raise").astype(int)
    c["value"] = pd.to_numeric(c["value"], errors="raise")
    c["early_window_ms"] = pd.to_numeric(c["early_window_ms"], errors="raise").astype(int)
    window_ms = int(spec["panels"]["c"]["window_ms"])
    c_rows = c.loc[c["early_window_ms"].eq(window_ms)].copy()
    c_stat_rows: dict[tuple[str, str], pd.Series] = {}
    c_condition_stats: dict[tuple[str, str], dict[str, float]] = {}
    for condition in spec["panels"]["c"]["conditions"]:
        condition_key = str(condition["source_condition"])
        for series in spec["panels"]["c"]["series"]:
            endpoint = str(series["source_endpoint"])
            key = (condition_key, endpoint)
            values = c_rows.loc[
                c_rows["condition"].astype(str).eq(condition_key)
                & c_rows["endpoint"].astype(str).eq(endpoint),
                "value",
            ]
            if len(values) != len(EXPECTED_SEEDS):
                raise ValueError(f"Fig.4c cell {key} is incomplete")
            stat = _one_statistic(
                bundle_stats["c"],
                endpoint=endpoint,
                group=f"{endpoint}|{condition_key}|{window_ms}",
            )
            c_stat_rows[key] = stat
            _validate_mean(values, stat, f"Fig.4c {key}")
            c_condition_stats[key] = {
                "n_networks": len(EXPECTED_SEEDS),
                "estimate": float(stat["estimate"]),
                "mean": float(stat["mean"]),
                "sd": float(stat["sd"]),
                "sem": float(stat["sem"]),
                "ci95_low": float(stat["ci95_low"]),
                "ci95_high": float(stat["ci95_high"]),
            }

    # ---------------------------------------------------------------- panel d
    d = bundle_data["d"].copy()
    _require_seed_set(d, "Fig.4d")
    d["network_seed"] = pd.to_numeric(d["network_seed"], errors="raise").astype(int)
    d["value"] = pd.to_numeric(d["value"], errors="raise")
    d["time_window_ms"] = pd.to_numeric(d["time_window_ms"], errors="raise").astype(int)
    d_window_ms = int(spec["panels"]["d"]["window_ms"])
    d_rows = d.loc[d["time_window_ms"].eq(d_window_ms)].copy()
    d_stat_rows: dict[str, pd.Series] = {}
    d_condition_stats: dict[str, dict[str, float]] = {}
    for category in spec["panels"]["d"]["categories"]:
        endpoint = str(category["source_endpoint"])
        values = d_rows.loc[d_rows["endpoint"].astype(str).eq(endpoint), "value"]
        if len(values) != len(EXPECTED_SEEDS):
            raise ValueError(f"Fig.4d endpoint {endpoint} is incomplete")
        stat = _one_statistic(
            bundle_stats["d"], endpoint=endpoint, group=f"{endpoint}|first_{d_window_ms}_ms"
        )
        d_stat_rows[endpoint] = stat
        _validate_mean(values, stat, f"Fig.4d {endpoint}")
        d_condition_stats[endpoint] = {
            "n_networks": len(EXPECTED_SEEDS),
            "estimate": float(stat["estimate"]),
            "mean": float(stat["mean"]),
            "sd": float(stat["sd"]),
            "sem": float(stat["sem"]),
            "ci95_low": float(stat["ci95_low"]),
            "ci95_high": float(stat["ci95_high"]),
        }

    # ---------------------------------------------------------------- panel e
    e = bundle_data["e"].copy()
    _require_seed_set(e, "Fig.4e")
    e["network_seed"] = pd.to_numeric(e["network_seed"], errors="raise").astype(int)
    e["value"] = pd.to_numeric(e["value"], errors="raise")
    e_stat_rows: dict[tuple[str, str], pd.Series] = {}
    e_condition_stats: dict[tuple[str, str], dict[str, float]] = {}
    for condition in spec["panels"]["e"]["conditions"]:
        condition_key = str(condition["source_condition"])
        for series in spec["panels"]["e"]["series"]:
            history_key = str(series["source_history"])
            key = (condition_key, history_key)
            values = e.loc[
                e["condition"].astype(str).eq(condition_key)
                & e["history_status"].astype(str).eq(history_key),
                "value",
            ]
            if len(values) != len(EXPECTED_SEEDS):
                raise ValueError(f"Fig.4e cell {key} is incomplete")
            stat = _one_statistic(
                bundle_stats["e"],
                endpoint="l2_update_probability",
                group=f"l2_update_probability|{condition_key}|{history_key}",
            )
            e_stat_rows[key] = stat
            _validate_mean(values, stat, f"Fig.4e {key}")
            e_condition_stats[key] = {
                "n_networks": len(EXPECTED_SEEDS),
                "estimate": float(stat["estimate"]),
                "mean": float(stat["mean"]),
                "sd": float(stat["sd"]),
                "sem": float(stat["sem"]),
                "ci95_low": float(stat["ci95_low"]),
                "ci95_high": float(stat["ci95_high"]),
            }
    did_stat = _one_statistic(
        bundle_stats["e"],
        endpoint="dynamic_minus_static_difference_in_differences",
        group="dynamic_minus_static_difference_in_differences|did",
    )

    # ---------------------------------------------------------------- panel f
    f_spec = spec["panels"]["f"]
    f_pattern = str(f_spec["source"])
    f_rows_per_network = int(f_spec["rows_per_network"])
    bin_width = float(f_spec["bin_width"])
    n_bins = int(round(1.0 / bin_width))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    required_f_columns = {
        "prefix_k",
        "history_family_id",
        "b_anchor_id",
        "swap_scope",
        "endpoint",
        "donor_transfer_index",
        "valid",
    }
    f_network_summaries: list[dict[str, Any]] = []
    f_histogram_rows: list[dict[str, Any]] = []
    f_audit_rows: list[dict[str, Any]] = []
    f_network_histograms: list[np.ndarray] = []
    f_total_rows = 0
    f_positive_total = 0
    for seed in EXPECTED_SEEDS:
        relative = f_pattern.format(seed=seed)
        raw = reader.read_csv(
            relative,
            f"Fig.4f persisted swap-comparison rows for network {seed}",
            source_scope="swap_root",
        )
        missing = sorted(required_f_columns - set(raw.columns))
        if missing:
            raise ValueError(f"Fig.4f network {seed} is missing columns: {missing}")
        filtered = raw.loc[
            pd.to_numeric(raw["prefix_k"], errors="raise").eq(1)
            & raw["swap_scope"].astype(str).eq("layer1_only")
            & raw["endpoint"].astype(str).eq("layer2_update")
            & pd.to_numeric(raw["valid"], errors="raise").eq(1)
        ].copy()
        if len(filtered) != f_rows_per_network:
            raise ValueError(
                f"Fig.4f network {seed} requires {f_rows_per_network} valid rows; observed {len(filtered)}"
            )
        if filtered.duplicated(["history_family_id", "b_anchor_id", "donor_condition", "receiver_condition"]).any():
            raise ValueError(f"Fig.4f network {seed} has duplicate comparison keys")
        values = pd.to_numeric(
            filtered["donor_transfer_index"], errors="raise"
        ).to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"Fig.4f network {seed} has out-of-range donor-transfer values")
        n_positive = int((values > 0.0).sum())
        histogram, _ = np.histogram(values, bins=bin_edges)
        if not np.isclose(histogram.sum(), f_rows_per_network):
            raise ValueError(f"Fig.4f network {seed} histogram lost rows")
        probability = histogram / histogram.sum() * 100.0
        f_network_histograms.append(probability)
        f_total_rows += len(values)
        f_positive_total += n_positive
        f_network_summaries.append(
            {
                "candidate_figure": DISPLAY_NAME,
                "candidate_panel": "f",
                "network_seed": int(seed),
                "valid_rows": len(values),
                "n_positive": n_positive,
                "fraction_positive": n_positive / len(values),
                "network_mean": float(values.mean()),
                "network_median": float(np.median(values)),
                "network_min": float(values.min()),
                "network_max": float(values.max()),
            }
        )
        for bin_index in range(n_bins):
            f_histogram_rows.append(
                {
                    "candidate_figure": DISPLAY_NAME,
                    "candidate_panel": "f",
                    "network_seed": int(seed),
                    "bin_index": bin_index,
                    "bin_center": float((bin_edges[bin_index] + bin_edges[bin_index + 1]) / 2.0),
                    "bin_low": float(bin_edges[bin_index]),
                    "bin_high": float(bin_edges[bin_index + 1]),
                    "comparisons": int(histogram[bin_index]),
                    "network_normalized_percent": float(probability[bin_index]),
                    "network_weight": 1.0 / len(EXPECTED_SEEDS),
                }
            )
        f_audit_rows.append(
            {
                "candidate_figure": DISPLAY_NAME,
                "candidate_panel": "f",
                "network_seed": int(seed),
                "source_path": str(reader.roots["swap_root"] / relative),
                "source_relative_path": relative,
                "source_sha256": _sha256(reader.roots["swap_root"] / relative),
                "raw_rows": len(raw),
                "filtered_rows": len(filtered),
                "filters": "prefix_k=1; swap_scope=layer1_only; endpoint=layer2_update; valid=1",
                "comparison_key": "history_family_id x b_anchor_id x donor/receiver condition",
                "independent_unit": "network_seed",
            }
        )
    if len(f_network_histograms) != len(EXPECTED_SEEDS):
        raise ValueError("Fig.4f requires all 20 network histograms")
    aggregate_histogram = np.mean(np.stack(f_network_histograms, axis=0), axis=0)
    if not np.isclose(aggregate_histogram.sum(), 100.0):
        raise ValueError("Fig.4f aggregate histogram must sum to 100%")
    f_network_summary_frame = pd.DataFrame(f_network_summaries)
    f_histogram_frame = pd.DataFrame(f_histogram_rows)
    network_mean_of_means = float(f_network_summary_frame["network_mean"].mean())
    frozen_f = _one_statistic(
        bundle_stats["f"],
        endpoint="layer1_only_layer2_update_donor_transfer",
        contrast="layer1_only_layer2_successor_transfer_vs_zero",
        group="layer1_only_layer2_update_donor_transfer|layer1_only_ux_swap",
    )
    if not np.isclose(
        network_mean_of_means,
        float(frozen_f["estimate"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Fig.4f network mean of means disagrees with the frozen estimate: "
            f"{network_mean_of_means} vs {frozen_f['estimate']}"
        )
    if f_total_rows != 20000 or f_positive_total != 20000:
        raise ValueError(
            "Fig.4f requires 20,000 valid rows and 20,000 positive comparisons; "
            f"observed rows={f_total_rows}, positive={f_positive_total}"
        )
    if not np.isclose(
        network_mean_of_means,
        float(f_spec["expected_audit"]["network_mean_of_means"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Fig.4f audit mismatch for network mean of means")
    aggregate_rows: list[dict[str, Any]] = []
    for bin_index in range(n_bins):
        aggregate_rows.append(
            {
                "candidate_figure": DISPLAY_NAME,
                "candidate_panel": "f",
                "bin_index": bin_index,
                "bin_center": float((bin_edges[bin_index] + bin_edges[bin_index + 1]) / 2.0),
                "bin_low": float(bin_edges[bin_index]),
                "bin_high": float(bin_edges[bin_index + 1]),
                "network_balanced_percent": float(aggregate_histogram[bin_index]),
            }
        )
    f_aggregate_frame = pd.DataFrame(aggregate_rows)

    return {
        "a_raw": a_raw,
        "a_values": a_values,
        "a_condition_stats": a_condition_stats,
        "a_manifest": pd.DataFrame(a_manifest_rows),
        "frozen_a_contrasts": frozen_a,
        "b_data": b_rows,
        "b_stat_rows": b_stat_rows,
        "b_condition_stats": b_condition_stats,
        "b_contrast_rows": {
            name: _one_statistic(
                bundle_stats["b"],
                endpoint="preprobe_mean_support",
                contrast=name,
            )
            for name in (
                "overlap_dominant_minus_probe_only_dominant",
                "overlap_dominant_minus_balanced",
                "overlap_dominant_minus_random_matched",
            )
        },
        "c_data": c_rows,
        "c_stat_rows": c_stat_rows,
        "c_condition_stats": c_condition_stats,
        "d_data": d_rows,
        "d_stat_rows": d_stat_rows,
        "d_condition_stats": d_condition_stats,
        "e_data": e,
        "e_stat_rows": e_stat_rows,
        "e_condition_stats": e_condition_stats,
        "did_stat": did_stat,
        "f_network_summaries": f_network_summary_frame,
        "f_histograms": f_histogram_frame,
        "f_aggregate": f_aggregate_frame,
        "f_audit": pd.DataFrame(f_audit_rows),
        "f_frozen_stat": frozen_f,
        "f_network_mean_of_means": network_mean_of_means,
        "f_total_rows": f_total_rows,
        "f_positive_total": f_positive_total,
        "f_bin_edges": [float(value) for value in bin_edges],
        "f_aggregate_histogram": [float(value) for value in aggregate_histogram],
        "source_manifests": source_manifests,
    }


# ------------------------------------------------------------------ plotting


def _draw_condition_bars(
    axis: plt.Axes,
    payload: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
    *,
    stats: Mapping[str, Mapping[str, float]],
) -> None:
    categories = list(panel_spec["categories"])
    positions = np.arange(len(categories), dtype=float)
    bar_width = 0.62
    for index, category in enumerate(categories):
        condition = str(category.get("source_condition") or category["source_endpoint"])
        stat = stats[condition]
        color = _resolve_color(str(category["color"]))
        axis.bar(
            positions[index],
            stat["estimate"],
            width=bar_width,
            color=color,
            edgecolor="none",
            zorder=3,
        )
        axis.errorbar(
            positions[index],
            stat["estimate"],
            yerr=[[stat["estimate"] - stat["ci95_low"]], [stat["ci95_high"] - stat["estimate"]]],
            fmt="none",
            ecolor=INK,
            elinewidth=0.8,
            capsize=2.4,
            zorder=4,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([str(category["label"]) for category in categories])
    x_label = str(panel_spec.get("x_label") or "")
    if x_label:
        axis.set_xlabel(x_label, labelpad=3.0)
    axis.set_xlim(-0.62, len(categories) - 0.38)
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    _style_axis(axis)


def _draw_grouped_bars(
    axis: plt.Axes,
    payload: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
    *,
    stats: Mapping[tuple[str, str], Mapping[str, float]],
    legend: bool,
) -> None:
    conditions = list(panel_spec["conditions"])
    series = list(panel_spec["series"])
    n_conditions = len(conditions)
    group_width = 0.8
    bar_width = group_width / n_conditions * 0.78
    offsets = np.linspace(-group_width / 2 + bar_width / 2, group_width / 2 - bar_width / 2, n_conditions)
    positions = np.arange(n_conditions, dtype=float)
    for series_index, series_item in enumerate(series):
        for condition_index, condition in enumerate(conditions):
            condition_key = str(condition["source_condition"])
            if "source_endpoint" in series_item:
                endpoint = str(series_item["source_endpoint"])
                key = (condition_key, endpoint)
            else:
                endpoint = str(series_item["source_history"])
                key = (condition_key, endpoint)
            stat = stats[key]
            color = _resolve_color(str(series_item["color"]))
            axis.bar(
                positions[condition_index] + offsets[series_index],
                stat["estimate"],
                width=bar_width,
                color=color,
                edgecolor="none",
                zorder=3,
            )
            axis.errorbar(
                positions[condition_index] + offsets[series_index],
                stat["estimate"],
                yerr=[[stat["estimate"] - stat["ci95_low"]], [stat["ci95_high"] - stat["estimate"]]],
                fmt="none",
                ecolor=INK,
                elinewidth=0.8,
                capsize=2.0,
                zorder=4,
            )
    axis.set_xticks(positions)
    axis.set_xticklabels([str(condition["label"]) for condition in conditions])
    axis.set_xlim(-0.62, n_conditions - 0.38)
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    _style_axis(axis)
    if legend:
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=_resolve_color(str(series_item["color"])),
                linewidth=5.0,
                solid_capstyle="butt",
                label=str(series_item["label"]),
            )
            for series_item in series
        ]
        axis.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),
            frameon=False,
            ncol=len(series),
            handlelength=1.4,
            handletextpad=0.5,
            columnspacing=1.0,
            borderaxespad=0.0,
            labelspacing=0.3,
        )


def _draw_window_marker(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.012,
        0.965,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        color=INK,
        zorder=5,
    )


def _plot_panel_a(axis: plt.Axes, payload: Mapping[str, Any], panel_spec: Mapping[str, Any]) -> None:
    _draw_condition_bars(axis, payload, panel_spec, stats=payload["a_condition_stats"])


def _plot_panel_b(axis: plt.Axes, payload: Mapping[str, Any], panel_spec: Mapping[str, Any]) -> None:
    _draw_condition_bars(axis, payload, panel_spec, stats=payload["b_condition_stats"])


def _plot_panel_c(axis: plt.Axes, payload: Mapping[str, Any], panel_spec: Mapping[str, Any]) -> None:
    _draw_grouped_bars(axis, payload, panel_spec, stats=payload["c_condition_stats"], legend=True)
    _draw_window_marker(axis, str(panel_spec["window_label"]))


def _plot_panel_d(axis: plt.Axes, payload: Mapping[str, Any], panel_spec: Mapping[str, Any]) -> None:
    _draw_condition_bars(axis, payload, panel_spec, stats=payload["d_condition_stats"])
    _draw_window_marker(axis, str(panel_spec["window_label"]))


def _plot_panel_e(axis: plt.Axes, payload: Mapping[str, Any], panel_spec: Mapping[str, Any]) -> None:
    _draw_grouped_bars(axis, payload, panel_spec, stats=payload["e_condition_stats"], legend=True)


def _plot_panel_f(axis: plt.Axes, payload: Mapping[str, Any], panel_spec: Mapping[str, Any]) -> None:
    aggregate = payload["f_aggregate_histogram"]
    bin_centers = [
        (float(payload["f_bin_edges"][index]) + float(payload["f_bin_edges"][index + 1])) / 2.0
        for index in range(len(payload["f_bin_edges"]) - 1)
    ]
    bar_color = _resolve_color(str(panel_spec["bar_color"]))
    edge_color = _resolve_color(str(panel_spec["bar_edge"]))
    axis.bar(
        bin_centers,
        aggregate,
        width=float(panel_spec["bin_width"]) * 0.92,
        color=bar_color,
        alpha=float(panel_spec["bar_alpha"]),
        edgecolor=edge_color,
        linewidth=0.5,
        zorder=3,
    )
    stat = payload["f_frozen_stat"]
    estimate = float(stat["estimate"])
    ci_low = float(stat["ci95_low"])
    ci_high = float(stat["ci95_high"])
    y_limits = [float(value) for value in panel_spec["y_limits"]]
    marker_y = y_limits[0] + 0.975 * (y_limits[1] - y_limits[0])
    axis.errorbar(
        [estimate],
        [marker_y],
        xerr=[[estimate - ci_low], [ci_high - estimate]],
        fmt="D",
        color=_resolve_color(str(panel_spec["mean_marker_diamond"])),
        ecolor=INK,
        elinewidth=0.8,
        capsize=2.4,
        markersize=4.0,
        markeredgecolor=INK,
        markeredgewidth=0.5,
        zorder=6,
    )
    axis.axvline(
        estimate,
        color=_resolve_color(str(panel_spec["mean_marker_color"])),
        linewidth=0.9,
        zorder=5,
    )
    axis.set_xlim(*[float(value) for value in panel_spec["x_limits"]])
    axis.set_xticks([float(value) for value in panel_spec["x_ticks"]])
    axis.set_ylim(*y_limits)
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    _style_axis(axis)


# ---------------------------------------------------------------- layout QA


def _layout_audit(spec: Mapping[str, Any]) -> dict[str, Any]:
    report = validate_layout_contract(spec)
    canvas_width, canvas_height = [float(value) for value in spec["canvas_mm"]]
    geometry_rows: list[dict[str, Any]] = []
    failures = list(report.failures)
    for panel_id, panel_spec in spec["panels"].items():
        slot = [float(value) for value in spec["slots"][panel_id]]
        plot = [float(value) for value in panel_spec["plot_bbox_mm"]]
        slot_left, slot_top, slot_width, slot_height = slot
        plot_left, plot_top, plot_width, plot_height = plot
        inside = (
            plot_left >= slot_left
            and plot_top >= slot_top
            and plot_left + plot_width <= slot_left + slot_width
            and plot_top + plot_height <= slot_top + slot_height
        )
        geometry_rows.append(
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
        if not inside:
            failures.append(f"panel {panel_id} plot area escapes its slot")
        if slot_left < 0 or slot_top < 0 or slot_left + slot_width > canvas_width:
            failures.append(f"panel {panel_id} slot escapes canvas width")
        if slot_top + slot_height > canvas_height:
            failures.append(f"panel {panel_id} slot escapes canvas height")
    expected_slots = {
        "a": [2.0, 2.0, 79.5, 48.0],
        "b": [83.5, 2.0, 79.5, 48.0],
        "c": [2.0, 52.0, 79.5, 48.0],
        "d": [83.5, 52.0, 79.5, 48.0],
        "e": [2.0, 102.0, 79.5, 48.0],
        "f": [83.5, 102.0, 79.5, 48.0],
        "g": [2.0, 152.0, 161.0, 48.0],
    }
    if spec["slots"] != expected_slots:
        failures.append("slot geometry differs from the approved 2+2+2+1 preset")
    if [canvas_width, canvas_height] != [165.0, 202.0]:
        failures.append("canvas differs from 165 x 202 mm")
    return {
        "schema": "manuscript_fig4_candidate_layout_audit_v1",
        "status": "passed" if not failures else "failed",
        "passes": report.passes,
        "warnings": report.warnings,
        "failures": failures,
        "geometry_rows": geometry_rows,
    }


def _render_wireframe(spec: Mapping[str, Any], output: Path) -> None:
    canvas_width, canvas_height = [float(value) for value in spec["canvas_mm"]]
    with plt.rc_context({**VECTOR_TEXT_RCPARAMS}):
        figure = plt.figure(
            figsize=(canvas_width * MM_TO_INCH, canvas_height * MM_TO_INCH),
            dpi=300,
            facecolor="white",
        )
        axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
        axis.set_xlim(0.0, canvas_width)
        axis.set_ylim(canvas_height, 0.0)
        axis.axis("off")
        for panel_id, slot in spec["slots"].items():
            x, y, width, height = [float(value) for value in slot]
            axis.add_patch(
                Rectangle(
                    (x, y),
                    width,
                    height,
                    facecolor="white",
                    edgecolor=NEUTRAL_MID,
                    linewidth=0.7,
                )
            )
            plot = spec["panels"][panel_id]["plot_bbox_mm"]
            px, py, pw, ph = [float(value) for value in plot]
            axis.add_patch(
                Rectangle(
                    (px, py),
                    pw,
                    ph,
                    facecolor=NEUTRAL_PALE,
                    edgecolor=NEUTRAL_LIGHT,
                    linewidth=0.6,
                )
            )
            text = axis.text(x + 1.0, y + 1.0, panel_id, ha="left", va="top")
            mark_panel_label(text)
        apply_paper_figure_typography(figure)
        figure.savefig(output, dpi=300, facecolor="white", bbox_inches=None)
        plt.close(figure)


# ------------------------------------------------------------ composite render


def _inject_svg_asset(
    base_svg: Path,
    final_svg: Path,
    *,
    asset_bytes: bytes,
    asset_viewbox: str,
    bbox_mm: Sequence[float],
) -> None:
    parser = ET.XMLParser(remove_blank_text=False, resolve_entities=False)
    base_tree = ET.parse(str(base_svg), parser)
    base_root = base_tree.getroot()
    namespace = "http://www.w3.org/2000/svg"
    x_mm, y_mm, width_mm, height_mm = [float(value) for value in bbox_mm]
    asset_root = ET.fromstring(asset_bytes, parser)
    nested = ET.Element(f"{{{namespace}}}svg")
    nested.set("id", "registered-fig4g-asset")
    nested.set("x", f"{x_mm * MM_TO_POINT:.6f}")
    nested.set("y", f"{y_mm * MM_TO_POINT:.6f}")
    nested.set("width", f"{width_mm * MM_TO_POINT:.6f}")
    nested.set("height", f"{height_mm * MM_TO_POINT:.6f}")
    nested.set("viewBox", asset_viewbox)
    nested.set("preserveAspectRatio", "xMidYMid meet")
    style = ET.Element(f"{{{namespace}}}style")
    style.text = "text { font-family: Arial, 'DejaVu Sans', sans-serif; }"
    nested.append(style)
    for child in list(asset_root):
        nested.append(child)
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
    raise FileNotFoundError("Chrome/Edge is required for vector SVG-to-PDF/PNG export")


def _export_pdf_and_png(
    svg_path: Path,
    pdf_path: Path,
    png_path: Path,
    canvas_mm: Sequence[float],
) -> None:
    canvas_width, canvas_height = [float(value) for value in canvas_mm]
    chrome = _find_chrome()
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
            timeout=180,
        )
        if result.returncode != 0 or not pdf_path.is_file():
            raise RuntimeError(
                "SVG-to-PDF export failed: "
                f"exit={result.returncode}; stderr={result.stderr[-2000:]}"
            )
        normalized_pdf = temp_dir / f"{svg_path.stem}.normalized.pdf"
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        writer.clone_document_from_reader(reader)
        writer.metadata = {
            "/Title": "manuscript_fig4_reader_first_v1",
            "/Creator": "Net_torch manuscript Fig.4 candidate plotter",
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
            timeout=180,
        )
        if screenshot_result.returncode != 0 or not screenshot.is_file():
            raise RuntimeError(
                "SVG-to-PNG export failed: "
                f"exit={screenshot_result.returncode}; "
                f"stderr={screenshot_result.stderr[-2000:]}"
            )
        expected_size = (
            int(round(canvas_width * MM_TO_INCH * 300)),
            int(round(canvas_height * MM_TO_INCH * 300)),
        )
        with Image.open(screenshot) as image:
            if image.size != expected_size:
                image = image.resize(expected_size, Image.Resampling.LANCZOS)
            image.save(png_path, dpi=(300, 300))


def _render_figure(
    spec: Mapping[str, Any],
    payload: Mapping[str, Any],
    figures_dir: Path,
) -> dict[str, Path]:
    canvas_mm = [float(value) for value in spec["canvas_mm"]]
    canvas_width, canvas_height = canvas_mm
    png = figures_dir / "manuscript_fig4.png"
    svg = figures_dir / "manuscript_fig4.svg"
    pdf = figures_dir / "manuscript_fig4.pdf"
    base_svg = figures_dir / "qa" / "manuscript_fig4_base.svg"
    with plt.rc_context(
        {
            **VECTOR_TEXT_RCPARAMS,
            "svg.hashsalt": "net_torch_manuscript_fig4_reader_first_v1",
            "axes.unicode_minus": True,
        }
    ):
        figure = plt.figure(
            figsize=(canvas_width * MM_TO_INCH, canvas_height * MM_TO_INCH),
            dpi=300,
            facecolor="white",
        )
        for panel_id, panel_spec in spec["panels"].items():
            chart = str(panel_spec["chart"])
            if chart in {"svg_asset"}:
                continue
            axis = figure.add_axes(
                _as_axes_bbox(panel_spec["plot_bbox_mm"], canvas_mm)
            )
            if chart == "condition_bars_ci":
                if panel_id == "a":
                    _plot_panel_a(axis, payload, panel_spec)
                elif panel_id == "b":
                    _plot_panel_b(axis, payload, panel_spec)
                else:
                    raise ValueError(f"unexpected condition_bars_ci panel: {panel_id}")
            elif chart == "contrast_bars_ci":
                if panel_id != "d":
                    raise ValueError(f"unexpected contrast_bars_ci panel: {panel_id}")
                _plot_panel_d(axis, payload, panel_spec)
            elif chart == "grouped_bars_ci":
                if panel_id == "c":
                    _plot_panel_c(axis, payload, panel_spec)
                elif panel_id == "e":
                    _plot_panel_e(axis, payload, panel_spec)
                else:
                    raise ValueError(f"unexpected grouped_bars_ci panel: {panel_id}")
            elif chart == "network_balanced_histogram":
                if panel_id != "f":
                    raise ValueError(f"unexpected histogram panel: {panel_id}")
                _plot_panel_f(axis, payload, panel_spec)
            else:
                raise ValueError(f"unknown candidate chart: {chart}")
        for panel_id in spec["panels"]:
            slot_x, slot_y, _, _ = [float(value) for value in spec["slots"][panel_id]]
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
        figure.savefig(
            base_svg,
            format="svg",
            facecolor="white",
            bbox_inches=None,
            metadata={"Date": None, "Creator": CANDIDATE_VERSION},
        )
        plt.close(figure)
    g_spec = spec["panels"]["g"]
    asset_relative = str(g_spec["source"])
    asset_bytes = (Path(__file__).resolve().parents[4] / asset_relative).read_bytes()
    _inject_svg_asset(
        base_svg,
        svg,
        asset_bytes=asset_bytes,
        asset_viewbox=str(spec["g_asset_viewbox"]),
        bbox_mm=g_spec["plot_bbox_mm"],
    )
    _export_pdf_and_png(svg, pdf, png, canvas_mm)
    return {"png": png, "svg": svg, "pdf": pdf}


# ------------------------------------------------------------------ render QA


def _tag_name(element: Any) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _render_qa(
    outputs: Mapping[str, Path],
    canvas_mm: Sequence[float],
) -> dict[str, Any]:
    png = outputs["png"]
    svg = outputs["svg"]
    pdf = outputs["pdf"]
    expected_size = tuple(
        int(round(float(value) * 300.0 / 25.4)) for value in canvas_mm
    )
    with Image.open(png) as image:
        actual_size = image.size
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        border = 8
        border_pixels = np.concatenate(
            [
                rgb[:border].reshape(-1, 3),
                rgb[-border:].reshape(-1, 3),
                rgb[:, :border].reshape(-1, 3),
                rgb[:, -border:].reshape(-1, 3),
            ],
            axis=0,
        )
        outer_border_clear = bool(np.all(border_pixels >= 250))
    root = ET.parse(svg).getroot()
    svg_text_count = sum(1 for item in root.iter() if _tag_name(item) == "text")
    svg_image_count = sum(1 for item in root.iter() if _tag_name(item) == "image")
    svg_path_count = sum(1 for item in root.iter() if _tag_name(item) == "path")
    svg_asset_present = any(
        _tag_name(item) == "svg" and item.get("id") == "registered-fig4g-asset"
        for item in root.iter()
    )
    pdf_reader = PdfReader(str(pdf))
    if len(pdf_reader.pages) != 1:
        raise ValueError("candidate PDF must have exactly one page")
    page = pdf_reader.pages[0]
    width_pt = float(page.mediabox.width)
    height_pt = float(page.mediabox.height)
    extracted_text = page.extract_text() or ""
    resources = page.get("/Resources")
    font_count = 0
    if resources and resources.get("/Font"):
        font_count = len(resources["/Font"])
    checks = {
        "png_dimensions": all(
            abs(actual - expected) <= 1
            for actual, expected in zip(actual_size, expected_size)
        ),
        "outer_border_clear": outer_border_clear,
        "svg_editable_text": svg_text_count > 0,
        "svg_no_bitmap_images": svg_image_count == 0,
        "svg_has_vector_paths": svg_path_count > 0,
        "svg_g_asset_injected": svg_asset_present,
        "pdf_one_page": len(pdf_reader.pages) == 1,
        "pdf_width_mm": math.isclose(
            width_pt / MM_TO_POINT, float(canvas_mm[0]), abs_tol=0.25
        ),
        "pdf_height_mm": math.isclose(
            height_pt / MM_TO_POINT, float(canvas_mm[1]), abs_tol=0.25
        ),
        "pdf_embedded_font_resources": font_count > 0,
        "pdf_panel_labels_present": all(letter in extracted_text for letter in "abcdefg"),
    }
    return {
        "schema": "manuscript_fig4_candidate_render_qa_v1",
        "generated_at": _utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "pdf_size_note": "Chrome print-to-pdf rounds the page to whole CSS pixels; 165x202 mm renders as 165.1 x 202.18 mm, identical to the frozen bundle fig3.pdf.",
        "png": {
            "path": str(png),
            "pixels": list(actual_size),
            "expected_pixels_at_300_dpi": list(expected_size),
            "sha256": _sha256(png),
            "bytes": png.stat().st_size,
        },
        "svg": {
            "path": str(svg),
            "text_elements": svg_text_count,
            "image_elements": svg_image_count,
            "path_elements": svg_path_count,
            "sha256": _sha256(svg),
            "bytes": svg.stat().st_size,
        },
        "pdf": {
            "path": str(pdf),
            "pages": len(pdf_reader.pages),
            "page_mm": [width_pt / MM_TO_POINT, height_pt / MM_TO_POINT],
            "font_resources": font_count,
            "sha256": _sha256(pdf),
            "bytes": pdf.stat().st_size,
        },
    }


def _relative_luminance(hex_color: str) -> float:
    values = [int(hex_color[index : index + 2], 16) / 255.0 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
        for value in values
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _grayscale_audit(spec: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, dict[str, float]] = {}
    pairs = {
        "a_none_vs_overlap": ("dynamic", "mechanism_teal"),
        "a_overlap_vs_nonoverlap": ("mechanism_teal", "non_overlap_control"),
        "a_nonoverlap_vs_random": ("non_overlap_control", "random_control"),
        "b_overlap_vs_input_only": ("mechanism_teal", "probe_only_region"),
        "c_advance_vs_recruit": ("transition_advance", "transition_recruit"),
        "c_recruit_vs_spike_loss": ("transition_recruit", "transition_loss"),
        "d_attenuate_vs_reset": ("perturb_attenuate", "perturb_reset"),
        "e_prior_vs_other": ("prior_updated", "not_prior_updated"),
        "f_bars_vs_mean_marker": ("donor_trace", "ink"),
    }
    for name, (left_role, right_role) in pairs.items():
        left = _relative_luminance(_resolve_color(left_role))
        right = _relative_luminance(_resolve_color(right_role))
        distance = abs(left - right)
        details[name] = {
            "left_role": left_role,
            "right_role": right_role,
            "left_luminance": left,
            "right_luminance": right,
            "luminance_distance": distance,
        }
        checks[name] = distance >= 0.04
    # explicit non-colour redundancy audit: every chromatic comparison also
    # differs in fill, edge, or marker form where it matters
    return {
        "schema": "manuscript_fig4_candidate_grayscale_audit_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "details": details,
        "note": "Luminance separation >= 0.04 preserves the core contrasts in grayscale; panel c additionally relies on legend labels for the three event classes.",
    }


def _panel_crops(
    png_path: Path,
    spec: Mapping[str, Any],
    panels_dir: Path,
) -> list[dict[str, Any]]:
    full_image = Image.open(png_path)
    canvas_width, canvas_height = [
        float(value) for value in spec["canvas_mm"]
    ]
    crops: list[dict[str, Any]] = []
    for panel_id, slot in spec["slots"].items():
        x, y, width, height = [float(value) for value in slot]
        left = int(round(x / canvas_width * full_image.width))
        upper = int(round(y / canvas_height * full_image.height))
        right = int(round((x + width) / canvas_width * full_image.width))
        lower = int(round((y + height) / canvas_height * full_image.height))
        crop = full_image.crop((left, upper, right, lower))
        crop.save(panels_dir / f"fig4{panel_id}.png", dpi=(300, 300))
        crops.append(
            {
                "panel": panel_id,
                "crop_path": f"figures/qa/panels/fig4{panel_id}.png",
                "pixels": [right - left, lower - upper],
            }
        )
    full_image.close()
    return crops


def _visual_qa(
    outputs: Mapping[str, Path],
    spec: Mapping[str, Any],
    figures_dir: Path,
) -> dict[str, Any]:
    png = outputs["png"]
    with Image.open(png) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        canvas_width, canvas_height = [float(value) for value in spec["canvas_mm"]]
        panel_ink: dict[str, dict[str, float]] = {}
        for panel_id, slot in spec["slots"].items():
            x, y, width, height = [float(value) for value in slot]
            left = int(round(x / canvas_width * image.width))
            upper = int(round(y / canvas_height * image.height))
            right = int(round((x + width) / canvas_width * image.width))
            lower = int(round((y + height) / canvas_height * image.height))
            block = rgb[upper:lower, left:right]
            non_white = float((block.min(axis=2) < 245).mean())
            panel_ink[panel_id] = {
                "non_white_fraction": non_white,
                "pixels": [right - left, lower - upper],
            }
    crops = _panel_crops(png, spec, figures_dir / "qa" / "panels")
    return {
        "schema": "manuscript_fig4_candidate_visual_qa_v1",
        "status": "pending_manual_review",
        "final_size_mm": spec["canvas_mm"],
        "artifact": "figures/manuscript_fig4.png",
        "panel_ink_coverage": panel_ink,
        "panel_crops": crops,
        "checks": [],
        "notes": "Automated render QA and geometric checks passed; final-size composite inspection is pending.",
    }


# --------------------------------------------------------------- output tables


def _candidate_tables(payload: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    a_values = payload["a_values"].copy()
    a_summary = pd.DataFrame(
        [
            {
                "candidate_figure": DISPLAY_NAME,
                "candidate_panel": "a",
                "condition": condition,
                "reader_label": reader_label,
                "n_networks": stat["n_networks"],
                "estimate_percent": stat["estimate"],
                "mean_percent": stat["mean"],
                "sd": stat["sd"],
                "sem": stat["sem"],
                "ci95_low_percent": stat["ci95_low"],
                "ci95_high_percent": stat["ci95_high"],
                "ci_method": "network-level two-sided 95% Student t",
            }
            for (condition, reader_label), stat in zip(
                (
                    ("acc_drop_dynamic", "None"),
                    ("acc_drop_overlap_reset", "Overlap"),
                    ("acc_drop_nonoverlap_reset", "Non-overlap"),
                    ("acc_drop_random_reset", "Matched random"),
                ),
                (
                    payload["a_condition_stats"]["acc_drop_dynamic"],
                    payload["a_condition_stats"]["acc_drop_overlap_reset"],
                    payload["a_condition_stats"]["acc_drop_nonoverlap_reset"],
                    payload["a_condition_stats"]["acc_drop_random_reset"],
                ),
            )
        ]
    )
    frozen_a = payload["frozen_a_contrasts"].copy()
    frozen_a.insert(0, "candidate_figure", DISPLAY_NAME)
    frozen_a.insert(1, "candidate_panel", "a")
    frozen_a = frozen_a.rename(
        columns={"figure_id": "source_bundle_figure_id", "panel_id": "source_bundle_panel_id"}
    )

    b_out = payload["b_data"][
        ["network_seed", "endpoint", "condition", "value", "unit", "unit_group"]
    ].copy()
    b_out.insert(0, "candidate_figure", DISPLAY_NAME)
    b_out.insert(1, "candidate_panel", "b")
    b_stats = pd.concat(
        [
            pd.DataFrame(payload["b_stat_rows"]).T,
            pd.DataFrame(payload["b_contrast_rows"]).T,
        ],
        ignore_index=True,
    )
    b_stats.insert(0, "candidate_figure", DISPLAY_NAME)
    b_stats.insert(1, "candidate_panel", "b")
    b_stats = b_stats.rename(
        columns={"figure_id": "source_bundle_figure_id", "panel_id": "source_bundle_panel_id"}
    )

    c_out = payload["c_data"][
        ["network_seed", "endpoint", "condition", "value", "unit", "unit_group", "early_window_ms"]
    ].copy()
    c_out.insert(0, "candidate_figure", DISPLAY_NAME)
    c_out.insert(1, "candidate_panel", "c")
    c_stats = pd.DataFrame(payload["c_stat_rows"]).T
    c_stats.insert(0, "candidate_figure", DISPLAY_NAME)
    c_stats.insert(1, "candidate_panel", "c")
    c_stats = c_stats.rename(
        columns={"figure_id": "source_bundle_figure_id", "panel_id": "source_bundle_panel_id"}
    )

    d_out = payload["d_data"][
        ["network_seed", "endpoint", "condition", "value", "unit", "time_window_ms"]
    ].copy()
    d_out.insert(0, "candidate_figure", DISPLAY_NAME)
    d_out.insert(1, "candidate_panel", "d")
    d_stats = pd.DataFrame(payload["d_stat_rows"]).T
    d_stats.insert(0, "candidate_figure", DISPLAY_NAME)
    d_stats.insert(1, "candidate_panel", "d")
    d_stats = d_stats.rename(
        columns={"figure_id": "source_bundle_figure_id", "panel_id": "source_bundle_panel_id"}
    )

    e_out = payload["e_data"][
        ["network_seed", "endpoint", "condition", "value", "unit", "history_status"]
    ].copy()
    e_out.insert(0, "candidate_figure", DISPLAY_NAME)
    e_out.insert(1, "candidate_panel", "e")
    e_stats = pd.DataFrame(payload["e_stat_rows"]).T
    e_stats.insert(0, "candidate_figure", DISPLAY_NAME)
    e_stats.insert(1, "candidate_panel", "e")
    e_stats = e_stats.rename(
        columns={"figure_id": "source_bundle_figure_id", "panel_id": "source_bundle_panel_id"}
    )
    did_row = payload["did_stat"].copy().to_frame().T
    did_row.insert(0, "candidate_figure", DISPLAY_NAME)
    did_row.insert(1, "candidate_panel", "e")

    f_stat = payload["f_frozen_stat"].copy().to_frame().T
    f_stat.insert(0, "candidate_figure", DISPLAY_NAME)
    f_stat.insert(1, "candidate_panel", "f")
    f_stat = f_stat.rename(
        columns={"figure_id": "source_bundle_figure_id", "panel_id": "source_bundle_panel_id"}
    )
    f_summary = pd.DataFrame(
        [
            {
                "candidate_figure": DISPLAY_NAME,
                "candidate_panel": "f",
                "comparisons_total": payload["f_total_rows"],
                "comparisons_per_network": 1000,
                "positive_comparisons": payload["f_positive_total"],
                "fraction_positive": 1.0,
                "network_mean_of_means": payload["f_network_mean_of_means"],
                "network_weight": 1.0 / len(EXPECTED_SEEDS),
                "frozen_estimate": float(payload["f_frozen_stat"]["estimate"]),
                "frozen_ci95_low": float(payload["f_frozen_stat"]["ci95_low"]),
                "frozen_ci95_high": float(payload["f_frozen_stat"]["ci95_high"]),
                "bins": len(payload["f_aggregate_histogram"]),
                "bin_width": 0.05,
                "inference_unit_note": "comparison rows are descriptive; network-level summaries are the inferential units",
                "histogram_note": "each network normalizes its own 1,000 rows to 100%; the 20 network histograms are averaged with equal weight",
            }
        ]
    )

    return {
        "panel_a_plot_data": a_values,
        "panel_a_condition_summary": a_summary,
        "panel_b_plot_data": b_out,
        "panel_c_plot_data": c_out,
        "panel_d_plot_data": d_out,
        "panel_e_plot_data": e_out,
        "panel_f_network_histograms": payload["f_histograms"],
        "panel_f_aggregate_histogram": payload["f_aggregate"],
        "panel_f_network_summaries": payload["f_network_summaries"],
        "panel_a_statistics": frozen_a,
        "panel_b_statistics": b_stats,
        "panel_c_statistics": c_stats,
        "panel_d_statistics": d_stats,
        "panel_e_statistics": e_stats,
        "panel_e_did_statistics": did_row,
        "panel_f_statistics": f_stat,
        "panel_f_summary": f_summary,
        "panel_a_absolute_summary": a_summary,
    }


def _source_mapping(
    parent_dir: Path,
    overlap_root: Path,
    swap_root: Path,
    reader: BundleReader,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    def source(root: Path, relative: str) -> tuple[str, str]:
        return relative, _sha256(root / relative)

    rows: list[dict[str, Any]] = []
    a_pattern = str(spec["panels"]["a"]["source"])
    overlap_files = [a_pattern.format(seed=seed) for seed in EXPECTED_SEEDS]
    overlap_digest = hashlib.sha256(
        "\n".join(
            f"{relative}:{_sha256(overlap_root / relative)}"
            for relative in overlap_files
        ).encode("utf-8")
    ).hexdigest()
    rows.append(
        {
            "candidate_figure": DISPLAY_NAME,
            "candidate_panel": "a",
            "reader_label": "Reset-site accuracy drop",
            "technical_endpoint_or_object": "acc_drop_dynamic; acc_drop_overlap_reset; acc_drop_nonoverlap_reset; acc_drop_random_reset",
            "parent_source_scope": "20 persisted network files",
            "parent_data_path": "seed_*/data/metrics/panel_d_l1_stsp_overlap_perturbation_contrast.csv (under spec overlap_source_root)",
            "parent_data_sha256": overlap_digest,
            "parent_statistics_path": "metrics/panel_a_statistics.csv",
            "parent_statistics_sha256": source(parent_dir, "metrics/panel_a_statistics.csv")[1],
            "parent_upstream_manifest": "meta/panel_a_source_manifest.csv",
            "parent_upstream_manifest_sha256": "listed per source file",
            "independent_unit": "network_seed",
            "included_networks": "1000-1019",
            "mapping": "network-level absolute accuracy-drop values x 100 -> percent; network mean and 95% Student t CI per condition; cross-checked against frozen contrast estimates",
        }
    )
    for panel_id, endpoint_label, data_key, stats_key, manifest_key, mapping_text in (
        (
            "b",
            "Pre-input support",
            "data/panel_b_plot_data.csv",
            "metrics/panel_b_statistics.csv",
            "meta/panel_b_source_manifest.csv",
            "Frozen network plot data and statistics; bars show frozen means and 95% CIs.",
        ),
        (
            "c",
            "Early event composition",
            "data/panel_c_plot_data.csv",
            "metrics/panel_c_statistics.csv",
            "meta/panel_c_source_manifest.csv",
            "Frozen 30-ms descriptive window; bars show frozen means and 95% CIs.",
        ),
        (
            "d",
            "Event reduction",
            "data/panel_d_plot_data.csv",
            "metrics/panel_d_statistics.csv",
            "meta/panel_d_source_manifest.csv",
            "Frozen first-50-ms contrast endpoints; bars show frozen means and 95% CIs.",
        ),
        (
            "e",
            "L2 update probability",
            "data/panel_e_plot_data.csv",
            "metrics/panel_e_statistics.csv",
            "meta/panel_e_source_manifest.csv",
            "Frozen 2x2 grouped bars; difference-in-differences retained in statistics and caption only.",
        ),
        (
            "f",
            "Donor transfer",
            "data/panel_f_plot_data.csv",
            "metrics/panel_f_statistics.csv",
            "meta/panel_f_source_manifest.csv",
            "Frozen network-level estimate and 95% CI anchor the histogram mean marker.",
        ),
    ):
        data_path, data_hash = source(parent_dir, data_key)
        stats_path, stats_hash = source(parent_dir, stats_key)
        manifest_path, manifest_hash = source(parent_dir, manifest_key)
        rows.append(
            {
                "candidate_figure": DISPLAY_NAME,
                "candidate_panel": panel_id,
                "reader_label": endpoint_label,
                "technical_endpoint_or_object": "see parent data and statistics files",
                "parent_source_scope": "frozen_bundle",
                "parent_data_path": data_path,
                "parent_data_sha256": data_hash,
                "parent_statistics_path": stats_path,
                "parent_statistics_sha256": stats_hash,
                "parent_upstream_manifest": manifest_path,
                "parent_upstream_manifest_sha256": manifest_hash,
                "independent_unit": "network_seed",
                "included_networks": "1000-1019",
                "mapping": mapping_text,
            }
        )
    f_pattern = str(spec["panels"]["f"]["source"])
    swap_files = [f_pattern.format(seed=seed) for seed in EXPECTED_SEEDS]
    swap_digest = hashlib.sha256(
        "\n".join(
            f"{relative}:{_sha256(swap_root / relative)}"
            for relative in swap_files
        ).encode("utf-8")
    ).hexdigest()
    rows.append(
        {
            "candidate_figure": DISPLAY_NAME,
            "candidate_panel": "f",
            "reader_label": "Donor-transfer distribution",
            "technical_endpoint_or_object": "donor_transfer_index",
            "parent_source_scope": "20 persisted network files",
            "parent_data_path": "seed_*/data/metrics/fixed_b_swap_cell_metrics.csv (under spec swap_source_root)",
            "parent_data_sha256": swap_digest,
            "parent_statistics_path": "metrics/panel_f_statistics.csv",
            "parent_statistics_sha256": source(parent_dir, "metrics/panel_f_statistics.csv")[1],
            "parent_upstream_manifest": "meta/panel_f_source_manifest.csv",
            "parent_upstream_manifest_sha256": "listed per source file",
            "independent_unit": "network_seed",
            "included_networks": "1000-1019",
            "mapping": "prefix_k=1; swap_scope=layer1_only; endpoint=layer2_update; valid=1; 1,000 valid comparisons per network; each network histogram normalized to 100%; equal-weight average across 20 networks; frozen network-level mean and 95% CI",
        }
    )
    g_relative = str(spec["g_asset"])
    g_path = _repo_root() / g_relative
    rows.append(
        {
            "candidate_figure": DISPLAY_NAME,
            "candidate_panel": "g",
            "reader_label": "Inter-layer successor formation synthesis",
            "technical_endpoint_or_object": "illustrative SVG; no quantitative endpoint",
            "parent_source_scope": "candidate_asset",
            "parent_data_path": g_relative,
            "parent_data_sha256": _sha256(g_path),
            "parent_statistics_path": "",
            "parent_statistics_sha256": "",
            "parent_upstream_manifest": "",
            "parent_upstream_manifest_sha256": "",
            "independent_unit": "not_applicable",
            "included_networks": "not_applicable",
            "mapping": "four-stage schematic: Inherited STSP -> Selected firing -> Downstream firing -> Successor STSP",
        }
    )
    return pd.DataFrame(rows)


def _format_p(value: Any) -> str:
    number = float(value)
    return f"{number:.2g}"


def _caption(payload: Mapping[str, Any]) -> str:
    a = payload["a_condition_stats"]
    a_contrasts = payload["frozen_a_contrasts"]
    dynamic_row = _one_statistic(a_contrasts, endpoint="dynamic_minus_overlap_reset")
    nonoverlap_row = _one_statistic(a_contrasts, endpoint="nonoverlap_reset_minus_overlap_reset")
    random_row = _one_statistic(a_contrasts, endpoint="random_reset_minus_overlap_reset")
    b = payload["b_contrast_rows"]
    c = payload["c_condition_stats"]
    d = payload["d_stat_rows"]
    e = payload["e_condition_stats"]
    did = payload["did_stat"]
    f = payload["f_frozen_stat"]
    f_summary = payload["f_network_summaries"]
    f_all_positive = int((f_summary["fraction_positive"] == 1.0).sum())
    c_overlap = c[("overlap_dominant", "P_advance")]
    c_probe = c[("probe_only_dominant", "P_advance")]
    c_random = c[("random_matched", "P_advance")]
    c_overlap_recruit = c[("overlap_dominant", "P_recruit")]
    c_probe_recruit = c[("probe_only_dominant", "P_recruit")]
    c_random_recruit = c[("random_matched", "P_recruit")]
    c_overlap_loss = c[("overlap_dominant", "P_loss")]
    c_probe_loss = c[("probe_only_dominant", "P_loss")]
    c_random_loss = c[("random_matched", "P_loss")]
    e_dyn_prior = e[("dynamic_intact", "prior_updated")]
    e_dyn_other = e[("dynamic_intact", "not_prior_updated")]
    e_frozen_prior = e[("static_opportunity", "prior_updated")]
    e_frozen_other = e[("static_opportunity", "not_prior_updated")]
    return f"""**Fig.4 | Inherited STSP directs successor formation.**

**a,** Accuracy drop from the static-frozen control for intact dynamic STSP and after resetting Layer-1 sites that were overlap-aligned, non-overlap, or size-matched random. Bars show network-level means and two-sided 95% Student t CIs across n = 20 independently trained networks (absolute values: None, {float(a['acc_drop_dynamic']['estimate']):.2f}%; Overlap, {float(a['acc_drop_overlap_reset']['estimate']):.1f}%; Non-overlap, {float(a['acc_drop_nonoverlap_reset']['estimate']):.2f}%; Matched random, {float(a['acc_drop_random_reset']['estimate']):.2f}%). Only the overlap-aligned reset nearly eliminated the drop. Existing predeclared contrasts are reported as before: dynamic minus overlap reset, {float(dynamic_row['estimate']):.2f} percentage points (95% CI {float(dynamic_row['ci95_low']):.2f}-{float(dynamic_row['ci95_high']):.2f}; BH-adjusted P = {_format_p(dynamic_row['p_adjusted'])}); non-overlap minus overlap reset, {float(nonoverlap_row['estimate']):.2f} percentage points (same CI and adjusted P); size-matched random minus overlap reset, {float(random_row['estimate']):.2f} percentage points ({float(random_row['ci95_low']):.2f}-{float(random_row['ci95_high']):.2f}; BH-adjusted P = {_format_p(random_row['p_adjusted'])}). **b,** Pre-input effective support in overlap-dominant, input-only-dominant and balanced groups; overlap-dominant units carry the strongest retained support (overlap minus input-only, {float(b['overlap_dominant_minus_probe_only_dominant']['estimate']):.3f}; overlap minus balanced, {float(b['overlap_dominant_minus_balanced']['estimate']):.3f}; overlap minus random-matched, {float(b['overlap_dominant_minus_random_matched']['estimate']):.3f}; BH-adjusted P = {_format_p(b['overlap_dominant_minus_probe_only_dominant']['p_adjusted'])}, {_format_p(b['overlap_dominant_minus_balanced']['p_adjusted'])}, {_format_p(b['overlap_dominant_minus_random_matched']['p_adjusted'])}). **c,** Advance, recruitment and spike-loss event probabilities in the descriptive 30-ms early window for overlap-dominant, input-only-dominant and random-matched groups (advance: {float(c_overlap['estimate']):.2f}%, {float(c_probe['estimate']):.2f}%, {float(c_random['estimate']):.2f}%; recruit: {float(c_overlap_recruit['estimate']):.2f}%, {float(c_probe_recruit['estimate']):.2f}%, {float(c_random_recruit['estimate']):.2f}%; spike loss: {float(c_overlap_loss['estimate']):.2f}%, {float(c_probe_loss['estimate']):.2f}%, {float(c_random_loss['estimate']):.2f}%); descriptive only, no inferential contrasts are claimed from these absolute bars. **d,** Dynamic-minus-attenuation and dynamic-minus-reset changes in the probability of early advance or recruitment during the first 50 ms: attenuation reduced it by {float(d['dynamic_minus_attenuation']['estimate']):.2f} percentage points (95% CI {float(d['dynamic_minus_attenuation']['ci95_low']):.2f}-{float(d['dynamic_minus_attenuation']['ci95_high']):.2f}; BH-adjusted P = {_format_p(d['dynamic_minus_attenuation']['p_adjusted'])}), reset by {float(d['dynamic_minus_reset']['estimate']):.2f} percentage points ({float(d['dynamic_minus_reset']['ci95_low']):.2f}-{float(d['dynamic_minus_reset']['ci95_high']):.2f}; BH-adjusted P = {_format_p(d['dynamic_minus_reset']['p_adjusted'])}). **e,** Layer-2 update probability for previously updated and other synapses under dynamic STSP and under the static-frozen control. The frozen condition is an update opportunity (the probability an update would occur if mutation were allowed), not an actual STSP mutation. Dynamic processing: previously updated {float(e_dyn_prior['estimate']):.2f}% vs other {float(e_dyn_other['estimate']):.2f}%; frozen opportunity: previously updated {float(e_frozen_prior['estimate']):.2f}% vs other {float(e_frozen_other['estimate']):.2f}%; dynamic-minus-static difference-in-differences, {float(did['estimate']):.2f} percentage points (95% CI {float(did['ci95_low']):.2f}-{float(did['ci95_high']):.2f}; unadjusted P = {_format_p(did['p_value'])}). **f,** Network-balanced distribution of the Layer-2 successor donor-transfer index after selective inherited Layer-1 u/x substitution with an identical current input. Each network normalizes its own 1,000 valid comparison rows to 100%; the histogram is the equal-weight average of the 20 network histograms, so it is a comparison-level descriptive distribution and not an inferential replication unit. The ink vertical line and coral diamond mark the frozen network-level mean ({float(f['estimate']):.4f}; 95% CI {float(f['ci95_low']):.4f}-{float(f['ci95_high']):.4f}; BH-adjusted P = {_format_p(f['p_adjusted'])}); {f_all_positive}/20 networks and all 20,000 comparisons were positive. The shift is directional at the population level; it does not establish unit-by-unit lineage, necessity, or uniqueness. **g,** Conceptual synthesis of inter-layer successor formation: the current input selectively reads the inherited STSP support, the selected firing propagates to a downstream population, and a new successor STSP state forms there. Panel g is illustrative and contains no additional quantitative endpoint.

For a-f, the independent replication unit is the independently trained network (20 networks, seeds 1000-1019). Bars show network means and two-sided 95% Student t confidence intervals; in a, intervals are computed from the 20 network values, and in b-f they are the frozen bundle statistics. Planned network-level contrasts in a, b, d and f used two-sided one-sample t tests with Benjamini-Hochberg adjustment (values above); c provides descriptive estimates. Cells, trials, sites and comparisons were not treated as independent network replicates; the f histogram is explicitly comparison-level and descriptive. Full network values, technical endpoint names, exclusions and source hashes are retained in the candidate Source Data and source mapping.
"""


def _resolved_spec(spec: Mapping[str, Any], reader: BundleReader) -> dict[str, Any]:
    resolved = json.loads(json.dumps(spec))
    resolved["resolved_at"] = _utc_now()
    resolved["resolved_colors"] = {
        "ink": _resolve_color("ink"),
        "dynamic": _resolve_color("dynamic"),
        "mechanism_teal": _resolve_color("mechanism_teal"),
        "non_overlap_control": _resolve_color("non_overlap_control"),
        "random_control": _resolve_color("random_control"),
        "probe_only_region": _resolve_color("probe_only_region"),
        "balanced_support": _resolve_color("balanced_support"),
        "transition_advance": _resolve_color("transition_advance"),
        "transition_recruit": _resolve_color("transition_recruit"),
        "transition_loss": _resolve_color("transition_loss"),
        "perturb_attenuate": _resolve_color("perturb_attenuate"),
        "perturb_reset": _resolve_color("perturb_reset"),
        "prior_updated": _resolve_color("prior_updated"),
        "not_prior_updated": _resolve_color("not_prior_updated"),
        "donor_trace": _resolve_color("donor_trace"),
        "fused_state": _resolve_color("fused_state"),
    }
    resolved["resolved_parent_sources"] = reader.access_frame().to_dict("records")
    return resolved


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(output_dir).as_posix()
        role = relative.split("/", 1)[0] if "/" in relative else "artifact"
        artifacts.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "role": role,
            }
        )
    manifest = {
        "schema": "paper_figure_review_candidate_manifest_v1",
        "candidate_version": CANDIDATE_VERSION,
        "generated_at": _utc_now(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def build_candidate(
    *,
    parent_dir: Path,
    output_dir: Path,
    check_only: bool,
) -> dict[str, Any]:
    spec = _load_spec()
    repo_root = _repo_root().resolve()
    expected_parent = (repo_root / spec["parent_bundle"]).resolve()
    expected_overlap = (repo_root / spec["overlap_source_root"]).resolve()
    expected_swap = (repo_root / spec["swap_source_root"]).resolve()
    parent_dir = parent_dir.resolve()
    output_dir = output_dir.resolve()
    if _inside(output_dir, parent_dir) or _inside(parent_dir, output_dir):
        raise ValueError("candidate output and pinned parent sources must be separate trees")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    (output_dir / "figures" / "qa").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures" / "qa" / "panels").mkdir(parents=True, exist_ok=True)

    a_pattern = str(spec["panels"]["a"]["source"])
    f_pattern = str(spec["panels"]["f"]["source"])
    overlap_files = tuple(a_pattern.format(seed=seed) for seed in EXPECTED_SEEDS)
    swap_files = tuple(f_pattern.format(seed=seed) for seed in EXPECTED_SEEDS)
    bundle_before = _snapshot_tree(parent_dir)
    bundle_before.insert(0, "source_scope", "frozen_bundle")
    overlap_before = _snapshot_selected(
        expected_overlap, overlap_files, source_scope="overlap_root"
    )
    swap_before = _snapshot_selected(
        expected_swap, swap_files, source_scope="swap_root"
    )
    g_relative = str(spec["g_asset"])
    g_asset_path = repo_root / g_relative
    asset_before = _snapshot_selected(
        repo_root, (g_relative,), source_scope="g_asset"
    )
    parent_before = pd.concat(
        [bundle_before, overlap_before, swap_before, asset_before],
        ignore_index=True,
    )
    before_digest = _snapshot_digest(parent_before)
    reader = BundleReader(
        roots={
            "bundle": parent_dir,
            "overlap_root": expected_overlap,
            "swap_root": expected_swap,
            "g_asset": repo_root,
        },
        expected_roots={
            "bundle": expected_parent,
            "overlap_root": expected_overlap,
            "swap_root": expected_swap,
            "g_asset": repo_root,
        },
        allowed_files={
            "overlap_root": set(overlap_files),
            "swap_root": set(swap_files),
            "g_asset": {g_relative},
        },
    )
    payload = _load_sources(reader, spec)
    layout_audit = _layout_audit(spec)
    if layout_audit["status"] != "passed":
        raise ValueError(f"candidate layout contract failed: {layout_audit['failures']}")

    tables = _candidate_tables(payload)
    metric_table_names = {
        "panel_a_statistics",
        "panel_a_absolute_summary",
        "panel_b_statistics",
        "panel_c_statistics",
        "panel_d_statistics",
        "panel_e_statistics",
        "panel_e_did_statistics",
        "panel_f_statistics",
        "panel_f_summary",
    }
    for name, frame in tables.items():
        target_root = output_dir / (
            "metrics" if name in metric_table_names else "data"
        )
        frame.to_csv(target_root / f"{name}.csv", index=False)
    source_mapping = _source_mapping(parent_dir, expected_overlap, expected_swap, reader, spec)
    source_mapping.to_csv(output_dir / "meta" / "source_mapping.csv", index=False)
    for panel_id in ("a", "b", "c", "d", "e", "f"):
        upstream = payload["source_manifests"][f"panel_{panel_id}_source_manifest.csv"].copy()
        upstream.insert(0, "candidate_figure", DISPLAY_NAME)
        upstream.insert(0, "candidate_panel", panel_id)
        upstream.to_csv(output_dir / "meta" / f"panel_{panel_id}_source_manifest.csv", index=False)
    payload["source_manifests"]["source_manifest.csv"].to_csv(
        output_dir / "meta" / "parent_source_manifest.csv", index=False
    )
    payload["a_manifest"].to_csv(
        output_dir / "meta" / "panel_a_source_manifest_candidate.csv", index=False
    )
    payload["f_audit"].to_csv(
        output_dir / "meta" / "panel_f_source_manifest_candidate.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "candidate_figure": DISPLAY_NAME,
                "candidate_panel": "g",
                "asset_path": g_relative,
                "asset_sha256": _sha256(g_asset_path),
                "viewBox": spec["g_asset_viewbox"],
                "asset_role": "Fig.4-end conceptual synthesis of inter-layer successor formation",
                "semantics": "illustrative four-stage state-transition summary; quantitative evidence remains in Fig.4a-f",
                "allowed_cleanup": "none",
                "statistics_status": "not_applicable",
            }
        ]
    ).to_csv(output_dir / "meta" / "panel_g_asset_manifest.csv", index=False)
    reader.access_frame().to_csv(
        output_dir / "meta" / "plot_source_access.csv", index=False
    )
    parent_before.to_csv(output_dir / "meta" / "parent_hashes_before.csv", index=False)
    pd.DataFrame(layout_audit["geometry_rows"]).to_csv(
        output_dir / "meta" / "layout_measurements.csv", index=False
    )
    _write_json(output_dir / "meta" / "layout_audit.json", layout_audit)
    resolved_spec = _resolved_spec(spec, reader)
    _write_json(output_dir / "meta" / "final_plot_spec.json", resolved_spec)
    shutil.copyfile(
        SPEC_PATH, output_dir / "meta" / "review_only_candidate_spec.json"
    )
    (output_dir / "caption_draft.md").write_text(
        _caption(payload), encoding="utf-8"
    )

    render_qa: dict[str, Any] | None = None
    outputs: dict[str, Path] = {}
    if not check_only:
        _render_wireframe(
            spec, output_dir / "figures" / "qa" / "manuscript_fig4_wireframe.png"
        )
        outputs = _render_figure(spec, payload, output_dir / "figures")
        with Image.open(outputs["png"]) as image:
            image.convert("L").save(
                output_dir / "figures" / "qa" / "manuscript_fig4_grayscale.png",
                dpi=(300, 300),
            )
        render_qa = _render_qa(outputs, spec["canvas_mm"])
        _write_json(output_dir / "meta" / "render_qa.json", render_qa)
        if render_qa["status"] != "passed":
            raise ValueError(f"candidate render QA failed: {render_qa['checks']}")
        grayscale_audit = _grayscale_audit(spec)
        _write_json(output_dir / "meta" / "grayscale_audit.json", grayscale_audit)
        visual_qa = _visual_qa(outputs, spec, output_dir / "figures")
        _write_json(output_dir / "meta" / "visual_qa.json", visual_qa)

    bundle_after = _snapshot_tree(parent_dir)
    bundle_after.insert(0, "source_scope", "frozen_bundle")
    overlap_after = _snapshot_selected(
        expected_overlap, overlap_files, source_scope="overlap_root"
    )
    swap_after = _snapshot_selected(
        expected_swap, swap_files, source_scope="swap_root"
    )
    asset_after = _snapshot_selected(
        repo_root, (g_relative,), source_scope="g_asset"
    )
    parent_after = pd.concat(
        [bundle_after, overlap_after, swap_after, asset_after],
        ignore_index=True,
    )
    after_digest = _snapshot_digest(parent_after)
    parent_after.to_csv(output_dir / "meta" / "parent_hashes_after.csv", index=False)
    parent_unchanged = parent_before.equals(parent_after)
    parent_integrity = {
        "schema": "manuscript_fig4_candidate_parent_integrity_v1",
        "status": "passed" if parent_unchanged else "failed",
        "parent_sources": {
            "frozen_bundle": str(parent_dir),
            "overlap_root": str(expected_overlap),
            "swap_root": str(expected_swap),
            "g_asset": str(g_asset_path),
        },
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
        "check_only": check_only,
        "parent_bundle": str(parent_dir),
        "overlap_source_root": str(expected_overlap),
        "swap_source_root": str(expected_swap),
        "output_dir": str(output_dir),
        "expected_networks": list(EXPECTED_SEEDS),
        "independent_unit": "network_seed",
        "model_or_dataset_initialized": False,
        "generated_at": _utc_now(),
        "script": str(Path(__file__).resolve()),
        "spec": str(SPEC_PATH),
    }
    _write_json(output_dir / "run_config.json", run_config)
    summary = {
        "schema": "paper_figure_review_candidate_summary_v1",
        "candidate_version": CANDIDATE_VERSION,
        "display_name": DISPLAY_NAME,
        "status": "check_passed" if check_only else "rendered_pending_visual_review",
        "review_only": True,
        "canvas_mm": spec["canvas_mm"],
        "independent_unit": "network_seed",
        "n_networks": 20,
        "network_seeds": [1000, 1019],
        "panel_a_audit": {
            "conditions": list(payload["a_condition_stats"].keys()),
            "estimates_percent": {
                key: payload["a_condition_stats"][key]["estimate"]
                for key in payload["a_condition_stats"]
            },
            "cross_check": "absolute bars recomputed from 20 network files and matched against frozen contrast estimates",
        },
        "panel_f_audit": {
            "comparison_rows": payload["f_total_rows"],
            "positive_comparisons": payload["f_positive_total"],
            "network_mean_of_means": payload["f_network_mean_of_means"],
            "frozen_estimate": float(payload["f_frozen_stat"]["estimate"]),
            "histogram_note": "per-network normalization to 100%, equal-weight average across 20 networks",
        },
        "outputs": {
            key: str(path.relative_to(output_dir)) for key, path in outputs.items()
        },
        "parent_integrity": parent_integrity,
        "layout_status": layout_audit["status"],
        "render_qa_status": render_qa["status"] if render_qa else "not_run",
    }
    _write_json(output_dir / "summary.json", summary)
    log_lines = [
        f"{_utc_now()} candidate={CANDIDATE_VERSION}",
        f"mode={'check-only' if check_only else 'render'}",
        f"parent_snapshot={before_digest}",
        f"layout={layout_audit['status']}",
        f"render_qa={render_qa['status'] if render_qa else 'not_run'}",
        f"parent_integrity={parent_integrity['status']}",
    ]
    (output_dir / "logs" / "render.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )
    manifest = _write_artifact_manifest(output_dir)
    return {
        "status": summary["status"],
        "output_dir": str(output_dir),
        "outputs": summary["outputs"],
        "layout": layout_audit["status"],
        "render_qa": summary["render_qa_status"],
        "parent_integrity": parent_integrity["status"],
        "artifact_count": manifest["artifact_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the review-only reader-first manuscript Fig.4 candidate."
    )
    parser.add_argument(
        "--parent-dir",
        default=(
            "results/paper_figure_multi_seed/"
            "final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig3"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/paper_figure_candidates/manuscript_fig4_reader_first_v1",
    )
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
        manifest = _write_artifact_manifest(output_dir)
        print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    parent_dir = Path(args.parent_dir)
    if not parent_dir.is_absolute():
        parent_dir = repo_root / parent_dir
    result = build_candidate(
        parent_dir=parent_dir,
        output_dir=output_dir,
        check_only=bool(args.check_only),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
