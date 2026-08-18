from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import MaxNLocator
from PIL import Image
from pypdf import PdfReader
from scipy.ndimage import gaussian_filter

from src.plotting.common.colors import NATURE_COMPATIBLE_PALETTE, get_plot_color
from src.plotting.paper_fig.layout_contract import validate_layout_contract
from src.plotting.paper_fig.typography import (
    VECTOR_TEXT_RCPARAMS,
    apply_paper_figure_typography,
    mark_panel_label,
)


CANDIDATE_VERSION = "manuscript_fig3_reader_first_v3"
EXPECTED_SEEDS = tuple(range(1000, 1020))
MM_TO_INCH = 1.0 / 25.4
MM_TO_POINT = 72.0 / 25.4
SPEC_PATH = (
    Path(__file__).resolve().parent
    / "specs"
    / "manuscript_fig3_reader_first_v3.json"
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
    return get_plot_color(role, context="manuscript_fig3")


@dataclass
class BundleReader:
    parent_dir: Path
    expected_parent_dir: Path
    decomposition_dir: Path
    expected_decomposition_dir: Path
    allowed_decomposition_files: tuple[str, ...]
    accesses: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.parent_dir = self.parent_dir.resolve()
        self.expected_parent_dir = self.expected_parent_dir.resolve()
        self.decomposition_dir = self.decomposition_dir.resolve()
        self.expected_decomposition_dir = self.expected_decomposition_dir.resolve()
        if self.parent_dir != self.expected_parent_dir:
            raise ValueError(
                "candidate plotting accepts only the parent bundle pinned by the review spec"
            )
        if self.decomposition_dir != self.expected_decomposition_dir:
            raise ValueError(
                "candidate plotting accepts only the decomposition root pinned by the review spec"
            )
        if not self.parent_dir.is_dir() or not self.decomposition_dir.is_dir():
            raise FileNotFoundError("a pinned parent source root is missing")

    def _resolve_internal(
        self,
        relative: str,
        purpose: str,
        source_scope: str,
    ) -> Path:
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise ValueError(f"absolute source path is forbidden: {relative}")
        if source_scope == "bundle":
            root = self.parent_dir
        elif source_scope == "decomposition":
            root = self.decomposition_dir
            if relative_path.as_posix() not in set(self.allowed_decomposition_files):
                raise PermissionError(f"unregistered decomposition source: {relative}")
        else:
            raise ValueError(f"unknown source scope: {source_scope}")
        path = (root / relative_path).resolve()
        allowed = _inside(path, root)
        if not allowed:
            raise PermissionError(f"plot source escapes its pinned parent root: {path}")
        if path.suffix.lower() not in {".csv", ".json"}:
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


def _stimulus_matrix(stimuli: pd.DataFrame, role: str) -> np.ndarray:
    required = {"stimulus_role", "pixel_x", "pixel_y", "normalized_intensity"}
    missing = sorted(required - set(stimuli.columns))
    if missing:
        raise ValueError(f"stimulus data is missing columns: {missing}")
    part = stimuli.loc[stimuli["stimulus_role"].astype(str).eq(role)].copy()
    if len(part) != 28 * 28:
        raise ValueError(f"stimulus role {role!r} must contain 784 persisted pixels")
    x = pd.to_numeric(part["pixel_x"], errors="raise").astype(int).to_numpy()
    y = pd.to_numeric(part["pixel_y"], errors="raise").astype(int).to_numpy()
    values = pd.to_numeric(
        part["normalized_intensity"], errors="raise"
    ).to_numpy(dtype=float)
    if (
        np.any(x < 0)
        or np.any(x >= 28)
        or np.any(y < 0)
        or np.any(y >= 28)
        or len(set(zip(x.tolist(), y.tolist()))) != 28 * 28
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError(f"stimulus role {role!r} has invalid persisted pixels")
    image = np.zeros((28, 28), dtype=float)
    image[y, x] = values
    return image


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


def _density_threshold(
    probability_mass: np.ndarray,
    target_mass: float,
) -> tuple[float, float]:
    flat = np.asarray(probability_mass, dtype=float).ravel()
    ordered = np.sort(flat)[::-1]
    cumulative = np.cumsum(ordered)
    index = int(np.searchsorted(cumulative, target_mass, side="left"))
    index = min(index, len(ordered) - 1)
    threshold = float(ordered[index])
    achieved = float(flat[flat >= threshold].sum())
    return threshold, achieved


def _build_network_balanced_density(
    joint: pd.DataFrame,
    *,
    x_field: str,
    y_field: str,
    x_reference: float,
    y_reference: float,
    panel_spec: Mapping[str, Any],
) -> dict[str, Any]:
    x_values = pd.to_numeric(joint[x_field], errors="raise").to_numpy(dtype=float)
    y_values = pd.to_numeric(joint[y_field], errors="raise").to_numpy(dtype=float)
    x_low = min(float(x_values.min()), x_reference)
    x_high = float(x_values.max())
    y_low = min(float(y_values.min()), y_reference)
    y_high = float(y_values.max())
    x_span = x_high - x_low
    y_span = y_high - y_low
    x_limits = (x_low - 0.025 * x_span, x_high + 0.035 * x_span)
    y_limits = (y_low - 0.025 * y_span, y_high + 0.035 * y_span)
    x_bins, y_bins = [int(value) for value in panel_spec["grid_bins"]]
    x_edges = np.linspace(x_limits[0], x_limits[1], x_bins + 1)
    y_edges = np.linspace(y_limits[0], y_limits[1], y_bins + 1)
    sigma = float(panel_spec["smoothing_sigma_cells"])
    network_grids: list[np.ndarray] = []
    grid_rows: list[dict[str, Any]] = []
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    for seed in EXPECTED_SEEDS:
        rows = joint.loc[joint["network_seed"].eq(seed)]
        histogram, _, _ = np.histogram2d(
            rows[x_field].to_numpy(dtype=float),
            rows[y_field].to_numpy(dtype=float),
            bins=[x_edges, y_edges],
        )
        if not np.isclose(histogram.sum(), 500.0):
            raise ValueError(f"Fig.3c network {seed} density did not receive 500 rows")
        probability = histogram / histogram.sum()
        probability = gaussian_filter(probability, sigma=sigma, mode="constant")
        probability = probability / probability.sum()
        network_grids.append(probability)
        for x_index, x_center in enumerate(x_centers):
            for y_index, y_center in enumerate(y_centers):
                grid_rows.append(
                    {
                        "candidate_figure": "Fig.3",
                        "candidate_panel": "c",
                        "network_seed": seed,
                        "x_center": float(x_center),
                        "y_center": float(y_center),
                        "probability_mass": float(probability[x_index, y_index]),
                        "network_weight": 1.0 / len(EXPECTED_SEEDS),
                    }
                )
    stack = np.stack(network_grids, axis=0)
    aggregate = stack.mean(axis=0)
    aggregate = aggregate / aggregate.sum()
    aggregate_rows: list[dict[str, Any]] = []
    for x_index, x_center in enumerate(x_centers):
        for y_index, y_center in enumerate(y_centers):
            aggregate_rows.append(
                {
                    "candidate_figure": "Fig.3",
                    "candidate_panel": "c",
                    "x_center": float(x_center),
                    "y_center": float(y_center),
                    "network_balanced_probability_mass": float(
                        aggregate[x_index, y_index]
                    ),
                }
            )
    contour_rows: list[dict[str, Any]] = []
    for mass in [float(value) for value in panel_spec["contour_masses"]]:
        threshold, achieved = _density_threshold(aggregate, mass)
        contour_rows.append(
            {
                "target_probability_mass": mass,
                "density_threshold": threshold,
                "achieved_probability_mass": achieved,
            }
        )
    return {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "aggregate": aggregate,
        "network_grid": pd.DataFrame(grid_rows),
        "aggregate_grid": pd.DataFrame(aggregate_rows),
        "contours": pd.DataFrame(contour_rows),
        "outline_mass": float(
            panel_spec.get("outline_mass", max(panel_spec["contour_masses"]))
        ),
        "x_limits": x_limits,
        "y_limits": y_limits,
    }


def _load_sources(
    reader: BundleReader,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    stimuli = reader.read_csv(
        "data/panel_a_input_stimuli.csv", "Fig.3a persisted stimuli"
    )
    behavior = reader.read_csv(
        "data/panel_b_plot_data.csv", "Fig.3b network opportunity rates"
    )
    updates = reader.read_csv(
        "data/panel_c_plot_data.csv", "Fig.3c-d network update metrics"
    )
    events = reader.read_csv(
        "data/panel_d_plot_data.csv", "Fig.3e network event conditions"
    )
    behavior_stats = reader.read_csv(
        "metrics/panel_b_statistics.csv", "Fig.3b frozen paired statistics"
    )
    update_stats = reader.read_csv(
        "metrics/panel_c_statistics.csv", "Fig.3c-d frozen statistics"
    )
    event_stats = reader.read_csv(
        "metrics/panel_d_statistics.csv", "Fig.3e frozen statistics"
    )
    source_manifests = {
        name: reader.read_csv(f"meta/{name}", f"parent provenance {name}")
        for name in (
            "panel_a_source_manifest.csv",
            "panel_b_source_manifest.csv",
            "panel_c_source_manifest.csv",
            "panel_d_source_manifest.csv",
            "source_manifest.csv",
        )
    }

    panel_c_spec = spec["panels"]["c"]
    source_pattern = str(panel_c_spec["source_pattern"])
    joint_frames: list[pd.DataFrame] = []
    decomposition_manifest_rows: list[dict[str, Any]] = []
    required_joint_columns = {
        "network_seed",
        "prefix_k",
        "valid",
        "history_family_id",
        "b_anchor_id",
        str(panel_c_spec["x_field"]),
        str(panel_c_spec["y_field"]),
    }
    for seed in EXPECTED_SEEDS:
        relative = source_pattern.format(seed=seed)
        raw = reader.read_csv(
            relative,
            f"Fig.3c persisted exact-B decomposition rows for network {seed}",
            source_scope="decomposition",
        )
        missing = sorted(required_joint_columns - set(raw.columns))
        if missing:
            raise ValueError(f"Fig.3c network {seed} is missing columns: {missing}")
        observed_seed = set(
            pd.to_numeric(raw["network_seed"], errors="raise").astype(int)
        )
        if observed_seed != {seed}:
            raise ValueError(
                f"Fig.3c source seed mismatch for network {seed}: {observed_seed}"
            )
        filtered = raw.loc[
            pd.to_numeric(raw["prefix_k"], errors="raise").eq(1)
            & pd.to_numeric(raw["valid"], errors="raise").eq(1)
        ].copy()
        if len(filtered) != 500:
            raise ValueError(
                f"Fig.3c network {seed} requires 500 valid one-step comparisons; "
                f"observed {len(filtered)}"
            )
        if filtered.duplicated(["history_family_id", "b_anchor_id"]).any():
            raise ValueError(f"Fig.3c network {seed} has duplicate comparison keys")
        if filtered[["history_family_id", "b_anchor_id"]].drop_duplicates().shape[0] != 500:
            raise ValueError(f"Fig.3c network {seed} lacks 500 unique comparisons")
        for field in (str(panel_c_spec["x_field"]), str(panel_c_spec["y_field"])):
            values = pd.to_numeric(filtered[field], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Fig.3c network {seed} has non-finite {field}")
            filtered[field] = values
        joint_frames.append(filtered)
        source_path = reader.decomposition_dir / relative
        decomposition_manifest_rows.append(
            {
                "candidate_figure": "Fig.3",
                "candidate_panel": "c",
                "network_seed": seed,
                "source_path": str(source_path),
                "source_relative_path": relative,
                "source_sha256": _sha256(source_path),
                "source_bytes": source_path.stat().st_size,
                "raw_rows": len(raw),
                "filtered_rows": len(filtered),
                "filters": "prefix_k=1; valid=1",
                "comparison_key": "history_family_id × b_anchor_id",
                "independent_unit": "network_seed",
            }
        )
    joint = pd.concat(joint_frames, ignore_index=True)
    if len(joint) != 10000 or joint["network_seed"].nunique() != 20:
        raise ValueError("Fig.3c requires 10000 rows from all 20 networks")
    decomposition_manifest = pd.DataFrame(decomposition_manifest_rows)

    behavior_manifest = source_manifests["panel_b_source_manifest.csv"]
    if "filters" not in behavior_manifest:
        raise ValueError("Fig.3b provenance does not declare its filters")
    has_one_step_filter = behavior_manifest["filters"].fillna("").astype(str).map(
        lambda value: "prefix_k=1"
        in {token.strip() for token in value.split(";")}
    )
    if not has_one_step_filter.all():
        raise ValueError("Fig.3b provenance does not retain the frozen one-step filter")

    identities = (
        stimuli[["stimulus_role", "label"]]
        .drop_duplicates()
        .set_index("stimulus_role")["label"]
        .astype(int)
        .to_dict()
    )
    if identities != {"A": 1, "B": 0, "C": 6}:
        raise ValueError(f"unexpected frozen stimulus identities: {identities}")
    images = {role: _stimulus_matrix(stimuli, role) for role in ("A", "B", "C")}
    image_meta: dict[str, dict[str, Any]] = {}
    for role in ("A", "B", "C"):
        rows = stimuli.loc[stimuli["stimulus_role"].astype(str).eq(role)]
        image_meta[role] = {
            "image_id": int(rows["image_id"].iloc[0]),
            "label": int(rows["label"].iloc[0]),
            "image_sha256": str(rows["image_sha256"].iloc[0]),
            "render_occurrences": int(rows["render_occurrences"].iloc[0]),
        }
    if image_meta["B"]["render_occurrences"] != 2:
        raise ValueError("the pinned B stimulus is not registered for two render occurrences")

    _require_seed_set(behavior, "Fig.3b")
    behavior["network_seed"] = pd.to_numeric(
        behavior["network_seed"], errors="raise"
    ).astype(int)
    behavior["value"] = pd.to_numeric(behavior["value"], errors="raise")
    if behavior.duplicated(
        ["network_seed", "outcome_type", "history_relation"]
    ).any():
        raise ValueError("Fig.3b has duplicate network/outcome/history rows")
    pivot = behavior.pivot(
        index="network_seed",
        columns=["outcome_type", "history_relation"],
        values="value",
    ).reindex(EXPECTED_SEEDS)
    required_columns = pd.MultiIndex.from_product(
        [["rescue", "loss"], ["aligned", "mismatched"]],
        names=["outcome_type", "history_relation"],
    )
    pivot = pivot.reindex(columns=required_columns)
    if pivot.isna().any(axis=None):
        raise ValueError("Fig.3b lacks a complete 20-network paired contrast")
    contrast_rows: list[dict[str, Any]] = []
    behavior_stat_rows: dict[str, pd.Series] = {}
    for endpoint in ("rescue", "loss"):
        statistic = _one_statistic(
            behavior_stats,
            endpoint=endpoint,
            contrast="aligned_minus_mismatched",
        )
        behavior_stat_rows[endpoint] = statistic
        contrast = pivot[(endpoint, "aligned")] - pivot[(endpoint, "mismatched")]
        _validate_mean(contrast, statistic, f"Fig.3b {endpoint}")
        for seed, value in contrast.items():
            contrast_rows.append(
                {
                    "figure_id": "Fig.3",
                    "panel_id": "b",
                    "network_seed": int(seed),
                    "reader_endpoint": "Rescue" if endpoint == "rescue" else "Loss",
                    "technical_endpoint": endpoint,
                    "aligned_percent": float(pivot.loc[seed, (endpoint, "aligned")]),
                    "mismatched_percent": float(
                        pivot.loc[seed, (endpoint, "mismatched")]
                    ),
                    "matching_history_change_percent": float(value),
                }
            )
    contrasts = pd.DataFrame(contrast_rows)

    _require_seed_set(updates, "Fig.3c-d")
    if "prefix_k" not in updates or set(
        pd.to_numeric(updates["prefix_k"], errors="raise").astype(int)
    ) != {1}:
        raise ValueError("Fig.3c-d requires the frozen one-step history rows")
    updates["network_seed"] = pd.to_numeric(
        updates["network_seed"], errors="raise"
    ).astype(int)
    updates["value"] = pd.to_numeric(updates["value"], errors="raise")
    update_stat_rows: dict[str, pd.Series] = {}
    for endpoint in (
        "same_B_common_update_cosine",
        "processing_residual_gamma_norm_ratio",
    ):
        rows = updates.loc[updates["endpoint"].astype(str).eq(endpoint)].copy()
        if len(rows) != len(EXPECTED_SEEDS) or rows["network_seed"].nunique() != 20:
            raise ValueError(f"Fig.3 update endpoint {endpoint!r} is incomplete")
        statistic = _one_statistic(update_stats, endpoint=endpoint)
        update_stat_rows[endpoint] = statistic
        _validate_mean(rows["value"], statistic, f"Fig.3 {endpoint}")

    x_field = str(panel_c_spec["x_field"])
    y_field = str(panel_c_spec["y_field"])
    network_joint_means = (
        joint.groupby("network_seed", as_index=False)[[x_field, y_field]].mean()
    )
    _validate_mean(
        network_joint_means[x_field],
        update_stat_rows["same_B_common_update_cosine"],
        "Fig.3c network-balanced update similarity",
    )
    _validate_mean(
        network_joint_means[y_field],
        update_stat_rows["processing_residual_gamma_norm_ratio"],
        "Fig.3c network-balanced history effect",
    )
    x_reference = float(
        update_stat_rows["same_B_common_update_cosine"]["null_value"]
    )
    y_reference = float(
        update_stat_rows["processing_residual_gamma_norm_ratio"]["null_value"]
    )
    both_above = joint[x_field].gt(x_reference) & joint[y_field].gt(y_reference)
    if int(both_above.sum()) != len(joint):
        raise ValueError(
            "Fig.3c expected every persisted exact-B comparison to exceed both criteria"
        )
    density = _build_network_balanced_density(
        joint,
        x_field=x_field,
        y_field=y_field,
        x_reference=x_reference,
        y_reference=y_reference,
        panel_spec=panel_c_spec,
    )

    _require_seed_set(events, "Fig.3d")
    if "prefix_k" not in events or set(
        pd.to_numeric(events["prefix_k"], errors="raise").astype(int)
    ) != {1}:
        raise ValueError("Fig.3d requires the frozen one-step history rows")
    events["network_seed"] = pd.to_numeric(
        events["network_seed"], errors="raise"
    ).astype(int)
    events["value"] = pd.to_numeric(events["value"], errors="raise")
    event_stat_rows: dict[str, pd.Series] = {}
    for condition in ("matched_random", "changed_events"):
        rows = events.loc[events["condition"].astype(str).eq(condition)].copy()
        if len(rows) != len(EXPECTED_SEEDS) or rows["network_seed"].nunique() != 20:
            raise ValueError(f"Fig.3d condition {condition!r} is incomplete")
        statistic = _one_statistic(
            event_stats,
            endpoint="residual_magnitude",
            group=f"residual_magnitude|{condition}",
        )
        event_stat_rows[condition] = statistic
        _validate_mean(rows["value"], statistic, f"Fig.3d {condition}")

    return {
        "stimuli": stimuli,
        "images": images,
        "image_meta": image_meta,
        "behavior": behavior,
        "contrasts": contrasts,
        "behavior_stats": behavior_stats,
        "behavior_stat_rows": behavior_stat_rows,
        "updates": updates,
        "update_stats": update_stats,
        "update_stat_rows": update_stat_rows,
        "joint": joint,
        "network_joint_means": network_joint_means,
        "density": density,
        "decomposition_manifest": decomposition_manifest,
        "both_above_criteria": int(both_above.sum()),
        "events": events,
        "event_stats": event_stats,
        "event_stat_rows": event_stat_rows,
        "source_manifests": source_manifests,
    }


def _draw_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linestyle: Any = "-",
    linewidth: float = 0.8,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=6.2,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=3,
        )
    )


def _draw_stimulus(
    axis: plt.Axes,
    image: np.ndarray,
    bbox: Sequence[float],
    *,
    edgecolor: str,
    linewidth: float,
) -> None:
    x, y, width, height = [float(value) for value in bbox]
    axis.imshow(
        image,
        extent=(x, x + width, y, y + height),
        origin="upper",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
        zorder=4,
    )
    axis.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor="none",
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=5,
        )
    )


def _draw_state_glyph(
    axis: plt.Axes,
    bbox: Sequence[float],
    *,
    color: str,
    active_nodes: set[int],
) -> None:
    x, y, width, height = [float(value) for value in bbox]
    nodes = [
        (0.12, 0.50),
        (0.34, 0.18),
        (0.34, 0.82),
        (0.62, 0.28),
        (0.62, 0.72),
        (0.88, 0.50),
    ]
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (1, 4)]
    for first, second in edges:
        x1, y1 = nodes[first]
        x2, y2 = nodes[second]
        axis.plot(
            [x + x1 * width, x + x2 * width],
            [y + y1 * height, y + y2 * height],
            color=color,
            alpha=0.50,
            linewidth=0.9,
            zorder=3,
        )
    radius = min(width, height) * 0.082
    for index, (node_x, node_y) in enumerate(nodes):
        axis.add_patch(
            Circle(
                (x + node_x * width, y + node_y * height),
                radius=radius,
                facecolor=color if index in active_nodes else "white",
                edgecolor=color,
                linewidth=1.0,
                zorder=5,
            )
        )


def _draw_choice_glyph(
    axis: plt.Axes,
    center: tuple[float, float],
    *,
    color: str,
) -> None:
    x, y = center
    for radius in (4.3, 2.6, 0.85):
        axis.add_patch(
            Circle(
                (x, y),
                radius=radius,
                facecolor="none",
                edgecolor=color,
                linewidth=0.9,
                zorder=4,
            )
        )
    axis.add_patch(
        FancyArrowPatch(
            (x + 6.1, y + 4.6),
            (x + 0.9, y + 0.7),
            arrowstyle="-|>",
            mutation_scale=6.5,
            linewidth=0.9,
            color=color,
            zorder=5,
        )
    )


def _plot_panel_a_v1(
    axis: plt.Axes,
    payload: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
) -> None:
    labels = panel_spec["labels"]
    colors = {key: _resolve_color(value) for key, value in panel_spec["colors"].items()}
    images = payload["images"]
    lane_y = {"A": 29.3, "C": 10.7}
    lane_style = {"A": "-", "C": (0, (3.0, 2.2))}
    lane_color = {"A": colors["history_1"], "C": colors["history_2"]}
    history_bbox = {
        "A": [0.8, 24.3, 10.0, 10.0],
        "C": [0.8, 5.7, 10.0, 10.0],
    }
    delay_bbox = {
        "A": [20.0, 24.2, 16.0, 10.2],
        "C": [20.0, 5.6, 16.0, 10.2],
    }
    inherited_bbox = {
        "A": [50.0, 24.3, 13.0, 10.0],
        "C": [50.0, 5.7, 13.0, 10.0],
    }
    b_bbox = {
        "A": [76.0, 24.3, 10.0, 10.0],
        "C": [76.0, 5.7, 10.0, 10.0],
    }
    post_bbox = {
        "A": [103.0, 24.3, 13.0, 10.0],
        "C": [103.0, 5.7, 13.0, 10.0],
    }
    inherited_pattern = {"A": {0, 2, 3, 5}, "C": {0, 1, 4, 5}}

    for role in ("A", "C"):
        y = lane_y[role]
        _draw_stimulus(
            axis,
            images[role],
            history_bbox[role],
            edgecolor=lane_color[role],
            linewidth=0.8,
        )
        axis.text(
            5.8,
            36.9 if role == "A" else 18.3,
            labels["history_1"] if role == "A" else labels["history_2"],
            ha="center",
            va="bottom",
            color=lane_color[role],
        )
        x, box_y, width, height = delay_bbox[role]
        axis.add_patch(
            FancyBboxPatch(
                (x, box_y),
                width,
                height,
                boxstyle="round,pad=0.25,rounding_size=0.9",
                facecolor=NEUTRAL_PALE,
                edgecolor=NEUTRAL_LIGHT,
                linewidth=0.7,
                zorder=2,
            )
        )
        axis.text(
            x + width / 2.0,
            box_y + height * 0.64,
            labels["no_input"],
            ha="center",
            va="center",
            color=INK,
        )
        axis.text(
            x + width / 2.0,
            box_y + height * 0.30,
            labels["delay"],
            ha="center",
            va="center",
            color=NEUTRAL_DARK,
        )
        _draw_state_glyph(
            axis,
            inherited_bbox[role],
            color=colors["inherited_state"],
            active_nodes=inherited_pattern[role],
        )
        _draw_stimulus(
            axis,
            images["B"],
            b_bbox[role],
            edgecolor=colors["current_b"],
            linewidth=0.9,
        )
        _draw_state_glyph(
            axis,
            post_bbox[role],
            color=colors["post_b_state"],
            active_nodes={1, 2, 3, 4},
        )
        _draw_choice_glyph(axis, (142.0, y), color=colors["choice"])
        _draw_arrow(
            axis,
            (11.7, y),
            (19.1, y),
            color=lane_color[role],
            linestyle=lane_style[role],
        )
        _draw_arrow(
            axis,
            (36.9, y),
            (49.1, y),
            color=lane_color[role],
            linestyle=lane_style[role],
        )
        _draw_arrow(
            axis,
            (63.9, y),
            (75.1, y),
            color=colors["current_b"],
            linestyle=lane_style[role],
        )
        _draw_arrow(
            axis,
            (89.6, y),
            (102.1, y),
            color=colors["current_b"],
            linestyle=lane_style[role],
        )
        _draw_arrow(
            axis,
            (116.9, y),
            (135.1, y),
            color=NEUTRAL_DARK,
            linestyle=lane_style[role],
        )

    axis.text(
        56.5,
        38.2,
        labels["inherited_state"],
        ha="center",
        va="bottom",
        color=colors["inherited_state"],
    )
    axis.text(
        81.0,
        38.2,
        labels["same_b"],
        ha="center",
        va="bottom",
        color=colors["current_b"],
    )
    axis.text(
        109.5,
        38.2,
        labels["state_after_b"],
        ha="center",
        va="bottom",
        color=INK,
    )
    axis.text(
        142.0,
        38.2,
        labels["choice_after_b"],
        ha="center",
        va="bottom",
        color=INK,
    )
    axis.plot([88.7, 88.7], [10.7, 29.3], color=NEUTRAL_MID, linewidth=0.7)
    axis.plot([86.9, 88.7], [10.7, 10.7], color=NEUTRAL_MID, linewidth=0.7)
    axis.plot([86.9, 88.7], [29.3, 29.3], color=NEUTRAL_MID, linewidth=0.7)
    axis.plot([92.0, 95.0], [19.3, 19.3], color=INK, linewidth=0.8)
    axis.plot([92.0, 95.0], [20.7, 20.7], color=INK, linewidth=0.8)
    axis.set_xlim(0.0, 155.0)
    axis.set_ylim(0.0, 40.0)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def _plot_panel_a(
    axis: plt.Axes,
    payload: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
) -> None:
    labels = panel_spec["labels"]
    colors = {
        key: _resolve_color(value) for key, value in panel_spec["colors"].items()
    }
    layout = panel_spec["schematic_layout"]
    x_shift = float(layout.get("content_shift_x", 0.0))
    images = payload["images"]
    lane_y = {key: float(value) for key, value in layout["lane_y"].items()}
    lane_style = {"A": "-", "C": (0, (3.0, 2.2))}
    lane_color = {"A": colors["history_1"], "C": colors["history_2"]}
    history_bbox = layout["history_bbox"]
    delay_bbox = layout["delay_bbox"]
    inherited_bbox = layout["inherited_bbox"]
    b_bbox = layout["b_bbox"]
    post_bbox = layout["post_bbox"]
    choice_center = layout["choice_center"]
    history_label_y = layout["history_label_y"]
    inherited_pattern = {"A": {0, 2, 3, 5}, "C": {0, 1, 4, 5}}

    for role in ("A", "C"):
        y = lane_y[role]
        history_box = [float(value) for value in history_bbox[role]]
        delay_box = [float(value) for value in delay_bbox[role]]
        inherited_box = [float(value) for value in inherited_bbox[role]]
        current_box = [float(value) for value in b_bbox[role]]
        post_box = [float(value) for value in post_bbox[role]]
        for box in (history_box, delay_box, inherited_box, current_box, post_box):
            box[0] += x_shift
        choice_xy = (
            float(choice_center[role][0]) + x_shift,
            float(choice_center[role][1]),
        )
        _draw_stimulus(
            axis,
            images[role],
            history_box,
            edgecolor=lane_color[role],
            linewidth=1.0,
        )
        axis.text(
            history_box[0] + history_box[2] / 2.0,
            float(history_label_y[role]),
            labels["history_1"] if role == "A" else labels["history_2"],
            ha="center",
            va="top" if role == "A" else "bottom",
            color=lane_color[role],
        )
        x, box_y, width, height = delay_box
        axis.add_patch(
            FancyBboxPatch(
                (x, box_y),
                width,
                height,
                boxstyle="round,pad=0.25,rounding_size=1.0",
                facecolor=NEUTRAL_PALE,
                edgecolor=NEUTRAL_LIGHT,
                linewidth=0.8,
                zorder=2,
            )
        )
        axis.text(
            x + width / 2.0,
            box_y + height * 0.64,
            labels["no_input"],
            ha="center",
            va="center",
            color=INK,
        )
        axis.text(
            x + width / 2.0,
            box_y + height * 0.30,
            labels["delay"],
            ha="center",
            va="center",
            color=NEUTRAL_DARK,
        )
        _draw_state_glyph(
            axis,
            inherited_box,
            color=colors["inherited_state"],
            active_nodes=inherited_pattern[role],
        )
        _draw_stimulus(
            axis,
            images["B"],
            current_box,
            edgecolor=colors["current_b"],
            linewidth=1.0,
        )
        _draw_state_glyph(
            axis,
            post_box,
            color=colors["post_b_state"],
            active_nodes={1, 2, 3, 4},
        )
        _draw_choice_glyph(axis, choice_xy, color=colors["choice"])
        _draw_arrow(
            axis,
            (history_box[0] + history_box[2] + 0.8, y),
            (delay_box[0] - 0.8, y),
            color=lane_color[role],
            linestyle=lane_style[role],
            linewidth=0.78,
        )
        _draw_arrow(
            axis,
            (delay_box[0] + delay_box[2] + 0.8, y),
            (inherited_box[0] - 0.8, y),
            color=lane_color[role],
            linestyle=lane_style[role],
            linewidth=0.78,
        )
        _draw_arrow(
            axis,
            (inherited_box[0] + inherited_box[2] + 0.8, y),
            (current_box[0] - 0.8, y),
            color=colors["current_b"],
            linestyle=lane_style[role],
            linewidth=0.78,
        )
        bracket_x = float(layout["equality"]["bracket_x"]) + x_shift
        _draw_arrow(
            axis,
            (bracket_x + 1.0, y),
            (post_box[0] - 0.8, y),
            color=colors["current_b"],
            linestyle=lane_style[role],
            linewidth=0.78,
        )
        _draw_arrow(
            axis,
            (post_box[0] + post_box[2] + 0.8, y),
            (choice_xy[0] - 5.3, y),
            color=NEUTRAL_DARK,
            linestyle=lane_style[role],
            linewidth=0.78,
        )

    header_y = float(layout["header_y"])
    axis.text(
        float(inherited_bbox["A"][0]) + float(inherited_bbox["A"][2]) / 2.0 + x_shift,
        header_y,
        labels["inherited_state"],
        ha="center",
        va="top",
        color=colors["inherited_state"],
    )
    axis.text(
        float(b_bbox["A"][0]) + float(b_bbox["A"][2]) / 2.0 + x_shift,
        header_y,
        labels["same_b"],
        ha="center",
        va="top",
        color=colors["current_b"],
    )
    axis.text(
        float(post_bbox["A"][0]) + float(post_bbox["A"][2]) / 2.0 + x_shift,
        header_y,
        labels["state_after_b"],
        ha="center",
        va="top",
        color=INK,
    )
    axis.text(
        float(choice_center["A"][0]) + x_shift,
        header_y,
        labels["choice_after_b"],
        ha="center",
        va="top",
        color=INK,
    )
    equality = layout["equality"]
    bracket_x = float(equality["bracket_x"]) + x_shift
    top_y, bottom_y = lane_y["A"], lane_y["C"]
    current_right = (
        float(b_bbox["A"][0]) + float(b_bbox["A"][2]) + x_shift
    )
    axis.plot([bracket_x, bracket_x], [bottom_y, top_y], color=NEUTRAL_MID, linewidth=0.8)
    axis.plot([current_right + 0.5, bracket_x], [bottom_y, bottom_y], color=NEUTRAL_MID, linewidth=0.8)
    axis.plot([current_right + 0.5, bracket_x], [top_y, top_y], color=NEUTRAL_MID, linewidth=0.8)
    equals_x = float(equality["equals_center_x"]) + x_shift
    equals_y = float(equality["equals_center_y"])
    axis.plot([equals_x - 1.8, equals_x + 1.8], [equals_y - 0.8, equals_y - 0.8], color=INK, linewidth=0.9)
    axis.plot([equals_x - 1.8, equals_x + 1.8], [equals_y + 0.8, equals_y + 0.8], color=INK, linewidth=0.9)
    bounds = [float(value) for value in layout["content_bounds"]]
    axis.set_xlim(bounds[0], bounds[0] + bounds[2])
    axis.set_ylim(bounds[1], bounds[1] + bounds[3])
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def _plot_panel_b(
    axis: plt.Axes,
    payload: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
) -> None:
    contrasts = payload["contrasts"]
    statistics = payload["behavior_stat_rows"]
    endpoint_order = list(panel_spec["endpoint_order"])
    labels = panel_spec["endpoint_labels"]
    colors = {key: _resolve_color(value) for key, value in panel_spec["colors"].items()}
    y_positions = {endpoint_order[0]: 1.0, endpoint_order[1]: 0.0}
    axis.axvline(0.0, color=NEUTRAL_MID, linewidth=0.8, zorder=0)
    for endpoint in endpoint_order:
        y = y_positions[endpoint]
        rows = contrasts.loc[contrasts["technical_endpoint"].eq(endpoint)]
        if len(rows) != len(EXPECTED_SEEDS):
            raise ValueError(f"Fig.3b endpoint {endpoint} lacks 20 network contrasts")
        statistic = statistics[endpoint]
        estimate = float(statistic["estimate"])
        low = float(statistic["ci95_low"])
        high = float(statistic["ci95_high"])
        color = colors[endpoint]
        axis.barh(
            y,
            estimate,
            left=0.0,
            height=0.28,
            color=color,
            edgecolor=INK,
            linewidth=0.45,
            zorder=2,
        )
        axis.errorbar(
            estimate,
            y,
            xerr=np.asarray([[estimate - low], [high - estimate]]),
            fmt="D",
            color=color,
            markerfacecolor=color,
            markeredgecolor=INK,
            markeredgewidth=0.45,
            markersize=4.1,
            ecolor=INK,
            elinewidth=1.25,
            capsize=2.8,
            capthick=1.0,
            zorder=4,
        )
        text = f"{estimate:+.1f}".replace("-", "−")
        axis.text(
            estimate + 0.6,
            y + 0.12,
            text,
            ha="left",
            va="bottom",
            color=color,
        )
    axis.set_yticks([y_positions[item] for item in endpoint_order])
    axis.set_yticklabels([labels[item] for item in endpoint_order])
    axis.set_ylim(-0.42, 1.45)
    axis.set_xlim(*[float(value) for value in panel_spec["x_limits"]])
    axis.set_xticks([float(value) for value in panel_spec["x_ticks"]])
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    _style_axis(axis)


def _plot_joint_density(
    axis: plt.Axes,
    payload: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
) -> None:
    density = payload["density"]
    aggregate = np.asarray(density["aggregate"], dtype=float)
    x_centers = np.asarray(density["x_centers"], dtype=float)
    y_centers = np.asarray(density["y_centers"], dtype=float)
    contours = density["contours"].set_index("target_probability_mass")
    contour_levels = [
        float(contours.loc[mass, "density_threshold"])
        for mass in (0.95, 0.8, 0.5)
    ]
    if not all(
        first < second for first, second in zip(contour_levels, contour_levels[1:])
    ):
        raise ValueError(
            f"Fig.3c contour levels are not strictly ordered: {contour_levels}"
        )
    outline_mass = float(panel_spec.get("outline_mass", 0.95))
    if outline_mass not in contours.index:
        raise ValueError(f"Fig.3c outline mass is unavailable: {outline_mass}")
    outline_threshold = float(contours.loc[outline_mass, "density_threshold"])
    density_color = _resolve_color(str(panel_spec["density_color"]))
    mean_color = _resolve_color(str(panel_spec["mean_color"]))
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    maximum = float(aggregate.max())
    floor_fraction = float(panel_spec.get("display_floor_fraction_of_max", 0.01))
    display_floor = maximum * floor_fraction
    if not 0.0 < display_floor < maximum:
        raise ValueError(f"Fig.3c density display floor is invalid: {display_floor}")
    density_field = np.ma.masked_less_equal(aggregate.T, display_floor)
    base_rgb = matplotlib.colors.to_rgb(density_color)
    continuous_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "fig3_joint_density",
        [
            (*base_rgb, 0.04),
            (*base_rgb, 0.22),
            (*base_rgb, 0.50),
            (*base_rgb, 0.84),
        ],
    )
    density_norm = matplotlib.colors.PowerNorm(
        gamma=float(panel_spec.get("power_gamma", 0.62)),
        vmin=display_floor,
        vmax=maximum,
        clip=True,
    )
    axis.pcolormesh(
        x_grid,
        y_grid,
        density_field,
        shading="nearest",
        cmap=continuous_cmap,
        norm=density_norm,
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=False,
        zorder=1,
    )
    axis.contour(
        x_grid,
        y_grid,
        aggregate.T,
        levels=[outline_threshold],
        colors=[density_color],
        linewidths=[0.75],
        alpha=0.82,
        zorder=2,
    )
    x_stat = payload["update_stat_rows"]["same_B_common_update_cosine"]
    y_stat = payload["update_stat_rows"]["processing_residual_gamma_norm_ratio"]
    x_mean = float(x_stat["estimate"])
    y_mean = float(y_stat["estimate"])
    x_low, x_high = float(x_stat["ci95_low"]), float(x_stat["ci95_high"])
    y_low, y_high = float(y_stat["ci95_low"]), float(y_stat["ci95_high"])
    x_reference = float(x_stat["null_value"])
    y_reference = float(y_stat["null_value"])
    axis.axvline(
        x_reference,
        color=NEUTRAL_MID,
        linewidth=0.75,
        linestyle=(0, (3.0, 2.3)),
        zorder=0,
    )
    axis.axhline(
        y_reference,
        color=NEUTRAL_MID,
        linewidth=0.75,
        linestyle=(0, (3.0, 2.3)),
        zorder=0,
    )
    axis.errorbar(
        x_mean,
        y_mean,
        xerr=np.asarray([[x_mean - x_low], [x_high - x_mean]]),
        yerr=np.asarray([[y_mean - y_low], [y_high - y_mean]]),
        fmt="D",
        color=mean_color,
        markerfacecolor=mean_color,
        markeredgecolor=INK,
        markeredgewidth=0.5,
        markersize=4.6,
        ecolor=INK,
        elinewidth=1.0,
        capsize=2.2,
        capthick=0.9,
        zorder=5,
    )
    axis.set_xlim(*[float(value) for value in density["x_limits"]])
    axis.set_ylim(*[float(value) for value in density["y_limits"]])
    axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=2.0)
    _style_axis(axis)


def _plot_bullet(
    axis: plt.Axes,
    payload: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
) -> None:
    endpoint = str(panel_spec["endpoint"])
    statistic = payload["update_stat_rows"][endpoint]
    estimate = float(statistic["estimate"])
    low = float(statistic["ci95_low"])
    high = float(statistic["ci95_high"])
    reference = float(statistic["null_value"])
    color = _resolve_color(str(panel_spec["color"]))
    x_limits = [float(value) for value in panel_spec["x_limits"]]
    axis.plot(
        x_limits,
        [0.0, 0.0],
        color=NEUTRAL_LIGHT,
        linewidth=4.8,
        solid_capstyle="butt",
        zorder=0,
    )
    axis.axvline(
        reference,
        color=NEUTRAL_MID,
        linewidth=0.8,
        linestyle=(0, (3.0, 2.3)),
        zorder=1,
    )
    axis.barh(
        0.0,
        estimate,
        left=0.0,
        height=0.20,
        color=color,
        edgecolor=INK,
        linewidth=0.45,
        zorder=2,
    )
    axis.errorbar(
        estimate,
        0.0,
        xerr=np.asarray([[estimate - low], [high - estimate]]),
        fmt="D",
        color=color,
        markerfacecolor=color,
        markeredgecolor=INK,
        markeredgewidth=0.45,
        markersize=4.2,
        elinewidth=1.1,
        capsize=3.0,
        capthick=0.9,
        zorder=4,
    )
    axis.set_xlim(*x_limits)
    axis.set_xticks([float(value) for value in panel_spec["x_ticks"]])
    axis.set_ylim(-0.48, 0.48)
    axis.set_yticks([])
    axis.set_xlabel(str(panel_spec["reader_label"]), labelpad=3.0)
    _style_axis(axis)
    axis.tick_params(axis="y", left=False)


def _plot_panel_d(
    axis: plt.Axes,
    payload: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
) -> None:
    order = list(panel_spec["condition_order"])
    labels = panel_spec["condition_labels"]
    colors = {key: _resolve_color(value) for key, value in panel_spec["colors"].items()}
    for index, condition in enumerate(order):
        statistic = payload["event_stat_rows"][condition]
        estimate = float(statistic["estimate"])
        low = float(statistic["ci95_low"])
        high = float(statistic["ci95_high"])
        axis.bar(
            index,
            estimate,
            width=0.48,
            color=colors[condition],
            edgecolor=INK,
            linewidth=0.45,
            zorder=2,
        )
        axis.errorbar(
            index,
            estimate,
            yerr=np.asarray([[estimate - low], [high - estimate]]),
            fmt="none",
            ecolor=INK,
            elinewidth=1.0,
            capsize=2.5,
            capthick=0.9,
            zorder=4,
        )
    axis.set_xticks(np.arange(len(order), dtype=float))
    axis.set_xticklabels([labels[item] for item in order])
    axis.set_xlim(-0.62, len(order) - 0.38)
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    _style_axis(axis)


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
        "a": [2.0, 2.0, 161.0, 48.0],
        "b": [2.0, 52.0, 52.333, 48.0],
        "c": [56.333, 52.0, 52.334, 48.0],
        "d": [110.667, 52.0, 52.333, 48.0],
    }
    if spec["slots"] != expected_slots:
        failures.append("slot geometry differs from the approved 1+3 preset")
    if [canvas_width, canvas_height] != [165.0, 102.0]:
        failures.append("canvas differs from 165 x 102 mm")
    return {
        "schema": "manuscript_fig3_candidate_layout_audit_v1",
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


def _resolved_spec(spec: Mapping[str, Any], reader: BundleReader) -> dict[str, Any]:
    resolved = json.loads(json.dumps(spec))
    resolved["resolved_at"] = _utc_now()
    resolved["resolved_colors"] = {
        "ink": _resolve_color("ink"),
        "neutral_dark": _resolve_color("neutral_dark"),
        "neutral_mid": _resolve_color("neutral_mid"),
        "neutral_light": _resolve_color("neutral_light"),
        "dynamic": _resolve_color("dynamic"),
        "mechanism_teal": _resolve_color("mechanism_teal"),
        "fused_state": _resolve_color("fused_state"),
        "comparison_coral": _resolve_color("comparison_coral"),
    }
    resolved["resolved_parent_sources"] = reader.access_frame().to_dict("records")
    return resolved


def _candidate_tables_v1(payload: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    stimuli = payload["stimuli"].copy()
    stimuli = stimuli.rename(
        columns={
            "figure_id": "source_bundle_figure_id",
            "panel_id": "source_bundle_panel_id",
        }
    )
    stimuli.insert(0, "candidate_figure", "Fig.3")
    stimuli.insert(1, "candidate_panel", "a")

    updates = payload["updates"]
    common = updates.loc[
        updates["endpoint"].astype(str).eq("same_B_common_update_cosine")
    ].copy()
    history = updates.loc[
        updates["endpoint"].astype(str).eq(
            "processing_residual_gamma_norm_ratio"
        )
    ].copy()
    common_out = common[
        [
            "network_seed",
            "value",
            "unit",
            "prefix_k",
            "endpoint",
            "source_endpoint",
        ]
    ].copy()
    common_out.insert(0, "candidate_figure", "Fig.3")
    common_out.insert(1, "candidate_panel", "c")
    common_out.insert(3, "reader_endpoint", "Update similarity")
    common_out = common_out.rename(
        columns={
            "endpoint": "bundle_endpoint_alias",
            "source_endpoint": "technical_source_endpoint",
        }
    )
    history_out = history[
        [
            "network_seed",
            "value",
            "unit",
            "prefix_k",
            "endpoint",
            "source_endpoint",
        ]
    ].copy()
    history_out.insert(0, "candidate_figure", "Fig.3")
    history_out.insert(1, "candidate_panel", "d")
    history_out.insert(3, "reader_endpoint", "History effect")
    history_out = history_out.rename(
        columns={
            "endpoint": "bundle_endpoint_alias",
            "source_endpoint": "technical_source_endpoint",
        }
    )

    events = payload["events"].copy()
    events = events.rename(
        columns={
            "figure_id": "source_bundle_figure_id",
            "panel_id": "source_bundle_panel_id",
        }
    )
    events.insert(0, "candidate_figure", "Fig.3")
    events.insert(1, "candidate_panel", "e")
    events["reader_condition"] = events["condition"].map(
        {"matched_random": "Matched control", "changed_events": "Changed spikes"}
    )
    events["reader_endpoint"] = "History effect"

    panel_a_stats = pd.DataFrame(
        [
            {
                "candidate_figure": "Fig.3",
                "candidate_panel": "a",
                "panel_type": "schematic",
                "statistics_status": "not_applicable",
            }
        ]
    )
    metric_tables: dict[str, pd.DataFrame] = {"panel_a_statistics": panel_a_stats}
    metric_sources = {
        "panel_b_statistics": pd.DataFrame(payload["behavior_stat_rows"]).T,
        "panel_c_statistics": pd.DataFrame(
            [payload["update_stat_rows"]["same_B_common_update_cosine"]]
        ),
        "panel_d_statistics": pd.DataFrame(
            [payload["update_stat_rows"]["processing_residual_gamma_norm_ratio"]]
        ),
        "panel_e_statistics": pd.DataFrame(payload["event_stat_rows"]).T,
    }
    for name, frame in metric_sources.items():
        panel_id = name.split("_")[1]
        frame = frame.copy()
        frame = frame.rename(
            columns={
                "figure_id": "source_bundle_figure_id",
                "panel_id": "source_bundle_panel_id",
            }
        )
        frame.insert(0, "candidate_figure", "Fig.3")
        frame.insert(1, "candidate_panel", panel_id)
        metric_tables[name] = frame

    return {
        "panel_a_stimuli": stimuli,
        "panel_b_network_contrasts": payload["contrasts"].copy(),
        "panel_c_update_similarity": common_out,
        "panel_d_history_effect": history_out,
        "panel_e_changed_spike_events": events,
        **metric_tables,
    }


def _candidate_tables(payload: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    stimuli = payload["stimuli"].copy().rename(
        columns={
            "figure_id": "source_bundle_figure_id",
            "panel_id": "source_bundle_panel_id",
        }
    )
    stimuli.insert(0, "candidate_figure", "Fig.3")
    stimuli.insert(1, "candidate_panel", "a")

    joint = payload["joint"].copy()
    joint = joint[
        [
            "network_seed",
            "prefix_k",
            "valid",
            "history_family_id",
            "b_anchor_id",
            "same_B_common_update_cosine",
            "processing_residual_gamma_energy_fraction",
        ]
    ]
    joint.insert(0, "candidate_figure", "Fig.3")
    joint.insert(1, "candidate_panel", "c")
    joint = joint.rename(
        columns={
            "same_B_common_update_cosine": "update_similarity",
            "processing_residual_gamma_energy_fraction": "history_effect",
        }
    )

    network_means = payload["network_joint_means"].copy().rename(
        columns={
            "same_B_common_update_cosine": "mean_update_similarity",
            "processing_residual_gamma_energy_fraction": "mean_history_effect",
        }
    )
    network_means.insert(0, "candidate_figure", "Fig.3")
    network_means.insert(1, "candidate_panel", "c")

    events = payload["events"].copy().rename(
        columns={
            "figure_id": "source_bundle_figure_id",
            "panel_id": "source_bundle_panel_id",
        }
    )
    events.insert(0, "candidate_figure", "Fig.3")
    events.insert(1, "candidate_panel", "d")
    events["reader_condition"] = events["condition"].map(
        {"matched_random": "Matched control", "changed_events": "Changed spikes"}
    )
    events["reader_endpoint"] = "History effect"

    panel_a_stats = pd.DataFrame(
        [
            {
                "candidate_figure": "Fig.3",
                "candidate_panel": "a",
                "panel_type": "schematic",
                "statistics_status": "not_applicable",
            }
        ]
    )
    metric_sources = {
        "panel_b_statistics": pd.DataFrame(payload["behavior_stat_rows"]).T,
        "panel_c_statistics": pd.DataFrame(
            [
                payload["update_stat_rows"]["same_B_common_update_cosine"],
                payload["update_stat_rows"][
                    "processing_residual_gamma_norm_ratio"
                ],
            ]
        ),
        "panel_d_statistics": pd.DataFrame(payload["event_stat_rows"]).T,
    }
    metric_tables: dict[str, pd.DataFrame] = {
        "panel_a_statistics": panel_a_stats
    }
    for name, frame in metric_sources.items():
        panel_id = name.split("_")[1]
        frame = frame.copy().rename(
            columns={
                "figure_id": "source_bundle_figure_id",
                "panel_id": "source_bundle_panel_id",
            }
        )
        frame.insert(0, "candidate_figure", "Fig.3")
        frame.insert(1, "candidate_panel", panel_id)
        metric_tables[name] = frame

    density = payload["density"]
    density_summary = pd.DataFrame(
        [
            {
                "candidate_figure": "Fig.3",
                "candidate_panel": "c",
                "independent_unit": "network_seed",
                "n_networks": len(EXPECTED_SEEDS),
                "comparisons_per_network": 500,
                "comparison_rows": len(payload["joint"]),
                "rows_above_both_criteria": payload["both_above_criteria"],
                "network_weight": 1.0 / len(EXPECTED_SEEDS),
                "grid_x_bins": len(density["x_centers"]),
                "grid_y_bins": len(density["y_centers"]),
                "smoothing_sigma_cells": 1.0,
                "aggregate_probability_mass": float(density["aggregate"].sum()),
                "inference_unit_note": "comparison rows are descriptive; networks are inferential units",
            }
        ]
    )
    contour_levels = density["contours"].copy()
    contour_levels.insert(0, "candidate_figure", "Fig.3")
    contour_levels.insert(1, "candidate_panel", "c")

    return {
        "panel_a_stimuli": stimuli,
        "panel_b_network_contrasts": payload["contrasts"].copy(),
        "panel_c_joint_comparisons": joint,
        "panel_c_network_means": network_means,
        "panel_c_network_density_grid": density["network_grid"].copy(),
        "panel_c_network_balanced_density": density["aggregate_grid"].copy(),
        "panel_d_changed_spike_events": events,
        "panel_c_density_summary": density_summary,
        "panel_c_contour_levels": contour_levels,
        **metric_tables,
    }


def _source_mapping_v1(
    parent_dir: Path,
    reader: BundleReader,
) -> pd.DataFrame:
    paths = {
        "a_data": "data/panel_a_input_stimuli.csv",
        "b_data": "data/panel_b_plot_data.csv",
        "b_stats": "metrics/panel_b_statistics.csv",
        "c_data": "data/panel_c_plot_data.csv",
        "c_stats": "metrics/panel_c_statistics.csv",
        "e_data": "data/panel_d_plot_data.csv",
        "e_stats": "metrics/panel_d_statistics.csv",
        "a_manifest": "meta/panel_a_source_manifest.csv",
        "b_manifest": "meta/panel_b_source_manifest.csv",
        "c_manifest": "meta/panel_c_source_manifest.csv",
        "e_manifest": "meta/panel_d_source_manifest.csv",
    }

    def source(name: str) -> tuple[str, str]:
        relative = paths[name]
        return relative, _sha256(parent_dir / relative)

    rows: list[dict[str, Any]] = []
    entries = [
        ("a", "Exact-B paired counterfactual", "Persisted MNIST pixels", "a_data", None, "a_manifest", "The same B matrix is rendered in both independent lanes."),
        ("b", "Wrong to correct", "rescue aligned-minus-mismatched", "b_data", "b_stats", "b_manifest", "Paired network contrast; all 20 values are shown as a deterministic rug."),
        ("b", "Correct to wrong", "loss aligned-minus-mismatched", "b_data", "b_stats", "b_manifest", "Paired network contrast; all 20 values are shown as a deterministic rug."),
        ("c", "Update similarity", "same_B_common_update_cosine", "c_data", "c_stats", "c_manifest", "Native-scale frozen estimate and 95% CI; network values remain in Source Data."),
        ("d", "History effect", "processing_residual_gamma_norm_ratio (source endpoint: processing_residual_gamma_energy_fraction)", "c_data", "c_stats", "c_manifest", "Native-scale frozen estimate and 95% CI; network values remain in Source Data."),
        ("e", "Matched control", "matched_random_gamma_mean_abs", "e_data", "e_stats", "e_manifest", "Frozen condition mean and 95% CI; raw network values are not drawn."),
        ("e", "Changed spikes", "changed_coordinate_gamma_mean_abs", "e_data", "e_stats", "e_manifest", "Frozen condition mean and 95% CI; raw network values are not drawn."),
    ]
    for panel_id, reader_label, endpoint, data_key, stats_key, manifest_key, transform in entries:
        data_path, data_hash = source(data_key)
        stats_path, stats_hash = ("", "") if stats_key is None else source(stats_key)
        manifest_path, manifest_hash = source(manifest_key)
        rows.append(
            {
                "candidate_figure": "Fig.3",
                "candidate_panel": panel_id,
                "reader_label": reader_label,
                "technical_endpoint_or_object": endpoint,
                "parent_data_path": data_path,
                "parent_data_sha256": data_hash,
                "parent_statistics_path": stats_path,
                "parent_statistics_sha256": stats_hash,
                "parent_upstream_manifest": manifest_path,
                "parent_upstream_manifest_sha256": manifest_hash,
                "independent_unit": "network_seed",
                "included_networks": "1000-1019",
                "mapping": transform,
            }
        )
    access_paths = set(reader.access_frame()["relative_path"].astype(str))
    missing_accesses = {
        value for value in paths.values() if value not in access_paths
    }
    if missing_accesses:
        raise ValueError(f"source mapping contains unread parent files: {missing_accesses}")
    return pd.DataFrame(rows)


def _source_mapping(
    parent_dir: Path,
    reader: BundleReader,
) -> pd.DataFrame:
    bundle_paths = {
        "a_data": "data/panel_a_input_stimuli.csv",
        "b_data": "data/panel_b_plot_data.csv",
        "b_stats": "metrics/panel_b_statistics.csv",
        "c_stats": "metrics/panel_c_statistics.csv",
        "d_data": "data/panel_d_plot_data.csv",
        "d_stats": "metrics/panel_d_statistics.csv",
        "a_manifest": "meta/panel_a_source_manifest.csv",
        "b_manifest": "meta/panel_b_source_manifest.csv",
        "c_manifest": "meta/panel_c_source_manifest.csv",
        "d_manifest": "meta/panel_d_source_manifest.csv",
    }

    def source(name: str) -> tuple[str, str]:
        relative = bundle_paths[name]
        return relative, _sha256(parent_dir / relative)

    rows: list[dict[str, Any]] = []
    bundle_entries = [
        ("a", "Exact-B paired counterfactual", "Persisted MNIST pixels", "a_data", None, "a_manifest", "The same B matrix is rendered in both independent lanes."),
        ("b", "Rescue", "rescue aligned-minus-mismatched", "b_data", "b_stats", "b_manifest", "Paired network contrast; artwork shows the frozen mean and paired 95% CI."),
        ("b", "Loss", "loss aligned-minus-mismatched", "b_data", "b_stats", "b_manifest", "Paired network contrast; artwork shows the frozen mean and paired 95% CI."),
        ("d", "Matched control", "matched_random_gamma_mean_abs", "d_data", "d_stats", "d_manifest", "Frozen condition mean and 95% CI; raw network values are not drawn."),
        ("d", "Changed spikes", "changed_coordinate_gamma_mean_abs", "d_data", "d_stats", "d_manifest", "Frozen condition mean and 95% CI; raw network values are not drawn."),
    ]
    for panel_id, reader_label, endpoint, data_key, stats_key, manifest_key, mapping in bundle_entries:
        data_path, data_hash = source(data_key)
        stats_path, stats_hash = ("", "") if stats_key is None else source(stats_key)
        manifest_path, manifest_hash = source(manifest_key)
        rows.append(
            {
                "candidate_figure": "Fig.3",
                "candidate_panel": panel_id,
                "reader_label": reader_label,
                "technical_endpoint_or_object": endpoint,
                "parent_source_scope": "frozen_bundle",
                "parent_data_path": data_path,
                "parent_data_sha256": data_hash,
                "parent_statistics_path": stats_path,
                "parent_statistics_sha256": stats_hash,
                "parent_upstream_manifest": manifest_path,
                "parent_upstream_manifest_sha256": manifest_hash,
                "independent_unit": "network_seed",
                "included_networks": "1000-1019",
                "mapping": mapping,
            }
        )
    decomposition_access = reader.access_frame().loc[
        reader.access_frame()["source_scope"].eq("decomposition")
    ].sort_values("relative_path")
    if len(decomposition_access) != len(EXPECTED_SEEDS):
        raise ValueError("Fig.3c source mapping requires 20 registered decomposition files")
    combined_digest = hashlib.sha256(
        "\n".join(
            f"{row.relative_path}:{row.sha256}"
            for row in decomposition_access.itertuples()
        ).encode("utf-8")
    ).hexdigest()
    rows.append(
        {
            "candidate_figure": "Fig.3",
            "candidate_panel": "c",
            "reader_label": "Update similarity × History effect",
            "technical_endpoint_or_object": "same_B_common_update_cosine × processing_residual_gamma_energy_fraction",
            "parent_source_scope": "20 persisted network files",
            "parent_data_path": "see meta/panel_c_source_manifest.csv",
            "parent_data_sha256": combined_digest,
            "parent_statistics_path": "metrics/panel_c_statistics.csv",
            "parent_statistics_sha256": source("c_stats")[1],
            "parent_upstream_manifest": "meta/panel_c_source_manifest.csv",
            "parent_upstream_manifest_sha256": "listed per source file",
            "independent_unit": "network_seed",
            "included_networks": "1000-1019",
            "mapping": "500 valid one-step comparisons per network → normalized network density → equal-weight mean across 20 networks",
        }
    )
    required_bundle_accesses = set(bundle_paths.values())
    observed_bundle_accesses = set(
        reader.access_frame().loc[
            reader.access_frame()["source_scope"].eq("bundle"), "relative_path"
        ].astype(str)
    )
    missing = required_bundle_accesses - observed_bundle_accesses
    if missing:
        raise ValueError(f"source mapping contains unread bundle files: {missing}")
    return pd.DataFrame(rows)


def _format_p(value: Any) -> str:
    number = float(value)
    return f"{number:.2g}"


def _caption_v1(payload: Mapping[str, Any]) -> str:
    b = payload["behavior_stat_rows"]
    c = payload["update_stat_rows"]["same_B_common_update_cosine"]
    d = payload["update_stat_rows"]["processing_residual_gamma_norm_ratio"]
    e = payload["event_stat_rows"]
    contrasts = payload["contrasts"]
    rescue_values = contrasts.loc[
        contrasts["technical_endpoint"].eq("rescue"),
        "matching_history_change_percent",
    ]
    loss_values = contrasts.loc[
        contrasts["technical_endpoint"].eq("loss"),
        "matching_history_change_percent",
    ]
    rescue_consistent = int((rescue_values > 0).sum())
    loss_consistent = int((loss_values < 0).sum())
    rescue_ties = int((rescue_values == 0).sum())
    b_rescue = b["rescue"]
    b_loss = b["loss"]
    matched = e["matched_random"]
    changed = e["changed_events"]
    return f"""**Fig.3 | Silent inherited history conditions processing of the same current input.**

**a,** Paired exact-B counterfactual. Two distinct one-item histories are followed by a 200-ms no-input interval and form differently patterned inherited STSP states. The same persisted B bitmap is then presented in two separate runs; state and choice are compared only after B. **b,** Matching-history effects on the two opportunity-defined behavioral changes, computed within each independently trained network as aligned minus mismatched history. Wrong-to-correct change was {float(b_rescue['estimate']):+.1f} percentage points (95% CI {float(b_rescue['ci95_low']):+.1f} to {float(b_rescue['ci95_high']):+.1f}); {rescue_consistent}/20 networks had the same positive direction and {rescue_ties} network had no difference. Correct-to-wrong change was {float(b_loss['estimate']):+.1f} percentage points (95% CI {float(b_loss['ci95_low']):+.1f} to {float(b_loss['ci95_high']):+.1f}); {loss_consistent}/20 networks had the same negative direction. Rugs show all 20 paired network contrasts without jitter. Frozen paired inference used two-sided one-sample t tests against zero with Benjamini-Hochberg adjustment (adjusted P = {_format_p(b_rescue['p_adjusted'])} and {_format_p(b_loss['p_adjusted'])}, respectively). **c,** Update similarity for the identical B (technical endpoint: `same_B_common_update_cosine`) on its native 0-1 scale; mean {float(c['estimate']):.4f}, 95% CI {float(c['ci95_low']):.4f}-{float(c['ci95_high']):.4f}, with the prespecified 0.5 reference. **d,** History effect (technical source endpoint: `processing_residual_gamma_energy_fraction`; bundle endpoint alias: `processing_residual_gamma_norm_ratio`) on an independent native scale; mean {float(d['estimate']):.4f}, 95% CI {float(d['ci95_low']):.4f}-{float(d['ci95_high']):.4f}, with the prespecified 0.05 reference. The values in c and d are different metrics and should not be read as directly comparable contributions. **e,** Mean absolute history-conditioned residual at matched control coordinates ({float(matched['estimate']):.5f}, 95% CI {float(matched['ci95_low']):.5f}-{float(matched['ci95_high']):.5f}) and changed spike-event coordinates ({float(changed['estimate']):.5f}, 95% CI {float(changed['ci95_low']):.5f}-{float(changed['ci95_high']):.5f}). This panel shows association/enrichment at changed spike events, not a causal effect of those events.

For b-e, the independent replication unit is the independently trained network (20 networks, seeds 1000-1019). Bars or diamonds show frozen network means and 95% t confidence intervals. Cells, trials, anchors, and events were not treated as independent replicates. Full network values, technical endpoint names, exclusions, and source hashes are retained in the candidate Source Data and source mapping.
"""


def _caption(payload: Mapping[str, Any]) -> str:
    behavior = payload["behavior_stat_rows"]
    updates = payload["update_stat_rows"]
    events = payload["event_stat_rows"]
    contrasts = payload["contrasts"]
    rescue_values = contrasts.loc[
        contrasts["technical_endpoint"].eq("rescue"),
        "matching_history_change_percent",
    ]
    loss_values = contrasts.loc[
        contrasts["technical_endpoint"].eq("loss"),
        "matching_history_change_percent",
    ]
    rescue_consistent = int((rescue_values > 0).sum())
    loss_consistent = int((loss_values < 0).sum())
    rescue_ties = int((rescue_values == 0).sum())
    rescue = behavior["rescue"]
    loss = behavior["loss"]
    common = updates["same_B_common_update_cosine"]
    history = updates["processing_residual_gamma_norm_ratio"]
    matched = events["matched_random"]
    changed = events["changed_events"]
    comparison_rows = len(payload["joint"])
    both_above = int(payload["both_above_criteria"])
    outline_mass_text = int(
        round(float(payload["density"]["outline_mass"]) * 100)
    )
    return f"""**Fig.3 | Silent inherited history conditions processing of the same current input.**

**a,** Paired exact-B counterfactual. Two distinct one-item histories are followed by a 200-ms no-input interval and form differently patterned inherited STSP states. The same persisted B bitmap is then presented in two separate runs; state and choice are compared only after B. **b,** Aligned-minus-mismatched effects on opportunity-defined Rescue and Loss. Rescue changed by {float(rescue['estimate']):+.1f} percentage points (95% CI {float(rescue['ci95_low']):+.1f} to {float(rescue['ci95_high']):+.1f}); {rescue_consistent}/20 networks had the same positive direction and {rescue_ties} network had no difference. Loss changed by {float(loss['estimate']):+.1f} percentage points (95% CI {float(loss['ci95_low']):+.1f} to {float(loss['ci95_high']):+.1f}); {loss_consistent}/20 networks had the same negative direction. Frozen paired inference used two-sided one-sample t tests against zero with Benjamini-Hochberg adjustment (adjusted P = {_format_p(rescue['p_adjusted'])} and {_format_p(loss['p_adjusted'])}, respectively). **c,** Joint exact-B comparison distribution of Update similarity (`same_B_common_update_cosine`) and History effect (`processing_residual_gamma_energy_fraction`, stored as a norm ratio). Each network contributed a normalized density from exactly 500 valid one-step comparisons; the displayed density is the equal-weight mean across 20 networks. Continuous tone represents the density field, and the thin outline encloses the highest-density {outline_mass_text}% probability mass. The diamond is the 20-network mean with frozen horizontal and vertical 95% CIs: Update similarity {float(common['estimate']):.4f} ({float(common['ci95_low']):.4f}-{float(common['ci95_high']):.4f}) and History effect {float(history['estimate']):.4f} ({float(history['ci95_low']):.4f}-{float(history['ci95_high']):.4f}). Dashed references are the prespecified {float(common['null_value']):g} and {float(history['null_value']):g} criteria. All {both_above:,}/{comparison_rows:,} persisted comparisons exceeded both criteria; comparison rows describe the joint distribution and are not independent inferential replicates. The cloud shape is descriptive and is not interpreted as a correlation or trade-off. **d,** Mean absolute history-conditioned residual at matched control coordinates ({float(matched['estimate']):.5f}, 95% CI {float(matched['ci95_low']):.5f}-{float(matched['ci95_high']):.5f}) and changed spike-event coordinates ({float(changed['estimate']):.5f}, 95% CI {float(changed['ci95_low']):.5f}-{float(changed['ci95_high']):.5f}). This panel shows association/enrichment at changed spike events, not a causal effect of those events.

For b-d, the independent replication unit is the independently trained network (20 networks, seeds 1000-1019). Cells, trials, anchors, comparisons, and events were not treated as independent network replicates. Full network values, comparison rows, density materialization, technical endpoint names, exclusions, and source hashes are retained in the candidate Source Data and source mapping.
"""


def _render_figure(
    spec: Mapping[str, Any],
    payload: Mapping[str, Any],
    figures_dir: Path,
) -> dict[str, Path]:
    canvas_mm = [float(value) for value in spec["canvas_mm"]]
    canvas_width, canvas_height = canvas_mm
    png = figures_dir / "manuscript_fig3.png"
    svg = figures_dir / "manuscript_fig3.svg"
    pdf = figures_dir / "manuscript_fig3.pdf"
    with plt.rc_context(
        {
            **VECTOR_TEXT_RCPARAMS,
            "svg.hashsalt": "net_torch_manuscript_fig3_reader_first_v3",
            "axes.unicode_minus": True,
            "image.composite_image": False,
        }
    ):
        figure = plt.figure(
            figsize=(canvas_width * MM_TO_INCH, canvas_height * MM_TO_INCH),
            dpi=300,
            facecolor="white",
        )
        for panel_id, panel_spec in spec["panels"].items():
            axis = figure.add_axes(
                _as_axes_bbox(panel_spec["plot_bbox_mm"], canvas_mm)
            )
            chart = str(panel_spec["chart"])
            if chart == "exact_b_paired_counterfactual":
                _plot_panel_a(axis, payload, panel_spec)
            elif chart == "horizontal_diverging_effect":
                _plot_panel_b(axis, payload, panel_spec)
            elif chart == "network_balanced_joint_density":
                _plot_joint_density(axis, payload, panel_spec)
            elif chart == "category_bars":
                _plot_panel_d(axis, payload, panel_spec)
            else:
                raise ValueError(f"unknown candidate chart: {chart}")
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
            svg,
            format="svg",
            facecolor="white",
            bbox_inches=None,
            metadata={"Date": None, "Creator": CANDIDATE_VERSION},
        )
        figure.savefig(
            pdf,
            format="pdf",
            facecolor="white",
            bbox_inches=None,
            metadata={
                "Creator": CANDIDATE_VERSION,
                "CreationDate": None,
                "ModDate": None,
            },
        )
        figure.savefig(
            png,
            format="png",
            dpi=300,
            facecolor="white",
            bbox_inches=None,
            metadata={"Software": CANDIDATE_VERSION},
        )
        plt.close(figure)
    return {"png": png, "svg": svg, "pdf": pdf}


def _tag_name(element: ET.Element) -> str:
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
        "svg_expected_stimulus_images": svg_image_count == 4,
        "svg_has_vector_paths": svg_path_count > 0,
        "pdf_one_page": len(pdf_reader.pages) == 1,
        "pdf_width_mm": math.isclose(
            width_pt / MM_TO_POINT, float(canvas_mm[0]), abs_tol=0.05
        ),
        "pdf_height_mm": math.isclose(
            height_pt / MM_TO_POINT, float(canvas_mm[1]), abs_tol=0.05
        ),
        "pdf_embedded_font_resources": font_count > 0,
        "pdf_panel_labels_present": all(letter in extracted_text for letter in "abcd"),
    }
    return {
        "schema": "manuscript_fig3_candidate_render_qa_v1",
        "generated_at": _utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
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
    decomposition_dir: Path,
    output_dir: Path,
    check_only: bool,
) -> dict[str, Any]:
    spec = _load_spec()
    repo_root = _repo_root().resolve()
    expected_parent = (repo_root / spec["parent_bundle"]).resolve()
    expected_decomposition = (
        repo_root / spec["decomposition_parent_root"]
    ).resolve()
    parent_dir = parent_dir.resolve()
    decomposition_dir = decomposition_dir.resolve()
    output_dir = output_dir.resolve()
    for source_root in (parent_dir, decomposition_dir):
        if _inside(output_dir, source_root) or _inside(source_root, output_dir):
            raise ValueError("candidate output and pinned parent sources must be separate trees")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    (output_dir / "figures" / "qa").mkdir(parents=True, exist_ok=True)

    source_pattern = str(spec["panels"]["c"]["source_pattern"])
    decomposition_files = tuple(
        source_pattern.format(seed=seed) for seed in EXPECTED_SEEDS
    )
    bundle_before = _snapshot_tree(parent_dir)
    bundle_before.insert(0, "source_scope", "frozen_bundle")
    decomposition_before = _snapshot_selected(
        decomposition_dir,
        decomposition_files,
        source_scope="decomposition",
    )
    parent_before = pd.concat(
        [bundle_before, decomposition_before], ignore_index=True
    )
    before_digest = _snapshot_digest(parent_before)
    reader = BundleReader(
        parent_dir=parent_dir,
        expected_parent_dir=expected_parent,
        decomposition_dir=decomposition_dir,
        expected_decomposition_dir=expected_decomposition,
        allowed_decomposition_files=decomposition_files,
    )
    payload = _load_sources(reader, spec)
    layout_audit = _layout_audit(spec)
    if layout_audit["status"] != "passed":
        raise ValueError(f"candidate layout contract failed: {layout_audit['failures']}")

    tables = _candidate_tables(payload)
    metric_table_names = {
        "panel_a_statistics",
        "panel_b_statistics",
        "panel_c_statistics",
        "panel_c_density_summary",
        "panel_c_contour_levels",
        "panel_d_statistics",
    }
    for name, frame in tables.items():
        target_root = output_dir / (
            "metrics" if name in metric_table_names else "data"
        )
        frame.to_csv(target_root / f"{name}.csv", index=False)
    source_mapping = _source_mapping(parent_dir, reader)
    source_mapping.to_csv(output_dir / "meta" / "source_mapping.csv", index=False)
    payload["decomposition_manifest"].to_csv(
        output_dir / "meta" / "panel_c_source_manifest.csv", index=False
    )
    payload["source_manifests"]["source_manifest.csv"].to_csv(
        output_dir / "meta" / "parent_source_manifest.csv", index=False
    )
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

    direction_counts = {}
    for endpoint, expected_direction in (("rescue", "positive"), ("loss", "negative")):
        values = payload["contrasts"].loc[
            payload["contrasts"]["technical_endpoint"].eq(endpoint),
            "matching_history_change_percent",
        ]
        direction_counts[endpoint] = {
            "positive": int((values > 0).sum()),
            "negative": int((values < 0).sum()),
            "ties": int((values == 0).sum()),
            "caption_direction": expected_direction,
        }

    render_qa: dict[str, Any] | None = None
    outputs: dict[str, Path] = {}
    if not check_only:
        _render_wireframe(
            spec, output_dir / "figures" / "qa" / "manuscript_fig3_wireframe.png"
        )
        outputs = _render_figure(spec, payload, output_dir / "figures")
        with Image.open(outputs["png"]) as image:
            image.convert("L").save(
                output_dir / "figures" / "qa" / "manuscript_fig3_grayscale.png",
                dpi=(300, 300),
            )
        render_qa = _render_qa(outputs, spec["canvas_mm"])
        _write_json(output_dir / "meta" / "render_qa.json", render_qa)
        if render_qa["status"] != "passed":
            raise ValueError(f"candidate render QA failed: {render_qa['checks']}")
        _write_json(
            output_dir / "meta" / "visual_qa.json",
            {
                "schema": "manuscript_fig3_candidate_visual_qa_v1",
                "status": "pending_manual_review",
                "final_size_mm": spec["canvas_mm"],
                "artifact": "figures/manuscript_fig3.png",
                "checks": [],
                "notes": "Automated render QA passed; final-size composite inspection is pending.",
            },
        )

    bundle_after = _snapshot_tree(parent_dir)
    bundle_after.insert(0, "source_scope", "frozen_bundle")
    decomposition_after = _snapshot_selected(
        decomposition_dir,
        decomposition_files,
        source_scope="decomposition",
    )
    parent_after = pd.concat(
        [bundle_after, decomposition_after], ignore_index=True
    )
    after_digest = _snapshot_digest(parent_after)
    parent_after.to_csv(output_dir / "meta" / "parent_hashes_after.csv", index=False)
    parent_unchanged = parent_before.equals(parent_after)
    parent_integrity = {
        "schema": "manuscript_fig3_candidate_parent_integrity_v1",
        "status": "passed" if parent_unchanged else "failed",
        "parent_sources": {
            "frozen_bundle": str(parent_dir),
            "decomposition_root": str(decomposition_dir),
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
        "display_name": "Fig.3",
        "plot_only": True,
        "check_only": check_only,
        "parent_bundle": str(parent_dir),
        "decomposition_parent_root": str(decomposition_dir),
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
        "display_name": "Fig.3",
        "status": "check_passed" if check_only else "rendered_pending_visual_review",
        "review_only": True,
        "canvas_mm": spec["canvas_mm"],
        "independent_unit": "network_seed",
        "n_networks": 20,
        "network_seeds": [1000, 1019],
        "direction_counts": direction_counts,
        "joint_distribution": {
            "comparison_rows": int(len(payload["joint"])),
            "comparisons_per_network": 500,
            "rows_above_both_criteria": int(payload["both_above_criteria"]),
            "density_weighting": "equal network weight",
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
        description="Render the review-only reader-first manuscript Fig.3 candidate."
    )
    parser.add_argument(
        "--parent-dir",
        default=(
            "results/paper_figure_multi_seed/"
            "final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig2"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/paper_figure_candidates/manuscript_fig3_reader_first_v3",
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
    pinned_decomposition = (
        repo_root / spec["decomposition_parent_root"]
    ).resolve()
    if args.refresh_manifest:
        for source_root in (pinned_parent, pinned_decomposition):
            if _inside(output_dir, source_root) or _inside(source_root, output_dir):
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
        decomposition_dir=pinned_decomposition,
        output_dir=output_dir,
        check_only=bool(args.check_only),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
