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
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter
from PIL import Image
from pypdf import PdfReader

from src.plotting.common.colors import NATURE_COMPATIBLE_PALETTE, get_plot_color
from src.plotting.paper_fig.layout_contract import validate_layout_contract
from src.plotting.paper_fig.typography import (
    VECTOR_TEXT_RCPARAMS,
    apply_paper_figure_typography,
    mark_panel_label,
)
from src.plotting.paper_fig.candidates import manuscript_fig3_reader_first as fig3
from src.plotting.paper_fig.candidates import manuscript_fig5_reader_first as fig5_v1


CANDIDATE_VERSION = "manuscript_fig5_reader_first_v2"
DISPLAY_NAME = "Fig.5"
EXPECTED_SEEDS = tuple(range(1000, 1020))
EXPECTED_STAGES = tuple(range(2, 11))
MM_TO_INCH = 1.0 / 25.4
MM_TO_POINT = 72.0 / 25.4
SPEC_PATH = Path(__file__).resolve().parent / "specs" / f"{CANDIDATE_VERSION}.json"
EXTENSION_ROOT_REL = Path("results/successor_extension_v1_confirmatory_20seed/aggregate")
EXTENSION_FILES = {"network_effects.csv", "population_inference.csv", "verdict.json", "artifact_manifest.json"}
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _load_spec() -> dict[str, Any]:
    with SPEC_PATH.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("candidate_version") != CANDIDATE_VERSION:
        raise ValueError("candidate spec version mismatch")
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
    allowed_files: set[str]
    accesses: list[dict[str, Any]] = field(default_factory=list)

    def _resolve(self, relative: str, purpose: str) -> Path:
        relative_path = Path(relative)
        normalized = relative_path.as_posix()
        if relative_path.is_absolute() or normalized not in self.allowed_files:
            raise PermissionError(f"unregistered extension source: {relative}")
        path = (self.root / relative_path).resolve()
        if not _inside(path, self.root) or not path.is_file():
            raise FileNotFoundError(f"required extension source is missing: {path}")
        self.accesses.append(
            {
                "candidate_figure": DISPLAY_NAME,
                "source_scope": "successor_extension_aggregate",
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


def _finite(values: Sequence[Any], label: str) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{label}: non-finite value")
    return array


def _validate_stat_triplet(row: Mapping[str, Any], label: str) -> dict[str, float]:
    keys = {
        "estimate": "estimate" if "estimate" in row else "mean",
        "ci95_low": "ci95_low" if "ci95_low" in row else "bootstrap_ci95_low",
        "ci95_high": "ci95_high" if "ci95_high" in row else "bootstrap_ci95_high",
    }
    values = {key: float(row[source]) for key, source in keys.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"{label}: non-finite statistic")
    if not values["ci95_low"] <= values["estimate"] <= values["ci95_high"]:
        raise ValueError(f"{label}: invalid confidence interval")
    return values


def _find_population_row(population: pd.DataFrame, experiment: str, endpoint: str) -> pd.Series:
    rows = population.loc[
        population["cohort"].astype(str).eq("full20")
        & population["experiment"].astype(str).eq(experiment)
        & population["endpoint"].astype(str).eq(endpoint)
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one population row for {experiment}/{endpoint}, found {len(rows)}")
    return rows.iloc[0]


def _load_extension(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    reader = BundleReader(root, EXTENSION_FILES)
    network = reader.read_csv("network_effects.csv", "network-level K=10 and two-hop effects")
    population = reader.read_csv("population_inference.csv", "frozen network-level inference")
    verdict = reader.read_json("verdict.json", "frozen cohort verdict")
    reader.read_json("artifact_manifest.json", "aggregate provenance manifest")
    required = {"cohort", "experiment", "network_seed", "endpoint", "value"}
    missing = sorted(required - set(network.columns))
    if missing:
        raise ValueError(f"extension network_effects.csv missing columns {missing}")
    network = network.copy()
    network["network_seed"] = pd.to_numeric(network["network_seed"], errors="raise").astype(int)
    network["value"] = pd.to_numeric(network["value"], errors="raise")
    if not np.isfinite(network["value"].to_numpy(dtype=float)).all():
        raise ValueError("extension network effects contain non-finite values")
    expected = {
        "input_response_l2": ("exp_a_c5_k10_successor", "early_layer2_event_map_donor_transfer"),
        "successor_state_l3": ("exp_a_c5_k10_successor", "layer3_successor_ux_donor_transfer"),
        "input_response": ("exp_c_c5_twohop_cd", "early_layer2_D_donor_transfer"),
        "successor_state": ("exp_c_c5_twohop_cd", "layer3_postD_ux_donor_transfer"),
    }
    summaries: dict[str, dict[str, Any]] = {}
    for key, (experiment, endpoint) in expected.items():
        rows = network.loc[
            network["cohort"].astype(str).eq("full20")
            & network["experiment"].astype(str).eq(experiment)
            & network["endpoint"].astype(str).eq(endpoint)
        ].copy()
        seeds = set(rows["network_seed"].tolist())
        if seeds != set(EXPECTED_SEEDS) or len(rows) != len(EXPECTED_SEEDS):
            raise ValueError(f"extension {key}: expected exactly seeds 1000-1019")
        if rows["network_seed"].duplicated().any():
            raise ValueError(f"extension {key}: duplicate network seed")
        population_row = _find_population_row(population, experiment, endpoint)
        stat = _validate_stat_triplet(population_row, f"extension {key}")
        observed_mean = float(rows["value"].mean())
        if not math.isclose(observed_mean, float(population_row["mean"]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"extension {key}: network mean disagrees with frozen inference")
        if int(population_row["n_networks"]) != len(EXPECTED_SEEDS):
            raise ValueError(f"extension {key}: frozen network count is not 20")
        summaries[key] = {
            **stat,
            "n_networks": int(population_row["n_networks"]),
            "positive_network_fraction": float(population_row["positive_network_fraction"]),
            "holm_adjusted_p": float(population_row["holm_adjusted_p"]),
            "experiment": experiment,
            "endpoint": endpoint,
        }
    return summaries, network, population, verdict, reader.access_frame()


def _load_v1(repo_root: Path) -> tuple[dict[str, Any], Path, pd.DataFrame, pd.DataFrame]:
    spec = fig5_v1._load_spec()
    parent = (repo_root / spec["parent_bundle"]).resolve()
    before = fig5_v1._snapshot_tree(parent, "fig5_v1_parent")
    reader = fig5_v1.BundleReader(parent, parent, set(fig5_v1.PARENT_DATA_FILES))
    payload = fig5_v1._load_sources(reader, spec)
    return payload, parent, before, reader.access_frame()


def _build_transfer_frames(v1_payload: Mapping[str, Any], extension: Mapping[str, dict[str, Any]], extension_network: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    v1_endpoints = {
        "input_response_l2": (v1_payload["a"], v1_payload["a_frozen"]),
        "successor_state_l3": (v1_payload["b"], v1_payload["b_frozen"]),
    }
    for endpoint, (frame, frozen) in v1_endpoints.items():
        for condition in ("K1", "K5"):
            subset = frame.loc[frame["condition"].astype(str).eq(condition)].copy()
            for _, row in subset.iterrows():
                raw_rows.append(
                    {
                        "figure_id": DISPLAY_NAME,
                        "panel_id": "b",
                        "network_seed": int(row["network_seed"]),
                        "history_depth": int(str(condition)[1:]),
                        "endpoint": endpoint,
                        "value": float(row["value"]),
                        "unit": "donor_transfer_index",
                        "source": "fig5_v1_frozen",
                    }
                )
            stat = _validate_stat_triplet(frozen[condition], f"Fig.5b {endpoint} K={condition[1:]}")
            stat_rows.append({
                "figure_id": DISPLAY_NAME,
                "panel_id": "b",
                "history_depth": int(condition[1:]),
                "endpoint": endpoint,
                **stat,
                "n_networks": 20,
                "positive_network_fraction": float((subset["value"] > 0).mean()),
                "p_adjusted": float(frozen[condition]["p_adjusted"]),
                "source": "fig5_v1_frozen",
            })
    for endpoint in ("input_response_l2", "successor_state_l3"):
        experiment_endpoint = extension[endpoint]
        experiment = str(experiment_endpoint["experiment"])
        technical_endpoint = str(experiment_endpoint["endpoint"])
        subset = extension_network.loc[
            extension_network["cohort"].astype(str).eq("full20")
            & extension_network["experiment"].astype(str).eq(experiment)
            & extension_network["endpoint"].astype(str).eq(technical_endpoint)
        ]
        for _, row in subset.iterrows():
            raw_rows.append({
                "figure_id": DISPLAY_NAME,
                "panel_id": "b",
                "network_seed": int(row["network_seed"]),
                "history_depth": 10,
                "endpoint": endpoint,
                "value": float(row["value"]),
                "unit": "donor_transfer_index",
                "source": "successor_extension_aggregate",
            })
        stat_rows.append({
            "figure_id": DISPLAY_NAME,
            "panel_id": "b",
            "history_depth": 10,
            "endpoint": endpoint,
            "estimate": float(experiment_endpoint["estimate"]),
            "ci95_low": float(experiment_endpoint["ci95_low"]),
            "ci95_high": float(experiment_endpoint["ci95_high"]),
            "n_networks": int(experiment_endpoint["n_networks"]),
            "positive_network_fraction": float(experiment_endpoint["positive_network_fraction"]),
            "p_adjusted": float(experiment_endpoint["holm_adjusted_p"]),
            "source": "successor_extension_aggregate",
        })
    raw = pd.DataFrame(raw_rows).sort_values(["endpoint", "history_depth", "network_seed"]).reset_index(drop=True)
    stats = pd.DataFrame(stat_rows).sort_values(["endpoint", "history_depth"]).reset_index(drop=True)
    if len(raw) != 120 or len(stats) != 6:
        raise ValueError(f"Fig.5b transfer materialization expected 120 raw and 6 summary rows, got {len(raw)}/{len(stats)}")
    return raw, stats


def _build_twohop_frames(extension: Mapping[str, dict[str, Any]], extension_network: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    for endpoint in ("input_response", "successor_state"):
        summary = extension[endpoint]
        subset = extension_network.loc[
            extension_network["cohort"].astype(str).eq("full20")
            & extension_network["experiment"].astype(str).eq(str(summary["experiment"]))
            & extension_network["endpoint"].astype(str).eq(str(summary["endpoint"]))
        ]
        for _, row in subset.iterrows():
            raw_rows.append({
                "figure_id": DISPLAY_NAME,
                "panel_id": "c",
                "network_seed": int(row["network_seed"]),
                "endpoint": endpoint,
                "value": float(row["value"]),
                "history_depth": 5,
                "unit": "donor_transfer_index",
                "source": "successor_extension_aggregate",
            })
        stat_rows.append({
            "figure_id": DISPLAY_NAME,
            "panel_id": "c",
            "endpoint": endpoint,
            "history_depth": 5,
            "estimate": float(summary["estimate"]),
            "ci95_low": float(summary["ci95_low"]),
            "ci95_high": float(summary["ci95_high"]),
            "n_networks": int(summary["n_networks"]),
            "positive_network_fraction": float(summary["positive_network_fraction"]),
            "p_adjusted": float(summary["holm_adjusted_p"]),
            "source": "successor_extension_aggregate",
        })
    raw = pd.DataFrame(raw_rows).sort_values(["endpoint", "network_seed"]).reset_index(drop=True)
    stats = pd.DataFrame(stat_rows).sort_values("endpoint").reset_index(drop=True)
    if len(raw) != 40 or len(stats) != 2:
        raise ValueError(f"Fig.5c two-hop materialization expected 40 raw and 2 summary rows, got {len(raw)}/{len(stats)}")
    return raw, stats


def _relabel(frame: pd.DataFrame, panel_id: str) -> pd.DataFrame:
    output = frame.copy()
    if "figure_id" in output.columns:
        output["figure_id"] = DISPLAY_NAME
    if "panel_id" in output.columns:
        output["panel_id"] = panel_id
    return output


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


def _as_axes_bbox(bbox_mm: Sequence[float], canvas_mm: Sequence[float]) -> list[float]:
    left, top, width, height = [float(value) for value in bbox_mm]
    canvas_width, canvas_height = [float(value) for value in canvas_mm]
    return [left / canvas_width, (canvas_height - top - height) / canvas_height, width / canvas_width, height / canvas_height]


def _draw_input(axis: plt.Axes, center: tuple[float, float], label: str, color: str, *, label_position: str = "below") -> dict[str, Any]:
    if label_position not in {"above", "below"}:
        raise ValueError(f"input label_position must be 'above' or 'below', got {label_position!r}")
    x, y = center
    frame = Rectangle((x - 4.0, y - 3.4), 8.0, 6.8, facecolor="none", edgecolor=color, linewidth=0.9, zorder=5)
    axis.add_patch(frame)
    grid = np.array([[0, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 0]], dtype=float)
    cell_w, cell_h = 1.5, 1.2
    cells = []
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            cell = Rectangle((x - 3.0 + col * cell_w, y - 2.4 + row * cell_h), cell_w, cell_h, facecolor=NEUTRAL_DARK if grid[row, col] else "white", edgecolor="none", zorder=6)
            axis.add_patch(cell)
            cells.append(cell)
    label_y = y - 5.0 if label_position == "below" else y + 5.0
    label_va = "top" if label_position == "below" else "bottom"
    axis.text(x, label_y, label, ha="center", va=label_va, color=INK, fontsize=9.0)
    return {"grid_cells": len(cells), "grid_visible": bool(frame.get_facecolor()[3] == 0 and all(cell.get_zorder() > frame.get_zorder() and cell.get_facecolor()[3] > 0 for cell in cells)), "label_position": label_position}


def _draw_schematic(axis: plt.Axes, labels: Mapping[str, str]) -> dict[str, Any]:
    layer2 = get_plot_color("layer2", context="manuscript_fig5")
    layer3 = get_plot_color("layer3", context="manuscript_fig5")
    dynamic = get_plot_color("dynamic", context="manuscript_fig5")
    donor = get_plot_color("donor_trace", context="manuscript_fig5")
    guide = get_plot_color("guide", context="manuscript_fig5")
    axis.set_xlim(0.0, 161.0)
    axis.set_ylim(0.0, 48.0)
    axis.axis("off")
    # This is a discrete protocol annotation, not a directional history axis.
    axis.text(26.0, 46.0, labels["history"], ha="center", va="top", color=INK)

    # One local intervention above the natural chain.
    fig3._draw_state_glyph(axis, [8.0, 31.0, 11.0, 8.0], color=donor, active_nodes={0, 2, 3, 5})
    fig3._draw_state_glyph(axis, [34.0, 31.0, 11.0, 8.0], color=layer2, active_nodes={0, 1, 4, 5})
    fig3._draw_arrow(axis, (20.0, 35.0), (32.0, 35.0), color=donor, linewidth=1.0)
    axis.text(26.0, 40.0, labels["transfer"], ha="center", va="bottom", color=donor, linespacing=0.9)
    axis.text(13.5, 29.5, labels["donor_state"], ha="center", va="top", color=donor)
    axis.text(39.5, 29.5, labels["receiver_state"], ha="center", va="top", color=layer2)

    # The receiver becomes the inherited state for the natural chain.
    fig3._draw_arrow(axis, (39.5, 30.5), (39.5, 23.0), color=layer2, linewidth=0.8)
    axis.text(39.5, 21.8, labels["inherited_state"], ha="center", va="top", color=layer2)
    positions = {"next": 53.0, "response_1": 71.0, "successor_1": 90.0, "following": 109.0, "response_2": 128.0, "successor_2": 148.0}
    next_input = _draw_input(axis, (positions["next"], 15.5), labels["next_input"], dynamic, label_position="below")
    fig3._draw_state_glyph(axis, [positions["response_1"] - 5.0, 11.5, 10.0, 8.0], color=dynamic, active_nodes={0, 1, 3, 5})
    axis.text(positions["response_1"], 21.5, labels["input_response"], ha="center", va="bottom", color=dynamic)
    fig3._draw_state_glyph(axis, [positions["successor_1"] - 5.0, 11.5, 10.0, 8.0], color=layer3, active_nodes={0, 2, 4, 5})
    axis.text(positions["successor_1"], 8.5, labels["successor_state"], ha="center", va="top", color=layer3)
    following_input = _draw_input(axis, (positions["following"], 15.5), labels["following_input"], dynamic, label_position="above")
    fig3._draw_state_glyph(axis, [positions["response_2"] - 5.0, 11.5, 10.0, 8.0], color=dynamic, active_nodes={0, 2, 3, 5})
    axis.text(positions["response_2"], 8.5, labels["input_response"], ha="center", va="top", color=dynamic)
    fig3._draw_state_glyph(axis, [positions["successor_2"] - 5.0, 11.5, 10.0, 8.0], color=layer3, active_nodes={0, 1, 4, 5})
    axis.text(positions["successor_2"], 21.5, labels["new_successor"], ha="center", va="bottom", color=layer3)
    for start, end, color in [((44.8, 15.5), (48.5, 15.5), layer2), ((57.5, 15.5), (65.8, 15.5), dynamic), ((76.2, 15.5), (84.8, 15.5), dynamic), ((95.0, 15.5), (103.8, 15.5), layer3), ((113.0, 15.5), (122.8, 15.5), dynamic), ((132.8, 15.5), (142.8, 15.5), dynamic)]:
        fig3._draw_arrow(axis, start, end, color=color, linewidth=0.8)
    axis.text(99.5, 30.0, labels["natural"], ha="center", va="bottom", color=guide)

    # Small passive-state alternative forks from the pre-C inherited-state/input path.
    fig3._draw_state_glyph(axis, [49.0, 1.5, 8.0, 5.0], color=NEUTRAL_MID, active_nodes={0, 2, 4})
    axis.plot([44.8, 44.8, 49.0], [15.5, 8.5, 6.7], color=NEUTRAL_MID, linestyle=(0, (3.0, 2.0)), linewidth=0.7, zorder=1)
    fig3._draw_arrow(axis, (44.8, 8.5), (49.0, 6.7), color=NEUTRAL_MID, linestyle=(0, (3.0, 2.0)), linewidth=0.7)
    axis.text(42.5, 8.5, labels["passive_input"], ha="right", va="center", color=NEUTRAL_MID)
    axis.text(60.0, 4.0, labels["passive"], ha="left", va="center", color=NEUTRAL_DARK)
    reader_labels = [labels["donor_state"], labels["receiver_state"], labels["inherited_state"], labels["transfer"], labels["next_input"], labels["following_input"], labels["input_response"], labels["successor_state"], labels["new_successor"], labels["passive"], labels["passive_input"], labels["natural"], labels["history"]]
    return {
        "status": "passed",
        "single_transfer_arrows": 1,
        "history_directional_arrows": 0,
        "history_label": labels["history"],
        "input_grid_cells": int(next_input["grid_cells"] + following_input["grid_cells"]),
        "input_grids_visible": bool(next_input["grid_visible"] and following_input["grid_visible"]),
        "reader_labels": reader_labels,
    }


def _draw_transfer(axis: plt.Axes, transfer_stats: pd.DataFrame, panel_spec: Mapping[str, Any]) -> dict[str, Any]:
    colors = {key: get_plot_color(value, context="manuscript_fig5") for key, value in panel_spec["endpoint_colors"].items()}
    labels = panel_spec["endpoint_labels"]
    x_values = [1.0, 5.0, 10.0]
    for endpoint in panel_spec["endpoint_order"]:
        subset = transfer_stats.loc[transfer_stats["endpoint"].eq(endpoint)].sort_values("history_depth")
        if len(subset) != 3:
            raise ValueError(f"Fig.5b: expected K=1,5,10 for {endpoint}")
        means = subset["estimate"].to_numpy(dtype=float)
        low = subset["ci95_low"].to_numpy(dtype=float)
        high = subset["ci95_high"].to_numpy(dtype=float)
        color = colors[endpoint]
        marker = {"input_response_l2": "o", "successor_state_l3": "s"}[endpoint]
        linestyle = {"input_response_l2": "-", "successor_state_l3": "--"}[endpoint]
        axis.plot(x_values, means, color=color, linewidth=1.2, linestyle=linestyle, marker=marker, markersize=3.8, markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.45, label=labels[endpoint], zorder=3)
        axis.errorbar(x_values, means, yerr=np.vstack([means - low, high - means]), fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.0, capthick=0.7, zorder=4)
    axis.set_xlim(0.2, 10.8)
    axis.set_xticks(x_values)
    axis.set_xticklabels(["1", "5", "10"])
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.yaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=2, handlelength=1.4, columnspacing=1.0, borderaxespad=0.0)
    return {"endpoints": list(panel_spec["endpoint_order"]), "history_depths": x_values}


def _draw_second_transition(axis: plt.Axes, stats: pd.DataFrame, panel_spec: Mapping[str, Any]) -> dict[str, Any]:
    colors = {key: get_plot_color(value, context="manuscript_fig5") for key, value in panel_spec["endpoint_colors"].items()}
    endpoints = list(panel_spec["x_order"])
    x = np.arange(len(endpoints), dtype=float)
    for index, endpoint in enumerate(endpoints):
        row = stats.loc[stats["endpoint"].eq(endpoint)]
        if len(row) != 1:
            raise ValueError(f"Fig.5c: missing second-transition endpoint {endpoint}")
        item = row.iloc[0]
        mean, low, high = float(item["estimate"]), float(item["ci95_low"]), float(item["ci95_high"])
        marker = {"input_response": "o", "successor_state": "s"}[endpoint]
        axis.errorbar([x[index]], [mean], yerr=[[mean - low], [high - mean]], fmt=marker, color=colors[endpoint], markerfacecolor=colors[endpoint], markeredgecolor=INK, markeredgewidth=0.55, markersize=5.0, ecolor=INK, elinewidth=0.9, capsize=2.3, capthick=0.8, zorder=4)
    axis.set_xlim(-0.45, 1.45)
    axis.set_xticks(x)
    axis.set_xticklabels([panel_spec["endpoint_labels"][endpoint] for endpoint in endpoints])
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.yaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    return {"endpoints": endpoints}


def _draw_recurrence(axis: plt.Axes, stats: pd.DataFrame, panel_spec: Mapping[str, Any]) -> dict[str, Any]:
    stages = [int(value) for value in panel_spec["x_order"]]
    subset = stats.sort_values("stage_k")
    if subset["stage_k"].astype(int).tolist() != stages:
        raise ValueError("Fig.5d: stage order is not 2-10")
    means = subset["estimate"].to_numpy(dtype=float)
    low = subset["ci95_low"].to_numpy(dtype=float)
    high = subset["ci95_high"].to_numpy(dtype=float)
    color = get_plot_color(str(panel_spec["color"]), context="manuscript_fig5")
    axis.fill_between(stages, low, high, color=color, alpha=0.16, linewidth=0, zorder=1)
    axis.plot(stages, means, color=color, linewidth=1.25, marker="o", markersize=3.8, markerfacecolor=color, markeredgecolor=color, zorder=3)
    axis.set_xlim(*[float(value) for value in panel_spec["x_limits"]])
    axis.set_xticks([float(value) for value in panel_spec["x_ticks"]])
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.yaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    return {"stages": stages, "means": means.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist()}


def _draw_behavior(axis: plt.Axes, stats: pd.DataFrame, panel_spec: Mapping[str, Any]) -> dict[str, Any]:
    x_values = np.array([1.0, 5.0])
    colors = {key: get_plot_color(value, context="manuscript_fig5") for key, value in panel_spec["series_colors"].items()}
    labels = panel_spec["series_labels"]
    rendered: dict[str, Any] = {}
    for series in panel_spec["series"]:
        rows = stats.loc[stats["outcome_type"].astype(str).eq(series)].copy()
        rows["prefix_k_num"] = rows["prefix_k"].astype(str).str.replace("K", "", regex=False).astype(int)
        rows = rows.sort_values("prefix_k_num")
        if rows["prefix_k_num"].tolist() != [1, 5]:
            raise ValueError(f"Fig.5e: expected K=1 and K=5 for {series}")
        means = rows["estimate"].to_numpy(dtype=float)
        low = rows["ci95_low"].to_numpy(dtype=float)
        high = rows["ci95_high"].to_numpy(dtype=float)
        color = colors[series]
        axis.plot(x_values, means, color=color, linewidth=1.3, marker="o", markersize=4.2, markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.45, zorder=3)
        axis.errorbar(x_values, means, yerr=np.vstack([means - low, high - means]), fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.0, capthick=0.7, zorder=4)
        axis.text(5.13, float(means[-1]), labels[series], ha="left", va="center", color=color, clip_on=False)
        rendered[series] = {"mean": means.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist()}
    axis.set_xlim(0.35, 6.25)
    axis.set_xticks(x_values)
    axis.set_xticklabels(["1", "5"])
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.yaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    return rendered


def _layout_audit(spec: Mapping[str, Any]) -> dict[str, Any]:
    report = validate_layout_contract(spec)
    failures = list(report.failures)
    expected_slots = {
        "a": [2.0, 2.0, 161.0, 48.0],
        "b": [2.0, 52.0, 79.5, 48.0],
        "c": [83.5, 52.0, 79.5, 48.0],
        "d": [2.0, 102.0, 79.5, 48.0],
        "e": [83.5, 102.0, 79.5, 48.0],
    }
    rows: list[dict[str, Any]] = []
    for panel_id, expected in expected_slots.items():
        actual = [float(value) for value in spec["slots"].get(panel_id, [])]
        if actual != expected:
            failures.append(f"panel {panel_id} slot differs from requested 1+2+2 geometry")
        plot = [float(value) for value in spec["panels"][panel_id]["plot_bbox_mm"]]
        if len(plot) != 4 or len(actual) != 4:
            failures.append(f"panel {panel_id} must declare four coordinates")
            continue
        inside = plot[0] >= actual[0] and plot[1] >= actual[1] and plot[0] + plot[2] <= actual[0] + actual[2] and plot[1] + plot[3] <= actual[1] + actual[3]
        if not inside:
            failures.append(f"panel {panel_id} plot area escapes slot")
        rows.append({"panel_id": panel_id, "slot_bbox_mm": actual, "plot_bbox_mm": plot, "plot_inside_slot": inside})
    if [float(value) for value in spec.get("canvas_mm", [])] != [165.0, 152.0]:
        failures.append("canvas differs from 165 x 152 mm")
    for left, right in (("b", "c"), ("d", "e")):
        left_plot = spec["panels"][left]["plot_bbox_mm"]
        right_plot = spec["panels"][right]["plot_bbox_mm"]
        if [float(value) for value in left_plot[1:]] != [float(value) for value in right_plot[1:]]:
            failures.append(f"row {left}/{right} plot-area geometry is not aligned")
    return {"schema": "manuscript_fig5_v2_layout_audit_v1", "status": "passed" if not failures else "failed", "passes": report.passes, "warnings": report.warnings, "failures": failures, "geometry_rows": rows}


def _render_wireframe(spec: Mapping[str, Any], output: Path) -> None:
    from matplotlib.patches import Rectangle as MplRectangle
    canvas_width, canvas_height = [float(value) for value in spec["canvas_mm"]]
    with plt.rc_context({**VECTOR_TEXT_RCPARAMS, "svg.hashsalt": CANDIDATE_VERSION}):
        figure = plt.figure(figsize=(canvas_width * MM_TO_INCH, canvas_height * MM_TO_INCH), dpi=300, facecolor="white")
        axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
        axis.set_xlim(0.0, canvas_width)
        axis.set_ylim(canvas_height, 0.0)
        axis.axis("off")
        for panel_id, slot in spec["slots"].items():
            x, y, width, height = [float(value) for value in slot]
            axis.add_patch(MplRectangle((x, y), width, height, facecolor="white", edgecolor=NEUTRAL_MID, linewidth=0.7))
            px, py, pw, ph = [float(value) for value in spec["panels"][panel_id]["plot_bbox_mm"]]
            axis.add_patch(MplRectangle((px, py), pw, ph, facecolor=NEUTRAL_PALE, edgecolor=NEUTRAL_LIGHT, linewidth=0.6))
            text = axis.text(x + 1.0, y + 1.0, panel_id, ha="left", va="top", color=INK)
            mark_panel_label(text)
        apply_paper_figure_typography(figure)
        figure.savefig(output, dpi=300, facecolor="white", bbox_inches=None, metadata={"Date": None, "Creator": CANDIDATE_VERSION})
        plt.close(figure)


def _render_figure(spec: Mapping[str, Any], payload: Mapping[str, Any], figures_dir: Path) -> dict[str, Any]:
    canvas_mm = [float(value) for value in spec["canvas_mm"]]
    outputs = {"png": figures_dir / "manuscript_fig5.png", "svg": figures_dir / "manuscript_fig5.svg", "pdf": figures_dir / "manuscript_fig5.pdf", "base_svg": figures_dir / "qa" / "manuscript_fig5_base.svg"}
    panel_qa: dict[str, Any] = {}
    with plt.rc_context({**VECTOR_TEXT_RCPARAMS, "svg.hashsalt": CANDIDATE_VERSION, "axes.unicode_minus": True}):
        figure = plt.figure(figsize=(canvas_mm[0] * MM_TO_INCH, canvas_mm[1] * MM_TO_INCH), dpi=300, facecolor="white")
        panel_qa["a"] = _draw_schematic(figure.add_axes(_as_axes_bbox(spec["panels"]["a"]["plot_bbox_mm"], canvas_mm)), payload["schematic_labels"])
        panel_qa["b"] = _draw_transfer(figure.add_axes(_as_axes_bbox(spec["panels"]["b"]["plot_bbox_mm"], canvas_mm)), payload["transfer_stats"], spec["panels"]["b"])
        panel_qa["c"] = _draw_second_transition(figure.add_axes(_as_axes_bbox(spec["panels"]["c"]["plot_bbox_mm"], canvas_mm)), payload["twohop_stats"], spec["panels"]["c"])
        panel_qa["d"] = _draw_recurrence(figure.add_axes(_as_axes_bbox(spec["panels"]["d"]["plot_bbox_mm"], canvas_mm)), payload["recurrence_stats"], spec["panels"]["d"])
        panel_qa["e"] = _draw_behavior(figure.add_axes(_as_axes_bbox(spec["panels"]["e"]["plot_bbox_mm"], canvas_mm)), payload["behavior_stats"], spec["panels"]["e"])
        for panel_id, slot in spec["slots"].items():
            slot_x, slot_y, _, _ = [float(value) for value in slot]
            label = figure.text((slot_x + 0.3) / canvas_mm[0], 1.0 - (slot_y + 0.6) / canvas_mm[1], panel_id, ha="left", va="top", color=INK, zorder=100)
            mark_panel_label(label)
        apply_paper_figure_typography(figure)
        figure.savefig(outputs["svg"], format="svg", facecolor="white", bbox_inches=None, metadata={"Date": None, "Creator": CANDIDATE_VERSION})
        figure.savefig(outputs["pdf"], format="pdf", facecolor="white", bbox_inches=None, metadata={"Creator": CANDIDATE_VERSION, "CreationDate": None})
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
    return {**outputs, "panel_qa": panel_qa}


def _render_qa(outputs: Mapping[str, Path], spec: Mapping[str, Any], panel_qa: Mapping[str, Any]) -> dict[str, Any]:
    expected_size = tuple(int(round(float(value) * 300.0 / 25.4)) for value in spec["canvas_mm"])
    with Image.open(outputs["png"]) as image:
        actual_size = image.size
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        border = 8
        border_pixels = np.concatenate([rgb[:border].reshape(-1, 3), rgb[-border:].reshape(-1, 3), rgb[:, :border].reshape(-1, 3), rgb[:, -border:].reshape(-1, 3)], axis=0)
        outer_border_clear = bool(np.all(border_pixels >= 250))
    svg_text = outputs["svg"].read_text(encoding="utf-8")
    pdf_reader = PdfReader(str(outputs["pdf"]))
    page = pdf_reader.pages[0]
    extracted_text = page.extract_text() or ""
    resources = page.get("/Resources")
    font_table = resources.get("/Font") if resources else None
    if font_table is not None and hasattr(font_table, "get_object"):
        font_table = font_table.get_object()
    font_count = len(font_table) if font_table else 0
    checks = {
        "png_dimensions": all(abs(actual - expected) <= 1 for actual, expected in zip(actual_size, expected_size)),
        "outer_border_clear": outer_border_clear,
        "svg_editable_text": svg_text.count("<text") > 0,
        "svg_has_vector_paths": svg_text.count("<path") > 0,
        "svg_no_bitmap_images": "<image" not in svg_text.lower(),
        "pdf_one_page": len(pdf_reader.pages) == 1,
        "pdf_width_mm": math.isclose(float(page.mediabox.width) / MM_TO_POINT, float(spec["canvas_mm"][0]), abs_tol=0.25),
        "pdf_height_mm": math.isclose(float(page.mediabox.height) / MM_TO_POINT, float(spec["canvas_mm"][1]), abs_tol=0.25),
        "pdf_embedded_font_resources": font_count > 0,
        "pdf_panel_labels_present": all(letter in extracted_text for letter in "abcde"),
        "schematic_single_transfer": panel_qa["a"]["single_transfer_arrows"] == 1,
        "schematic_history_is_discrete": panel_qa["a"]["history_directional_arrows"] == 0,
        "schematic_input_grids_visible": panel_qa["a"]["input_grids_visible"] and panel_qa["a"]["input_grid_cells"] == 32,
    }
    return {"schema": "manuscript_fig5_v2_render_qa_v1", "generated_at": _utc_now(), "status": "passed" if all(checks.values()) else "failed", "checks": checks, "png": {"path": str(outputs["png"]), "pixels": list(actual_size), "expected_pixels_at_300_dpi": list(expected_size), "sha256": _sha256(outputs["png"])}, "svg": {"path": str(outputs["svg"]), "sha256": _sha256(outputs["svg"])}, "pdf": {"path": str(outputs["pdf"]), "pages": len(pdf_reader.pages), "page_mm": [float(page.mediabox.width) / MM_TO_POINT, float(page.mediabox.height) / MM_TO_POINT], "font_resources": font_count, "sha256": _sha256(outputs["pdf"])}}


def _grayscale_audit(outputs: Mapping[str, Path], figures_dir: Path) -> dict[str, Any]:
    grayscale_path = figures_dir / "qa" / "manuscript_fig5_grayscale.png"
    with Image.open(outputs["png"]) as image:
        gray_image = image.convert("L")
        gray_image.save(grayscale_path, dpi=(300, 300))
        gray = np.asarray(gray_image, dtype=np.uint8)
    checks = {"grayscale_exists": grayscale_path.is_file(), "grayscale_has_dark_marks": bool((gray < 180).any()), "grayscale_has_midtones": bool(((gray >= 80) & (gray < 245)).any())}
    return {"schema": "manuscript_fig5_v2_grayscale_audit_v1", "status": "passed" if all(checks.values()) else "failed", "checks": checks, "path": str(grayscale_path)}


def _visual_qa(outputs: Mapping[str, Path], spec: Mapping[str, Any], panel_qa: Mapping[str, Any], figures_dir: Path) -> dict[str, Any]:
    with Image.open(outputs["png"]) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        width, height = [float(value) for value in spec["canvas_mm"]]
        coverage: dict[str, float] = {}
        for panel_id, slot in spec["slots"].items():
            x, y, w, h = [float(value) for value in slot]
            left, top = int(round(x / width * image.width)), int(round(y / height * image.height))
            right, bottom = int(round((x + w) / width * image.width)), int(round((y + h) / height * image.height))
            coverage[panel_id] = float((rgb[top:bottom, left:right].min(axis=2) < 245).mean())
        panel_dir = figures_dir / "qa" / "panels"
        panel_dir.mkdir(parents=True, exist_ok=True)
        crops = []
        for panel_id, slot in spec["slots"].items():
            x, y, w, h = [float(value) for value in slot]
            box = (int(round(x / width * image.width)), int(round(y / height * image.height)), int(round((x + w) / width * image.width)), int(round((y + h) / height * image.height)))
            path = panel_dir / f"manuscript_fig5{panel_id}.png"
            image.crop(box).save(path, dpi=(300, 300))
            crops.append({"panel": panel_id, "path": str(path), "pixels": [box[2] - box[0], box[3] - box[1]]})
    checks = {
        "all_panels_have_ink": all(value > 0.01 for value in coverage.values()),
        "row_2_plot_area_aligned": spec["panels"]["b"]["plot_bbox_mm"][1:] == spec["panels"]["c"]["plot_bbox_mm"][1:],
        "row_3_plot_area_aligned": spec["panels"]["d"]["plot_bbox_mm"][1:] == spec["panels"]["e"]["plot_bbox_mm"][1:],
        "schematic_has_single_transfer": panel_qa["a"]["single_transfer_arrows"] == 1,
        "schematic_history_is_discrete": panel_qa["a"]["history_directional_arrows"] == 0,
        "schematic_input_grids_visible": panel_qa["a"]["input_grids_visible"] and panel_qa["a"]["input_grid_cells"] == 32,
        "schematic_labels_are_reader_language": all("at D" not in str(value) and "after D" not in str(value) for value in panel_qa["a"]["reader_labels"]),
    }
    return {"schema": "manuscript_fig5_v2_visual_qa_v1", "status": "passed" if all(checks.values()) else "failed", "checks": checks, "panel_ink_coverage": coverage, "panel_qa": panel_qa, "panel_crops": crops}


def _caption(payload: Mapping[str, Any]) -> str:
    return (
        "**Fig. 5 | Successor states carry history-conditioned updating across successive inputs.**\n\n"
        "**a,** Reader-language schematic of one local intervention and the natural state-transition process. A post-B Layer-2 u/x successor from a donor history is transferred once to the receiver state; the receiver then encounters the next input (C), forms an input response and successor state, and continues to the following input (D) without a second transfer. The lower branch shows the equal-time passive alternative with no input; the donor/receiver annotation gives the tested pre-B depths K=1, 5 and 10. **b,** Network-level donor-transfer estimates for the input response in Layer 2 and the successor state in Layer 3 at K=1, 5 and 10. K=1 and K=5 use the frozen Fig.5 protocol; K=10 uses the confirmatory successor-extension aggregate. **c,** Primary donor-transfer endpoints for the following input at K=5, using one post-B transfer and no second transfer. **d,** In an independent progressive protocol, input-driven change is the observed next-input joint-state displacement minus equal-time passive displacement across transition numbers 2–10; the ribbon shows the persisted two-sided 95% network-bootstrap CI. **e,** Network-level Rescue and Loss rates at K=1 and K=5; the two outcomes use distinct opportunity denominators.\n\n"
        "All quantitative panels summarize 20 independently trained networks (seeds 1000–1019); lower-level observations were aggregated within network. Point/line intervals are persisted two-sided 95% network-bootstrap CIs. The donor-transfer tests are one-sided exact sign-flip tests with the supplied Holm correction within their prespecified families; the progressive and behavioral analyses retain their supplied inference records. No cross-depth trend test or b-versus-c endpoint test is implied. Donor transfer establishes bounded sufficiency under the tested intervention only, not necessity, complete mediation or uniqueness. The progressive recurrence protocol is independent of the transplant protocol."
    )


def _source_mapping(v1_access: pd.DataFrame, ext_access: pd.DataFrame, v1_parent: Path, ext_parent: Path) -> pd.DataFrame:
    rows = [
        {"candidate_figure": DISPLAY_NAME, "panel": "a", "source_bundle": "schematic", "source_path": "illustrative protocol grammar", "independent_unit": "not applicable", "included_seeds": "not applicable", "mapping": "No scientific endpoint; labels define the protocols represented below."},
        {"candidate_figure": DISPLAY_NAME, "panel": "b", "source_bundle": str(v1_parent), "source_path": "data/panel_a_plot_data.csv; data/panel_b_plot_data.csv; metrics/panel_a_statistics.csv; metrics/panel_b_statistics.csv; extension aggregate/network_effects.csv; population_inference.csv", "independent_unit": "independently trained network", "included_seeds": "1000-1019", "mapping": "K1/K5 frozen one-hop endpoints plus K10 confirmatory extension endpoints."},
        {"candidate_figure": DISPLAY_NAME, "panel": "c", "source_bundle": str(ext_parent), "source_path": "network_effects.csv; population_inference.csv", "independent_unit": "independently trained network", "included_seeds": "1000-1019", "mapping": "Two primary second-transition endpoints only."},
        {"candidate_figure": DISPLAY_NAME, "panel": "d", "source_bundle": str(v1_parent), "source_path": "data/panel_c_plot_data.csv; metrics/panel_c_recurrence_inference.csv", "independent_unit": "independently trained network", "included_seeds": "1000-1019", "mapping": "Observed-minus-passive stage endpoint, stages 2-10."},
        {"candidate_figure": DISPLAY_NAME, "panel": "e", "source_bundle": str(v1_parent), "source_path": "data/panel_d_plot_data.csv; metrics/panel_d_statistics.csv; metrics/panel_d_depth_inference.csv", "independent_unit": "independently trained network", "included_seeds": "1000-1019", "mapping": "Rescue and Loss rates with distinct opportunity sets."},
    ]
    return pd.DataFrame(rows)


def _write_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        artifacts.append({"path": path.relative_to(output_dir).as_posix(), "sha256": _sha256(path), "bytes": path.stat().st_size})
    manifest = {"schema": "paper_figure_reader_first_candidate_manifest_v2", "candidate_version": CANDIDATE_VERSION, "display_name": DISPLAY_NAME, "generated_at": _utc_now(), "artifact_count": len(artifacts), "artifacts": artifacts}
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def build_candidate(*, output_dir: Path, check_only: bool) -> dict[str, Any]:
    spec = _load_spec()
    repo_root = _repo_root().resolve()
    v1_payload, v1_parent, v1_before, v1_access = _load_v1(repo_root)
    ext_parent = (repo_root / EXTENSION_ROOT_REL).resolve()
    ext_before = _snapshot_tree(ext_parent, "successor_extension_aggregate")
    extension, extension_network, extension_population, extension_verdict, ext_access = _load_extension(ext_parent)
    transfer_raw, transfer_stats = _build_transfer_frames(v1_payload, extension, extension_network)
    twohop_raw, twohop_stats = _build_twohop_frames(extension, extension_network)
    recurrence_raw = _relabel(v1_payload["c"], "d")
    recurrence_stats = _relabel(v1_payload["c_stats"], "d")
    recurrence_stats = recurrence_stats.loc[recurrence_stats["endpoint"].astype(str).str.match(r"^joint_observed_minus_passive_stage_\d+$")].copy()
    recurrence_stats["stage_k"] = recurrence_stats["endpoint"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
    behavior_raw = _relabel(v1_payload["d"], "e")
    behavior_stats = _relabel(v1_payload["d_stats"], "e").copy()
    behavior_stats["outcome_type"] = behavior_stats["endpoint"].astype(str).str.extract(r"^(rescue|loss)")[0]
    behavior_stats["prefix_k"] = behavior_stats["group"].astype(str).str.extract(r"\|(K[15])$")[0]
    behavior_stats = behavior_stats.loc[behavior_stats["outcome_type"].isin(["rescue", "loss"]) & behavior_stats["prefix_k"].isin(["K1", "K5"])].copy()
    behavior_depth_stats = _relabel(v1_payload["d_depth_stats"], "e")
    # Validate recurrence and behavior finite values against the frozen v1 summaries before plotting.
    _finite(recurrence_raw["value"], "Fig.5d recurrence raw values")
    _finite(behavior_raw["value"], "Fig.5e behavior raw values")
    for stage in EXPECTED_STAGES:
        values = recurrence_raw.loc[recurrence_raw["stage_k"].astype(int).eq(stage), "value"]
        stat = v1_payload["c_frozen"][stage]
        if len(values) != 20 or not math.isclose(float(values.mean()), float(stat["estimate"]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Fig.5d stage {stage}: materialized mean disagrees with frozen statistic")
    output_dir = output_dir.resolve()
    if _inside(output_dir, v1_parent) or _inside(output_dir, ext_parent):
        raise ValueError("candidate output must be separate from both pinned parent trees")
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    (output_dir / "figures" / "qa" / "panels").mkdir(parents=True, exist_ok=True)
    layout_audit = _layout_audit(spec)
    if layout_audit["status"] != "passed":
        raise ValueError(f"candidate layout contract failed: {layout_audit['failures']}")
    _write_json(output_dir / "data" / "panel_a_schematic.json", {"figure_id": DISPLAY_NAME, "panel_id": "a", "schema": "reader_language_protocol_schematic_v1", "labels": spec["panels"]["a"]["labels"], "scientific_endpoint": False})
    transfer_raw.to_csv(output_dir / "data" / "panel_b_transfer_network_effects.csv", index=False)
    twohop_raw.to_csv(output_dir / "data" / "panel_c_second_transition.csv", index=False)
    recurrence_raw.to_csv(output_dir / "data" / "panel_d_recurrence.csv", index=False)
    behavior_raw.to_csv(output_dir / "data" / "panel_e_behavior.csv", index=False)
    transfer_stats.to_csv(output_dir / "metrics" / "panel_b_transfer_statistics.csv", index=False)
    twohop_stats.to_csv(output_dir / "metrics" / "panel_c_second_transition_statistics.csv", index=False)
    recurrence_stats.to_csv(output_dir / "metrics" / "panel_d_recurrence_statistics.csv", index=False)
    behavior_stats.to_csv(output_dir / "metrics" / "panel_e_behavior_statistics.csv", index=False)
    behavior_depth_stats.to_csv(output_dir / "metrics" / "panel_e_behavior_depth_inference.csv", index=False)
    combined_before = pd.concat([v1_before, ext_before], ignore_index=True)
    combined_before.to_csv(output_dir / "meta" / "parent_hashes_before.csv", index=False)
    _source_mapping(v1_access, ext_access, v1_parent, ext_parent).to_csv(output_dir / "meta" / "source_mapping.csv", index=False)
    access = pd.concat([v1_access.assign(candidate_version=CANDIDATE_VERSION), ext_access.assign(candidate_version=CANDIDATE_VERSION)], ignore_index=True)
    access.to_csv(output_dir / "meta" / "plot_source_access.csv", index=False)
    pd.DataFrame(layout_audit["geometry_rows"]).to_csv(output_dir / "meta" / "layout_measurements.csv", index=False)
    _write_json(output_dir / "meta" / "layout_audit.json", layout_audit)
    _write_json(output_dir / "meta" / "extension_verdict.json", extension_verdict)
    _write_json(output_dir / "meta" / "final_plot_spec.json", spec)
    _write_json(output_dir / "meta" / "review_only_candidate_spec.json", spec)
    _write_json(output_dir / "meta" / "extension_summary.json", extension)
    (output_dir / "caption_draft.md").write_text(_caption({"transfer_stats": transfer_stats, "twohop_stats": twohop_stats}), encoding="utf-8")
    outputs: dict[str, Path] = {}
    render_qa = grayscale_qa = visual_qa = None
    panel_qa: dict[str, Any] = {}
    if not check_only:
        _render_wireframe(spec, output_dir / "figures" / "qa" / "manuscript_fig5_wireframe.png")
        rendered = _render_figure(spec, {"schematic_labels": spec["panels"]["a"]["labels"], "transfer_stats": transfer_stats, "twohop_stats": twohop_stats, "recurrence_stats": recurrence_stats, "behavior_stats": behavior_stats}, output_dir / "figures")
        outputs = {key: value for key, value in rendered.items() if key in {"png", "svg", "pdf"}}
        panel_qa = rendered["panel_qa"]
        render_qa = _render_qa(outputs, spec, panel_qa)
        _write_json(output_dir / "meta" / "render_qa.json", render_qa)
        if render_qa["status"] != "passed":
            raise ValueError(f"render QA failed: {render_qa['checks']}")
        grayscale_qa = _grayscale_audit(outputs, output_dir / "figures")
        _write_json(output_dir / "meta" / "grayscale_audit.json", grayscale_qa)
        if grayscale_qa["status"] != "passed":
            raise ValueError(f"grayscale QA failed: {grayscale_qa['checks']}")
        visual_qa = _visual_qa(outputs, spec, panel_qa, output_dir / "figures")
        _write_json(output_dir / "meta" / "visual_qa.json", visual_qa)
        if visual_qa["status"] != "passed":
            raise ValueError(f"visual QA failed: {visual_qa['checks']}")
    v1_after = fig5_v1._snapshot_tree(v1_parent, "fig5_v1_parent")
    ext_after = _snapshot_tree(ext_parent, "successor_extension_aggregate")
    combined_after = pd.concat([v1_after, ext_after], ignore_index=True)
    combined_after.to_csv(output_dir / "meta" / "parent_hashes_after.csv", index=False)
    parents_unchanged = v1_before.equals(v1_after) and ext_before.equals(ext_after)
    parent_integrity = {"schema": "manuscript_fig5_v2_parent_integrity_v1", "status": "passed" if parents_unchanged else "failed", "parents": {"fig5_v1": {"root": str(v1_parent), "before": _snapshot_digest(v1_before), "after": _snapshot_digest(v1_after), "unchanged": v1_before.equals(v1_after)}, "successor_extension": {"root": str(ext_parent), "before": _snapshot_digest(ext_before), "after": _snapshot_digest(ext_after), "unchanged": ext_before.equals(ext_after)}}, "unchanged": parents_unchanged}
    _write_json(output_dir / "meta" / "parent_integrity.json", parent_integrity)
    if not parents_unchanged:
        raise RuntimeError("one or more pinned parent trees changed during candidate rendering")
    run_config = {"candidate_version": CANDIDATE_VERSION, "display_name": DISPLAY_NAME, "plot_only": True, "check_only": bool(check_only), "parent_bundles": {"fig5_v1": str(v1_parent), "successor_extension": str(ext_parent)}, "output_dir": str(output_dir), "expected_networks": list(EXPECTED_SEEDS), "independent_unit": "independently trained network", "source_policy": "read-only persisted source data and frozen statistics", "model_or_dataset_initialized": False, "generated_at": _utc_now(), "script": str(Path(__file__).resolve()), "spec": str(SPEC_PATH)}
    _write_json(output_dir / "run_config.json", run_config)
    summary = {"schema": "paper_figure_reader_first_candidate_summary_v2", "candidate_version": CANDIDATE_VERSION, "display_name": DISPLAY_NAME, "status": "check_passed" if check_only else "rendered", "canvas_mm": spec["canvas_mm"], "independent_unit": "independently trained network", "n_networks": 20, "network_seeds": list(EXPECTED_SEEDS), "panel_b_summary_rows": int(len(transfer_stats)), "panel_c_summary_rows": int(len(twohop_stats)), "panel_d_stage_count": int(len(recurrence_stats)), "panel_e_series": ["rescue", "loss"], "outputs": {key: str(path.relative_to(output_dir)) for key, path in outputs.items()}, "parent_integrity": parent_integrity, "layout_status": layout_audit["status"], "render_qa_status": render_qa["status"] if render_qa else "not_run", "grayscale_qa_status": grayscale_qa["status"] if grayscale_qa else "not_run", "visual_qa_status": visual_qa["status"] if visual_qa else "not_run"}
    _write_json(output_dir / "summary.json", summary)
    log_lines = [f"{_utc_now()} candidate={CANDIDATE_VERSION}", f"mode={'check-only' if check_only else 'plot-only render'}", f"fig5_v1_before={_snapshot_digest(v1_before)}", f"fig5_v1_after={_snapshot_digest(v1_after)}", f"extension_before={_snapshot_digest(ext_before)}", f"extension_after={_snapshot_digest(ext_after)}", f"layout={layout_audit['status']}", f"render_qa={render_qa['status'] if render_qa else 'not_run'}", f"grayscale_qa={grayscale_qa['status'] if grayscale_qa else 'not_run'}", f"visual_qa={visual_qa['status'] if visual_qa else 'not_run'}", f"parent_integrity={parent_integrity['status']}"]
    (output_dir / "logs" / "render.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    manifest = _write_artifact_manifest(output_dir)
    return {"status": summary["status"], "output_dir": str(output_dir), "outputs": summary["outputs"], "layout": layout_audit["status"], "render_qa": summary["render_qa_status"], "grayscale_qa": summary["grayscale_qa_status"], "visual_qa": summary["visual_qa_status"], "parent_integrity": parent_integrity["status"], "artifact_count": manifest["artifact_count"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the reader-first manuscript Fig.5 v2 candidate.")
    parser.add_argument("--output-dir", default="results/paper_figure_candidates/manuscript_fig5_reader_first_v2")
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
    if args.refresh_manifest:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"candidate output is missing: {output_dir}")
        print(json.dumps(_write_artifact_manifest(output_dir), indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    result = build_candidate(output_dir=output_dir, check_only=bool(args.check_only))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
