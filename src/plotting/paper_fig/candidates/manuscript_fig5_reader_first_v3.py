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
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
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
from src.plotting.paper_fig.candidates import manuscript_fig5_reader_first as fig5_v1


CANDIDATE_VERSION = "manuscript_fig5_reader_first_v3"
DISPLAY_NAME = "Fig.5"
EXPECTED_SEEDS = tuple(range(1000, 1020))
EXPECTED_STAGES = tuple(range(2, 11))
MM_TO_INCH = 1.0 / 25.4
MM_TO_POINT = 72.0 / 25.4
SPEC_PATH = Path(__file__).resolve().parent / "specs" / f"{CANDIDATE_VERSION}.json"
EXTENSION_ROOT_REL = Path("results/successor_extension_v1_confirmatory_20seed/aggregate")
EXTENSION_COHORT_REL = Path("results/successor_extension_v1_confirmatory_20seed")
EXTENSION_FILES = {"network_effects.csv", "population_inference.csv", "verdict.json", "artifact_manifest.json"}
SEED_EXP_B = "data/metrics/exp_b_k10_l1_overlap_intervention/summary.json"
SEED_EXP_C = "data/metrics/exp_c_c5_twohop_cd/summary.json"
INK = NATURE_COMPATIBLE_PALETTE["ink"]
WHITE = NATURE_COMPATIBLE_PALETTE["white"]
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
    source_scope: str = "successor_extension_aggregate"

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
                "source_scope": self.source_scope,
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


def _snapshot_files(root: Path, relative_paths: Sequence[str], source_scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for relative in sorted(set(relative_paths)):
        path = (root / relative).resolve()
        if not _inside(path, root) or not path.is_file():
            raise FileNotFoundError(f"required source is missing: {path}")
        rows.append({"source_scope": source_scope, "path": Path(relative).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return pd.DataFrame(rows, columns=["source_scope", "path", "bytes", "sha256"])


def _load_seed_summaries(repo_root: Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], pd.DataFrame, Path, pd.DataFrame]:
    root = (repo_root / EXTENSION_COHORT_REL).resolve()
    relative_paths = [f"seed_{seed}/{SEED_EXP_B}" for seed in EXPECTED_SEEDS] + [f"seed_{seed}/{SEED_EXP_C}" for seed in EXPECTED_SEEDS]
    reader = BundleReader(root, set(relative_paths), source_scope="successor_extension_seed_summaries")
    overlap: dict[int, dict[str, Any]] = {}
    twohop: dict[int, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        overlap[seed] = reader.read_json(f"seed_{seed}/{SEED_EXP_B}", "per-network overlap intervention summary")
        twohop[seed] = reader.read_json(f"seed_{seed}/{SEED_EXP_C}", "per-network two-hop descriptive gate summary")
        if int(overlap[seed].get("network_seed", seed)) != seed or int(twohop[seed].get("network_seed", seed)) != seed:
            raise ValueError(f"seed summary network identity mismatch for {seed}")
    return overlap, twohop, reader.access_frame(), root, _snapshot_files(root, relative_paths, "successor_extension_seed_summaries")


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
                        "panel_id": "a",
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
                "panel_id": "a",
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
                "panel_id": "a",
                "network_seed": int(row["network_seed"]),
                "history_depth": 10,
                "endpoint": endpoint,
                "value": float(row["value"]),
                "unit": "donor_transfer_index",
                "source": "successor_extension_aggregate",
            })
        stat_rows.append({
            "figure_id": DISPLAY_NAME,
            "panel_id": "a",
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


def _build_overlap_frames(
    seed_overlap: Mapping[int, Mapping[str, Any]],
    extension_population: pd.DataFrame,
    extension_network: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint_specs = {
        "input_response": ("exp_b_k10_l1_overlap_intervention", "early_layer2_b_history_contrast_attenuation"),
        "post_input_state": ("exp_b_k10_l1_overlap_intervention", "post_b_layer2_ux_history_contrast_attenuation"),
    }
    condition_fields = {
        "overlap": "mean_overlap_attenuation",
        "non_overlap": "mean_nonoverlap_attenuation",
        "random": "mean_random_attenuation",
    }
    raw_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    for endpoint, (experiment, technical_endpoint) in endpoint_specs.items():
        values_by_condition: dict[str, list[float]] = {condition: [] for condition in condition_fields}
        overlap_margin_values: list[float] = []
        for seed in EXPECTED_SEEDS:
            summary = seed_overlap[seed]
            endpoint_summary = summary.get("endpoints", {}).get(technical_endpoint)
            if endpoint_summary is None:
                raise ValueError(f"Fig.5b: missing per-seed overlap endpoint {technical_endpoint} for {seed}")
            overlap_value = float(endpoint_summary["mean_overlap_attenuation"])
            overlap_margin = float(endpoint_summary["mean_overlap_specific_margin"])
            if not math.isclose(overlap_value, overlap_margin, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Fig.5b {endpoint}/{seed}: overlap attenuation disagrees with overlap-specific margin")
            overlap_margin_values.append(overlap_margin)
            for condition, field_name in condition_fields.items():
                value = float(endpoint_summary[field_name])
                if not math.isfinite(value):
                    raise ValueError(f"Fig.5b: non-finite overlap value for {endpoint}/{condition}/{seed}")
                values_by_condition[condition].append(value)
                raw_rows.append({
                    "figure_id": DISPLAY_NAME,
                    "panel_id": "b",
                    "network_seed": seed,
                    "endpoint": endpoint,
                    "removed_sites": condition,
                    "value": value,
                    "unit": "history_effect_removed_fraction",
                    "history_depth": 10,
                    "source": "successor_extension_per_seed_summary",
                })
        aggregate_rows = extension_network.loc[
            extension_network["cohort"].astype(str).eq("full20")
            & extension_network["experiment"].astype(str).eq(experiment)
            & extension_network["endpoint"].astype(str).eq(technical_endpoint)
        ].copy()
        if set(aggregate_rows["network_seed"].astype(int)) != set(EXPECTED_SEEDS) or len(aggregate_rows) != len(EXPECTED_SEEDS):
            raise ValueError(f"Fig.5b {endpoint}: aggregate network rows are not a complete 20-network cohort")
        aggregate_values = aggregate_rows.set_index(aggregate_rows["network_seed"].astype(int))["value"].astype(float)
        for seed, value in zip(EXPECTED_SEEDS, overlap_margin_values):
            if not math.isclose(float(aggregate_values.loc[seed]), value, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"Fig.5b {endpoint}/{seed}: aggregate network value disagrees with per-network overlap margin")
        population_row = _find_population_row(extension_population, experiment, technical_endpoint)
        population_stat = _validate_stat_triplet(population_row, f"Fig.5b {endpoint}/overlap")
        if int(population_row["n_networks"]) != len(EXPECTED_SEEDS):
            raise ValueError(f"Fig.5b {endpoint}/overlap: aggregate inference is not a 20-network cohort")
        if not math.isclose(float(population_stat["estimate"]), float(np.mean(overlap_margin_values)), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Fig.5b {endpoint}/overlap: aggregate mean disagrees with complete per-network margins")
        for condition, values in values_by_condition.items():
            if condition == "overlap":
                stat = population_stat
                source = "successor_extension_aggregate_and_complete_per_network_overlap_margins"
                interval_source = "confirmatory_supplied_network_bootstrap"
            else:
                stat = dict(zip(("estimate", "ci95_low", "ci95_high"), _bootstrap_mean_ci(values)))
                if not all(math.isclose(float(stat[key]), 0.0, abs_tol=1e-12) for key in ("estimate", "ci95_low", "ci95_high")):
                    raise ValueError(f"Fig.5b {endpoint}/{condition}: candidate bootstrap control interval is not exactly zero")
                source = "complete_persisted_per_network_summary"
                interval_source = "candidate_network_bootstrap_20000"
            stat_rows.append({
                "figure_id": DISPLAY_NAME,
                "panel_id": "b",
                "endpoint": endpoint,
                "removed_sites": condition,
                "history_depth": 10,
                **stat,
                "n_networks": len(EXPECTED_SEEDS),
                "positive_network_fraction": float(np.mean(np.asarray(values) > 0.0)),
                "source": source,
                "interval_source": interval_source,
                "bootstrap_draws": 20000 if interval_source == "candidate_network_bootstrap_20000" else None,
                "inference_role": "confirmatory_endpoint" if condition == "overlap" else "control_summary",
            })
    raw = pd.DataFrame(raw_rows).sort_values(["endpoint", "removed_sites", "network_seed"]).reset_index(drop=True)
    stats = pd.DataFrame(stat_rows).sort_values(["endpoint", "removed_sites"]).reset_index(drop=True)
    if len(raw) != 120 or len(stats) != 6:
        raise ValueError(f"Fig.5b overlap materialization expected 120 raw and 6 summary rows, got {len(raw)}/{len(stats)}")
    return raw, stats


def _bootstrap_mean_ci(values: Sequence[float], *, seed: int = 20260726, draws: int = 20000) -> tuple[float, float, float]:
    array = _finite(values, "bootstrap values")
    if len(array) != len(EXPECTED_SEEDS):
        raise ValueError(f"bootstrap requires exactly {len(EXPECTED_SEEDS)} network values")
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return float(array.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _build_gate_frame(seed_twohop: Mapping[int, Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    values: list[float] = []
    raw_rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        summary = seed_twohop[seed]
        value = float(summary.get("gate_mean_early_layer2_C_donor_transfer"))
        if not math.isfinite(value):
            raise ValueError(f"Fig.5c: non-finite next-response gate for {seed}")
        values.append(value)
        raw_rows.append({
            "figure_id": DISPLAY_NAME,
            "panel_id": "c",
            "network_seed": seed,
            "endpoint": "next_response",
            "value": value,
            "history_depth": 5,
            "unit": "donor_transfer_index",
            "source": "successor_extension_per_seed_summary",
        })
    estimate, low, high = _bootstrap_mean_ci(values)
    stats = pd.DataFrame([{
        "figure_id": DISPLAY_NAME,
        "panel_id": "c",
        "endpoint": "next_response",
        "history_depth": 5,
        "estimate": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "n_networks": len(EXPECTED_SEEDS),
        "positive_network_fraction": float(np.mean(np.asarray(values) > 0.0)),
        "source": "complete_persisted_per_network_summary",
        "interval_source": "candidate_network_bootstrap_20000",
        "bootstrap_draws": 20000,
        "inference_role": "descriptive_gate",
    }])
    return pd.DataFrame(raw_rows), stats


def _build_twohop_frames(extension: Mapping[str, dict[str, Any]], extension_network: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    endpoint_map = {"input_response": "following_response", "successor_state": "new_successor"}
    for internal_endpoint, endpoint in endpoint_map.items():
        summary = extension[internal_endpoint]
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
            "interval_source": "confirmatory_supplied_network_bootstrap",
            "bootstrap_draws": 20000,
            "inference_role": "confirmatory_endpoint",
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


def _draw_transfer(axis: plt.Axes, transfer_stats: pd.DataFrame, panel_spec: Mapping[str, Any]) -> dict[str, Any]:
    colors = {key: get_plot_color(value, context="manuscript_fig5") for key, value in panel_spec["endpoint_colors"].items()}
    labels = panel_spec["endpoint_labels"]
    history_depths = [1, 5, 10]
    x = np.arange(len(history_depths), dtype=float)
    offsets = np.linspace(-0.18, 0.18, len(panel_spec["endpoint_order"]))
    width = 0.32
    for offset, endpoint in zip(offsets, panel_spec["endpoint_order"]):
        subset = transfer_stats.loc[transfer_stats["endpoint"].eq(endpoint)].sort_values("history_depth")
        if subset["history_depth"].astype(int).tolist() != history_depths:
            raise ValueError(f"Fig.5a: expected K=1,5,10 for {endpoint}")
        means = subset["estimate"].to_numpy(dtype=float)
        low = subset["ci95_low"].to_numpy(dtype=float)
        high = subset["ci95_high"].to_numpy(dtype=float)
        color = colors[endpoint]
        axis.bar(x + offset, means, width=width, color=color, edgecolor=INK, linewidth=0.45, label=labels[endpoint], zorder=2)
        axis.errorbar(x + offset, means, yerr=np.vstack([means - low, high - means]), fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.0, capthick=0.7, zorder=4)
    axis.axhline(0.0, color=NEUTRAL_LIGHT, linewidth=0.7, zorder=1)
    axis.set_xlim(-0.55, 2.55)
    axis.set_xticks(x)
    axis.set_xticklabels([str(value) for value in history_depths])
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.yaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=2, handlelength=1.4, columnspacing=1.0, borderaxespad=0.0)
    return {"endpoints": list(panel_spec["endpoint_order"]), "history_depths": history_depths, "bars": 6}


def _draw_overlap(axis: plt.Axes, overlap_stats: pd.DataFrame, panel_spec: Mapping[str, Any]) -> dict[str, Any]:
    colors = {key: get_plot_color(value, context="manuscript_fig5") for key, value in panel_spec["condition_colors"].items()}
    endpoint_order = list(panel_spec["endpoint_order"])
    condition_order = list(panel_spec["condition_order"])
    x = np.arange(len(endpoint_order), dtype=float)
    offsets = np.linspace(-0.25, 0.25, len(condition_order))
    width = 0.22
    zero_caps: list[dict[str, Any]] = []
    for offset, condition in zip(offsets, condition_order):
        rows = overlap_stats.loc[overlap_stats["removed_sites"].eq(condition)].set_index("endpoint").reindex(endpoint_order)
        if rows["estimate"].isna().any():
            raise ValueError(f"Fig.5b: missing overlap condition {condition}")
        means = rows["estimate"].to_numpy(dtype=float) * 100.0
        low = rows["ci95_low"].to_numpy(dtype=float) * 100.0
        high = rows["ci95_high"].to_numpy(dtype=float) * 100.0
        color = colors[condition]
        positions = x + offset
        axis.bar(positions, means, width=width, color=color, edgecolor=INK, linewidth=0.45, label=panel_spec["condition_labels"][condition], zorder=2)
        axis.errorbar(positions, means, yerr=np.vstack([means - low, high - means]), fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.0, capthick=0.7, zorder=4)
        for endpoint_index, mean in enumerate(means):
            if math.isclose(float(mean), 0.0, abs_tol=1e-12):
                # The cap is the true top edge of a zero-height bar, not a marker
                # or a fabricated positive bar height.
                left = float(positions[endpoint_index] - width / 2.0)
                right = float(positions[endpoint_index] + width / 2.0)
                axis.plot([left, right], [0.0, 0.0], color=color, linewidth=1.25, solid_capstyle="butt", zorder=5)
                zero_caps.append({"condition": condition, "endpoint": endpoint_order[endpoint_index], "color": color, "width": float(width)})
    axis.axhline(0.0, color=NEUTRAL_LIGHT, linewidth=0.7, zorder=1)
    axis.set_xlim(-0.55, 1.55)
    axis.set_xticks(x)
    axis.set_xticklabels([panel_spec["endpoint_labels"][endpoint] for endpoint in endpoint_order])
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    if panel_spec.get("x_label"):
        axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.yaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    legend_handles = [
        Patch(facecolor=colors[condition], edgecolor=INK, linewidth=0.45, label=panel_spec["condition_labels"][condition])
        for condition in condition_order
    ]
    axis.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.05), frameon=False, ncol=3, handlelength=1.2, columnspacing=0.8, borderaxespad=0.0)
    return {
        "endpoints": endpoint_order,
        "conditions": condition_order,
        "bars": 6,
        "zero_caps": zero_caps,
        "legend_handle_types": [type(handle).__name__ for handle in legend_handles],
    }


def _draw_propagation(axis: plt.Axes, stats: pd.DataFrame, panel_spec: Mapping[str, Any]) -> dict[str, Any]:
    endpoint_order = list(panel_spec["endpoint_order"])
    colors = {key: get_plot_color(value, context="manuscript_fig5") for key, value in panel_spec["endpoint_colors"].items()}
    rows = stats.set_index("endpoint").reindex(endpoint_order)
    if rows["estimate"].isna().any():
        raise ValueError("Fig.5c: missing propagation endpoint")
    y = np.arange(len(endpoint_order), dtype=float)
    means = rows["estimate"].to_numpy(dtype=float)
    low = rows["ci95_low"].to_numpy(dtype=float)
    high = rows["ci95_high"].to_numpy(dtype=float)
    filled = dict(panel_spec["endpoint_filled"])
    for index, endpoint in enumerate(endpoint_order):
        if filled[endpoint]:
            axis.barh(index, means[index], height=0.48, color=colors[endpoint], edgecolor=INK, linewidth=0.45, zorder=2)
        else:
            axis.barh(index, means[index], height=0.48, facecolor="none", edgecolor=colors[endpoint], linewidth=1.0, zorder=2)
        axis.errorbar(means[index], index, xerr=[[means[index] - low[index]], [high[index] - means[index]]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.0, capthick=0.7, zorder=4)
    axis.axvline(0.0, color=NEUTRAL_LIGHT, linewidth=0.7, zorder=1)
    axis.set_xlim(*[float(value) for value in panel_spec["x_limits"]])
    y_axis_labels = [str(panel_spec["endpoint_labels"][endpoint]) for endpoint in endpoint_order]
    axis.set_yticks(y)
    axis.set_yticklabels(y_axis_labels)
    axis.invert_yaxis()
    axis.set_ylim(len(endpoint_order) - 0.5, -0.5)
    axis.set_xticks([float(value) for value in panel_spec["x_ticks"]])
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.xaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    return {
        "endpoints": endpoint_order,
        "bars": len(endpoint_order),
        "endpoint_filled": filled,
        "descriptive_gate": "next_response",
        "y_axis_labels": y_axis_labels,
    }


def _draw_recurrence(axis: plt.Axes, stats: pd.DataFrame, panel_spec: Mapping[str, Any]) -> dict[str, Any]:
    stages = [int(value) for value in panel_spec["x_order"]]
    colors = {key: get_plot_color(value, context="manuscript_fig5") for key, value in panel_spec["condition_colors"].items()}
    rendered: dict[str, Any] = {}
    for condition in panel_spec["condition_order"]:
        subset = stats.loc[stats["condition"].eq(condition)].sort_values("stage_k")
        if subset["stage_k"].astype(int).tolist() != stages:
            raise ValueError(f"Fig.5d: stage order is not 2-10 for {condition}")
        means = subset["estimate"].to_numpy(dtype=float)
        low = subset["ci95_low"].to_numpy(dtype=float)
        high = subset["ci95_high"].to_numpy(dtype=float)
        color = colors[condition]
        linestyle = panel_spec["condition_linestyles"][condition]
        marker = panel_spec["condition_markers"][condition]
        axis.fill_between(stages, low, high, color=color, alpha=0.12 if condition == "observed" else 0.08, linewidth=0, zorder=1)
        axis.plot(stages, means, color=color, linewidth=1.25, linestyle=linestyle, marker=marker, markersize=3.6, markerfacecolor=color if condition == "observed" else WHITE, markeredgecolor=color, markeredgewidth=0.65, label=panel_spec["condition_labels"][condition], zorder=3)
        rendered[condition] = {"means": means.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist()}
    axis.set_xlim(*[float(value) for value in panel_spec["x_limits"]])
    axis.set_xticks([float(value) for value in panel_spec["x_ticks"]])
    axis.set_ylim(*[float(value) for value in panel_spec["y_limits"]])
    axis.set_yticks([float(value) for value in panel_spec["y_ticks"]])
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.yaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    if len(panel_spec["condition_order"]) > 1:
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=2, handlelength=1.4, columnspacing=0.9, borderaxespad=0.0)
    return {
        "stages": stages,
        "conditions": list(panel_spec["condition_order"]),
        "legend_rendered": axis.get_legend() is not None,
        "rendered": rendered,
    }


def _draw_behavior(axis: plt.Axes, stats: pd.DataFrame, panel_spec: Mapping[str, Any]) -> dict[str, Any]:
    colors = {key: get_plot_color(value, context="manuscript_fig5") for key, value in panel_spec["series_colors"].items()}
    row_positions = {series: float(index) for index, series in enumerate(panel_spec["series"])}
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
        y = row_positions[series]
        axis.annotate("", xy=(means[1], y), xytext=(means[0], y), arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.4, "shrinkA": 4, "shrinkB": 4}, zorder=2)
        axis.errorbar(means[0], y, xerr=[[means[0] - low[0]], [high[0] - means[0]]], fmt="o", color=color, markerfacecolor=WHITE, markeredgecolor=color, markeredgewidth=0.9, markersize=5.0, ecolor=INK, elinewidth=0.8, capsize=2.0, capthick=0.7, zorder=4)
        axis.errorbar(means[1], y, xerr=[[means[1] - low[1]], [high[1] - means[1]]], fmt="o", color=color, markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.55, markersize=5.0, ecolor=INK, elinewidth=0.8, capsize=2.0, capthick=0.7, zorder=4)
        rendered[series] = {"mean": means.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist()}
    axis.set_xlim(*[float(value) for value in panel_spec["x_limits"]])
    axis.set_ylim(-0.45, len(panel_spec["series"]) - 0.55)
    axis.invert_yaxis()
    axis.set_yticks([row_positions[series] for series in panel_spec["series"]])
    axis.set_yticklabels([panel_spec["series_labels"][series] for series in panel_spec["series"]])
    axis.set_xticks([float(value) for value in panel_spec["x_ticks"]])
    axis.set_xlabel(str(panel_spec["x_label"]), labelpad=3.0)
    axis.set_ylabel(str(panel_spec["y_label"]), labelpad=3.0)
    axis.xaxis.set_major_formatter(FuncFormatter(_numeric_tick))
    _style_axis(axis)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=WHITE, markeredgecolor=INK, markersize=5.0, label="K=1"), Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=INK, markeredgecolor=INK, markersize=5.0, label="K=5")]
    axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=2, handlelength=1.0, columnspacing=0.9, borderaxespad=0.0)
    return {
        "series_order": list(panel_spec["series"]),
        "top_to_bottom": list(panel_spec["series"]),
        "y_inverted": True,
        "rendered": rendered,
    }


def _build_recurrence_frames(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = raw.copy()
    required = {"network_seed", "stage_k", "condition", "value", "summary_mean", "summary_ci95_low", "summary_ci95_high"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Fig.5d raw recurrence data missing {missing}")
    data["network_seed"] = pd.to_numeric(data["network_seed"], errors="raise").astype(int)
    data["stage_k"] = pd.to_numeric(data["stage_k"], errors="raise").astype(int)
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    data["condition"] = data["condition"].astype(str)
    if set(data["condition"]) != {"observed", "passive"} or len(data) != 360:
        raise ValueError("Fig.5d: expected 360 observed/passive network-stage rows")
    stat_rows: list[dict[str, Any]] = []
    for condition in ("observed", "passive"):
        for stage in EXPECTED_STAGES:
            rows = data.loc[data["condition"].eq(condition) & data["stage_k"].eq(stage)]
            if len(rows) != len(EXPECTED_SEEDS) or set(rows["network_seed"]) != set(EXPECTED_SEEDS):
                raise ValueError(f"Fig.5d: incomplete {condition} stage {stage}")
            values = rows["value"].to_numpy(dtype=float)
            estimate = float(values.mean())
            summary_mean = float(rows["summary_mean"].iloc[0])
            low = float(rows["summary_ci95_low"].iloc[0])
            high = float(rows["summary_ci95_high"].iloc[0])
            if not math.isclose(estimate, summary_mean, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Fig.5d: {condition} stage {stage} mean disagrees with persisted summary")
            if not low <= estimate <= high:
                raise ValueError(f"Fig.5d: invalid persisted CI for {condition} stage {stage}")
            stat_rows.append({
                "figure_id": DISPLAY_NAME,
                "panel_id": "d",
                "condition": condition,
                "stage_k": stage,
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "n_networks": len(EXPECTED_SEEDS),
                "source": "fig5_v1_frozen",
            })
    data["figure_id"] = DISPLAY_NAME
    data["panel_id"] = "d"
    return data.sort_values(["condition", "stage_k", "network_seed"]).reset_index(drop=True), pd.DataFrame(stat_rows)


def _layout_audit(spec: Mapping[str, Any]) -> dict[str, Any]:
    report = validate_layout_contract(spec)
    failures = list(report.failures)
    expected_slots = {
        "a": [2.0, 2.0, 79.5, 48.0],
        "b": [83.5, 2.0, 79.5, 48.0],
        "c": [2.0, 52.0, 52.333, 48.0],
        "d": [56.333, 52.0, 52.334, 48.0],
        "e": [110.667, 52.0, 52.333, 48.0],
    }
    expected_plots = {
        "a": [14.0, 13.0, 65.5, 30.0],
        "b": [95.5, 13.0, 65.5, 30.0],
        "c": [17.0, 62.0, 34.333, 28.0],
        "d": [69.333, 62.0, 36.334, 28.0],
        "e": [123.667, 62.0, 36.333, 28.0],
    }
    rows: list[dict[str, Any]] = []
    for panel_id, expected in expected_slots.items():
        actual = [float(value) for value in spec["slots"].get(panel_id, [])]
        if actual != expected:
            failures.append(f"panel {panel_id} slot differs from requested 2+3 geometry")
        plot = [float(value) for value in spec["panels"][panel_id]["plot_bbox_mm"]]
        if len(plot) != 4 or len(actual) != 4:
            failures.append(f"panel {panel_id} must declare four coordinates")
            continue
        if plot != expected_plots[panel_id]:
            failures.append(f"panel {panel_id} plot bbox differs from frozen v3 geometry: expected {expected_plots[panel_id]}")
        inside = plot[0] >= actual[0] and plot[1] >= actual[1] and plot[0] + plot[2] <= actual[0] + actual[2] and plot[1] + plot[3] <= actual[1] + actual[3]
        if not inside:
            failures.append(f"panel {panel_id} plot area escapes slot")
        rows.append({"panel_id": panel_id, "slot_bbox_mm": actual, "plot_bbox_mm": plot, "plot_inside_slot": inside, "plot_geometry_exact": plot == expected_plots[panel_id]})
    if [float(value) for value in spec.get("canvas_mm", [])] != [165.0, 102.0]:
        failures.append("canvas differs from 165 x 102 mm")
    return {"schema": "manuscript_fig5_v3_layout_audit_v2", "status": "passed" if not failures else "failed", "passes": report.passes, "warnings": report.warnings, "failures": failures, "geometry_rows": rows}


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
        panel_qa["a"] = _draw_transfer(figure.add_axes(_as_axes_bbox(spec["panels"]["a"]["plot_bbox_mm"], canvas_mm)), payload["transfer_stats"], spec["panels"]["a"])
        panel_qa["b"] = _draw_overlap(figure.add_axes(_as_axes_bbox(spec["panels"]["b"]["plot_bbox_mm"], canvas_mm)), payload["overlap_stats"], spec["panels"]["b"])
        panel_qa["c"] = _draw_propagation(figure.add_axes(_as_axes_bbox(spec["panels"]["c"]["plot_bbox_mm"], canvas_mm)), payload["propagation_stats"], spec["panels"]["c"])
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
        "five_quantitative_panels": set(panel_qa) == set("abcde"),
        "panel_a_bars": panel_qa["a"]["bars"] == 6,
        "panel_b_bars": panel_qa["b"]["bars"] == 6,
        "panel_b_zero_controls_visible": {item["condition"] for item in panel_qa["b"]["zero_caps"]} == {"non_overlap", "random"} and len(panel_qa["b"]["zero_caps"]) == 4,
        "panel_b_legend_uses_colored_patches": panel_qa["b"]["legend_handle_types"] == ["Patch", "Patch", "Patch"],
        "panel_c_bars": panel_qa["c"]["bars"] == 3,
        "panel_c_gate_open": panel_qa["c"]["endpoint_filled"]["next_response"] is False,
        "panel_c_y_axis_labels": panel_qa["c"]["y_axis_labels"] == ["Next\nresponse", "Following\nresponse", "New\nsuccessor"],
        "panel_d_observed_only": panel_qa["d"]["conditions"] == ["observed"] and panel_qa["d"]["legend_rendered"] is False,
        "panel_e_series": panel_qa["e"]["top_to_bottom"] == ["rescue", "loss"] and panel_qa["e"]["y_inverted"] is True,
    }
    return {"schema": "manuscript_fig5_v3_render_qa_v2", "generated_at": _utc_now(), "status": "passed" if all(checks.values()) else "failed", "checks": checks, "png": {"path": str(outputs["png"]), "pixels": list(actual_size), "expected_pixels_at_300_dpi": list(expected_size), "sha256": _sha256(outputs["png"])}, "svg": {"path": str(outputs["svg"]), "sha256": _sha256(outputs["svg"])}, "pdf": {"path": str(outputs["pdf"]), "pages": len(pdf_reader.pages), "page_mm": [float(page.mediabox.width) / MM_TO_POINT, float(page.mediabox.height) / MM_TO_POINT], "font_resources": font_count, "sha256": _sha256(outputs["pdf"])}}


def _grayscale_audit(outputs: Mapping[str, Path], figures_dir: Path) -> dict[str, Any]:
    grayscale_path = figures_dir / "qa" / "manuscript_fig5_grayscale.png"
    with Image.open(outputs["png"]) as image:
        gray_image = image.convert("L")
        gray_image.save(grayscale_path, dpi=(300, 300))
        gray = np.asarray(gray_image, dtype=np.uint8)
    checks = {"grayscale_exists": grayscale_path.is_file(), "grayscale_has_dark_marks": bool((gray < 180).any()), "grayscale_has_midtones": bool(((gray >= 80) & (gray < 245)).any())}
    return {"schema": "manuscript_fig5_v3_grayscale_audit_v1", "status": "passed" if all(checks.values()) else "failed", "checks": checks, "path": str(grayscale_path)}


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
    expected_bottom_plots = {
        "c": [17.0, 62.0, 34.333, 28.0],
        "d": [69.333, 62.0, 36.334, 28.0],
        "e": [123.667, 62.0, 36.333, 28.0],
    }
    checks = {
        "all_panels_have_ink": all(value > 0.01 for value in coverage.values()),
        "row_1_plot_area_aligned": spec["panels"]["a"]["plot_bbox_mm"][1:] == spec["panels"]["b"]["plot_bbox_mm"][1:],
        "row_2_plot_top_bottom_aligned": all(
            abs(float(spec["panels"]["c"]["plot_bbox_mm"][index]) - float(spec["panels"]["d"]["plot_bbox_mm"][index])) <= 1.1e-3
            and abs(float(spec["panels"]["d"]["plot_bbox_mm"][index]) - float(spec["panels"]["e"]["plot_bbox_mm"][index])) <= 1.1e-3
            for index in (1, 3)
        ),
        "row_2_plot_geometry_exact": all(spec["panels"][panel_id]["plot_bbox_mm"] == expected for panel_id, expected in expected_bottom_plots.items()),
        "five_quantitative_panels": set(panel_qa) == set("abcde"),
        "panel_a_bars": panel_qa["a"]["bars"] == 6,
        "panel_b_bars": panel_qa["b"]["bars"] == 6,
        "panel_b_zero_controls_visible": {item["condition"] for item in panel_qa["b"]["zero_caps"]} == {"non_overlap", "random"} and len(panel_qa["b"]["zero_caps"]) == 4,
        "panel_b_legend_uses_colored_patches": panel_qa["b"]["legend_handle_types"] == ["Patch", "Patch", "Patch"],
        "panel_c_bars": panel_qa["c"]["bars"] == 3,
        "panel_c_gate_open": panel_qa["c"]["descriptive_gate"] == "next_response" and panel_qa["c"]["endpoint_filled"] == {"next_response": False, "following_response": True, "new_successor": True},
        "panel_c_y_axis_labels": panel_qa["c"]["y_axis_labels"] == ["Next\nresponse", "Following\nresponse", "New\nsuccessor"],
        "panel_d_observed_only": panel_qa["d"]["conditions"] == ["observed"] and panel_qa["d"]["legend_rendered"] is False,
        "panel_e_series": panel_qa["e"]["top_to_bottom"] == ["rescue", "loss"] and panel_qa["e"]["y_inverted"] is True,
    }
    return {"schema": "manuscript_fig5_v3_visual_qa_v2", "status": "passed" if all(checks.values()) else "failed", "checks": checks, "panel_ink_coverage": coverage, "panel_qa": panel_qa, "panel_crops": crops}


def _caption(payload: Mapping[str, Any]) -> str:
    return (
        "**Fig. 5 | Successor states carry history-conditioned updating across successive inputs.**\n\n"
        "**a,** Mean donor-transfer index for the Layer-2 input response and Layer-3 successor state at history depths K=1, 5 and 10. **b,** Mean history-effect attenuation after removing overlap, non-overlap or random sites at history depth K=10 for the input response and post-input state. **c,** Mean donor-transfer index across the next response, following response and new successor at history depth K=5. **d,** Mean input-associated centered-cosine displacement across transition stages 2–10 in the progressive protocol. **e,** Mean behavioral Rescue and Loss rates at K=1 and K=5; the outcomes use distinct opportunity denominators.\n\n"
        "All quantitative panels summarize 20 independently trained networks (seeds 1000–1019); lower-level observations were aggregated within network. Bars, trajectories and arrows show means with two-sided 95% network-bootstrap CIs. Confirmatory endpoints retain their supplied intervals. The panel-c next-response descriptive gate and the panel-b zero-valued non-overlap and random controls use candidate-derived 20,000-draw network bootstrap intervals from complete persisted per-network summaries. The panel-b overlap endpoint and panel-c following-response and new-successor endpoints use supplied confirmatory aggregate intervals. Donor-transfer tests use one-sided exact sign-flip tests with the supplied Holm correction within prespecified families; no cross-depth trend test or b-versus-c endpoint test is implied. Donor transfer establishes bounded sufficiency under the tested intervention only, not necessity, complete mediation or uniqueness. The progressive recurrence protocol is independent of the transplant protocol."
    )


def _source_mapping(v1_access: pd.DataFrame, ext_access: pd.DataFrame, seed_access: pd.DataFrame, v1_parent: Path, ext_parent: Path, seed_parent: Path) -> pd.DataFrame:
    rows = [
        {"candidate_figure": DISPLAY_NAME, "panel": "a", "source_bundle": str(v1_parent), "source_path": "data/panel_a_plot_data.csv; data/panel_b_plot_data.csv; metrics/panel_a_statistics.csv; metrics/panel_b_statistics.csv; successor-extension aggregate K10 endpoints", "independent_unit": "independently trained network", "included_seeds": "1000-1019", "mapping": "Layer-2 input response and Layer-3 successor transfer at K1/K5/K10."},
        {"candidate_figure": DISPLAY_NAME, "panel": "b", "source_bundle": str(seed_parent), "source_path": "seed_1000..1019/data/metrics/exp_b_k10_l1_overlap_intervention/summary.json plus aggregate population inference", "independent_unit": "independently trained network", "included_seeds": "1000-1019", "mapping": "Overlap, non-overlap and random removal attenuation for two endpoints."},
        {"candidate_figure": DISPLAY_NAME, "panel": "c", "source_bundle": str(seed_parent), "source_path": "seed_1000..1019 exp_c descriptive gate summaries plus aggregate network_effects.csv/population_inference.csv", "independent_unit": "independently trained network", "included_seeds": "1000-1019", "mapping": "Next-response gate followed by following-response and new-successor primary endpoints."},
        {"candidate_figure": DISPLAY_NAME, "panel": "d", "source_bundle": str(v1_parent), "source_path": "data/panel_c_plot_data.csv", "independent_unit": "independently trained network", "included_seeds": "1000-1019", "mapping": "Input-associated state displacement across stages 2-10; passive values remain persisted as an analytic-zero provenance reference and are not drawn."},
        {"candidate_figure": DISPLAY_NAME, "panel": "e", "source_bundle": str(v1_parent), "source_path": "data/panel_d_plot_data.csv; metrics/panel_d_statistics.csv; metrics/panel_d_depth_inference.csv", "independent_unit": "independently trained network", "included_seeds": "1000-1019", "mapping": "Rescue and Loss rates with distinct opportunity sets."},
    ]
    return pd.DataFrame(rows)


def _write_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        artifacts.append({"path": path.relative_to(output_dir).as_posix(), "sha256": _sha256(path), "bytes": path.stat().st_size})
    manifest = {"schema": "paper_figure_reader_first_candidate_manifest_v3", "candidate_version": CANDIDATE_VERSION, "display_name": DISPLAY_NAME, "generated_at": _utc_now(), "artifact_count": len(artifacts), "artifacts": artifacts}
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def build_candidate(*, output_dir: Path, check_only: bool) -> dict[str, Any]:
    spec = _load_spec()
    repo_root = _repo_root().resolve()
    v1_payload, v1_parent, v1_before, v1_access = _load_v1(repo_root)
    ext_parent = (repo_root / EXTENSION_ROOT_REL).resolve()
    ext_before = _snapshot_tree(ext_parent, "successor_extension_aggregate")
    extension, extension_network, extension_population, extension_verdict, ext_access = _load_extension(ext_parent)
    seed_overlap, seed_twohop, seed_access, seed_parent, seed_before = _load_seed_summaries(repo_root)
    transfer_raw, transfer_stats = _build_transfer_frames(v1_payload, extension, extension_network)
    overlap_raw, overlap_stats = _build_overlap_frames(seed_overlap, extension_population, extension_network)
    twohop_raw, twohop_stats = _build_twohop_frames(extension, extension_network)
    gate_raw, gate_stats = _build_gate_frame(seed_twohop)
    propagation_raw = pd.concat([gate_raw, twohop_raw], ignore_index=True)
    propagation_stats = pd.concat([gate_stats, twohop_stats], ignore_index=True)
    propagation_stats = propagation_stats.sort_values("endpoint").reset_index(drop=True)
    recurrence_raw, recurrence_stats = _build_recurrence_frames(v1_payload["raw_c"])
    behavior_raw = _relabel(v1_payload["d"], "e")
    behavior_stats = _relabel(v1_payload["d_stats"], "e").copy()
    behavior_stats["outcome_type"] = behavior_stats["endpoint"].astype(str).str.extract(r"^(rescue|loss)")[0]
    behavior_stats["prefix_k"] = behavior_stats["group"].astype(str).str.extract(r"\|(K[15])$")[0]
    behavior_stats = behavior_stats.loc[behavior_stats["outcome_type"].isin(["rescue", "loss"]) & behavior_stats["prefix_k"].isin(["K1", "K5"])].copy()
    behavior_depth_stats = _relabel(v1_payload["d_depth_stats"], "e")
    # Validate finite persisted values before plotting.
    _finite(recurrence_raw["value"], "Fig.5d recurrence raw values")
    _finite(behavior_raw["value"], "Fig.5e behavior raw values")
    if len(recurrence_stats) != 18 or len(overlap_stats) != 6 or len(propagation_stats) != 3:
        raise ValueError("Fig.5 v3: incomplete quantitative panel statistics")
    output_dir = output_dir.resolve()
    if _inside(output_dir, v1_parent) or _inside(output_dir, ext_parent) or _inside(output_dir, seed_parent):
        raise ValueError("candidate output must be separate from all pinned parent trees")
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    (output_dir / "figures" / "qa" / "panels").mkdir(parents=True, exist_ok=True)
    layout_audit = _layout_audit(spec)
    if layout_audit["status"] != "passed":
        raise ValueError(f"candidate layout contract failed: {layout_audit['failures']}")
    transfer_raw.to_csv(output_dir / "data" / "panel_a_transfer_network_effects.csv", index=False)
    overlap_raw.to_csv(output_dir / "data" / "panel_b_overlap_network_effects.csv", index=False)
    propagation_raw.to_csv(output_dir / "data" / "panel_c_propagation_network_effects.csv", index=False)
    recurrence_raw.to_csv(output_dir / "data" / "panel_d_recurrence.csv", index=False)
    behavior_raw.to_csv(output_dir / "data" / "panel_e_behavior.csv", index=False)
    transfer_stats.to_csv(output_dir / "metrics" / "panel_a_transfer_statistics.csv", index=False)
    overlap_stats.to_csv(output_dir / "metrics" / "panel_b_overlap_statistics.csv", index=False)
    propagation_stats.to_csv(output_dir / "metrics" / "panel_c_propagation_statistics.csv", index=False)
    recurrence_stats.to_csv(output_dir / "metrics" / "panel_d_recurrence_statistics.csv", index=False)
    behavior_stats.to_csv(output_dir / "metrics" / "panel_e_behavior_statistics.csv", index=False)
    behavior_depth_stats.to_csv(output_dir / "metrics" / "panel_e_behavior_depth_inference.csv", index=False)
    combined_before = pd.concat([v1_before, ext_before, seed_before], ignore_index=True)
    combined_before.to_csv(output_dir / "meta" / "parent_hashes_before.csv", index=False)
    _source_mapping(v1_access, ext_access, seed_access, v1_parent, ext_parent, seed_parent).to_csv(output_dir / "meta" / "source_mapping.csv", index=False)
    access = pd.concat([v1_access.assign(candidate_version=CANDIDATE_VERSION), ext_access.assign(candidate_version=CANDIDATE_VERSION), seed_access.assign(candidate_version=CANDIDATE_VERSION)], ignore_index=True)
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
        rendered = _render_figure(spec, {"transfer_stats": transfer_stats, "overlap_stats": overlap_stats, "propagation_stats": propagation_stats, "recurrence_stats": recurrence_stats, "behavior_stats": behavior_stats}, output_dir / "figures")
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
    seed_after = _snapshot_files(seed_parent, seed_before["path"].astype(str).tolist(), "successor_extension_seed_summaries")
    combined_after = pd.concat([v1_after, ext_after, seed_after], ignore_index=True)
    combined_after.to_csv(output_dir / "meta" / "parent_hashes_after.csv", index=False)
    parents_unchanged = v1_before.equals(v1_after) and ext_before.equals(ext_after) and seed_before.equals(seed_after)
    parent_integrity = {"schema": "manuscript_fig5_v3_parent_integrity_v1", "status": "passed" if parents_unchanged else "failed", "parents": {"fig5_v1": {"root": str(v1_parent), "before": _snapshot_digest(v1_before), "after": _snapshot_digest(v1_after), "unchanged": v1_before.equals(v1_after)}, "successor_extension_aggregate": {"root": str(ext_parent), "before": _snapshot_digest(ext_before), "after": _snapshot_digest(ext_after), "unchanged": ext_before.equals(ext_after)}, "successor_extension_seed_summaries": {"root": str(seed_parent), "before": _snapshot_digest(seed_before), "after": _snapshot_digest(seed_after), "unchanged": seed_before.equals(seed_after)}}, "unchanged": parents_unchanged}
    _write_json(output_dir / "meta" / "parent_integrity.json", parent_integrity)
    if not parents_unchanged:
        raise RuntimeError("one or more pinned parent trees changed during candidate rendering")
    run_config = {"candidate_version": CANDIDATE_VERSION, "display_name": DISPLAY_NAME, "plot_only": True, "check_only": bool(check_only), "parent_bundles": {"fig5_v1": str(v1_parent), "successor_extension_aggregate": str(ext_parent), "successor_extension_seed_summaries": str(seed_parent)}, "output_dir": str(output_dir), "expected_networks": list(EXPECTED_SEEDS), "independent_unit": "independently trained network", "source_policy": "read-only persisted source data and frozen statistics", "interval_sources": {"panel_b_overlap": {"overlap": "confirmatory_supplied_network_bootstrap", "non_overlap": "candidate_network_bootstrap_20000", "random": "candidate_network_bootstrap_20000"}, "panel_c": {"next_response": "candidate_network_bootstrap_20000", "following_response": "confirmatory_supplied_network_bootstrap", "new_successor": "confirmatory_supplied_network_bootstrap"}}, "model_or_dataset_initialized": False, "generated_at": _utc_now(), "script": str(Path(__file__).resolve()), "spec": str(SPEC_PATH)}
    _write_json(output_dir / "run_config.json", run_config)
    summary = {"schema": "paper_figure_reader_first_candidate_summary_v3", "candidate_version": CANDIDATE_VERSION, "display_name": DISPLAY_NAME, "status": "check_passed" if check_only else "rendered", "canvas_mm": spec["canvas_mm"], "independent_unit": "independently trained network", "n_networks": 20, "network_seeds": list(EXPECTED_SEEDS), "panel_a_summary_rows": int(len(transfer_stats)), "panel_b_summary_rows": int(len(overlap_stats)), "panel_c_summary_rows": int(len(propagation_stats)), "panel_d_stage_count": int(recurrence_stats["stage_k"].nunique()), "panel_e_series": ["rescue", "loss"], "interval_sources": run_config["interval_sources"], "outputs": {key: str(path.relative_to(output_dir)) for key, path in outputs.items()}, "parent_integrity": parent_integrity, "layout_status": layout_audit["status"], "render_qa_status": render_qa["status"] if render_qa else "not_run", "grayscale_qa_status": grayscale_qa["status"] if grayscale_qa else "not_run", "visual_qa_status": visual_qa["status"] if visual_qa else "not_run"}
    _write_json(output_dir / "summary.json", summary)
    log_lines = [f"{_utc_now()} candidate={CANDIDATE_VERSION}", f"mode={'check-only' if check_only else 'plot-only render'}", f"fig5_v1_before={_snapshot_digest(v1_before)}", f"fig5_v1_after={_snapshot_digest(v1_after)}", f"extension_aggregate_before={_snapshot_digest(ext_before)}", f"extension_aggregate_after={_snapshot_digest(ext_after)}", f"extension_seed_before={_snapshot_digest(seed_before)}", f"extension_seed_after={_snapshot_digest(seed_after)}", f"layout={layout_audit['status']}", f"render_qa={render_qa['status'] if render_qa else 'not_run'}", f"grayscale_qa={grayscale_qa['status'] if grayscale_qa else 'not_run'}", f"visual_qa={visual_qa['status'] if visual_qa else 'not_run'}", f"parent_integrity={parent_integrity['status']}"]
    (output_dir / "logs" / "render.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    manifest = _write_artifact_manifest(output_dir)
    return {"status": summary["status"], "output_dir": str(output_dir), "outputs": summary["outputs"], "layout": layout_audit["status"], "render_qa": summary["render_qa_status"], "grayscale_qa": summary["grayscale_qa_status"], "visual_qa": summary["visual_qa_status"], "parent_integrity": parent_integrity["status"], "artifact_count": manifest["artifact_count"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the reader-first manuscript Fig.5 v3 candidate.")
    parser.add_argument("--output-dir", default="results/paper_figure_candidates/manuscript_fig5_reader_first_v3")
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
