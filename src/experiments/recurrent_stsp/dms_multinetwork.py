"""Persist and aggregate independent-network DMS replication runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import statistics
from typing import Dict, Iterable, List, Sequence, Union

from .connectivity import SparseRecurrentConnectivity
from .recording import atomic_json_dump


PathLike = Union[str, os.PathLike]


@dataclass(frozen=True)
class DmsNetworkEntry:
    graph_seed: int
    connectivity_path: str
    run_directory: str

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def write_dms_network_manifest(
    path: PathLike,
    entries: Sequence[DmsNetworkEntry],
    *,
    experiment_config_path: PathLike,
) -> Path:
    """Freeze graph identities and per-network destinations before rollout."""

    if len(entries) < 2:
        raise ValueError("A multi-network manifest requires at least two networks.")
    seeds = [entry.graph_seed for entry in entries]
    if len(seeds) != len(set(seeds)) or min(seeds) < 0:
        raise ValueError("Graph seeds must be unique and non-negative.")
    run_directories = [str(Path(entry.run_directory).resolve()) for entry in entries]
    if len(run_directories) != len(set(run_directories)):
        raise ValueError("Per-network run directories must be unique.")
    config = Path(experiment_config_path).resolve()
    payload = {
        "schema_version": 1,
        "interpretation_boundary": (
            "multi-network replication of the causal role of pre-query STSP; "
            "the Fig.4 sequential STSP-to-next-window-firing mechanism was not "
            "tested, and the result is not confirmatory inference"
        ),
        "experiment_config": str(config),
        "networks": [
            {
                **entry.as_dict(),
                "connectivity_path": str(Path(entry.connectivity_path).resolve()),
                "run_directory": str(Path(entry.run_directory).resolve()),
                "status": "pending",
            }
            for entry in entries
        ],
        "tasks": {
            "per_network_runs": {
                "depends_on": [str(config), "each persisted connectivity_path"],
                "outputs": ["each run_directory/analysis_metrics.json"],
            },
            "network_level_aggregation": {
                "depends_on": ["all per-network analysis_metrics.json"],
                "outputs": ["network_rows.json", "network_level_metrics.json"],
                "status": "pending",
            },
        },
    }
    return atomic_json_dump(payload, path)


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _network_row(graph_seed: int, run_directory: Path) -> Dict[str, object]:
    metrics = _load_json(run_directory / "analysis_metrics.json")
    behavior = metrics["behavior_decoder"]["test"]["balanced_accuracy"]
    probe = metrics["probe_only_control_decoder"]["test"]["balanced_accuracy"]
    firing = metrics["delay_firing_sample_decoder"]["test"]["balanced_accuracy"]
    stsp = metrics["pre_query_stsp_sample_decoder"]["test"]["balanced_accuracy"]
    successor = metrics["post_query_stsp_match_decoder"]["test"][
        "balanced_accuracy"
    ]
    causal = metrics["causal_controls"]
    return {
        "graph_seed": graph_seed,
        "run_directory": str(run_directory.resolve()),
        "behavior_balanced_accuracy": behavior,
        "probe_only_balanced_accuracy": probe,
        "behavior_minus_probe": behavior - probe,
        "delay_firing_sample_balanced_accuracy": firing,
        "pre_query_stsp_sample_balanced_accuracy": stsp,
        "stsp_minus_delay_firing": stsp - firing,
        "post_query_stsp_match_balanced_accuracy": successor,
        "reset_balanced_accuracy": causal["reset_test_balanced_accuracy"],
        "dynamic_minus_reset": behavior - causal["reset_test_balanced_accuracy"],
        "swap_accuracy_against_donor_label": causal[
            "swap_accuracy_against_donor_label"
        ],
        "swap_donor_projection": causal["mean_swap_donor_projection"],
        "swap_donor_directed_score_change": causal[
            "mean_swap_donor_directed_score_change"
        ],
        "mean_delay_population_rate_hz": metrics["mean_delay_population_rate_hz"],
        "gates": metrics["gates"],
        "all_pilot_gates_pass": metrics["all_pilot_gates_pass"],
    }


def _summary(values: Iterable[float]) -> Dict[str, float]:
    resolved = [float(value) for value in values]
    if not resolved or any(not math.isfinite(value) for value in resolved):
        raise ValueError("Network-level endpoint values must be finite.")
    return {
        "mean": statistics.mean(resolved),
        "sample_sd": statistics.stdev(resolved) if len(resolved) > 1 else 0.0,
        "minimum": min(resolved),
        "maximum": max(resolved),
    }


def aggregate_dms_network_results(
    network_manifest_path: PathLike,
    *,
    output_directory: PathLike,
) -> Dict[str, object]:
    """Verify frozen inputs and aggregate one already-analyzed row per graph."""

    manifest_path = Path(network_manifest_path)
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported DMS network-manifest schema.")
    entries = manifest.get("networks")
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("A DMS network manifest must contain at least two entries.")
    declared_seeds = [int(entry["graph_seed"]) for entry in entries]
    if len(declared_seeds) != len(set(declared_seeds)) or min(declared_seeds) < 0:
        raise ValueError("Manifest graph seeds must be unique and non-negative.")
    run_directories = [
        str(Path(entry["run_directory"]).resolve()) for entry in entries
    ]
    if len(run_directories) != len(set(run_directories)):
        raise ValueError("Manifest run directories must be unique.")
    expected_config = _load_json(Path(manifest["experiment_config"]))
    rows: List[Dict[str, object]] = []
    reference_network_config = None
    for entry in entries:
        graph_path = Path(entry["connectivity_path"])
        run_directory = Path(entry["run_directory"])
        if not graph_path.is_file():
            raise FileNotFoundError("Missing frozen graph: {}".format(graph_path))
        if not (run_directory / "analysis_metrics.json").is_file():
            raise FileNotFoundError(
                "Missing per-network analysis: {}".format(run_directory)
            )
        actual_config = _load_json(run_directory / "experiment_config.json")
        if actual_config != expected_config:
            raise ValueError("Per-network DMS experiment configs are not identical.")
        graph = SparseRecurrentConnectivity.load(graph_path)
        if graph.config.seed != int(entry["graph_seed"]):
            raise ValueError("A persisted graph seed does not match the frozen manifest.")
        graph_config = asdict(graph.config)
        graph_config.pop("seed")
        if reference_network_config is None:
            reference_network_config = graph_config
        elif graph_config != reference_network_config:
            raise ValueError("Independent graphs differ in more than their seed.")
        rows.append(_network_row(graph.config.seed, run_directory))
        entry["status"] = "complete"

    endpoints = (
        "behavior_balanced_accuracy",
        "probe_only_balanced_accuracy",
        "behavior_minus_probe",
        "delay_firing_sample_balanced_accuracy",
        "pre_query_stsp_sample_balanced_accuracy",
        "stsp_minus_delay_firing",
        "post_query_stsp_match_balanced_accuracy",
        "reset_balanced_accuracy",
        "dynamic_minus_reset",
        "swap_accuracy_against_donor_label",
        "swap_donor_projection",
        "swap_donor_directed_score_change",
        "mean_delay_population_rate_hz",
    )
    gate_names = tuple(rows[0]["gates"].keys())
    aggregate = {
        "interpretation_boundary": (
            "multi-network replication of the causal role of pre-query STSP; "
            "the Fig.4 sequential STSP-to-next-window-firing mechanism was not "
            "tested, and the result is not confirmatory inference"
        ),
        "network_count": len(rows),
        "graph_seeds": [row["graph_seed"] for row in rows],
        "endpoint_summaries": {
            endpoint: _summary(row[endpoint] for row in rows)
            for endpoint in endpoints
        },
        "gate_pass_counts": {
            gate: sum(bool(row["gates"][gate]) for row in rows)
            for gate in gate_names
        },
        "all_pilot_gates_pass_count": sum(
            bool(row["all_pilot_gates_pass"]) for row in rows
        ),
        "three_graph_causal_role_replication_supported": len(rows) >= 3 and all(
            bool(row["all_pilot_gates_pass"]) for row in rows
        ),
        "fig4_sequential_mechanism_tested": False,
    }
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(
        {"schema_version": 1, "networks": rows},
        destination / "network_rows.json",
    )
    atomic_json_dump(aggregate, destination / "network_level_metrics.json")
    manifest["tasks"]["network_level_aggregation"]["status"] = "complete"
    manifest["interpretation_boundary"] = aggregate["interpretation_boundary"]
    atomic_json_dump(manifest, manifest_path)
    return aggregate


__all__ = [
    "DmsNetworkEntry",
    "aggregate_dms_network_results",
    "write_dms_network_manifest",
]
