"""Behavior-first delayed-match-to-sample experiment for recurrent STSP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import platform
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from .checkpoint import (
    NetworkCheckpoint,
    capture_network_checkpoint,
    replace_plastic_state,
    reset_plastic_state_to_no_event_baseline,
    restore_network_checkpoint,
)
from .connectivity import SparseRecurrentConnectivity
from .dms_trials import DmsExperimentConfig, DmsTrial, build_dms_trial_manifest
from .linear_decoder import (
    apply_ridge_decoder,
    balanced_accuracy,
    fit_ridge_decoder,
)
from .protocol import ExternalInputEngine, ItemLoadingSignal, WorkingMemoryProtocolConfig
from .recording import atomic_json_dump, atomic_torch_save
from .runner import SimulationRunConfig
from .scheduler import SparseRecurrentNetwork


PathLike = Union[str, os.PathLike]


@dataclass
class _DmsBoundary:
    trial: DmsTrial
    protocol: WorkingMemoryProtocolConfig
    network: NetworkCheckpoint
    external: Dict[str, object]
    delay_firing_rates_hz: torch.Tensor
    pre_stsp_features: torch.Tensor


@dataclass
class _DmsBranch:
    mode: str
    recipient_trial_id: str
    donor_trial_id: Optional[str]
    response_features_hz: torch.Tensor
    response_population_rates_hz: torch.Tensor
    post_stsp_features: torch.Tensor
    total_spikes: int
    plastic_edge_events: int
    static_edge_events: int


def _grid_step(time_ms: float, dt_ms: float) -> int:
    scaled = time_ms / dt_ms
    rounded = int(math.floor(scaled + 0.5))
    if not math.isclose(scaled, rounded, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("DMS times must lie on the simulation grid.")
    return rounded


def _population_counts(spikes: torch.Tensor, network_config) -> torch.Tensor:
    selective = network_config.selective_population_size
    selected = spikes[: network_config.n_memories * selective]
    return selected.reshape(network_config.n_memories, selective).sum(dim=1)


def _advance_one_step(
    network: SparseRecurrentNetwork, external: ExternalInputEngine, event_step: int
):
    external_ex, external_in, external_current = external.pop_current()
    result = network.step(
        external_spikes_ex_pa=external_ex,
        external_spikes_in_pa=external_in,
        external_current_0_pa=external_current,
    )
    external.emit(event_step)
    external.advance()
    return result


class _StspPopulationFeatureExtractor:
    """Summarize continuous STSP over fixed source/target population groups."""

    def __init__(
        self,
        network: SparseRecurrentNetwork,
        source_populations: Sequence[int],
        edges_per_source_population: int,
    ) -> None:
        self.network = network
        self.source_populations = tuple(source_populations)
        self.target_group_count = network.config.n_memories + 1
        graph = network.scheduler.connectivity.plastic
        population_size = network.config.selective_population_size
        edge_ids: List[torch.Tensor] = []
        group_ids: List[torch.Tensor] = []
        for source_slot, population in enumerate(self.source_populations):
            source_start = population * population_size
            source_stop = source_start + population_size
            first = int(graph.row_ptr[source_start].item())
            last = int(graph.row_ptr[source_stop].item())
            available = last - first
            count = min(edges_per_source_population, available)
            if count <= 0:
                raise ValueError("A DMS source population has no plastic edges.")
            offsets = torch.div(
                torch.arange(count, dtype=torch.int64, device=network.device)
                * available,
                count,
                rounding_mode="floor",
            )
            chosen = offsets + first
            targets = graph.targets[chosen].to(torch.int64)
            target_groups = torch.div(
                targets, population_size, rounding_mode="floor"
            ).clamp_max(network.config.n_memories)
            edge_ids.append(chosen)
            group_ids.append(source_slot * self.target_group_count + target_groups)
        self.edge_ids = torch.cat(edge_ids)
        self.group_ids = torch.cat(group_ids)
        self.group_count = len(self.source_populations) * self.target_group_count
        counts = torch.bincount(self.group_ids, minlength=self.group_count)
        self.group_counts = counts.to(dtype=network.dtype).clamp_min_(1.0)

    @torch.no_grad()
    def extract(self) -> torch.Tensor:
        scheduler = self.network.scheduler
        graph = scheduler.connectivity.plastic
        state = scheduler.plastic_state
        indices = self.edge_ids
        time_ms = self.network.step_index * self.network.config.dt_ms
        elapsed = torch.clamp_min(time_ms - state.last_spike_time_ms[indices], 0.0)
        baseline_u = self.network.config.stsp_u
        u = baseline_u + (state.u[indices] - baseline_u) * torch.exp(
            -elapsed / graph.tau_fac_ms[indices]
        )
        x = 1.0 + (state.x[indices] - 1.0) * torch.exp(
            -elapsed / graph.tau_rec_ms[indices]
        )
        next_u = u + baseline_u * (1.0 - u)
        release = graph.weights[indices] * next_u * x
        summaries = []
        for values in (u, x, release):
            sums = torch.zeros(
                self.group_count, dtype=self.network.dtype, device=self.network.device
            )
            sums.scatter_add_(0, self.group_ids, values)
            summaries.append(sums / self.group_counts)
        return torch.cat(summaries).to(dtype=torch.float64, device="cpu")


def _trial_protocol(
    connectivity: SparseRecurrentConnectivity,
    experiment: DmsExperimentConfig,
    trial: DmsTrial,
) -> Tuple[WorkingMemoryProtocolConfig, float, float]:
    sample_stop_ms = experiment.sample_origin_ms + experiment.cue_duration_ms
    query_origin_ms = sample_stop_ms + trial.delay_ms
    total_time_ms = query_origin_ms + (
        experiment.response_bin_ms * experiment.response_bin_count
    )
    signals = [ItemLoadingSignal(trial.sample_population, experiment.sample_origin_ms)]
    if trial.distracted:
        distractor_origin_ms = sample_stop_ms + (
            trial.delay_ms - experiment.cue_duration_ms
        ) / 2.0
        signals.append(
            ItemLoadingSignal(experiment.distractor_population, distractor_origin_ms)
        )
    signals.append(ItemLoadingSignal(trial.probe_population, query_origin_ms))
    protocol = WorkingMemoryProtocolConfig(
        total_time_ms=total_time_ms,
        background_stop_ms=total_time_ms,
        poisson_input=experiment.poisson_input,
        item_loading=tuple(signals),
        cue_duration_ms=experiment.cue_duration_ms,
        eta_end_origin_ms=total_time_ms + connectivity.config.dt_ms,
        seed=trial.input_seed,
    )
    return protocol, query_origin_ms, total_time_ms


def _capture_trial_boundary(
    network: SparseRecurrentNetwork,
    connectivity: SparseRecurrentConnectivity,
    experiment: DmsExperimentConfig,
    trial: DmsTrial,
    stsp_features: _StspPopulationFeatureExtractor,
) -> _DmsBoundary:
    network.reset()
    protocol, query_origin_ms, _ = _trial_protocol(connectivity, experiment, trial)
    external = ExternalInputEngine(
        connectivity.config,
        protocol,
        device=network.device,
        dtype=network.dtype,
    )
    query_step = _grid_step(query_origin_ms, connectivity.config.dt_ms)
    silent_steps = _grid_step(experiment.silent_window_ms, connectivity.config.dt_ms)
    delay_counts = torch.zeros(
        connectivity.config.n_memories,
        dtype=torch.int64,
        device=network.device,
    )
    for event_step in range(1, query_step + 1):
        result = _advance_one_step(network, external, event_step)
        if event_step > query_step - silent_steps:
            delay_counts += _population_counts(result.spikes, connectivity.config)
    delay_rates = delay_counts.to(torch.float64) / (
        connectivity.config.selective_population_size
        * (experiment.silent_window_ms / 1_000.0)
    )
    return _DmsBoundary(
        trial=trial,
        protocol=protocol,
        network=capture_network_checkpoint(network),
        external=external.state_dict(),
        delay_firing_rates_hz=delay_rates.to("cpu"),
        pre_stsp_features=stsp_features.extract(),
    )


def _run_trial_branch(
    network: SparseRecurrentNetwork,
    connectivity: SparseRecurrentConnectivity,
    experiment: DmsExperimentConfig,
    recipient: _DmsBoundary,
    stsp_features: _StspPopulationFeatureExtractor,
    *,
    mode: str,
    donor: Optional[_DmsBoundary] = None,
) -> _DmsBranch:
    restore_network_checkpoint(network, recipient.network)
    if mode == "reset":
        reset_plastic_state_to_no_event_baseline(network)
    elif mode == "swap":
        if donor is None:
            raise ValueError("A swap branch requires a donor boundary.")
        replace_plastic_state(network, donor.network.plastic)
    elif mode != "dynamic_sham":
        raise ValueError("Unknown DMS branch mode: {}".format(mode))
    external = ExternalInputEngine(
        connectivity.config,
        recipient.protocol,
        device=network.device,
        dtype=network.dtype,
    )
    external.load_state_dict(recipient.external)
    _, query_origin_ms, total_time_ms = _trial_protocol(
        connectivity, experiment, recipient.trial
    )
    query_step = _grid_step(query_origin_ms, connectivity.config.dt_ms)
    total_step = _grid_step(total_time_ms, connectivity.config.dt_ms)
    bin_steps = _grid_step(experiment.response_bin_ms, connectivity.config.dt_ms)
    counts = torch.zeros(
        (experiment.response_bin_count, connectivity.config.n_memories),
        dtype=torch.int64,
        device=network.device,
    )
    total_spikes = torch.zeros((), dtype=torch.int64, device=network.device)
    plastic_events = 0
    static_events = 0
    for event_step in range(recipient.network.step_index + 1, total_step + 1):
        result = _advance_one_step(network, external, event_step)
        total_spikes += result.spikes.sum()
        plastic_events += result.dispatch.plastic_events
        static_events += result.dispatch.static_events
        relative_step = event_step - query_step - 1
        if relative_step >= 0:
            bin_index = relative_step // bin_steps
            if bin_index < experiment.response_bin_count:
                counts[bin_index] += _population_counts(
                    result.spikes, connectivity.config
                )
    rates = counts.to(torch.float64) / (
        connectivity.config.selective_population_size
        * (experiment.response_bin_ms / 1_000.0)
    )
    return _DmsBranch(
        mode=mode,
        recipient_trial_id=recipient.trial.trial_id,
        donor_trial_id=None if donor is None else donor.trial.trial_id,
        response_features_hz=rates.flatten().to("cpu"),
        response_population_rates_hz=rates.to("cpu"),
        post_stsp_features=stsp_features.extract(),
        total_spikes=int(total_spikes.item()),
        plastic_edge_events=plastic_events,
        static_edge_events=static_events,
    )


def _runtime_payload(runtime: SimulationRunConfig, elapsed_seconds: float) -> Dict[str, object]:
    return {
        "elapsed_seconds": elapsed_seconds,
        "device": runtime.device,
        "dtype": runtime.dtype,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(torch.device(runtime.device))
            if torch.device(runtime.device).type == "cuda" and torch.cuda.is_available()
            else None
        ),
    }


def run_dms_trial_simulation(
    connectivity: SparseRecurrentConnectivity,
    output_directory: PathLike,
    *,
    experiment: DmsExperimentConfig = DmsExperimentConfig(),
    runtime: SimulationRunConfig = SimulationRunConfig(),
    overwrite: bool = False,
    connectivity_path: Optional[PathLike] = None,
    progress_interval_pairs: int = 10,
) -> Dict[str, object]:
    """Persist balanced trial features without fitting any decoder."""

    if max((*experiment.task_populations, experiment.distractor_population)) >= (
        connectivity.config.n_memories
    ):
        raise ValueError("DMS populations lie outside the connectivity memory groups.")
    destination = Path(output_directory)
    feature_path = destination / "trial_features.pt"
    if feature_path.exists() and not overwrite:
        raise FileExistsError(
            "DMS trial features already exist; pass overwrite=True to replace them."
        )
    destination.mkdir(parents=True, exist_ok=True)
    trials = build_dms_trial_manifest(experiment)
    atomic_json_dump(experiment.as_dict(), destination / "experiment_config.json")
    atomic_json_dump(
        {"schema_version": 1, "trials": [trial.as_dict() for trial in trials]},
        destination / "trial_manifest.json",
    )
    dtype = torch.float32 if runtime.dtype == "float32" else torch.float64
    network = SparseRecurrentNetwork(
        connectivity,
        device=runtime.device,
        dtype=dtype,
        source_chunk_size=runtime.source_chunk_size,
    )
    extractor = _StspPopulationFeatureExtractor(
        network,
        experiment.task_populations,
        experiment.stsp_edges_per_source_population,
    )
    pairs: Dict[str, List[DmsTrial]] = {}
    for trial in trials:
        pairs.setdefault(trial.pair_id, []).append(trial)
    dynamic: Dict[str, Tuple[_DmsBoundary, _DmsBranch]] = {}
    controls: Dict[str, Dict[str, _DmsBranch]] = {}
    started = time.perf_counter()
    for pair_index, (pair_id, pair_trials) in enumerate(pairs.items(), start=1):
        boundaries = {
            trial.trial_id: _capture_trial_boundary(
                network, connectivity, experiment, trial, extractor
            )
            for trial in pair_trials
        }
        for trial in pair_trials:
            boundary = boundaries[trial.trial_id]
            branch = _run_trial_branch(
                network,
                connectivity,
                experiment,
                boundary,
                extractor,
                mode="dynamic_sham",
            )
            dynamic[trial.trial_id] = (boundary, branch)
        if pair_trials[0].split == "test":
            for trial in pair_trials:
                recipient = boundaries[trial.trial_id]
                donor_trial = next(
                    candidate
                    for candidate in pair_trials
                    if candidate.trial_id != trial.trial_id
                )
                donor = boundaries[donor_trial.trial_id]
                controls[trial.trial_id] = {
                    "reset": _run_trial_branch(
                        network,
                        connectivity,
                        experiment,
                        recipient,
                        extractor,
                        mode="reset",
                    ),
                    "swap": _run_trial_branch(
                        network,
                        connectivity,
                        experiment,
                        recipient,
                        extractor,
                        mode="swap",
                        donor=donor,
                    ),
                }
        if progress_interval_pairs > 0 and (
            pair_index % progress_interval_pairs == 0 or pair_index == len(pairs)
        ):
            print(
                "DMS pairs {}/{} elapsed {:.1f}s".format(
                    pair_index, len(pairs), time.perf_counter() - started
                ),
                flush=True,
            )

    ordered = [dynamic[trial.trial_id] for trial in trials]
    test_trials = [trial for trial in trials if trial.split == "test"]
    payload = {
        "schema_version": 1,
        "trial_metadata": [trial.as_dict() for trial in trials],
        "feature_schema": {
            "response": "response_bin x selective_population firing rate (Hz)",
            "delay_firing": "last silent_window selective-population rate (Hz)",
            "stsp": "continuous mean u, x, next-release by source/target population",
            "stsp_source_populations": list(experiment.task_populations),
            "stsp_target_group_count": extractor.target_group_count,
        },
        "dynamic": {
            "response_features_hz": torch.stack(
                [branch.response_features_hz for _, branch in ordered]
            ),
            "response_population_rates_hz": torch.stack(
                [branch.response_population_rates_hz for _, branch in ordered]
            ),
            "delay_firing_features_hz": torch.stack(
                [boundary.delay_firing_rates_hz for boundary, _ in ordered]
            ),
            "pre_stsp_features": torch.stack(
                [boundary.pre_stsp_features for boundary, _ in ordered]
            ),
            "post_stsp_features": torch.stack(
                [branch.post_stsp_features for _, branch in ordered]
            ),
            "total_spikes": torch.tensor(
                [branch.total_spikes for _, branch in ordered], dtype=torch.int64
            ),
            "plastic_edge_events": torch.tensor(
                [branch.plastic_edge_events for _, branch in ordered],
                dtype=torch.int64,
            ),
            "static_edge_events": torch.tensor(
                [branch.static_edge_events for _, branch in ordered],
                dtype=torch.int64,
            ),
        },
        "test_controls": {
            "trial_ids": [trial.trial_id for trial in test_trials],
            "donor_trial_ids": [
                controls[trial.trial_id]["swap"].donor_trial_id for trial in test_trials
            ],
            "reset_response_features_hz": torch.stack(
                [
                    controls[trial.trial_id]["reset"].response_features_hz
                    for trial in test_trials
                ]
            ),
            "reset_post_stsp_features": torch.stack(
                [
                    controls[trial.trial_id]["reset"].post_stsp_features
                    for trial in test_trials
                ]
            ),
            "swap_response_features_hz": torch.stack(
                [
                    controls[trial.trial_id]["swap"].response_features_hz
                    for trial in test_trials
                ]
            ),
            "swap_post_stsp_features": torch.stack(
                [
                    controls[trial.trial_id]["swap"].post_stsp_features
                    for trial in test_trials
                ]
            ),
        },
    }
    atomic_torch_save(payload, feature_path)
    elapsed = time.perf_counter() - started
    run_info = _runtime_payload(runtime, elapsed)
    run_info.update(
        {
            "network_seed": connectivity.config.seed,
            "network_neurons": connectivity.config.n_neurons,
            "network_edges": connectivity.num_edges,
            "plastic_edges": connectivity.plastic.num_edges,
            "trial_count": len(trials),
            "pair_count": len(pairs),
        }
    )
    atomic_json_dump(run_info, destination / "run_info.json")
    graph_input = (
        str(Path(connectivity_path).resolve())
        if connectivity_path is not None
        else "in-memory-connectivity"
    )
    atomic_json_dump(
        {
            "schema_version": 1,
            "tasks": {
                "trial_manifest": {
                    "depends_on": ["experiment_config"],
                    "outputs": ["trial_manifest.json"],
                },
                "per_network_trial_simulation": {
                    "depends_on": [
                        graph_input,
                        "experiment_config.json",
                        "trial_manifest.json",
                    ],
                    "outputs": ["trial_features.pt", "run_info.json"],
                },
                "decoder_analysis": {
                    "depends_on": ["trial_features.pt", "trial_manifest.json"],
                    "outputs": ["decoders.pt", "analysis_metrics.json"],
                    "status": "not-run-by-simulation-task",
                },
            },
        },
        destination / "artifact_manifest.json",
    )
    return {
        "output_directory": str(destination.resolve()),
        "trial_count": len(trials),
        "test_control_count": len(test_trials),
        "elapsed_seconds": elapsed,
        "features": str(feature_path.resolve()),
    }


def _decoder_json(metrics: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"predictions", "logits"}
    }


def _stratified_behavior(
    metadata: Sequence[Dict[str, object]],
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> List[Dict[str, object]]:
    rows = []
    for delay in sorted({float(row["delay_ms"]) for row in metadata}):
        for distracted in sorted({bool(row["distracted"]) for row in metadata}):
            indices = [
                index
                for index, row in enumerate(metadata)
                if row["split"] == "test"
                and float(row["delay_ms"]) == delay
                and bool(row["distracted"]) == distracted
            ]
            if not indices:
                continue
            index_tensor = torch.tensor(indices, dtype=torch.int64)
            rows.append(
                {
                    "delay_ms": delay,
                    "distracted": distracted,
                    "n_trials": len(indices),
                    "balanced_accuracy": balanced_accuracy(
                        labels[index_tensor], predictions[index_tensor]
                    ),
                }
            )
    return rows


def analyze_dms_trial_features(output_directory: PathLike) -> Dict[str, object]:
    """Fit frozen readouts from persisted features without rerunning simulation."""

    destination = Path(output_directory)
    payload = torch.load(
        destination / "trial_features.pt", map_location="cpu", weights_only=False
    )
    metadata = payload["trial_metadata"]
    dynamic = payload["dynamic"]
    splits = [str(row["split"]) for row in metadata]
    match_labels = torch.tensor(
        [int(bool(row["is_match"])) for row in metadata], dtype=torch.int64
    )
    sample_labels = torch.tensor(
        [int(row["sample_population"]) for row in metadata], dtype=torch.int64
    )
    probe_labels = torch.tensor(
        [int(row["probe_population"]) for row in metadata], dtype=torch.int64
    )
    probe_classes = torch.unique(probe_labels, sorted=True)
    probe_only = (probe_labels[:, None] == probe_classes[None, :]).to(torch.float64)
    behavior_model, behavior_metrics = fit_ridge_decoder(
        dynamic["response_features_hz"], match_labels, splits
    )
    probe_model, probe_metrics = fit_ridge_decoder(probe_only, match_labels, splits)
    firing_model, firing_metrics = fit_ridge_decoder(
        dynamic["delay_firing_features_hz"], sample_labels, splits
    )
    stsp_model, stsp_metrics = fit_ridge_decoder(
        dynamic["pre_stsp_features"], sample_labels, splits
    )
    successor_model, successor_metrics = fit_ridge_decoder(
        dynamic["post_stsp_features"], match_labels, splits
    )
    controls = payload["test_controls"]
    id_to_index = {
        str(row["trial_id"]): index for index, row in enumerate(metadata)
    }
    test_indices = torch.tensor(
        [id_to_index[str(trial_id)] for trial_id in controls["trial_ids"]],
        dtype=torch.int64,
    )
    donor_indices = torch.tensor(
        [id_to_index[str(trial_id)] for trial_id in controls["donor_trial_ids"]],
        dtype=torch.int64,
    )
    reset_predictions, reset_logits = apply_ridge_decoder(
        behavior_model, controls["reset_response_features_hz"]
    )
    swap_predictions, swap_logits = apply_ridge_decoder(
        behavior_model, controls["swap_response_features_hz"]
    )
    dynamic_logits = behavior_metrics["logits"]
    dynamic_scores = dynamic_logits[:, 1] - dynamic_logits[:, 0]
    reset_scores = reset_logits[:, 1] - reset_logits[:, 0]
    swap_scores = swap_logits[:, 1] - swap_logits[:, 0]
    projections: List[float] = []
    signed_movements: List[float] = []
    for control_index, (recipient_index, donor_index) in enumerate(
        zip(test_indices.tolist(), donor_indices.tolist())
    ):
        recipient_score = float(dynamic_scores[recipient_index].item())
        donor_score = float(dynamic_scores[donor_index].item())
        swapped_score = float(swap_scores[control_index].item())
        denominator = donor_score - recipient_score
        if abs(denominator) > 1e-12:
            projections.append((swapped_score - recipient_score) / denominator)
        donor_direction = 1.0 if bool(match_labels[donor_index].item()) else -1.0
        signed_movements.append((swapped_score - recipient_score) * donor_direction)
    test_labels = match_labels[test_indices]
    donor_labels = match_labels[donor_indices]
    behavior_test = float(behavior_metrics["test"]["balanced_accuracy"])
    probe_test = float(probe_metrics["test"]["balanced_accuracy"])
    reset_test = balanced_accuracy(test_labels, reset_predictions)
    swap_donor_accuracy = balanced_accuracy(donor_labels, swap_predictions)
    firing_test = float(firing_metrics["test"]["balanced_accuracy"])
    stsp_test = float(stsp_metrics["test"]["balanced_accuracy"])
    gates = {
        "behavior_feasibility": behavior_test >= 0.60 and behavior_test > probe_test,
        "activity_silent_pattern": stsp_test > firing_test,
        "reset_disrupts_behavior": reset_test < behavior_test,
        "swap_moves_toward_donor": bool(signed_movements)
        and sum(signed_movements) / len(signed_movements) > 0.0,
    }
    metrics: Dict[str, object] = {
        "interpretation_boundary": (
            "single-network pilot of behavior, silent-state association, and the "
            "causal role of pre-query STSP; it does not test the Fig.4 sequential "
            "STSP-to-next-window-firing mechanism or network-level inference"
        ),
        "behavior_decoder": _decoder_json(behavior_metrics),
        "probe_only_control_decoder": _decoder_json(probe_metrics),
        "delay_firing_sample_decoder": _decoder_json(firing_metrics),
        "pre_query_stsp_sample_decoder": _decoder_json(stsp_metrics),
        "post_query_stsp_match_decoder": _decoder_json(successor_metrics),
        "test_behavior_by_condition": _stratified_behavior(
            metadata,
            match_labels,
            behavior_metrics["predictions"],
        ),
        "causal_controls": {
            "dynamic_test_balanced_accuracy": behavior_test,
            "reset_test_balanced_accuracy": reset_test,
            "reset_accuracy_change": reset_test - behavior_test,
            "reset_mean_decision_score": float(reset_scores.mean().item()),
            "swap_accuracy_against_donor_label": swap_donor_accuracy,
            "mean_swap_donor_projection": (
                sum(projections) / len(projections) if projections else None
            ),
            "mean_swap_donor_directed_score_change": (
                sum(signed_movements) / len(signed_movements)
                if signed_movements
                else None
            ),
        },
        "gates": gates,
        "all_pilot_gates_pass": all(gates.values()),
        "mean_delay_population_rate_hz": float(
            dynamic["delay_firing_features_hz"].to(torch.float64).mean().item()
        ),
    }
    atomic_torch_save(
        {
            "schema_version": 1,
            "behavior": behavior_model,
            "probe_only_control": probe_model,
            "delay_firing_sample": firing_model,
            "pre_query_stsp_sample": stsp_model,
            "post_query_stsp_match": successor_model,
        },
        destination / "decoders.pt",
    )
    atomic_json_dump(metrics, destination / "analysis_metrics.json")
    manifest_path = destination / "artifact_manifest.json"
    if manifest_path.exists():
        import json

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["tasks"]["decoder_analysis"]["status"] = "complete"
        atomic_json_dump(manifest, manifest_path)
    return metrics


__all__ = [
    "analyze_dms_trial_features",
    "run_dms_trial_simulation",
]
