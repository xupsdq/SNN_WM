"""Artifact-oriented simulation runner for the recurrent STSP backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Dict, Optional, Union

import torch

from .connectivity import SparseRecurrentConnectivity, generate_sparse_connectivity
from .protocol import ExternalInputEngine, WorkingMemoryProtocolConfig
from .recording import (
    ContinuousStspProbeRecorder,
    SparseSpikeRecorder,
    SpikeRecordingConfig,
    StspProbeRecordingConfig,
    TaskEvaluationConfig,
    atomic_json_dump,
    atomic_torch_save,
    evaluate_task,
)
from .scheduler import SparseRecurrentNetwork


PathLike = Union[str, os.PathLike]


@dataclass(frozen=True)
class SimulationRunConfig:
    """Runtime-only settings; scientific inputs live in the other configs."""

    device: str = "cuda"
    dtype: str = "float32"
    source_chunk_size: int = 256
    progress_interval_steps: int = 10_000

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64.")
        if self.source_chunk_size <= 0 or self.progress_interval_steps < 0:
            raise ValueError("Runtime chunk/progress settings are invalid.")


def _resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_or_build_connectivity(
    path: PathLike,
    *,
    config=None,
    reuse: str = "require",
) -> SparseRecurrentConnectivity:
    """Resolve a persisted graph with explicit, loss-safe reuse semantics."""

    destination = Path(path)
    if reuse not in ("require", "auto", "rebuild"):
        raise ValueError("reuse must be require, auto, or rebuild.")
    if destination.exists() and reuse != "rebuild":
        graph = SparseRecurrentConnectivity.load(destination)
        if config is not None and graph.config != config:
            raise ValueError("Persisted connectivity does not match requested config.")
        return graph
    if reuse == "require":
        raise FileNotFoundError(
            "Connectivity artifact is required but missing: {}".format(destination)
        )
    graph = generate_sparse_connectivity(config)
    graph.save(destination)
    return graph


def evaluate_run(
    run_directory: PathLike,
    evaluation: Optional[TaskEvaluationConfig] = None,
) -> Dict[str, object]:
    """Downstream decoder task: read persisted spikes and write only metrics."""

    run_dir = Path(run_directory)
    spike_path = run_dir / "data" / "spikes.pt"
    if not spike_path.is_file():
        raise FileNotFoundError("Spike artifact is missing: {}".format(spike_path))
    if evaluation is None:
        config_path = run_dir / "run_config.json"
        if not config_path.is_file():
            raise FileNotFoundError("Run configuration is missing: {}".format(config_path))
        with config_path.open("r", encoding="utf-8") as handle:
            run_config = json.load(handle)
        evaluation = TaskEvaluationConfig(**run_config["evaluation"])
    spike_payload = torch.load(spike_path, map_location="cpu", weights_only=True)
    metrics = evaluate_task(spike_payload, evaluation)
    metrics_path = run_dir / "metrics" / "task_metrics.json"
    atomic_json_dump(metrics, metrics_path)
    return metrics


def run_simulation(
    connectivity: SparseRecurrentConnectivity,
    run_directory: PathLike,
    *,
    protocol: Optional[WorkingMemoryProtocolConfig] = None,
    spike_recording: Optional[SpikeRecordingConfig] = None,
    stsp_recording: Optional[StspProbeRecordingConfig] = None,
    evaluation: Optional[TaskEvaluationConfig] = None,
    runtime: Optional[SimulationRunConfig] = None,
    duration_ms: Optional[float] = None,
    overwrite: bool = False,
    connectivity_path: Optional[PathLike] = None,
) -> Dict[str, object]:
    """Run simulation, persist reusable data, then invoke the decoder task."""

    resolved_protocol = protocol or WorkingMemoryProtocolConfig.upstream_run_protocol(
        seed=connectivity.config.seed
    )
    resolved_spikes = spike_recording or SpikeRecordingConfig(
        populations=tuple(range(connectivity.config.n_memories))
    )
    resolved_stsp = stsp_recording or StspProbeRecordingConfig(
        populations=tuple(range(connectivity.config.n_memories))
    )
    resolved_evaluation = evaluation or TaskEvaluationConfig()
    resolved_runtime = runtime or SimulationRunConfig()
    if duration_ms is not None:
        if duration_ms <= 0.0 or duration_ms > resolved_protocol.total_time_ms:
            raise ValueError("duration_ms must lie in (0, protocol.total_time_ms].")
        resolved_protocol = replace(
            resolved_protocol,
            total_time_ms=duration_ms,
            background_stop_ms=min(resolved_protocol.background_stop_ms, duration_ms),
        )
    scaled_steps = resolved_protocol.total_time_ms / connectivity.config.dt_ms
    total_steps = int(math.floor(scaled_steps + 0.5))
    if not math.isclose(scaled_steps, total_steps, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("Simulation duration must lie on the network time grid.")

    run_dir = Path(run_directory)
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            "Run directory is non-empty; pass overwrite=True to replace named artifacts."
        )
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    run_config_payload = {
        "schema_version": 1,
        "network": asdict(connectivity.config),
        "protocol": resolved_protocol.as_dict(),
        "spike_recording": asdict(resolved_spikes),
        "stsp_recording": asdict(resolved_stsp),
        "evaluation": asdict(resolved_evaluation),
        "runtime": asdict(resolved_runtime),
        "total_steps": total_steps,
    }
    atomic_json_dump(run_config_payload, run_dir / "run_config.json")

    requested_device = torch.device(resolved_runtime.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this environment.")
    dtype = _resolve_dtype(resolved_runtime.dtype)
    started_wall = time.time()
    atomic_json_dump(
        {
            "status": "running",
            "started_unix_s": started_wall,
            "connectivity_path": (
                None if connectivity_path is None else str(Path(connectivity_path).resolve())
            ),
        },
        run_dir / "meta" / "run_info.json",
    )

    network = SparseRecurrentNetwork(
        connectivity,
        device=requested_device,
        dtype=dtype,
        source_chunk_size=resolved_runtime.source_chunk_size,
    )
    external = ExternalInputEngine(
        connectivity.config,
        resolved_protocol,
        device=requested_device,
        dtype=dtype,
    )
    spike_recorder = SparseSpikeRecorder(
        connectivity.config, resolved_spikes, device=requested_device
    )
    stsp_recorder = ContinuousStspProbeRecorder(network.scheduler, resolved_stsp)
    active_source_events = 0
    plastic_edge_events = 0
    static_edge_events = 0

    _synchronize(requested_device)
    simulation_started = time.perf_counter()
    for event_step in range(1, total_steps + 1):
        external_ex, external_in, external_current = external.pop_current()
        result = network.step(
            external_spikes_ex_pa=external_ex,
            external_spikes_in_pa=external_in,
            external_current_0_pa=external_current,
        )
        external.emit(event_step)
        external.advance()
        spike_recorder.record(result.spikes, event_step)
        stsp_recorder.record(event_step)
        active_source_events += result.dispatch.active_sources
        plastic_edge_events += result.dispatch.plastic_events
        static_edge_events += result.dispatch.static_events
        interval = resolved_runtime.progress_interval_steps
        if interval and (event_step % interval == 0 or event_step == total_steps):
            print(
                "recurrent-stsp step {}/{} ({:.1f} ms)".format(
                    event_step, total_steps, result.time_ms
                ),
                flush=True,
            )
    _synchronize(requested_device)
    simulation_elapsed = time.perf_counter() - simulation_started

    spike_payload = spike_recorder.payload()
    stsp_payload = stsp_recorder.payload()
    spike_path = atomic_torch_save(spike_payload, run_dir / "data" / "spikes.pt")
    stsp_path = atomic_torch_save(
        stsp_payload, run_dir / "data" / "stsp_probes.pt"
    )
    metrics = evaluate_task(spike_payload, resolved_evaluation)
    metrics_path = atomic_json_dump(
        metrics, run_dir / "metrics" / "task_metrics.json"
    )
    completed_wall = time.time()
    run_info = {
        "status": "complete",
        "started_unix_s": started_wall,
        "completed_unix_s": completed_wall,
        "wall_time_s": completed_wall - started_wall,
        "simulation_time_s": simulation_elapsed,
        "simulated_time_ms": resolved_protocol.total_time_ms,
        "total_steps": total_steps,
        "device": str(requested_device),
        "dtype": resolved_runtime.dtype,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cuda_version": torch.version.cuda,
        "connectivity_path": (
            None if connectivity_path is None else str(Path(connectivity_path).resolve())
        ),
        "connectivity_edges": connectivity.num_edges,
        "plastic_edges": connectivity.plastic.num_edges,
        "static_edges": connectivity.static.num_edges,
        "active_source_events": active_source_events,
        "plastic_edge_events": plastic_edge_events,
        "static_edge_events": static_edge_events,
        "recorded_spikes": int(torch.as_tensor(spike_payload["times_ms"]).numel()),
        "stsp_probe_edges": int(torch.as_tensor(stsp_payload["edge_ids"]).numel()),
        "stsp_snapshots": int(torch.as_tensor(stsp_payload["times_ms"]).numel()),
        "external_signals": external.description_dicts(),
        "compatibility_notes": [
            "late excitatory offset retains upstream use of inhibitory tau_m",
            "periodic Poisson input implements upstream intent because its original branch constructs a noise_generator with a rate field and drops list entries",
        ],
    }
    run_info_path = atomic_json_dump(run_info, run_dir / "meta" / "run_info.json")
    manifest = {
        "schema_version": 1,
        "tasks": {
            "simulate": {
                "depends_on": [
                    str(Path(connectivity_path).resolve())
                    if connectivity_path is not None
                    else "in_memory_connectivity",
                    "run_config.json",
                ],
                "outputs": ["data/spikes.pt", "data/stsp_probes.pt", "meta/run_info.json"],
            },
            "evaluate": {
                "depends_on": ["data/spikes.pt", "run_config.json"],
                "outputs": ["metrics/task_metrics.json"],
            },
            "plot": {
                "depends_on": [
                    "data/spikes.pt",
                    "data/stsp_probes.pt",
                    "run_config.json",
                ],
                "outputs": ["figures/*"],
                "plot_only": True,
            },
        }
    }
    manifest_path = atomic_json_dump(manifest, run_dir / "artifact_manifest.json")
    summary = {
        "status": "complete",
        "success": metrics["success"],
        "winner_population": metrics["winner_population"],
        "target_population": resolved_evaluation.target_population,
        "target_rate_hz": metrics["target_rate_hz"],
        "target_margin_hz": metrics["target_margin_hz"],
        "steps": total_steps,
        "simulated_time_ms": resolved_protocol.total_time_ms,
        "artifacts": {
            "spikes": str(spike_path.resolve()),
            "stsp_probes": str(stsp_path.resolve()),
            "metrics": str(metrics_path.resolve()),
            "run_info": str(run_info_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
    }
    atomic_json_dump(summary, run_dir / "summary.json")
    return summary


__all__ = [
    "SimulationRunConfig",
    "evaluate_run",
    "load_or_build_connectivity",
    "run_simulation",
]
