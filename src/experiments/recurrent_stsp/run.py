"""Command-line DAG entry points for the reusable recurrent STSP workflow."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Optional, Sequence

from .config import TiddiaNetworkConfig
from .connectivity import SparseRecurrentConnectivity
from .plot_artifacts import plot_run_artifacts
from .protocol import (
    ItemLoadingSignal,
    PeriodicReadoutInterval,
    RandomNonspecificSignal,
    WorkingMemoryProtocolConfig,
)
from .recording import (
    SpikeRecordingConfig,
    StspProbeRecordingConfig,
    TaskEvaluationConfig,
)
from .runner import (
    SimulationRunConfig,
    evaluate_run,
    load_or_build_connectivity,
    run_simulation,
)


def _parse_item(value: str) -> ItemLoadingSignal:
    try:
        population, origin = value.split("@", maxsplit=1)
        return ItemLoadingSignal(int(population), float(origin))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("item must use POPULATION@ORIGIN_MS") from exc


def _parse_random(value: str) -> RandomNonspecificSignal:
    try:
        origin, fraction = value.split("@", maxsplit=1)
        return RandomNonspecificSignal(float(origin), float(fraction))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "random noise must use ORIGIN_MS@FRACTION"
        ) from exc


def _parse_interval(value: str) -> PeriodicReadoutInterval:
    try:
        start, stop = value.split(":", maxsplit=1)
        return PeriodicReadoutInterval(float(start), float(stop))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "periodic interval must use START_MS:STOP_MS"
        ) from exc


def _network_from_arguments(arguments) -> TiddiaNetworkConfig:
    constructor = (
        TiddiaNetworkConfig.heterogeneous_run_config
        if arguments.network_profile == "heterogeneous-upstream"
        else TiddiaNetworkConfig
    )
    overrides = {}
    for argument_name, config_name in (
        ("n_exc", "n_exc"),
        ("n_inh", "n_inh"),
        ("n_memories", "n_memories"),
        ("coding_fraction", "coding_fraction"),
        ("connection_probability", "connection_probability"),
        ("seed", "seed"),
    ):
        value = getattr(arguments, argument_name, None)
        if value is not None:
            overrides[config_name] = value
    return constructor(**overrides)


def _add_network_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--network-profile",
        choices=("default", "heterogeneous-upstream"),
        default="heterogeneous-upstream",
    )
    parser.add_argument("--n-exc", type=int)
    parser.add_argument("--n-inh", type=int)
    parser.add_argument("--n-memories", type=int)
    parser.add_argument("--coding-fraction", type=float)
    parser.add_argument("--connection-probability", type=float)
    parser.add_argument("--seed", type=int)


def _protocol_from_arguments(
    arguments, network_seed: int
) -> WorkingMemoryProtocolConfig:
    protocol = WorkingMemoryProtocolConfig.upstream_run_protocol(seed=network_seed)
    if arguments.item is not None:
        protocol = replace(protocol, item_loading=tuple(arguments.item))
    if arguments.readout is not None:
        protocol = replace(
            protocol, nonspecific_readout_origins_ms=tuple(arguments.readout)
        )
    if arguments.random_noise is not None:
        protocol = replace(protocol, random_nonspecific=tuple(arguments.random_noise))
    if arguments.periodic is not None:
        protocol = replace(protocol, periodic_intervals=tuple(arguments.periodic))
    if arguments.current_input:
        protocol = replace(protocol, poisson_input=False)
    return protocol


def _add_simulation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--connectivity", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--duration-ms", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--current-input", action="store_true")
    parser.add_argument("--item", action="append", type=_parse_item)
    parser.add_argument("--readout", action="append", type=float)
    parser.add_argument("--random-noise", action="append", type=_parse_random)
    parser.add_argument("--periodic", action="append", type=_parse_interval)
    parser.add_argument("--spike-start-ms", type=float, default=100.0)
    parser.add_argument("--recording-fraction", type=float, default=1.0)
    parser.add_argument("--stsp-start-ms", type=float, default=100.0)
    parser.add_argument("--stsp-interval-ms", type=float, default=5.0)
    parser.add_argument("--stsp-source-fraction", type=float, default=0.10)
    parser.add_argument("--stsp-edges-per-population", type=int, default=2_048)
    parser.add_argument("--target-population", type=int, default=0)
    parser.add_argument("--evaluation-start-ms", type=float, default=3_350.0)
    parser.add_argument("--evaluation-stop-ms", type=float, default=5_200.0)
    parser.add_argument("--minimum-target-rate-hz", type=float, default=3.0)
    parser.add_argument("--minimum-margin-hz", type=float, default=1.0)
    parser.add_argument("--progress-interval-steps", type=int, default=10_000)


def _simulate(arguments, graph: SparseRecurrentConnectivity) -> dict:
    populations = tuple(range(graph.config.n_memories))
    return run_simulation(
        graph,
        arguments.output_directory,
        protocol=_protocol_from_arguments(arguments, graph.config.seed),
        spike_recording=SpikeRecordingConfig(
            populations=populations,
            fraction_per_population=arguments.recording_fraction,
            start_ms=arguments.spike_start_ms,
        ),
        stsp_recording=StspProbeRecordingConfig(
            populations=populations,
            source_fraction_per_population=arguments.stsp_source_fraction,
            max_edges_per_population=arguments.stsp_edges_per_population,
            start_ms=arguments.stsp_start_ms,
            snapshot_interval_ms=arguments.stsp_interval_ms,
        ),
        evaluation=TaskEvaluationConfig(
            target_population=arguments.target_population,
            window_start_ms=arguments.evaluation_start_ms,
            window_stop_ms=arguments.evaluation_stop_ms,
            minimum_target_rate_hz=arguments.minimum_target_rate_hz,
            minimum_margin_hz=arguments.minimum_margin_hz,
        ),
        runtime=SimulationRunConfig(
            device=arguments.device,
            dtype=arguments.dtype,
            progress_interval_steps=arguments.progress_interval_steps,
        ),
        duration_ms=arguments.duration_ms,
        overwrite=arguments.overwrite,
        connectivity_path=arguments.connectivity,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and run artifact-oriented recurrent STSP simulations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-connectivity")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--reuse", choices=("require", "auto", "rebuild"), default="auto")
    _add_network_options(build)

    simulate = subparsers.add_parser("simulate")
    _add_simulation_options(simulate)

    workflow = subparsers.add_parser("workflow")
    _add_network_options(workflow)
    _add_simulation_options(workflow)
    workflow.add_argument(
        "--reuse-connectivity",
        choices=("require", "auto", "rebuild"),
        default="require",
    )
    workflow.add_argument("--plot", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("run_directory", type=Path)
    evaluate.add_argument("--target-population", type=int)
    evaluate.add_argument("--evaluation-start-ms", type=float)
    evaluate.add_argument("--evaluation-stop-ms", type=float)
    evaluate.add_argument("--minimum-target-rate-hz", type=float)
    evaluate.add_argument("--minimum-margin-hz", type=float)

    plot = subparsers.add_parser("plot")
    plot.add_argument("run_directory", type=Path)
    plot.add_argument("--output-directory", type=Path)
    plot.add_argument("--max-raster-points", type=int, default=1_000_000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "build-connectivity":
        graph = load_or_build_connectivity(
            arguments.output,
            config=_network_from_arguments(arguments),
            reuse=arguments.reuse,
        )
        result = {
            "path": str(arguments.output.resolve()),
            "neurons": graph.config.n_neurons,
            "edges": graph.num_edges,
            "plastic_edges": graph.plastic.num_edges,
            "static_edges": graph.static.num_edges,
        }
    elif arguments.command == "simulate":
        graph = SparseRecurrentConnectivity.load(arguments.connectivity)
        result = _simulate(arguments, graph)
    elif arguments.command == "workflow":
        graph = load_or_build_connectivity(
            arguments.connectivity,
            config=_network_from_arguments(arguments),
            reuse=arguments.reuse_connectivity,
        )
        result = _simulate(arguments, graph)
        if arguments.plot:
            result["plots"] = plot_run_artifacts(arguments.output_directory)
    elif arguments.command == "evaluate":
        evaluation_values = (
            arguments.target_population,
            arguments.evaluation_start_ms,
            arguments.evaluation_stop_ms,
            arguments.minimum_target_rate_hz,
            arguments.minimum_margin_hz,
        )
        if all(value is None for value in evaluation_values):
            evaluation = None
        else:
            config_path = arguments.run_directory / "run_config.json"
            with config_path.open("r", encoding="utf-8") as handle:
                stored_evaluation = json.load(handle)["evaluation"]
            evaluation = TaskEvaluationConfig(
                target_population=(
                    stored_evaluation["target_population"]
                    if arguments.target_population is None
                    else arguments.target_population
                ),
                window_start_ms=(
                    stored_evaluation["window_start_ms"]
                    if arguments.evaluation_start_ms is None
                    else arguments.evaluation_start_ms
                ),
                window_stop_ms=(
                    stored_evaluation["window_stop_ms"]
                    if arguments.evaluation_stop_ms is None
                    else arguments.evaluation_stop_ms
                ),
                minimum_target_rate_hz=(
                    stored_evaluation["minimum_target_rate_hz"]
                    if arguments.minimum_target_rate_hz is None
                    else arguments.minimum_target_rate_hz
                ),
                minimum_margin_hz=(
                    stored_evaluation["minimum_margin_hz"]
                    if arguments.minimum_margin_hz is None
                    else arguments.minimum_margin_hz
                ),
            )
        result = evaluate_run(arguments.run_directory, evaluation)
    else:
        result = plot_run_artifacts(
            arguments.run_directory,
            output_directory=arguments.output_directory,
            max_raster_points=arguments.max_raster_points,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
