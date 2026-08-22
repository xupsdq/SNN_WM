import json

import pytest
import torch

from src.experiments.recurrent_stsp import (
    ContinuousStspProbeRecorder,
    ExternalInputEngine,
    ItemLoadingSignal,
    PeriodicReadoutInterval,
    RandomNonspecificSignal,
    SparseRecurrentNetwork,
    SpikeRecordingConfig,
    StspProbeRecordingConfig,
    TaskEvaluationConfig,
    TiddiaNetworkConfig,
    WorkingMemoryProtocolConfig,
    evaluate_run,
    evaluate_task,
    generate_sparse_connectivity,
    noise_current_parameters,
    poisson_rate_and_weight,
    run_simulation,
)
from src.experiments.recurrent_stsp.plot_artifacts import plot_run_artifacts
from src.experiments.recurrent_stsp.runner import SimulationRunConfig


def _small_config(**overrides):
    values = dict(
        n_exc=40,
        n_inh=10,
        n_memories=2,
        coding_fraction=0.20,
        connection_probability=0.20,
        delay_min_ms=0.10,
        delay_max_ms=0.10,
        seed=123,
    )
    values.update(overrides)
    return TiddiaNetworkConfig(**values)


def test_upstream_external_parameter_formulas_are_preserved():
    mean_pa, std_pa = noise_current_parameters(23.7, 1.0, 15.0)
    assert mean_pa == pytest.approx(395.0)
    assert std_pa == pytest.approx((2.0 / 15.0) ** 0.5 * 250.0)

    rate_hz, weight_pa = poisson_rate_and_weight(23.7, 1.0, 15.0)
    expected_rate_per_ms = 23.7**2 / (2.0 * (15.0 + 2.0))
    assert rate_hz == pytest.approx(expected_rate_per_ms * 1_000.0)
    assert weight_pa == pytest.approx(395.0 / (expected_rate_per_ms * 2.0))


def test_every_upstream_optional_input_family_compiles():
    config = _small_config()
    protocol = WorkingMemoryProtocolConfig(
        total_time_ms=1_000.0,
        background_stop_ms=1_000.0,
        item_loading=(ItemLoadingSignal(1, 100.0),),
        nonspecific_readout_origins_ms=(200.0,),
        random_nonspecific=(RandomNonspecificSignal(300.0, 0.25),),
        periodic_intervals=(PeriodicReadoutInterval(400.0, 1_000.0),),
        eta_end_origin_ms=900.0,
    )
    engine = ExternalInputEngine(config, protocol, device="cpu")
    descriptions = {item.name: item for item in engine.descriptions}
    assert set(descriptions) == {
        "background_exc",
        "background_inh",
        "late_exc_offset",
        "item_loading_0",
        "nonspecific_readout_0",
        "random_nonspecific_0",
        "periodic_readout_0",
        "periodic_readout_1",
    }
    assert descriptions["item_loading_0"].target_count == config.selective_population_size
    assert descriptions["random_nonspecific_0"].target_count == int(0.25 * config.n_exc)


def test_current_generator_delay_and_neuron_step_current_order():
    config = _small_config(connection_probability=0.0)
    graph = generate_sparse_connectivity(config)
    network = SparseRecurrentNetwork(graph, device="cpu")
    initial = network.exc_state.absolute_voltage(network.exc_params).clone()
    first = network.step(external_current_0_pa=10.0)
    second = network.step()
    assert torch.equal(first.voltage_mv[: config.n_exc], initial)
    assert bool((second.voltage_mv[: config.n_exc] > initial).all())

    protocol = WorkingMemoryProtocolConfig(
        total_time_ms=0.5,
        background_stop_ms=0.5,
        poisson_input=False,
        sigma_exc_mv=0.0,
        sigma_inh_mv=0.0,
        item_loading=(),
        eta_end_origin_ms=1.0,
    )
    engine = ExternalInputEngine(config, protocol, device="cpu")
    delivered = []
    for event_step in range(1, 4):
        _, _, current = engine.pop_current()
        delivered.append(current)
        engine.emit(event_step)
        engine.advance()
    assert torch.count_nonzero(delivered[0]) == 0
    assert torch.count_nonzero(delivered[1]) == 0
    assert torch.count_nonzero(delivered[2]) == config.n_neurons

    double_engine = ExternalInputEngine(
        config, protocol, device="cpu", dtype=torch.float64
    )
    double_engine.emit(1)
    double_engine.advance()
    double_engine.emit(2)
    double_engine.advance()
    _, _, double_current = double_engine.pop_current()
    assert double_current.dtype == torch.float64


def test_stsp_probe_analytically_recovers_between_spikes():
    config = _small_config()
    graph = generate_sparse_connectivity(config)
    network = SparseRecurrentNetwork(graph, device="cpu")
    recorder = ContinuousStspProbeRecorder(
        network.scheduler,
        StspProbeRecordingConfig(
            populations=(0,),
            source_fraction_per_population=1.0,
            max_edges_per_population=4,
            start_ms=0.0,
            snapshot_interval_ms=1.0,
        ),
    )
    edge_id = int(recorder.edge_ids[0])
    network.scheduler.plastic_state.u[edge_id] = 0.50
    network.scheduler.plastic_state.x[edge_id] = 0.20
    network.scheduler.plastic_state.last_spike_time_ms[edge_id] = 0.0
    recorder.record(20)
    payload = recorder.payload()
    expected_u = config.stsp_u + (0.50 - config.stsp_u) * torch.exp(
        torch.tensor(-1.0 / float(recorder.tau_fac[0]))
    )
    expected_x = 1.0 + (0.20 - 1.0) * torch.exp(
        torch.tensor(-1.0 / float(recorder.tau_rec[0]))
    )
    assert payload["u"][0, 0] == pytest.approx(float(expected_u))
    assert payload["x"][0, 0] == pytest.approx(float(expected_x))
    assert payload["ux"][0, 0] == pytest.approx(float(expected_u * expected_x))


def test_decoder_has_explicit_success_condition():
    payload = {
        "times_ms": torch.tensor([2.0, 3.0, 4.0, 2.5]),
        "population_ids": torch.tensor([1, 1, 1, 0]),
        "recorded_neuron_population_ids": torch.tensor([0, 0, 1, 1]),
    }
    result = evaluate_task(
        payload,
        TaskEvaluationConfig(
            target_population=1,
            window_start_ms=1.0,
            window_stop_ms=5.0,
            minimum_target_rate_hz=100.0,
            minimum_margin_hz=100.0,
        ),
    )
    assert result["winner_population"] == 1
    assert result["target_rate_hz"] == pytest.approx(375.0)
    assert result["target_margin_hz"] == pytest.approx(250.0)
    assert result["success"] is True


def test_artifact_dag_smoke_and_plot_only_leaf(tmp_path):
    config = _small_config()
    graph = generate_sparse_connectivity(config)
    graph_path = tmp_path / "graph.pt"
    graph.save(graph_path)
    run_dir = tmp_path / "run"
    protocol = WorkingMemoryProtocolConfig(
        total_time_ms=5.0,
        background_stop_ms=5.0,
        poisson_input=False,
        sigma_exc_mv=0.0,
        sigma_inh_mv=0.0,
        item_loading=(),
        eta_end_origin_ms=10.0,
    )
    summary = run_simulation(
        graph,
        run_dir,
        protocol=protocol,
        spike_recording=SpikeRecordingConfig(
            populations=(0, 1), start_ms=0.0, chunk_steps=16
        ),
        stsp_recording=StspProbeRecordingConfig(
            populations=(0, 1),
            source_fraction_per_population=1.0,
            max_edges_per_population=4,
            start_ms=0.0,
            snapshot_interval_ms=1.0,
        ),
        evaluation=TaskEvaluationConfig(
            target_population=0,
            window_start_ms=1.0,
            window_stop_ms=4.0,
            minimum_target_rate_hz=0.0,
            minimum_margin_hz=0.0,
        ),
        runtime=SimulationRunConfig(
            device="cpu", progress_interval_steps=0, source_chunk_size=16
        ),
        connectivity_path=graph_path,
    )
    assert summary["status"] == "complete"
    assert summary["steps"] == 100
    for relative in (
        "run_config.json",
        "data/spikes.pt",
        "data/stsp_probes.pt",
        "metrics/task_metrics.json",
        "meta/run_info.json",
        "artifact_manifest.json",
        "summary.json",
    ):
        assert (run_dir / relative).is_file()
    with (run_dir / "artifact_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["tasks"]["plot"]["plot_only"] is True
    reevaluated = evaluate_run(run_dir)
    assert reevaluated["evaluation_config"]["window_start_ms"] == 1.0

    plots = plot_run_artifacts(run_dir, max_raster_points=100)
    assert plots["plot_only"] is True
    assert len(plots["raster"]["outputs"]) == 3
    assert len(plots["stsp_ux"]["outputs"]) == 3
