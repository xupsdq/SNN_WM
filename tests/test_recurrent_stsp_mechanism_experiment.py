import json

import torch

from src.experiments.recurrent_stsp import (
    ExternalInputEngine,
    ItemLoadingSignal,
    MatchedQueryExperimentConfig,
    PlasticStateSnapshot,
    SimulationRunConfig,
    SparseRecurrentNetwork,
    TiddiaNetworkConfig,
    WorkingMemoryProtocolConfig,
    capture_network_checkpoint,
    capture_plastic_state,
    generate_sparse_connectivity,
    replace_plastic_state,
    restore_network_checkpoint,
    run_matched_query_substitution,
)


def _config():
    return TiddiaNetworkConfig(
        n_exc=40,
        n_inh=10,
        n_memories=3,
        coding_fraction=0.20,
        connection_probability=0.20,
        delay_min_ms=0.10,
        delay_max_ms=0.10,
        seed=777,
    )


def _step(network, external, event_step):
    external_ex, external_in, external_current = external.pop_current()
    result = network.step(
        external_spikes_ex_pa=external_ex,
        external_spikes_in_pa=external_in,
        external_current_0_pa=external_current,
    )
    external.emit(event_step)
    external.advance()
    return result


def test_network_and_external_checkpoint_replay_is_exact():
    config = _config()
    graph = generate_sparse_connectivity(config)
    protocol = WorkingMemoryProtocolConfig(
        total_time_ms=20.0,
        background_stop_ms=20.0,
        item_loading=(ItemLoadingSignal(0, 2.0),),
        cue_duration_ms=3.0,
        eta_end_origin_ms=21.0,
        seed=config.seed,
    )
    network = SparseRecurrentNetwork(graph, device="cpu")
    external = ExternalInputEngine(config, protocol, device="cpu")
    for event_step in range(1, 101):
        _step(network, external, event_step)
    network_checkpoint = capture_network_checkpoint(network)
    external_checkpoint = external.state_dict()

    first_pass = []
    for event_step in range(101, 141):
        result = _step(network, external, event_step)
        first_pass.append((result.spikes.clone(), result.voltage_mv.clone()))
    first_plastic = capture_plastic_state(network)

    restore_network_checkpoint(network, network_checkpoint)
    external.load_state_dict(external_checkpoint)
    second_pass = []
    for event_step in range(101, 141):
        result = _step(network, external, event_step)
        second_pass.append((result.spikes.clone(), result.voltage_mv.clone()))
    second_plastic = capture_plastic_state(network)

    for first, second in zip(first_pass, second_pass):
        assert torch.equal(first[0], second[0])
        assert torch.equal(first[1], second[1])
    assert torch.equal(first_plastic.u, second_plastic.u)
    assert torch.equal(first_plastic.x, second_plastic.x)
    assert torch.equal(
        first_plastic.last_spike_time_ms, second_plastic.last_spike_time_ms
    )


def test_stsp_substitution_leaves_neurons_and_delay_buffers_unchanged():
    config = _config()
    graph = generate_sparse_connectivity(config)
    network = SparseRecurrentNetwork(graph, device="cpu")
    recipient = capture_network_checkpoint(network)
    donor = PlasticStateSnapshot(
        time_ms=recipient.time_ms,
        u=torch.full_like(recipient.plastic.u, 0.75),
        x=torch.full_like(recipient.plastic.x, 0.25),
        last_spike_time_ms=recipient.plastic.last_spike_time_ms.clone(),
    )
    voltage_before = network.exc_state.v_m_relative.clone()
    recurrent_before = network.scheduler.delay_buffer.excitatory.clone()
    replace_plastic_state(network, donor)
    assert torch.equal(network.exc_state.v_m_relative, voltage_before)
    assert torch.equal(network.scheduler.delay_buffer.excitatory, recurrent_before)
    assert torch.equal(network.scheduler.plastic_state.u, donor.u)
    assert torch.equal(network.scheduler.plastic_state.x, donor.x)


def test_matched_query_experiment_writes_branch_dag(tmp_path):
    config = _config()
    graph = generate_sparse_connectivity(config)
    output = tmp_path / "matched_query"
    summary = run_matched_query_substitution(
        graph,
        output,
        experiment=MatchedQueryExperimentConfig(
            history_populations=(0, 1),
            query_population=2,
            history_origin_ms=1.0,
            query_origin_ms=6.0,
            cue_duration_ms=2.0,
            total_time_ms=10.0,
            delay_readout_start_ms=4.0,
            response_window_start_ms=6.0,
            response_window_stop_ms=10.0,
            stsp_sample_edges=8,
        ),
        runtime=SimulationRunConfig(
            device="cpu", source_chunk_size=16, progress_interval_steps=0
        ),
    )
    assert summary["status"] == "complete"
    for filename in (
        "experiment_config.json",
        "branches.json",
        "metrics.json",
        "state_samples.pt",
        "run_info.json",
        "artifact_manifest.json",
    ):
        assert (output / filename).is_file()
    with (output / "branches.json").open(encoding="utf-8") as handle:
        branches = json.load(handle)
    assert set(branches) == {
        "history0_from_stsp0",
        "history1_from_stsp1",
        "history0_from_stsp1",
        "history1_from_stsp0",
    }
    with (output / "artifact_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    task = manifest["tasks"]["matched_query_simulation"]
    assert task["depends_on"][0] == "experiment_config.json"
    assert task["outputs"] == [
        "branches.json",
        "state_samples.pt",
        "metrics.json",
        "run_info.json",
    ]
