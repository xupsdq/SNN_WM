import os

import pytest
import torch

from src.experiments.recurrent_stsp import (
    ConnectionBlockRecord,
    PlasticCsrEdges,
    SparseEventScheduler,
    SparseRecurrentConnectivity,
    SparseRecurrentNetwork,
    StaticCsrEdges,
    TiddiaNetworkConfig,
    expected_edge_counts,
    generate_sparse_connectivity,
)


def _small_config(**overrides):
    values = {
        "n_exc": 80,
        "n_inh": 20,
        "n_memories": 2,
        "coding_fraction": 0.25,
        "connection_probability": 0.2,
        "allow_multapses": False,
        "dt_ms": 0.1,
        "delay_min_ms": 0.1,
        "delay_max_ms": 0.3,
        "sampling_target_chunk": 16,
        "seed": 1234,
    }
    values.update(overrides)
    return TiddiaNetworkConfig(**values)


def test_default_and_upstream_heterogeneous_configs_have_exact_20m_edges():
    expected = {"plastic": 12_800_000, "static": 7_200_000, "total": 20_000_000}
    assert expected_edge_counts(TiddiaNetworkConfig()) == expected
    assert expected_edge_counts(TiddiaNetworkConfig.heterogeneous_run_config()) == expected


def test_small_fixed_indegree_graph_is_deterministic_and_source_major():
    config = _small_config()
    first = generate_sparse_connectivity(config)
    second = generate_sparse_connectivity(config)

    assert first.num_edges == 1_920
    assert first.plastic.num_edges == 1_200
    assert first.static.num_edges == 720
    assert torch.equal(first.plastic.row_ptr, second.plastic.row_ptr)
    assert torch.equal(first.plastic.targets, second.plastic.targets)
    assert torch.equal(first.plastic.delay_steps, second.plastic.delay_steps)
    assert torch.equal(first.static.row_ptr, second.static.row_ptr)
    assert torch.equal(first.static.targets, second.static.targets)

    plastic_indegree = torch.bincount(
        first.plastic.targets.to(torch.int64), minlength=config.n_neurons
    )
    static_indegree = torch.bincount(
        first.static.targets.to(torch.int64), minlength=config.n_neurons
    )
    assert torch.equal(plastic_indegree[: config.n_exc], torch.full((80,), 15))
    assert torch.equal(static_indegree[: config.n_exc], torch.full((80,), 4))
    assert torch.equal(static_indegree[config.n_exc :], torch.full((20,), 20))
    assert int(first.plastic.row_ptr[config.n_exc].item()) == first.plastic.num_edges
    assert torch.all(first.plastic.row_ptr[1:] >= first.plastic.row_ptr[:-1])


def test_connectivity_artifact_round_trip(tmp_path):
    graph = generate_sparse_connectivity(_small_config())
    path = graph.save(tmp_path / "connectivity.pt")
    loaded = SparseRecurrentConnectivity.load(path)

    assert loaded.config == graph.config
    assert loaded.blocks == graph.blocks
    assert torch.equal(loaded.plastic.targets, graph.plastic.targets)
    assert torch.equal(loaded.plastic.initial_u, graph.plastic.initial_u)
    assert torch.equal(loaded.static.weights, graph.static.weights)


def test_scaled_heterogeneous_run_config_generates_bounded_edge_parameters():
    config = TiddiaNetworkConfig.heterogeneous_run_config(
        n_exc=80,
        n_inh=20,
        n_memories=2,
        coding_fraction=0.25,
        connection_probability=0.2,
        dt_ms=0.1,
        delay_min_ms=0.1,
        delay_max_ms=0.3,
        sampling_target_chunk=16,
        seed=4321,
    )
    graph = generate_sparse_connectivity(config)

    assert graph.num_edges == expected_edge_counts(config)["total"]
    assert torch.all((graph.plastic.initial_u >= 0) & (graph.plastic.initial_u <= 1))
    assert torch.all((graph.plastic.initial_x >= 0) & (graph.plastic.initial_x <= 1))
    assert torch.std(graph.plastic.initial_u) > 0
    assert torch.std(graph.plastic.initial_x) > 0
    assert torch.std(graph.plastic.weights) > 0


def _two_edge_graph():
    config = TiddiaNetworkConfig(
        n_exc=2,
        n_inh=1,
        n_memories=1,
        coding_fraction=0.5,
        connection_probability=0.5,
        dt_ms=0.1,
        delay_min_ms=0.1,
        delay_max_ms=0.2,
        stsp_u=0.2,
        stsp_initial_u=0.2,
    )
    plastic = PlasticCsrEdges(
        row_ptr=torch.tensor([0, 1, 1, 1], dtype=torch.int64),
        targets=torch.tensor([1], dtype=torch.int32),
        weights=torch.tensor([100.0], dtype=torch.float64),
        delay_steps=torch.tensor([2], dtype=torch.int16),
        initial_u=torch.tensor([0.2], dtype=torch.float64),
        initial_x=torch.tensor([1.0], dtype=torch.float64),
        tau_rec_ms=torch.tensor([200.0], dtype=torch.float64),
        tau_fac_ms=torch.tensor([1500.0], dtype=torch.float64),
    )
    static = StaticCsrEdges(
        row_ptr=torch.tensor([0, 0, 0, 1], dtype=torch.int64),
        targets=torch.tensor([1], dtype=torch.int32),
        weights=torch.tensor([-50.0], dtype=torch.float64),
        delay_steps=torch.tensor([1], dtype=torch.int16),
    )
    blocks = (
        ConnectionBlockRecord(
            name="plastic",
            plastic=True,
            source_start=0,
            source_count=1,
            target_start=1,
            target_count=1,
            indegree=1,
            edge_count=1,
            weight_mode="manual",
            facilitated=True,
        ),
        ConnectionBlockRecord(
            name="static",
            plastic=False,
            source_start=2,
            source_count=1,
            target_start=1,
            target_count=1,
            indegree=1,
            edge_count=1,
            weight_mode="manual",
            facilitated=False,
        ),
    )
    graph = SparseRecurrentConnectivity(config, plastic, static, blocks)
    graph.validate()
    return graph


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_sparse_scheduler_updates_stsp_and_delivers_each_delay_slot(device):
    scheduler = SparseEventScheduler(_two_edge_graph(), device=device, dtype=torch.float64)
    spikes = torch.tensor([True, False, True], device=device)

    stats = scheduler.dispatch_spikes(spikes, time_ms=10.0)

    assert stats.active_sources == 2
    assert stats.plastic_events == 1
    assert stats.static_events == 1
    assert scheduler.plastic_state.u.item() == pytest.approx(0.36, abs=1e-14)
    assert scheduler.plastic_state.x.item() == pytest.approx(0.64, abs=1e-14)

    scheduler.advance()
    excitatory, inhibitory = scheduler.pop_current()
    assert excitatory[1].item() == 0.0
    assert inhibitory[1].item() == pytest.approx(-50.0, abs=1e-14)

    scheduler.advance()
    excitatory, inhibitory = scheduler.pop_current()
    assert excitatory[1].item() == pytest.approx(36.0, abs=1e-12)
    assert inhibitory[1].item() == 0.0


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_sparse_recurrent_network_closes_the_spike_event_loop(device):
    network = SparseRecurrentNetwork(_two_edge_graph(), device=device, dtype=torch.float64)
    drive = torch.tensor([60_000.0, 0.0, 0.0], dtype=torch.float64, device=device)

    first = network.step(external_current_pa=drive)
    second = network.step()
    third = network.step()
    fourth = network.step()

    assert bool(first.spikes[0].item())
    assert first.dispatch.plastic_events == 1
    assert second.voltage_mv[1].item() == 0.0
    assert third.voltage_mv[1].item() == 0.0
    assert fourth.voltage_mv[1].item() > 0.0


@pytest.mark.skipif(
    os.environ.get("RUN_FULL_SCALE_STSP") != "1",
    reason="Set RUN_FULL_SCALE_STSP=1 for the 10k-neuron/20M-edge allocation test.",
)
def test_full_scale_20m_graph_and_cuda_dispatch():
    graph = generate_sparse_connectivity(TiddiaNetworkConfig())
    assert graph.num_edges == 20_000_000
    assert graph.plastic.num_edges == 12_800_000
    assert graph.static.num_edges == 7_200_000
    assert graph.storage_bytes < 512 * 1024**2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    scheduler = SparseEventScheduler(graph, device=device, dtype=torch.float32)
    spikes = torch.zeros(graph.config.n_neurons, dtype=torch.bool, device=device)
    spikes[torch.tensor([0, 7999, 8000, 9999], device=device)] = True
    stats = scheduler.dispatch_spikes(spikes, time_ms=0.05)

    assert stats.active_sources == 4
    assert stats.total_events > 0
    assert scheduler.storage_bytes < 1024 * 1024**2
    if device == "cuda":
        assert torch.cuda.max_memory_allocated() < 2 * 1024**3
