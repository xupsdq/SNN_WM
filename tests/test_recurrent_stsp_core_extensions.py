import torch

from src.experiments.recurrent_stsp import (
    ExternalInputEngine,
    IafPscExpParameters,
    IndexedPlasticState,
    ItemLoadingSignal,
    PatternedAssemblySignal,
    SparseEventScheduler,
    SparseRecurrentNetwork,
    TiddiaNetworkConfig,
    WorkingMemoryProtocolConfig,
    capture_indexed_plastic_state,
    capture_network_checkpoint,
    capture_raw_indexed_plastic_state,
    event_induced_release_delta,
    fit_ridge_decoder,
    generate_sparse_connectivity,
    iaf_psc_exp_step,
    indexed_next_release,
    make_iaf_psc_exp_state,
    network_checkpoint_from_dict,
    network_checkpoint_to_dict,
    passive_evolve_indexed_state,
    replace_indexed_plastic_state,
    replay_indexed_plastic_state,
    replay_raw_indexed_plastic_state,
    restore_network_checkpoint,
)


def _small_graph():
    return generate_sparse_connectivity(
        TiddiaNetworkConfig(
            n_exc=80,
            n_inh=20,
            n_memories=5,
            coding_fraction=0.1,
            connection_probability=0.2,
            allow_multapses=False,
            dt_ms=1.0,
            delay_min_ms=1.0,
            delay_max_ms=2.0,
            seed=7_041,
        )
    )


def _plastic_source_and_edges(graph):
    source = next(
        index
        for index in range(graph.config.n_exc)
        if int(graph.plastic.row_ptr[index + 1] - graph.plastic.row_ptr[index]) > 0
    )
    first = int(graph.plastic.row_ptr[source].item())
    stop = int(graph.plastic.row_ptr[source + 1].item())
    return source, torch.arange(first, stop, dtype=torch.int64)


def test_patterned_signal_replays_exact_psc_events():
    graph = _small_graph()
    signal = PatternedAssemblySignal(
        name="frozen",
        origin_ms=0.0,
        target_neuron_ids=(0, 1),
        event_steps_relative=(1, 1, 2),
        event_target_indices=(0, 1, 0),
        event_amplitudes_pa=(3.0, 5.0, 7.0),
        target_delay_steps=(1, 2),
        stream_id=4,
    )
    protocol = WorkingMemoryProtocolConfig(
        total_time_ms=4.0,
        background_stop_ms=4.0,
        poisson_input=True,
        item_loading=(),
        patterned_loading=(signal,),
        eta_end_origin_ms=5.0,
        seed=4,
    )
    engine = ExternalInputEngine(graph.config, protocol, dtype=torch.float64)
    suppressed = ("background_exc", "background_inh", "late_exc_offset")

    engine.emit(1, suppressed_signal_names=suppressed)
    engine.advance()
    excitatory, _, _ = engine.pop_current()
    assert excitatory[0] == 3.0
    assert excitatory[1] == 0.0

    engine.emit(2, suppressed_signal_names=suppressed)
    engine.advance()
    excitatory, _, _ = engine.pop_current()
    assert excitatory[0] == 7.0
    assert excitatory[1] == 5.0


def test_external_input_checkpoint_restores_rng_and_delayed_events():
    graph = _small_graph()
    protocol = WorkingMemoryProtocolConfig(
        total_time_ms=8.0,
        background_stop_ms=8.0,
        item_loading=(ItemLoadingSignal(0, 1.0, stream_id=0),),
        cue_duration_ms=4.0,
        eta_end_origin_ms=9.0,
        seed=99,
    )
    reference = ExternalInputEngine(graph.config, protocol, dtype=torch.float64)
    restored = ExternalInputEngine(graph.config, protocol, dtype=torch.float64)
    reference.emit(2)
    state = reference.state_dict(storage_device="cpu")
    restored.load_state_dict(state)

    reference.emit(3)
    restored.emit(3)
    reference_state = reference.state_dict(storage_device="cpu")
    restored_state = restored.state_dict(storage_device="cpu")

    for left, right in zip(reference_state["signals"], restored_state["signals"]):
        assert torch.equal(left["generator_state"], right["generator_state"])
    assert torch.equal(
        reference_state["spike_buffer"]["excitatory"],
        restored_state["spike_buffer"]["excitatory"],
    )
    assert torch.equal(
        reference_state["current_buffer"]["excitatory"],
        restored_state["current_buffer"]["excitatory"],
    )


def test_network_checkpoint_dict_round_trip_replays_exactly(tmp_path):
    graph = _small_graph()
    source, _ = _plastic_source_and_edges(graph)
    forced = torch.zeros(graph.config.n_neurons, dtype=torch.bool)
    forced[source] = True
    network = SparseRecurrentNetwork(graph, dtype=torch.float64)
    network.step(forced_dispatch_spikes=forced, replace_emitted_spikes=True)
    checkpoint = capture_network_checkpoint(network, storage_device="cpu")
    path = tmp_path / "checkpoint.pt"
    torch.save(network_checkpoint_to_dict(checkpoint), path)
    restored_checkpoint = network_checkpoint_from_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )

    reference = SparseRecurrentNetwork(graph, dtype=torch.float64)
    restored = SparseRecurrentNetwork(graph, dtype=torch.float64)
    restore_network_checkpoint(reference, checkpoint)
    restore_network_checkpoint(restored, restored_checkpoint)
    expected = reference.step()
    actual = restored.step()

    assert torch.equal(expected.spikes, actual.spikes)
    assert torch.equal(reference.exc_state.v_m_relative, restored.exc_state.v_m_relative)
    assert torch.equal(reference.inh_state.v_m_relative, restored.inh_state.v_m_relative)
    assert torch.equal(
        reference.scheduler.plastic_state.u, restored.scheduler.plastic_state.u
    )
    assert torch.equal(
        reference.scheduler.delay_buffer.excitatory,
        restored.scheduler.delay_buffer.excitatory,
    )


def test_indexed_replay_matches_live_scheduler_and_passive_counterfactual():
    graph = _small_graph()
    scheduler = SparseEventScheduler(graph, dtype=torch.float64)
    source, edge_ids = _plastic_source_and_edges(graph)
    edges = scheduler.connectivity.plastic
    start = IndexedPlasticState(
        0.0,
        edge_ids,
        edges.initial_u[edge_ids].clone(),
        edges.initial_x[edge_ids].clone(),
    )
    spikes = torch.zeros(graph.config.n_neurons, dtype=torch.bool)
    spikes[source] = True
    for event_time in (1.0, 3.0):
        scheduler.dispatch_spikes(spikes, time_ms=event_time)
    end_time = 5.0
    state = scheduler.plastic_state
    elapsed = end_time - state.last_spike_time_ms[edge_ids]
    live = IndexedPlasticState(
        end_time,
        edge_ids,
        graph.config.stsp_u
        + (state.u[edge_ids] - graph.config.stsp_u)
        * torch.exp(-elapsed / edges.tau_fac_ms[edge_ids]),
        1.0
        + (state.x[edge_ids] - 1.0)
        * torch.exp(-elapsed / edges.tau_rec_ms[edge_ids]),
    )
    replay = replay_indexed_plastic_state(
        edges,
        start,
        spike_times_ms=torch.tensor([1.0, 3.0], dtype=torch.float64),
        spike_sources=torch.tensor([source, source]),
        end_time_ms=end_time,
        baseline_u=graph.config.stsp_u,
    )
    passive = passive_evolve_indexed_state(
        edges, start, end_time_ms=end_time, baseline_u=graph.config.stsp_u
    )
    delta = event_induced_release_delta(
        edges, start, live, baseline_u=graph.config.stsp_u
    )

    assert torch.allclose(replay.u, live.u)
    assert torch.allclose(replay.x, live.x)
    assert torch.allclose(
        delta,
        indexed_next_release(edges, live, baseline_u=graph.config.stsp_u)
        - indexed_next_release(edges, passive, baseline_u=graph.config.stsp_u),
    )


def test_raw_replay_and_indexed_replacement_preserve_declared_boundaries():
    graph = _small_graph()
    network = SparseRecurrentNetwork(graph, dtype=torch.float64)
    source, edge_ids = _plastic_source_and_edges(graph)
    start = capture_raw_indexed_plastic_state(network, edge_ids)
    forced = torch.zeros(graph.config.n_neurons, dtype=torch.bool)
    forced[source] = True
    event_times = []
    for _ in range(3):
        network.step(forced_dispatch_spikes=forced, replace_emitted_spikes=True)
        event_times.append(network.step_index * graph.config.dt_ms)
    live = capture_indexed_plastic_state(network, edge_ids)
    replay = replay_raw_indexed_plastic_state(
        graph.plastic,
        start,
        spike_times_ms=torch.tensor(event_times, dtype=torch.float64),
        spike_sources=torch.full((3,), source, dtype=torch.int64),
        end_time_ms=live.time_ms,
        baseline_u=graph.config.stsp_u,
    )
    assert torch.equal(replay.u, live.u)
    assert torch.equal(replay.x, live.x)

    untouched = network.scheduler.plastic_state.u.clone()
    selected = edge_ids[: min(3, edge_ids.numel())]
    donor = IndexedPlasticState(
        live.time_ms,
        selected,
        torch.full((selected.numel(),), 0.8, dtype=torch.float64),
        torch.full((selected.numel(),), 0.2, dtype=torch.float64),
    )
    replace_indexed_plastic_state(network, donor)
    changed = torch.zeros_like(untouched, dtype=torch.bool)
    changed[selected] = True
    assert torch.equal(network.scheduler.plastic_state.u[~changed], untouched[~changed])
    assert torch.equal(network.scheduler.plastic_state.u[selected], donor.u)


def test_spike_override_replaces_threshold_decision_without_changing_integration():
    params = IafPscExpParameters(
        tau_m=10.0,
        t_ref=2.0,
        e_l=0.0,
        v_th=1.0,
        v_reset=0.0,
        c_m=1.0,
        tau_syn_ex=2.0,
        tau_syn_in=2.0,
        dt=1.0,
    )
    state = make_iaf_psc_exp_state((2,), params, dtype=torch.float64)
    state.v_m_relative[:] = torch.tensor([2.0, 0.0], dtype=torch.float64)
    next_state, spikes = iaf_psc_exp_step(
        state,
        params,
        spike_override=torch.tensor([False, True]),
    )

    assert torch.equal(spikes, torch.tensor([False, True]))
    assert next_state.v_m_relative[0] > params.v_th - params.e_l
    assert next_state.v_m_relative[1] == params.v_reset - params.e_l


def test_ridge_decoder_does_not_fit_or_select_on_test_rows():
    splits = ["train"] * 8 + ["validation"] * 4 + ["test"] * 4
    labels = torch.tensor([0, 1] * 8)
    features = torch.stack(
        (labels.to(torch.float64), 1.0 - labels.to(torch.float64)), dim=1
    )
    model, metrics = fit_ridge_decoder(features, labels, splits)
    perturbed = features.clone()
    perturbed[-4:] = torch.tensor(
        [[1e6, -1e6], [-1e6, 1e6], [1e6, -1e6], [-1e6, 1e6]],
        dtype=torch.float64,
    )
    second_model, second_metrics = fit_ridge_decoder(perturbed, labels, splits)

    assert torch.equal(model["feature_mean"], second_model["feature_mean"])
    assert torch.equal(model["feature_std"], second_model["feature_std"])
    assert torch.equal(model["weights"], second_model["weights"])
    assert metrics["selected_ridge_lambda"] == second_metrics["selected_ridge_lambda"]
