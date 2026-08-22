import json
from dataclasses import replace

import pytest
import torch

from src.experiments.recurrent_stsp import (
    DmsExperimentConfig,
    DmsNetworkEntry,
    SimulationRunConfig,
    SparseRecurrentNetwork,
    TiddiaNetworkConfig,
    analyze_dms_trial_features,
    aggregate_dms_network_results,
    build_dms_trial_manifest,
    fit_ridge_decoder,
    generate_sparse_connectivity,
    reset_plastic_state_to_no_event_baseline,
    run_dms_trial_simulation,
    write_dms_network_manifest,
)


def _dms_config():
    return DmsExperimentConfig(
        task_populations=(0, 1, 2),
        distractor_population=3,
        sample_origin_ms=1.0,
        cue_duration_ms=1.0,
        delays_ms=(2.0,),
        distractor_conditions=(False,),
        response_bin_ms=1.0,
        response_bin_count=2,
        silent_window_ms=1.0,
        pairs_per_probe=(2, 2, 2),
        stsp_edges_per_source_population=8,
        seed=91,
    )


def _network_config():
    return TiddiaNetworkConfig(
        n_exc=60,
        n_inh=15,
        n_memories=4,
        coding_fraction=0.20,
        connection_probability=0.20,
        delay_min_ms=0.10,
        delay_max_ms=0.10,
        seed=411,
    )


def test_dms_manifest_balances_current_probe_sample_and_labels():
    config = _dms_config()
    trials = build_dms_trial_manifest(config)
    assert len(trials) == 36
    for split in ("train", "validation", "test"):
        subset = [trial for trial in trials if trial.split == split]
        assert sum(trial.is_match for trial in subset) == len(subset) // 2
        for population in config.task_populations:
            by_probe = [trial for trial in subset if trial.probe_population == population]
            by_sample = [trial for trial in subset if trial.sample_population == population]
            assert sum(trial.is_match for trial in by_probe) == len(by_probe) // 2
            assert sum(trial.is_match for trial in by_sample) == len(by_sample) // 2
    for pair_id in {trial.pair_id for trial in trials}:
        pair = [trial for trial in trials if trial.pair_id == pair_id]
        assert len(pair) == 2
        assert pair[0].input_seed == pair[1].input_seed
        assert pair[0].probe_population == pair[1].probe_population
        assert pair[0].is_match != pair[1].is_match


def test_ridge_decoder_never_uses_test_rows_for_fit_or_selection():
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


def test_no_event_stsp_reset_preserves_nonplastic_network_state():
    graph = generate_sparse_connectivity(_network_config())
    network = SparseRecurrentNetwork(graph, device="cpu")
    network.scheduler.plastic_state.u.add_(0.2)
    network.scheduler.plastic_state.x.mul_(0.5)
    voltage = network.exc_state.v_m_relative.clone()
    recurrent = network.scheduler.delay_buffer.excitatory.clone()
    reset_plastic_state_to_no_event_baseline(network)
    assert torch.equal(network.exc_state.v_m_relative, voltage)
    assert torch.equal(network.scheduler.delay_buffer.excitatory, recurrent)
    assert torch.all(network.scheduler.plastic_state.u <= 1.0)
    assert torch.all(network.scheduler.plastic_state.x <= 1.0)


def test_dms_simulation_and_decoder_are_separate_dag_tasks(tmp_path):
    graph = generate_sparse_connectivity(_network_config())
    output = tmp_path / "dms"
    simulation = run_dms_trial_simulation(
        graph,
        output,
        experiment=_dms_config(),
        runtime=SimulationRunConfig(
            device="cpu", source_chunk_size=16, progress_interval_steps=0
        ),
        progress_interval_pairs=0,
    )
    assert simulation["trial_count"] == 36
    assert (output / "trial_features.pt").is_file()
    assert not (output / "analysis_metrics.json").exists()
    with (output / "artifact_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["tasks"]["decoder_analysis"]["status"] == (
        "not-run-by-simulation-task"
    )

    metrics = analyze_dms_trial_features(output)
    assert (output / "decoders.pt").is_file()
    assert (output / "analysis_metrics.json").is_file()
    assert set(metrics["gates"]) == {
        "behavior_feasibility",
        "activity_silent_pattern",
        "reset_disrupts_behavior",
        "swap_moves_toward_donor",
    }
    with (output / "artifact_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["tasks"]["decoder_analysis"]["status"] == "complete"


def test_multinetwork_aggregation_verifies_graphs_and_uses_one_row_each(tmp_path):
    root = tmp_path / "replication"
    root.mkdir()
    experiment_config_path = root / "experiment_config.json"
    experiment_config_path.write_text(
        json.dumps(_dms_config().as_dict()), encoding="utf-8"
    )
    entries = []
    behavior_values = (0.75, 0.85, 0.95)
    for seed, behavior in zip((411, 412, 413), behavior_values):
        graph_path = root / "graph_{}.pt".format(seed)
        generate_sparse_connectivity(replace(_network_config(), seed=seed)).save(
            graph_path
        )
        run_directory = root / "run_{}".format(seed)
        run_directory.mkdir()
        (run_directory / "experiment_config.json").write_text(
            json.dumps(_dms_config().as_dict()), encoding="utf-8"
        )
        metrics = {
            "behavior_decoder": {"test": {"balanced_accuracy": behavior}},
            "probe_only_control_decoder": {
                "test": {"balanced_accuracy": 0.5}
            },
            "delay_firing_sample_decoder": {
                "test": {"balanced_accuracy": 1.0 / 3.0}
            },
            "pre_query_stsp_sample_decoder": {
                "test": {"balanced_accuracy": 1.0}
            },
            "post_query_stsp_match_decoder": {
                "test": {"balanced_accuracy": 0.9}
            },
            "causal_controls": {
                "reset_test_balanced_accuracy": 0.5,
                "swap_accuracy_against_donor_label": 0.8,
                "mean_swap_donor_projection": 0.9,
                "mean_swap_donor_directed_score_change": 0.4,
            },
            "mean_delay_population_rate_hz": 0.6,
            "gates": {
                "behavior_feasibility": True,
                "activity_silent_pattern": True,
                "reset_disrupts_behavior": True,
                "swap_moves_toward_donor": True,
            },
            "all_pilot_gates_pass": True,
        }
        (run_directory / "analysis_metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        entries.append(
            DmsNetworkEntry(seed, str(graph_path), str(run_directory))
        )
    manifest_path = root / "network_manifest.json"
    write_dms_network_manifest(
        manifest_path,
        entries,
        experiment_config_path=experiment_config_path,
    )
    aggregate = aggregate_dms_network_results(
        manifest_path,
        output_directory=root / "aggregate",
    )
    assert aggregate["network_count"] == 3
    assert aggregate["all_pilot_gates_pass_count"] == 3
    assert aggregate["three_graph_causal_role_replication_supported"] is True
    assert aggregate["fig4_sequential_mechanism_tested"] is False
    assert aggregate["endpoint_summaries"]["behavior_balanced_accuracy"][
        "mean"
    ] == 0.85

    with manifest_path.open(encoding="utf-8") as handle:
        invalid_manifest = json.load(handle)
    invalid_manifest["networks"][1]["graph_seed"] = invalid_manifest["networks"][0][
        "graph_seed"
    ]
    manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        aggregate_dms_network_results(
            manifest_path,
            output_directory=root / "invalid_aggregate",
        )
