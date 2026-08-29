"""Stripped Masse-LIF + STSP public behaviour and smoke DAG."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.experiments.masse_delayed_cue_lif.config import (
    NO_STSP_RECURRENT_SCALE,
    STSP_U_FAC,
    stripped_no_stsp_config,
    stripped_stsp_config,
)
from src.experiments.masse_delayed_cue_lif.decode import decode_run, recompute_decode_from_features
from src.experiments.masse_delayed_cue_lif.metrics import weighted_cross_entropy
from src.experiments.masse_delayed_cue_lif.model import RecurrentLifSfa
from src.experiments.masse_delayed_cue_lif.plot import plot_run
from src.experiments.masse_delayed_cue_lif.run import main
from src.experiments.masse_delayed_cue_lif.task import expand_rows, generate_trial_table
from src.experiments.masse_delayed_cue_lif.train import (
    clip_gradients_per_parameter,
    parameter_grads_are_usable,
)


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_stripped(use_stsp: bool):
    builder = stripped_stsp_config if use_stsp else stripped_no_stsp_config
    config = builder(
        device="cpu",
        n_hidden=4,
        n_train=32,
        n_val=32,
        n_test=32,
        batch_size=4,
        max_epochs=1,
    )
    rows = generate_trial_table(config)[:4]
    inputs, targets, weights = expand_rows(rows, config)
    return config, inputs, targets, weights


def test_stripped_no_stsp_has_no_current_sfa_or_syn_state():
    config, inputs, _targets, _weights = _tiny_stripped(False)
    model = RecurrentLifSfa(config)
    logits, state = model(inputs)
    assert logits.shape[0] == config.n_steps
    assert state.current is None
    assert state.adaptation is None
    assert state.syn_x is None
    assert state.syn_u is None
    assert int(model.sfa_mask.sum().item()) == 0


def test_stsp_spike_jump_is_order_u_not_dt_u():
    config, _inputs, _targets, _weights = _tiny_stripped(True)
    model = RecurrentLifSfa(config)
    state = model.initial_state(1, torch.device("cpu"), torch.float32)
    spiked = torch.zeros(1, config.n_hidden)
    spiked[0, 0] = 1.0
    state = state._replace(spikes=spiked)
    _logit, next_state = model.step(torch.zeros(1, 30), state)
    assert next_state.syn_u is not None and next_state.syn_x is not None
    delta_u = float(next_state.syn_u[0, 0] - STSP_U_FAC)
    assert delta_u > 0.05
    assert delta_u < 0.5
    assert float(next_state.syn_x[0, 0]) < 0.95
    assert delta_u > 20.0 * (config.dt_ms / 1000.0) * STSP_U_FAC


def test_stsp_state_is_per_neuron_not_per_edge():
    config, inputs, _targets, _weights = _tiny_stripped(True)
    model = RecurrentLifSfa(config)
    _logits, state = model(inputs)
    assert state.syn_x is not None and state.syn_u is not None
    assert state.syn_x.shape == (4, config.n_hidden)
    assert state.syn_u.shape == (4, config.n_hidden)


def test_stripped_stsp_recurrent_weights_get_gradients():
    config, inputs, targets, weights = _tiny_stripped(True)
    model = RecurrentLifSfa(config)
    logits, state = model(inputs)
    loss = weighted_cross_entropy(logits, targets, weights) + config.spike_cost * state.spike_power
    loss.backward()
    assert model.recurrent_linear.weight.grad is not None
    assert torch.isfinite(model.recurrent_linear.weight.grad).all()
    assert model.recurrent_linear.weight.grad.abs().sum() > 0


def _two_step_from_spike_leaf(model: RecurrentLifSfa, spikes: torch.Tensor) -> torch.Tensor:
    state = model.initial_state(1, spikes.device, spikes.dtype)._replace(spikes=spikes)
    inputs = torch.zeros(1, 30, device=spikes.device, dtype=spikes.dtype)
    _logit1, state = model.step(inputs, state)
    logit2, _state = model.step(inputs, state)
    return logit2


def test_no_stsp_does_not_backprop_through_recurrent_spikes():
    config, *_ = _tiny_stripped(False)
    model = RecurrentLifSfa(config)
    spikes = torch.full((1, config.n_hidden), 0.3, requires_grad=True)
    _two_step_from_spike_leaf(model, spikes).sum().backward()
    assert spikes.grad is None or float(spikes.grad.abs().sum().item()) == 0.0


def test_stsp_backprop_reaches_prior_spikes_through_syn_state():
    config, *_ = _tiny_stripped(True)
    model = RecurrentLifSfa(config)
    spikes = torch.full((1, config.n_hidden), 0.3, requires_grad=True)
    _two_step_from_spike_leaf(model, spikes).sum().backward()
    assert spikes.grad is not None
    assert torch.isfinite(spikes.grad).all()
    assert float(spikes.grad.abs().sum().item()) > 0.0


def test_per_parameter_clip_keeps_readout_grad_when_recurrent_grad_explodes():
    config, *_ = _tiny_stripped(True)
    model = RecurrentLifSfa(config)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    model.recurrent_linear.weight.grad.fill_(1.0e6)
    model.readout_linear.weight.grad.fill_(0.5)
    clip_gradients_per_parameter(model, max_norm=0.1)
    readout_norm = float(model.readout_linear.weight.grad.norm().item())
    recurrent_norm = float(model.recurrent_linear.weight.grad.norm().item())
    assert readout_norm == pytest.approx(0.1, rel=0.05)
    assert recurrent_norm == pytest.approx(0.1, rel=0.05)


def test_nonfinite_global_grad_norm_is_not_usable():
    config, *_ = _tiny_stripped(True)
    model = RecurrentLifSfa(config)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    model.recurrent_linear.weight.grad[0, 0] = torch.inf
    assert parameter_grads_are_usable(model) is False


def test_no_stsp_recurrent_weights_are_scaled_by_one_third():
    stsp = RecurrentLifSfa(_tiny_stripped(True)[0])
    no_stsp = RecurrentLifSfa(_tiny_stripped(False)[0])
    stsp_norm = torch.linalg.vector_norm(stsp.recurrent_linear.weight.detach())
    no_stsp_norm = torch.linalg.vector_norm(no_stsp.recurrent_linear.weight.detach())
    ratio = float(no_stsp_norm / stsp_norm.clamp_min(1e-8))
    assert ratio == pytest.approx(NO_STSP_RECURRENT_SCALE, rel=0.15)


@cuda_only
def test_stripped_stsp_smoke_dag_writes_decode_and_plot_is_leaf(tmp_path):
    output = tmp_path / "stripped_stsp"
    common = [
        "--output-directory",
        str(output),
        "--profile",
        "stripped_stsp",
        "--n-hidden",
        "16",
        "--n-train",
        "32",
        "--n-val",
        "32",
        "--n-test",
        "32",
        "--batch-size",
        "32",
        "--max-epochs",
        "1",
        "--device",
        "cuda",
    ]
    assert main(["build-trials", *common]) == 0
    assert main(["train", *common]) == 0
    assert main(["evaluate", *common]) == 0
    assert main(["decode", *common]) == 0
    before = {
        name: _hash_file(output / name)
        for name in (
            "data/checkpoints/best.pt",
            "data/test_predictions.csv",
            "metrics/test_metrics.json",
            "metrics/decode_metrics.json",
        )
    }
    assert main(["plot", *common]) == 0
    after_plot = {name: _hash_file(output / name) for name in before}
    assert after_plot == before
    decode = json.loads((output / "metrics" / "decode_metrics.json").read_text(encoding="utf-8"))
    assert "spike_decode" in decode
    assert "stsp_decode" in decode
    assert "shuffle" in decode
    recomputed = recompute_decode_from_features(output)
    assert recomputed["spike_decode"]["overall"] == pytest.approx(decode["spike_decode"]["overall"])
    assert recomputed["stsp_decode"]["overall"] == pytest.approx(decode["stsp_decode"]["overall"])
    plot_again = plot_run(output)
    assert plot_again["plot_only"] is True
    for name, digest in before.items():
        assert _hash_file(output / name) == digest
    assert (output / "figures" / "decode_accuracy.png").is_file()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert "legal_negative" not in summary or summary.get("behavior_gate_passed") in (True, False)


@cuda_only
def test_stripped_no_stsp_smoke_records_legal_negative_without_stsp_decode(tmp_path):
    output = tmp_path / "stripped_no_stsp"
    common = [
        "--output-directory",
        str(output),
        "--profile",
        "stripped_no_stsp",
        "--n-hidden",
        "16",
        "--n-train",
        "32",
        "--n-val",
        "32",
        "--n-test",
        "32",
        "--batch-size",
        "32",
        "--max-epochs",
        "1",
        "--device",
        "cuda",
    ]
    assert main(["build-trials", *common]) == 0
    assert main(["train", *common]) == 0
    assert main(["evaluate", *common]) == 0
    assert main(["decode", *common]) == 0
    decode = json.loads((output / "metrics" / "decode_metrics.json").read_text(encoding="utf-8"))
    assert "spike_decode" in decode
    assert "stsp_decode" not in decode
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert "legal_negative" in summary
    run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["use_stsp"] is False
    assert run_config["use_synaptic_current"] is False
    assert run_config["sfa_ratio"] == 0.0
