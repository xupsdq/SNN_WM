"""Public behaviour of the recurrent LIF/SFA model."""

from __future__ import annotations

import torch

from src.experiments.masse_delayed_cue_lif.config import overfit_config
from src.experiments.masse_delayed_cue_lif.metrics import weighted_cross_entropy
from src.experiments.masse_delayed_cue_lif.model import RecurrentLifSfa
from src.experiments.masse_delayed_cue_lif.task import expand_rows, generate_trial_table


def _tiny_batch():
    config = overfit_config(device="cpu")
    rows = generate_trial_table(config)[:4]
    inputs, targets, weights = expand_rows(rows, config)
    return config, inputs, targets, weights


def test_output_shape_is_time_by_batch_by_three():
    config, inputs, _targets, _weights = _tiny_batch()
    model = RecurrentLifSfa(config)
    logits, _state = model(inputs)
    assert logits.shape == (config.n_steps, 4, 3)


def test_loss_is_finite_and_recurrent_and_input_weights_get_gradients():
    config, inputs, targets, weights = _tiny_batch()
    model = RecurrentLifSfa(config)
    logits, _state = model(inputs)
    loss = weighted_cross_entropy(logits, targets, weights)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.recurrent_linear.weight.grad is not None
    assert model.input_linear.weight.grad is not None
    assert torch.isfinite(model.recurrent_linear.weight.grad).all()
    assert torch.isfinite(model.input_linear.weight.grad).all()
    assert model.recurrent_linear.weight.grad.abs().sum() > 0
    assert model.input_linear.weight.grad.abs().sum() > 0
    assert torch.all(model.recurrent_weight().diag() == 0)


def test_reset_state_is_deterministic_and_carry_over_is_not_silent():
    config, inputs, _targets, _weights = _tiny_batch()
    model = RecurrentLifSfa(config)
    model.eval()
    first, _ = model(inputs)
    second, state = model(inputs)
    assert torch.equal(first, second)
    other = torch.flip(inputs, dims=[1])
    carried, _ = model(other, state=state)
    fresh, _ = model(other)
    assert not torch.equal(carried, fresh)


def test_checkpoint_reload_reproduces_logits(tmp_path):
    config, inputs, _targets, _weights = _tiny_batch()
    model = RecurrentLifSfa(config)
    model.eval()
    logits, _ = model(inputs)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = RecurrentLifSfa(config)
    restored.load_state_dict(torch.load(path, map_location="cpu"))
    restored.eval()
    replay, _ = restored(inputs)
    torch.testing.assert_close(replay, logits, rtol=0.0, atol=0.0)
