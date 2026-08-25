"""Overfit self-check and smoke DAG workflow for Masse delayed-cue LIF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.experiments.masse_delayed_cue_lif.config import overfit_config
from src.experiments.masse_delayed_cue_lif.evaluate import evaluate_run
from src.experiments.masse_delayed_cue_lif.model import RecurrentLifSfa
from src.experiments.masse_delayed_cue_lif.plot import plot_run
from src.experiments.masse_delayed_cue_lif.run import build_trials, main
from src.experiments.masse_delayed_cue_lif.task import expand_rows, load_trial_table
from src.experiments.masse_delayed_cue_lif.train import train_run


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@cuda_only
def test_small_network_overfits_all_32_conditions(tmp_path):
    config = overfit_config()
    build_trials(tmp_path, config)
    result = train_run(tmp_path, config)
    assert result["last_train_loss"] < result["first_train_loss"]
    assert result["final_train_trial_accuracy"] >= 0.80
    checkpoint = torch.load(
        tmp_path / "data" / "checkpoints" / "best.pt",
        map_location="cuda",
        weights_only=False,
    )
    rows = load_trial_table(tmp_path / "data" / "trials.csv", split="train")
    inputs, _targets, _weights = expand_rows(rows, config)
    model = RecurrentLifSfa(config).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        first, _ = model(inputs.cuda())
        second, _ = model(inputs.cuda())
    torch.testing.assert_close(first, second, rtol=0.0, atol=1e-5)


@cuda_only
def test_smoke_dag_writes_required_artifacts_and_plot_is_leaf(tmp_path):
    output = tmp_path / "smoke"
    common = [
        "--output-directory",
        str(output),
        "--profile",
        "smoke",
        "--n-hidden",
        "32",
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
    before = {
        name: _hash_file(output / name)
        for name in (
            "data/checkpoints/best.pt",
            "data/test_predictions.csv",
            "metrics/test_metrics.json",
        )
    }
    assert main(["plot", *common]) == 0
    after_plot = {name: _hash_file(output / name) for name in before}
    assert after_plot == before
    for relative in (
        "run_config.json",
        "summary.json",
        "artifact_manifest.json",
        "meta/run_info.json",
        "data/trials.csv",
        "data/checkpoints/best.pt",
        "data/checkpoints/last.pt",
        "data/train_history.json",
        "data/test_predictions.csv",
        "metrics/test_metrics.json",
        "figures/training_curves.png",
        "figures/condition_accuracy.png",
        "figures/rule_match_confusion.png",
        "figures/example_trial_timeline.png",
    ):
        assert (output / relative).is_file(), relative
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tasks"]["plot"]["plot_only"] is True
    metrics = json.loads((output / "metrics" / "test_metrics.json").read_text(encoding="utf-8"))
    recomputed = evaluate_run(output)
    assert recomputed["trial_accuracy"] == pytest.approx(metrics["trial_accuracy"])
    plot_again = plot_run(output)
    assert plot_again["plot_only"] is True
    for name, digest in before.items():
        assert _hash_file(output / name) == digest
    history = json.loads((output / "data" / "train_history.json").read_text(encoding="utf-8"))
    assert history["epochs"]
    assert all(epoch["train_loss"] == epoch["train_loss"] for epoch in history["epochs"])
