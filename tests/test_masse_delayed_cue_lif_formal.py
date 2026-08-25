"""Formal Masse delayed-cue LIF gates on a persisted results bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from src.experiments.masse_delayed_cue_lif.artifacts import read_predictions_csv
from src.experiments.masse_delayed_cue_lif.evaluate import evaluate_run
from src.experiments.masse_delayed_cue_lif.metrics import formal_gates_status, summarize_predictions
from src.experiments.masse_delayed_cue_lif.plot import plot_run

FORMAL_ROOT = Path("results/masse_delayed_cue_lif/formal")
_FORMAL_METRICS = FORMAL_ROOT / "metrics" / "test_metrics.json"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_gates_status_accepts_threshold_and_rejects_below():
    metrics = {
        "timepoint_accuracy": 0.91,
        "dms_timepoint_accuracy": 0.91,
        "dmrs90_timepoint_accuracy": 0.91,
        "cross_condition_trial_accuracy": {
            "dms_match0": 0.86,
            "dms_match1": 0.86,
            "dmrs90_match0": 0.86,
            "dmrs90_match1": 0.86,
        },
    }
    assert formal_gates_status(metrics)["passed"] is True
    failing = dict(metrics)
    failing["dmrs90_timepoint_accuracy"] = 0.89
    status = formal_gates_status(failing)
    assert status["passed"] is False
    assert status["failed"]


@pytest.mark.skipif(not _FORMAL_METRICS.is_file(), reason="formal bundle is not present")
def test_formal_bundle_meets_gates_recomputes_metrics_and_plot_is_leaf():
    metrics = json.loads(_FORMAL_METRICS.read_text(encoding="utf-8"))
    records = read_predictions_csv(FORMAL_ROOT / "data" / "test_predictions.csv")
    recomputed = summarize_predictions(records)
    assert recomputed["trial_accuracy"] == pytest.approx(metrics["trial_accuracy"])
    assert recomputed["dms_trial_accuracy"] == pytest.approx(metrics["dms_trial_accuracy"])
    assert recomputed["dmrs90_trial_accuracy"] == pytest.approx(metrics["dmrs90_trial_accuracy"])
    for key, value in metrics["cross_condition_trial_accuracy"].items():
        assert recomputed["cross_condition_trial_accuracy"][key] == pytest.approx(value)
    gates = formal_gates_status(metrics)
    assert gates["passed"], gates["failed"]
    before = {
        name: _hash_file(FORMAL_ROOT / name)
        for name in (
            "data/checkpoints/best.pt",
            "data/test_predictions.csv",
            "metrics/test_metrics.json",
        )
    }
    plot_again = plot_run(FORMAL_ROOT)
    assert plot_again["plot_only"] is True
    after_plot = {name: _hash_file(FORMAL_ROOT / name) for name in before}
    assert after_plot == before
    with tempfile.TemporaryDirectory() as tmp:
        replay_root = Path(tmp) / "formal"
        shutil.copytree(FORMAL_ROOT, replay_root)
        replay = evaluate_run(replay_root)
        replay_records = read_predictions_csv(replay_root / "data" / "test_predictions.csv")
        assert replay["trial_accuracy"] == pytest.approx(metrics["trial_accuracy"])
        assert replay["timepoint_accuracy"] == pytest.approx(metrics["timepoint_accuracy"])
        assert [row["predicted_class"] for row in replay_records] == [
            row["predicted_class"] for row in records
        ]
    for name, digest in before.items():
        assert _hash_file(FORMAL_ROOT / name) == digest
