"""Overfit self-check and smoke DAG workflow for Masse delayed-cue LIF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.experiments.masse_delayed_cue_lif.artifacts import (
    command_is_fresh,
    file_sha256,
    input_lineage,
    read_json,
    require_identical_trial_tables,
    save_run_config,
    write_json,
    write_manifest,
)
from src.experiments.masse_delayed_cue_lif.config import (
    formal_config,
    overfit_config,
    stripped_stsp_config,
)
from src.experiments.masse_delayed_cue_lif.evaluate import evaluate_run
from src.experiments.masse_delayed_cue_lif.model import RecurrentLifSfa
from src.experiments.masse_delayed_cue_lif.plot import plot_run
from src.experiments.masse_delayed_cue_lif.run import build_trials, main
from src.experiments.masse_delayed_cue_lif.task import (
    expand_rows,
    generate_trial_table,
    load_trial_table,
    save_trial_table,
)
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


def _tiny_overfit_cpu():
    return overfit_config(device="cpu", max_epochs=1)


def _write_plot_input_stubs(root: Path, config) -> None:
    save_run_config(root, config)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "data" / "train_history.json").write_text("{}\n", encoding="utf-8")
    (root / "data" / "test_predictions.csv").write_text("trial_id\n", encoding="utf-8")
    (root / "metrics" / "test_metrics.json").write_text("{}\n", encoding="utf-8")


def _stamp_evaluate_bundle(root: Path, *, checkpoint_bytes: bytes, trials_text: str) -> None:
    save_run_config(root, stripped_stsp_config(device="cpu", n_hidden=4, n_train=32, n_val=32, n_test=32, batch_size=4, max_epochs=1))
    (root / "data" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "data" / "trials.csv").write_text(trials_text, encoding="utf-8")
    (root / "data" / "checkpoints" / "best.pt").write_bytes(checkpoint_bytes)
    (root / "data" / "test_predictions.csv").write_text("trial_id\n1\n", encoding="utf-8")
    metrics = {"trial_accuracy": 0.5}
    metrics["lineage"] = input_lineage(root, ("run_config.json", "data/trials.csv", "data/checkpoints/best.pt"))
    write_json(root / "metrics" / "test_metrics.json", metrics)


def test_reuse_trials_copies_matching_table_and_records_hash(tmp_path):
    config = _tiny_overfit_cpu()
    source = tmp_path / "source.csv"
    save_trial_table(source, generate_trial_table(config))
    output = tmp_path / "run"
    summary = build_trials(output, config, reuse_trials=source)
    copied = output / "data" / "trials.csv"
    assert file_sha256(copied) == file_sha256(source)
    assert summary["n_train"] == 32
    assert summary["n_val"] == 32
    assert summary["n_test"] == 32
    assert summary["n_rows"] == 96
    assert summary["trials_sha256"] == file_sha256(source)
    rows = load_trial_table(copied)
    assert [row["trial_id"] for row in rows] == list(range(96))


def test_reuse_trials_rejects_schema_split_and_duplicate_ids(tmp_path):
    config = _tiny_overfit_cpu()
    output = tmp_path / "run"
    bad_schema = tmp_path / "bad_schema.csv"
    bad_schema.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        build_trials(output, config, reuse_trials=bad_schema)

    wrong_counts = tmp_path / "wrong_counts.csv"
    small = overfit_config(device="cpu", n_train=32, n_val=32, n_test=32)
    save_trial_table(wrong_counts, generate_trial_table(small))
    mismatched = overfit_config(device="cpu", n_train=32, n_val=32, n_test=64)
    with pytest.raises(ValueError, match="split"):
        build_trials(output, mismatched, reuse_trials=wrong_counts)

    dup = tmp_path / "dup.csv"
    rows = generate_trial_table(config)
    rows[1]["trial_id"] = rows[0]["trial_id"]
    save_trial_table(dup, rows)
    with pytest.raises(ValueError, match="trial_id"):
        build_trials(output, config, reuse_trials=dup)


def test_identical_trial_tables_required_for_matched_pair(tmp_path):
    config = _tiny_overfit_cpu()
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    other = tmp_path / "other.csv"
    save_trial_table(left, generate_trial_table(config))
    right.write_bytes(left.read_bytes())
    save_trial_table(other, generate_trial_table(overfit_config(device="cpu", trial_table_seed=99)))
    assert require_identical_trial_tables(left, right) == file_sha256(left)
    with pytest.raises(ValueError, match="identical"):
        require_identical_trial_tables(left, other)


def test_evaluate_is_not_fresh_when_checkpoint_or_trials_change(tmp_path):
    root = tmp_path / "bundle"
    _stamp_evaluate_bundle(root, checkpoint_bytes=b"ckpt-a", trials_text="trials-v1\n")
    assert command_is_fresh(root, "evaluate") is True
    (root / "data" / "checkpoints" / "best.pt").write_bytes(b"ckpt-b")
    assert command_is_fresh(root, "evaluate") is False
    _stamp_evaluate_bundle(root, checkpoint_bytes=b"ckpt-a", trials_text="trials-v1\n")
    (root / "data" / "trials.csv").write_text("trials-v2\n", encoding="utf-8")
    assert command_is_fresh(root, "evaluate") is False
    _stamp_evaluate_bundle(root, checkpoint_bytes=b"ckpt-a", trials_text="trials-v1\n")
    (root / "metrics" / "test_metrics.json").unlink()
    assert command_is_fresh(root, "evaluate") is False


def test_stripped_plot_check_only_requires_decode_formal_does_not(tmp_path):
    stripped = tmp_path / "stripped"
    _write_plot_input_stubs(
        stripped,
        stripped_stsp_config(
            device="cpu",
            n_hidden=4,
            n_train=32,
            n_val=32,
            n_test=32,
            batch_size=4,
            max_epochs=1,
        ),
    )
    with pytest.raises(FileNotFoundError, match="decode_metrics"):
        main(
            [
                "plot",
                "--check-only",
                "--output-directory",
                str(stripped),
                "--profile",
                "stripped_stsp",
                "--device",
                "cpu",
            ]
        )
    (stripped / "metrics" / "decode_metrics.json").write_text("{}\n", encoding="utf-8")
    assert (
        main(
            [
                "plot",
                "--check-only",
                "--output-directory",
                str(stripped),
                "--profile",
                "stripped_stsp",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    formal = tmp_path / "formal"
    _write_plot_input_stubs(formal, formal_config(device="cpu"))
    assert (
        main(
            [
                "plot",
                "--check-only",
                "--output-directory",
                str(formal),
                "--profile",
                "formal",
                "--device",
                "cpu",
            ]
        )
        == 0
    )


def test_stripped_manifest_declares_decode_plot_contract(tmp_path):
    stripped = tmp_path / "stripped"
    save_run_config(
        stripped,
        stripped_stsp_config(
            device="cpu",
            n_hidden=4,
            n_train=32,
            n_val=32,
            n_test=32,
            batch_size=4,
            max_epochs=1,
        ),
    )
    write_manifest(stripped)
    manifest = read_json(stripped / "artifact_manifest.json")
    assert "metrics/decode_metrics.json" in manifest["tasks"]["plot"]["depends_on"]
    assert "figures/decode_accuracy.png" in manifest["tasks"]["plot"]["outputs"]

    formal = tmp_path / "formal"
    save_run_config(formal, formal_config(device="cpu"))
    write_manifest(formal)
    formal_manifest = read_json(formal / "artifact_manifest.json")
    assert "metrics/decode_metrics.json" not in formal_manifest["tasks"]["plot"]["depends_on"]
    assert "figures/decode_accuracy.png" not in formal_manifest["tasks"]["plot"]["outputs"]


def test_stripped_plot_run_fails_without_decode(tmp_path):
    root = tmp_path / "stripped"
    _write_plot_input_stubs(
        root,
        stripped_stsp_config(
            device="cpu",
            n_hidden=4,
            n_train=32,
            n_val=32,
            n_test=32,
            batch_size=4,
            max_epochs=1,
        ),
    )
    with pytest.raises(FileNotFoundError, match="decode_metrics"):
        plot_run(root)
