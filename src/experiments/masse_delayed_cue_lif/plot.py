"""Plot-only leaf: consume training history, predictions, metrics, and run config."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.plotting.common.colors import NATURE_COMPATIBLE_PALETTE

from .artifacts import (
    input_lineage,
    layout_for,
    profile_requires_decode_plot,
    read_json,
    require_files,
    required_plot_inputs,
    write_manifest,
)
from .config import (
    CLASS_MATCH,
    RULE_START_MS,
    RULE_STOP_MS,
    SAMPLE_START_MS,
    SAMPLE_STOP_MS,
    TEST_START_MS,
    TEST_STOP_MS,
    MasseDelayedCueConfig,
)
from .task import expand_trial


def _load_config(run_directory: Path) -> MasseDelayedCueConfig:
    from .artifacts import load_run_config

    return load_run_config(run_directory)


def _style(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3.0, width=0.8)
    axis.set_facecolor("white")


def _export(fig, prefix: Path) -> list[str]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("png", "pdf", "svg"):
        path = prefix.with_suffix("." + extension)
        fig.savefig(path, dpi=300, facecolor="white")
        outputs.append(str(path.resolve()))
    plt.close(fig)
    return outputs


def _plot_training_curves(history: dict[str, Any], prefix: Path) -> list[str]:
    epochs = [row["epoch"] for row in history["epochs"]]
    train_loss = [row["train_loss"] for row in history["epochs"]]
    train_acc = [row["train_trial_accuracy"] for row in history["epochs"]]
    val_acc = [row["val_trial_accuracy"] for row in history["epochs"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    axes[0].plot(epochs, train_loss, color=NATURE_COMPATIBLE_PALETTE["primary_navy"], lw=1.6)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train loss")
    axes[1].plot(
        epochs,
        train_acc,
        color=NATURE_COMPATIBLE_PALETTE["primary_navy"],
        lw=1.6,
        label="Train",
    )
    axes[1].plot(
        epochs,
        val_acc,
        color=NATURE_COMPATIBLE_PALETTE["comparison_coral"],
        lw=1.6,
        label="Validation",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Trial accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False)
    for axis in axes:
        _style(axis)
    return _export(fig, prefix)


def _plot_condition_accuracy(metrics: dict[str, Any], prefix: Path) -> list[str]:
    labels = [
        "Overall",
        "DMS",
        "DMRS90",
        "Match",
        "Nonmatch",
    ]
    values = [
        metrics["trial_accuracy"],
        metrics["dms_trial_accuracy"],
        metrics["dmrs90_trial_accuracy"],
        metrics["match_trial_accuracy"],
        metrics["nonmatch_trial_accuracy"],
    ]
    crosses = metrics["cross_condition_trial_accuracy"]
    cross_labels = list(crosses.keys())
    cross_values = [crosses[key] for key in cross_labels]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    axes[0].bar(labels, values, color=NATURE_COMPATIBLE_PALETTE["primary_navy"])
    axes[1].bar(cross_labels, cross_values, color=NATURE_COMPATIBLE_PALETTE["mechanism_teal"])
    for axis in axes:
        _style(axis)
        axis.set_ylim(0.0, 1.0)
        axis.tick_params(axis="x", rotation=30)
        axis.set_ylabel("Trial accuracy")
    return _export(fig, prefix)


def _plot_confusion(records: list[dict[str, str]], prefix: Path) -> list[str]:
    labels = ["DMS match", "DMS nonmatch", "DMRS90 match", "DMRS90 nonmatch"]
    key = {
        ("DMS", "1"): 0,
        ("DMS", "0"): 1,
        ("DMRS90", "1"): 2,
        ("DMRS90", "0"): 3,
    }
    counts = np.zeros((4, 4), dtype=np.int64)
    for row in records:
        true_index = key[(row["rule"], str(int(row["match"])))]
        pred_match = "1" if int(row["predicted_class"]) == CLASS_MATCH else "0"
        pred_index = key[(row["rule"], pred_match)]
        counts[true_index, pred_index] += 1
    fig, axis = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
    image = axis.imshow(counts, cmap="Blues")
    axis.set_xticks(range(4), labels, rotation=30, ha="right")
    axis.set_yticks(range(4), labels)
    axis.set_xlabel("Predicted condition")
    axis.set_ylabel("True condition")
    for i in range(4):
        for j in range(4):
            axis.text(j, i, str(counts[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    _style(axis)
    return _export(fig, prefix)


def _row_from_prediction(record: dict[str, str]) -> dict[str, object]:
    return {
        "split": record.get("split", "test"),
        "trial_id": int(record["trial_id"]),
        "sample_direction": int(record["sample_direction"]),
        "rule": record["rule"],
        "match": int(record["match"]),
        "test_direction": int(record["test_direction"]),
        "input_seed": 0,
    }


def _plot_example_trial(
    records: list[dict[str, str]],
    config: MasseDelayedCueConfig,
    prefix: Path,
) -> list[str]:
    row = _row_from_prediction(records[0])
    inputs, targets, weights = expand_trial(row, config)
    time_ms = np.arange(config.n_steps) * config.dt_ms
    motion = inputs[:, :24].max(dim=1).values.numpy()
    rule = inputs[:, 24:].max(dim=1).values.numpy()
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.4), sharex=True, constrained_layout=True)
    axes[0].plot(time_ms, motion, color=NATURE_COMPATIBLE_PALETTE["primary_navy"], lw=1.4)
    axes[0].set_ylabel("Motion input")
    axes[1].plot(time_ms, rule, color=NATURE_COMPATIBLE_PALETTE["mechanism_teal"], lw=1.4)
    axes[1].set_ylabel("Rule input")
    axes[2].plot(
        time_ms,
        targets.numpy(),
        color=NATURE_COMPATIBLE_PALETTE["comparison_coral"],
        lw=1.4,
        label="target",
    )
    axes[2].plot(
        time_ms,
        weights.numpy(),
        color=NATURE_COMPATIBLE_PALETTE["ink"],
        lw=1.0,
        label="loss weight",
    )
    axes[2].set_ylabel("Target / weight")
    axes[2].set_xlabel("Time (ms)")
    axes[2].legend(frameon=False, loc="upper left")
    for axis in axes:
        _style(axis)
        for start, stop in (
            (SAMPLE_START_MS, SAMPLE_STOP_MS),
            (RULE_START_MS, RULE_STOP_MS),
            (TEST_START_MS, TEST_STOP_MS),
        ):
            axis.axvspan(start, stop, color=NATURE_COMPATIBLE_PALETTE["primary_pale"], zorder=0)
    axes[0].set_title(
        f"trial {row['trial_id']} {row['rule']} match={row['match']} "
        f"sample={row['sample_direction']} test={row['test_direction']}"
    )
    return _export(fig, prefix)


def _plot_decode(decode: dict[str, Any], prefix: Path) -> list[str]:
    spike = decode.get("spike_decode", {})
    stsp = decode.get("stsp_decode", {})
    labels = ["Spike overall", "Spike DMS", "Spike DMRS90"]
    values = [spike.get("overall", 0.0), spike.get("dms", 0.0), spike.get("dmrs90", 0.0)]
    if stsp:
        labels.extend(["STSP overall", "STSP DMS", "STSP DMRS90"])
        values.extend([stsp.get("overall", 0.0), stsp.get("dms", 0.0), stsp.get("dmrs90", 0.0)])
    fig, axis = plt.subplots(figsize=(6.4, 2.8), constrained_layout=True)
    axis.bar(labels, values, color=NATURE_COMPATIBLE_PALETTE["mechanism_teal"])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Sample decode accuracy")
    axis.tick_params(axis="x", rotation=30)
    _style(axis)
    return _export(fig, prefix)


def plot_run(run_directory: Path) -> dict[str, Any]:
    run_directory = Path(run_directory)
    config = _load_config(run_directory)
    plot_inputs = required_plot_inputs(config)
    require_files(run_directory, plot_inputs)
    layout = layout_for(run_directory)
    history = read_json(layout.data_dir / "train_history.json")
    metrics = read_json(layout.metrics_dir / "test_metrics.json")
    with (layout.data_dir / "test_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    outputs = {
        "training_curves": _plot_training_curves(history, layout.figures_dir / "training_curves"),
        "condition_accuracy": _plot_condition_accuracy(metrics, layout.figures_dir / "condition_accuracy"),
        "rule_match_confusion": _plot_confusion(records, layout.figures_dir / "rule_match_confusion"),
        "example_trial_timeline": _plot_example_trial(
            records, config, layout.figures_dir / "example_trial_timeline"
        ),
    }
    decode_path = layout.metrics_dir / "decode_metrics.json"
    if profile_requires_decode_plot(config.profile) or decode_path.is_file():
        decode = read_json(decode_path)
        outputs["decode_accuracy"] = _plot_decode(decode, layout.figures_dir / "decode_accuracy")
    write_manifest(
        run_directory,
        extra={
            "plot_only": True,
            "plot_lineage": input_lineage(run_directory, plot_inputs),
        },
    )
    return {"plot_only": True, "outputs": outputs}
