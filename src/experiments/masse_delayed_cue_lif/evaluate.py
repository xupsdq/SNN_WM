"""Evaluate node: freeze best.pt and score the test split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    REQUIRED_EVAL_INPUTS,
    config_identity,
    layout_for,
    load_checkpoint,
    load_run_config,
    require_files,
    write_json,
    write_manifest,
    write_predictions_csv,
    write_summary,
)
from .config import MasseDelayedCueConfig
from .metrics import (
    attach_timepoint_breakdown,
    fixation_accuracy,
    summarize_predictions,
    trial_predictions,
)
from .model import RecurrentLifSfa
from .task import expand_trial, load_trial_table


@torch.no_grad()
def evaluate_run(run_directory: Path, config: MasseDelayedCueConfig | None = None) -> dict[str, Any]:
    run_directory = Path(run_directory)
    require_files(run_directory, REQUIRED_EVAL_INPUTS)
    stored = load_run_config(run_directory)
    if config is None:
        config = stored
    layout = layout_for(run_directory)
    device = torch.device(config.device)
    checkpoint = load_checkpoint(layout.data_dir / "checkpoints" / "best.pt", map_location=device)
    if checkpoint.get("identity") != config_identity(config):
        raise ValueError("checkpoint identity does not match run_config.json")

    model = RecurrentLifSfa(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = load_trial_table(layout.data_dir / "trials.csv", split="test")
    logits_chunks = []
    target_chunks = []
    batch_size = min(config.batch_size, len(rows))
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        inputs = torch.stack([expand_trial(row, config)[0] for row in chunk], dim=1).to(device)
        targets = torch.stack([expand_trial(row, config)[1] for row in chunk], dim=1).to(device)
        logits, _ = model(inputs)
        logits_chunks.append(logits)
        target_chunks.append(targets)
    logits = torch.cat(logits_chunks, dim=1)
    targets = torch.cat(target_chunks, dim=1)
    records = trial_predictions(logits, rows, config)
    metrics = summarize_predictions(
        records,
        fixation_acc=fixation_accuracy(logits, targets, config),
    )
    attach_timepoint_breakdown(metrics, logits, targets, rows, config)
    metrics["identity"] = config_identity(config)
    metrics["checkpoint_epoch"] = int(checkpoint["epoch"])
    write_predictions_csv(layout.data_dir / "test_predictions.csv", records)
    write_json(layout.metrics_dir / "test_metrics.json", metrics)
    write_manifest(run_directory)
    write_summary(
        run_directory,
        {
            "status": "evaluated",
            "profile": config.profile,
            "test_trial_accuracy": metrics["trial_accuracy"],
            "test_timepoint_accuracy": metrics["timepoint_accuracy"],
            "checkpoint_epoch": metrics["checkpoint_epoch"],
        },
    )
    return metrics
