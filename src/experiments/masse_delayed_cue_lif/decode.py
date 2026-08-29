"""Mechanism node: delay-end sample decoding and STSP shuffle intervention."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .artifacts import (
    REQUIRED_DECODE_INPUTS,
    attach_input_lineage,
    config_identity,
    layout_for,
    load_checkpoint,
    load_run_config,
    require_files,
    write_json,
    write_manifest,
    write_summary,
)
from .config import DELAY_DECODE_START_MS, DELAY_DECODE_STOP_MS, RULE_DMRS90, RULE_DMS, MasseDelayedCueConfig
from .metrics import (
    attach_timepoint_breakdown,
    formal_gates_status,
    summarize_predictions,
    trial_predictions,
)
from .model import RecurrentLifSfa
from .task import expand_trial, load_trial_table, window_indices

N_SHUFFLES = 5
DECODE_SPLIT_SEED = 0


def _subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float((y_true[mask] == y_pred[mask]).mean())


def _fit_linear_decoder(
    features: np.ndarray,
    labels: np.ndarray,
    rules: np.ndarray,
) -> dict[str, float]:
    unique, counts = np.unique(labels, return_counts=True)
    stratify = labels if unique.size > 1 and int(counts.min()) >= 2 else None
    train_index, test_index = train_test_split(
        np.arange(len(labels)),
        test_size=0.5,
        random_state=DECODE_SPLIT_SEED,
        stratify=stratify,
    )
    decoder = LogisticRegression(max_iter=400, solver="lbfgs")
    decoder.fit(features[train_index], labels[train_index])
    predicted = decoder.predict(features[test_index])
    y_true = labels[test_index]
    rule_test = rules[test_index]
    return {
        "overall": _subset_accuracy(y_true, predicted, np.ones(len(y_true), dtype=bool)),
        "dms": _subset_accuracy(y_true, predicted, rule_test == RULE_DMS),
        "dmrs90": _subset_accuracy(y_true, predicted, rule_test == RULE_DMRS90),
        "n_test": int(len(test_index)),
    }


def _collect_traces(
    model: RecurrentLifSfa,
    rows: list[dict[str, object]],
    config: MasseDelayedCueConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None, torch.Tensor]:
    windows = window_indices(config)
    decode_window = windows["delay_end_decode"]
    spike_means = []
    efficacy_means = []
    logit_chunks = []
    batch_size = min(config.batch_size, len(rows))
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        inputs = torch.stack([expand_trial(row, config)[0] for row in chunk], dim=1).to(device)
        logits, _state, spikes, efficacy = model(inputs, record_traces=True)
        logit_chunks.append(logits)
        spike_means.append(spikes[decode_window].mean(dim=0).detach().cpu())
        if efficacy is not None:
            efficacy_means.append(efficacy[decode_window].mean(dim=0).detach().cpu())
    spikes_feature = torch.cat(spike_means, dim=0).numpy()
    efficacy_feature = torch.cat(efficacy_means, dim=0).numpy() if efficacy_means else None
    return spikes_feature, efficacy_feature, torch.cat(logit_chunks, dim=1)


@torch.no_grad()
def _shuffle_task_accuracy(
    model: RecurrentLifSfa,
    rows: list[dict[str, object]],
    config: MasseDelayedCueConfig,
    device: torch.device,
) -> dict[str, float]:
    if not config.use_stsp:
        return {}
    windows = window_indices(config)
    shuffle_at = int(windows["delay_end_decode"].start)
    accuracies = []
    dms_accuracies = []
    dmrs_accuracies = []
    batch_size = min(config.batch_size, len(rows))
    for shuffle_index in range(N_SHUFFLES):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(shuffle_index)
        logit_chunks = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            inputs = torch.stack([expand_trial(row, config)[0] for row in chunk], dim=1).to(device)
            logits, _ = model(inputs, shuffle_stsp_at=shuffle_at, shuffle_generator=generator)
            logit_chunks.append(logits)
        logits = torch.cat(logit_chunks, dim=1)
        records = trial_predictions(logits, rows, config)
        summary = summarize_predictions(records)
        accuracies.append(summary["trial_accuracy"])
        dms_accuracies.append(summary["dms_trial_accuracy"])
        dmrs_accuracies.append(summary["dmrs90_trial_accuracy"])
    return {
        "trial_accuracy": float(np.mean(accuracies)),
        "dms_trial_accuracy": float(np.mean(dms_accuracies)),
        "dmrs90_trial_accuracy": float(np.mean(dmrs_accuracies)),
        "n_shuffles": N_SHUFFLES,
    }


@torch.no_grad()
def decode_run(run_directory: Path, config: MasseDelayedCueConfig | None = None) -> dict[str, Any]:
    run_directory = Path(run_directory)
    require_files(run_directory, REQUIRED_DECODE_INPUTS)
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
    spike_features, efficacy_features, logits = _collect_traces(model, rows, config, device)
    labels = np.array([int(row["sample_direction"]) for row in rows], dtype=np.int64)
    rules = np.array([str(row["rule"]) for row in rows])
    payload: dict[str, Any] = {
        "identity": config_identity(config),
        "profile": config.profile,
        "spike_decode": _fit_linear_decoder(spike_features, labels, rules),
        "window_ms": [DELAY_DECODE_START_MS, DELAY_DECODE_STOP_MS],
    }
    if efficacy_features is not None:
        payload["stsp_decode"] = _fit_linear_decoder(efficacy_features, labels, rules)
        payload["shuffle"] = _shuffle_task_accuracy(model, rows, config, device)
        targets = torch.stack([expand_trial(row, config)[1] for row in rows], dim=1).to(device)
        baseline = summarize_predictions(trial_predictions(logits, rows, config))
        attach_timepoint_breakdown(baseline, logits, targets, rows, config)
        payload["unshuffled_trial_accuracy"] = baseline["trial_accuracy"]
    attach_input_lineage(payload, run_directory, REQUIRED_DECODE_INPUTS)
    torch.save(
        {
            "spike_features": spike_features,
            "efficacy_features": efficacy_features,
            "labels": labels,
            "rules": rules,
        },
        layout.data_dir / "delay_end_features.pt",
    )
    write_json(layout.metrics_dir / "decode_metrics.json", payload)
    write_manifest(run_directory)
    summary: dict[str, Any] = {}
    summary_path = run_directory / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["decode"] = {
        "spike_overall": payload["spike_decode"]["overall"],
        "stsp_overall": payload.get("stsp_decode", {}).get("overall"),
    }
    write_summary(run_directory, summary)
    return payload


def recompute_decode_from_features(run_directory: Path) -> dict[str, Any]:
    layout = layout_for(Path(run_directory))
    stored = torch.load(layout.data_dir / "delay_end_features.pt", weights_only=False, map_location="cpu")
    payload = {"spike_decode": _fit_linear_decoder(stored["spike_features"], stored["labels"], stored["rules"])}
    if stored["efficacy_features"] is not None:
        payload["stsp_decode"] = _fit_linear_decoder(
            stored["efficacy_features"], stored["labels"], stored["rules"]
        )
    return payload


def attach_behavior_gate(
    summary: dict[str, Any],
    metrics: dict[str, Any],
    config: MasseDelayedCueConfig,
) -> dict[str, Any]:
    gates = formal_gates_status(metrics)
    if config.profile == "stripped_stsp":
        summary["behavior_gate_passed"] = gates["passed"]
        summary["behavior_gate_failed"] = gates["failed"]
    elif config.profile == "stripped_no_stsp":
        summary["behavior_gate_passed"] = gates["passed"]
        summary["legal_negative"] = not gates["passed"]
        summary["behavior_gate_failed"] = gates["failed"]
    return summary
