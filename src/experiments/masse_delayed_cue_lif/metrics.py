"""Task metrics from logits, targets, and time weights."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .config import CLASS_FIXATION, CLASS_MATCH, CLASS_NONMATCH, RULE_DMRS90, RULE_DMS, MasseDelayedCueConfig
from .task import window_indices


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    n_steps, batch_size, n_classes = logits.shape
    flat_logits = logits.reshape(n_steps * batch_size, n_classes)
    flat_targets = targets.reshape(n_steps * batch_size)
    flat_weights = weights.reshape(n_steps * batch_size)
    per_step = torch.nn.functional.cross_entropy(flat_logits, flat_targets, reduction="none")
    denom = flat_weights.sum().clamp_min(1e-8)
    return (per_step * flat_weights).sum() / denom


def trial_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: MasseDelayedCueConfig,
) -> torch.Tensor:
    windows = window_indices(config)
    mean_logits = logits[windows["valid_test"]].mean(dim=0)
    trial_targets = targets[windows["valid_test"]][0]
    return torch.nn.functional.cross_entropy(mean_logits, trial_targets)


def _valid_test_logits(
    logits: torch.Tensor,
    config: MasseDelayedCueConfig,
) -> torch.Tensor:
    windows = window_indices(config)
    return logits[windows["valid_test"]]


def timepoint_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    predicted = logits.argmax(dim=-1)
    valid = weights > 0
    if int(valid.sum()) == 0:
        return 0.0
    correct = (predicted == targets) & valid
    return float(correct.sum().item() / valid.sum().item())


def trial_predictions(
    logits: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    config: MasseDelayedCueConfig,
) -> list[dict[str, Any]]:
    valid = _valid_test_logits(logits, config)
    mean_logits = valid.mean(dim=0)
    predicted = mean_logits.argmax(dim=-1).detach().cpu().tolist()
    records = []
    for index, row in enumerate(rows):
        target = CLASS_MATCH if int(row["match"]) == 1 else CLASS_NONMATCH
        pred = int(predicted[index])
        records.append(
            {
                "split": row["split"],
                "trial_id": int(row["trial_id"]),
                "sample_direction": int(row["sample_direction"]),
                "rule": row["rule"],
                "match": int(row["match"]),
                "test_direction": int(row["test_direction"]),
                "target_class": target,
                "predicted_class": pred,
                "correct": int(pred == target),
                "logit_fixation": float(mean_logits[index, CLASS_FIXATION].item()),
                "logit_nonmatch": float(mean_logits[index, CLASS_NONMATCH].item()),
                "logit_match": float(mean_logits[index, CLASS_MATCH].item()),
            }
        )
    return records


def _subset_accuracy(records: Sequence[Mapping[str, Any]], predicate) -> float:
    selected = [row for row in records if predicate(row)]
    if not selected:
        return 0.0
    return float(sum(int(row["correct"]) for row in selected) / len(selected))


def summarize_predictions(
    records: Sequence[Mapping[str, Any]],
    *,
    timepoint_acc: float | None = None,
    fixation_acc: float | None = None,
) -> dict[str, Any]:
    overall = _subset_accuracy(records, lambda row: True)
    dms = _subset_accuracy(records, lambda row: row["rule"] == RULE_DMS)
    dmrs = _subset_accuracy(records, lambda row: row["rule"] == RULE_DMRS90)
    match = _subset_accuracy(records, lambda row: int(row["match"]) == 1)
    nonmatch = _subset_accuracy(records, lambda row: int(row["match"]) == 0)
    crosses = {}
    for rule in (RULE_DMS, RULE_DMRS90):
        for match_value in (0, 1):
            key = f"{rule.lower()}_match{match_value}"
            crosses[key] = _subset_accuracy(
                records,
                lambda row, rule=rule, match_value=match_value: row["rule"] == rule
                and int(row["match"]) == match_value,
            )
    payload = {
        "n_trials": len(records),
        "trial_accuracy": overall,
        "dms_trial_accuracy": dms,
        "dmrs90_trial_accuracy": dmrs,
        "match_trial_accuracy": match,
        "nonmatch_trial_accuracy": nonmatch,
        "cross_condition_trial_accuracy": crosses,
    }
    if timepoint_acc is not None:
        payload["timepoint_accuracy"] = float(timepoint_acc)
    if fixation_acc is not None:
        payload["fixation_accuracy"] = float(fixation_acc)
    return payload


def fixation_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: MasseDelayedCueConfig,
) -> float:
    windows = window_indices(config)
    fix_logits = logits[windows["fixation"]]
    fix_targets = targets[windows["fixation"]]
    predicted = fix_logits.argmax(dim=-1)
    return float((predicted == fix_targets).float().mean().item())


def timepoint_accuracy_valid_test(
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: MasseDelayedCueConfig,
) -> float:
    windows = window_indices(config)
    valid_logits = logits[windows["valid_test"]]
    valid_targets = targets[windows["valid_test"]]
    predicted = valid_logits.argmax(dim=-1)
    return float((predicted == valid_targets).float().mean().item())


def timepoint_accuracy_valid_test_subset(
    logits: torch.Tensor,
    targets: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    config: MasseDelayedCueConfig,
    predicate,
) -> float:
    selected = [index for index, row in enumerate(rows) if predicate(row)]
    if not selected:
        return 0.0
    windows = window_indices(config)
    index = torch.tensor(selected, device=logits.device, dtype=torch.long)
    valid_logits = logits[windows["valid_test"]][:, index]
    valid_targets = targets[windows["valid_test"]][:, index]
    predicted = valid_logits.argmax(dim=-1)
    return float((predicted == valid_targets).float().mean().item())


def attach_timepoint_breakdown(
    summary: dict[str, Any],
    logits: torch.Tensor,
    targets: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    config: MasseDelayedCueConfig,
) -> dict[str, Any]:
    summary["timepoint_accuracy"] = timepoint_accuracy_valid_test(logits, targets, config)
    summary["dms_timepoint_accuracy"] = timepoint_accuracy_valid_test_subset(
        logits, targets, rows, config, lambda row: row["rule"] == RULE_DMS
    )
    summary["dmrs90_timepoint_accuracy"] = timepoint_accuracy_valid_test_subset(
        logits, targets, rows, config, lambda row: row["rule"] == RULE_DMRS90
    )
    crosses: dict[str, float] = {}
    for rule in (RULE_DMS, RULE_DMRS90):
        for match_value in (0, 1):
            key = f"{rule.lower()}_match{match_value}"
            crosses[key] = timepoint_accuracy_valid_test_subset(
                logits,
                targets,
                rows,
                config,
                lambda row, rule=rule, match_value=match_value: row["rule"] == rule
                and int(row["match"]) == match_value,
            )
    summary["cross_condition_timepoint_accuracy"] = crosses
    return summary


FORMAL_OVERALL_TIMEPOINT_MIN = 0.90
FORMAL_CROSS_CONDITION_MIN = 0.85


def formal_gates_status(metrics: Mapping[str, Any]) -> dict[str, Any]:
    crosses = metrics.get("cross_condition_trial_accuracy", {})
    failed: list[str] = []
    checks = {
        "timepoint_accuracy": float(metrics.get("timepoint_accuracy", 0.0)),
        "dms_timepoint_accuracy": float(metrics.get("dms_timepoint_accuracy", 0.0)),
        "dmrs90_timepoint_accuracy": float(metrics.get("dmrs90_timepoint_accuracy", 0.0)),
    }
    for name, value in checks.items():
        if value < FORMAL_OVERALL_TIMEPOINT_MIN:
            failed.append(f"{name}={value:.4f} < {FORMAL_OVERALL_TIMEPOINT_MIN:.2f}")
    for key in ("dms_match0", "dms_match1", "dmrs90_match0", "dmrs90_match1"):
        value = float(crosses.get(key, 0.0))
        if value < FORMAL_CROSS_CONDITION_MIN:
            failed.append(f"{key}={value:.4f} < {FORMAL_CROSS_CONDITION_MIN:.2f}")
    return {
        "passed": not failed,
        "failed": failed,
        "checks": {**checks, **{key: float(crosses.get(key, 0.0)) for key in (
            "dms_match0",
            "dms_match1",
            "dmrs90_match0",
            "dmrs90_match1",
        )}},
    }
