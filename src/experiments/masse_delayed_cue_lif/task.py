"""Deterministic Masse delayed-cue DMS+DMRS trial table and expansion."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .config import (
    CLASS_FIXATION,
    CLASS_MATCH,
    CLASS_NONMATCH,
    DELAY_DECODE_START_MS,
    DELAY_DECODE_STOP_MS,
    GRACE_MS,
    KAPPA,
    N_DIRECTIONS,
    N_INPUT,
    N_MOTION_CHANNELS,
    N_RULE_CHANNELS,
    RULE_DMRS90,
    RULE_DMS,
    RULES,
    RULE_START_MS,
    RULE_STOP_MS,
    SAMPLE_START_MS,
    SAMPLE_STOP_MS,
    TEST_LOSS_WEIGHT,
    TEST_START_MS,
    TEST_STOP_MS,
    TUNING_HEIGHT,
    MasseDelayedCueConfig,
)


TRIAL_COLUMNS = (
    "split",
    "trial_id",
    "sample_direction",
    "rule",
    "match",
    "test_direction",
    "input_seed",
)


def matching_direction(sample_direction: int, rule: str) -> int:
    if rule == RULE_DMS:
        return int(sample_direction) % N_DIRECTIONS
    if rule == RULE_DMRS90:
        return (int(sample_direction) + 2) % N_DIRECTIONS
    raise ValueError(f"Unknown rule {rule!r}")


def ms_to_index(time_ms: float, dt_ms: float) -> int:
    return int(round(time_ms / dt_ms))


def preferred_directions() -> np.ndarray:
    return np.linspace(0.0, 360.0, N_MOTION_CHANNELS, endpoint=False)


def stimulus_directions() -> np.ndarray:
    return np.linspace(0.0, 360.0, N_DIRECTIONS, endpoint=False)


def motion_tuning(direction_index: int) -> np.ndarray:
    stim_deg = stimulus_directions()[int(direction_index) % N_DIRECTIONS]
    pref_deg = preferred_directions()
    cosine = np.cos(np.deg2rad(stim_deg - pref_deg))
    return (np.exp(KAPPA * cosine) / np.exp(KAPPA)).astype(np.float32)


def rule_tuning(rule: str) -> np.ndarray:
    pattern = np.zeros(N_RULE_CHANNELS, dtype=np.float32)
    if rule == RULE_DMS:
        pattern[: N_RULE_CHANNELS // 2] = 1.0
    elif rule == RULE_DMRS90:
        pattern[N_RULE_CHANNELS // 2 :] = 1.0
    else:
        raise ValueError(f"Unknown rule {rule!r}")
    return pattern


def _basic_conditions() -> list[tuple[int, str, int]]:
    return [
        (sample, rule, match)
        for sample in range(N_DIRECTIONS)
        for rule in RULES
        for match in (0, 1)
    ]


def _nonmatch_test_direction(match_direction: int, input_seed: int) -> int:
    options = [index for index in range(N_DIRECTIONS) if index != match_direction]
    local = np.random.default_rng(int(input_seed))
    return int(local.choice(options))


def generate_trial_table(config: MasseDelayedCueConfig) -> list[dict[str, object]]:
    counts = {"train": config.n_train, "val": config.n_val, "test": config.n_test}
    for split, count in counts.items():
        if count % 32 != 0:
            raise ValueError(f"{split} trial count {count} must be a multiple of 32")

    rng = np.random.default_rng(config.trial_table_seed)
    conditions = _basic_conditions()
    rows: list[dict[str, object]] = []
    trial_id = 0
    input_seed = int(config.trial_table_seed) * 1_000_000 + 1

    for split, count in counts.items():
        repeats = count // 32
        template = conditions * repeats
        order = rng.permutation(len(template))
        for template_index in order:
            sample_direction, rule, match = template[int(template_index)]
            match_dir = matching_direction(sample_direction, rule)
            if match == 1:
                test_direction = match_dir
            else:
                test_direction = _nonmatch_test_direction(match_dir, input_seed)
            rows.append(
                {
                    "split": split,
                    "trial_id": trial_id,
                    "sample_direction": int(sample_direction),
                    "rule": rule,
                    "match": int(match),
                    "test_direction": int(test_direction),
                    "input_seed": int(input_seed),
                }
            )
            trial_id += 1
            input_seed += 1
    return rows


def save_trial_table(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRIAL_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in TRIAL_COLUMNS})
    return path


def validate_trial_table(path: Path, config: MasseDelayedCueConfig) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"trial table does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if fields != TRIAL_COLUMNS:
            raise ValueError(
                f"trial table schema mismatch: expected {TRIAL_COLUMNS}, got {fields}"
            )
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"trial table is empty: {path}")
    trial_ids: list[int] = []
    input_seeds: list[int] = []
    split_counts: Counter[str] = Counter()
    for index, raw in enumerate(raw_rows):
        missing = [column for column in TRIAL_COLUMNS if raw.get(column, "") == ""]
        if missing:
            raise ValueError(f"trial table row {index} missing fields: {missing}")
        split = str(raw["split"])
        if split not in {"train", "val", "test"}:
            raise ValueError(f"trial table row {index} has unknown split {split!r}")
        rule = str(raw["rule"])
        if rule not in RULES:
            raise ValueError(f"trial table row {index} has unknown rule {rule!r}")
        trial_id = int(raw["trial_id"])
        sample_direction = int(raw["sample_direction"])
        test_direction = int(raw["test_direction"])
        match = int(raw["match"])
        input_seed = int(raw["input_seed"])
        if match not in (0, 1):
            raise ValueError(f"trial table row {index} has invalid match {match}")
        if not 0 <= sample_direction < N_DIRECTIONS:
            raise ValueError(f"trial table row {index} has invalid sample_direction")
        if not 0 <= test_direction < N_DIRECTIONS:
            raise ValueError(f"trial table row {index} has invalid test_direction")
        trial_ids.append(trial_id)
        input_seeds.append(input_seed)
        split_counts[split] += 1
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("trial table contains duplicate trial_id values")
    if len(set(input_seeds)) != len(input_seeds):
        raise ValueError("trial table contains duplicate input_seed values")
    expected = {"train": config.n_train, "val": config.n_val, "test": config.n_test}
    actual = {split: int(split_counts.get(split, 0)) for split in expected}
    if actual != expected:
        raise ValueError(
            f"trial table split counts {actual} do not match config {expected}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "n_rows": len(raw_rows),
        "n_train": actual["train"],
        "n_val": actual["val"],
        "n_test": actual["test"],
        "trials_sha256": digest,
    }


def load_trial_table(path: Path, split: str | None = None) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            row = {
                "split": raw["split"],
                "trial_id": int(raw["trial_id"]),
                "sample_direction": int(raw["sample_direction"]),
                "rule": raw["rule"],
                "match": int(raw["match"]),
                "test_direction": int(raw["test_direction"]),
                "input_seed": int(raw["input_seed"]),
            }
            if split is None or row["split"] == split:
                rows.append(row)
    return rows


def window_indices(config: MasseDelayedCueConfig) -> dict[str, slice]:
    dt = config.dt_ms
    sample = slice(ms_to_index(SAMPLE_START_MS, dt), ms_to_index(SAMPLE_STOP_MS, dt))
    rule = slice(ms_to_index(RULE_START_MS, dt), ms_to_index(RULE_STOP_MS, dt))
    test = slice(ms_to_index(TEST_START_MS, dt), ms_to_index(TEST_STOP_MS, dt))
    grace = slice(
        ms_to_index(TEST_START_MS, dt),
        ms_to_index(TEST_START_MS + GRACE_MS, dt),
    )
    valid_test = slice(
        ms_to_index(TEST_START_MS + GRACE_MS, dt),
        ms_to_index(TEST_STOP_MS, dt),
    )
    delay_end_decode = slice(
        ms_to_index(DELAY_DECODE_START_MS, dt),
        ms_to_index(DELAY_DECODE_STOP_MS, dt),
    )
    fixation = slice(0, ms_to_index(TEST_START_MS, dt))
    return {
        "sample": sample,
        "rule": rule,
        "test": test,
        "grace": grace,
        "valid_test": valid_test,
        "delay_end_decode": delay_end_decode,
        "fixation": fixation,
    }


def expand_trial(
    row: dict[str, object],
    config: MasseDelayedCueConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_steps = config.n_steps
    windows = window_indices(config)
    inputs = np.zeros((n_steps, N_INPUT), dtype=np.float32)
    targets = np.full(n_steps, CLASS_FIXATION, dtype=np.int64)
    weights = np.ones(n_steps, dtype=np.float32)

    motion = motion_tuning(int(row["sample_direction"])) * TUNING_HEIGHT * config.input_gain
    test_motion = motion_tuning(int(row["test_direction"])) * TUNING_HEIGHT * config.input_gain
    rule = rule_tuning(str(row["rule"])) * TUNING_HEIGHT * config.input_gain

    inputs[windows["sample"], :N_MOTION_CHANNELS] = motion
    inputs[windows["rule"], N_MOTION_CHANNELS:] = rule
    inputs[windows["test"], :N_MOTION_CHANNELS] = test_motion

    test_class = CLASS_MATCH if int(row["match"]) == 1 else CLASS_NONMATCH
    targets[windows["test"]] = test_class
    weights[windows["grace"]] = 0.0
    weights[windows["valid_test"]] = TEST_LOSS_WEIGHT

    if config.input_noise:
        raise ValueError("Input noise is disabled for the first delivery")

    return (
        torch.from_numpy(inputs),
        torch.from_numpy(targets),
        torch.from_numpy(weights),
    )


def expand_rows(
    rows: Iterable[dict[str, object]],
    config: MasseDelayedCueConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expanded = [expand_trial(row, config) for row in rows]
    inputs = torch.stack([item[0] for item in expanded], dim=1)
    targets = torch.stack([item[1] for item in expanded], dim=1)
    weights = torch.stack([item[2] for item in expanded], dim=1)
    return inputs, targets, weights
