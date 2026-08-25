"""Task-table semantics for Masse delayed-cue DMS+DMRS."""

from __future__ import annotations

from collections import Counter

import torch

from src.experiments.masse_delayed_cue_lif.config import (
    GRACE_MS,
    N_DIRECTIONS,
    RULE_DMRS90,
    RULE_DMS,
    RULE_START_MS,
    RULE_STOP_MS,
    SAMPLE_START_MS,
    SAMPLE_STOP_MS,
    TEST_LOSS_WEIGHT,
    TEST_START_MS,
    TEST_STOP_MS,
    smoke_config,
)
from src.experiments.masse_delayed_cue_lif.task import (
    expand_trial,
    generate_trial_table,
    matching_direction,
    window_indices,
)


def _independent_match_direction(sample_direction: int, rule: str) -> int:
    if rule == "DMS":
        return sample_direction % 8
    if rule == "DMRS90":
        return (sample_direction + 2) % 8
    raise AssertionError(rule)


def test_each_split_covers_32_conditions_evenly():
    rows = generate_trial_table(smoke_config())
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split]
        keys = [(row["sample_direction"], row["rule"], row["match"]) for row in selected]
        counts = Counter(keys)
        assert len(counts) == 32
        assert len(set(counts.values())) == 1


def test_dms_match_test_equals_sample():
    rows = generate_trial_table(smoke_config())
    matched = [row for row in rows if row["rule"] == "DMS" and int(row["match"]) == 1]
    assert matched
    for row in matched:
        assert row["test_direction"] == row["sample_direction"]
        assert row["test_direction"] == _independent_match_direction(
            int(row["sample_direction"]), "DMS"
        )


def test_dmrs90_match_test_equals_sample_plus_two_mod_eight():
    rows = generate_trial_table(smoke_config())
    matched = [row for row in rows if row["rule"] == "DMRS90" and int(row["match"]) == 1]
    assert matched
    for row in matched:
        expected = _independent_match_direction(int(row["sample_direction"]), "DMRS90")
        assert row["test_direction"] == expected
        assert matching_direction(int(row["sample_direction"]), RULE_DMRS90) == expected


def test_nonmatch_never_equals_rule_match_direction():
    rows = generate_trial_table(smoke_config())
    nonmatch = [row for row in rows if int(row["match"]) == 0]
    assert nonmatch
    for row in nonmatch:
        match_dir = _independent_match_direction(int(row["sample_direction"]), str(row["rule"]))
        assert row["test_direction"] != match_dir
        assert 0 <= int(row["test_direction"]) < N_DIRECTIONS


def test_sample_rule_and_test_occupy_specified_windows_only():
    config = smoke_config()
    row = generate_trial_table(config)[0]
    inputs, targets, weights = expand_trial(row, config)
    windows = window_indices(config)
    motion = inputs[:, :24]
    rule = inputs[:, 24:]
    assert torch.count_nonzero(motion[: windows["sample"].start]) == 0
    assert torch.count_nonzero(motion[windows["sample"]]) > 0
    assert torch.count_nonzero(motion[windows["sample"].stop : windows["test"].start]) == 0
    assert torch.count_nonzero(motion[windows["test"]]) > 0
    assert torch.count_nonzero(rule[: windows["rule"].start]) == 0
    assert torch.count_nonzero(rule[windows["rule"]]) > 0
    assert torch.count_nonzero(rule[windows["rule"].stop :]) == 0
    assert torch.equal(targets[windows["fixation"]], torch.zeros_like(targets[windows["fixation"]]))
    test_class = 2 if int(row["match"]) == 1 else 1
    assert torch.equal(targets[windows["test"]], torch.full_like(targets[windows["test"]], test_class))
    assert torch.all(weights[windows["grace"]] == 0)
    assert torch.all(weights[windows["valid_test"]] == TEST_LOSS_WEIGHT)
    assert windows["sample"].start == int(round(SAMPLE_START_MS / config.dt_ms))
    assert windows["sample"].stop == int(round(SAMPLE_STOP_MS / config.dt_ms))
    assert windows["rule"].start == int(round(RULE_START_MS / config.dt_ms))
    assert windows["rule"].stop == int(round(RULE_STOP_MS / config.dt_ms))
    assert windows["test"].start == int(round(TEST_START_MS / config.dt_ms))
    assert windows["grace"].stop == int(round((TEST_START_MS + GRACE_MS) / config.dt_ms))
    assert windows["test"].stop == int(round(TEST_STOP_MS / config.dt_ms))
    _ = RULE_DMS


def test_first_delay_does_not_leak_rule():
    config = smoke_config()
    rows = generate_trial_table(config)
    windows = window_indices(config)
    dms = next(row for row in rows if row["rule"] == RULE_DMS)
    dmrs = next(row for row in rows if row["rule"] == RULE_DMRS90)
    dms_input, _, _ = expand_trial(dms, config)
    dmrs_input, _, _ = expand_trial(dmrs, config)
    pre_rule = slice(windows["sample"].stop, windows["rule"].start)
    assert torch.equal(dms_input[pre_rule, 24:], torch.zeros_like(dms_input[pre_rule, 24:]))
    assert torch.equal(dmrs_input[pre_rule, 24:], torch.zeros_like(dmrs_input[pre_rule, 24:]))
    assert not torch.equal(dms_input[windows["rule"], 24:], dmrs_input[windows["rule"], 24:])


def test_same_seed_same_table_and_splits_do_not_share_ids():
    first = generate_trial_table(smoke_config(trial_table_seed=0))
    second = generate_trial_table(smoke_config(trial_table_seed=0))
    third = generate_trial_table(smoke_config(trial_table_seed=1))
    assert first == second
    assert first != third
    trial_ids = [row["trial_id"] for row in first]
    input_seeds = [row["input_seed"] for row in first]
    assert len(trial_ids) == len(set(trial_ids))
    assert len(input_seeds) == len(set(input_seeds))
    by_split = {}
    for row in first:
        by_split.setdefault(row["split"], {"ids": set(), "seeds": set()})
        by_split[row["split"]]["ids"].add(row["trial_id"])
        by_split[row["split"]]["seeds"].add(row["input_seed"])
    assert by_split["train"]["ids"].isdisjoint(by_split["val"]["ids"])
    assert by_split["train"]["ids"].isdisjoint(by_split["test"]["ids"])
    assert by_split["val"]["ids"].isdisjoint(by_split["test"]["ids"])
    assert by_split["train"]["seeds"].isdisjoint(by_split["val"]["seeds"])
    assert by_split["train"]["seeds"].isdisjoint(by_split["test"]["seeds"])
    assert by_split["val"]["seeds"].isdisjoint(by_split["test"]["seeds"])
