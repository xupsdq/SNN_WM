"""Balanced delayed-match-to-sample trial manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class DmsExperimentConfig:
    """Frozen task and recording choices for the recurrent DMS experiment."""

    task_populations: Tuple[int, ...] = (0, 1, 2, 3)
    distractor_population: int = 4
    sample_origin_ms: float = 50.0
    cue_duration_ms: float = 75.0
    delays_ms: Tuple[float, ...] = (125.0, 375.0, 750.0)
    distractor_conditions: Tuple[bool, ...] = (False, True)
    response_bin_ms: float = 50.0
    response_bin_count: int = 3
    silent_window_ms: float = 50.0
    pairs_per_probe: Tuple[int, int, int] = (3, 3, 3)
    stsp_edges_per_source_population: int = 1_024
    poisson_input: bool = True
    seed: int = 921_734

    def __post_init__(self) -> None:
        if len(self.task_populations) < 3:
            raise ValueError("DMS requires at least three task populations.")
        if len(set(self.task_populations)) != len(self.task_populations):
            raise ValueError("Task populations must be unique.")
        if min(self.task_populations) < 0:
            raise ValueError("Task populations must be non-negative.")
        if self.distractor_population in self.task_populations:
            raise ValueError("The distractor population must not be a task item.")
        if self.distractor_population < 0:
            raise ValueError("The distractor population must be non-negative.")
        for name in (
            "sample_origin_ms",
            "cue_duration_ms",
            "response_bin_ms",
            "silent_window_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be finite and positive.".format(name))
        if not self.delays_ms or any(
            not math.isfinite(value) or value <= 0.0 for value in self.delays_ms
        ):
            raise ValueError("DMS delays must be finite and positive.")
        if len(set(self.delays_ms)) != len(self.delays_ms):
            raise ValueError("DMS delays must be unique.")
        if not self.distractor_conditions:
            raise ValueError("At least one distractor condition is required.")
        if self.response_bin_count <= 0:
            raise ValueError("response_bin_count must be positive.")
        if len(self.pairs_per_probe) != len(_SPLITS):
            raise ValueError("pairs_per_probe must define train/validation/test.")
        cycle = len(self.task_populations) - 1
        if any(value <= 0 or value % cycle != 0 for value in self.pairs_per_probe):
            raise ValueError(
                "Each split's pairs_per_probe must be a positive multiple of "
                "len(task_populations) - 1 so sample identity remains label-balanced."
            )
        if self.stsp_edges_per_source_population <= 0:
            raise ValueError("stsp_edges_per_source_population must be positive.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DmsTrial:
    """One immutable trial description; paired trials share future noise."""

    trial_id: str
    pair_id: str
    split: str
    sample_population: int
    probe_population: int
    is_match: bool
    delay_ms: float
    distracted: bool
    pair_repetition: int
    input_seed: int

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _paired_seed(base_seed: int, indices: Sequence[int]) -> int:
    state = np.random.SeedSequence([base_seed, *indices]).generate_state(
        1, dtype=np.uint64
    )[0]
    return int(state % np.uint64(2**63 - 1))


def build_dms_trial_manifest(config: DmsExperimentConfig) -> List[DmsTrial]:
    """Build label/current-input balanced, pairwise-noise-matched trials."""

    trials: List[DmsTrial] = []
    task = config.task_populations
    for split_index, (split, repetitions) in enumerate(
        zip(_SPLITS, config.pairs_per_probe)
    ):
        for delay_index, delay_ms in enumerate(config.delays_ms):
            for distractor_index, distracted in enumerate(
                config.distractor_conditions
            ):
                for probe_index, probe in enumerate(task):
                    alternatives = tuple(item for item in task if item != probe)
                    for repetition in range(repetitions):
                        pair_id = (
                            "{}-d{:g}-x{}-p{}-r{:03d}".format(
                                split,
                                delay_ms,
                                int(distracted),
                                probe,
                                repetition,
                            )
                        )
                        input_seed = _paired_seed(
                            config.seed,
                            (
                                split_index,
                                delay_index,
                                distractor_index,
                                probe_index,
                                repetition,
                            ),
                        )
                        nonmatch_sample = alternatives[
                            repetition % len(alternatives)
                        ]
                        for is_match, sample in (
                            (True, probe),
                            (False, nonmatch_sample),
                        ):
                            trial_id = "{}-{}".format(
                                pair_id, "match" if is_match else "nonmatch"
                            )
                            trials.append(
                                DmsTrial(
                                    trial_id=trial_id,
                                    pair_id=pair_id,
                                    split=split,
                                    sample_population=sample,
                                    probe_population=probe,
                                    is_match=is_match,
                                    delay_ms=delay_ms,
                                    distracted=bool(distracted),
                                    pair_repetition=repetition,
                                    input_seed=input_seed,
                                )
                            )
    validate_dms_trial_manifest(trials, config)
    return trials


def validate_dms_trial_manifest(
    trials: Sequence[DmsTrial], config: DmsExperimentConfig
) -> None:
    """Reject label imbalance and pair/future-input mismatches before simulation."""

    if not trials:
        raise ValueError("The DMS trial manifest is empty.")
    ids = [trial.trial_id for trial in trials]
    if len(ids) != len(set(ids)):
        raise ValueError("DMS trial IDs must be unique.")
    pair_members: Dict[str, List[DmsTrial]] = {}
    for trial in trials:
        pair_members.setdefault(trial.pair_id, []).append(trial)
    for pair_id, members in pair_members.items():
        if len(members) != 2 or {member.is_match for member in members} != {
            False,
            True,
        }:
            raise ValueError("Pair {} must contain one match and one non-match.".format(pair_id))
        immutable = {
            (
                member.split,
                member.probe_population,
                member.delay_ms,
                member.distracted,
                member.input_seed,
            )
            for member in members
        }
        if len(immutable) != 1:
            raise ValueError("Paired trials must share probe, timing, and input seed.")

    task = set(config.task_populations)
    for split in _SPLITS:
        for delay in config.delays_ms:
            for distracted in config.distractor_conditions:
                subset = [
                    trial
                    for trial in trials
                    if trial.split == split
                    and trial.delay_ms == delay
                    and trial.distracted == distracted
                ]
                if not subset:
                    raise ValueError("Every declared DMS condition must contain trials.")
                labels = [trial.is_match for trial in subset]
                if labels.count(True) != labels.count(False):
                    raise ValueError("Match labels must be balanced in every condition.")
                for probe in task:
                    by_probe = [trial for trial in subset if trial.probe_population == probe]
                    probe_labels = [trial.is_match for trial in by_probe]
                    if probe_labels.count(True) != probe_labels.count(False):
                        raise ValueError("Every probe must be label-balanced.")
                for sample in task:
                    by_sample = [trial for trial in subset if trial.sample_population == sample]
                    sample_labels = [trial.is_match for trial in by_sample]
                    if sample_labels.count(True) != sample_labels.count(False):
                        raise ValueError("Every sample must be label-balanced.")


__all__ = [
    "DmsExperimentConfig",
    "DmsTrial",
    "build_dms_trial_manifest",
    "validate_dms_trial_manifest",
]
