from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config.units import ms


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


@dataclass(frozen=True)
class Fig4Config:
    model_path: str
    dataset_root: str
    output_root: str
    network_seed: int
    device: str = "auto"
    split: str = "test"
    dt: float = 0.001
    sample_ms: int = 200
    delay_ms: int = 400
    probe_ms: int = 100
    batch_size: int = 16
    max_pairs: int = 500
    num_similarity_bins: int = 5
    num_overlap_bins: int = 3
    overlap_mask_mode: str = "encoded_spike"
    foreground_threshold: float = 0.1
    dilation_radius: int = 1
    random_mask_candidates: int = 32
    n_null: int = 100
    save_full_trace: bool = False
    save_l3_trace: bool = True
    run_pair_sampling: bool = False
    run_rollouts: bool = False
    run_similarity_entry: bool = False
    run_overlap_localization: bool = False
    run_overlap_accuracy_identification: bool = False
    run_decision_spike_displacement: bool = False
    run_decision_deflection: bool = False
    run_overlap_perturbation: bool = False
    run_supplement: bool = False
    num_iso_similarity_bins: int = 20
    overlap_tail_quantile: float = 0.33
    match_similarity_caliper: float = 0.02
    match_energy_caliper: float = 0.15
    match_require_probe_label: bool = False
    match_require_class_pair: bool = False
    require_distinct_pair_labels: bool = True
    min_matches_per_network: int = 20
    n_match_permutations: int = 2000
    save_debug_figures: bool = False
    show_progress: bool = True
    enable_condition_batch: bool = False
    use_legacy_similarity_bias_method: bool = True
    use_legacy_overlap_perturbation_method: bool = True
    use_legacy_l3_accumulator_method: bool = True
    legacy_exact_mode: bool = True
    l3_mask_mode: str = "1x1"
    l3_region_batch_size: int = 16
    temporal_pool: str = "mean"
    save_case_count: int = 4
    run_l3_region_deletion: bool = True
    run_l3_region_replacement: bool = True
    smoke: bool = False

    @property
    def sample_steps(self) -> int:
        return _ms_to_steps(self.sample_ms, self.dt)

    @property
    def delay_steps(self) -> int:
        return _ms_to_steps(self.delay_ms, self.dt)

    @property
    def probe_steps(self) -> int:
        return _ms_to_steps(self.probe_ms, self.dt)


@dataclass
class ExperimentContext:
    cfg: Fig4Config
    seed_dir: Path
    config_dir: Path
    trial_specs_dir: Path
    raw_dir: Path
    metrics_dir: Path
    debug_dir: Path
    device: torch.device
    dataset: Any
    class_index: dict[int, list[int]]
    net: Any
    encoder: Any
    warnings: list[str]
    output_files: dict[str, str]
    completed_modules: dict[str, bool]
    run_log: list[str]
    availability: dict[str, Any] = field(default_factory=dict)
    n_pairs: int = 0


@dataclass
class OverlapReentryDMSBank:
    pair_trials: pd.DataFrame
    perturbation_masks: pd.DataFrame
    rollout_manifest: pd.DataFrame
    condition_metrics: pd.DataFrame
    traces: dict[str, np.ndarray]
    vectors: dict[str, np.ndarray]


@dataclass
class SimilarityBiasCompatibleBank:
    pair_trials: pd.DataFrame
    trial_metrics: pd.DataFrame
    repeat_metrics: pd.DataFrame
    voltage_vectors: dict[str, np.ndarray]


@dataclass
class OverlapPerturbationCompatibleBank:
    pair_trials: pd.DataFrame
    perturbation_masks: pd.DataFrame
    rollout_manifest: pd.DataFrame
    condition_metrics: pd.DataFrame
    traces: dict[str, np.ndarray]
    vectors: dict[str, np.ndarray]
    l3_replay_capture_manifest: pd.DataFrame = field(default_factory=pd.DataFrame)
    l3_replay_captures: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class L3AccumulatorReplayBank:
    pair_trials: pd.DataFrame
    region_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    pair_vectors: dict[str, np.ndarray]
    region_effects: dict[str, np.ndarray]


__all__ = [
    "ExperimentContext",
    "Fig4Config",
    "L3AccumulatorReplayBank",
    "OverlapPerturbationCompatibleBank",
    "OverlapReentryDMSBank",
    "SimilarityBiasCompatibleBank",
]
