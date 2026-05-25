from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.config.units import ms


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


@dataclass(frozen=True)
class Fig5Config:
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
    batch_size: int = 8
    max_trials: int = 500
    foreground_threshold: float = 0.1
    min_overlap_area: int = 4
    min_probe_only_area: int = 4
    medium_q_low: float = 0.35
    medium_q_high: float = 0.65
    early_window_ms: int = 15
    drive_score_threshold: float = 0.05
    local_kernel_radius: int = 2
    peak_support_q: float = 0.20
    perturbation_mode: str = "attenuate_reset"
    perturbation_attenuation_factor: float = 0.5
    fig5d_include_balanced: bool = False
    event_align_pre_steps: int = 8
    event_align_post_steps: int = 12
    chain_pre_spike_steps: int = 4
    chain_post_spike_steps: int = 6
    n_null: int = 100
    save_full_traces: bool = False
    save_spike_cache: bool = False
    run_trial_sampling: bool = False
    run_preprobe_support: bool = False
    run_early_firing: bool = False
    run_local_events: bool = False
    run_support_perturbation: bool = False
    run_supplement: bool = False
    save_debug_figures: bool = False
    show_progress: bool = True
    enable_branch_batch: bool = False
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

    @property
    def early_window_steps(self) -> int:
        return min(self.probe_steps, _ms_to_steps(self.early_window_ms, self.dt))


@dataclass(frozen=True)
class TrialSpec:
    trial_id: int
    sample_image_id: int
    probe_image_id: int
    sample_label: int
    probe_label: int
    overlap_area: int
    probe_only_area: int
    overlap_quantile: float
    selected_trial_group: str
    input_energy_sample: float
    input_energy_probe: float
    pixel_similarity: float
    dice_overlap: float
    class_pair: str
    trial_seed: int


@dataclass(frozen=True)
class UnitGroupEntry:
    trial_id: int
    unit_id: int
    row: int
    col: int
    unit_group: str
    overlap_drive_score: float
    probe_only_drive_score: float
    support_value: float


@dataclass(frozen=True)
class LocalEventEntry:
    trial_id: int
    event_id: int
    winner_unit_idx: int
    loser_unit_idx: int
    winner_time: int
    loser_time_dynamic: int
    loser_time_static: int


@dataclass(frozen=True)
class PerturbationSetEntry:
    trial_id: int
    condition: str
    unit_id: int
    unit_group: str
    original_support: float
    perturbed_support: float


@dataclass
class ExperimentContext:
    cfg: Fig5Config
    seed_dir: Path
    config_dir: Path
    trial_specs_dir: Path
    raw_dir: Path
    metrics_dir: Path
    debug_dir: Path
    device: torch.device
    dataset: Any
    class_index: dict[int, list[int]]
    net: Any | None
    encoder: Any | None
    warnings: list[str]
    output_files: dict[str, str]
    completed_modules: dict[str, bool]
    run_log: list[str]
    availability: dict[str, Any] = field(default_factory=dict)
    n_trials: int = 0
    n_events: int = 0


@dataclass
class BranchTrace:
    spikes: np.ndarray
    v_effective: np.ndarray
    inhibition: np.ndarray
    layer3_spikes: np.ndarray
    prediction: int
    first_fire_time: int


@dataclass
class LocalSupportCompetitionBank:
    trials: pd.DataFrame
    support_maps: dict[int, np.ndarray]
    branch_traces: dict[int, dict[str, BranchTrace]]
    boundary_states: dict[int, Mapping[str, Mapping[str, torch.Tensor]]]
    unit_groups: pd.DataFrame
    perturbation_sets: pd.DataFrame
    perturbation_ux_audit: pd.DataFrame
    l1_stsp_perturbation_audit: pd.DataFrame


__all__ = [
    "BranchTrace",
    "ExperimentContext",
    "Fig5Config",
    "LocalEventEntry",
    "LocalSupportCompetitionBank",
    "PerturbationSetEntry",
    "TrialSpec",
    "UnitGroupEntry",
]
