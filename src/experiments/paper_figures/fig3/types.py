from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.config.units import ms


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


@dataclass(frozen=True)
class Fig3Config:
    model_path: str
    dataset_root: str
    output_root: str
    network_seed: int
    device: str = "auto"
    split: str = "test"
    dt: float = 0.001
    sequence_lengths: tuple[int, ...] = (10,)
    primary_sequence_length: int = 10
    main_sequence_length: int = 10
    main_only_seq_len_10: bool = True
    sample_ms: int = 200
    delay_ms: int = 200
    ping_ms: int = 30
    ping_amp: float = 1.0
    ping_repeats: int = 1
    ping_main_state_conditions: tuple[str, ...] = ("S_final", "S0")
    weak_probe_ms: int = 100
    weak_probe_keep_probs: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    weak_probe_repeats: int = 5
    weak_probe_mask_space: str = "encoded_spikes"
    weak_probe_use_same_mask_across_states: bool = True
    weak_probe_scale: float = 0.35
    weak_probe_noise: float = 0.0
    weak_probe_metric_mode: str = "fig2_compat"
    weak_probe_target_source: str = "sequence_member_random"
    weak_probe_memory_scope: str = "final_only"
    num_sequences: int = 100
    batch_size: int = 16
    peak_q: float = 0.20
    valley_q: float = 0.20
    n_null: int = 100
    weak_cue_target_source: str = "sequence_member_random"
    weak_cue_keep_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    weak_cue_repeats: int = 5
    weak_cue_mask_mode: str = "rank_within_target_foreground"
    foreground_threshold: float = 0.1
    functional_restore_mode: str = "stsp_only"
    partial_cue_keep_fraction: float = 0.10
    partial_cue_keep_fraction_sweep: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3)
    partial_cue_repeats: int = 20
    target_position: str = "K-1"
    run_state_bank: bool = False
    run_progressive_update: bool = False
    run_peak_valley_landscape: bool = False
    run_neutral_ping: bool = False
    run_weak_probe: bool = False
    run_region_ping: bool = False
    run_region_ping_s0_control: bool = False
    run_region_ping_amp_sweep: bool = False
    run_peak_aligned_completion: bool = False
    run_peak_cue_main: bool = False
    run_population_morphology_supplement: bool = False
    run_structural_weak_cue: bool = False
    run_structural_weak_cue_supplement: bool = False
    run_supplement: bool = False
    save_debug_figures: bool = False
    save_spike_cache: bool = False
    save_all_layer_state_bank: bool = False
    show_progress: bool = True
    use_encode_cache: bool = True
    enable_condition_batch: bool = False
    smoke: bool = False
    peak_cue_main_keep_fraction: float = 0.10
    region_ping_q: float = 0.20
    region_ping_support_metric: str = "gain_ratio_map"
    region_ping_conditions: tuple[str, ...] = ("peak", "valley", "random")
    region_ping_repeats: int = 5
    region_ping_amp_sweep: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5)
    region_ping_use_random_matched: bool = True
    weak_probe_include_singleton: bool = True

    @property
    def sample_steps(self) -> int:
        return _ms_to_steps(self.sample_ms, self.dt)

    @property
    def delay_steps(self) -> int:
        return _ms_to_steps(self.delay_ms, self.dt)

    @property
    def ping_steps(self) -> int:
        return _ms_to_steps(self.ping_ms, self.dt)

    @property
    def weak_probe_steps(self) -> int:
        return _ms_to_steps(self.weak_probe_ms, self.dt)


@dataclass
class ExperimentContext:
    cfg: Fig3Config
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
    n_sequences: int = 0


@dataclass
class MultiItemSequenceLandscapeBank:
    sequence_trials: pd.DataFrame
    sequence_meta: pd.DataFrame
    arrays: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]]
    singleton_refs: dict[int, dict[int, dict[str, dict[str, np.ndarray]]]]
    singleton_boundaries: dict[int, dict[int, Mapping[str, Mapping[str, torch.Tensor]]]]
    boundaries: dict[int, dict[str, Mapping[str, Mapping[str, torch.Tensor]]]]
    landscapes: dict[int, dict[str, np.ndarray]]

    def get(self, sequence_id: int, state: str, layer: str, variable: str) -> np.ndarray:
        if variable == "g":
            return self.arrays[int(sequence_id)][state][layer]["g"]
        return self.arrays[int(sequence_id)][state][layer][variable]


__all__ = ["ExperimentContext", "Fig3Config", "MultiItemSequenceLandscapeBank"]
