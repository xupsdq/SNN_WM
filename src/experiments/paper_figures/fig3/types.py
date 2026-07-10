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
    num_sequences: int = 40
    batch_size: int = 16
    peak_q: float = 0.20
    valley_q: float = 0.20
    n_null: int = 100
    weak_cue_target_source: str = "sequence_member_random"
    weak_cue_keep_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    weak_cue_repeats: int = 10
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
    enable_state_bank_batch: bool = False
    state_bank_singleton_batch_size: int = 4
    smoke: bool = False
    peak_cue_main_keep_fraction: float = 0.10
    region_ping_q: float = 0.20
    region_ping_support_metric: str = "gain_ratio_map"
    region_ping_conditions: tuple[str, ...] = ("peak", "valley", "random")
    region_ping_repeats: int = 5
    region_ping_amp_sweep: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5)
    region_ping_use_random_matched: bool = True
    weak_probe_include_singleton: bool = True
    boundary_sequence_lengths: tuple[int, ...] = (3, 5, 7, 10)
    boundary_delay_grid_ms: tuple[int, ...] = (100, 200, 300, 400, 600, 800, 1200, 1500)
    morphology_layer: str = "layer1"
    morphology_variable: str = "g"
    weak_cue_main_keep_prob: float = 0.5
    access_null_quantile: float = 0.95
    cue_specificity_seq_len: int = 7
    cue_specificity_delay_ms: int = 400
    cue_specificity_keep_prob: float = 0.5
    cue_specificity_readout_batch_size: int = 6
    cue_specificity_cue_types: tuple[str, ...] = ("matched", "mismatched", "unseen")

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

    def resolve_sequence_id(self, sequence_id: int, condition_id: str | None = None, delay_ms: int | None = None) -> int:
        if condition_id is None and delay_ms is None:
            return int(sequence_id)
        if "condition_id" not in self.sequence_meta.columns:
            return int(sequence_id)
        meta = self.sequence_meta.copy()
        meta = meta[meta["condition_id"].astype(str).eq(str(condition_id))]
        if delay_ms is not None and "delay_ms" in meta.columns:
            meta = meta[meta["delay_ms"].astype(int).eq(int(delay_ms))]
        source_col = "source_sequence_id" if "source_sequence_id" in meta.columns else "sequence_id"
        meta = meta[meta[source_col].astype(int).eq(int(sequence_id))]
        if meta.empty:
            raise KeyError(
                f"Missing Fig.3 boundary-bank mapping for sequence_id={sequence_id}, "
                f"condition_id={condition_id}, delay_ms={delay_ms}"
            )
        return int(meta.iloc[0]["sequence_id"])

    def sequence_meta_row(self, sequence_id: int, condition_id: str | None = None, delay_ms: int | None = None) -> pd.Series:
        resolved = self.resolve_sequence_id(sequence_id, condition_id=condition_id, delay_ms=delay_ms)
        part = self.sequence_meta[self.sequence_meta["sequence_id"].astype(int).eq(int(resolved))]
        if part.empty:
            raise KeyError(f"Missing Fig.3 sequence_meta row for resolved sequence_id={resolved}")
        return part.iloc[0]

    def get(self, sequence_id: int, state: str, layer: str, variable: str, condition_id: str | None = None, delay_ms: int | None = None) -> np.ndarray:
        sequence_id = self.resolve_sequence_id(sequence_id, condition_id=condition_id, delay_ms=delay_ms)
        if variable == "g":
            return self.arrays[int(sequence_id)][state][layer]["g"]
        return self.arrays[int(sequence_id)][state][layer][variable]

    def singleton_ref(self, sequence_id: int, position: int, layer: str, variable: str, condition_id: str | None = None, delay_ms: int | None = None) -> np.ndarray:
        sequence_id = self.resolve_sequence_id(sequence_id, condition_id=condition_id, delay_ms=delay_ms)
        return self.singleton_refs[int(sequence_id)][int(position)][layer][variable]

    def boundary_for(self, sequence_id: int, state: str, condition_id: str | None = None, delay_ms: int | None = None) -> Mapping[str, Mapping[str, torch.Tensor]]:
        sequence_id = self.resolve_sequence_id(sequence_id, condition_id=condition_id, delay_ms=delay_ms)
        return self.boundaries[int(sequence_id)][state]

    def singleton_boundary_for(self, sequence_id: int, position: int, condition_id: str | None = None, delay_ms: int | None = None) -> Mapping[str, Mapping[str, torch.Tensor]] | None:
        sequence_id = self.resolve_sequence_id(sequence_id, condition_id=condition_id, delay_ms=delay_ms)
        return self.singleton_boundaries.get(int(sequence_id), {}).get(int(position))


__all__ = ["ExperimentContext", "Fig3Config", "MultiItemSequenceLandscapeBank"]
