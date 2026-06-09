from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as legacy
from src.experiments.paper_figures.fig3.types import ExperimentContext


def build_boundary_condition_specs(ctx: ExperimentContext, sequence_trials: pd.DataFrame) -> pd.DataFrame:
    available_lengths = sorted(int(v) for v in sequence_trials["seq_len"].dropna().unique())
    requested_lengths = [int(v) for v in ctx.cfg.boundary_sequence_lengths]
    lengths = [value for value in requested_lengths if value in set(available_lengths)]
    if not lengths:
        lengths = available_lengths
    rows: list[dict[str, Any]] = []
    for seq_len in lengths:
        n_sequences = int(sequence_trials[sequence_trials["seq_len"].astype(int).eq(int(seq_len))]["sequence_id"].nunique())
        for delay_ms in ctx.cfg.boundary_delay_grid_ms:
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "condition_id": f"K{int(seq_len)}_D{int(delay_ms)}",
                    "seq_len": int(seq_len),
                    "delay_ms": int(delay_ms),
                    "sample_ms": int(ctx.cfg.sample_ms),
                    "n_sequences": int(n_sequences),
                    "morphology_layer": str(ctx.cfg.morphology_layer),
                    "morphology_variable": str(ctx.cfg.morphology_variable),
                    "smoke": bool(ctx.cfg.smoke),
                }
            )
    out = pd.DataFrame(rows)
    legacy._save_csv(ctx, out, ctx.trial_specs_dir / "boundary_condition_specs.csv")
    ctx.completed_modules["boundary_condition_specs"] = True
    return out


def build_access_job_specs(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    condition_specs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    job_id = 0
    repeats = max(1, int(ctx.cfg.weak_cue_repeats))
    if ctx.cfg.smoke:
        repeats = min(repeats, 2)
    ping_repeats = max(1, int(ctx.cfg.ping_repeats))
    for _, condition in condition_specs.iterrows():
        seq_len = int(condition["seq_len"])
        delay_ms = int(condition["delay_ms"])
        condition_id = str(condition["condition_id"])
        matching_sequences = sequence_trials[sequence_trials["seq_len"].astype(int).eq(seq_len)]
        for seq_id, group in matching_sequences.groupby("sequence_id", sort=True):
            ordered = group.sort_values("stage_k")
            labels = [int(v) for v in ordered["item_label"].tolist()]
            image_ids = [int(v) for v in ordered["item_image_id"].tolist()]
            for target_position, (image_id, label) in enumerate(zip(image_ids, labels), start=1):
                for repeat_id in range(repeats):
                    rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "job_id": int(job_id),
                            "job_type": "weak_cue",
                            "condition_id": condition_id,
                            "sequence_id": int(seq_id),
                            "seq_len": seq_len,
                            "delay_ms": delay_ms,
                            "target_position": int(target_position),
                            "target_image_id": int(image_id),
                            "target_label": int(label),
                            "keep_prob": float(ctx.cfg.weak_cue_main_keep_prob),
                            "repeat_id": int(repeat_id),
                            "mask_seed": int(_stable_seed(ctx.cfg.network_seed, seq_id, target_position, repeat_id, 17)),
                            "ping_repeat": -1,
                        }
                    )
                    job_id += 1
            for ping_repeat in range(ping_repeats):
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "job_id": int(job_id),
                        "job_type": "neutral_ping",
                        "condition_id": condition_id,
                        "sequence_id": int(seq_id),
                        "seq_len": seq_len,
                        "delay_ms": delay_ms,
                        "target_position": 0,
                        "target_image_id": -1,
                        "target_label": -1,
                        "keep_prob": float("nan"),
                        "repeat_id": -1,
                        "mask_seed": -1,
                        "ping_repeat": int(ping_repeat),
                    }
                )
                job_id += 1
    out = pd.DataFrame(rows)
    legacy._save_csv(ctx, out, ctx.trial_specs_dir / "access_job_specs.csv")
    ctx.completed_modules["access_job_specs"] = True
    return out


def _stable_seed(network_seed: int, sequence_id: int, target_position: int, repeat_id: int, offset: int) -> int:
    value = (
        int(network_seed) * 1_000_003
        + int(sequence_id) * 10_007
        + int(target_position) * 101
        + int(repeat_id) * 37
        + int(offset)
    )
    return int(np.mod(value, 2**31 - 1))


__all__ = ["build_access_job_specs", "build_boundary_condition_specs"]
