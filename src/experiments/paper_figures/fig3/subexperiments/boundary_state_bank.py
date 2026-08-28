from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig3.subexperiments.state_bank import run_multiitem_sequence_state_bank
from src.experiments.paper_figures.fig3.types import ExperimentContext, MultiItemSequenceLandscapeBank


def materialize_boundary_state_bank(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank | None,
    condition_specs: pd.DataFrame,
    sequence_trials: pd.DataFrame | None = None,
) -> MultiItemSequenceLandscapeBank:
    if bank is not None and "condition_id" in bank.sequence_meta.columns:
        _write_boundary_manifest(ctx, bank.sequence_meta)
        ctx.completed_modules["boundary_state_bank"] = True
        return bank
    unsupported_lengths = sorted(
        int(v)
        for v in condition_specs["seq_len"].dropna().unique()
        if bank is not None and int(v) not in set(int(x) for x in bank.sequence_meta["seq_len"].dropna().unique())
    )
    if unsupported_lengths:
        raise ValueError(f"Fig.3 boundary_state_bank missing captured sequence lengths: {unsupported_lengths}")
    delays = sorted(int(v) for v in condition_specs["delay_ms"].dropna().unique())
    source_by_delay: dict[int, MultiItemSequenceLandscapeBank] = {}
    if bank is not None and int(ctx.cfg.delay_ms) in set(delays):
        source_by_delay[int(ctx.cfg.delay_ms)] = bank
    missing_delays = [delay for delay in delays if delay not in source_by_delay]
    if missing_delays:
        if sequence_trials is None:
            raise ValueError(
                "Fig.3 boundary_state_bank requires sequence_trials to materialize multiple delay values "
                f"without regenerating downstream tasks; missing delays={missing_delays}."
            )
        for delay in missing_delays:
            delay_ctx = replace(ctx, cfg=replace(ctx.cfg, delay_ms=int(delay)))
            source_by_delay[int(delay)] = run_multiitem_sequence_state_bank(delay_ctx, sequence_trials)

    max_source_id = 0
    for source_bank in source_by_delay.values():
        if not source_bank.sequence_meta.empty:
            max_source_id = max(max_source_id, int(source_bank.sequence_meta["sequence_id"].astype(int).max()))
    stride = max(max_source_id + 1, 1000)
    arrays = {}
    singleton_refs = {}
    singleton_boundaries = {}
    boundaries = {}
    landscapes = {}
    meta_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for condition_index, (_, condition) in enumerate(condition_specs.iterrows(), start=1):
        seq_len = int(condition["seq_len"])
        delay_ms = int(condition["delay_ms"])
        condition_id = str(condition["condition_id"])
        source_bank = source_by_delay[delay_ms]
        for _, meta in source_bank.sequence_meta[source_bank.sequence_meta["seq_len"].astype(int).eq(seq_len)].iterrows():
            source_seq_id = int(meta["sequence_id"])
            boundary_seq_id = int(condition_index * stride + source_seq_id)
            arrays[boundary_seq_id] = source_bank.arrays[source_seq_id]
            singleton_refs[boundary_seq_id] = source_bank.singleton_refs[source_seq_id]
            singleton_boundaries[boundary_seq_id] = source_bank.singleton_boundaries[source_seq_id]
            boundaries[boundary_seq_id] = source_bank.boundaries[source_seq_id]
            landscapes[boundary_seq_id] = source_bank.landscapes.get(source_seq_id, {})
            meta_row = dict(meta)
            meta_row.update(
                {
                    "sequence_id": boundary_seq_id,
                    "source_sequence_id": source_seq_id,
                    "condition_id": condition_id,
                    "delay_ms": delay_ms,
                    "sample_ms": int(ctx.cfg.sample_ms),
                    "morphology_layer": str(ctx.cfg.morphology_layer),
                    "morphology_variable": str(ctx.cfg.morphology_variable),
                }
            )
            meta_rows.append(meta_row)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "condition_id": condition_id,
                    "sequence_id": boundary_seq_id,
                    "source_sequence_id": source_seq_id,
                    "seq_len": seq_len,
                    "delay_ms": delay_ms,
                    "source_state_condition": "S_final",
                    "morphology_layer": str(ctx.cfg.morphology_layer),
                    "morphology_variable": str(ctx.cfg.morphology_variable),
                    "boundary_source": "fig3.state_bank",
                }
            )
    _save_csv(ctx, pd.DataFrame(rows), ctx.raw_dir / "boundary_state_bank_manifest.csv")
    ctx.completed_modules["boundary_state_bank"] = True
    return MultiItemSequenceLandscapeBank(
        sequence_trials=sequence_trials.reset_index(drop=True).copy() if sequence_trials is not None else bank.sequence_trials.reset_index(drop=True).copy(),
        sequence_meta=pd.DataFrame(meta_rows),
        arrays=arrays,
        singleton_refs=singleton_refs,
        singleton_boundaries=singleton_boundaries,
        boundaries=boundaries,
        landscapes=landscapes,
    )


def _write_boundary_manifest(ctx: ExperimentContext, sequence_meta: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    for _, meta in sequence_meta.iterrows():
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "condition_id": str(meta.get("condition_id", "")),
                "sequence_id": int(meta["sequence_id"]),
                "source_sequence_id": int(meta.get("source_sequence_id", meta["sequence_id"])),
                "seq_len": int(meta["seq_len"]),
                "delay_ms": int(meta.get("delay_ms", ctx.cfg.delay_ms)),
                "source_state_condition": "S_final",
                "morphology_layer": str(meta.get("morphology_layer", ctx.cfg.morphology_layer)),
                "morphology_variable": str(meta.get("morphology_variable", ctx.cfg.morphology_variable)),
                "boundary_source": "fig3.boundary_state_bank",
            }
        )
    _save_csv(ctx, pd.DataFrame(rows), ctx.raw_dir / "boundary_state_bank_manifest.csv")


__all__ = ["materialize_boundary_state_bank"]
