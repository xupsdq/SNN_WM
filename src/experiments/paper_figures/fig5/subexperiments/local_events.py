from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config.units import ms
from src.experiments.paper_figures.common.bundle_io import relative_to_root as _rel, save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig5.constants import (
    LATE_PRE_WINDOW_MS,
    MAX_WINNERS_PER_TRIAL,
    PANEL_C_EVENT_COLUMNS,
    PRIMARY_PRE_WINDOW_MS,
    SUPP_EVENT_AUDIT_COLUMNS,
)
from src.experiments.paper_figures.fig5.subexperiments.helpers import (
    _advanced_or_recruited_units,
    _aligned_delta,
    _delayed_or_lost_units,
    _event_audit_row,
    _first_spike_map,
    _is_loser_suppressed,
    _latency_delta,
    _nearest_loser,
    _neighborhood_radius_robustness,
    _progress,
    _spikes_earlier,
)
from src.experiments.paper_figures.fig5.types import ExperimentContext, LocalSupportCompetitionBank


def compute_event_aligned_metrics(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> None:
    event_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    raw_trace_payload: dict[str, np.ndarray] = {}
    time_axis_steps = np.arange(-ctx.cfg.event_align_pre_steps, ctx.cfg.event_align_post_steps + 1, dtype=int)
    time_axis_ms = time_axis_steps.astype(float) * float(ctx.cfg.dt / ms)
    raw_trace_payload["time_axis_steps"] = time_axis_steps
    raw_trace_payload["time_axis_ms"] = time_axis_ms
    event_id = 0

    for trial in _progress(bank.trials.itertuples(index=False), total=len(bank.trials), desc="fig5 local events", enabled=ctx.cfg.show_progress):
        trial_id = int(trial.trial_id)
        groups = bank.unit_groups[bank.unit_groups["trial_id"].eq(trial_id)]
        dynamic = bank.branch_traces[trial_id]["dynamic_intact"]
        static = bank.branch_traces[trial_id]["static_frozen"]
        first_dyn = _first_spike_map(dynamic.spikes)
        first_sta = _first_spike_map(static.spikes)
        winners = groups[
            groups["unit_group"].eq("overlap_dominant")
            & groups["unit_id"].isin(_advanced_or_recruited_units(first_dyn, first_sta))
        ].copy()
        losers = groups[
            groups["unit_group"].isin(["probe_only_dominant", "balanced"])
            & groups["unit_id"].isin(_delayed_or_lost_units(first_dyn, first_sta))
        ].copy()
        if losers.empty:
            losers = groups[groups["unit_group"].eq("probe_only_dominant")].copy()
        ranked_winners = winners.sort_values("support_value", ascending=False)
        for win in ranked_winners.iloc[MAX_WINNERS_PER_TRIAL:].itertuples(index=False):
            audit_rows.append(_event_audit_row(ctx, trial_id, -1, "winner_loser_pair", False, "trial_rank_gt_3", win.unit_group, "", float(win.overlap_drive_score), float("nan")))
        for selection_rank, win in enumerate(ranked_winners.head(MAX_WINNERS_PER_TRIAL).itertuples(index=False), start=1):
            loser = _nearest_loser(win, losers, ctx.cfg.local_kernel_radius)
            if loser is None:
                loser = _nearest_loser(win, losers, max(ctx.cfg.local_kernel_radius, 6))
            if loser is None:
                audit_rows.append(_event_audit_row(ctx, trial_id, event_id, "winner_loser_pair", False, "no_local_loser", win.unit_group, "", float(win.overlap_drive_score), float("nan")))
                continue
            t0 = int(first_dyn[int(win.row), int(win.col)])
            if t0 < 0:
                continue
            winner_delta_v = _aligned_delta(dynamic.v_effective[:, int(win.row), int(win.col)], static.v_effective[:, int(win.row), int(win.col)], t0, ctx)
            loser_delta_v = _aligned_delta(dynamic.v_effective[:, int(loser.row), int(loser.col)], static.v_effective[:, int(loser.row), int(loser.col)], t0, ctx)
            loser_inh = _aligned_delta(dynamic.inhibition[:, int(loser.row), int(loser.col)], static.inhibition[:, int(loser.row), int(loser.col)], t0, ctx)
            if not all(np.isfinite(values).all() for values in (winner_delta_v, loser_delta_v, loser_inh)):
                audit_rows.append(_event_audit_row(ctx, trial_id, -1, "winner_loser_pair", False, "incomplete_alignment_window", win.unit_group, loser.unit_group, float(win.overlap_drive_score), float(loser.overlap_drive_score)))
                continue
            winner_minus_loser = winner_delta_v - loser_delta_v
            primary_mask = _window_mask(time_axis_ms, PRIMARY_PRE_WINDOW_MS)
            late_mask = _window_mask(time_axis_ms, LATE_PRE_WINDOW_MS)
            post = slice(ctx.cfg.event_align_pre_steps + 1, None)
            winner_full_pre = _masked_mean(winner_delta_v, primary_mask)
            loser_full_pre = _masked_mean(loser_delta_v, primary_mask)
            row = {
                "network_seed": int(ctx.cfg.network_seed),
                "trial_id": trial_id,
                "event_id": int(event_id),
                "selection_rank_within_trial": int(selection_rank),
                "winner_unit_idx": int(win.unit_id),
                "loser_unit_idx": int(loser.unit_id),
                "winner_group": str(win.unit_group),
                "loser_group": str(loser.unit_group),
                "winner_first_spike_dynamic": int(first_dyn[int(win.row), int(win.col)]),
                "winner_first_spike_static": int(first_sta[int(win.row), int(win.col)]),
                "loser_first_spike_dynamic": int(first_dyn[int(loser.row), int(loser.col)]),
                "loser_first_spike_static": int(first_sta[int(loser.row), int(loser.col)]),
                "winner_pre_spike_delta_v_mean": winner_full_pre,
                "winner_pre_spike_boost": bool(winner_full_pre > 0.0),
                "winner_full_pre_delta_v_mean": winner_full_pre,
                "loser_full_pre_delta_v_mean": loser_full_pre,
                "winner_minus_loser_full_pre_delta_v_mean": _masked_mean(winner_minus_loser, primary_mask),
                "winner_late_pre_delta_v_mean": _masked_mean(winner_delta_v, late_mask),
                "loser_late_pre_delta_v_mean": _masked_mean(loser_delta_v, late_mask),
                "winner_minus_loser_late_pre_delta_v_mean": _masked_mean(winner_minus_loser, late_mask),
                "complete_alignment_window": True,
                "winner_spikes_earlier": bool(_spikes_earlier(first_dyn[int(win.row), int(win.col)], first_sta[int(win.row), int(win.col)])),
                "loser_post_winner_delta_v_mean": float(np.nanmean(loser_delta_v[post])),
                "loser_post_winner_inh_rise": float(np.nanmean(loser_inh[post])),
                "loser_post_winner_suppressed": bool(np.nanmean(loser_delta_v[post]) < 0.0 or _is_loser_suppressed(first_dyn[int(loser.row), int(loser.col)], first_sta[int(loser.row), int(loser.col)])),
                "winner_loser_latency_gap": _latency_delta(first_dyn[int(loser.row), int(loser.col)], first_dyn[int(win.row), int(win.col)]),
                "neighborhood_radius": int(ctx.cfg.local_kernel_radius),
                "local_distance": float(abs(int(win.row) - int(loser.row)) + abs(int(win.col) - int(loser.col))),
            }
            event_rows.append(row)
            audit_rows.append(_event_audit_row(ctx, trial_id, event_id, "winner_loser_pair", True, "", win.unit_group, loser.unit_group, float(win.overlap_drive_score), float(loser.overlap_drive_score)))
            raw_trace_payload[f"event_{event_id}_winner_delta_v"] = winner_delta_v.astype(np.float32)
            raw_trace_payload[f"event_{event_id}_loser_delta_v"] = loser_delta_v.astype(np.float32)
            raw_trace_payload[f"event_{event_id}_winner_minus_loser_delta_v"] = winner_minus_loser.astype(np.float32)
            raw_trace_payload[f"event_{event_id}_loser_inhibition"] = loser_inh.astype(np.float32)
            for t_ms, value in zip(time_axis_ms, winner_delta_v):
                trace_rows.append(_event_trace_row(ctx, trial_id, event_id, t_ms, "winner_delta_v", value))
            for t_ms, value in zip(time_axis_ms, loser_delta_v):
                trace_rows.append(_event_trace_row(ctx, trial_id, event_id, t_ms, "loser_delta_v", value))
            for t_ms, value in zip(time_axis_ms, winner_minus_loser):
                trace_rows.append(_event_trace_row(ctx, trial_id, event_id, t_ms, "winner_minus_loser_delta_v", value))
            for t_ms, value in zip(time_axis_ms, loser_inh):
                trace_rows.append(_event_trace_row(ctx, trial_id, event_id, t_ms, "loser_inhibition", value))
            event_id += 1
    ctx.n_events = int(event_id)
    extra_event_columns = [
        "selection_rank_within_trial",
        "winner_full_pre_delta_v_mean",
        "loser_full_pre_delta_v_mean",
        "winner_minus_loser_full_pre_delta_v_mean",
        "winner_late_pre_delta_v_mean",
        "loser_late_pre_delta_v_mean",
        "winner_minus_loser_late_pre_delta_v_mean",
        "complete_alignment_window",
    ]
    event_columns = list(dict.fromkeys([*PANEL_C_EVENT_COLUMNS, *extra_event_columns]))
    events = pd.DataFrame(event_rows, columns=event_columns)
    audit = pd.DataFrame(audit_rows, columns=SUPP_EVENT_AUDIT_COLUMNS)
    trials = _trial_event_summary(events)
    network = _network_event_summary(ctx, bank, events, trials, audit)
    _save_csv(ctx, events, ctx.metrics_dir / "panel_c_winner_loser_event_metrics.csv")
    _save_csv(ctx, trials, ctx.metrics_dir / "panel_c_winner_loser_trial_summary.csv")
    _save_csv(ctx, network, ctx.metrics_dir / "panel_c_winner_loser_network_summary.csv")
    _save_csv(ctx, _trial_first_trace_summary(ctx, trace_rows), ctx.metrics_dir / "panel_c_event_trace_summary.csv")
    _save_csv(ctx, audit, ctx.metrics_dir / "supp_event_selection_audit.csv")
    _save_csv(ctx, _neighborhood_radius_robustness(ctx, events), ctx.metrics_dir / "supp_neighborhood_radius_robustness.csv")
    if ctx.cfg.save_full_traces:
        np.savez_compressed(ctx.raw_dir / "panel_c_event_aligned_traces.npz", **raw_trace_payload)
        ctx.output_files["panel_c_event_aligned_traces"] = _rel(ctx.raw_dir / "panel_c_event_aligned_traces.npz", ctx.seed_dir)
    ctx.completed_modules["local_events"] = True


def _window_mask(time_axis_ms: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    values = np.asarray(time_axis_ms, dtype=float)
    return (values >= float(window[0])) & (values <= float(window[1]))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    if not selected.size or not np.isfinite(selected).all():
        return float("nan")
    return float(selected.mean())


def _event_trace_row(ctx: ExperimentContext, trial_id: int, event_id: int, time_ms: float, trace_type: str, value: float) -> dict[str, Any]:
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "trial_id": int(trial_id),
        "event_id": int(event_id),
        "time_ms": float(time_ms),
        "trace_type": str(trace_type),
        "value": float(value),
    }


def _trial_event_summary(events: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "winner_full_pre_delta_v_mean",
        "loser_full_pre_delta_v_mean",
        "winner_minus_loser_full_pre_delta_v_mean",
        "winner_late_pre_delta_v_mean",
        "loser_late_pre_delta_v_mean",
        "winner_minus_loser_late_pre_delta_v_mean",
    ]
    columns = ["network_seed", "trial_id", "n_events", *metric_columns]
    if events.empty:
        return pd.DataFrame(columns=columns)
    summary = events.groupby(["network_seed", "trial_id"], as_index=False)[metric_columns].mean()
    counts = events.groupby(["network_seed", "trial_id"], as_index=False).size().rename(columns={"size": "n_events"})
    return summary.merge(counts, on=["network_seed", "trial_id"], how="left")[columns]


def _network_event_summary(
    ctx: ExperimentContext,
    bank: LocalSupportCompetitionBank,
    events: pd.DataFrame,
    trials: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "winner_full_pre_delta_v_mean",
        "loser_full_pre_delta_v_mean",
        "winner_minus_loser_full_pre_delta_v_mean",
        "winner_late_pre_delta_v_mean",
        "loser_late_pre_delta_v_mean",
        "winner_minus_loser_late_pre_delta_v_mean",
    ]
    excluded = audit.loc[~audit["included"].astype(bool)] if not audit.empty else pd.DataFrame()
    exclusion_counts = excluded["exclusion_reason"].astype(str).value_counts() if not excluded.empty else pd.Series(dtype=int)
    row: dict[str, Any] = {
        "network_seed": int(ctx.cfg.network_seed),
        "n_trials_total": int(len(bank.trials)),
        "n_trials_eligible": int(trials["trial_id"].nunique()) if not trials.empty else 0,
        "n_events_eligible": int(len(events)),
        "n_candidates_excluded": int(len(excluded)),
        "n_candidates_rank_excluded": int(exclusion_counts.get("trial_rank_gt_3", 0)),
        "n_candidates_no_local_loser": int(exclusion_counts.get("no_local_loser", 0)),
        "n_candidates_incomplete_window": int(exclusion_counts.get("incomplete_alignment_window", 0)),
        "max_winners_per_trial": int(MAX_WINNERS_PER_TRIAL),
        "primary_window_start_ms": float(PRIMARY_PRE_WINDOW_MS[0]),
        "primary_window_end_ms": float(PRIMARY_PRE_WINDOW_MS[1]),
        "descriptive_window_start_ms": float(LATE_PRE_WINDOW_MS[0]),
        "descriptive_window_end_ms": float(LATE_PRE_WINDOW_MS[1]),
        "aggregation": "event_to_trial_to_network",
        "selection_scope": "outcome_conditioned_selected_local_events",
    }
    for column in metric_columns:
        row[column] = float(pd.to_numeric(trials[column], errors="coerce").mean()) if not trials.empty else float("nan")
    return pd.DataFrame([row])


def _trial_first_trace_summary(ctx: ExperimentContext, rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = ["network_seed", "time_ms", "trace_type", "mean_value", "sem_value", "n_events", "n_trials"]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    trial = frame.groupby(["trial_id", "time_ms", "trace_type"], as_index=False)["value"].mean()
    grouped = trial.groupby(["time_ms", "trace_type"], sort=True)["value"]
    summary = grouped.agg(mean_value="mean", sem_value="sem", n_trials="count").reset_index()
    summary["sem_value"] = summary["sem_value"].fillna(0.0)
    counts = frame.groupby(["time_ms", "trace_type"], as_index=False).size().rename(columns={"size": "n_events"})
    summary = summary.merge(counts, on=["time_ms", "trace_type"], how="left")
    summary.insert(0, "network_seed", int(ctx.cfg.network_seed))
    return summary[columns]
