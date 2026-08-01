from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from scipy import optimize

from src.experiments.common.ping_common import LAYER_KEYS
from src.experiments.paper_figures.fig3_multiitem_peak_landscape_experiment import (
    _progress,
    _save_csv,
)
from src.experiments.common.monitored_dms import (
    restore_functional_probe_state_in_place,
    snapshot_boundary_state,
)
from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import (
    _centered_cosine,
    _cosine_distance,
    _layer_input_shapes_for_batch,
    _step_network_once,
    stsp_boundary_from_bank,
)
from src.experiments.paper_figures.fig3.types import (
    ExperimentContext,
    MultiItemSequenceLandscapeBank,
)

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as _legacy

# Keep module-level names identical while Fig.3 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value


def _natural_decay_boundary(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    stage_k: int,
) -> Mapping[str, Mapping[str, torch.Tensor]]:
    shapes = _layer_input_shapes_for_batch(boundary, 1)
    restore_functional_probe_state_in_place(
        ctx.net,
        shapes,
        boundary,
        mode="full_boundary",
        device=ctx.device,
    )
    current_time = int(stage_k - 1) * int(ctx.cfg.sample_steps + ctx.cfg.delay_steps)
    zero = torch.zeros((1, 1, 28, 28), dtype=torch.float32, device=ctx.device)
    with torch.no_grad():
        for _ in range(int(ctx.cfg.sample_steps + ctx.cfg.delay_steps)):
            current_time = _step_network_once(ctx.net, zero, current_time)
    return snapshot_boundary_state(ctx.net)


def _prefix_effective_count(
    bank: MultiItemSequenceLandscapeBank,
    sequence_id: int,
    stage_k: int,
    state: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    baseline = bank.get(sequence_id, "S0", "layer1", "g").astype(np.float64, copy=False).reshape(-1)
    target = np.asarray(state, dtype=np.float64).reshape(-1) - baseline
    refs = [
        bank.singleton_refs[sequence_id][position]["layer1"]["g"].astype(np.float64, copy=False).reshape(-1)
        - baseline
        for position in range(1, int(stage_k) + 1)
    ]
    coefficients, residual = optimize.nnls(np.column_stack(refs), target)
    coefficient_sum = float(coefficients.sum())
    if coefficient_sum <= 1e-12:
        proportions = np.zeros_like(coefficients)
        n_eff = 0.0
    else:
        proportions = coefficients / coefficient_sum
        n_eff = float(1.0 / np.sum(proportions * proportions))
    relative_error = float(
        residual / max(np.linalg.norm(target), 1e-12)
    )
    return n_eff, relative_error, proportions


def compute_progressive_update_metrics(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    item_weight_rows: list[dict[str, Any]] = []
    sequence_meta = bank.sequence_meta
    if "delay_ms" in sequence_meta.columns:
        sequence_meta = sequence_meta.loc[
            sequence_meta["delay_ms"].astype(int).eq(int(ctx.cfg.delay_ms))
        ]
    selected = sequence_meta.loc[
        sequence_meta["seq_len"].astype(int).eq(
            int(ctx.cfg.main_sequence_length)
        )
    ].sort_values("sequence_id")
    selected = selected.head(
        max(1, int(ctx.cfg.progressive_max_sequences))
    )
    selected_ids = set(selected["sequence_id"].astype(int).tolist())

    for _, meta in _progress(
        sequence_meta.iterrows(),
        total=len(sequence_meta),
        desc="fig3 progressive sequences",
        enabled=ctx.cfg.show_progress,
    ):
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        source_sequence_id = int(meta.get("source_sequence_id", seq_id))
        condition_id = str(meta.get("condition_id", ""))
        delay_ms = int(meta.get("delay_ms", ctx.cfg.delay_ms))
        decay_by_stage: dict[int, Mapping[str, Mapping[str, torch.Tensor]]] = {}
        if bool(ctx.cfg.progressive_natural_decay) and seq_id in selected_ids:
            for stage_k in range(2, seq_len + 1):
                previous_boundary = stsp_boundary_from_bank(
                    bank,
                    seq_id,
                    f"S_{stage_k - 1}",
                )
                decay_by_stage[stage_k] = _natural_decay_boundary(
                    ctx,
                    previous_boundary,
                    stage_k=stage_k,
                )

        for layer in LAYER_KEYS:
            for variable in ("g", "u", "x"):
                prev = bank.get(seq_id, "S0", layer, variable)
                prev_com = 0.0
                for stage_k in range(1, seq_len + 1):
                    state = bank.get(seq_id, f"S_{stage_k}", layer, variable)
                    ref = bank.singleton_refs[seq_id][stage_k][layer][variable]
                    state_disp = _cosine_distance(state, prev)
                    ref_disp = _cosine_distance(ref, prev)
                    if stage_k in decay_by_stage:
                        decay_layer = decay_by_stage[stage_k][layer]
                        decay_tensor = (
                            decay_layer["u"] * decay_layer["x"]
                            if variable == "g"
                            else decay_layer[variable]
                        )
                        decay_state = (
                            decay_tensor.detach()
                            .cpu()
                            .to(torch.float32)
                            .numpy()
                        )
                        decay_disp = _cosine_distance(decay_state, prev)
                    else:
                        decay_disp = float("nan")
                    sims = [
                        max(
                            0.0,
                            _centered_cosine(
                                state,
                                bank.singleton_refs[seq_id][pos][layer][variable],
                            ),
                        )
                        for pos in range(1, stage_k + 1)
                    ]
                    weights = np.asarray(sims, dtype=float)
                    weights = weights / max(float(weights.sum()), 1e-12)
                    positions = np.arange(1, stage_k + 1, dtype=float)
                    anchor_com = float(np.sum(positions * weights))
                    entropy = float(
                        -np.sum(weights * np.log(np.maximum(weights, 1e-12)))
                        / max(np.log(max(stage_k, 2)), 1e-12)
                    )
                    rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "sequence_id": seq_id,
                            "source_sequence_id": source_sequence_id,
                            "condition_id": condition_id,
                            "delay_ms": delay_ms,
                            "seq_len": seq_len,
                            "stage_k": stage_k,
                            "layer": layer,
                            "state_variable": variable,
                            "state_displacement": state_disp,
                            "singleton_displacement": ref_disp,
                            "natural_decay_displacement": decay_disp,
                            "observed_minus_natural_decay": state_disp - decay_disp,
                            "stepwise_update_ratio": float(state_disp / max(ref_disp, 1e-12)),
                            "anchor_COM": anchor_com,
                            "anchor_shift": float(anchor_com - prev_com),
                            "similarity_entropy": entropy,
                        }
                    )
                    if layer == "layer1" and variable == "g" and seq_id in selected_ids:
                        n_eff, nnls_error, item_weights = (
                            _prefix_effective_count(
                                bank,
                                seq_id,
                                stage_k,
                                state,
                            )
                        )
                        recency_bias = 0.0
                        if stage_k > 1:
                            midpoint = 0.5 * (stage_k + 1)
                            recency_bias = float((anchor_com - midpoint) / (0.5 * (stage_k - 1)))
                        prefix_rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "sequence_id": seq_id,
                                "source_sequence_id": source_sequence_id,
                                "condition_id": condition_id,
                                "delay_ms": delay_ms,
                                "seq_len": seq_len,
                                "stage_k": stage_k,
                                "prefix_state": f"S_{stage_k}",
                                "state_displacement": state_disp,
                                "singleton_displacement": ref_disp,
                                "natural_decay_displacement": decay_disp,
                                "observed_minus_natural_decay": state_disp - decay_disp,
                                "N_eff": n_eff,
                                "nnls_relative_error": nnls_error,
                                "anchor_COM": anchor_com,
                                "anchor_shift": float(anchor_com - prev_com),
                                "recency_bias": recency_bias,
                                "similarity_entropy": entropy,
                            }
                        )
                        for item_position, item_weight in enumerate(
                            item_weights,
                            start=1,
                        ):
                            item_weight_rows.append(
                                {
                                    "network_seed": int(ctx.cfg.network_seed),
                                    "sequence_id": seq_id,
                                    "source_sequence_id": source_sequence_id,
                                    "condition_id": condition_id,
                                    "delay_ms": delay_ms,
                                    "seq_len": seq_len,
                                    "stage_k": stage_k,
                                    "item_position": item_position,
                                    "item_weight": float(item_weight),
                                    "is_latest": int(
                                        item_position == stage_k
                                    ),
                                }
                            )
                    prev = state
                    prev_com = anchor_com

    progressive = pd.DataFrame(rows)
    prefix = pd.DataFrame(prefix_rows)
    item_weights = pd.DataFrame(item_weight_rows)
    summary = (
        prefix.groupby(["network_seed", "stage_k"], observed=True, as_index=False)
        .agg(
            mean_state_displacement=("state_displacement", "mean"),
            mean_natural_decay_displacement=("natural_decay_displacement", "mean"),
            mean_observed_minus_natural_decay=("observed_minus_natural_decay", "mean"),
            mean_N_eff=("N_eff", "mean"),
            mean_anchor_COM=("anchor_COM", "mean"),
            mean_recency_bias=("recency_bias", "mean"),
            mean_similarity_entropy=("similarity_entropy", "mean"),
            n_sequences=("sequence_id", "nunique"),
        )
    )
    _save_csv(ctx, progressive, ctx.metrics_dir / "panel_b_progressive_update_metrics.csv")
    _save_csv(ctx, prefix, ctx.metrics_dir / "panel_b_prefix_trajectory_metrics.csv")
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_b_prefix_trajectory_summary.csv")
    ctx.completed_modules["progressive_update"] = True
    _save_csv(
        ctx,
        item_weights,
        ctx.metrics_dir / "panel_b_prefix_item_weights.csv",
    )
    return {
        "progressive_update_metrics": progressive,
        "prefix_trajectory_metrics": prefix,
        "prefix_trajectory_summary": summary,
        "prefix_item_weights": item_weights,
    }
