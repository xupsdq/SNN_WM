from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import optimize

from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
from src.experiments.common.input_masks import entry_mask_from_image
from src.experiments.common.monitored_dms import restore_functional_probe_state_in_place, snapshot_boundary_state
from src.experiments.paper_figures.fig3.cache_keys import dataframe_hash
from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import (
    _encode_cached,
    _centered_cosine,
    _layer_input_shapes_for_batch,
    _run_ping_from_boundary,
    _step_network_once,
    run_probe_readout_from_boundary,
    stsp_boundary_from_bank,
)
from src.experiments.paper_figures.fig3.types import ExperimentContext, MultiItemSequenceLandscapeBank

FORMATION_CONDITIONS = (
    "dynamic_intact",
    "sham_boundary_roundtrip",
    "attenuate_overlap",
    "reset_overlap",
    "reset_nonoverlap_matched",
    "reset_random_matched",
)


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def _image_tensor(ctx: ExperimentContext, image_id: int) -> torch.Tensor:
    return ctx.dataset[int(image_id)][0].detach().to(torch.float32)


def _sequence_rows(bank: MultiItemSequenceLandscapeBank, sequence_id: int) -> pd.DataFrame:
    rows = bank.sequence_trials.loc[
        bank.sequence_trials["sequence_id"].astype(int).eq(int(sequence_id))
    ].copy()
    if rows.empty and "source_sequence_id" in bank.sequence_meta.columns:
        meta = bank.sequence_meta.loc[
            bank.sequence_meta["sequence_id"].astype(int).eq(int(sequence_id))
        ]
        if len(meta) == 1:
            source_sequence_id = int(meta.iloc[0]["source_sequence_id"])
            rows = bank.sequence_trials.loc[
                bank.sequence_trials["sequence_id"].astype(int).eq(source_sequence_id)
            ].copy()
    aliases = {
        "stage_k": "serial_position",
        "item_image_id": "image_id",
        "item_label": "label",
    }
    rows = rows.rename(
        columns={
            source: target
            for source, target in aliases.items()
            if target not in rows.columns and source in rows.columns
        }
    )
    required = {"serial_position", "image_id", "label"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Formation sequence rows missing required columns: {missing}")
    return rows.sort_values("serial_position").reset_index(drop=True)


def _layer1_map(boundary: Mapping[str, Mapping[str, torch.Tensor]]) -> np.ndarray:
    state = boundary["layer1"]
    product = state["u"] * state["x"]
    arr = product.detach().cpu().to(torch.float32).numpy()
    if arr.shape[0] != 1:
        raise ValueError(
            f"Formation intervention expects a singleton boundary, got layer1 STSP shape={arr.shape}"
        )
    return arr.reshape(1, 2, 28, 28).mean(axis=1)[0]


def _bank_layer1_map(
    bank: MultiItemSequenceLandscapeBank,
    sequence_id: int,
    state_condition: str,
) -> np.ndarray:
    arr = bank.get(
        int(sequence_id),
        str(state_condition),
        "layer1",
        "g",
    ).astype(np.float32, copy=False)
    return arr.reshape(2, 28, 28).mean(axis=0)


def _entry_mask(ctx: ExperimentContext, image_id: int, cache: dict[tuple[Any, ...], np.ndarray]) -> np.ndarray:
    return entry_mask_from_image(
        _image_tensor(ctx, image_id),
        mode=str(ctx.cfg.formation_mask_mode),
        encoder=ctx.encoder,
        steps=int(ctx.cfg.sample_steps),
        device=ctx.device,
        foreground_threshold=float(ctx.cfg.foreground_threshold),
        cache=cache,
        image_id=int(image_id),
    )


def _incoming_energy_map(ctx: ExperimentContext, image_id: int, cache: dict[tuple[Any, ...], torch.Tensor]) -> np.ndarray:
    spikes = _encode_cached(ctx, [int(image_id)], int(ctx.cfg.sample_steps), cache=cache)
    arr = spikes.detach().cpu().to(torch.float32).numpy()
    if arr.shape[-2:] != (28, 28):
        raise ValueError(f"Expected 28x28 encoded input, got shape={arr.shape}")
    return arr.sum(axis=tuple(range(arr.ndim - 2)))


def _select_nearest(
    pool: np.ndarray,
    support: np.ndarray,
    incoming: np.ndarray,
    *,
    n_sites: int,
    target_support: float,
    target_incoming: float,
    match_incoming: bool,
    seed: int,
) -> np.ndarray:
    flat_pool = np.flatnonzero(np.asarray(pool, dtype=bool).reshape(-1))
    if int(n_sites) <= 0 or flat_pool.size < int(n_sites):
        return np.zeros_like(pool, dtype=bool)
    support_flat = np.asarray(support, dtype=float).reshape(-1)
    incoming_flat = np.asarray(incoming, dtype=float).reshape(-1)
    support_scale = max(float(np.nanstd(support_flat[flat_pool])), 1e-8)
    incoming_scale = max(float(np.nanstd(incoming_flat[flat_pool])), 1e-8)
    score = ((support_flat[flat_pool] - float(target_support)) / support_scale) ** 2
    if match_incoming:
        score = score + ((incoming_flat[flat_pool] - float(target_incoming)) / incoming_scale) ** 2
    rng = np.random.default_rng(int(seed))
    score = score + rng.uniform(0.0, 1e-9, size=score.shape)
    selected = flat_pool[np.argsort(score, kind="stable")[: int(n_sites)]]
    mask = np.zeros(pool.size, dtype=bool)
    mask[selected] = True
    return mask.reshape(pool.shape)


def _mask_stats(mask: np.ndarray, support: np.ndarray, incoming: np.ndarray) -> tuple[int, float, float]:
    selected = np.asarray(mask, dtype=bool)
    return (
        int(selected.sum()),
        float(np.asarray(support, dtype=float)[selected].sum()),
        float(np.asarray(incoming, dtype=float)[selected].sum()),
    )


def build_formation_intervention_specs(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
) -> pd.DataFrame:
    sequence_mask = bank.sequence_meta["seq_len"].astype(int).eq(
        int(ctx.cfg.formation_sequence_length)
    )
    if "delay_ms" in bank.sequence_meta.columns:
        sequence_mask &= bank.sequence_meta["delay_ms"].astype(int).eq(
            int(ctx.cfg.delay_ms)
        )
    sequence_meta = (
        bank.sequence_meta.loc[sequence_mask]
        .sort_values("sequence_id")
        .head(max(1, int(ctx.cfg.formation_max_sequences)))
    )
    entry_cache: dict[tuple[Any, ...], np.ndarray] = {}
    encoded_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    rows: list[dict[str, Any]] = []

    for meta in sequence_meta.itertuples(index=False):
        sequence_id = int(meta.sequence_id)
        seq_rows = _sequence_rows(bank, sequence_id)
        seq_len = int(len(seq_rows))
        terminal_stage = min(int(ctx.cfg.formation_terminal_stage), seq_len)
        if terminal_stage < 2:
            continue
        image_ids = seq_rows["image_id"].astype(int).tolist()
        masks = [_entry_mask(ctx, image_id, entry_cache) for image_id in image_ids]
        baseline = _bank_layer1_map(bank, sequence_id, "S0")

        for stage_k in range(2, terminal_stage + 1):
            previous = _bank_layer1_map(bank, sequence_id, f"S_{stage_k - 1}")
            support = np.maximum(previous - baseline, 0.0)
            history_mask = np.logical_or.reduce(masks[: stage_k - 1])
            incoming_mask = np.asarray(masks[stage_k - 1], dtype=bool)
            incoming_energy = _incoming_energy_map(ctx, image_ids[stage_k - 1], encoded_cache)
            overlap_pool = history_mask & incoming_mask & (support > 0.0)
            if not overlap_pool.any():
                overlap_pool = history_mask & incoming_mask
            nonoverlap_pool = history_mask & ~incoming_mask
            random_pool = ~overlap_pool
            n_sites = min(
                int(overlap_pool.sum()),
                int(nonoverlap_pool.sum()),
                int(random_pool.sum()),
            )
            if n_sites <= 0:
                continue

            target_candidates = np.flatnonzero(overlap_pool.reshape(-1))
            target_order = np.argsort(support.reshape(-1)[target_candidates], kind="stable")[::-1]
            target_indices = target_candidates[target_order[:n_sites]]
            target_mask = np.zeros(overlap_pool.size, dtype=bool)
            target_mask[target_indices] = True
            target_mask = target_mask.reshape(overlap_pool.shape)
            _, target_support_mass, target_input_energy = _mask_stats(target_mask, support, incoming_energy)
            target_support_mean = target_support_mass / float(n_sites)
            target_input_mean = target_input_energy / float(n_sites)

            nonoverlap_mask = _select_nearest(
                nonoverlap_pool,
                support,
                incoming_energy,
                n_sites=n_sites,
                target_support=target_support_mean,
                target_incoming=target_input_mean,
                match_incoming=False,
                seed=_stable_seed(ctx.cfg.network_seed, sequence_id, stage_k, "nonoverlap"),
            )
            random_mask = _select_nearest(
                random_pool,
                support,
                incoming_energy,
                n_sites=n_sites,
                target_support=target_support_mean,
                target_incoming=target_input_mean,
                match_incoming=True,
                seed=_stable_seed(ctx.cfg.network_seed, sequence_id, stage_k, "random"),
            )

            for condition in FORMATION_CONDITIONS:
                if condition == "reset_nonoverlap_matched":
                    selected = nonoverlap_mask
                    intervention_class = "reset_nonoverlap"
                elif condition == "reset_random_matched":
                    selected = random_mask
                    intervention_class = "reset_random"
                elif condition == "attenuate_overlap":
                    selected = target_mask
                    intervention_class = "attenuate_overlap"
                elif condition == "reset_overlap":
                    selected = target_mask
                    intervention_class = "reset_overlap"
                elif condition == "sham_boundary_roundtrip":
                    selected = target_mask
                    intervention_class = "sham"
                else:
                    selected = target_mask
                    intervention_class = "none"
                selected_n, selected_support_mass, selected_input_energy = _mask_stats(selected, support, incoming_energy)
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": sequence_id,
                        "seq_len": seq_len,
                        "delay_ms": int(getattr(meta, "delay_ms", ctx.cfg.delay_ms)),
                        "stage_k": stage_k,
                        "condition": condition,
                        "intervention_class": intervention_class,
                        "selected_flat_indices": ";".join(str(v) for v in np.flatnonzero(selected.reshape(-1))),
                        "selected_site_count": selected_n,
                        "target_site_count": int(n_sites),
                        "target_support_mass": target_support_mass,
                        "selected_support_mass": selected_support_mass,
                        "target_incoming_energy": target_input_energy,
                        "selected_incoming_energy": selected_input_energy,
                        "support_match_error": abs(selected_support_mass - target_support_mass) / max(abs(target_support_mass), 1e-8),
                        "incoming_match_error": abs(selected_input_energy - target_input_energy) / max(abs(target_input_energy), 1e-8),
                        "history_area": int(history_mask.sum()),
                        "incoming_area": int(incoming_mask.sum()),
                        "overlap_area": int((history_mask & incoming_mask).sum()),
                        "attenuation_factor": float(ctx.cfg.formation_attenuation),
                        "intervention_seed": _stable_seed(ctx.cfg.network_seed, sequence_id, stage_k, condition),
                        "selection_valid": bool(selected_n == n_sites and n_sites > 0),
                    }
                )

    columns = [
        "network_seed",
        "sequence_id",
        "seq_len",
        "delay_ms",
        "stage_k",
        "condition",
        "intervention_class",
        "selected_flat_indices",
        "selected_site_count",
        "target_site_count",
        "target_support_mass",
        "selected_support_mass",
        "target_incoming_energy",
        "selected_incoming_energy",
        "support_match_error",
        "incoming_match_error",
        "history_area",
        "incoming_area",
        "overlap_area",
        "attenuation_factor",
        "intervention_seed",
        "selection_valid",
    ]
    specs = pd.DataFrame(rows, columns=columns)
    if specs.empty:
        raise RuntimeError("Formation intervention specification produced no valid rows.")
    if not specs["selection_valid"].astype(bool).all():
        raise RuntimeError("Formation intervention specification contains invalid matched selections.")
    return specs


def _clone_boundary(boundary: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        layer: {key: value.detach().clone() for key, value in state.items()}
        for layer, state in boundary.items()
    }




def _mask_from_row(row: pd.Series) -> np.ndarray:
    mask = np.zeros(28 * 28, dtype=bool)
    text = str(row["selected_flat_indices"]).strip()
    if text:
        indices = np.fromstring(text, sep=";", dtype=np.int64)
        mask[indices] = True
    return mask.reshape(28, 28)


def _repeat_boundary(
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    repeat_count: int,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        layer: {
            key: torch.cat(
                [
                    value.detach().clone()
                    for _ in range(int(repeat_count))
                ],
                dim=0,
            )
            for key, value in state.items()
        }
        for layer, state in boundary.items()
    }


def _apply_formation_intervention(
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    baseline: Mapping[str, Mapping[str, torch.Tensor]],
    row: pd.Series,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    out = _clone_boundary(boundary)
    condition = str(row["condition"])
    mask = _mask_from_row(row)
    channel_mask = np.broadcast_to(mask[None, ...], (2, 28, 28)).reshape(-1)
    index = torch.as_tensor(np.flatnonzero(channel_mask), dtype=torch.long)
    before = {key: out["layer1"][key].detach().clone() for key in ("u", "x")}

    if condition == "attenuate_overlap":
        factor = float(row["attenuation_factor"])
        for key in ("u", "x"):
            current = out["layer1"][key]
            current_flat = current.reshape(current.shape[0], -1)
            base_flat = baseline["layer1"][key].to(
                current.device,
                dtype=current.dtype,
            ).reshape(current.shape[0], -1)
            current_flat[:, index] = base_flat[:, index] + factor * (
                current_flat[:, index] - base_flat[:, index]
            )
    elif condition in {"reset_overlap", "reset_nonoverlap_matched", "reset_random_matched"}:
        for key in ("u", "x"):
            current = out["layer1"][key]
            current_flat = current.reshape(current.shape[0], -1)
            base_flat = baseline["layer1"][key].to(
                current.device,
                dtype=current.dtype,
            ).reshape(current.shape[0], -1)
            current_flat[:, index] = base_flat[:, index]
    elif condition not in {"dynamic_intact", "sham_boundary_roundtrip"}:
        raise ValueError(f"Unsupported formation condition: {condition}")

    inside_deltas: list[float] = []
    outside_deltas: list[float] = []
    outside_index = torch.as_tensor(np.flatnonzero(~channel_mask), dtype=torch.long)
    for key in ("u", "x"):
        after = out["layer1"][key]
        delta = (after - before[key]).abs().reshape(after.shape[0], -1)
        inside_deltas.append(
            float(delta[:, index].max().item()) if index.numel() else 0.0
        )
        outside_deltas.append(
            float(delta[:, outside_index].max().item())
            if outside_index.numel()
            else 0.0
        )
    return out, {
        "intervention_max_abs_inside": max(inside_deltas, default=0.0),
        "intervention_max_abs_outside": max(outside_deltas, default=0.0),
        "pre_intervention_hash": dataframe_hash(
            pd.DataFrame({"value": (before["u"] * before["x"]).reshape(-1).cpu().numpy()})
        ),
        "post_intervention_hash": dataframe_hash(
            pd.DataFrame(
                {
                    "value": (
                        out["layer1"]["u"] * out["layer1"]["x"]
                    ).reshape(-1).cpu().numpy()
                }
            )
        ),
    }


def _full_restore(ctx: ExperimentContext, boundary: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    shapes = _layer_input_shapes_for_batch(boundary, 1)
    restore_functional_probe_state_in_place(
        ctx.net,
        shapes,
        boundary,
        mode="full_boundary",
        device=ctx.device,
    )


def _run_item_transition(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    sample_spikes: torch.Tensor,
    *,
    stage_k: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], int, int]:
    _full_restore(ctx, boundary)
    ctx.net.layer3.reset_decision_state()
    current_time = int(stage_k - 1) * int(ctx.cfg.sample_steps + ctx.cfg.delay_steps)
    with torch.no_grad():
        for t_idx in range(int(sample_spikes.shape[1])):
            current_time = _step_network_once(ctx.net, sample_spikes[:, t_idx].to(ctx.device, dtype=torch.float32), current_time)
        zero = torch.zeros_like(sample_spikes[:, 0], dtype=torch.float32, device=ctx.device)
        for _ in range(int(ctx.cfg.delay_steps)):
            current_time = _step_network_once(ctx.net, zero, current_time)
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size=1)
    return snapshot_boundary_state(ctx.net), int(pred[0].item()), int(fire[0].item())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float64).reshape(-1)
    vb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return 0.0 if denom <= 0.0 else float(np.dot(va, vb) / denom)


def _boundary_vector(
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    layer: str = "layer1",
) -> np.ndarray:
    state = boundary[layer]
    product = state["u"] * state["x"]
    return (
        product.detach()
        .cpu()
        .to(torch.float32)
        .numpy()
        .reshape(-1)
        .astype(np.float64, copy=False)
    )


def _weak_cue_spikes(
    ctx: ExperimentContext,
    image_id: int,
    *,
    sequence_id: int,
    stage_k: int,
    cue_role: str,
    cue_repeat: int,
    cache: dict[tuple[Any, ...], torch.Tensor],
) -> torch.Tensor:
    spikes = _encode_cached(ctx, [int(image_id)], int(ctx.cfg.weak_probe_steps), cache=cache).clone()
    generator = torch.Generator(device=spikes.device)
    generator.manual_seed(
        _stable_seed(
            ctx.cfg.network_seed,
            sequence_id,
            stage_k,
            cue_role,
            cue_repeat,
        )
    )
    keep = torch.rand(spikes.shape, generator=generator, device=spikes.device) < float(ctx.cfg.formation_weak_probe_keep_fraction)
    return spikes * keep.to(spikes.dtype)


def _effective_item_count(
    bank: MultiItemSequenceLandscapeBank,
    sequence_id: int,
    stage_k: int,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[float, float]:
    baseline = bank.get(sequence_id, "S0", "layer1", "g").astype(np.float64, copy=False)
    target = _boundary_vector(boundary) - baseline.reshape(-1)
    refs = [
        bank.singleton_refs[int(sequence_id)][position]["layer1"]["g"].astype(np.float64, copy=False).reshape(-1)
        - baseline.reshape(-1)
        for position in range(1, int(stage_k) + 1)
    ]
    design = np.column_stack(refs)
    coefficients, residual = optimize.nnls(design, target)
    coefficient_sum = float(coefficients.sum())
    if coefficient_sum <= 1e-12:
        n_eff = 0.0
    else:
        proportions = coefficients / coefficient_sum
        n_eff = float(1.0 / np.sum(proportions * proportions))
    return n_eff, float(residual / max(np.linalg.norm(target), 1e-12))


def _pair_specificity_rows(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
    pre_boundaries: Mapping[
        tuple[int, str, int],
        Mapping[str, Mapping[str, torch.Tensor]],
    ],
    formed_boundaries: Mapping[
        tuple[int, str, int],
        Mapping[str, Mapping[str, torch.Tensor]],
    ],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sequence_ids = sorted({key[0] for key in formed_boundaries})
    for sequence_id in sequence_ids:
        baseline = (
            bank.get(sequence_id, "S0", "layer1", "g")
            .astype(np.float64, copy=False)
            .reshape(-1)
        )
        for condition in FORMATION_CONDITIONS:
            stage_keys = sorted(
                key
                for key in formed_boundaries
                if key[0] == sequence_id and key[1] == condition
            )
            for key in stage_keys:
                stage_k = int(key[2])
                x_history = _boundary_vector(pre_boundaries[key]) - baseline
                x_current = (
                    bank.singleton_refs[sequence_id][stage_k]["layer1"]["g"]
                    .astype(np.float64, copy=False)
                    .reshape(-1)
                    - baseline
                )
                y = _boundary_vector(formed_boundaries[key]) - baseline
                y_raw = _boundary_vector(formed_boundaries[key])
                history_raw = (
                    bank.get(
                        sequence_id,
                        f"S_{stage_k - 1}",
                        "layer1",
                        "g",
                    )
                    .astype(np.float64, copy=False)
                    .reshape(-1)
                )
                incoming_raw = (
                    bank.singleton_refs[sequence_id][stage_k]["layer1"]["g"]
                    .astype(np.float64, copy=False)
                    .reshape(-1)
                )
                pair_raw = 0.5 * (history_raw + incoming_raw)
                history_similarity = _centered_cosine(y_raw, history_raw)
                incoming_similarity = _centered_cosine(y_raw, incoming_raw)
                pair_similarity = _centered_cosine(y_raw, pair_raw)
                design = np.column_stack(
                    [x_history, x_current, np.ones_like(x_history)]
                )
                coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
                true_residual = float(
                    np.linalg.norm(y - design @ coefficients)
                    / max(np.linalg.norm(y), 1e-12)
                )
                candidates = [
                    other for other in sequence_ids if other != sequence_id
                ]
                rng = np.random.default_rng(
                    _stable_seed(
                        ctx.cfg.network_seed,
                        sequence_id,
                        condition,
                        stage_k,
                        "shuffle",
                    )
                )
                if len(candidates) > int(ctx.cfg.formation_n_shuffle):
                    candidates = [
                        int(value)
                        for value in rng.choice(
                            candidates,
                            size=int(ctx.cfg.formation_n_shuffle),
                            replace=False,
                        )
                    ]
                shuffled: list[float] = []
                for other in candidates:
                    other_baseline = (
                        bank.get(other, "S0", "layer1", "g")
                        .astype(np.float64, copy=False)
                        .reshape(-1)
                    )
                    other_current = (
                        bank.singleton_refs[other][stage_k]["layer1"]["g"]
                        .astype(np.float64, copy=False)
                        .reshape(-1)
                        - other_baseline
                    )
                    null_design = np.column_stack(
                        [x_history, other_current, np.ones_like(x_history)]
                    )
                    null_coefficients, *_ = np.linalg.lstsq(
                        null_design,
                        y,
                        rcond=None,
                    )
                    shuffled.append(
                        float(
                            np.linalg.norm(y - null_design @ null_coefficients)
                            / max(np.linalg.norm(y), 1e-12)
                        )
                    )
                shuffled_mean = (
                    float(np.mean(shuffled)) if shuffled else float("nan")
                )
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": sequence_id,
                        "stage_k": stage_k,
                        "condition": condition,
                        "history_similarity": history_similarity,
                        "incoming_similarity": incoming_similarity,
                        "dual_constituent_similarity": min(
                            history_similarity,
                            incoming_similarity,
                        ),
                        "pair_composite_similarity": pair_similarity,
                        "WPRI": (
                            pair_similarity
                            - max(history_similarity, incoming_similarity)
                        ),
                        "pair_residual_specificity": (
                            shuffled_mean - true_residual
                        ),
                        "residual_true_pair": true_residual,
                        "residual_shuffled_pair_mean": shuffled_mean,
                        "n_shuffled_pairs": int(len(shuffled)),
                    }
                )
    return pd.DataFrame(rows)


def _condition_summary(
    ctx: ExperimentContext,
    stage: pd.DataFrame,
    access: pd.DataFrame,
    pair: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    terminal_stage = int(ctx.cfg.formation_terminal_stage)
    for condition in FORMATION_CONDITIONS:
        stage_cond = stage.loc[stage["condition"].eq(condition)]
        access_cond = access.loc[access["condition"].eq(condition)]
        pair_cond = pair.loc[pair["condition"].eq(condition)]
        old_access = access_cond.loc[(access_cond["stage_k"].eq(2)) & access_cond["cue_role"].eq("old_item")]
        new_access = access_cond.loc[
            (access_cond["stage_k"].eq(2))
            & access_cond["cue_role"].eq("new_item")
        ]
        stage2_item_access = (
            access_cond.loc[access_cond["stage_k"].eq(2)]
            .groupby(
                ["sequence_id", "cue_role"],
                observed=True,
            )["cue_pred_is_target"]
            .mean()
            .unstack("cue_role")
        )
        joint_item_access = stage2_item_access[
            ["old_item", "new_item"]
        ].min(axis=1)
        stage2_access = access_cond.loc[
            access_cond["stage_k"].eq(2)
        ].drop_duplicates(["sequence_id", "condition", "stage_k"])
        terminal = access_cond.loc[access_cond["stage_k"].eq(terminal_stage)]
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "condition": condition,
                "mean_pre_b_retention": float(stage_cond.loc[stage_cond["stage_k"].eq(2), "pre_b_retention"].mean()),
                "mean_b_accuracy": float(stage_cond.loc[stage_cond["stage_k"].eq(2), "b_pred_is_target"].mean()),
                "mean_pair_residual_specificity": float(pair_cond["pair_residual_specificity"].mean()),
                "mean_stage2_pair_residual_specificity": float(
                    pair_cond.loc[
                        pair_cond["stage_k"].eq(2),
                        "pair_residual_specificity",
                    ].mean()
                ),
                "mean_terminal_pair_residual_specificity": float(
                    pair_cond.loc[
                        pair_cond["stage_k"].eq(terminal_stage),
                        "pair_residual_specificity",
                    ].mean()
                ),
                "mean_stage2_dual_constituent_similarity": float(
                    pair_cond.loc[
                        pair_cond["stage_k"].eq(2),
                        "dual_constituent_similarity",
                    ].mean()
                ),
                "mean_stage2_WPRI": float(
                    pair_cond.loc[
                        pair_cond["stage_k"].eq(2),
                        "WPRI",
                    ].mean()
                ),
                "mean_stage2_N_eff": float(stage2_access["N_eff"].mean()),
                "mean_stage2_nnls_relative_error": float(
                    stage2_access["nnls_relative_error"].mean()
                ),
                "mean_old_item_cue_accuracy": float(old_access["cue_pred_is_target"].mean()),
                "mean_new_item_cue_accuracy": float(
                    new_access["cue_pred_is_target"].mean()
                ),
                "mean_joint_item_cue_accuracy": float(
                    joint_item_access.mean()
                ),
                "mean_terminal_N_eff": float(terminal["N_eff"].mean()),
                "mean_terminal_old_item_cue_accuracy": float(terminal.loc[terminal["cue_role"].eq("old_item"), "cue_pred_is_target"].mean()),
                "mean_state_fidelity_cosine": float(stage_cond["state_fidelity_cosine"].mean()),
                "n_sequences": int(stage_cond["sequence_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def run_formation_necessity(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
    specs: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    encoded_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    stage_rows: list[dict[str, Any]] = []
    access_rows: list[dict[str, Any]] = []
    formed_boundaries: dict[tuple[int, str, int], dict[str, dict[str, torch.Tensor]]] = {}
    pre_boundaries: dict[
        tuple[int, str, int],
        dict[str, dict[str, torch.Tensor]],
    ] = {}

    for (sequence_id, condition), group in specs.groupby(["sequence_id", "condition"], sort=True, observed=True):
        sequence_id = int(sequence_id)
        condition = str(condition)
        seq_rows = _sequence_rows(bank, sequence_id)
        target_by_position = {
            int(row.serial_position): (int(row.image_id), int(row.label))
            for row in seq_rows.itertuples(index=False)
        }
        prior_labels: list[int] = [target_by_position[1][1]]
        current_boundary = stsp_boundary_from_bank(bank, sequence_id, "S_1")
        baseline = stsp_boundary_from_bank(bank, sequence_id, "S0")

        for _, spec_row in group.sort_values("stage_k").iterrows():
            stage_k = int(spec_row["stage_k"])
            image_id, target_label = target_by_position[stage_k]
            intervened, audit = _apply_formation_intervention(current_boundary, baseline, spec_row)
            pre_boundaries[(sequence_id, condition, stage_k)] = _clone_boundary(
                intervened
            )
            pre_pred, pre_fire, _, _, _ = _run_ping_from_boundary(ctx, intervened)
            sample_spikes = _encode_cached(ctx, [image_id], int(ctx.cfg.sample_steps), cache=encoded_cache)
            formed, b_pred, b_fire = _run_item_transition(ctx, intervened, sample_spikes, stage_k=stage_k)
            canonical = stsp_boundary_from_bank(bank, sequence_id, f"S_{stage_k}")
            fidelity = _cosine(_boundary_vector(formed), _boundary_vector(canonical))
            formed_boundaries[(sequence_id, condition, stage_k)] = formed
            stage_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": sequence_id,
                    "seq_len": int(len(seq_rows)),
                    "stage_k": stage_k,
                    "condition": condition,
                    "intervention_class": str(spec_row["intervention_class"]),
                    "pre_b_prediction": pre_pred,
                    "pre_b_fire_time": pre_fire,
                    "pre_b_retention": int(pre_pred in prior_labels),
                    "b_target_label": target_label,
                    "b_prediction": b_pred,
                    "b_pred_is_target": int(b_pred == target_label),
                    "b_fire_time": b_fire,
                    "state_fidelity_cosine": fidelity,
                    "selected_site_count": int(spec_row["selected_site_count"]),
                    "support_match_error": float(spec_row["support_match_error"]),
                    "incoming_match_error": float(spec_row["incoming_match_error"]),
                    **audit,
                }
            )

            if stage_k in {2, int(ctx.cfg.formation_terminal_stage)}:
                n_eff, nnls_error = _effective_item_count(
                    bank,
                    sequence_id,
                    stage_k,
                    formed,
                )
                repeated_boundary = _repeat_boundary(
                    formed,
                    int(ctx.cfg.formation_weak_probe_repeats),
                )
                cue_targets = {
                    "old_item": target_by_position[1],
                    "new_item": target_by_position[stage_k],
                }
                for cue_role, (cue_image_id, cue_label) in cue_targets.items():
                    cue_spikes = torch.cat(
                        [
                            _weak_cue_spikes(
                                ctx,
                                cue_image_id,
                                sequence_id=sequence_id,
                                stage_k=stage_k,
                                cue_role=cue_role,
                                cue_repeat=cue_repeat,
                                cache=encoded_cache,
                            )
                            for cue_repeat in range(
                                int(ctx.cfg.formation_weak_probe_repeats)
                            )
                        ],
                        dim=0,
                    )
                    cue_pred, cue_fire = run_probe_readout_from_boundary(
                        ctx,
                        repeated_boundary,
                        cue_spikes,
                        probe_scale=float(ctx.cfg.weak_probe_scale),
                        probe_noise=float(ctx.cfg.weak_probe_noise),
                        seed=_stable_seed(
                            ctx.cfg.network_seed,
                            sequence_id,
                            stage_k,
                            cue_role,
                            "readout",
                        ),
                    )
                    for cue_repeat, (prediction, fire_time) in enumerate(
                        zip(cue_pred, cue_fire)
                    ):
                        access_rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "sequence_id": sequence_id,
                                "stage_k": stage_k,
                                "condition": condition,
                                "cue_role": cue_role,
                                "cue_repeat": cue_repeat,
                                "cue_target_label": cue_label,
                                "cue_prediction": int(prediction),
                                "cue_pred_is_target": int(
                                    prediction == cue_label
                                ),
                                "cue_fire_time": int(fire_time),
                                "N_eff": n_eff,
                                "nnls_relative_error": nnls_error,
                            }
                        )
            prior_labels.append(target_label)
            current_boundary = formed

    stage = pd.DataFrame(stage_rows)
    access = pd.DataFrame(access_rows)
    pair = _pair_specificity_rows(
        ctx,
        bank,
        pre_boundaries,
        formed_boundaries,
    )
    summary = _condition_summary(ctx, stage, access, pair)
    if stage.empty or access.empty or pair.empty or summary.empty:
        raise RuntimeError("Formation necessity task produced an empty required output table.")
    return {
        "formation_stage_readout": stage,
        "formation_access_readout": access,
        "formation_pair_specificity": pair,
        "formation_condition_summary": summary,
    }


__all__ = [
    "FORMATION_CONDITIONS",
    "build_formation_intervention_specs",
    "run_formation_necessity",
]
