from __future__ import annotations

"""State-bank producer for the fixed-set, fixed-latest temporal-order pilot.

Reuses the project's shared simulation infrastructure (SDNN_Network,
DoGSpikeEncoder, model loader, MNIST skeleton dataset, snapshot helpers) with
the exact Fig.6 Layer-2 structural-analysis timing: sample_ms=200, delay_ms=200,
dt=1 ms; S0 captured after a zero-input pre-run of K x (sample+delay); terminal
state captured after the last item's delay.

Captured per order trial (Layer 2): S0 u/x, terminal u/x, and the
baseline-subtracted u/x concatenation. Per singleton reference (item x slot):
terminal u/x and baseline-subtracted concatenation.
"""


import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from src.config.units import ms as ms_unit
from src.core.network import SDNN_Network
from src.data.encoding import DoGSpikeEncoder
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.ping_common import LAYER_KEYS, prepare_network_state, snapshot_ux_state
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.paper_figures.fig6b_order_specificity.types import (
    SEQUENCE_LENGTH,
    OrderSpecificityConfig,
    SimulationContext,
)

LAYER = "layer2"
STATE_VARIABLES = ("u", "x")


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms_unit) / float(dt))))


def build_dataset_index(cfg: OrderSpecificityConfig) -> tuple[object, dict[int, list[int]]]:
    """Dataset + class index only; no model loading (used by the specs task)."""
    dataset = load_mnist_skeleton_dataset(cfg.dataset_root, cfg.split)
    return dataset, build_class_index(dataset, 10)


def build_simulation_context(cfg: OrderSpecificityConfig) -> SimulationContext:
    seed_everything(int(cfg.network_seed or 0))
    device = resolve_device(cfg.device)
    dataset = load_mnist_skeleton_dataset(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, 10)
    model_path = Path(cfg.model_path) if cfg.model_path else None
    warnings: list[str] = []
    if model_path is not None and model_path.exists():
        net, encoder = load_model_and_encoder(
            model_path,
            device=device,
            dt=cfg.dt,
            max_duration_ms=max(cfg.sample_ms, 100),
        )
    elif cfg.smoke:
        seed_everything(int(cfg.network_seed or 0))
        net = SDNN_Network(device=str(device)).to(device)
        net.eval()
        encoder = DoGSpikeEncoder(dt=cfg.dt, max_duration=max(cfg.sample_ms, 100) * ms_unit, device=str(device))
        warnings.append(
            "Model checkpoint missing; smoke mode used an untrained repo SDNN_Network instance. "
            "Rollouts are real but are not manuscript evidence."
        )
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return SimulationContext(
        cfg=cfg,
        device=device,
        net=net,
        encoder=encoder,
        dataset=dataset,
        class_index=class_index,
        warnings=warnings,
    )


def _step_network_once(net, input_t: torch.Tensor, current_time: int) -> int:
    s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode="dynamic", ping_drive=None)
    s1p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode="dynamic")
    s2p = net.pool2(s2.float())
    net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode="dynamic")
    return current_time + 1


def _encode_image_ids(ctx: SimulationContext, image_ids: Iterable[int], *, cache: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    cfg = ctx.cfg
    steps = cfg.sample_steps
    out: dict[int, torch.Tensor] = {}
    to_encode = [int(v) for v in image_ids if int(v) not in cache]
    if to_encode:
        images = torch.stack([ctx.dataset[int(v)][0].detach().to(torch.float32) for v in to_encode], dim=0).to(ctx.device)
        spikes = encode_images(ctx.encoder, images, steps)
        for image_id, row in zip(to_encode, spikes):
            cache[int(image_id)] = row.detach()
    for image_id in image_ids:
        out[int(image_id)] = cache[int(image_id)]
    return out


def _order_batch_tensor(ctx: SimulationContext, encode_cache: dict[int, torch.Tensor], ordered_ids_rows: list[list[int]]) -> torch.Tensor:
    """Build a (B, K, T, C, H, W) spike tensor for a batch of order trials."""
    batches = []
    for ordered_ids in ordered_ids_rows:
        spikes = _encode_image_ids(ctx, ordered_ids, cache=encode_cache)
        item_spikes = torch.stack([spikes[int(image_id)] for image_id in ordered_ids], dim=0)  # (K, T, C, H, W)
        batches.append(item_spikes.unsqueeze(0))
    return torch.cat(batches, dim=0).contiguous()


def _spike_shape(ctx: SimulationContext, encode_cache: dict[int, torch.Tensor]) -> tuple[int, int, int]:
    """(channels, height, width) of encoded spikes for one image."""
    image_id = int(ctx.dataset[0][1] is not None and 0)
    spikes = _encode_image_ids(ctx, [image_id], cache=encode_cache)
    first = next(iter(spikes.values()))
    _, channels, height, width = first.shape
    return int(channels), int(height), int(width)


def _ref_batch_tensor(ctx: SimulationContext, encode_cache: dict[int, torch.Tensor], ref_specs: list[tuple[int, int]]) -> torch.Tensor:
    """Build a (B, K, T, C, H, W) tensor; each row presents its item at its slot only."""
    cfg = ctx.cfg
    steps = cfg.sample_steps
    channels, height, width = _spike_shape(ctx, encode_cache)
    rows = []
    for image_id, slot in ref_specs:
        row = torch.zeros(SEQUENCE_LENGTH, steps, channels, height, width, device=ctx.device)
        spikes = _encode_image_ids(ctx, [image_id], cache=encode_cache)
        row[int(slot) - 1] = spikes[int(image_id)]
        rows.append(row.unsqueeze(0))
    return torch.cat(rows, dim=0).contiguous()


def _capture_s0(ctx: SimulationContext, batch_size: int, ref_shape: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    """S0: fresh state after K x (sample + delay) steps of zero input."""
    cfg = ctx.cfg
    channels, height, width = ref_shape
    zero_input = torch.zeros((batch_size, channels, height, width), device=ctx.device)
    prepare_network_state(ctx.net, batch_size, channels, height, width)
    with torch.no_grad():
        for _ in range(cfg.sequence_length):
            for _ in range(cfg.sample_steps + cfg.delay_steps):
                _step_network_once(ctx.net, zero_input, 0)
    return snapshot_ux_state(ctx.net, batch_size)


def _capture_order_trials(ctx: SimulationContext, spikes_batch: torch.Tensor) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run a batch of order trials; return (S0 u/x, terminal u/x) per row."""
    cfg = ctx.cfg
    batch_size, seq_len, steps, channels, height, width = spikes_batch.shape
    zero_input = torch.zeros((batch_size, channels, height, width), device=ctx.device)
    prepare_network_state(ctx.net, batch_size, channels, height, width)
    with torch.no_grad():
        for _ in range(seq_len):
            for _ in range(cfg.sample_steps + cfg.delay_steps):
                _step_network_once(ctx.net, zero_input, 0)
    s0 = snapshot_ux_state(ctx.net, batch_size)
    prepare_network_state(ctx.net, batch_size, channels, height, width)
    current_time = 0
    with torch.no_grad():
        for idx in range(seq_len):
            for t in range(steps):
                current_time = _step_network_once(ctx.net, spikes_batch[:, idx, t, ...], current_time)
            for _ in range(cfg.delay_steps):
                current_time = _step_network_once(ctx.net, zero_input, current_time)
    terminal = snapshot_ux_state(ctx.net, batch_size)
    return s0, terminal


def _capture_ref_trials(ctx: SimulationContext, spikes_batch: torch.Tensor) -> dict[str, np.ndarray]:
    """Run a batch of singleton-reference trials (item at its slot only)."""
    cfg = ctx.cfg
    batch_size, seq_len, steps, channels, height, width = spikes_batch.shape
    zero_input = torch.zeros((batch_size, channels, height, width), device=ctx.device)
    prepare_network_state(ctx.net, batch_size, channels, height, width)
    current_time = 0
    with torch.no_grad():
        for idx in range(seq_len):
            for t in range(steps):
                current_time = _step_network_once(ctx.net, spikes_batch[:, idx, t, ...], current_time)
            for _ in range(cfg.delay_steps):
                current_time = _step_network_once(ctx.net, zero_input, current_time)
    return snapshot_ux_state(ctx.net, batch_size)


def _layer2_ux(snapshot: Mapping[str, Mapping[str, np.ndarray]], row: int) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(snapshot[LAYER]["u"][row], dtype=np.float64).reshape(-1),
            np.asarray(snapshot[LAYER]["x"][row], dtype=np.float64).reshape(-1),
        ]
    )


def capture_network_state_bank(
    ctx: SimulationContext,
    sequence_specs: pd.DataFrame,
    reference_specs: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Any]:
    """Simulate all order and reference trials for one network and persist the bank."""
    cfg = ctx.cfg
    out_dir.mkdir(parents=True, exist_ok=True)
    network_seed = int(cfg.network_seed or 0)
    # Specs are shared across networks; each bank simulates only its own rows.
    sequence_specs = sequence_specs[sequence_specs["network_seed"].astype(int).eq(network_seed)].copy()
    reference_specs = reference_specs[reference_specs["network_seed"].astype(int).eq(network_seed)].copy()
    if sequence_specs.empty or reference_specs.empty:
        raise RuntimeError(f"No stimulus specs for network_seed={network_seed}")
    encode_cache: dict[int, torch.Tensor] = {}

    payload: dict[str, np.ndarray] = {}
    manifest_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    def _store(key: str, arr: np.ndarray, *, set_id: int, condition_type: str, order_index: int = -1, ref_role: str = "", ref_slot: int = -1) -> None:
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        payload[key] = arr
        manifest_rows.append(
            {
                "network_seed": network_seed,
                "set_id": int(set_id),
                "condition_type": condition_type,
                "order_index": int(order_index) if condition_type == "order" else -1,
                "ref_role": ref_role if condition_type == "reference" else "",
                "ref_slot": int(ref_slot) if condition_type == "reference" else -1,
                "layer": LAYER,
                "state_variable": key.split("_")[-1],
                "shape": "x".join(str(v) for v in arr.shape),
                "storage_file": "state_bank_layer2.npz",
                "storage_key": key,
                "captured_after": "terminal_delay" if not key.endswith("_S0_u") and not key.endswith("_S0_x") else "baseline_pre_run",
                "sample_ms": int(cfg.sample_ms),
                "delay_ms": int(cfg.delay_ms),
            }
        )

    # --- S0 baseline (shared by order trials and references) -------------
    channels, height, width = _spike_shape(ctx, encode_cache)
    s0_global = _capture_s0(ctx, 1, (channels, height, width))
    s0_ux = _layer2_ux(s0_global, 0)

    # --- Order trials ------------------------------------------------------
    order_rows = sequence_specs.sort_values(["set_id", "order_index"], kind="stable").to_dict("records")
    for start in range(0, len(order_rows), max(1, int(cfg.batch_size))):
        chunk = order_rows[start : start + int(cfg.batch_size)]
        ids_rows = [[int(v) for v in str(row["ordered_item_ids"]).split(";")] for row in chunk]
        spikes_batch = _order_batch_tensor(ctx, encode_cache, ids_rows)
        s0_batch, terminal_batch = _capture_order_trials(ctx, spikes_batch)
        for local_idx, row in enumerate(chunk):
            set_id = int(row["set_id"])
            order_index = int(row["order_index"])
            terminal_ux = _layer2_ux(terminal_batch, local_idx)
            s0_row_ux = _layer2_ux(s0_batch, local_idx)
            sub_ux = terminal_ux - s0_row_ux
            prefix = f"order_{set_id:02d}_{order_index}"
            _store(f"{prefix}_S0_u", s0_batch[LAYER]["u"][local_idx], set_id=set_id, condition_type="order", order_index=order_index)
            _store(f"{prefix}_S0_x", s0_batch[LAYER]["x"][local_idx], set_id=set_id, condition_type="order", order_index=order_index)
            _store(f"{prefix}_Sfinal_u", terminal_batch[LAYER]["u"][local_idx], set_id=set_id, condition_type="order", order_index=order_index)
            _store(f"{prefix}_Sfinal_x", terminal_batch[LAYER]["x"][local_idx], set_id=set_id, condition_type="order", order_index=order_index)
            _store(f"{prefix}_sub_ux", sub_ux, set_id=set_id, condition_type="order", order_index=order_index)
            meta_rows.append(
                {
                    "network_seed": network_seed,
                    "set_id": set_id,
                    "order_index": order_index,
                    "order_name": str(row["order_name"]),
                    "ordered_item_ids": str(row["ordered_item_ids"]),
                    "ordered_item_labels": str(row["ordered_item_labels"]),
                    "latest_item_id": int(row["latest_item_id"]),
                    "latest_item_label": int(row["latest_item_label"]),
                    "seq_len": int(row["seq_len"]),
                    "sample_ms": int(cfg.sample_ms),
                    "delay_ms": int(cfg.delay_ms),
                    "sequence_seed": int(row["sequence_seed"]),
                    "storage_prefix": prefix,
                }
            )

    # --- Singleton references (item x slot, shared across candidate orders) --
    ref_rows = reference_specs.sort_values(["set_id", "temporal_slot"], kind="stable").to_dict("records")
    for start in range(0, len(ref_rows), max(1, int(cfg.batch_size))):
        chunk = ref_rows[start : start + int(cfg.batch_size)]
        ref_specs = [(int(row["item_image_id"]), int(row["temporal_slot"])) for row in chunk]
        spikes_batch = _ref_batch_tensor(ctx, encode_cache, ref_specs)
        terminal_batch = _capture_ref_trials(ctx, spikes_batch)
        for local_idx, row in enumerate(chunk):
            set_id = int(row["set_id"])
            role = str(row["item_role"])
            slot = int(row["temporal_slot"])
            ref_ux = _layer2_ux(terminal_batch, local_idx)
            sub_ux = ref_ux - s0_ux
            prefix = f"ref_{set_id:02d}_{role}_{slot}"
            _store(f"{prefix}_u", terminal_batch[LAYER]["u"][local_idx], set_id=set_id, condition_type="reference", ref_role=role, ref_slot=slot)
            _store(f"{prefix}_x", terminal_batch[LAYER]["x"][local_idx], set_id=set_id, condition_type="reference", ref_role=role, ref_slot=slot)
            _store(f"{prefix}_sub_ux", sub_ux, set_id=set_id, condition_type="reference", ref_role=role, ref_slot=slot)
            meta_rows.append(
                {
                    "network_seed": network_seed,
                    "set_id": set_id,
                    "item_role": role,
                    "item_image_id": int(row["item_image_id"]),
                    "item_label": int(row["item_label"]),
                    "temporal_slot": slot,
                    "reference_seed": int(row["reference_seed"]),
                    "seq_len": int(row["seq_len"]),
                    "sample_ms": int(cfg.sample_ms),
                    "delay_ms": int(cfg.delay_ms),
                    "storage_prefix": prefix,
                }
            )

    np.savez_compressed(out_dir / "state_bank_layer2.npz", **payload)
    pd.DataFrame(manifest_rows).to_csv(out_dir / "state_bank_manifest.csv", index=False, encoding="utf-8")
    pd.DataFrame(meta_rows).to_csv(out_dir / "sequence_meta.csv", index=False, encoding="utf-8")
    summary = {
        "network_seed": network_seed,
        "n_order_trials": int(len(order_rows)),
        "n_reference_trials": int(len(ref_rows)),
        "n_stored_arrays": int(len(payload)),
        "sample_ms": int(cfg.sample_ms),
        "delay_ms": int(cfg.delay_ms),
        "layer": LAYER,
        "dim_sub_ux": int(payload["order_00_0_sub_ux"].shape[0]),
        "warnings": list(ctx.warnings),
    }
    (out_dir / "capture_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


__all__ = [
    "LAYER",
    "STATE_VARIABLES",
    "build_dataset_index",
    "build_simulation_context",
    "capture_network_state_bank",
]
