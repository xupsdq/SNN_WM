from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.paper_figures.fig1.subexperiments.helpers import (
    _encode_cached,
    _iter_batches,
)
from src.experiments.paper_figures.fig1.cache_keys import trial_specs_hash
from src.experiments.paper_figures.fig1.types import ExperimentContext


OUTPUT_NAME = "supp_time_binned_firing_rates.csv"
LAYER_KEYS = ("layer1", "layer2", "layer3")


def _ms_to_steps(value_ms: int, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


def _time_binned_batch(
    ctx: ExperimentContext,
    sample_spikes: torch.Tensor,
    batch: pd.DataFrame,
    *,
    bin_ms: int,
) -> list[dict[str, Any]]:
    bin_steps = _ms_to_steps(bin_ms, ctx.cfg.dt)
    sample_steps = int(ctx.cfg.dms_sample_steps)
    delay_steps = int(ctx.cfg.dms_delay_steps)
    total_steps = sample_steps + delay_steps
    if sample_steps % bin_steps or delay_steps % bin_steps:
        raise ValueError(
            "Fig.1 time-binned firing requires both stimulus and delay durations "
            f"to be divisible by bin_ms; sample_steps={sample_steps}, "
            f"delay_steps={delay_steps}, bin_steps={bin_steps}"
        )

    net = ctx.net
    batch_size, _, channels, height, width = sample_spikes.shape
    prepare_network_state(net, batch_size, channels, height, width)
    zero_input = torch.zeros(
        (batch_size, channels, height, width),
        device=ctx.device,
    )
    counts = {
        layer: torch.zeros(batch_size, dtype=torch.float64, device=ctx.device)
        for layer in LAYER_KEYS
    }
    trial_ids = batch["trial_id"].to_numpy(dtype=np.int64)
    rows: list[dict[str, Any]] = []

    def flush(bin_index: int) -> None:
        start_step = int(bin_index * bin_steps)
        end_step = int(start_step + bin_steps)
        start_ms = float(start_step * ctx.cfg.dt / ms)
        end_ms = float(end_step * ctx.cfg.dt / ms)
        midpoint_ms = 0.5 * (start_ms + end_ms)
        phase = (
            "stimulus"
            if end_step <= sample_steps
            else (
                "early_delay"
                if end_step <= sample_steps + delay_steps // 2
                else "late_delay"
            )
        )
        for layer in LAYER_KEYS:
            layer_counts = counts[layer].detach().cpu().numpy()
            for trial_id, spike_count in zip(trial_ids, layer_counts):
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "trial_id": int(trial_id),
                        "layer": layer,
                        "phase": phase,
                        "bin_start_ms": start_ms,
                        "bin_end_ms": end_ms,
                        "time_ms": midpoint_ms,
                        "time_window_ms": int(bin_ms),
                        "stimulus_start_ms": 0.0,
                        "stimulus_end_ms": float(sample_steps * ctx.cfg.dt / ms),
                        "spike_count": float(spike_count),
                        "spike_rate_hz": float(
                            spike_count / (float(bin_steps) * ctx.cfg.dt)
                        ),
                    }
                )
            counts[layer].zero_()

    with torch.no_grad():
        for time_step in range(total_steps):
            input_t = (
                sample_spikes[:, time_step, ...]
                if time_step < sample_steps
                else zero_input
            )
            s1, _ = net.layer1.forward_step(
                input_t,
                time_step,
                training=False,
                monitor=False,
                stsp_mode="dynamic",
            )
            s1p = net.pool1(s1.float())
            s2, _ = net.layer2.forward_step(
                s1p,
                time_step,
                training=False,
                monitor=False,
                stsp_mode="dynamic",
            )
            s2p = net.pool2(s2.float())
            s3, _ = net.layer3.forward_step(
                s2p,
                time_step,
                training=False,
                monitor=False,
                stsp_mode="dynamic",
            )
            for layer, spikes_t in zip(LAYER_KEYS, (s1, s2, s3)):
                counts[layer].add_(
                    spikes_t.detach().to(torch.float64).flatten(start_dim=1).sum(dim=1)
                )
            if (time_step + 1) % bin_steps == 0:
                flush(time_step // bin_steps)
    return rows


def run_time_binned_firing_rate_control(
    ctx: ExperimentContext,
    dms_trials: pd.DataFrame,
    *,
    bin_ms: int,
) -> None:
    if int(bin_ms) <= 0:
        raise ValueError(f"firing bin must be positive, got {bin_ms}")
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    batches = _iter_batches(dms_trials, int(ctx.cfg.dms_batch_size))
    total_batches = int(
        np.ceil(len(dms_trials) / max(1, int(ctx.cfg.dms_batch_size)))
    )
    iterator = batches
    if ctx.cfg.show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(
            batches,
            total=total_batches,
            desc="fig1 50 ms firing-rate batches",
        )
    for batch in iterator:
        sample_spikes = _encode_cached(
            ctx,
            batch["sample_image_id"].to_numpy(),
            int(ctx.cfg.dms_sample_steps),
            cache=encode_cache,
        )
        rows.extend(
            _time_binned_batch(
                ctx,
                sample_spikes,
                batch,
                bin_ms=int(bin_ms),
            )
        )

    frame = pd.DataFrame(rows)
    expected_bins = (
        int(ctx.cfg.dms_sample_ms) + int(ctx.cfg.dms_delay_ms)
    ) // int(bin_ms)
    expected_rows = len(dms_trials) * len(LAYER_KEYS) * expected_bins
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"Fig.1 time-binned firing row count mismatch: "
            f"expected={expected_rows}, observed={len(frame)}"
        )
    key = ["network_seed", "trial_id", "layer", "bin_start_ms"]
    duplicate_count = int(frame.duplicated(key, keep=False).sum())
    if duplicate_count:
        raise RuntimeError(
            f"Fig.1 time-binned firing contains duplicate rows: {duplicate_count}"
        )
    frame = frame.sort_values(key, kind="mergesort").reset_index(drop=True)
    frame["dms_trial_specs_digest"] = trial_specs_hash({"dms": dms_trials})
    output = ctx.metrics_dir / OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8", lineterminator="\n")
    ctx.completed_modules["firing_rate_control"] = True
    ctx.n_trials["time_binned_firing_rate_control"] = int(len(dms_trials))
    ctx.output_files["time_binned_firing_rates"] = output.relative_to(
        ctx.seed_dir
    ).as_posix()
    ctx.run_log.append(
        f"time-binned firing rate ready bin_ms={int(bin_ms)} rows={len(frame)}"
    )


__all__ = [
    "OUTPUT_NAME",
    "run_time_binned_firing_rate_control",
]
