from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch

from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig1.constants import NUM_CLASSES
from src.experiments.paper_figures.fig1.subexperiments.helpers import _encode_cached, _iter_batches, _progress
from src.experiments.paper_figures.fig1.types import ExperimentContext

def run_baseline_eval(ctx: ExperimentContext, trials: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    batches = _iter_batches(trials, ctx.cfg.batch_size)
    for batch in _progress(
        batches,
        total=math.ceil(len(trials) / ctx.cfg.batch_size),
        desc="fig1 baseline batches",
        enabled=ctx.cfg.show_progress,
    ):
        spikes = _encode_cached(ctx, batch["image_id"].to_numpy(), ctx.cfg.sample_steps, cache=encode_cache)
        with torch.no_grad():
            _ = ctx.net(spikes, layer_idx=3, monitor=False)
        pred, fire_t = decode_prediction_and_fire_time_from_layer3(ctx.net, len(batch))
        pred_np = pred.numpy().astype(int, copy=False)
        fire_np = fire_t.numpy().astype(int, copy=False)
        for i, rec in enumerate(batch.to_dict("records")):
            label = int(rec["label"])
            prediction = int(pred_np[i])
            silent = prediction < 0
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": int(rec["trial_id"]),
                    "image_id": int(rec["image_id"]),
                    "label": label,
                    "prediction": prediction,
                    "correct": int(prediction == label),
                    "first_fire_time_ms": -1 if silent else int(fire_np[i]),
                    "silent": int(silent),
                }
            )
    pred_df = pd.DataFrame(rows)
    _save_csv(ctx, pred_df, ctx.raw_dir / "panel_b_baseline_trial_predictions.csv")

    n = max(1, len(pred_df))
    n_correct = int(pred_df["correct"].sum()) if not pred_df.empty else 0
    n_silent = int(pred_df["silent"].sum()) if not pred_df.empty else 0
    metrics = pd.DataFrame(
        [
            {
                "network_seed": int(ctx.cfg.network_seed),
                "overall_recall": float(n_correct / n),
                "error_rate": float(1.0 - n_correct / n),
                "n_trials": int(len(pred_df)),
                "n_correct": n_correct,
                "n_silent": n_silent,
                "silent_rate": float(n_silent / n),
            }
        ]
    )
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_b_baseline_metrics_by_network.csv")

    recall_rows = []
    for digit in range(NUM_CLASSES):
        sub = pred_df[pred_df["label"] == digit]
        denom = max(1, len(sub))
        recall_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "digit": digit,
                "class_recall": float(sub["correct"].sum() / denom),
                "n_trials": int(len(sub)),
                "n_correct": int(sub["correct"].sum()),
            }
        )
    _save_csv(ctx, pd.DataFrame(recall_rows), ctx.metrics_dir / "supp_class_recall_by_digit.csv")

    conf_rows = []
    for true_label in range(NUM_CLASSES):
        for pred_label in range(-1, NUM_CLASSES):
            count = int(((pred_df["label"] == true_label) & (pred_df["prediction"] == pred_label)).sum())
            conf_rows.append({"network_seed": int(ctx.cfg.network_seed), "true_label": true_label, "pred_label": pred_label, "count": count})
    _save_csv(ctx, pd.DataFrame(conf_rows), ctx.metrics_dir / "supp_confusion_matrix_long.csv")
    ctx.completed_modules["baseline"] = True
