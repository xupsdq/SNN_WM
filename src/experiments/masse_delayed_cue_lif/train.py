"""Train node: consume a persisted trial table and write checkpoints."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .artifacts import (
    REQUIRED_TRAIN_INPUTS,
    config_identity,
    layout_for,
    load_run_config,
    require_files,
    save_checkpoint,
    save_run_config,
    write_json,
    write_manifest,
    write_summary,
)
from .config import MasseDelayedCueConfig
from .metrics import (
    attach_timepoint_breakdown,
    summarize_predictions,
    trial_cross_entropy,
    trial_predictions,
    weighted_cross_entropy,
)
from .model import RecurrentLifSfa
from .task import expand_rows, load_trial_table, window_indices


class TrialTensorDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], config: MasseDelayedCueConfig):
        self.rows = rows
        self.config = config
        inputs, targets, weights = expand_rows(rows, config)
        self.inputs = inputs
        self.targets = targets
        self.weights = weights

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "inputs": self.inputs[:, index],
            "targets": self.targets[:, index],
            "weights": self.weights[:, index],
            "row": self.rows[index],
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    inputs = torch.stack([item["inputs"] for item in batch], dim=1)
    targets = torch.stack([item["targets"] for item in batch], dim=1)
    weights = torch.stack([item["weights"] for item in batch], dim=1)
    rows = [item["row"] for item in batch]
    return {"inputs": inputs, "targets": targets, "weights": weights, "rows": rows}


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "inputs": batch["inputs"].to(device),
        "targets": batch["targets"].to(device),
        "weights": batch["weights"].to(device),
        "rows": batch["rows"],
    }


def _loader(
    rows: list[dict[str, object]],
    config: MasseDelayedCueConfig,
    *,
    shuffle: bool,
    generator: torch.Generator | None,
    dataset: TrialTensorDataset | None = None,
) -> DataLoader:
    if dataset is None:
        dataset = TrialTensorDataset(rows, config)
    return DataLoader(
        dataset,
        batch_size=min(config.batch_size, max(1, len(dataset))),
        shuffle=shuffle,
        generator=generator,
        collate_fn=_collate,
        num_workers=0,
        drop_last=False,
    )


@torch.no_grad()
def _evaluate_split(
    model: RecurrentLifSfa,
    rows: list[dict[str, object]],
    config: MasseDelayedCueConfig,
    device: torch.device,
    dataset: TrialTensorDataset | None = None,
) -> dict[str, Any]:
    model.eval()
    loader = _loader(rows, config, shuffle=False, generator=None, dataset=dataset)
    all_logits = []
    all_targets = []
    all_rows: list[dict[str, object]] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        logits, _ = model(batch["inputs"])
        all_logits.append(logits)
        all_targets.append(batch["targets"])
        all_rows.extend(batch["rows"])
    logits = torch.cat(all_logits, dim=1)
    targets = torch.cat(all_targets, dim=1)
    records = trial_predictions(logits, all_rows, config)
    summary = summarize_predictions(records)
    attach_timepoint_breakdown(summary, logits, targets, all_rows, config)
    summary["records"] = records
    return summary


def train_run(run_directory: Path, config: MasseDelayedCueConfig | None = None) -> dict[str, Any]:
    run_directory = Path(run_directory)
    require_files(run_directory, REQUIRED_TRAIN_INPUTS)
    stored = load_run_config(run_directory)
    if config is None:
        config = stored
    layout = layout_for(run_directory)
    save_run_config(run_directory, config)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    train_rows = load_trial_table(run_directory / "data" / "trials.csv", split="train")
    val_rows = load_trial_table(run_directory / "data" / "trials.csv", split="val")
    torch.manual_seed(config.model_init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.model_init_seed)

    model = RecurrentLifSfa(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    order_generator = torch.Generator()
    order_generator.manual_seed(config.train_order_seed)
    train_dataset = TrialTensorDataset(train_rows, config)
    val_dataset = None if config.profile == "overfit" else TrialTensorDataset(val_rows, config)

    history: list[dict[str, Any]] = []
    best_val = -1.0
    best_epoch = -1
    epochs_without_improve = 0
    checkpoint_dir = layout.data_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    for epoch in range(config.max_epochs):
        model.train()
        loader = _loader(
            train_rows, config, shuffle=True, generator=order_generator, dataset=train_dataset
        )
        epoch_loss = 0.0
        n_batches = 0
        epoch_started = time.perf_counter()
        train_logit_chunks: list[torch.Tensor] = []
        train_target_chunks: list[torch.Tensor] = []
        train_epoch_rows: list[dict[str, object]] = []
        for batch in loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch["inputs"])
            loss = weighted_cross_entropy(logits, batch["targets"], batch["weights"])
            loss = loss + 5.0 * trial_cross_entropy(logits, batch["targets"], config)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            finite_grads = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
            if finite_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
            train_logit_chunks.append(logits.detach())
            train_target_chunks.append(batch["targets"])
            train_epoch_rows.extend(batch["rows"])

        train_logits = torch.cat(train_logit_chunks, dim=1)
        train_targets = torch.cat(train_target_chunks, dim=1)
        train_records = trial_predictions(train_logits, train_epoch_rows, config)
        train_metrics = summarize_predictions(train_records)
        attach_timepoint_breakdown(
            train_metrics, train_logits, train_targets, train_epoch_rows, config
        )
        if config.profile == "overfit":
            val_metrics = train_metrics
        else:
            val_metrics = _evaluate_split(
                model, val_rows, config, device, dataset=val_dataset
            )
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(n_batches, 1),
            "train_trial_accuracy": train_metrics["trial_accuracy"],
            "val_trial_accuracy": val_metrics["trial_accuracy"],
            "val_timepoint_accuracy": val_metrics["timepoint_accuracy"],
        }
        history.append(record)
        print(
            (
                f"[masse_delayed_cue_lif] epoch {epoch + 1}/{config.max_epochs} "
                f"loss={record['train_loss']:.4f} "
                f"train_trial={record['train_trial_accuracy']:.3f} "
                f"val_trial={record['val_trial_accuracy']:.3f} "
                f"val_tp={record['val_timepoint_accuracy']:.3f} "
                f"best_val={max(best_val, record['val_trial_accuracy']):.3f} "
                f"elapsed={time.perf_counter() - epoch_started:.1f}s"
            ),
            file=sys.stderr,
            flush=True,
        )
        metrics_payload = {"train": train_metrics, "val": val_metrics, "history": record}
        write_last = config.profile != "overfit" or train_metrics["trial_accuracy"] >= 0.80 or epoch + 1 == config.max_epochs
        if write_last:
            save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics=metrics_payload,
            )
        improved = val_metrics["trial_accuracy"] > best_val
        if improved:
            best_val = val_metrics["trial_accuracy"]
            best_epoch = epoch
            epochs_without_improve = 0
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics=metrics_payload,
            )
        else:
            epochs_without_improve += 1
        if (
            config.early_stopping_patience is not None
            and epochs_without_improve >= config.early_stopping_patience
        ):
            break
        if config.profile == "overfit" and train_metrics["trial_accuracy"] >= 0.80:
            break

    if not last_path.is_file():
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=max(best_epoch, 0),
            config=config,
            metrics={"history": history[-1] if history else {}},
        )

    history_payload = {
        "identity": config_identity(config),
        "best_epoch": best_epoch,
        "best_val_trial_accuracy": best_val,
        "epochs": history,
    }
    write_json(layout.data_dir / "train_history.json", history_payload)
    write_manifest(run_directory)
    summary = {
        "status": "trained",
        "profile": config.profile,
        "best_epoch": best_epoch,
        "best_val_trial_accuracy": best_val,
        "n_epochs": len(history),
        "identity": config_identity(config),
    }
    write_summary(run_directory, summary)
    return {
        **summary,
        "history": history,
        "final_train_trial_accuracy": history[-1]["train_trial_accuracy"] if history else 0.0,
        "first_train_loss": history[0]["train_loss"] if history else None,
        "last_train_loss": history[-1]["train_loss"] if history else None,
        "best_path": str(best_path),
        "last_path": str(last_path),
        "windows": {key: [value.start, value.stop] for key, value in window_indices(config).items()},
    }
