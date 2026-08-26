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
    attach_input_lineage,
    config_identity,
    layout_for,
    load_checkpoint,
    load_run_config,
    read_json,
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


def parameter_grads_are_usable(model: torch.nn.Module) -> bool:
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not grads:
        return False
    if not all(torch.isfinite(grad).all() for grad in grads):
        return False
    total_norm = torch.norm(torch.stack([grad.detach().float().norm() for grad in grads]))
    return bool(torch.isfinite(total_norm))


def clip_gradients_per_parameter(model: torch.nn.Module, max_norm: float = 0.1) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            torch.nn.utils.clip_grad_norm_([parameter], max_norm=max_norm)


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


def _slim_split_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "records"}


def _history_from_checkpoint(metrics: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not metrics:
        return []
    history = metrics.get("history")
    if isinstance(history, list):
        return [row for row in history if isinstance(row, dict)]
    if isinstance(history, dict) and "epoch" in history:
        return [history]
    return []


def _restore_training(
    *,
    model: RecurrentLifSfa,
    optimizer: torch.optim.Optimizer,
    config: MasseDelayedCueConfig,
    device: torch.device,
    last_path: Path,
    best_path: Path,
    history_path: Path,
) -> tuple[int, list[dict[str, Any]], float, int, int]:
    history: list[dict[str, Any]] = []
    best_val = -1.0
    best_epoch = -1
    epochs_without_improve = 0
    start_epoch = 0
    if not last_path.is_file():
        return start_epoch, history, best_val, best_epoch, epochs_without_improve
    checkpoint = load_checkpoint(last_path, map_location=device)
    if checkpoint.get("identity") != config_identity(config):
        print(
            "[masse_delayed_cue_lif] ignoring last.pt because identity does not match",
            file=sys.stderr,
            flush=True,
        )
        return start_epoch, history, best_val, best_epoch, epochs_without_improve
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    if history_path.is_file():
        stored = read_json(history_path)
        history = [row for row in stored.get("epochs", []) if isinstance(row, dict)]
        if stored.get("best_epoch") is not None:
            best_epoch = int(stored["best_epoch"])
        if stored.get("best_val_trial_accuracy") is not None:
            best_val = float(stored["best_val_trial_accuracy"])
    if not history:
        history = _history_from_checkpoint(checkpoint.get("metrics"))
    if best_path.is_file():
        best = load_checkpoint(best_path, map_location="cpu")
        if best.get("identity") == config_identity(config):
            best_epoch = int(best["epoch"])
            val_metrics = best.get("metrics", {}).get("val", {})
            if isinstance(val_metrics, dict) and "trial_accuracy" in val_metrics:
                best_val = float(val_metrics["trial_accuracy"])
    if best_val < 0.0 and history:
        best_row = max(history, key=lambda row: float(row.get("val_trial_accuracy", -1.0)))
        best_val = float(best_row.get("val_trial_accuracy", -1.0))
        best_epoch = int(best_row.get("epoch", best_epoch))
    if best_val >= 0.0:
        epochs_without_improve = 0
        for row in reversed(history):
            if float(row.get("val_trial_accuracy", -1.0)) >= best_val - 1e-12:
                break
            epochs_without_improve += 1
    print(
        (
            f"[masse_delayed_cue_lif] resume from epoch {start_epoch + 1}/"
            f"{config.max_epochs} best_val={best_val:.3f} "
            f"stale={epochs_without_improve}"
        ),
        file=sys.stderr,
        flush=True,
    )
    return start_epoch, history, best_val, best_epoch, epochs_without_improve


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

    checkpoint_dir = layout.data_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    history_path = layout.data_dir / "train_history.json"
    start_epoch, history, best_val, best_epoch, epochs_without_improve = _restore_training(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
        last_path=last_path,
        best_path=best_path,
        history_path=history_path,
    )
    patience = config.early_stopping_patience
    already_stopped = (
        patience is not None
        and history
        and epochs_without_improve >= patience
    )
    if already_stopped or start_epoch >= config.max_epochs:
        start_epoch = config.max_epochs

    for epoch in range(start_epoch, config.max_epochs):
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
            logits, state = model(batch["inputs"])
            loss = weighted_cross_entropy(logits, batch["targets"], batch["weights"])
            loss = loss + 5.0 * trial_cross_entropy(logits, batch["targets"], config)
            if config.spike_cost > 0.0 and state.spike_power is not None:
                loss = loss + float(config.spike_cost) * state.spike_power
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            if parameter_grads_are_usable(model):
                clip_gradients_per_parameter(model)
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
        improved = val_metrics["trial_accuracy"] > best_val
        if improved:
            best_val = val_metrics["trial_accuracy"]
            best_epoch = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        metrics_payload = {
            "train": _slim_split_metrics(train_metrics),
            "val": _slim_split_metrics(val_metrics),
            "history": history,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "epochs_without_improve": epochs_without_improve,
        }
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
        if improved:
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics=metrics_payload,
            )
        history_payload = {
            "identity": config_identity(config),
            "best_epoch": best_epoch,
            "best_val_trial_accuracy": best_val,
            "epochs": history,
        }
        attach_input_lineage(history_payload, run_directory, REQUIRED_TRAIN_INPUTS)
        write_json(history_path, history_payload)
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
    attach_input_lineage(history_payload, run_directory, REQUIRED_TRAIN_INPUTS)
    write_json(history_path, history_payload)
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
