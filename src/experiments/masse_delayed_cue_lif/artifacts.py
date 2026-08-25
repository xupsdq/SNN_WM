"""JSON, checkpoint, and manifest helpers for the delayed-cue DAG."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.experiments.common.results import ResultLayout, prepare_result_layout
from src.experiments.common.run_info import build_run_info, write_run_info

from .config import MasseDelayedCueConfig, config_from_mapping


REQUIRED_TRAIN_INPUTS = ("run_config.json", "data/trials.csv")
REQUIRED_EVAL_INPUTS = (
    "run_config.json",
    "data/trials.csv",
    "data/checkpoints/best.pt",
)
REQUIRED_PLOT_INPUTS = (
    "run_config.json",
    "data/train_history.json",
    "data/test_predictions.csv",
    "metrics/test_metrics.json",
)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_files(run_directory: Path, relative_paths: tuple[str, ...]) -> None:
    missing = [name for name in relative_paths if not (run_directory / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required artifacts in {run_directory}: {joined}"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def config_identity(config: MasseDelayedCueConfig) -> str:
    payload = {
        key: value
        for key, value in config.to_dict().items()
        if key not in {"device"}
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def layout_for(run_directory: Path) -> ResultLayout:
    return prepare_result_layout(run_directory)


def load_run_config(run_directory: Path) -> MasseDelayedCueConfig:
    payload = read_json(run_directory / "run_config.json")
    return config_from_mapping(payload["task"] if "task" in payload else payload)


def save_run_config(run_directory: Path, config: MasseDelayedCueConfig) -> Path:
    payload = {
        "experiment": "masse_delayed_cue_lif",
        "identity": config_identity(config),
        **config.to_dict(),
    }
    return write_json(run_directory / "run_config.json", payload)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: MasseDelayedCueConfig,
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "config": config.to_dict(),
        "identity": config_identity(config),
        "metrics": dict(metrics or {}),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def write_predictions_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("prediction rows must not be empty")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_predictions_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def default_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "masse_delayed_cue_lif",
        "tasks": {
            "build-trials": {
                "depends_on": [],
                "outputs": ["run_config.json", "data/trials.csv"],
            },
            "train": {
                "depends_on": ["run_config.json", "data/trials.csv"],
                "outputs": [
                    "data/checkpoints/best.pt",
                    "data/checkpoints/last.pt",
                    "data/train_history.json",
                ],
            },
            "evaluate": {
                "depends_on": [
                    "data/checkpoints/best.pt",
                    "data/trials.csv",
                    "run_config.json",
                ],
                "outputs": ["data/test_predictions.csv", "metrics/test_metrics.json"],
            },
            "plot": {
                "depends_on": [
                    "run_config.json",
                    "data/train_history.json",
                    "data/test_predictions.csv",
                    "metrics/test_metrics.json",
                ],
                "outputs": [
                    "figures/training_curves.png",
                    "figures/condition_accuracy.png",
                    "figures/rule_match_confusion.png",
                    "figures/example_trial_timeline.png",
                ],
                "plot_only": True,
            },
        },
    }


def write_manifest(run_directory: Path, extra: Mapping[str, Any] | None = None) -> Path:
    payload = default_manifest()
    if extra:
        if extra.get("plot_only"):
            payload["tasks"]["plot"]["plot_only"] = True
        for key, value in extra.items():
            if key != "plot_only":
                payload[key] = value
    return write_json(run_directory / "artifact_manifest.json", payload)


def write_summary(run_directory: Path, payload: Mapping[str, Any]) -> Path:
    return write_json(run_directory / "summary.json", dict(payload))


def record_run_info(
    run_directory: Path,
    *,
    command: str,
    config: MasseDelayedCueConfig,
    status: str = "running",
) -> Path:
    layout = layout_for(run_directory)
    payload = build_run_info(
        experiment_name="masse_delayed_cue_lif",
        output_dir=run_directory,
        entry_script="src.experiments.masse_delayed_cue_lif.run",
        seed=config.trial_table_seed,
        dataset="masse_delayed_cue_dms_dmrs",
        command=command,
        status=status,
    )
    payload["profile"] = config.profile
    payload["identity"] = config_identity(config)
    return write_run_info(layout.meta_dir, payload)
