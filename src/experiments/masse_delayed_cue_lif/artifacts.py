"""JSON, checkpoint, and manifest helpers for the delayed-cue DAG."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
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
REQUIRED_DECODE_INPUTS = (
    "run_config.json",
    "data/trials.csv",
    "data/checkpoints/best.pt",
)
REQUIRED_PLOT_OUTPUTS = (
    "figures/training_curves.png",
    "figures/condition_accuracy.png",
    "figures/rule_match_confusion.png",
    "figures/example_trial_timeline.png",
)
DECODE_PLOT_INPUT = "metrics/decode_metrics.json"
DECODE_PLOT_OUTPUTS = ("figures/decode_accuracy.png",)
TRAIN_COMPLETE_STATUSES = frozenset({"trained", "evaluated"})
COMMAND_LINEAGE_INPUTS = {
    "train": REQUIRED_TRAIN_INPUTS,
    "evaluate": REQUIRED_EVAL_INPUTS,
    "decode": REQUIRED_DECODE_INPUTS,
}
COMMAND_LINEAGE_STAMP = {
    "train": "data/train_history.json",
    "evaluate": "metrics/test_metrics.json",
    "decode": "metrics/decode_metrics.json",
}
COMMAND_OUTPUTS = {
    "train": ("data/checkpoints/best.pt", "data/train_history.json"),
    "evaluate": ("data/test_predictions.csv", "metrics/test_metrics.json"),
    "decode": ("metrics/decode_metrics.json",),
}


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


def input_lineage(run_directory: Path, relative_paths: Sequence[str]) -> dict[str, Any]:
    run_directory = Path(run_directory)
    files: dict[str, str | None] = {}
    for relative in relative_paths:
        path = run_directory / relative
        files[relative] = file_sha256(path) if path.is_file() else None
    identity = None
    config_path = run_directory / "run_config.json"
    if config_path.is_file():
        try:
            identity = read_json(config_path).get("identity")
        except (OSError, json.JSONDecodeError, AttributeError):
            identity = None
    return {"identity": identity, "files": files}


def attach_input_lineage(
    payload: dict[str, Any],
    run_directory: Path,
    relative_paths: Sequence[str],
) -> dict[str, Any]:
    payload["lineage"] = input_lineage(run_directory, relative_paths)
    return payload


def profile_requires_decode_plot(profile: str) -> bool:
    return str(profile).startswith("stripped_")


def required_plot_inputs(config: MasseDelayedCueConfig) -> tuple[str, ...]:
    if profile_requires_decode_plot(config.profile):
        return REQUIRED_PLOT_INPUTS + (DECODE_PLOT_INPUT,)
    return REQUIRED_PLOT_INPUTS


def required_plot_outputs(config: MasseDelayedCueConfig | None) -> tuple[str, ...]:
    if config is not None and profile_requires_decode_plot(config.profile):
        return REQUIRED_PLOT_OUTPUTS + DECODE_PLOT_OUTPUTS
    return REQUIRED_PLOT_OUTPUTS


def _config_for_manifest(run_directory: Path) -> MasseDelayedCueConfig | None:
    config_path = Path(run_directory) / "run_config.json"
    if not config_path.is_file():
        return None
    try:
        return load_run_config(run_directory)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def require_identical_trial_tables(*paths: Path) -> str:
    if len(paths) < 2:
        raise ValueError("need at least two trial tables to compare")
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing trial tables: {', '.join(missing)}")
    hashes = {Path(path): file_sha256(Path(path)) for path in paths}
    unique = set(hashes.values())
    if len(unique) != 1:
        detail = ", ".join(f"{path}={digest[:12]}" for path, digest in hashes.items())
        raise ValueError(f"trial tables are not identical: {detail}")
    return next(iter(unique))


def command_is_fresh(run_directory: Path, command: str) -> bool:
    run_directory = Path(run_directory)
    if command == "plot":
        config = _config_for_manifest(run_directory)
        if config is None:
            return False
        outputs = required_plot_outputs(config)
        if any(not (run_directory / relative).is_file() for relative in outputs):
            return False
        try:
            stored = read_json(run_directory / "artifact_manifest.json").get("plot_lineage")
        except (OSError, json.JSONDecodeError):
            return False
        current = input_lineage(run_directory, required_plot_inputs(config))
        return stored == current
    outputs = COMMAND_OUTPUTS.get(command)
    inputs = COMMAND_LINEAGE_INPUTS.get(command)
    stamp = COMMAND_LINEAGE_STAMP.get(command)
    if outputs is None or inputs is None or stamp is None:
        return False
    if any(not (run_directory / relative).is_file() for relative in outputs):
        return False
    if command == "train":
        try:
            status = str(read_json(run_directory / "summary.json").get("status", ""))
        except (OSError, json.JSONDecodeError):
            return False
        if status not in TRAIN_COMPLETE_STATUSES:
            return False
    try:
        stored = read_json(run_directory / stamp).get("lineage")
    except (OSError, json.JSONDecodeError, AttributeError):
        return False
    current = input_lineage(run_directory, inputs)
    if stored:
        return stored == current
    if command != "train":
        return False
    return _train_pre_lineage_is_fresh(run_directory)


def _train_pre_lineage_is_fresh(run_directory: Path) -> bool:
    checkpoint_path = run_directory / "data" / "checkpoints" / "best.pt"
    try:
        config = load_run_config(run_directory)
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError):
        return False
    return checkpoint.get("identity") == config_identity(config)


def config_identity(config: MasseDelayedCueConfig) -> str:
    payload = {
        key: value
        for key, value in config.to_dict().items()
        if key not in {"device"}
    }
    legacy_dynamics = (
        bool(payload.get("use_synaptic_current", True))
        and not bool(payload.get("use_stsp", False))
        and float(payload.get("recurrent_weight_scale", 1.0)) == 1.0
        and float(payload.get("spike_cost", 0.0)) == 0.0
    )
    if legacy_dynamics:
        for key in ("use_synaptic_current", "use_stsp", "recurrent_weight_scale", "spike_cost"):
            payload.pop(key, None)
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
    handle, temporary_name = tempfile.mkstemp(suffix=".pt")
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        destination_tmp = path.with_suffix(path.suffix + ".tmp")
        destination_tmp.write_bytes(temporary.read_bytes())
        destination_tmp.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
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


def default_manifest(config: MasseDelayedCueConfig | None = None) -> dict[str, Any]:
    plot_depends = list(
        required_plot_inputs(config) if config is not None else REQUIRED_PLOT_INPUTS
    )
    plot_outputs = list(required_plot_outputs(config))
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
            "decode": {
                "depends_on": [
                    "data/checkpoints/best.pt",
                    "data/trials.csv",
                    "run_config.json",
                ],
                "outputs": [
                    "metrics/decode_metrics.json",
                    "data/delay_end_features.pt",
                ],
            },
            "plot": {
                "depends_on": plot_depends,
                "outputs": plot_outputs,
                "plot_only": True,
            },
        },
    }


def write_manifest(run_directory: Path, extra: Mapping[str, Any] | None = None) -> Path:
    payload = default_manifest(_config_for_manifest(run_directory))
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
