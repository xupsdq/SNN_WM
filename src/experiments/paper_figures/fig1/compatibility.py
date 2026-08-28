from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.experiments.paper_figures.fig1.artifacts import (
    load_delay_feature_bank,
    load_dms_boundary_bank,
    load_trial_specs_artifact,
    read_cache_key,
)
from src.experiments.paper_figures.fig1.cache_keys import cache_key_digest
from src.experiments.paper_figures.fig1.constants import FIGURE_ID
from src.experiments.paper_figures.fig1.schemas import (
    TASK_DELAY_FEATURE_BANK,
    TASK_DMS_BOUNDARY_BANK,
    TASK_TRIAL_SPECS,
)


PERSISTED_ARTIFACT_TASKS = (
    TASK_TRIAL_SPECS,
    TASK_DELAY_FEATURE_BANK,
    TASK_DMS_BOUNDARY_BANK,
)


@dataclass(frozen=True)
class ResultBundleCompatibility:
    bundle_root: str
    experiment_id: str
    network_seed: int
    downstream_outputs_compatible: bool
    missing_output_files: tuple[str, ...]
    reusable_artifact_tasks: tuple[str, ...]
    missing_artifact_tasks: tuple[str, ...]
    invalid_artifact_tasks: dict[str, str]
    can_reuse_all_persisted_artifacts: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_result_bundle(bundle_root: str | Path) -> ResultBundleCompatibility:
    root = Path(bundle_root).resolve()
    manifest = _read_json_object(root / "artifact_manifest.json")
    summary = _read_json_object(root / "summary.json")

    experiment_id = str(manifest.get("experiment_id", ""))
    if experiment_id != FIGURE_ID:
        raise ValueError(f"Expected Fig1 experiment_id={FIGURE_ID!r}, found {experiment_id!r}: {root}")
    summary_figure = str(summary.get("figure", ""))
    if summary_figure != FIGURE_ID:
        raise ValueError(f"Expected Fig1 summary figure={FIGURE_ID!r}, found {summary_figure!r}: {root}")

    manifest_seed = int(manifest.get("network_seed"))
    summary_seed = int(summary.get("network_seed"))
    if manifest_seed != summary_seed:
        raise ValueError(
            f"Fig1 bundle network_seed mismatch: manifest={manifest_seed}, summary={summary_seed}: {root}"
        )

    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"Fig1 artifact manifest files must be a mapping: {root / 'artifact_manifest.json'}")
    missing_outputs: list[str] = []
    for name, relative in sorted(files.items(), key=lambda item: str(item[0])):
        path = _safe_bundle_path(root, relative, label=str(name))
        if not path.is_file():
            missing_outputs.append(str(relative).replace("\\", "/"))

    validators: dict[str, Callable[[Path], Any]] = {
        TASK_TRIAL_SPECS: load_trial_specs_artifact,
        TASK_DELAY_FEATURE_BANK: load_delay_feature_bank,
        TASK_DMS_BOUNDARY_BANK: load_dms_boundary_bank,
    }
    reusable: list[str] = []
    missing_artifacts: list[str] = []
    invalid_artifacts: dict[str, str] = {}
    artifact_root = root / "data" / "intermediates"
    for task_id in PERSISTED_ARTIFACT_TASKS:
        task_dir = artifact_root / task_id
        if not task_dir.is_dir():
            missing_artifacts.append(task_id)
            continue
        try:
            _validate_embedded_cache_key(task_dir)
            validators[task_id](task_dir)
        except Exception as exc:
            invalid_artifacts[task_id] = f"{type(exc).__name__}: {exc}"
        else:
            reusable.append(task_id)

    return ResultBundleCompatibility(
        bundle_root=str(root),
        experiment_id=experiment_id,
        network_seed=manifest_seed,
        downstream_outputs_compatible=not missing_outputs,
        missing_output_files=tuple(missing_outputs),
        reusable_artifact_tasks=tuple(reusable),
        missing_artifact_tasks=tuple(missing_artifacts),
        invalid_artifact_tasks=invalid_artifacts,
        can_reuse_all_persisted_artifacts=(
            len(reusable) == len(PERSISTED_ARTIFACT_TASKS)
            and not missing_artifacts
            and not invalid_artifacts
        ),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Fig1 bundle file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _safe_bundle_path(root: Path, value: Any, *, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError(f"Fig1 manifest path for {label!r} must be relative: {value!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Fig1 manifest path for {label!r} escapes bundle root: {value!r}") from exc
    return resolved


def _validate_embedded_cache_key(task_dir: Path) -> None:
    payload = read_cache_key(task_dir)
    found = str(payload.get("cache_key_digest", ""))
    expected = cache_key_digest(payload["cache_key"])
    if found != expected:
        raise RuntimeError(
            f"Artifact cache key digest mismatch in {task_dir}: expected {expected}, found {found}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a historical Fig1 result bundle without modifying it.")
    parser.add_argument("bundle_root")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = inspect_result_bundle(args.bundle_root)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PERSISTED_ARTIFACT_TASKS",
    "ResultBundleCompatibility",
    "inspect_result_bundle",
    "main",
]
