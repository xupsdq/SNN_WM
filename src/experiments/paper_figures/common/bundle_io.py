from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


JsonSafeFn = Callable[[Any], Any]


def resolve_seed_dir(output_root: Path, network_seed: int) -> Path:
    """Resolve a paper-figure output root to the seed-level bundle directory."""
    if output_root.name.startswith("seed_"):
        return output_root
    return output_root / f"seed_{int(network_seed):03d}"


def prepare_seed_dirs(seed_dir: Path, *, include_root_layout: bool = False) -> dict[str, Path]:
    """Create the standard paper-figure seed bundle directories."""
    paths = {
        "config": seed_dir / "config",
        "trial_specs": seed_dir / "data" / "trial_specs",
        "raw": seed_dir / "data" / "raw",
        "metrics": seed_dir / "data" / "metrics",
        "debug": seed_dir / "debug_figures",
        "meta": seed_dir / "meta",
    }
    if include_root_layout:
        paths.update(
            {
                "root_figures": seed_dir / "figures",
                "root_logs": seed_dir / "logs",
                "root_metrics": seed_dir / "metrics",
            }
        )
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def relative_to_root(path: Path, root: Path, *, normalize_outside_root: bool = False) -> str:
    """Return a manifest-friendly path relative to root when possible."""
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        text = str(path)
        return text.replace("\\", "/") if normalize_outside_root else text


def json_safe(value: Any) -> Any:
    """Convert common scientific Python values into JSON-serializable values."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_file(payload: Mapping[str, Any], path: Path, *, json_safe_fn: JsonSafeFn | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    converter = json_safe if json_safe_fn is None else json_safe_fn
    path.write_text(json.dumps(converter(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def save_csv_with_registry(
    ctx: Any,
    df: pd.DataFrame,
    path: Path,
    *,
    normalize_outside_root: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    ctx.output_files[path.stem] = relative_to_root(path, ctx.seed_dir, normalize_outside_root=normalize_outside_root)


def write_run_log(ctx: Any, *, now_text: str) -> None:
    ctx.run_log.append(f"{now_text} completed modules={sorted(k for k, v in ctx.completed_modules.items() if v)}")
    path = ctx.seed_dir / "run_log.txt"
    path.write_text("\n".join(ctx.run_log) + "\n", encoding="utf-8")
    ctx.output_files["run_log"] = "run_log.txt"


def write_artifact_manifest(ctx: Any, *, experiment_id: str, title: str | None = None) -> Path:
    ctx.output_files["artifact_manifest"] = "artifact_manifest.json"
    payload = {
        "experiment_id": str(experiment_id),
        "title": str(title or experiment_id),
        "network_seed": int(getattr(ctx.cfg, "network_seed")),
        "files": dict(sorted(ctx.output_files.items())),
    }
    path = ctx.seed_dir / "artifact_manifest.json"
    write_json_file(payload, path)
    return path


def record_optional_missing(ctx: Any, output_name: str, reason: str, *, message_label: str) -> None:
    missing = ctx.availability.setdefault("supplement_alias_missing_reasons", {})
    missing[output_name] = reason
    message = f"Optional {message_label} {output_name} is empty: {reason}"
    if message not in ctx.warnings:
        ctx.warnings.append(message)


def write_empty_csv_with_warning(
    ctx: Any,
    dst: Path,
    columns: Sequence[str],
    reason: str,
    *,
    message_label: str,
) -> None:
    record_optional_missing(ctx, dst.name, reason, message_label=message_label)
    save_csv_with_registry(ctx, pd.DataFrame(columns=list(columns)), dst)


def copy_csv_alias(
    ctx: Any,
    src: Path,
    dst: Path,
    *,
    empty_columns: Sequence[str],
    reason: str,
    message_label: str,
) -> None:
    if not src.exists():
        write_empty_csv_with_warning(ctx, dst, empty_columns, reason, message_label=message_label)
        return
    df = pd.read_csv(src)
    if df.empty:
        write_empty_csv_with_warning(ctx, dst, list(df.columns) if len(df.columns) else empty_columns, reason, message_label=message_label)
        return
    save_csv_with_registry(ctx, df, dst)


__all__ = [
    "copy_csv_alias",
    "json_safe",
    "prepare_seed_dirs",
    "record_optional_missing",
    "relative_to_root",
    "resolve_seed_dir",
    "save_csv_with_registry",
    "write_artifact_manifest",
    "write_empty_csv_with_warning",
    "write_json_file",
    "write_run_log",
]
