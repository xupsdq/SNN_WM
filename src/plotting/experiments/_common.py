from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config.yaml_loader import load_yaml_file, nested_get
from src.plotting.common.io import apply_publication_style, save_figure_all_formats, validate_required_columns

Plotter = Callable[[Path], Mapping[str, Figure]]


def require_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Required artifact not found: {path_obj}")
    return path_obj


def resolve_bundle_file(input_dir: Path, relative_name: str | Path) -> Path:
    rel = Path(relative_name)
    if rel.is_absolute():
        return require_path(rel)
    candidates = [
        input_dir / rel,
        input_dir / "data" / rel,
        input_dir / "metrics" / rel,
        input_dir / "meta" / rel,
        input_dir / "arrays" / rel,
        input_dir / "logs" / rel,
        input_dir / "log" / rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Required bundle artifact not found: {rel}; searched: {searched}")


def optional_bundle_file(input_dir: Path, relative_name: str | Path) -> Path | None:
    try:
        return resolve_bundle_file(input_dir, relative_name)
    except FileNotFoundError:
        return None


def read_csv_validated(path: str | Path, required_columns: Sequence[str] = ()) -> pd.DataFrame:
    csv_path = require_path(path)
    df = pd.read_csv(csv_path)
    if required_columns:
        validate_required_columns(df, list(required_columns))
    return df


def read_bundle_csv(input_dir: Path, relative_name: str | Path, required_columns: Sequence[str] = ()) -> pd.DataFrame:
    return read_csv_validated(resolve_bundle_file(input_dir, relative_name), required_columns=required_columns)


def load_json(path: str | Path) -> Any:
    return json.loads(require_path(path).read_text(encoding="utf-8"))


def load_bundle_json(input_dir: Path, relative_name: str | Path) -> Any:
    return load_json(resolve_bundle_file(input_dir, relative_name))


def load_bundle_npz(input_dir: Path, relative_name: str | Path) -> dict[str, np.ndarray]:
    npz_path = resolve_bundle_file(input_dir, relative_name)
    with np.load(npz_path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def save_named_figures(figures: Mapping[str, Figure], output_dir: Path) -> dict[str, dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, dict[str, str]] = {}
    for stem, fig in figures.items():
        try:
            saved[stem] = save_figure_all_formats(fig, output_dir / stem)
        finally:
            plt.close(fig)
    return saved


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _read_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _relativize_files(root: Path) -> list[str]:
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
    return files


def _paths_are_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _relativize_saved_paths(saved_figures: Mapping[str, Mapping[str, str]], root: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for stem, formats in saved_figures.items():
        out[stem] = {}
        for ext, value in formats.items():
            path = Path(value)
            out[stem][ext] = path.resolve().relative_to(root.resolve()).as_posix() if _paths_are_within(path, root) else str(path)
    return out


def _ensure_plot_bundle_metadata(
    input_dir: Path,
    *,
    experiment_id: str,
    output_dir: Path,
    saved_figures: Mapping[str, Mapping[str, str]],
) -> None:
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (input_dir / name).mkdir(parents=True, exist_ok=True)

    run_log = input_dir / "logs" / "plot_replay.log"
    run_log.write_text(
        "\n".join(
            [
                f"timestamp={_timestamp_now()}",
                f"experiment_id={experiment_id}",
                f"input_dir={input_dir}",
                f"output_dir={output_dir}",
                f"command={subprocess.list2cmdline(sys.argv)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_info_path = input_dir / "meta" / "run_info.json"
    if not run_info_path.exists():
        run_info = {
            "command": subprocess.list2cmdline(sys.argv),
            "config_file": None,
            "dataset": "",
            "entry_script": f"python -m src.plotting.experiments.{experiment_id}_plot",
            "experiment_name": experiment_id,
            "finished_at": _timestamp_now(),
            "git_commit": _read_git_commit(),
            "model_path": None,
            "output_dir": str(input_dir.resolve()),
            "seed": None,
            "started_at": _timestamp_now(),
            "status": "plot-only-replay",
        }
        run_info_path.write_text(json.dumps(_to_json_safe(run_info), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    manifest_path = input_dir / "artifact_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                manifest = {}
        except Exception:
            manifest = {}
    else:
        manifest = {}
    files = set(_relativize_files(input_dir))
    files.add("artifact_manifest.json")
    manifest.update(
        {
            "experiment_id": experiment_id,
            "files": sorted(files),
            "plot_output_dir": output_dir.relative_to(input_dir).as_posix() if _paths_are_within(output_dir, input_dir) else str(output_dir),
            "plot_outputs": _relativize_saved_paths(saved_figures, input_dir),
            "plot_replayed_at": _timestamp_now(),
        }
    )
    manifest_path.write_text(json.dumps(_to_json_safe(manifest), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def apply_plot_style() -> None:
    apply_publication_style()


def empty_figure(message: str, *, title: str | None = None, figsize: tuple[float, float] = (6.0, 4.0)) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    return fig


def mean_sem(df: pd.DataFrame, group_cols: Sequence[str], value_col: str) -> pd.DataFrame:
    validate_required_columns(df, [*group_cols, value_col])
    grouped = df.groupby(list(group_cols), sort=True)[value_col]
    out = grouped.agg(["mean", "count", "std"]).reset_index()
    out["sem"] = out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    return out


def default_input_dir(experiment_id: str) -> Path:
    return Path("results") / experiment_id


def build_plot_parser(experiment_id: str, title: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Plot-only entrypoint for {title or experiment_id}.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def _resolve_from_config(config: dict[str, Any] | None, *path: str, default: Any = None) -> Any:
    if not config:
        return default
    value = nested_get(config, *path, default=None)
    if value is not None:
        return value
    if len(path) == 1:
        return config.get(path[0], default)
    return default


def _apply_plot_config_defaults(args: argparse.Namespace, experiment_id: str) -> argparse.Namespace:
    config_payload = load_yaml_file(args.config) if args.config else {}
    config_input_dir = _resolve_from_config(
        config_payload,
        "input_dir",
        default=_resolve_from_config(config_payload, "experiment", "output_dir"),
    )
    args.input_dir = args.input_dir or config_input_dir or str(default_input_dir(experiment_id))
    args.output_dir = args.output_dir or _resolve_from_config(
        config_payload,
        "plotting",
        "output_dir",
        default=_resolve_from_config(config_payload, "output_dir"),
    )
    return args


def main_for(experiment_id: str, plotter: Plotter, *, title: str | None = None) -> int:
    parser = build_plot_parser(experiment_id, title=title)
    args = _apply_plot_config_defaults(parser.parse_args(), experiment_id)
    input_dir = Path(args.input_dir).resolve()
    require_path(input_dir / "summary.json")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / "figures"
    apply_plot_style()
    figures = plotter(input_dir)
    if not figures:
        raise RuntimeError(f"{experiment_id}: plotter returned no figures")
    saved = save_named_figures(figures, output_dir)
    _ensure_plot_bundle_metadata(input_dir, experiment_id=experiment_id, output_dir=output_dir, saved_figures=saved)
    return 0


__all__ = [
    "apply_plot_style",
    "default_input_dir",
    "empty_figure",
    "load_bundle_json",
    "load_bundle_npz",
    "load_json",
    "main_for",
    "mean_sem",
    "optional_bundle_file",
    "read_bundle_csv",
    "read_csv_validated",
    "require_path",
    "resolve_bundle_file",
    "save_named_figures",
]
