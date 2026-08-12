from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from .builders import BuilderContext, FIGURE_BUILDERS, PanelResult
from .schema import sha256_file, write_csv, write_json


BUILDER_VERSION = "final_six_statistics_v2.4.0"
FINAL_TASK_ID = "final-statistics"
FIGURE_IDS = tuple(f"fig{index}" for index in range(1, 7))
DEFAULT_RELATIVE_OUTPUT = Path(
    "results/paper_figure_multi_seed/final_six_figures"
)
FORBIDDEN_FINAL_STATISTICS_RUNTIME_MODULES = (
    "torch",
    "src.core.network",
    "src.data.encoding",
    "src.experiments.common.mnist_loader",
    "src.experiments.common.model_io",
)

PANEL_CLAIMS = {
    "fig1": {
        "a": "fixed STSP-SNN architecture",
        "b": "network accuracy across fixed seeds",
        "c": "50 ms firing-rate trajectory and delay-period disappearance",
        "d": "delay-period u/x content decoding",
        "e": "error-pool attribution composition",
    },
    "fig2": {
        "a": "distinct A/C histories followed by identical B and paired post-B state/outcome comparison",
        "b": "history-aligned rescue and loss",
        "c": "common update and history effect",
        "d": "changed-event versus size-matched-random residual magnitude",
    },
    "fig3": {
        "a": "overlap-specific causal gate",
        "b": "pre-input retained support",
        "c": "descriptive 30-ms advance, recruit and loss probabilities",
        "d": "Layer-1 STSP contribution to early processing",
        "e": "Layer-2 history-dependent write-back",
        "f": "Layer-1-only causal entry into Layer-2 successor formation",
        "g": "conceptual synthesis of iterative inherited-state updating",
    },
    "fig4": {
        "a": "C5 early Layer-2 processing donor transfer",
        "b": "C5 post-C Layer-3 successor donor transfer",
        "c": "successive observed versus persisted passive displacement",
        "d": "relation-balanced K1-to-K5 rescue and loss shifts",
    },
    "fig5": {
        "a": "both pair constituents retained",
        "b": "experienced-pair specificity",
        "c": "effective component number across load",
        "d": "latest-item-only exclusion",
        "e": "Layer-1 g effective area across load and delay",
        "f": "coefficient-free morphology specificity across load and delay",
    },
    "fig6": {
        "a": "pair partial-cue recovery",
        "b": "multi-item serial access",
        "c": "matched versus same-label-novel and unseen cue specificity",
        "d": "functional K-by-delay boundary",
        "e": "exact-area-and-energy-matched high-STSP-overlap contribution",
        "f": "primary 10-ms STSP-by-overlap expression gate",
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_output(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def is_final_statistics_request(argv: Optional[Sequence[str]]) -> bool:
    values = list(sys.argv[1:] if argv is None else argv)
    for index, value in enumerate(values):
        if value == "--task" and index + 1 < len(values):
            return values[index + 1] == FINAL_TASK_ID
        if value.startswith("--task="):
            return value.split("=", 1)[1] == FINAL_TASK_ID
    return False


def canonical_runner_main(
    figure_id: str,
    argv: Optional[Sequence[str]] = None,
) -> int:
    if figure_id not in FIGURE_IDS:
        raise ValueError(f"Unsupported final figure id: {figure_id}")
    loaded_runtime_modules = [
        module
        for module in FORBIDDEN_FINAL_STATISTICS_RUNTIME_MODULES
        if module in sys.modules
    ]
    if loaded_runtime_modules:
        raise RuntimeError(
            "final-statistics must branch before model/dataset runtime imports; "
            f"already loaded={loaded_runtime_modules}"
        )
    parser = argparse.ArgumentParser(
        description=f"Build load-only frozen statistics for {figure_id}.",
        allow_abbrev=False,
    )
    parser.add_argument("--task", required=True, choices=[FINAL_TASK_ID])
    parser.add_argument("--reuse-artifacts", required=True, choices=["require"])
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RELATIVE_OUTPUT),
        help="Exact final-six bundle root.",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    repo_root = _repo_root()
    output_root = _resolve_output(repo_root, args.output_dir)
    summary = build_figure_statistics(
        figure_id,
        repo_root=repo_root,
        output_root=output_root,
        reuse_artifacts=args.reuse_artifacts,
        check_only=bool(args.check_only),
    )
    summary["load_only_branch"] = True
    summary["model_or_dataset_initialized"] = False
    summary["forbidden_runtime_modules_loaded"] = loaded_runtime_modules
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def build_figure_statistics(
    figure_id: str,
    *,
    repo_root: Path,
    output_root: Path,
    reuse_artifacts: str,
    check_only: bool = False,
) -> dict[str, Any]:
    if figure_id not in FIGURE_BUILDERS:
        raise ValueError(f"No final statistics builder for {figure_id}")
    if reuse_artifacts != "require":
        raise ValueError(
            "Final-six statistics are load-only and require --reuse-artifacts require"
        )
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    if check_only:
        context = BuilderContext(
            repo_root=repo_root,
            output_root=output_root,
            figure_id=figure_id,
            builder_version=BUILDER_VERSION,
        )
        panels = FIGURE_BUILDERS[figure_id](context)
        return _check_summary(figure_id, panels)
    _prepare_output_root(output_root)
    figure_dir = output_root / figure_id
    if figure_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing final figure bundle: {figure_dir}"
        )
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{figure_id}_statistics_", dir=str(output_root))
    )
    try:
        context = BuilderContext(
            repo_root=repo_root,
            output_root=staging_dir,
            figure_id=figure_id,
            builder_version=BUILDER_VERSION,
        )
        panels = FIGURE_BUILDERS[figure_id](context)
        _write_figure_bundle(
            repo_root=repo_root,
            figure_dir=staging_dir,
            figure_id=figure_id,
            panels=panels,
        )
        staging_dir.replace(figure_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    _refresh_top_level(repo_root=repo_root, output_root=output_root)
    return {
        "figure_id": figure_id,
        "status": "statistics_ready",
        "output_dir": str(figure_dir),
        "panel_count": len(panels),
        "quantitative_panels": sum(
            panel.panel_type == "quantitative" for panel in panels
        ),
        "builder_version": BUILDER_VERSION,
    }


def _check_summary(figure_id: str, panels: Sequence[PanelResult]) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "status": "check_passed",
        "load_only_branch": True,
        "model_or_dataset_initialized": False,
        "panel_count": len(panels),
        "quantitative_panels": sum(
            panel.panel_type == "quantitative" for panel in panels
        ),
        "cohort_panels": [
            panel.panel_id for panel in panels if panel.cohort_record is not None
        ],
        "builder_version": BUILDER_VERSION,
    }


def _prepare_output_root(output_root: Path) -> None:
    marker = output_root / "meta/final_six_builder.json"
    if output_root.exists():
        existing = list(output_root.iterdir())
        if existing and not marker.exists():
            raise FileExistsError(
                f"Refusing to reuse a nonempty unregistered output root: {output_root}"
            )
        if marker.exists():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("builder_version") != BUILDER_VERSION:
                raise RuntimeError(
                    f"Output root belongs to builder {payload.get('builder_version')}, "
                    f"not {BUILDER_VERSION}"
                )
    for relative in ("logs", "meta"):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        write_json(
            marker,
            {
                "builder_version": BUILDER_VERSION,
                "created_at": _now(),
                "purpose": "frozen final Fig.1-Fig.6 statistics and plot-only bundle",
            },
        )


def _write_figure_bundle(
    *,
    repo_root: Path,
    figure_dir: Path,
    figure_id: str,
    panels: Sequence[PanelResult],
) -> None:
    for relative in (
        "data",
        "metrics",
        "figures/panels",
        "figures/qa",
        "logs",
        "meta",
    ):
        (figure_dir / relative).mkdir(parents=True, exist_ok=True)
    panel_index_rows: list[dict[str, Any]] = []
    source_frames: list[pd.DataFrame] = []
    cohort_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    for panel in panels:
        panel_id = panel.panel_id
        plot_relative = ""
        if panel.plot_data is not None:
            plot_relative = f"data/panel_{panel_id}_plot_data.csv"
            write_csv(figure_dir / plot_relative, panel.plot_data)
            schema_rows.append(
                _schema_record(
                    figure_id,
                    panel_id,
                    plot_relative,
                    panel.plot_data,
                    "plot_data",
                )
            )
        statistics_relative = f"metrics/panel_{panel_id}_statistics.csv"
        write_csv(figure_dir / statistics_relative, panel.statistics)
        schema_rows.append(
            _schema_record(
                figure_id,
                panel_id,
                statistics_relative,
                panel.statistics,
                "statistics",
            )
        )
        source_relative = f"meta/panel_{panel_id}_source_manifest.csv"
        write_csv(figure_dir / source_relative, panel.source_manifest)
        source_frames.append(panel.source_manifest)
        extra_paths: list[str] = []
        for name, frame in sorted(panel.extra_data.items()):
            relative = f"data/{name}"
            write_csv(figure_dir / relative, frame)
            extra_paths.append(relative)
            schema_rows.append(
                _schema_record(
                    figure_id,
                    panel_id,
                    relative,
                    frame,
                    "panel_auxiliary_data",
                )
            )
        for name, frame in sorted(panel.extra_metrics.items()):
            relative = f"metrics/{name}"
            write_csv(figure_dir / relative, frame)
            extra_paths.append(relative)
            schema_rows.append(
                _schema_record(
                    figure_id,
                    panel_id,
                    relative,
                    frame,
                    "panel_auxiliary_metrics",
                )
            )
        if "asset_manifest" in panel.panel_meta:
            relative = f"meta/panel_{panel_id}_asset_manifest.csv"
            write_csv(figure_dir / relative, panel.panel_meta["asset_manifest"])
            extra_paths.append(relative)
        if "protocol_source_manifest" in panel.panel_meta:
            relative = f"meta/panel_{panel_id}_source_manifest_protocol.csv"
            write_csv(
                figure_dir / relative,
                panel.panel_meta["protocol_source_manifest"],
            )
            # Frozen prompt also requires the unqualified Fig.4a filename.
            write_csv(
                figure_dir / f"meta/panel_{panel_id}_source_manifest.csv",
                panel.panel_meta["protocol_source_manifest"],
            )
            extra_paths.append(relative)
        if panel.cohort_record is not None:
            cohort_rows.append(panel.cohort_record)
        panel_index_rows.append(
            {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "panel_type": panel.panel_type,
                "claim": PANEL_CLAIMS[figure_id][panel_id],
                "plot_data_csv": plot_relative,
                "statistics_csv": statistics_relative,
                "source_manifest_csv": source_relative,
                "auxiliary_csvs": ";".join(extra_paths),
                "cohort_status": (
                    panel.cohort_record["status"]
                    if panel.cohort_record is not None
                    else "not_applicable"
                ),
            }
        )
        _write_panel_manifest(
            figure_dir=figure_dir,
            figure_id=figure_id,
            panel_id=panel_id,
            paths=[
                value
                for value in [
                    plot_relative,
                    statistics_relative,
                    source_relative,
                    *extra_paths,
                ]
                if value
            ],
        )
    panel_index = pd.DataFrame(panel_index_rows)
    write_csv(figure_dir / "meta/panel_index.csv", panel_index)
    cohort = pd.DataFrame(cohort_rows)
    write_csv(figure_dir / "meta/cohort_validation.csv", cohort)
    schema = pd.DataFrame(schema_rows)
    write_csv(figure_dir / "meta/schema_validation.csv", schema)
    source_manifest = pd.concat(source_frames, ignore_index=True, sort=False)
    write_csv(figure_dir / "meta/source_manifest.csv", source_manifest)
    parent_hashes = (
        source_manifest.loc[
            :,
            [
                "figure_id",
                "panel_id",
                "source_path",
                "source_sha256",
                "source_bytes",
            ],
        ]
        .drop_duplicates()
        .sort_values(["figure_id", "panel_id", "source_path"], kind="mergesort")
    )
    write_csv(figure_dir / "meta/parent_hashes_before.csv", parent_hashes)
    run_config = {
        "figure_id": figure_id,
        "task": FINAL_TASK_ID,
        "reuse_artifacts": "require",
        "expected_seeds": list(range(1000, 1020)),
        "builder_version": BUILDER_VERSION,
        "built_at": _now(),
        "git_commit": _git_commit(repo_root),
        "scientific_chain": (
            "inherit -> identical-input conditioning -> successor formation -> "
            "successor reuse and recurrence; morphology and conditional function are "
            "parallel outcome modules"
        ),
        "model_or_dataset_initialized": False,
    }
    write_json(figure_dir / "run_config.json", run_config)
    summary = {
        "figure_id": figure_id,
        "status": "statistics_ready",
        "panel_count": len(panels),
        "quantitative_panel_count": sum(
            panel.panel_type == "quantitative" for panel in panels
        ),
        "schematic_panel_count": sum(
            panel.panel_type == "schematic" for panel in panels
        ),
        "cohort_validation": (
            "pass"
            if cohort.empty or cohort["status"].eq("pass").all()
            else "fail"
        ),
        "parent_file_count": int(parent_hashes["source_path"].nunique()),
        "plot_status": "pending",
    }
    write_json(figure_dir / "summary.json", summary)
    (figure_dir / "logs/statistics_build.log").write_text(
        f"{_now()} {figure_id} statistics build passed; "
        f"quantitative_panels={summary['quantitative_panel_count']}\n",
        encoding="utf-8",
    )
    _write_artifact_manifest(figure_dir)


def _schema_record(
    figure_id: str,
    panel_id: str,
    relative_path: str,
    frame: pd.DataFrame,
    role: str,
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "path": relative_path,
        "role": role,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": ";".join(str(column) for column in frame.columns),
        "duplicate_full_rows": int(frame.duplicated().sum()),
        "status": "pass",
    }


def _write_panel_manifest(
    *,
    figure_dir: Path,
    figure_id: str,
    panel_id: str,
    paths: Sequence[str],
) -> None:
    rows: list[dict[str, Any]] = []
    for relative in paths:
        path = figure_dir / relative
        rows.append(
            {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
                "role": _artifact_role(relative),
            }
        )
    write_csv(figure_dir / f"meta/panel_{panel_id}_manifest.csv", pd.DataFrame(rows))


def _artifact_role(relative: str) -> str:
    if "/figures/" in f"/{relative}" or relative.startswith("figures/"):
        return "figure"
    if "/metrics/" in f"/{relative}" or relative.startswith("metrics/"):
        return "metrics"
    if "/data/" in f"/{relative}" or relative.startswith("data/"):
        return "data"
    if "/meta/" in f"/{relative}" or relative.startswith("meta/"):
        return "metadata"
    if "/logs/" in f"/{relative}" or relative.startswith("logs/"):
        return "log"
    return "artifact"


def _write_artifact_manifest(root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
                "role": _artifact_role(relative),
            }
        )
    write_json(
        root / "artifact_manifest.json",
        {
            "builder_version": BUILDER_VERSION,
            "generated_at": _now(),
            "artifact_count": len(rows),
            "artifacts": rows,
        },
    )


def _refresh_top_level(*, repo_root: Path, output_root: Path) -> None:
    built = [figure_id for figure_id in FIGURE_IDS if (output_root / figure_id).is_dir()]
    panel_frames = [
        pd.read_csv(output_root / figure_id / "meta/panel_index.csv")
        for figure_id in built
    ]
    source_frames = [
        pd.read_csv(output_root / figure_id / "meta/source_manifest.csv")
        for figure_id in built
    ]
    cohort_frames = [
        pd.read_csv(output_root / figure_id / "meta/cohort_validation.csv")
        for figure_id in built
    ]
    schema_frames = [
        pd.read_csv(output_root / figure_id / "meta/schema_validation.csv")
        for figure_id in built
    ]
    panel_index = pd.concat(panel_frames, ignore_index=True, sort=False)
    source_manifest = pd.concat(source_frames, ignore_index=True, sort=False)
    cohort = pd.concat(cohort_frames, ignore_index=True, sort=False)
    schema = pd.concat(schema_frames, ignore_index=True, sort=False)
    write_csv(output_root / "panel_index.csv", panel_index)
    write_csv(output_root / "source_manifest.csv", source_manifest)
    write_csv(output_root / "meta/cohort_validation.csv", cohort)
    write_csv(output_root / "meta/schema_validation.csv", schema)
    parent_hashes = (
        source_manifest.loc[
            :,
            [
                "figure_id",
                "panel_id",
                "source_path",
                "source_sha256",
                "source_bytes",
            ],
        ]
        .drop_duplicates()
        .sort_values(["figure_id", "panel_id", "source_path"], kind="mergesort")
    )
    write_csv(output_root / "meta/parent_hashes_before.csv", parent_hashes)
    expected_quantitative = 29
    observed_quantitative = int(panel_index["panel_type"].eq("quantitative").sum())
    write_json(
        output_root / "run_config.json",
        {
            "task": FINAL_TASK_ID,
            "reuse_artifacts": "require",
            "builder_version": BUILDER_VERSION,
            "expected_seeds": list(range(1000, 1020)),
            "built_figures": built,
            "git_commit": _git_commit(repo_root),
            "updated_at": _now(),
            "model_or_dataset_initialized": False,
        },
    )
    write_json(
        output_root / "summary.json",
        {
            "status": (
                "statistics_ready"
                if built == list(FIGURE_IDS)
                and observed_quantitative == expected_quantitative
                else "statistics_partial"
            ),
            "built_figures": built,
            "panel_count": int(len(panel_index)),
            "quantitative_panel_count": observed_quantitative,
            "expected_quantitative_panel_count": expected_quantitative,
            "cohort_validation": (
                "pass" if cohort.empty or cohort["status"].eq("pass").all() else "fail"
            ),
            "schema_validation": (
                "pass" if schema.empty or schema["status"].eq("pass").all() else "fail"
            ),
            "parent_file_count": int(parent_hashes["source_path"].nunique()),
            "plot_status": "pending",
        },
    )
    with (output_root / "logs/statistics_build.log").open("a", encoding="utf-8") as handle:
        handle.write(
            f"{_now()} refreshed top-level statistics bundle; figures={','.join(built)}; "
            f"quantitative_panels={observed_quantitative}\n"
        )
    _write_artifact_manifest(output_root)


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the frozen final-six statistics bundle without model execution.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--figure",
        required=True,
        choices=[*FIGURE_IDS, "all"],
    )
    parser.add_argument("--reuse-artifacts", required=True, choices=["require"])
    parser.add_argument("--output-dir", default=str(DEFAULT_RELATIVE_OUTPUT))
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    repo_root = _repo_root()
    output_root = _resolve_output(repo_root, args.output_dir)
    selected = FIGURE_IDS if args.figure == "all" else (args.figure,)
    summaries: list[dict[str, Any]] = []
    for figure_id in selected:
        summaries.append(
            build_figure_statistics(
                figure_id,
                repo_root=repo_root,
                output_root=output_root,
                reuse_artifacts=args.reuse_artifacts,
                check_only=bool(args.check_only),
            )
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
