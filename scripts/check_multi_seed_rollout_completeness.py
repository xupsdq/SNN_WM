from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_ROOT = Path("results/multi_seed_rollout")
DEFAULT_SEEDS = tuple(range(1000, 1020))
FIGURE_ORDER = ("fig1", "fig2", "fig3", "fig4", "fig5", "fig6")


@dataclass(frozen=True)
class FigureConfig:
    experiment_dir: str
    upstream_tasks: tuple[str, ...]
    main_specs: tuple[str, ...]
    supp_specs: tuple[str, ...]


FIGURES: dict[str, FigureConfig] = {
    "fig1": FigureConfig(
        experiment_dir="fig1_functional_stsp_substrate",
        upstream_tasks=("trial_specs", "dms_boundary_bank"),
        main_specs=("fig1.yaml",),
        supp_specs=("fig1_supp.yaml", "fig1_supp_s2.yaml"),
    ),
    "fig2": FigureConfig(
        experiment_dir="fig2_pair_fused_stsp_state",
        upstream_tasks=("pair_trial_specs", "state_bank", "completion_delay_boundary_bank"),
        main_specs=("fig2.yaml",),
        supp_specs=("fig2_supp.yaml",),
    ),
    "fig3": FigureConfig(
        experiment_dir="fig3_multiitem_peak_landscape",
        upstream_tasks=("state_bank", "boundary_condition_specs"),
        main_specs=("fig3.yaml",),
        supp_specs=("fig3_delay_supp.yaml", "fig3_morphology_serial_supp.yaml", "fig3_supp.yaml"),
    ),
    "fig4": FigureConfig(
        experiment_dir="fig4_overlap_reentry",
        upstream_tasks=("pair_sampling", "rollouts"),
        main_specs=("fig4.yaml",),
        supp_specs=("fig4_supp.yaml",),
    ),
    "fig5": FigureConfig(
        experiment_dir="fig5_local_support_competition",
        upstream_tasks=("trial_sampling", "preprobe_support_bank"),
        main_specs=("fig5.yaml",),
        supp_specs=("fig5_supp.yaml",),
    ),
    "fig6": FigureConfig(
        experiment_dir="fig6_peak_amplified_reentry",
        upstream_tasks=("sequence_trials", "sequence_bank"),
        main_specs=("fig6.yaml",),
        supp_specs=("fig6_supp.yaml",),
    ),
}


SHARED_SEQUENCE_ROOT_TASKS = ("shared_sequence_specs", "shared_sequence_root_bank")
REQUIRED_BUNDLE_DIRS = ("data", "logs", "meta")
REQUIRED_BUNDLE_FILES = ("summary.json", "run_config.json", "artifact_manifest.json", "meta/run_info.json")
MANUAL_SOURCE_PREFIXES = ("manual_assets/",)
IGNORED_SOURCE_VALUES = {"", "none", "null"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _is_nonempty_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size <= 0:
        return False
    if path.suffix.lower() == ".csv":
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.reader(handle)
                rows = 0
                for row in reader:
                    if any(str(cell).strip() for cell in row):
                        rows += 1
                    if rows > 1:
                        return True
                return False
        except Exception:
            return False
    return True


def _is_source_relevant(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().replace("\\", "/")
    if text.lower() in IGNORED_SOURCE_VALUES:
        return False
    return not text.startswith(MANUAL_SOURCE_PREFIXES)


def _as_source_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip().replace("\\", "/") for item in value if _is_source_relevant(item)]
    if isinstance(value, str):
        return [value.strip().replace("\\", "/")] if _is_source_relevant(value) else []
    return []


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _load_spec_requirements(spec_path: Path) -> list[dict[str, Any]]:
    if not spec_path.is_file():
        return [{"kind": "spec", "spec": spec_path.name, "label": spec_path.name, "alternatives": [], "missing_spec": True}]
    with spec_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{spec_path} must contain a YAML mapping")

    requirements: list[dict[str, Any]] = []
    for rel in _as_source_list(payload.get("required_outputs")):
        requirements.append(
            {
                "kind": "required_output",
                "spec": spec_path.name,
                "label": f"{spec_path.name}:required:{rel}",
                "alternatives": [rel],
                "producer_tasks": [],
            }
        )

    for node in _walk_dicts(payload.get("panels", {})):
        panel_id = str(node.get("panel_id") or node.get("panel") or "").strip()
        label = f"{spec_path.name}:{panel_id or 'panel'}"
        producer_tasks = _normalise_tasks(node.get("producer_task"))
        alternatives = _as_source_list(node.get("source_priority"))
        if not alternatives:
            alternatives = _as_source_list(node.get("source"))
        if not alternatives:
            continue
        requirements.append(
            {
                "kind": "panel_source",
                "spec": spec_path.name,
                "label": label,
                "alternatives": alternatives,
                "producer_tasks": producer_tasks,
            }
        )
    return requirements


def _normalise_tasks(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _check_source_group(seed_root: Path, requirement: dict[str, Any]) -> dict[str, Any]:
    alternatives = [str(item) for item in requirement.get("alternatives", [])]
    existing = [rel for rel in alternatives if _is_nonempty_file(seed_root / rel)]
    return {
        **requirement,
        "ok": bool(existing) and not bool(requirement.get("missing_spec")),
        "existing": existing,
        "missing": [rel for rel in alternatives if rel not in existing],
    }


def _artifact_task_ok(task_dir: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    if not task_dir.is_dir():
        return False, [str(task_dir)]
    cache_key = task_dir / "cache_key.json"
    if not _is_nonempty_file(cache_key):
        missing.append(str(cache_key))
    manifest_candidates = list(task_dir.glob("*manifest*.csv"))
    if not any(_is_nonempty_file(path) for path in manifest_candidates):
        missing.append(f"{task_dir}/*manifest*.csv")
    return not missing, missing


def _check_upstream(seed_root: Path, tasks: tuple[str, ...]) -> dict[str, Any]:
    artifact_root = seed_root / "data" / "intermediates"
    task_status: dict[str, Any] = {}
    for task in tasks:
        ok, missing = _artifact_task_ok(artifact_root / task)
        task_status[task] = {"ok": ok, "missing": missing}
    return {
        "ok": all(item["ok"] for item in task_status.values()),
        "artifact_root": str(artifact_root),
        "tasks": task_status,
    }


def _check_bundle(seed_root: Path) -> dict[str, Any]:
    missing_dirs = [name for name in REQUIRED_BUNDLE_DIRS if not (seed_root / name).is_dir()]
    missing_files = [name for name in REQUIRED_BUNDLE_FILES if not _is_nonempty_file(seed_root / name)]
    summary: dict[str, Any] = {}
    summary_error = ""
    if _is_nonempty_file(seed_root / "summary.json"):
        try:
            summary = _load_json(seed_root / "summary.json")
        except Exception as exc:
            summary_error = str(exc)
    else:
        summary_error = "summary.json missing or empty"
    run_info_status = ""
    run_info_path = seed_root / "meta" / "run_info.json"
    if _is_nonempty_file(run_info_path):
        try:
            run_info = _load_json(run_info_path)
            run_info_status = str(run_info.get("status", ""))
        except Exception:
            run_info_status = ""
    return {
        "ok": not missing_dirs and not missing_files and not summary_error and run_info_status in {"", "success"},
        "missing_dirs": missing_dirs,
        "missing_files": missing_files,
        "summary_error": summary_error,
        "run_info_status": run_info_status,
        "summary_missing_main": list(summary.get("missing_for_main_figure") or []),
        "summary_missing_supp": list(summary.get("missing_for_supplementary") or []),
        "completed_modules": sorted(str(key) for key, value in (summary.get("completed_modules") or {}).items() if value),
    }


def _check_spec_group(seed_root: Path, spec_names: tuple[str, ...], specs_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for spec_name in spec_names:
        for requirement in _load_spec_requirements(specs_dir / spec_name):
            checks.append(_check_source_group(seed_root, requirement))
    missing = [item for item in checks if not item["ok"]]
    producer_tasks = sorted({task for item in checks for task in item.get("producer_tasks", []) if task})
    return {
        "ok": not missing,
        "total_requirements": len(checks),
        "missing_count": len(missing),
        "missing": missing,
        "producer_tasks": producer_tasks,
    }


def _figure_seed_root(root: Path, fig: str, config: FigureConfig, seed: int) -> Path:
    return root / fig / config.experiment_dir / f"seed_{seed}"


def _check_figure(root: Path, fig: str, config: FigureConfig, seed: int, specs_dir: Path) -> dict[str, Any]:
    seed_root = _figure_seed_root(root, fig, config, seed)
    exists = seed_root.is_dir()
    if not exists:
        return {
            "figure": fig,
            "seed": seed,
            "seed_root": str(seed_root),
            "exists": False,
            "bundle_ok": False,
            "upstream_ok": False,
            "main_ok": False,
            "supp_ok": False,
            "complete_ok": False,
            "bundle": {"ok": False, "missing_dirs": [], "missing_files": [str(seed_root)]},
            "upstream": {"ok": False, "tasks": {}},
            "main": {"ok": False, "missing": []},
            "supp": {"ok": False, "missing": []},
        }

    bundle = _check_bundle(seed_root)
    upstream = _check_upstream(seed_root, config.upstream_tasks)
    main = _check_spec_group(seed_root, config.main_specs, specs_dir)
    supp = _check_spec_group(seed_root, config.supp_specs, specs_dir)
    return {
        "figure": fig,
        "seed": seed,
        "seed_root": str(seed_root),
        "exists": True,
        "bundle_ok": bool(bundle["ok"]),
        "upstream_ok": bool(upstream["ok"]),
        "main_ok": bool(main["ok"]),
        "supp_ok": bool(supp["ok"]),
        "complete_ok": bool(bundle["ok"] and upstream["ok"] and main["ok"] and supp["ok"]),
        "bundle": bundle,
        "upstream": upstream,
        "main": main,
        "supp": supp,
    }


def _check_shared_sequence_root(root: Path, seed: int) -> dict[str, Any]:
    seed_root = root / "shared_sequence_root" / f"seed_{seed}"
    if not seed_root.is_dir():
        return {
            "seed": seed,
            "seed_root": str(seed_root),
            "exists": False,
            "summary_ok": False,
            "upstream_ok": False,
            "complete_ok": False,
            "missing": [str(seed_root)],
        }
    summary_path = seed_root / "shared_sequence_root_summary.json"
    summary_ok = _is_nonempty_file(summary_path)
    upstream = _check_upstream(seed_root, SHARED_SEQUENCE_ROOT_TASKS)
    return {
        "seed": seed,
        "seed_root": str(seed_root),
        "exists": True,
        "summary_ok": summary_ok,
        "upstream_ok": bool(upstream["ok"]),
        "complete_ok": bool(summary_ok and upstream["ok"]),
        "upstream": upstream,
        "missing": [] if summary_ok else [str(summary_path)],
    }


def _discover_seeds(root: Path, figures: tuple[str, ...]) -> tuple[int, ...]:
    seeds: set[int] = set()
    for fig in figures:
        config = FIGURES.get(fig)
        if config is None:
            continue
        base = root / fig / config.experiment_dir
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and child.name.startswith("seed_"):
                try:
                    seeds.add(int(child.name.split("_", 1)[1]))
                except ValueError:
                    continue
    shared = root / "shared_sequence_root"
    if shared.is_dir():
        for child in shared.iterdir():
            if child.is_dir() and child.name.startswith("seed_"):
                try:
                    seeds.add(int(child.name.split("_", 1)[1]))
                except ValueError:
                    continue
    return tuple(sorted(seeds))


def build_report(root: Path, *, figures: tuple[str, ...], seeds: tuple[int, ...], specs_dir: Path) -> dict[str, Any]:
    figure_checks: list[dict[str, Any]] = []
    for fig in figures:
        config = FIGURES[fig]
        for seed in seeds:
            figure_checks.append(_check_figure(root, fig, config, seed, specs_dir))

    shared_checks = [_check_shared_sequence_root(root, seed) for seed in seeds]
    by_figure: dict[str, Any] = {}
    for fig in figures:
        rows = [row for row in figure_checks if row["figure"] == fig]
        by_figure[fig] = {
            "seeds": len(rows),
            "existing": sum(1 for row in rows if row["exists"]),
            "bundle_ok": sum(1 for row in rows if row["bundle_ok"]),
            "upstream_ok": sum(1 for row in rows if row["upstream_ok"]),
            "main_ok": sum(1 for row in rows if row["main_ok"]),
            "supp_ok": sum(1 for row in rows if row["supp_ok"]),
            "complete_ok": sum(1 for row in rows if row["complete_ok"]),
            "missing_seeds": [row["seed"] for row in rows if not row["complete_ok"]],
        }

    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "seeds": list(seeds),
        "figures": list(figures),
        "summary": {
            "figures": by_figure,
            "shared_sequence_root": {
                "seeds": len(shared_checks),
                "existing": sum(1 for row in shared_checks if row["exists"]),
                "complete_ok": sum(1 for row in shared_checks if row["complete_ok"]),
                "missing_seeds": [row["seed"] for row in shared_checks if not row["complete_ok"]],
            },
        },
        "figure_checks": figure_checks,
        "shared_sequence_root_checks": shared_checks,
    }


def _short_missing(row: dict[str, Any], *, limit: int = 8) -> str:
    items: list[str] = []
    if not row["exists"]:
        return "seed root missing"
    for task, status in row.get("upstream", {}).get("tasks", {}).items():
        if not status.get("ok"):
            items.append(f"upstream:{task}")
    for group_name in ("main", "supp"):
        for item in row.get(group_name, {}).get("missing", []):
            label = str(item.get("label", "source"))
            alternatives = item.get("alternatives") or []
            sample = alternatives[0] if alternatives else "missing spec"
            items.append(f"{group_name}:{label}:{sample}")
    for rel in row.get("bundle", {}).get("summary_missing_main", []):
        items.append(f"summary_main:{rel}")
    for rel in row.get("bundle", {}).get("summary_missing_supp", []):
        items.append(f"summary_supp:{rel}")
    if not items:
        return ""
    clipped = items[:limit]
    suffix = f"; +{len(items) - limit} more" if len(items) > limit else ""
    return "; ".join(clipped) + suffix


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "multi_seed_rollout_completeness.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    csv_path = output_dir / "multi_seed_rollout_completeness.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "figure",
            "seed",
            "exists",
            "bundle_ok",
            "upstream_ok",
            "main_ok",
            "supp_ok",
            "complete_ok",
            "missing_summary",
            "seed_root",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["figure_checks"]:
            writer.writerow(
                {
                    "figure": row["figure"],
                    "seed": row["seed"],
                    "exists": row["exists"],
                    "bundle_ok": row["bundle_ok"],
                    "upstream_ok": row["upstream_ok"],
                    "main_ok": row["main_ok"],
                    "supp_ok": row["supp_ok"],
                    "complete_ok": row["complete_ok"],
                    "missing_summary": _short_missing(row),
                    "seed_root": row["seed_root"],
                }
            )

    md_path = output_dir / "multi_seed_rollout_completeness.md"
    md_path.write_text(_markdown_report(report), encoding="utf-8")


def _markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Multi-Seed Rollout Completeness")
    lines.append("")
    lines.append(f"- Checked at UTC: `{report['checked_at_utc']}`")
    lines.append(f"- Root: `{report['root']}`")
    lines.append(f"- Seeds: `{', '.join(str(seed) for seed in report['seeds'])}`")
    lines.append("")
    lines.append("## Figure Summary")
    lines.append("")
    lines.append("| figure | seeds | existing | upstream ok | main ok | supplement ok | complete ok | missing seeds |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for fig, item in report["summary"]["figures"].items():
        missing = ", ".join(str(seed) for seed in item["missing_seeds"]) or "-"
        lines.append(
            f"| {fig} | {item['seeds']} | {item['existing']} | {item['upstream_ok']} | "
            f"{item['main_ok']} | {item['supp_ok']} | {item['complete_ok']} | {missing} |"
        )
    shared = report["summary"]["shared_sequence_root"]
    missing_shared = ", ".join(str(seed) for seed in shared["missing_seeds"]) or "-"
    lines.append(
        f"| shared_sequence_root | {shared['seeds']} | {shared['existing']} | {shared['complete_ok']} | n/a | n/a | {shared['complete_ok']} | {missing_shared} |"
    )
    lines.append("")
    lines.append("## Missing Details")
    lines.append("")
    for row in report["figure_checks"]:
        if row["complete_ok"]:
            continue
        lines.append(f"### {row['figure']} seed_{row['seed']}")
        if not row["exists"]:
            lines.append(f"- Missing seed root: `{row['seed_root']}`")
            lines.append("")
            continue
        missing = _short_missing(row, limit=30)
        lines.append(f"- Summary: {missing or 'no missing detail'}")
        lines.append(f"- Seed root: `{row['seed_root']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        range_sep = ".." if ".." in item else "-" if "-" in item and not item.startswith("-") else ""
        if range_sep:
            start_text, end_text = item.split(range_sep, 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(item))
    return tuple(sorted(dict.fromkeys(seeds)))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check paper-figure multi-seed rollout completeness.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--specs-dir", default="src/plotting/paper_fig/specs")
    parser.add_argument("--figures", default=",".join(FIGURE_ORDER), help="Comma-separated figures to check.")
    parser.add_argument("--seeds", default="", help="Comma-separated seeds or ranges. Defaults to 1000-1019.")
    parser.add_argument("--discover-seeds", action="store_true", help="Use seed directories present under --root instead of the fixed 20-seed target.")
    parser.add_argument("--output-dir", default="", help="Directory for JSON/CSV/Markdown reports. Defaults to --root.")
    parser.add_argument("--json-only", action="store_true", help="Print JSON to stdout and do not write report files.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    root = Path(args.root)
    specs_dir = Path(args.specs_dir)
    figures = tuple(fig.strip() for fig in str(args.figures).split(",") if fig.strip())
    unknown = [fig for fig in figures if fig not in FIGURES]
    if unknown:
        raise SystemExit(f"Unknown figures: {', '.join(unknown)}")
    if args.seeds:
        seeds = _parse_csv_ints(args.seeds)
    elif args.discover_seeds:
        seeds = _discover_seeds(root, figures)
    else:
        seeds = DEFAULT_SEEDS
    report = build_report(root, figures=figures, seeds=seeds, specs_dir=specs_dir)
    if args.json_only:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        output_dir = Path(args.output_dir) if args.output_dir else root
        write_outputs(report, output_dir)
        for fig, item in report["summary"]["figures"].items():
            print(
                f"{fig}: existing={item['existing']}/{item['seeds']} upstream={item['upstream_ok']}/{item['seeds']} "
                f"main={item['main_ok']}/{item['seeds']} supp={item['supp_ok']}/{item['seeds']} complete={item['complete_ok']}/{item['seeds']}"
            )
        shared = report["summary"]["shared_sequence_root"]
        print(f"shared_sequence_root: complete={shared['complete_ok']}/{shared['seeds']}")
        print(f"Wrote reports to {output_dir.resolve()}")
    any_incomplete = any(item["complete_ok"] != item["seeds"] for item in report["summary"]["figures"].values())
    any_incomplete = any_incomplete or report["summary"]["shared_sequence_root"]["complete_ok"] != report["summary"]["shared_sequence_root"]["seeds"]
    return 1 if any_incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
