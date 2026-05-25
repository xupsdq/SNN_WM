from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_EXPERIMENTS = REPO_ROOT / "src" / "experiments"
SRC_PLOTTING = REPO_ROOT / "src" / "plotting"

SKIP_DIR_NAMES = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
ROOT_EXPERIMENT_NON_EXPERIMENTS = {"__init__", "catalog"}
OVERSIZED_WARNING_LIMIT = 1200


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def iter_python_files(*roots: Path) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            rel_parts = path.relative_to(REPO_ROOT).parts
            if any(part in SKIP_DIR_NAMES for part in rel_parts):
                continue
            yield path


def root_experiment_modules() -> set[str]:
    modules: set[str] = set()
    for path in sorted(SRC_EXPERIMENTS.glob("*.py")):
        if path.stem in ROOT_EXPERIMENT_NON_EXPERIMENTS:
            continue
        modules.add(path.stem)
    return modules


def import_module_names(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    names: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                names.append((node.module, int(node.lineno)))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.append((str(alias.name), int(node.lineno)))
    return names


def imported_root_experiment(module_name: str, root_modules: set[str]) -> str | None:
    prefix = "src.experiments."
    if not module_name.startswith(prefix):
        return None
    rest = module_name[len(prefix) :]
    top = rest.split(".", 1)[0]
    return top if top in root_modules else None


def is_archive_file(path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(SRC_EXPERIMENTS.resolve())
    except ValueError:
        return False
    return bool(rel.parts and rel.parts[0] == "archive")


def is_paper_figure_file(path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(SRC_EXPERIMENTS.resolve())
    except ValueError:
        return False
    return bool(rel.parts and rel.parts[0] == "paper_figures")


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8-sig").splitlines())


def source_category(path: Path) -> str:
    rel = path.resolve().relative_to(REPO_ROOT.resolve())
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "src" and parts[1] == "experiments":
        if len(parts) == 3:
            return "experiments/root"
        return f"experiments/{parts[2]}"
    if len(parts) >= 3 and parts[0] == "src" and parts[1] == "plotting":
        return f"plotting/{parts[2]}"
    return "/".join(parts[:2])


def build_report(oversized_limit: int = OVERSIZED_WARNING_LIMIT) -> dict[str, Any]:
    root_modules = root_experiment_modules()
    files = list(iter_python_files(SRC_EXPERIMENTS, SRC_PLOTTING))
    failures: list[str] = []
    warnings: list[str] = []
    passes: list[str] = []
    cross_imports: list[dict[str, Any]] = []
    paper_figure_root_imports: list[dict[str, Any]] = []
    oversized_modules: list[dict[str, Any]] = []
    category_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "lines": 0})
    parse_failures: list[str] = []

    for path in files:
        rel = repo_relative(path)
        n_lines = line_count(path)
        category = source_category(path)
        category_counts[category]["files"] += 1
        category_counts[category]["lines"] += int(n_lines)
        if n_lines > int(oversized_limit) and ("paper_figures" in rel or "src/experiments/" in rel):
            oversized_modules.append({"path": rel, "lines": int(n_lines), "limit": int(oversized_limit)})
        try:
            imports = import_module_names(path)
        except SyntaxError as exc:
            parse_failures.append(f"{rel}: {exc}")
            continue
        for module_name, lineno in imports:
            imported_root = imported_root_experiment(module_name, root_modules)
            if imported_root is None:
                continue
            item = {
                "path": rel,
                "line": int(lineno),
                "module": module_name,
                "root_experiment": imported_root,
            }
            if is_paper_figure_file(path):
                paper_figure_root_imports.append(item)
            elif not is_archive_file(path):
                cross_imports.append(item)

    for item in cross_imports:
        failures.append(
            f"active experiment imports root experiment module: {item['path']}:{item['line']} -> {item['module']}"
        )
    for item in paper_figure_root_imports:
        failures.append(
            f"paper_figures imports root experiment module instead of shared/common: {item['path']}:{item['line']} -> {item['module']}"
        )
    for item in parse_failures:
        failures.append(f"python parse failure: {item}")
    for item in oversized_modules:
        warnings.append(f"oversized module: {item['path']} has {item['lines']} lines (limit {item['limit']})")

    if not cross_imports:
        passes.append("no active non-archive experiment module imports another root experiment module")
    if not paper_figure_root_imports:
        passes.append("paper_figures do not import root experiment modules directly")
    if not parse_failures:
        passes.append("all scanned Python files parsed successfully")

    return {
        "ok": not failures,
        "counts": {
            "python_files": len(files),
            "root_experiment_modules": len(root_modules),
            "active_cross_root_imports": len(cross_imports),
            "paper_figure_root_imports": len(paper_figure_root_imports),
            "oversized_modules": len(oversized_modules),
            "parse_failures": len(parse_failures),
        },
        "categories": dict(sorted(category_counts.items())),
        "root_experiment_modules": sorted(root_modules),
        "active_cross_root_imports": cross_imports,
        "paper_figure_root_imports": paper_figure_root_imports,
        "oversized_modules": oversized_modules,
        "parse_failures": parse_failures,
        "passes": passes,
        "warnings": warnings,
        "failures": failures,
    }


def print_text_report(report: dict[str, Any]) -> None:
    print("Experiment structure audit")
    for key, value in report["counts"].items():
        print(f"{key}={value}")
    print("categories=")
    for category, payload in report["categories"].items():
        print(f"  {category}: files={payload['files']} lines={payload['lines']}")
    for message in report["passes"]:
        print(f"PASS: {message}")
    for message in report["warnings"]:
        print(f"WARN: {message}")
    for message in report["failures"]:
        print(f"FAIL: {message}")
    print(f"RESULT: {'PASS' if report['ok'] else 'FAIL'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit src experiment structure, shared boundaries, and cross-experiment imports.")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when structural failures are present.")
    parser.add_argument("--oversized-limit", type=int, default=OVERSIZED_WARNING_LIMIT)
    args = parser.parse_args()

    report = build_report(oversized_limit=int(args.oversized_limit))
    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_text_report(report)
    return 1 if args.strict and not report["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
