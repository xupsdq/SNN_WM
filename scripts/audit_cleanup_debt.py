from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCAN_ROOTS = (
    Path("src") / "plotting" / "paper_fig",
    Path("src") / "experiments" / "paper_figures",
)
DEFAULT_OVERSIZED_LIMIT = 2000
SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        absolute_root = REPO_ROOT / root
        if not absolute_root.exists():
            continue
        for path in sorted(absolute_root.rglob("*.py")):
            if any(part in SKIP_DIR_NAMES for part in path.relative_to(REPO_ROOT).parts):
                continue
            yield path


def function_definitions(path: Path) -> dict[str, list[int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    seen: dict[str, list[int]] = defaultdict(list)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seen[node.name].append(int(node.lineno))
        elif isinstance(node, ast.ClassDef):
            class_seen: dict[str, list[int]] = defaultdict(list)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_seen[f"{node.name}.{item.name}"].append(int(item.lineno))
            for name, lines in class_seen.items():
                seen[name].extend(lines)
    return dict(seen)


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def build_report(oversized_limit: int = DEFAULT_OVERSIZED_LIMIT) -> dict[str, Any]:
    files = list(iter_python_files(DEFAULT_SCAN_ROOTS))
    duplicate_functions: list[dict[str, Any]] = []
    oversized_modules: list[dict[str, Any]] = []
    parse_failures: list[str] = []
    for path in files:
        try:
            defs = function_definitions(path)
        except SyntaxError as exc:
            parse_failures.append(f"{repo_relative(path)}: {exc}")
            continue
        for name, lines in sorted(defs.items()):
            if len(lines) > 1:
                duplicate_functions.append({"path": repo_relative(path), "function": name, "lines": lines})
        n_lines = line_count(path)
        if n_lines > int(oversized_limit):
            oversized_modules.append({"path": repo_relative(path), "lines": n_lines, "limit": int(oversized_limit)})

    failures = [f"duplicate function definition: {item['path']}::{item['function']} at lines {item['lines']}" for item in duplicate_functions]
    failures.extend(f"python parse failure: {item}" for item in parse_failures)
    warnings = [f"oversized module: {item['path']} has {item['lines']} lines (limit {item['limit']})" for item in oversized_modules]
    passes: list[str] = []
    if not duplicate_functions and not parse_failures:
        passes.append("no duplicate function definitions found in paper_fig cleanup scan roots")
    return {
        "ok": not failures,
        "counts": {
            "python_files": len(files),
            "duplicate_functions": len(duplicate_functions),
            "oversized_modules": len(oversized_modules),
            "parse_failures": len(parse_failures),
        },
        "scan_roots": [root.as_posix() for root in DEFAULT_SCAN_ROOTS],
        "oversized_limit": int(oversized_limit),
        "duplicate_functions": duplicate_functions,
        "oversized_modules": oversized_modules,
        "parse_failures": parse_failures,
        "passes": passes,
        "warnings": warnings,
        "failures": failures,
    }


def print_text_report(report: dict[str, Any]) -> None:
    print("Cleanup debt audit")
    for key, value in report["counts"].items():
        print(f"{key}={value}")
    for message in report["passes"]:
        print(f"PASS: {message}")
    for message in report["warnings"]:
        print(f"WARN: {message}")
    for message in report["failures"]:
        print(f"FAIL: {message}")
    print(f"RESULT: {'PASS' if report['ok'] else 'FAIL'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit cleanup debt such as duplicate function definitions and oversized paper-figure modules.")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--strict", action="store_true", help="Return non-zero for duplicate functions or parse failures.")
    parser.add_argument("--oversized-limit", type=int, default=DEFAULT_OVERSIZED_LIMIT)
    args = parser.parse_args()

    report = build_report(oversized_limit=int(args.oversized_limit))
    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_text_report(report)
    return 1 if args.strict and not report["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
