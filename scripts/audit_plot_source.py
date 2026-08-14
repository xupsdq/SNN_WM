from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable, Sequence


FORBIDDEN_IMPORT_PREFIXES = (
    "src.experiments",
    "src.core",
    "src.data",
    "src.config",
    "torch",
)
FORBIDDEN_TEXT = (
    "results/multi_seed_rollout",
    "results\\multi_seed_rollout",
    "results/paper_figure_multi_seed/fig",
    "results\\paper_figure_multi_seed\\fig",
    ".npz",
    ".pth",
    ".pt\"",
    "load_model",
    "load_checkpoint",
    "load_mnist",
    "simulation",
    "run_rollout",
)


def _iter_python_files(targets: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        if target.is_file() and target.suffix == ".py":
            files.append(target)
        elif target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
        else:
            raise FileNotFoundError(f"plot source target does not exist: {target}")
    return sorted(set(path.resolve() for path in files))


class PlotAuditVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.failures: list[str] = []
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_import(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._check_import(str(node.module or ""), node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        name = self._call_name(node.func)
        if name.endswith("read_csv"):
            allowed_central_reader = (
                self.class_stack[-1:] == ["BundleReader"]
                and self.function_stack[-1:] == ["read_csv"]
                and name == "pd.read_csv"
            ) or name == "reader.read_csv"
            if not allowed_central_reader:
                self.failures.append(
                    f"{self.path}:{node.lineno}: CSV read bypasses BundleReader allowlist ({name})"
                )
        self.generic_visit(node)

    def _check_import(self, module: str, line: int) -> None:
        if any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in FORBIDDEN_IMPORT_PREFIXES
        ):
            self.failures.append(
                f"{self.path}:{line}: forbidden plotting import {module!r}"
            )

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = PlotAuditVisitor._call_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""


def audit(paths: Sequence[Path]) -> dict[str, object]:
    files = _iter_python_files(paths)
    failures: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for token in FORBIDDEN_TEXT:
            if token.lower() in lowered:
                failures.append(f"{path}: forbidden source token {token!r}")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path}:{exc.lineno}: syntax error: {exc.msg}")
            continue
        visitor = PlotAuditVisitor(path)
        visitor.visit(tree)
        failures.extend(visitor.failures)
    return {
        "schema": "plot_source_audit_v1",
        "files_scanned": len(files),
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit plot-only source for forbidden experiment and parent-data access."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional UTF-8 JSON report path.",
    )
    args = parser.parse_args(argv)
    report = audit(args.paths)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
