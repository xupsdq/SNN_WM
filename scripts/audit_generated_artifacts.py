from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

GENERATED_EXTENSIONS = {
    ".bak",
    ".bin",
    ".ckpt",
    ".csv",
    ".feather",
    ".h5",
    ".jpeg",
    ".joblib",
    ".jpg",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".svg",
    ".tmp",
    ".tsv",
    ".xlsx",
}

GENERATED_ROOTS = {
    ".pytest_tmp",
    "artifacts",
    "cache",
    "cache_data",
    "logs",
    "mlruns",
    "not_use",
    "outputs",
    "results",
    "useful_fig_results",
}

SKIP_DIR_NAMES = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "env",
    "venv",
}

SOURCE_OUTPUT_DIR_NAMES = {
    "figure",
    "figures",
    "log",
    "logs",
    "metrics",
    "mlruns",
    "output",
    "outputs",
    "runs",
    "wandb",
}

SOURCE_ROOTS = ("src", "scripts", "tools", "configs")
MANUSCRIPT_ROOT = "fig"
ARCHIVE_ROOT = "archive"
MANUAL_ASSET_ROOTS = (
    Path("src") / "plotting" / "paper_fig" / "manual_assets",
)


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def iter_paths(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIR_NAMES for part in rel_parts):
            continue
        yield path


def is_under_any(path: Path, roots: Iterable[Path]) -> bool:
    return any(is_relative_to(path, root) for root in roots)


def is_git_ignored(path: Path) -> bool:
    rel = repo_relative(path)
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "--", rel],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def root_name(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return ""
    return rel.parts[0] if rel.parts else ""


def scan_source_output_dirs() -> list[str]:
    found: list[str] = []
    for root_name_item in SOURCE_ROOTS:
        root = REPO_ROOT / root_name_item
        if not root.exists():
            continue
        for path in iter_paths(root):
            if not path.is_dir():
                continue
            if path.name.lower() in SOURCE_OUTPUT_DIR_NAMES:
                found.append(repo_relative(path))
    return sorted(found)


def scan_generated_files() -> tuple[list[str], list[str], list[str]]:
    manual_roots = tuple(REPO_ROOT / item for item in MANUAL_ASSET_ROOTS)
    unignored: list[str] = []
    ignored: list[str] = []
    manuscript_inputs: list[str] = []
    for path in iter_paths(REPO_ROOT):
        if not path.is_file() or path.suffix.lower() not in GENERATED_EXTENSIONS:
            continue
        top = root_name(path)
        if top in GENERATED_ROOTS or top == ARCHIVE_ROOT:
            continue
        if top == MANUSCRIPT_ROOT:
            manuscript_inputs.append(repo_relative(path))
            continue
        if is_under_any(path, manual_roots):
            continue
        rel = repo_relative(path)
        if is_git_ignored(path):
            ignored.append(rel)
        else:
            unignored.append(rel)
    return sorted(unignored), sorted(ignored), sorted(manuscript_inputs)


def existing_generated_roots() -> list[str]:
    found: list[str] = []
    for name in sorted(GENERATED_ROOTS | {ARCHIVE_ROOT}):
        path = REPO_ROOT / name
        if path.exists():
            found.append(repo_relative(path))
    return found


def build_report() -> dict[str, Any]:
    source_output_dirs = scan_source_output_dirs()
    unignored_generated_files, ignored_generated_files, manuscript_inputs = scan_generated_files()
    failures: list[str] = []
    warnings: list[str] = []
    passes: list[str] = []

    for path in source_output_dirs:
        failures.append(f"active source contains default generated output directory: {path}")
    for path in unignored_generated_files:
        failures.append(f"generated-looking artifact outside allowed roots is not ignored/documented: {path}")
    for path in ignored_generated_files:
        warnings.append(f"generated-looking artifact outside allowed roots is covered by .gitignore: {path}")
    if manuscript_inputs:
        warnings.append("manuscript input materials under fig/: " + ", ".join(manuscript_inputs))

    if not source_output_dirs:
        passes.append("no default generated output directories found under active source roots")
    if not unignored_generated_files:
        passes.append("no unignored generated-looking files found outside allowed roots")

    return {
        "ok": not failures,
        "counts": {
            "source_output_dirs": len(source_output_dirs),
            "unignored_generated_files": len(unignored_generated_files),
            "ignored_generated_files": len(ignored_generated_files),
            "manuscript_inputs": len(manuscript_inputs),
            "generated_roots_present": len(existing_generated_roots()),
        },
        "generated_roots_present": existing_generated_roots(),
        "source_output_dirs": source_output_dirs,
        "unignored_generated_files": unignored_generated_files,
        "ignored_generated_files": ignored_generated_files,
        "manuscript_inputs": manuscript_inputs,
        "passes": passes,
        "warnings": warnings,
        "failures": failures,
    }


def print_text_report(report: dict[str, Any]) -> None:
    print("Generated artifact audit")
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
    parser = argparse.ArgumentParser(description="Audit generated artifacts and default output directories outside results/archive.")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when active-source boundary failures are present.")
    args = parser.parse_args()

    report = build_report()
    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_text_report(report)
    return 1 if args.strict and not report["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
