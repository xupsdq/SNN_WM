from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "net_torch_cleanup_plan_v1"
BATCH_ID = "20260814_batch2_low_risk"
DEFAULT_CUTOFF = date(2026, 7, 14)
DEFAULT_OUTPUT = Path("archive/move_ledgers/cleanup_plan_20260814_batch2.json")
WHOLE_TREE_TARGETS = {
    "__pycache__": "Regenerable root-level Python bytecode.",
    "tools": "Directory contains only regenerable Python bytecode.",
    ".mplconfig": "Regenerable Matplotlib font cache.",
    "tmp/pdf_deps": "Regenerable temporary pypdf dependency installation.",
    "tmp/second_batch_validation": (
        "Task-local replay output; promoted-runner validation is recorded in the "
        "temporary-provenance manifest."
    ),
}
EMPTY_TREE_TARGETS = ("--help", ".agents", ".codex-tmp", "output")
SELECTIVE_ROOT = ".pytest_tmp"
FORBIDDEN_PREFIXES = (
    ".codex/tmp",
    "cache",
    "cache_data",
    "data/MNIST",
    "docs/paper",
    ".pi/manuscript_review",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_files(root: Path) -> Iterable[Path]:
    if root.is_file() and not root.is_symlink():
        yield root
        return
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        kept: list[str] = []
        for name in dirnames:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise RuntimeError(f"Symlink is not allowed in cleanup target: {candidate}")
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            candidate = directory_path / name
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise RuntimeError(f"Symlink is not allowed in cleanup target: {candidate}")
            if stat.S_ISREG(info.st_mode):
                yield candidate
            else:
                raise RuntimeError(f"Non-regular cleanup entry: {candidate}")


def file_record(root: Path, path: Path) -> dict[str, Any]:
    info = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": int(info.st_size),
        "mtime": datetime.fromtimestamp(info.st_mtime).astimezone().isoformat(
            timespec="microseconds"
        ),
        "mtime_ns": int(info.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def tree_record(root: Path, relative: str, reason: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"Expected cleanup directory is unavailable: {relative}")
    files = [file_record(root, item) for item in sorted(regular_files(path))]
    if not files:
        raise RuntimeError(f"Whole-tree cleanup target has no files: {relative}")
    return {
        "path": relative,
        "scope": "whole_tree",
        "reason": reason,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def empty_tree_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"Expected empty-directory tree is unavailable: {relative}")
    directories: list[str] = []
    for directory, dirnames, filenames in os.walk(path, followlinks=False):
        directory_path = Path(directory)
        if filenames:
            raise RuntimeError(f"Empty-directory target contains files: {relative}")
        for name in dirnames:
            if (directory_path / name).is_symlink():
                raise RuntimeError(f"Empty-directory target contains symlink: {relative}")
        directories.append(directory_path.relative_to(root).as_posix())
    return {
        "path": relative,
        "scope": "empty_tree",
        "reason": "Directory tree contains no files or symlinks.",
        "directory_count": len(directories),
        "directories": sorted(directories),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the approved low-risk cleanup plan.")
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cutoff", type=date.fromisoformat, default=DEFAULT_CUTOFF)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    cutoff_timestamp = datetime.combine(
        args.cutoff, time.min, tzinfo=datetime.now().astimezone().tzinfo
    ).timestamp()

    whole_trees = [
        tree_record(root, path, reason)
        for path, reason in WHOLE_TREE_TARGETS.items()
    ]
    pytest_root = root / SELECTIVE_ROOT
    if not pytest_root.is_dir() or pytest_root.is_symlink():
        raise RuntimeError(f"Selective cleanup root is unavailable: {SELECTIVE_ROOT}")
    selected: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    for path in sorted(regular_files(pytest_root)):
        record = file_record(root, path)
        if path.stat().st_mtime < cutoff_timestamp:
            selected.append(record)
        else:
            protected.append(record)
    if not selected or not protected:
        raise RuntimeError("Selective pytest cleanup must contain old and protected-new files")

    empty_trees = [empty_tree_record(root, path) for path in EMPTY_TREE_TARGETS]
    selected_paths = {
        item["path"]
        for tree in whole_trees
        for item in tree["files"]
    } | {item["path"] for item in selected}
    for path in selected_paths:
        if path.startswith(FORBIDDEN_PREFIXES):
            raise RuntimeError(f"Forbidden path entered cleanup plan: {path}")

    payload = {
        "schema": SCHEMA_VERSION,
        "batch_id": BATCH_ID,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "approval_scope": (
            "temporary-provenance-promotion plus low-risk cleanup; exclude .codex/tmp, "
            "cache, cache_data, data/MNIST, active manuscripts and .pi/manuscript_review"
        ),
        "cutoff": args.cutoff.isoformat(),
        "cutoff_semantics": "strict file mtime < local midnight",
        "execution_status": "approved_not_executed",
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "whole_tree_targets": whole_trees,
        "selective_targets": [
            {
                "path": SELECTIVE_ROOT,
                "scope": "files_before_cutoff",
                "reason": "Regenerable pytest outputs; retain post-cutoff failure receipts.",
                "selected_file_count": len(selected),
                "selected_bytes": sum(item["bytes"] for item in selected),
                "selected_files": selected,
                "protected_file_count": len(protected),
                "protected_bytes": sum(item["bytes"] for item in protected),
                "protected_files": protected,
            }
        ],
        "empty_tree_targets": empty_trees,
        "totals": {
            "files_selected": sum(tree["file_count"] for tree in whole_trees)
            + len(selected),
            "bytes_selected": sum(tree["total_bytes"] for tree in whole_trees)
            + sum(item["bytes"] for item in selected),
            "empty_roots_selected": len(empty_trees),
            "protected_files": len(protected),
            "delete_actions_outside_plan": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"plan={output}")
    print(f"files_selected={payload['totals']['files_selected']}")
    print(f"bytes_selected={payload['totals']['bytes_selected']}")
    print(f"protected_files={payload['totals']['protected_files']}")
    print("forbidden_targets_selected=0")


if __name__ == "__main__":
    main()
