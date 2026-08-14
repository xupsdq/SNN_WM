from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "net_torch_cleanup_execution_v1"
EXPECTED_PLAN_SHA256 = "09ab8b03b25cd7eee28d4aa808556ceebef8c715196d3cfc0730dd10bd02236f"
EXPECTED_BATCH_ID = "20260814_batch2_low_risk"
DEFAULT_PLAN = Path("archive/move_ledgers/cleanup_plan_20260814_batch2.json")
DEFAULT_RECEIPT = Path("archive/move_ledgers/cleanup_execution_20260814_batch2.json")
STAGING_RELATIVE = Path("archive/move_ledgers/.cleanup_batch2_staging")
EXPECTED_WHOLE_TARGETS = {
    "__pycache__",
    "tools",
    ".mplconfig",
    "tmp/pdf_deps",
    "tmp/second_batch_validation",
}
EXPECTED_EMPTY_TARGETS = {"--help", ".agents", ".codex-tmp", "output"}
EXPECTED_SELECTIVE_ROOT = ".pytest_tmp"
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


def inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def repo_path(root: Path, relative: str) -> Path:
    candidate = (root / Path(relative)).resolve(strict=False)
    if not inside(root, candidate):
        raise RuntimeError(f"Path escapes repository: {relative}")
    return candidate


def regular_files(root: Path) -> Iterable[Path]:
    if root.is_file() and not root.is_symlink():
        yield root
        return
    if not root.is_dir() or root.is_symlink():
        return
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        kept: list[str] = []
        for name in dirnames:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise RuntimeError(f"Symlink appeared in cleanup target: {candidate}")
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            candidate = directory_path / name
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise RuntimeError(f"Symlink appeared in cleanup target: {candidate}")
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(f"Non-regular cleanup entry: {candidate}")
            yield candidate


def file_paths(records: list[dict[str, Any]]) -> set[str]:
    return {str(item["path"]) for item in records}


def verify_file(root: Path, record: dict[str, Any]) -> None:
    relative = str(record["path"])
    if relative.startswith(FORBIDDEN_PREFIXES):
        raise RuntimeError(f"Forbidden path entered cleanup execution: {relative}")
    path = repo_path(root, relative)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Planned file is missing or not regular: {relative}")
    info = path.stat()
    if int(info.st_size) != int(record["bytes"]):
        raise RuntimeError(f"Size changed after cleanup approval: {relative}")
    if int(info.st_mtime_ns) != int(record["mtime_ns"]):
        raise RuntimeError(f"mtime changed after cleanup approval: {relative}")
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Hash changed after cleanup approval: {relative}")


def current_relative_files(root: Path, target: Path) -> set[str]:
    return {item.relative_to(root).as_posix() for item in regular_files(target)}


def current_directories(root: Path, target: Path) -> set[str]:
    directories: set[str] = set()
    for directory, dirnames, filenames in os.walk(target, followlinks=False):
        directory_path = Path(directory)
        if filenames:
            raise RuntimeError(f"Empty-tree target now contains files: {target}")
        for name in dirnames:
            if (directory_path / name).is_symlink():
                raise RuntimeError(f"Empty-tree target now contains symlink: {target}")
        directories.add(directory_path.relative_to(root).as_posix())
    return directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the approved batch-2 low-risk cleanup through an atomic staging area."
        )
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Delete the staged, verified low-risk targets. Default is preflight only.",
    )
    return parser.parse_args()


def preflight(root: Path, plan_path: Path, receipt_path: Path) -> dict[str, Any]:
    if sha256_file(plan_path) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("Cleanup plan hash does not match the approved batch")
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    if plan.get("schema") != "net_torch_cleanup_plan_v1":
        raise RuntimeError("Unexpected cleanup plan schema")
    if plan.get("batch_id") != EXPECTED_BATCH_ID:
        raise RuntimeError("Unexpected cleanup batch ID")
    if plan.get("execution_status") != "approved_not_executed":
        raise RuntimeError("Cleanup plan is not in approved-not-executed state")
    if tuple(plan.get("forbidden_prefixes", ())) != FORBIDDEN_PREFIXES:
        raise RuntimeError("Cleanup plan forbidden-prefix contract changed")
    if receipt_path.exists():
        raise RuntimeError(f"Cleanup receipt already exists: {receipt_path}")

    whole = plan["whole_tree_targets"]
    selective = plan["selective_targets"]
    empty = plan["empty_tree_targets"]
    if {item["path"] for item in whole} != EXPECTED_WHOLE_TARGETS:
        raise RuntimeError("Whole-tree cleanup target set changed")
    if len(selective) != 1 or selective[0]["path"] != EXPECTED_SELECTIVE_ROOT:
        raise RuntimeError("Selective cleanup root changed")
    if {item["path"] for item in empty} != EXPECTED_EMPTY_TARGETS:
        raise RuntimeError("Empty-tree cleanup target set changed")

    for item in whole:
        target = repo_path(root, item["path"])
        planned = file_paths(item["files"])
        if current_relative_files(root, target) != planned:
            raise RuntimeError(f"Whole-tree membership changed: {item['path']}")
        for record in item["files"]:
            verify_file(root, record)
        if len(planned) != int(item["file_count"]):
            raise RuntimeError(f"Whole-tree file count mismatch: {item['path']}")
        if sum(int(record["bytes"]) for record in item["files"]) != int(
            item["total_bytes"]
        ):
            raise RuntimeError(f"Whole-tree byte count mismatch: {item['path']}")

    selective_item = selective[0]
    selected = selective_item["selected_files"]
    protected = selective_item["protected_files"]
    selected_paths = file_paths(selected)
    protected_paths = file_paths(protected)
    if selected_paths & protected_paths:
        raise RuntimeError("Selective selected/protected sets overlap")
    actual_selective = current_relative_files(
        root, repo_path(root, selective_item["path"])
    )
    if actual_selective != selected_paths | protected_paths:
        raise RuntimeError("Selective cleanup membership changed")
    for record in [*selected, *protected]:
        verify_file(root, record)
    if len(selected) != int(selective_item["selected_file_count"]):
        raise RuntimeError("Selective cleanup file count mismatch")
    if sum(int(record["bytes"]) for record in selected) != int(
        selective_item["selected_bytes"]
    ):
        raise RuntimeError("Selective cleanup byte count mismatch")

    for item in empty:
        target = repo_path(root, item["path"])
        if not target.is_dir() or target.is_symlink():
            raise RuntimeError(f"Empty-tree target is missing: {item['path']}")
        if current_directories(root, target) != set(item["directories"]):
            raise RuntimeError(f"Empty-tree membership changed: {item['path']}")

    selected_records = [
        record for item in whole for record in item["files"]
    ] + selected
    selected_record_paths = file_paths(selected_records)
    if any(path.startswith(FORBIDDEN_PREFIXES) for path in selected_record_paths):
        raise RuntimeError("Forbidden path entered selected cleanup records")
    if len(selected_record_paths) != len(selected_records):
        raise RuntimeError("Cleanup plan contains duplicate selected files")
    totals = plan["totals"]
    if len(selected_records) != int(totals["files_selected"]):
        raise RuntimeError("Cleanup plan total file count mismatch")
    if sum(int(item["bytes"]) for item in selected_records) != int(
        totals["bytes_selected"]
    ):
        raise RuntimeError("Cleanup plan total byte count mismatch")
    if len(protected) != int(totals["protected_files"]):
        raise RuntimeError("Cleanup plan protected file count mismatch")

    staging = repo_path(root, STAGING_RELATIVE.as_posix())
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(f"Cleanup staging path already exists: {staging}")
    return plan


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def prune_empty_directories(root: Path) -> int:
    removed = 0
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            continue
        removed += 1
    return removed


def execute(
    root: Path,
    plan_path: Path,
    receipt_path: Path,
    plan: dict[str, Any],
) -> None:
    staging = repo_path(root, STAGING_RELATIVE.as_posix())
    whole = plan["whole_tree_targets"]
    selective = plan["selective_targets"][0]
    empty = plan["empty_tree_targets"]
    protected = selective["protected_files"]
    moved_whole: list[tuple[Path, Path]] = []
    moved_selective: list[tuple[Path, Path]] = []
    moved_empty: list[tuple[Path, Path]] = []
    deletion_started = False

    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "batch_id": EXPECTED_BATCH_ID,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "status": "staging",
        "approval_scope": plan["approval_scope"],
        "plan_path": plan_path.relative_to(root).as_posix(),
        "plan_sha256": sha256_file(plan_path),
        "files_deleted": 0,
        "bytes_deleted": 0,
        "empty_roots_deleted": 0,
        "protected_files_preserved": 0,
        "forbidden_targets_deleted": 0,
        "whole_tree_targets": [item["path"] for item in whole],
        "selective_root": selective["path"],
        "empty_tree_targets": [item["path"] for item in empty],
        "excluded_targets": list(FORBIDDEN_PREFIXES),
    }
    write_receipt(receipt_path, receipt)

    try:
        for item in whole:
            source = repo_path(root, item["path"])
            destination = staging / "whole" / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            moved_whole.append((source, destination))

        for record in selective["selected_files"]:
            source = repo_path(root, record["path"])
            destination = staging / "selective" / record["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            moved_selective.append((source, destination))

        for item in empty:
            source = repo_path(root, item["path"])
            destination = staging / "empty" / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            moved_empty.append((source, destination))

        staged_records = [
            record for item in whole for record in item["files"]
        ] + selective["selected_files"]
        for record in staged_records:
            staged = staging / (
                "selective"
                if record["path"].startswith(EXPECTED_SELECTIVE_ROOT + "/")
                else "whole"
            ) / record["path"]
            if not staged.is_file() or staged.is_symlink():
                raise RuntimeError(f"Staged file is missing: {record['path']}")
            if staged.stat().st_size != int(record["bytes"]):
                raise RuntimeError(f"Staged file size mismatch: {record['path']}")
            if sha256_file(staged) != record["sha256"]:
                raise RuntimeError(f"Staged file hash mismatch: {record['path']}")
        for record in protected:
            verify_file(root, record)

        pruned = prune_empty_directories(repo_path(root, selective["path"]))
        receipt.update(
            {
                "status": "staged_verified",
                "staged_files": len(staged_records),
                "staged_bytes": sum(int(item["bytes"]) for item in staged_records),
                "pytest_empty_directories_pruned": pruned,
                "protected_files_preserved": len(protected),
            }
        )
        write_receipt(receipt_path, receipt)

        deletion_started = True
        shutil.rmtree(staging)
        if staging.exists():
            raise RuntimeError("Cleanup staging directory still exists after deletion")

        for source, _ in [*moved_whole, *moved_selective, *moved_empty]:
            if source.exists() or source.is_symlink():
                raise RuntimeError(f"Deleted source unexpectedly remains: {source}")
        for record in protected:
            verify_file(root, record)

        receipt.update(
            {
                "status": "completed_verified",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "files_deleted": int(plan["totals"]["files_selected"]),
                "bytes_deleted": int(plan["totals"]["bytes_selected"]),
                "empty_roots_deleted": len(empty),
                "protected_files_preserved": len(protected),
                "forbidden_targets_deleted": 0,
            }
        )
        write_receipt(receipt_path, receipt)
    except Exception as original_error:
        rollback_errors: list[str] = []
        if not deletion_started:
            for source, destination in reversed(moved_empty):
                try:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.rename(source)
                except Exception as error:  # pragma: no cover - emergency path
                    rollback_errors.append(f"{source}: {error}")
            for source, destination in reversed(moved_selective):
                try:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.rename(source)
                except Exception as error:  # pragma: no cover - emergency path
                    rollback_errors.append(f"{source}: {error}")
            for source, destination in reversed(moved_whole):
                try:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.rename(source)
                except Exception as error:  # pragma: no cover - emergency path
                    rollback_errors.append(f"{source}: {error}")
            if staging.exists():
                try:
                    shutil.rmtree(staging)
                except Exception as error:  # pragma: no cover - emergency path
                    rollback_errors.append(f"{staging}: {error}")
        receipt.update(
            {
                "status": (
                    "failed_rolled_back"
                    if not deletion_started and not rollback_errors
                    else "failed_after_delete_started"
                    if deletion_started
                    else "failed_partial_rollback"
                ),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(original_error),
                "rollback_errors": rollback_errors,
            }
        )
        write_receipt(receipt_path, receipt)
        raise


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    receipt_path = args.receipt if args.receipt.is_absolute() else root / args.receipt
    plan_path = plan_path.resolve()
    receipt_path = receipt_path.resolve(strict=False)
    if not inside(root, plan_path) or not inside(root, receipt_path):
        raise RuntimeError("Plan and receipt must remain inside the repository")

    plan = preflight(root, plan_path, receipt_path)
    totals = plan["totals"]
    print(
        "preflight=pass "
        f"files={totals['files_selected']} "
        f"bytes={totals['bytes_selected']} "
        f"protected_files={totals['protected_files']} "
        "forbidden_targets=0"
    )
    if not args.execute:
        print("mode=preflight_only files_deleted=0")
        return
    execute(root, plan_path, receipt_path, plan)
    print(
        "mode=executed_verified "
        f"files_deleted={totals['files_selected']} "
        f"bytes_deleted={totals['bytes_selected']} "
        f"protected_files={totals['protected_files']} "
        "forbidden_targets_deleted=0"
    )
    print(f"receipt={receipt_path}")


if __name__ == "__main__":
    main()
