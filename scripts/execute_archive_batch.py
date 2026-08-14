from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

from repository_archive_audit import (
    DEFAULT_CUTOFF,
    collect_stats,
    iter_regular_files,
    sha256_file,
    tree_sha256,
)


SCHEMA_VERSION = "net_torch_archive_execution_v1"
APPROVED_PLAN_SHA256 = "cb2cb9ce2f3f57639053725149f073f36c92cf92abf9dbf94cd92dff1c48841c"
MAX_WINDOWS_DESTINATION_CHARS = 248
DEFAULT_PLAN = Path("archive/move_ledgers/archive_plan_20260814_approved.csv")
DEFAULT_RECEIPT = Path("archive/move_ledgers/archive_execution_20260814.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and execute the approved archive-only batch. "
            "This program has no deletion operation."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform verified archive moves. Without this flag, run preflight only.",
    )
    return parser.parse_args()


def inside_root(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_under_root(root: Path, value: str) -> Path:
    candidate = (root / Path(value)).resolve(strict=False)
    if not inside_root(root, candidate):
        raise RuntimeError(f"Path escapes repository root: {value}")
    return candidate


def read_plan(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["proposed_action"] == "archive"]
    if not selected:
        raise RuntimeError("Approved plan contains no archive rows")
    return selected


def allowed_mapping(source: str, destination: str) -> bool:
    exact = {
        "reviews": "archive/reviews_202606",
        "fig": "docs/archive/paper/legacy-assets_202605",
        "sandbox_document.xml": (
            "archive/work_history/document_extraction_202604/sandbox_document.xml"
        ),
    }
    if source in exact:
        return destination == exact[source]
    if source.startswith("tmp/fig3_"):
        return destination == (
            "archive/work_history/experiment_probes_202606/"
            + Path(source).name
        )
    return False


def assert_no_symlink_component(root: Path, path: Path) -> None:
    relative = path.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise RuntimeError(f"Symlink component is not allowed: {cursor}")


def expected_int(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"Invalid integer field {field!r} for {row.get('source_path')}") from exc


def windows_path_text(path: Path) -> str | None:
    parts = path.parts
    if len(parts) >= 4 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1:
        return parts[2].upper() + ":\\" + "\\".join(parts[3:])
    if os.name == "nt":
        return str(path)
    return None


def max_destination_windows_chars(source: Path, destination: Path) -> int | None:
    if source.is_file():
        targets = [destination]
    else:
        targets = [destination / item.relative_to(source) for item in iter_regular_files(source)]
    lengths = [len(text) for target in targets if (text := windows_path_text(target))]
    return max(lengths) if lengths else None


def preflight(root: Path, plan_path: Path) -> list[dict[str, Any]]:
    if sha256_file(plan_path) != APPROVED_PLAN_SHA256:
        raise RuntimeError("Plan hash does not match the explicitly approved dry-run")

    rows = read_plan(plan_path)
    cutoff_timestamp = datetime.combine(
        DEFAULT_CUTOFF,
        time.min,
        tzinfo=datetime.now().astimezone().tzinfo,
    ).timestamp()
    checked: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    destination_paths: list[Path] = []

    for row in rows:
        source_text = row["source_path"]
        destination_text = row["destination_path"]
        if row["approval_required"] != "yes_move_batch":
            raise RuntimeError(f"Archive row lacks move approval gate: {source_text}")
        if row["execution_status"] != "dry_run_only":
            raise RuntimeError(f"Unexpected source-plan status: {source_text}")
        if not allowed_mapping(source_text, destination_text):
            raise RuntimeError(
                f"Path is outside the approved first-batch mapping: {source_text}"
            )

        source = resolve_under_root(root, source_text)
        destination = resolve_under_root(root, destination_text)
        assert_no_symlink_component(root, source)
        assert_no_symlink_component(root, destination.parent)
        if not source.exists():
            raise RuntimeError(f"Archive source is missing: {source_text}")
        if source.is_symlink():
            raise RuntimeError(f"Archive source is a symlink: {source_text}")
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"Archive destination already exists: {destination_text}")

        stats = collect_stats(source, cutoff_timestamp)
        comparisons = {
            "file_count": stats.file_count,
            "total_bytes": stats.total_bytes,
            "old_file_count": stats.old_file_count,
            "old_bytes": stats.old_bytes,
            "symlink_count": stats.symlink_count,
            "scan_errors": stats.error_count,
        }
        for field, actual in comparisons.items():
            expected = expected_int(row, field)
            if actual != expected:
                raise RuntimeError(
                    f"Preflight mismatch for {source_text}: {field}={actual}, expected {expected}"
                )
        if stats.old_file_count != stats.file_count:
            raise RuntimeError(f"Not every file is strictly older than cutoff: {source_text}")
        actual_hash = tree_sha256(source)
        if actual_hash != row["tree_sha256"]:
            raise RuntimeError(f"Content hash changed after approval: {source_text}")
        max_windows_chars = max_destination_windows_chars(source, destination)
        if (
            max_windows_chars is not None
            and max_windows_chars > MAX_WINDOWS_DESTINATION_CHARS
        ):
            raise RuntimeError(
                f"Destination exceeds conservative Windows path limit for {source_text}: "
                f"{max_windows_chars} > {MAX_WINDOWS_DESTINATION_CHARS}"
            )

        source_paths.append(source)
        destination_paths.append(destination)
        checked.append(
            {
                "source_path": source_text,
                "destination_path": destination_text,
                "restore_path": source_text,
                "approved_action": row["proposed_action"],
                "reason": row["reason"],
                "evidence": row["evidence"],
                "file_count": stats.file_count,
                "total_bytes": stats.total_bytes,
                "tree_sha256": actual_hash,
                "max_destination_windows_chars": max_windows_chars,
                "status": "preflight_passed",
            }
        )

    if len(set(source_paths)) != len(source_paths):
        raise RuntimeError("Approved plan contains duplicate sources")
    if len(set(destination_paths)) != len(destination_paths):
        raise RuntimeError("Approved plan contains duplicate destinations")
    for index, source in enumerate(source_paths):
        for other in source_paths[index + 1 :]:
            if inside_root(source, other) or inside_root(other, source):
                raise RuntimeError(f"Archive sources overlap: {source} and {other}")

    return checked


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def execute(root: Path, plan_path: Path, receipt_path: Path, entries: list[dict[str, Any]]) -> None:
    if receipt_path.exists():
        raise RuntimeError(f"Execution receipt already exists: {receipt_path}")

    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "approval_scope": "first_archive_batch_only_no_cache_deletion",
        "plan_path": plan_path.relative_to(root).as_posix(),
        "plan_sha256": sha256_file(plan_path),
        "cutoff": DEFAULT_CUTOFF.isoformat(),
        "status": "executing",
        "archive_rows": len(entries),
        "files_moved": 0,
        "bytes_moved": 0,
        "files_deleted": 0,
        "delete_actions_executed": 0,
        "entries": entries,
    }
    write_receipt(receipt_path, payload)

    moved: list[dict[str, Any]] = []
    try:
        for entry in entries:
            source = resolve_under_root(root, entry["source_path"])
            destination = resolve_under_root(root, entry["destination_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            moved.append(entry)
            if source.exists() or not destination.exists():
                raise RuntimeError(f"Move did not reach expected state: {entry['source_path']}")
            if tree_sha256(destination) != entry["tree_sha256"]:
                raise RuntimeError(f"Post-move hash mismatch: {entry['destination_path']}")
            entry["status"] = "moved_verified"
            entry["moved_at"] = datetime.now(timezone.utc).isoformat()
            payload["files_moved"] += entry["file_count"]
            payload["bytes_moved"] += entry["total_bytes"]
            write_receipt(receipt_path, payload)
    except Exception as original_error:
        rollback_errors: list[str] = []
        for entry in reversed(moved):
            source = resolve_under_root(root, entry["source_path"])
            destination = resolve_under_root(root, entry["destination_path"])
            try:
                source.parent.mkdir(parents=True, exist_ok=True)
                destination.rename(source)
                if tree_sha256(source) != entry["tree_sha256"]:
                    raise RuntimeError("rollback hash mismatch")
                entry["status"] = "rolled_back_verified"
            except Exception as rollback_error:  # pragma: no cover - emergency path
                entry["status"] = "rollback_failed"
                rollback_errors.append(f"{entry['source_path']}: {rollback_error}")
        payload["status"] = "failed_rolled_back" if not rollback_errors else "failed_partial"
        payload["error"] = str(original_error)
        payload["rollback_errors"] = rollback_errors
        payload["completed_at"] = datetime.now(timezone.utc).isoformat()
        payload["files_moved"] = 0 if not rollback_errors else payload["files_moved"]
        payload["bytes_moved"] = 0 if not rollback_errors else payload["bytes_moved"]
        write_receipt(receipt_path, payload)
        raise

    payload["status"] = "completed_verified"
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_receipt(receipt_path, payload)


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    receipt_path = args.receipt if args.receipt.is_absolute() else root / args.receipt
    plan_path = plan_path.resolve()
    receipt_path = receipt_path.resolve(strict=False)
    if not inside_root(root, plan_path) or not inside_root(root, receipt_path):
        raise RuntimeError("Plan and receipt must remain inside the repository")

    entries = preflight(root, plan_path)
    print(
        "preflight=pass "
        f"archive_rows={len(entries)} "
        f"files={sum(entry['file_count'] for entry in entries)} "
        f"bytes={sum(entry['total_bytes'] for entry in entries)} "
        "delete_actions=0"
    )
    if not args.execute:
        print("mode=preflight_only files_moved=0 files_deleted=0")
        return

    execute(root, plan_path, receipt_path, entries)
    print(
        "mode=executed_verified "
        f"archive_rows={len(entries)} "
        f"files_moved={sum(entry['file_count'] for entry in entries)} "
        "files_deleted=0"
    )
    print(f"receipt={receipt_path}")


if __name__ == "__main__":
    main()
