from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tarfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable


SCHEMA = "net_torch_parent_artifact_recovery_audit_v1"
DEFAULT_REGISTER = Path("docs/paper/PARENT_ARTIFACT_GAP_REGISTER_20260814.json")
DEFAULT_OUTPUT = Path("archive/move_ledgers/parent_artifact_recovery_audit_20260814.json")
TARGET_BASENAMES = {
    "panel_b_early_firing_transition_metrics.csv",
    "panel_d_l1_stsp_perturbation_unit_transitions.csv",
}
ZIP_SUFFIXES = {".zip"}
TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


def sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


def repo_relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def bucket_for(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0] if relative.parts else "."


def iter_regular_files(root: Path, errors: list[dict[str, str]]) -> Iterable[Path]:
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        kept: list[str] = []
        for name in dirnames:
            candidate = directory_path / name
            if candidate == root / ".git":
                continue
            try:
                if not candidate.is_symlink():
                    kept.append(name)
            except OSError as exc:
                errors.append({"path": repo_relative(root, candidate), "error": repr(exc)})
        dirnames[:] = kept
        for name in filenames:
            candidate = directory_path / name
            try:
                mode = candidate.lstat().st_mode
            except OSError as exc:
                errors.append({"path": repo_relative(root, candidate), "error": repr(exc)})
                continue
            if stat.S_ISREG(mode):
                yield candidate


def is_tar_path(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in TAR_SUFFIXES)


def candidate_record(
    *,
    location: str,
    size: int,
    sha256: str,
    expected_by_size: dict[int, list[dict[str, Any]]],
    expected_by_hash: dict[str, dict[str, Any]],
    source: str,
) -> dict[str, Any]:
    exact = expected_by_hash.get(sha256)
    return {
        "source": source,
        "location": location,
        "bytes": int(size),
        "sha256": sha256,
        "exact_match": exact is not None and int(exact["expected_bytes"]) == int(size),
        "matches_expected_path": exact["path"] if exact is not None and int(exact["expected_bytes"]) == int(size) else None,
        "same_size_expected_paths": [row["path"] for row in expected_by_size.get(int(size), [])],
    }


def inspect_zip(
    root: Path,
    path: Path,
    expected_by_size: dict[int, list[dict[str, Any]]],
    expected_by_hash: dict[str, dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    matches: list[dict[str, Any]] = []
    member_count = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member_count += 1
            basename = PurePosixPath(info.filename.replace("\\", "/")).name
            if basename not in TARGET_BASENAMES and int(info.file_size) not in expected_by_size:
                continue
            with archive.open(info, "r") as handle:
                digest = sha256_stream(handle)
            matches.append(
                candidate_record(
                    location=f"{repo_relative(root, path)}!/{info.filename}",
                    size=int(info.file_size),
                    sha256=digest,
                    expected_by_size=expected_by_size,
                    expected_by_hash=expected_by_hash,
                    source="zip_member",
                )
            )
    return member_count, matches


def inspect_tar(
    root: Path,
    path: Path,
    expected_by_size: dict[int, list[dict[str, Any]]],
    expected_by_hash: dict[str, dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    matches: list[dict[str, Any]] = []
    member_count = 0
    with tarfile.open(path, "r:*") as archive:
        for info in archive:
            if not info.isfile():
                continue
            member_count += 1
            basename = PurePosixPath(info.name.replace("\\", "/")).name
            if basename not in TARGET_BASENAMES and int(info.size) not in expected_by_size:
                continue
            handle = archive.extractfile(info)
            if handle is None:
                continue
            with handle:
                digest = sha256_stream(handle)
            matches.append(
                candidate_record(
                    location=f"{repo_relative(root, path)}!/{info.name}",
                    size=int(info.size),
                    sha256=digest,
                    expected_by_size=expected_by_size,
                    expected_by_hash=expected_by_hash,
                    source="tar_member",
                )
            )
    return member_count, matches


def inspect_git_objects(
    root: Path,
    expected_by_size: dict[int, list[dict[str, Any]]],
    expected_by_hash: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "blob_count": 0,
        "same_size_blob_count": 0,
        "same_name_history_entries": [],
        "candidates": [],
        "errors": [],
    }
    try:
        listing = subprocess.run(
            ["git", "cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        names = subprocess.run(
            ["git", "rev-list", "--objects", "--all"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        result["errors"].append(repr(exc))
        return result

    result["available"] = True
    object_paths: dict[str, list[str]] = defaultdict(list)
    for line in names.stdout.splitlines():
        object_id, separator, object_path = line.partition(" ")
        if separator:
            object_paths[object_id].append(object_path)
            if PurePosixPath(object_path.replace("\\", "/")).name in TARGET_BASENAMES:
                result["same_name_history_entries"].append({"object_id": object_id, "path": object_path})

    for line in listing.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[1] != "blob":
            continue
        object_id, _, size_text = parts
        result["blob_count"] += 1
        size = int(size_text)
        paths = object_paths.get(object_id, [])
        same_name = any(PurePosixPath(value.replace("\\", "/")).name in TARGET_BASENAMES for value in paths)
        if size not in expected_by_size and not same_name:
            continue
        result["same_size_blob_count"] += int(size in expected_by_size)
        try:
            payload = subprocess.run(
                ["git", "cat-file", "blob", object_id],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
            digest = hashlib.sha256(payload).hexdigest()
        except (OSError, subprocess.CalledProcessError) as exc:
            result["errors"].append(f"{object_id}: {exc!r}")
            continue
        record = candidate_record(
            location=f"git:{object_id}",
            size=size,
            sha256=digest,
            expected_by_size=expected_by_size,
            expected_by_hash=expected_by_hash,
            source="git_blob",
        )
        record["historical_paths"] = paths
        result["candidates"].append(record)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only search for missing Fig.5 parent artifacts.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--register", type=Path, default=DEFAULT_REGISTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    register_path = args.register if args.register.is_absolute() else root / args.register
    output_path = args.output if args.output.is_absolute() else root / args.output
    register = json.loads(register_path.read_text(encoding="utf-8-sig"))
    expected = list(register["missing_files"])
    expected_by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    expected_by_hash: dict[str, dict[str, Any]] = {}
    for row in expected:
        expected_by_size[int(row["expected_bytes"])].append(row)
        expected_by_hash[str(row["expected_sha256"])] = row

    scan_errors: list[dict[str, str]] = []
    candidates: list[dict[str, Any]] = []
    archives: list[Path] = []
    bucket_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"file_count": 0, "total_bytes": 0})
    file_count = 0
    total_bytes = 0
    hashed_file_count = 0

    for path in iter_regular_files(root, scan_errors):
        if path == output_path:
            continue
        try:
            size = int(path.stat().st_size)
        except OSError as exc:
            scan_errors.append({"path": repo_relative(root, path), "error": repr(exc)})
            continue
        file_count += 1
        total_bytes += size
        bucket = bucket_for(root, path)
        bucket_stats[bucket]["file_count"] += 1
        bucket_stats[bucket]["total_bytes"] += size
        lower = path.name.lower()
        if path.suffix.lower() in ZIP_SUFFIXES or is_tar_path(path):
            archives.append(path)
        if path.name not in TARGET_BASENAMES and size not in expected_by_size:
            continue
        try:
            digest = sha256_file(path)
            hashed_file_count += 1
        except OSError as exc:
            scan_errors.append({"path": repo_relative(root, path), "error": repr(exc)})
            continue
        candidates.append(
            candidate_record(
                location=repo_relative(root, path),
                size=size,
                sha256=digest,
                expected_by_size=expected_by_size,
                expected_by_hash=expected_by_hash,
                source="filesystem",
            )
        )

    archive_errors: list[dict[str, str]] = []
    archive_candidates: list[dict[str, Any]] = []
    archive_member_count = 0
    zip_count = tar_count = 0
    for path in archives:
        try:
            if path.suffix.lower() in ZIP_SUFFIXES:
                count, found = inspect_zip(root, path, expected_by_size, expected_by_hash)
                zip_count += 1
            elif is_tar_path(path):
                count, found = inspect_tar(root, path, expected_by_size, expected_by_hash)
                tar_count += 1
            else:
                continue
            archive_member_count += count
            archive_candidates.extend(found)
        except (OSError, tarfile.TarError, zipfile.BadZipFile, RuntimeError) as exc:
            archive_errors.append({"path": repo_relative(root, path), "error": repr(exc)})

    git_scan = inspect_git_objects(root, expected_by_size, expected_by_hash)
    all_candidates = candidates + archive_candidates + list(git_scan["candidates"])
    exact_matches = [row for row in all_candidates if row["exact_match"]]
    same_basename = [
        row
        for row in all_candidates
        if PurePosixPath(row["location"].split("!/", 1)[-1].replace("\\", "/")).name in TARGET_BASENAMES
    ]
    expected_status = []
    matched_paths = {str(row["matches_expected_path"]) for row in exact_matches}
    for row in expected:
        target = root / row["path"]
        expected_status.append(
            {
                **row,
                "target_exists_now": target.is_file(),
                "byte_identical_copy_found": row["path"] in matched_paths,
            }
        )

    payload = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only",
        "repo_root": str(root),
        "register": repo_relative(root, register_path),
        "register_sha256": sha256_file(register_path),
        "scope": {
            "filesystem": "all regular files under repository root except .git; symlink directories are not followed",
            "archives": "direct regular-file members of ZIP and TAR-family archives found in filesystem scope",
            "git": "all local Git blobs plus all-history path-name check",
            "external_backups": "not searched outside repository root",
        },
        "safety": {
            "files_moved": 0,
            "files_deleted": 0,
            "parent_artifacts_restored": 0,
            "protected_manuscripts_modified": 0,
        },
        "expected": {
            "file_count": len(expected),
            "total_bytes": sum(int(row["expected_bytes"]) for row in expected),
            "unique_sizes": len(expected_by_size),
            "unique_sha256": len(expected_by_hash),
            "status": expected_status,
        },
        "filesystem_scan": {
            "file_count": file_count,
            "total_bytes": total_bytes,
            "hashed_candidate_count": hashed_file_count,
            "bucket_stats": dict(sorted(bucket_stats.items())),
            "candidates": candidates,
            "errors": scan_errors,
        },
        "archive_scan": {
            "archive_count": len(archives),
            "zip_count": zip_count,
            "tar_count": tar_count,
            "member_count": archive_member_count,
            "candidates": archive_candidates,
            "errors": archive_errors,
        },
        "git_scan": git_scan,
        "result": {
            "byte_identical_copy_count": len(exact_matches),
            "byte_identical_matches": exact_matches,
            "same_basename_candidate_count": len(same_basename),
            "same_basename_candidates": same_basename,
            "resolved_expected_file_count": len(matched_paths),
            "unresolved_expected_file_count": len(expected) - len(matched_paths),
            "status": "byte_identical_copies_found" if exact_matches else "no_byte_identical_copy_found_in_repository_scope",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output={output_path}")
    print(f"files_scanned={file_count}")
    print(f"archives_scanned={len(archives)} members={archive_member_count}")
    print(f"byte_identical_copy_count={len(exact_matches)}")
    print(f"same_basename_candidate_count={len(same_basename)}")
    print(f"scan_errors={len(scan_errors) + len(archive_errors) + len(git_scan['errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
