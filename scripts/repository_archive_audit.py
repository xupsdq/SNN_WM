from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile


SCHEMA_VERSION = "net_torch_archive_audit_v1"
DEFAULT_CUTOFF = date(2026, 7, 14)


@dataclass(frozen=True)
class PathStats:
    file_count: int
    total_bytes: int
    old_file_count: int
    old_bytes: int
    oldest_mtime: str
    latest_mtime: str
    symlink_count: int
    error_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_mtime(timestamp: float | None) -> str:
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def iter_regular_files(path: Path) -> Iterable[Path]:
    if path.is_symlink():
        return
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    for directory, dirnames, filenames in os.walk(path, followlinks=False):
        directory_path = Path(directory)
        dirnames[:] = [
            name for name in dirnames if not (directory_path / name).is_symlink()
        ]
        for filename in filenames:
            candidate = directory_path / filename
            try:
                mode = candidate.lstat().st_mode
            except OSError:
                continue
            if stat.S_ISREG(mode):
                yield candidate


def collect_stats(path: Path, cutoff_timestamp: float) -> PathStats:
    file_count = total_bytes = old_file_count = old_bytes = 0
    symlink_count = error_count = 0
    oldest: float | None = None
    latest: float | None = None

    if path.is_symlink():
        symlink_count = 1
    elif path.is_file():
        candidates: Iterable[Path] = (path,)
        for candidate in candidates:
            try:
                info = candidate.stat()
            except OSError:
                error_count += 1
                continue
            file_count += 1
            total_bytes += info.st_size
            oldest = info.st_mtime
            latest = info.st_mtime
            if info.st_mtime < cutoff_timestamp:
                old_file_count += 1
                old_bytes += info.st_size
    elif path.is_dir():
        for directory, dirnames, filenames in os.walk(path, followlinks=False):
            directory_path = Path(directory)
            kept_dirnames: list[str] = []
            for name in dirnames:
                candidate = directory_path / name
                try:
                    if candidate.is_symlink():
                        symlink_count += 1
                    else:
                        kept_dirnames.append(name)
                except OSError:
                    error_count += 1
            dirnames[:] = kept_dirnames
            for name in filenames:
                candidate = directory_path / name
                try:
                    info = candidate.lstat()
                except OSError:
                    error_count += 1
                    continue
                if stat.S_ISLNK(info.st_mode):
                    symlink_count += 1
                    continue
                if not stat.S_ISREG(info.st_mode):
                    continue
                file_count += 1
                total_bytes += info.st_size
                oldest = info.st_mtime if oldest is None else min(oldest, info.st_mtime)
                latest = info.st_mtime if latest is None else max(latest, info.st_mtime)
                if info.st_mtime < cutoff_timestamp:
                    old_file_count += 1
                    old_bytes += info.st_size

    return PathStats(
        file_count=file_count,
        total_bytes=total_bytes,
        old_file_count=old_file_count,
        old_bytes=old_bytes,
        oldest_mtime=iso_mtime(oldest),
        latest_mtime=iso_mtime(latest),
        symlink_count=symlink_count,
        error_count=error_count,
    )


def tree_sha256(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for candidate in sorted(iter_regular_files(path), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(path).as_posix()
        info = candidate.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(info.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(candidate).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def repository_relative(root: Path, value: str) -> Path:
    normalized = value.replace("\\", "/")
    for prefix in (
        "Y:/python_project/Net_torch/",
        "/mnt/y/python_project/Net_torch/",
    ):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return root / Path(normalized)


def collect_lineage_gaps(root: Path, authority: dict[str, Any]) -> dict[str, Any]:
    evidence = authority["evidence"]
    main_root = root / evidence["main_statistics_bundle"]["path"]
    supplementary_root = root / evidence["supplementary_s1_s6_bundle"]["path"]

    references: list[dict[str, Any]] = []
    for manifest in sorted(main_root.glob("fig*/meta/source_manifest.csv")):
        for row in read_csv_rows(manifest):
            references.append(
                {
                    "path": row["source_path"],
                    "sha256": row["source_sha256"],
                    "bytes": int(row["source_bytes"]),
                    "referenced_by": manifest.relative_to(root).as_posix(),
                }
            )

    supplementary_manifest = supplementary_root / "artifact_manifest.json"
    supplementary_data = json.loads(
        supplementary_manifest.read_text(encoding="utf-8-sig")
    )
    for row in supplementary_data.get("inputs", []):
        references.append(
            {
                "path": row["path"],
                "sha256": row["sha256"],
                "bytes": int(row["bytes"]),
                "referenced_by": supplementary_manifest.relative_to(root).as_posix(),
            }
        )

    grouped: dict[str, dict[str, Any]] = {}
    for row in references:
        absolute = repository_relative(root, row["path"])
        if absolute.is_file():
            continue
        normalized = absolute.relative_to(root).as_posix()
        existing = grouped.get(normalized)
        if existing is None:
            grouped[normalized] = {
                "path": normalized,
                "expected_sha256": row["sha256"],
                "expected_bytes": row["bytes"],
                "referenced_by": [row["referenced_by"]],
            }
            continue
        if (
            existing["expected_sha256"] != row["sha256"]
            or existing["expected_bytes"] != row["bytes"]
        ):
            raise RuntimeError(f"Conflicting missing-source identity: {normalized}")
        if row["referenced_by"] not in existing["referenced_by"]:
            existing["referenced_by"].append(row["referenced_by"])

    missing = sorted(grouped.values(), key=lambda row: row["path"])
    return {
        "schema": "paper_parent_artifact_gap_register_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "open",
        "scope": "Current main-statistics and Supplementary Fig. S1-S6 source manifests",
        "unique_missing_file_count": len(missing),
        "unique_missing_bytes": sum(row["expected_bytes"] for row in missing),
        "missing_filename_counts": {
            name: sum(1 for row in missing if Path(row["path"]).name == name)
            for name in sorted({Path(row["path"]).name for row in missing})
        },
        "impact": {
            "derived_bundles_present": True,
            "plot_only_replay_from_derived_bundles": "not invalidated by this register",
            "full_upstream_require_replay": "blocked until the missing parents are restored or the lineage contract is formally revised",
            "frozen_bundle_mutation_allowed": False,
        },
        "required_resolution": [
            "Restore byte-identical parents at their recorded paths and verify SHA-256, or",
            "approve a new versioned lineage contract that explicitly retires full-parent replay while preserving the frozen derived bundles.",
        ],
        "missing_files": missing,
    }


def verify_docx_media(root: Path, authority: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for document_key in ("main_baseline", "supplementary_baseline"):
        document = authority["manuscript"][document_key]
        docx_path = root / document["path"]
        with ZipFile(docx_path) as archive:
            for mapping in document["embedded_figure_mapping"]:
                media_bytes = archive.read(mapping["media_part"])
                media_hash = hashlib.sha256(media_bytes).hexdigest()
                source_path = root / mapping["source_path"]
                source_hash = sha256_file(source_path)
                expected_hash = mapping["sha256"]
                status = (
                    "pass"
                    if media_hash == source_hash == expected_hash
                    else "fail"
                )
                checks.append(
                    {
                        "document": document["path"],
                        "figure": mapping["figure"],
                        "media_part": mapping["media_part"],
                        "source_path": mapping["source_path"],
                        "expected_sha256": expected_hash,
                        "embedded_sha256": media_hash,
                        "source_sha256": source_hash,
                        "status": status,
                    }
                )
    failed = [row for row in checks if row["status"] != "pass"]
    if failed:
        raise RuntimeError(f"DOCX embedded-figure verification failed: {failed}")
    return checks


def verify_duplicate_tree(source: Path, canonical: Path) -> int:
    source_files = {
        path.relative_to(source).as_posix(): path for path in iter_regular_files(source)
    }
    canonical_files = {
        path.relative_to(canonical).as_posix(): path
        for path in iter_regular_files(canonical)
    }
    if not source_files or set(source_files) != set(canonical_files):
        raise RuntimeError(
            f"Duplicate-tree membership mismatch: {source} vs {canonical}"
        )
    for relative, source_path in source_files.items():
        canonical_path = canonical_files[relative]
        if (
            source_path.stat().st_size != canonical_path.stat().st_size
            or sha256_file(source_path) != sha256_file(canonical_path)
        ):
            raise RuntimeError(
                f"Duplicate-tree hash mismatch for {relative}: {source} vs {canonical}"
            )
    return len(source_files)


def archive_candidate_rows(
    root: Path, cutoff_timestamp: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    duplicate_mnist_files = verify_duplicate_tree(
        root / "data/MNIST/raw", root / "MNIST/raw"
    )

    def add(
        source: str,
        action: str,
        destination: str,
        reason: str,
        evidence: str,
        approval: str,
        age_scope: str = "whole_path",
        hash_tree: bool = False,
    ) -> None:
        path = root / source
        if not path.exists() and not path.is_symlink():
            return
        stats = collect_stats(path, cutoff_timestamp)
        if action == "archive" and stats.file_count == 0:
            return
        if age_scope == "files_before_cutoff" and stats.old_file_count == 0:
            return
        rows.append(
            {
                "source_path": source,
                "proposed_action": action,
                "destination_path": destination,
                "age_scope": age_scope,
                "file_count": stats.file_count,
                "total_bytes": stats.total_bytes,
                "old_file_count": stats.old_file_count,
                "old_bytes": stats.old_bytes,
                "oldest_file_mtime": stats.oldest_mtime,
                "latest_file_mtime": stats.latest_mtime,
                "symlink_count": stats.symlink_count,
                "scan_errors": stats.error_count,
                "tree_sha256": tree_sha256(path) if hash_tree and path.exists() else "",
                "reason": reason,
                "evidence": evidence,
                "approval_required": approval,
                "execution_status": "dry_run_only",
            }
        )

    add(
        "reviews",
        "archive",
        "archive/reviews_202606",
        "Completed pre-V6 review runs with no current code, manuscript, or manifest consumer.",
        "All 738 regular files are older than the cutoff; manual consumer scan on 2026-08-14 found no live reference.",
        "yes_move_batch",
        hash_tree=True,
    )
    add(
        "fig",
        "archive",
        "docs/archive/paper/legacy-assets_202605",
        "Legacy manuscript PDF and panel mapping workbook, superseded by docs/paper and current result manifests.",
        "Both files predate the cutoff and no live consumer was found.",
        "yes_move_batch",
        hash_tree=True,
    )
    add(
        "sandbox_document.xml",
        "archive",
        "archive/work_history/document_extraction_202604/sandbox_document.xml",
        "Unconsumed historical document-extraction artifact.",
        "Ignored by project policy, older than the cutoff, and absent from live consumer references.",
        "yes_move_batch",
        hash_tree=True,
    )

    for candidate in sorted(root.glob("tmp/fig3_*")):
        if not candidate.is_file() or candidate.stat().st_mtime >= cutoff_timestamp:
            continue
        relative = candidate.relative_to(root).as_posix()
        add(
            relative,
            "archive",
            f"archive/work_history/experiment_probes_202606/{candidate.name}",
            "Historical Fig.3 probe or run receipt retained only for provenance.",
            "File predates the cutoff and current runtime entrypoints no longer consume tmp/ probes.",
            "yes_move_batch",
            hash_tree=True,
        )

    cleanup_candidates = [
        (
            ".codex/tmp",
            "files_before_cutoff",
            "Agent scratch data; scientific provenance must be promoted elsewhere before cleanup.",
        ),
        (
            ".pytest_tmp",
            "files_before_cutoff",
            "Regenerable test outputs.",
        ),
        ("cache", "whole_path", "Regenerable runtime cache."),
        ("cache_data", "whole_path", "Unreferenced legacy tensor cache."),
        (
            "data/MNIST",
            "whole_path",
            f"All {duplicate_mnist_files} files are byte-identical duplicates of canonical MNIST/raw files (membership, size and SHA-256 verified).",
        ),
        ("__pycache__", "whole_path", "Regenerable Python bytecode."),
        ("tools", "whole_path", "Directory contains only regenerable bytecode."),
        (".mplconfig", "whole_path", "Regenerable Matplotlib cache."),
        ("tmp/pdf_deps", "whole_path", "Regenerable temporary PDF dependencies."),
    ]
    for source, age_scope, reason in cleanup_candidates:
        add(
            source,
            "delete_after_explicit_approval",
            "",
            reason,
            "Not a scientific archive object; retaining it under archive would add recoverable clutter.",
            "explicit_delete_approval",
            age_scope=age_scope,
        )

    for source in ("--help", ".agents", ".codex-tmp", "output"):
        path = root / source
        stats = collect_stats(path, cutoff_timestamp)
        if path.is_dir() and stats.file_count == 0 and stats.symlink_count == 0:
            add(
                source,
                "remove_empty_directory_after_approval",
                "",
                "Empty root-level directory tree.",
                "No regular files, symlinks, or consumers.",
                "explicit_delete_approval",
            )

    return rows


def keep_rows(authority: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(path: str, classification: str, reason: str, evidence: str) -> None:
        rows.append(
            {
                "path": path,
                "classification": classification,
                "reason": reason,
                "evidence": evidence,
                "status": "protected_no_move",
            }
        )

    add(
        authority["manuscript"]["main_baseline"]["path"],
        "paper_baseline",
        "Formal V6 baseline while V6.1 remains under author-confirmed construction.",
        "docs/paper/PAPER_AUTHORITY.json",
    )
    add(
        authority["manuscript"]["main_working"]["path"],
        "paper_working",
        "In-progress V6.1; never archive or hash-freeze during active editing.",
        "docs/paper/revisions/V6_1_CONFIRMATION_LOG_20260814.md",
    )
    for key in ("main_candidate_source", "supplementary_baseline", "supplementary_candidate_source"):
        entry = authority["manuscript"][key]
        add(entry["path"], entry["status"], entry["role"], "docs/paper/PAPER_AUTHORITY.json")

    for path in (
        "docs/paper/CORE_SCIENTIFIC_LOGIC_CONTRACT.md",
        "docs/paper/RESULTS_EVIDENCE_BOUNDARIES.md",
        "docs/paper/results_state_transition_program",
        "docs/paper/revisions",
        "docs/paper/submission_packages/communications_biology_20260801_final_six_results_candidate",
        "src",
        "tests",
        "scripts",
        "src/plotting/paper_fig/assets",
        "results/multi_snn",
        "results/multi_seed_rollout",
    ):
        add(
            path,
            "active_or_full_dag",
            "Current paper contract, runtime, dataset, checkpoint, figure asset, or full pre-submission DAG lineage.",
            "Project AGENTS.md plus current paper authority and source manifests.",
        )

    evidence = authority["evidence"]
    for key in (
        "redesign_bundle",
        "redesign_parent_bundle",
        "main_statistics_bundle",
        "supplementary_s1_s6_bundle",
        "supplementary_s7_bundle",
    ):
        entry = evidence[key]
        add(entry["path"], "current_evidence_bundle", entry["role"], "docs/paper/PAPER_AUTHORITY.json")

    for path in evidence["direct_parent_roots"]:
        add(
            path,
            "current_direct_parent",
            "At least one current source or artifact manifest points into this root.",
            "Current main/supplementary source-manifest closure.",
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_archive_execution_receipt(root: Path, path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8-sig"))
    if receipt.get("schema") != "net_torch_archive_execution_v1":
        raise RuntimeError(f"Unexpected archive execution schema: {path}")
    if receipt.get("status") != "completed_verified":
        raise RuntimeError(f"Archive execution is not complete: {path}")
    if receipt.get("files_deleted") != 0 or receipt.get("delete_actions_executed") != 0:
        raise RuntimeError(f"Archive-only receipt unexpectedly records deletion: {path}")

    files_moved = bytes_moved = 0
    for entry in receipt.get("entries", []):
        destination = root / entry["destination_path"]
        if not destination.exists():
            raise RuntimeError(f"Archived destination is missing: {destination}")
        if tree_sha256(destination) != entry["tree_sha256"]:
            raise RuntimeError(f"Archived destination hash mismatch: {destination}")
        if entry.get("status") != "moved_verified":
            raise RuntimeError(f"Archive receipt entry is not verified: {destination}")
        files_moved += int(entry["file_count"])
        bytes_moved += int(entry["total_bytes"])

    if files_moved != int(receipt["files_moved"]):
        raise RuntimeError("Archive receipt file count is internally inconsistent")
    if bytes_moved != int(receipt["bytes_moved"]):
        raise RuntimeError("Archive receipt byte count is internally inconsistent")
    return {
        "receipt": path.relative_to(root).as_posix(),
        "status": receipt["status"],
        "archive_rows": int(receipt["archive_rows"]),
        "files_moved": files_moved,
        "bytes_moved": bytes_moved,
        "files_deleted": 0,
        "completed_at": receipt["completed_at"],
    }


def verify_cleanup_execution_receipt(root: Path, path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8-sig"))
    if receipt.get("schema") != "net_torch_cleanup_execution_v1":
        raise RuntimeError(f"Unexpected cleanup execution schema: {path}")
    if receipt.get("status") != "completed_verified":
        raise RuntimeError(f"Cleanup execution is not complete: {path}")
    if receipt.get("forbidden_targets_deleted") != 0:
        raise RuntimeError(f"Cleanup receipt records forbidden deletion: {path}")

    plan_path = root / receipt["plan_path"]
    if sha256_file(plan_path) != receipt["plan_sha256"]:
        raise RuntimeError("Cleanup plan hash no longer matches its execution receipt")
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    totals = plan["totals"]
    if int(receipt["files_deleted"]) != int(totals["files_selected"]):
        raise RuntimeError("Cleanup receipt file count is inconsistent with its plan")
    if int(receipt["bytes_deleted"]) != int(totals["bytes_selected"]):
        raise RuntimeError("Cleanup receipt byte count is inconsistent with its plan")

    for item in [*plan["whole_tree_targets"], *plan["empty_tree_targets"]]:
        if (root / item["path"]).exists():
            raise RuntimeError(f"Completed cleanup target still exists: {item['path']}")
    selective = plan["selective_targets"][0]
    for record in selective["selected_files"]:
        if (root / record["path"]).exists():
            raise RuntimeError(f"Selected cleanup file still exists: {record['path']}")
    for record in selective["protected_files"]:
        protected = root / record["path"]
        if not protected.is_file() or sha256_file(protected) != record["sha256"]:
            raise RuntimeError(f"Protected cleanup-exclusion file changed: {record['path']}")

    return {
        "receipt": path.relative_to(root).as_posix(),
        "status": receipt["status"],
        "files_deleted": int(receipt["files_deleted"]),
        "bytes_deleted": int(receipt["bytes_deleted"]),
        "empty_roots_deleted": int(receipt["empty_roots_deleted"]),
        "protected_files_preserved": int(receipt["protected_files_preserved"]),
        "forbidden_targets_deleted": 0,
        "temporary_provenance_promotion": receipt.get(
            "temporary_provenance_promotion"
        ),
        "completed_at": receipt["completed_at"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a paper-first archive dry-run and lineage-gap register."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--cutoff",
        type=date.fromisoformat,
        default=DEFAULT_CUTOFF,
        help="Strict file-mtime cutoff (YYYY-MM-DD); default: 2026-07-14.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("archive/move_ledgers"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    cutoff_timestamp = datetime.combine(
        args.cutoff, time.min, tzinfo=datetime.now().astimezone().tzinfo
    ).timestamp()

    authority_path = root / "docs/paper/PAPER_AUTHORITY.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8-sig"))

    baseline = authority["manuscript"]["main_baseline"]
    if sha256_file(root / baseline["path"]) != baseline["sha256"]:
        raise RuntimeError("Formal V6 baseline hash no longer matches PAPER_AUTHORITY.json")
    supplementary = authority["manuscript"]["supplementary_baseline"]
    if sha256_file(root / supplementary["path"]) != supplementary["sha256"]:
        raise RuntimeError("Supplementary baseline hash no longer matches PAPER_AUTHORITY.json")
    for contract_name in ("core_scientific_logic", "results_evidence_boundaries"):
        contract = authority["contracts"][contract_name]
        if sha256_file(root / contract["path"]) != contract["sha256"]:
            raise RuntimeError(
                f"Contract hash no longer matches PAPER_AUTHORITY.json: {contract['path']}"
            )

    media_checks = verify_docx_media(root, authority)
    gap_register = collect_lineage_gaps(root, authority)
    gap_path = root / "docs/paper/PARENT_ARTIFACT_GAP_REGISTER_20260814.json"
    gap_path.write_text(
        json.dumps(gap_register, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    archive_rows = archive_candidate_rows(root, cutoff_timestamp)
    keep_manifest = keep_rows(authority)
    completed_archive = verify_archive_execution_receipt(
        root, output_dir / "archive_execution_20260814.json"
    )
    completed_cleanup = verify_cleanup_execution_receipt(
        root, output_dir / "cleanup_execution_20260814_batch2.json"
    )

    archive_fields = [
        "source_path",
        "proposed_action",
        "destination_path",
        "age_scope",
        "file_count",
        "total_bytes",
        "old_file_count",
        "old_bytes",
        "oldest_file_mtime",
        "latest_file_mtime",
        "symlink_count",
        "scan_errors",
        "tree_sha256",
        "reason",
        "evidence",
        "approval_required",
        "execution_status",
    ]
    keep_fields = ["path", "classification", "reason", "evidence", "status"]
    archive_plan_path = output_dir / "archive_plan_20260814.csv"
    keep_path = output_dir / "keep_manifest_20260814.csv"
    write_csv(archive_plan_path, archive_rows, archive_fields)
    write_csv(keep_path, keep_manifest, keep_fields)

    summary = {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "cutoff": args.cutoff.isoformat(),
        "mode": "dry_run_only",
        "files_moved": 0,
        "files_deleted": 0,
        "files_moved_this_audit": 0,
        "files_deleted_this_audit": 0,
        "deletion_approved": False,
        "remaining_deletion_approved": False,
        "completed_archive_execution": completed_archive,
        "completed_cleanup_execution": completed_cleanup,
        "authority_path": authority_path.relative_to(root).as_posix(),
        "lineage_gap_register": gap_path.relative_to(root).as_posix(),
        "unique_missing_parent_files": gap_register["unique_missing_file_count"],
        "unique_missing_parent_bytes": gap_register["unique_missing_bytes"],
        "docx_media_checks": media_checks,
        "docx_media_status": "pass",
        "archive_candidate_rows": sum(
            row["proposed_action"] == "archive" for row in archive_rows
        ),
        "archive_candidate_bytes": sum(
            row["total_bytes"]
            for row in archive_rows
            if row["proposed_action"] == "archive"
        ),
        "cleanup_candidate_old_bytes": sum(
            row["old_bytes"]
            for row in archive_rows
            if row["proposed_action"]
            in {
                "delete_after_explicit_approval",
                "remove_empty_directory_after_approval",
            }
        ),
        "archive_plan": archive_plan_path.relative_to(root).as_posix(),
        "keep_manifest": keep_path.relative_to(root).as_posix(),
        "protected_path_count": len(keep_manifest),
        "next_gate": (
            "Archive batch 1 and low-risk cleanup batch 2 are complete. Remaining high-risk "
            "cleanup candidates are not approved and require a new explicit gate."
            if completed_cleanup is not None
            else "First archive batch is complete. Remaining cleanup candidates require explicit approval."
            if completed_archive is not None
            else "Human review and explicit approval are required before any move or deletion."
        ),
    }
    summary_path = output_dir / "archive_audit_summary_20260814.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"authority={authority_path}")
    print(f"gap_register={gap_path}")
    print(f"archive_plan={archive_plan_path}")
    print(f"keep_manifest={keep_path}")
    print(f"summary={summary_path}")
    print(f"missing_parent_files={gap_register['unique_missing_file_count']}")
    print(f"docx_media_checks={len(media_checks)} pass")
    if completed_archive is not None:
        print(f"completed_archive_files_moved={completed_archive['files_moved']}")
        print(f"completed_archive_bytes_moved={completed_archive['bytes_moved']}")
    if completed_cleanup is not None:
        print(f"completed_cleanup_files_deleted={completed_cleanup['files_deleted']}")
        print(f"completed_cleanup_bytes_deleted={completed_cleanup['bytes_deleted']}")
    print("files_moved_this_audit=0")
    print("files_deleted=0")


if __name__ == "__main__":
    main()
