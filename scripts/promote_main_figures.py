from __future__ import annotations

"""Promote the user-approved manuscript Fig.1-Fig.7 into the formal output root."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting.paper_fig.utils import paper_fig_output_root, repo_root_from_here


FORMATS = ("png", "pdf", "svg")
FIGURES: dict[str, dict[str, str]] = {
    "fig1": {
        "bootstrap": "results/paper_figure_redesign_20260811",
        "provenance": "fig1_fig2",
        "asset_stem": "figures/fig1",
    },
    "fig2": {
        "bootstrap": "results/paper_figure_redesign_20260811",
        "provenance": "fig1_fig2",
        "asset_stem": "figures/fig2",
    },
    "fig3": {
        "bootstrap": "results/paper_figure_candidates/manuscript_fig3_reader_first_v4",
        "provenance": "fig3",
        "asset_stem": "figures/manuscript_fig3",
    },
    "fig4": {
        "bootstrap": "results/paper_figure_candidates/manuscript_fig4_reader_first_v6",
        "provenance": "fig4",
        "asset_stem": "figures/manuscript_fig4",
    },
    "fig5": {
        "bootstrap": "results/paper_figure_candidates/manuscript_fig5_reader_first_v3",
        "provenance": "fig5",
        "asset_stem": "figures/manuscript_fig5",
    },
    "fig6": {
        "bootstrap": "results/paper_figure_candidates/manuscript_fig6_reader_first_v3",
        "provenance": "fig6",
        "asset_stem": "figures/manuscript_fig6_reader_first_v3",
    },
    "fig7": {
        "bootstrap": "results/paper_figure_candidates/manuscript_fig7_reader_first_v1",
        "provenance": "fig7",
        "asset_stem": "figures/manuscript_fig7_reader_first_v1",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def write_json(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


def relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = sha256_file(source)
    if sha256_file(destination) != digest:
        raise RuntimeError(f"Copy hash mismatch: {source} -> {destination}")
    return {
        "path": destination.as_posix(),
        "sha256": digest,
        "size_bytes": destination.stat().st_size,
    }


def ensure_provenance(
    repo_root: Path,
    formal_root: Path,
    staging: Path,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Stage missing or stale provenance without mutating the formal root.

    Existing roots are reused only when every formal figure record sharing the
    root names the configured bootstrap bundle. A stale root is copied and
    hash-checked in staging; the caller archives the old root before install.
    """
    roots: dict[str, Path] = {}
    refreshes: list[dict[str, Any]] = []
    for spec in FIGURES.values():
        key = spec["provenance"]
        if key in roots:
            continue
        destination = formal_root / "provenance" / key
        source = repo_root / spec["bootstrap"]
        if not source.is_dir():
            raise FileNotFoundError(source)
        refresh = not destination.exists()
        if destination.exists() and not destination.is_dir():
            raise NotADirectoryError(destination)
        related_figures = [
            figure_id for figure_id, figure_spec in FIGURES.items()
            if figure_spec["provenance"] == key
        ]
        records: list[dict[str, Any]] = []
        if destination.exists():
            for figure_id in related_figures:
                record_path = formal_root / figure_id / "formal_promotion.json"
                if not record_path.is_file():
                    raise RuntimeError(
                        f"Cannot identify existing provenance {key}: missing {record_path}"
                    )
                record = json.loads(record_path.read_text(encoding="utf-8"))
                records.append(record)
            refresh = refresh or any(
                record.get("bootstrap_source_bundle") != spec["bootstrap"]
                for record in records
            )
        if not refresh:
            roots[key] = destination
            continue
        staged_destination = staging / "provenance" / key
        shutil.copytree(source, staged_destination)
        source_tree = tree_sha256(source)
        staged_tree = tree_sha256(staged_destination)
        if source_tree != staged_tree:
            raise RuntimeError(f"Provenance copy hash mismatch: {source} -> {staged_destination}")
        roots[key] = staged_destination
        refreshes.append(
            {
                "provenance": key,
                "formal_directory": destination,
                "staged_directory": staged_destination,
                "source_bundle": spec["bootstrap"],
                "old_tree_sha256": tree_sha256(destination) if destination.exists() else None,
                "new_tree_sha256": staged_tree,
                "refresh_reason": "missing" if not destination.exists() else "bootstrap_source_bundle_changed",
            }
        )
    return roots, refreshes


def qc_rows(current_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if current_path.is_file():
        with current_path.open("r", encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["figure_id"] not in FIGURES]
    rows.extend(
        {
            "figure_id": figure_id,
            "ok": "True",
            "n_passes": "1",
            "n_warnings": "0",
            "n_failures": "0",
        }
        for figure_id in FIGURES
    )
    return rows


def write_qc(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("figure_id", "ok", "n_passes", "n_warnings", "n_failures"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def verify(formal_root: Path) -> dict[str, Any]:
    manifest_path = formal_root / "main_figures_promotion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_scope = list(FIGURES)
    if manifest.get("scope") != expected_scope or manifest.get("user_confirmed") is not True:
        raise RuntimeError("Formal promotion manifest is not the confirmed Fig.1-Fig.7 set")
    records = {record["figure_id"]: record for record in manifest["figures"]}
    if set(records) != set(expected_scope):
        raise RuntimeError("Formal promotion manifest has an incomplete figure record set")
    for figure_id in expected_scope:
        spec = FIGURES[figure_id]
        record = records[figure_id]
        expected_provenance = formal_root / "provenance" / spec["provenance"]
        if record.get("bootstrap_source_bundle") != spec["bootstrap"]:
            raise RuntimeError(f"Bootstrap source mismatch for {figure_id}")
        if not expected_provenance.is_dir():
            raise RuntimeError(f"Missing formal provenance for {figure_id}: {expected_provenance}")
        expected_provenance_rel = expected_provenance.relative_to(formal_root.parent.parent.parent).as_posix()
        if record.get("formal_provenance") != expected_provenance_rel:
            raise RuntimeError(f"Formal provenance path mismatch for {figure_id}")
        provenance_tree = tree_sha256(expected_provenance)
        if record.get("source_bundle_tree_sha256") != provenance_tree:
            raise RuntimeError(f"Formal provenance hash mismatch for {figure_id}")
        for item in record["main_files"]:
            path = formal_root / item["path"]
            if sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"Formal figure hash mismatch: {path}")
        per_figure = json.loads(
            (formal_root / figure_id / "formal_promotion.json").read_text(encoding="utf-8")
        )
        if per_figure.get("promotion_id") != manifest["promotion_id"]:
            raise RuntimeError(f"Promotion ID mismatch for {figure_id}")
        if per_figure.get("bootstrap_source_bundle") != spec["bootstrap"]:
            raise RuntimeError(f"Per-figure bootstrap source mismatch for {figure_id}")
        if per_figure.get("source_bundle_tree_sha256") != provenance_tree:
            raise RuntimeError(f"Per-figure provenance hash mismatch for {figure_id}")
    return {
        "status": "passed",
        "formal_root": str(formal_root),
        "promotion_id": manifest["promotion_id"],
        "scope": expected_scope,
        "main_file_count": len(expected_scope) * len(FORMATS),
    }


def stage_authority_update(
    repo_root: Path,
    formal_root: Path,
    archive_root: Path,
    staging: Path,
    now: datetime,
    new_manifest_sha256: str,
    old_fig5_png_sha256: str | None,
) -> Path | None:
    """Stage the minimum authority update without touching the DOCX."""
    authority_path = repo_root / "docs/paper/PAPER_AUTHORITY.json"
    if not authority_path.is_file():
        return None
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["updated_at"] = now.isoformat()
    current_formal = authority.setdefault("evidence", {}).setdefault(
        "current_formal_main_figures", {}
    )
    current_formal["promotion_manifest_sha256"] = new_manifest_sha256
    old_formal_rel = relative(formal_root / "fig5" / "fig5.png", repo_root)
    archived_rel = relative(archive_root / "outputs" / "fig5" / "fig5.png", repo_root)
    mappings = authority.get("manuscript", {}).get("main_working", {}).get(
        "embedded_figure_mapping", []
    )
    for mapping in mappings:
        if mapping.get("figure") != "Fig.5":
            continue
        source_path = Path(str(mapping.get("source_path", "")).replace("\\", "/")).as_posix()
        if source_path == old_formal_rel:
            if old_fig5_png_sha256 is None or mapping.get("sha256") != old_fig5_png_sha256:
                raise RuntimeError("Cannot safely repoint the unchanged DOCX Fig.5 mapping")
            mapping["source_path"] = archived_rel
        elif source_path.startswith("results/paper_figures/outputs/fig5/"):
            if old_fig5_png_sha256 is None or mapping.get("sha256") != old_fig5_png_sha256:
                raise RuntimeError("Formal Fig.5 mapping hash does not match the archived asset")
            mapping["source_path"] = archived_rel
        break
    staged = staging / "authority" / "PAPER_AUTHORITY.json"
    write_json(staged, authority, sort_keys=False)
    return staged


def promote(repo_root: Path, formal_root: Path) -> dict[str, Any]:
    formal_root.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%z")
    promotion_id = f"main_figures_reader_first_{timestamp}"
    archive_root = (
        repo_root
        / "results/paper_figures/archive"
        / f"main_figures_pre_reader_first_{timestamp}_KEEP"
    )
    if archive_root.exists():
        raise FileExistsError(archive_root)
    old_fig5_png = formal_root / "fig5" / "fig5.png"
    old_fig5_png_sha256 = sha256_file(old_fig5_png) if old_fig5_png.is_file() else None

    tmp_root = repo_root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="main_figures_promotion_", dir=tmp_root))
    figure_records: list[dict[str, Any]] = []
    try:
        provenance_roots, provenance_refreshes = ensure_provenance(
            repo_root, formal_root, staging
        )
        for figure_id, spec in FIGURES.items():
            provenance = provenance_roots[spec["provenance"]]
            formal_provenance = formal_root / "provenance" / spec["provenance"]
            formal_directory = staging / figure_id
            main_files = []
            for extension in FORMATS:
                source = provenance / f"{spec['asset_stem']}.{extension}"
                destination = formal_directory / f"{figure_id}.{extension}"
                item = copy_verified(source, destination)
                item["path"] = f"{figure_id}/{figure_id}.{extension}"
                main_files.append(item)
            artifact_manifest = provenance / "artifact_manifest.json"
            record = {
                "figure_id": figure_id,
                "formal_directory": f"results/paper_figures/outputs/{figure_id}",
                "main_files": main_files,
                "formal_provenance": relative(formal_provenance, repo_root),
                "bootstrap_source_bundle": spec["bootstrap"],
                "source_bundle_tree_sha256": tree_sha256(provenance),
                "source_bundle_artifact_manifest_sha256": (
                    sha256_file(artifact_manifest) if artifact_manifest.is_file() else None
                ),
                "experiments_rerun": False,
                "scientific_data_changed": False,
                "user_confirmed": True,
                "docx_embedded": False,
            }
            write_json(
                formal_directory / "formal_promotion.json",
                {
                    "schema": "main_figure_formal_promotion_v2",
                    "promotion_id": promotion_id,
                    "promoted_at": now.isoformat(),
                    **record,
                },
            )
            figure_records.append(record)

        manifest = {
            "schema": "main_figures_formal_promotion_v2",
            "promotion_id": promotion_id,
            "promoted_at": now.isoformat(),
            "formal_root": relative(formal_root, repo_root),
            "archive_root": relative(archive_root, repo_root),
            "scope": list(FIGURES),
            "figures": figure_records,
            "plot_only_outputs_promoted": True,
            "experiments_rerun": False,
            "scientific_data_changed": False,
            "user_confirmed": True,
            "docx_embedding_updated": False,
        }
        write_json(staging / "main_figures_promotion_manifest.json", manifest)
        write_qc(
            staging / "all_figures_qc_summary.csv",
            qc_rows(formal_root / "all_figures_qc_summary.csv"),
        )
        authority_staged = stage_authority_update(
            repo_root,
            formal_root,
            archive_root,
            staging,
            now,
            sha256_file(staging / "main_figures_promotion_manifest.json"),
            old_fig5_png_sha256,
        )

        archived: list[dict[str, Any]] = []
        archive_outputs = archive_root / "outputs"
        archive_outputs.mkdir(parents=True, exist_ok=False)
        for figure_id in FIGURES:
            source = formal_root / figure_id
            if source.exists():
                destination = archive_outputs / figure_id
                file_count = sum(1 for path in source.rglob("*") if path.is_file())
                byte_count = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
                digest = tree_sha256(source)
                shutil.move(str(source), str(destination))
                if tree_sha256(destination) != digest:
                    raise RuntimeError(f"Formal archive hash mismatch: {source}")
                archived.append(
                    {
                        "original_directory": relative(source, repo_root),
                        "archived_directory": relative(destination, repo_root),
                        "file_count": file_count,
                        "byte_count": byte_count,
                        "tree_sha256": digest,
                    }
                )
        archived_root_files = []
        for name in ("main_figures_promotion_manifest.json", "all_figures_qc_summary.csv"):
            source = formal_root / name
            if source.is_file():
                destination = archive_outputs / name
                digest = sha256_file(source)
                shutil.move(str(source), str(destination))
                if sha256_file(destination) != digest:
                    raise RuntimeError(f"Formal archive hash mismatch: {source}")
                archived_root_files.append(
                    {
                        "original_path": relative(source, repo_root),
                        "archived_path": relative(destination, repo_root),
                        "sha256": digest,
                    }
                )
        archived_provenance: list[dict[str, Any]] = []
        archive_provenance_root = archive_root / "provenance"
        for refresh in provenance_refreshes:
            source = Path(refresh["formal_directory"])
            if not source.exists():
                if refresh["old_tree_sha256"] is not None:
                    raise RuntimeError(f"Stale provenance disappeared before archive: {source}")
                continue
            destination = archive_provenance_root / str(refresh["provenance"])
            destination.parent.mkdir(parents=True, exist_ok=False)
            digest = tree_sha256(source)
            if refresh["old_tree_sha256"] != digest:
                raise RuntimeError(f"Stale provenance hash changed before archive: {source}")
            file_count = sum(1 for path in source.rglob("*") if path.is_file())
            byte_count = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
            shutil.move(str(source), str(destination))
            if tree_sha256(destination) != digest:
                raise RuntimeError(f"Provenance archive hash mismatch: {source}")
            archived_provenance.append(
                {
                    "provenance": refresh["provenance"],
                    "original_directory": relative(source, repo_root),
                    "archived_directory": relative(destination, repo_root),
                    "file_count": file_count,
                    "byte_count": byte_count,
                    "tree_sha256": digest,
                    "replacement_source_bundle": refresh["source_bundle"],
                }
            )
        archive_manifest = {
            "schema": "pre_reader_first_main_figure_archive_v2",
            "archive_root": relative(archive_root, repo_root),
            "archived_at": now.isoformat(),
            "reason": "Replaced by the user-confirmed latest manuscript Fig.1-Fig.7.",
            "recoverable": True,
            "directories": archived,
            "root_files": archived_root_files,
            "provenance": archived_provenance,
        }
        write_json(archive_root / "archive_manifest.json", archive_manifest)
        write_json(
            repo_root / "archive/move_ledgers" / f"main_figures_reader_first_{timestamp}.json",
            archive_manifest,
        )

        for refresh in provenance_refreshes:
            staged_provenance = Path(refresh["staged_directory"])
            destination = Path(refresh["formal_directory"])
            if destination.exists():
                raise RuntimeError(f"Formal provenance destination was not archived: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged_provenance), str(destination))
            if tree_sha256(destination) != refresh["new_tree_sha256"]:
                raise RuntimeError(f"Installed provenance hash mismatch: {destination}")
        for figure_id in FIGURES:
            shutil.move(str(staging / figure_id), str(formal_root / figure_id))
        for name in ("main_figures_promotion_manifest.json", "all_figures_qc_summary.csv"):
            shutil.move(str(staging / name), str(formal_root / name))
        if authority_staged is not None:
            os.replace(
                str(authority_staged),
                str(repo_root / "docs/paper/PAPER_AUTHORITY.json"),
            )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    result = verify(formal_root)
    result["archive_root"] = relative(archive_root, repo_root)
    result["provenance_refreshed"] = [item["provenance"] for item in provenance_refreshes]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote or verify the user-confirmed formal manuscript Fig.1-Fig.7."
    )
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=paper_fig_output_root(),
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = repo_root_from_here().resolve()
    formal_root = args.formal_root
    if not formal_root.is_absolute():
        formal_root = (repo_root / formal_root).resolve()
    if repo_root not in formal_root.parents:
        raise ValueError("Formal root must stay inside the repository")
    result = verify(formal_root) if args.verify_only else promote(repo_root, formal_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
