from __future__ import annotations

import json
from pathlib import Path

from scripts import promote_main_figures as promotion


def _write_candidate(root: Path, bundle: str, *, payload: bytes = b"new") -> Path:
    source = root / "results" / "paper_figure_candidates" / bundle
    (source / "figures").mkdir(parents=True)
    for extension in promotion.FORMATS:
        (source / "figures" / f"manuscript_fig5.{extension}").write_bytes(payload + extension.encode())
    (source / "artifact_manifest.json").write_text("{}\n", encoding="utf-8")
    return source


def _write_formal(root: Path, *, bootstrap: str, old_payload: bytes = b"old") -> Path:
    formal = root / "results" / "paper_figures" / "outputs"
    (formal / "fig5").mkdir(parents=True)
    for extension in promotion.FORMATS:
        (formal / "fig5" / f"fig5.{extension}").write_bytes(old_payload + extension.encode())
    (formal / "fig5" / "formal_promotion.json").write_text(
        json.dumps({"figure_id": "fig5", "bootstrap_source_bundle": bootstrap}) + "\n",
        encoding="utf-8",
    )
    provenance = formal / "provenance" / "fig5"
    provenance.mkdir(parents=True)
    (provenance / "old.txt").write_bytes(old_payload)
    return formal


def test_stale_provenance_is_staged_and_unchanged_root_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        promotion,
        "FIGURES",
        {"fig5": {"bootstrap": "results/paper_figure_candidates/v3", "provenance": "fig5", "asset_stem": "figures/manuscript_fig5"}},
    )
    source = _write_candidate(tmp_path, "v3")
    formal = _write_formal(tmp_path, bootstrap="results/paper_figure_candidates/v1")
    staging = tmp_path / "tmp" / "staging"
    staging.mkdir(parents=True)
    roots, refreshes = promotion.ensure_provenance(tmp_path, formal, staging)
    assert roots["fig5"] == staging / "provenance" / "fig5"
    assert [item["provenance"] for item in refreshes] == ["fig5"]
    assert promotion.tree_sha256(roots["fig5"]) == promotion.tree_sha256(source)
    assert (formal / "provenance" / "fig5" / "old.txt").read_bytes() == b"old"

    (formal / "fig5" / "formal_promotion.json").write_text(
        json.dumps({"figure_id": "fig5", "bootstrap_source_bundle": "results/paper_figure_candidates/v3"}) + "\n",
        encoding="utf-8",
    )
    second_staging = tmp_path / "tmp" / "staging_2"
    second_staging.mkdir(parents=True)
    roots, refreshes = promotion.ensure_provenance(tmp_path, formal, second_staging)
    assert roots["fig5"] == formal / "provenance" / "fig5"
    assert refreshes == []


def test_promotion_archives_old_output_and_stale_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        promotion,
        "FIGURES",
        {"fig5": {"bootstrap": "results/paper_figure_candidates/v3", "provenance": "fig5", "asset_stem": "figures/manuscript_fig5"}},
    )
    candidate = _write_candidate(tmp_path, "v3", payload=b"new")
    formal = _write_formal(tmp_path, bootstrap="results/paper_figure_candidates/v1", old_payload=b"old")
    result = promotion.promote(tmp_path, formal)

    archive_root = tmp_path / result["archive_root"]
    assert result["provenance_refreshed"] == ["fig5"]
    assert (formal / "fig5" / "fig5.png").read_bytes() == (candidate / "figures" / "manuscript_fig5.png").read_bytes()
    assert (archive_root / "outputs" / "fig5" / "fig5.png").read_bytes() == b"oldpng"
    assert (archive_root / "provenance" / "fig5" / "old.txt").read_bytes() == b"old"
    archive_manifest = json.loads((archive_root / "archive_manifest.json").read_text(encoding="utf-8"))
    assert archive_manifest["provenance"][0]["provenance"] == "fig5"
    assert promotion.verify(formal)["status"] == "passed"
