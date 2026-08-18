from __future__ import annotations

"""Artifact persistence for the Fig.6b order-specificity pilot DAG.

Persisted source specs -> validated reusable artifacts -> downstream outputs:
- sequence_specs task persists data/sequence_specs.csv + data/singleton_reference_specs.csv
  and an artifact copy with a cache key + digest;
- state_bank task persists data/intermediates/seed_<n>/... and an artifact copy;
- analysis is load-only for parents: missing, stale or corrupt parents must fail
  loudly in require mode.
"""


import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def cache_key_digest(key: Mapping[str, Any]) -> str:
    payload = repr(sorted(key.items())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_cache_key(artifact_dir: Path, key: Mapping[str, Any]) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "cache_key.json"
    path.write_text(json.dumps(key, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def cache_key_matches(artifact_dir: Path, expected_key: Mapping[str, Any]) -> bool:
    path = artifact_dir / "cache_key.json"
    if not path.exists():
        return False
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(sorted(stored.items()) == sorted(expected_key.items()))


def artifact_dir_for(root: Path, task: str, network_seed: int | None = None) -> Path:
    base = root / "artifacts" / task
    return base / f"seed_{int(network_seed)}" if network_seed is not None else base


class ArtifactState:
    """Tracks the source of every loaded/built artifact for summary metadata."""

    def __init__(self) -> None:
        self.sources: dict[str, dict[str, str]] = {}

    def record(self, name: str, source: str, artifact_dir: Path, digest: str, cache_key: Mapping[str, Any]) -> None:
        self.sources[name] = {
            "source": source,
            "artifact_dir": str(Path(artifact_dir).resolve()),
            "digest": digest,
            "cache_key_digest": cache_key_digest(cache_key),
        }


def save_specs_artifact(
    artifact_dir: Path,
    sequence_specs: pd.DataFrame,
    reference_specs: pd.DataFrame,
    cache_key: Mapping[str, Any],
    *,
    digest: str,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sequence_specs.to_csv(artifact_dir / "sequence_specs.csv", index=False, encoding="utf-8")
    reference_specs.to_csv(artifact_dir / "singleton_reference_specs.csv", index=False, encoding="utf-8")
    write_cache_key(artifact_dir, cache_key)
    (artifact_dir / "digest.json").write_text(json.dumps({"digest": digest}, sort_keys=True), encoding="utf-8")


def load_specs_artifact(
    artifact_dir: Path,
    expected_key: Mapping[str, Any],
    *,
    expected_digest: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cache_key_matches(artifact_dir, expected_key):
        raise RuntimeError(
            f"Sequence-specs artifact missing or stale: {artifact_dir} "
            f"(expected cache key {cache_key_digest(expected_key)})"
        )
    seq_path = artifact_dir / "sequence_specs.csv"
    ref_path = artifact_dir / "singleton_reference_specs.csv"
    if not seq_path.exists() or not ref_path.exists():
        raise RuntimeError(f"Sequence-specs artifact incomplete: {artifact_dir}")
    digest_path = artifact_dir / "digest.json"
    if expected_digest and digest_path.exists():
        stored_digest = str(json.loads(digest_path.read_text(encoding="utf-8")).get("digest", ""))
        if stored_digest != expected_digest:
            raise RuntimeError(
                f"Sequence-specs artifact digest mismatch in {artifact_dir}: "
                f"expected {expected_digest}, found {stored_digest}"
            )
    sequence_specs = pd.read_csv(seq_path)
    reference_specs = pd.read_csv(ref_path)
    if len(sequence_specs) == 0 or len(reference_specs) == 0:
        raise RuntimeError(f"Sequence-specs artifact empty: {artifact_dir}")
    return sequence_specs, reference_specs


def save_bank_artifact(
    artifact_dir: Path,
    bank_dir: Path,
    cache_key: Mapping[str, Any],
    *,
    digest: str,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("state_bank_layer2.npz", "state_bank_manifest.csv", "sequence_meta.csv", "capture_summary.json"):
        source = bank_dir / filename
        if source.exists():
            target = artifact_dir / filename
            target.write_bytes(source.read_bytes())
    write_cache_key(artifact_dir, cache_key)
    (artifact_dir / "digest.json").write_text(json.dumps({"digest": digest}, sort_keys=True), encoding="utf-8")


def load_bank_artifact(
    artifact_dir: Path,
    expected_key: Mapping[str, Any],
    *,
    expected_digest: str,
) -> Path:
    if not cache_key_matches(artifact_dir, expected_key):
        raise RuntimeError(
            f"State-bank artifact missing or stale: {artifact_dir} "
            f"(expected cache key {cache_key_digest(expected_key)})"
        )
    for filename in ("state_bank_layer2.npz", "state_bank_manifest.csv", "sequence_meta.csv"):
        if not (artifact_dir / filename).exists():
            raise RuntimeError(f"State-bank artifact incomplete: {artifact_dir} missing {filename}")
    digest_path = artifact_dir / "digest.json"
    if expected_digest and digest_path.exists():
        stored_digest = str(json.loads(digest_path.read_text(encoding="utf-8")).get("digest", ""))
        if stored_digest != expected_digest:
            raise RuntimeError(
                f"State-bank artifact digest mismatch in {artifact_dir}: "
                f"expected {expected_digest}, found {stored_digest}"
            )
    return artifact_dir


def bank_digest(bank_dir: Path) -> str:
    parts = []
    for filename in ("state_bank_layer2.npz", "state_bank_manifest.csv", "sequence_meta.csv"):
        path = bank_dir / filename
        if path.exists():
            parts.append(f"{filename}:{file_sha256(path)}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def copy_bank_to_bundle(bank_dir: Path, bundle_bank_dir: Path) -> None:
    bundle_bank_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("state_bank_layer2.npz", "state_bank_manifest.csv", "sequence_meta.csv", "capture_summary.json"):
        source = bank_dir / filename
        if source.exists():
            target = bundle_bank_dir / filename
            target.write_bytes(source.read_bytes())


__all__ = [
    "ArtifactState",
    "artifact_dir_for",
    "bank_digest",
    "cache_key_digest",
    "cache_key_matches",
    "copy_bank_to_bundle",
    "file_sha256",
    "load_bank_artifact",
    "load_specs_artifact",
    "save_bank_artifact",
    "save_specs_artifact",
    "write_cache_key",
]
