from __future__ import annotations

"""Load and verify the frozen Fig.6b formal-analysis specification.

The companion SHA-256 file makes any post-freeze edit explicit. Formal runtime
code must call :func:`load_frozen_formal_spec` before scoring or materializing a
20-network analysis bundle.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


FORMAL_SPEC_PATH = Path(__file__).with_name("formal_analysis_spec.json")
FORMAL_SPEC_SHA256_PATH = Path(__file__).with_name("formal_analysis_spec.sha256")
FORMAL_SPEC_SCHEMA = "fig6b_order_specificity_formal_analysis_spec_v1"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_formal_spec() -> dict[str, Any]:
    if not FORMAL_SPEC_PATH.exists():
        raise FileNotFoundError(f"Frozen formal-analysis spec is missing: {FORMAL_SPEC_PATH}")
    if not FORMAL_SPEC_SHA256_PATH.exists():
        raise FileNotFoundError(
            f"Frozen formal-analysis spec digest is missing: {FORMAL_SPEC_SHA256_PATH}"
        )
    digest_tokens = FORMAL_SPEC_SHA256_PATH.read_text(encoding="utf-8").strip().split()
    if not digest_tokens:
        raise RuntimeError(f"Frozen spec digest file is empty: {FORMAL_SPEC_SHA256_PATH}")
    expected_digest = digest_tokens[0].lower()
    observed_digest = _sha256_file(FORMAL_SPEC_PATH)
    if observed_digest != expected_digest:
        raise RuntimeError(
            "Frozen Fig.6b formal-analysis spec digest mismatch: "
            f"expected {expected_digest}, observed {observed_digest}. "
            "Treat any change as an explicit new spec version before formal scoring."
        )
    payload = json.loads(FORMAL_SPEC_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != FORMAL_SPEC_SCHEMA:
        raise RuntimeError(
            f"Unexpected formal spec schema: {payload.get('schema')!r}; "
            f"expected {FORMAL_SPEC_SCHEMA!r}"
        )
    if payload.get("status") != "frozen_before_formal_scoring":
        raise RuntimeError(f"Formal spec is not frozen: status={payload.get('status')!r}")
    seeds = payload.get("design", {}).get("network_seeds", [])
    if seeds != list(range(1000, 1020)):
        raise RuntimeError(f"Formal spec must contain exactly network seeds 1000-1019, found {seeds}")
    return payload


__all__ = [
    "FORMAL_SPEC_PATH",
    "FORMAL_SPEC_SCHEMA",
    "FORMAL_SPEC_SHA256_PATH",
    "load_frozen_formal_spec",
]
