from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CSV_PARTS = {"trial_specs", "metrics", "raw"}
VOLATILE_CSV_COLUMNS = {"artifact_digest", "cache_key_digest", "fit_seconds", "sha256", "table_digest"}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    old_root = Path(args.old_root)
    new_root = Path(args.new_root)
    ignore_rel_prefixes = tuple(_normalize_rel_prefix(prefix) for prefix in args.ignore_rel_prefix)
    include_rels = tuple(_normalize_rel_prefix(rel) for rel in args.include_rel)
    errors: list[str] = []
    _compare_csv_sets(
        old_root,
        new_root,
        errors,
        atol=float(args.atol),
        rtol=float(args.rtol),
        ignore_rel_prefixes=ignore_rel_prefixes,
        include_rels=include_rels,
    )
    _compare_summaries(old_root, new_root, errors, include_rels=include_rels)
    _compare_npz_sets(
        old_root,
        new_root,
        errors,
        atol=float(args.atol),
        rtol=float(args.rtol),
        ignore_rel_prefixes=ignore_rel_prefixes,
        include_rels=include_rels,
    )
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print("PASS regression outputs match")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare old/new paper-figure smoke outputs.")
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument(
        "--ignore-rel-prefix",
        action="append",
        default=[],
        help="Relative path prefix to exclude from CSV/NPZ set comparison. May be repeated.",
    )
    parser.add_argument(
        "--include-rel",
        action="append",
        default=[],
        help="Relative file path to compare. When set, only listed paths are compared. May be repeated.",
    )
    return parser.parse_args(argv)


def _normalize_rel_prefix(prefix: str) -> str:
    return str(prefix).replace("\\", "/").strip("/")


def _is_ignored(rel: str, ignore_rel_prefixes: Iterable[str]) -> bool:
    rel_norm = rel.replace("\\", "/").strip("/")
    return any(rel_norm == prefix or rel_norm.startswith(f"{prefix}/") for prefix in ignore_rel_prefixes if prefix)


def _is_included(rel: str, include_rels: Iterable[str]) -> bool:
    include = {str(item).replace("\\", "/").strip("/") for item in include_rels if item}
    if not include:
        return True
    return rel.replace("\\", "/").strip("/") in include


def _interesting_csvs(
    root: Path,
    *,
    ignore_rel_prefixes: Iterable[str] = (),
    include_rels: Iterable[str] = (),
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in root.rglob("*.csv"):
        rel = path.relative_to(root).as_posix()
        if _is_ignored(rel, ignore_rel_prefixes):
            continue
        if include_rels:
            if _is_included(rel, include_rels):
                out[rel] = path
        elif any(part in CSV_PARTS for part in path.parts):
            out[rel] = path
    return out


def _npz_files(
    root: Path,
    *,
    ignore_rel_prefixes: Iterable[str] = (),
    include_rels: Iterable[str] = (),
) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*.npz")
        if not _is_ignored(path.relative_to(root).as_posix(), ignore_rel_prefixes)
        and _is_included(path.relative_to(root).as_posix(), include_rels)
    }


def _compare_csv_sets(
    old_root: Path,
    new_root: Path,
    errors: list[str],
    *,
    atol: float,
    rtol: float,
    ignore_rel_prefixes: Iterable[str] = (),
    include_rels: Iterable[str] = (),
) -> None:
    old = _interesting_csvs(old_root, ignore_rel_prefixes=ignore_rel_prefixes, include_rels=include_rels)
    new = _interesting_csvs(new_root, ignore_rel_prefixes=ignore_rel_prefixes, include_rels=include_rels)
    for rel in sorted(set(old) | set(new)):
        if rel not in old:
            errors.append(f"csv only in new: {rel}")
            continue
        if rel not in new:
            errors.append(f"csv missing in new: {rel}")
            continue
        _compare_csv(rel, old[rel], new[rel], errors, atol=atol, rtol=rtol)


def _compare_csv(rel: str, old_path: Path, new_path: Path, errors: list[str], *, atol: float, rtol: float) -> None:
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    if list(old.columns) != list(new.columns):
        errors.append(f"{rel}: column schema differs old={list(old.columns)} new={list(new.columns)}")
        return
    if len(old) != len(new):
        errors.append(f"{rel}: row count differs old={len(old)} new={len(new)}")
        return
    if old.empty and new.empty:
        return
    sort_cols = list(old.columns)
    old = old.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)
    new = new.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)
    for col in old.columns:
        if col in VOLATILE_CSV_COLUMNS:
            continue
        a = old[col]
        b = new[col]
        if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
            av = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
            bv = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
            if not np.allclose(av, bv, atol=atol, rtol=rtol, equal_nan=True):
                errors.append(f"{rel}: float column differs: {col}")
        elif pd.api.types.is_integer_dtype(a) or pd.api.types.is_bool_dtype(a):
            if not np.array_equal(a.to_numpy(), b.to_numpy()):
                errors.append(f"{rel}: exact column differs: {col}")
        else:
            if not np.array_equal(a.fillna("<NA>").astype(str).to_numpy(), b.fillna("<NA>").astype(str).to_numpy()):
                errors.append(f"{rel}: string column differs: {col}")


def _compare_summaries(
    old_root: Path,
    new_root: Path,
    errors: list[str],
    *,
    include_rels: Iterable[str] = (),
) -> None:
    old = _summary_files(old_root, include_rels=include_rels)
    new = _summary_files(new_root, include_rels=include_rels)
    for rel in sorted(set(old) | set(new)):
        if rel not in old:
            errors.append(f"summary only in new: {rel}")
            continue
        if rel not in new:
            errors.append(f"summary missing in new: {rel}")
            continue
        with old[rel].open("r", encoding="utf-8") as handle:
            old_payload = json.load(handle)
        with new[rel].open("r", encoding="utf-8") as handle:
            new_payload = json.load(handle)
        if old_payload.get("completed_modules") != new_payload.get("completed_modules"):
            errors.append(f"{rel}: completed_modules differs")
        _check_output_files(rel, new_root / Path(rel).parent, new_payload, errors)


def _summary_files(root: Path, *, include_rels: Iterable[str] = ()) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("summary.json")
        if _is_included(path.relative_to(root).as_posix(), include_rels)
    }


def _check_output_files(rel: str, root: Path, payload: dict, errors: list[str]) -> None:
    output_files = payload.get("output_files", {})
    if not isinstance(output_files, dict):
        errors.append(f"{rel}: output_files is not a dict")
        return
    for key, value in output_files.items():
        if isinstance(value, str) and value and not (root / value).exists():
            errors.append(f"{rel}: output_files[{key}] missing path {value}")


def _compare_npz_sets(
    old_root: Path,
    new_root: Path,
    errors: list[str],
    *,
    atol: float,
    rtol: float,
    ignore_rel_prefixes: Iterable[str] = (),
    include_rels: Iterable[str] = (),
) -> None:
    old = _npz_files(old_root, ignore_rel_prefixes=ignore_rel_prefixes, include_rels=include_rels)
    new = _npz_files(new_root, ignore_rel_prefixes=ignore_rel_prefixes, include_rels=include_rels)
    for rel in sorted(set(old) | set(new)):
        if rel not in old:
            errors.append(f"npz only in new: {rel}")
            continue
        if rel not in new:
            errors.append(f"npz missing in new: {rel}")
            continue
        _compare_npz(rel, old[rel], new[rel], errors, atol=atol, rtol=rtol)


def _compare_npz(rel: str, old_path: Path, new_path: Path, errors: list[str], *, atol: float, rtol: float) -> None:
    with np.load(old_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
        old_keys = set(old.files)
        new_keys = set(new.files)
        if old_keys != new_keys:
            errors.append(f"{rel}: npz keys differ old_only={sorted(old_keys - new_keys)} new_only={sorted(new_keys - old_keys)}")
            return
        for key in sorted(old_keys):
            a = old[key]
            b = new[key]
            if a.shape != b.shape:
                errors.append(f"{rel}:{key}: shape differs old={a.shape} new={b.shape}")
                continue
            if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
                if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
                    errors.append(f"{rel}:{key}: float array differs")
            elif not np.array_equal(a, b):
                errors.append(f"{rel}:{key}: array differs")


if __name__ == "__main__":
    raise SystemExit(main())
