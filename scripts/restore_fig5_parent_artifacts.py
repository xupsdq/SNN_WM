from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verify_fig5_parent_reconstruction import (
    TARGET_NAMES,
    expected_for_seed,
    seed_list,
    sha256_file,
    verify_seed,
)


CONFIRMATION = "RESTORE_40_BYTE_IDENTICAL_FIG5_PARENTS"
PINNED_REGISTER_SHA256 = "cbd78cb3b9526825c48dfe06718e84c099122d5b6426ed46e8a97df9115e5f5f"
CANONICAL_ROOT = Path("results/paper_figure_multi_seed/fig5_local_support_competition")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def path_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def check_existing(path: Path, identity: dict[str, Any]) -> str:
    if path.is_symlink():
        raise RuntimeError(f"Refusing symlink target: {path}")
    if not path.exists():
        return "missing"
    if not path.is_file():
        raise RuntimeError(f"Refusing non-file target: {path}")
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != int(identity["expected_bytes"]) or actual_sha256 != identity["expected_sha256"]:
        raise RuntimeError(
            f"Refusing to overwrite mismatched target: {path} "
            f"bytes={actual_bytes} sha256={actual_sha256}"
        )
    return "already_byte_identical"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atomically restore the 40 verified Fig.5 parent CSVs after explicit authorization."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--register",
        type=Path,
        default=Path("archive/move_ledgers/parent_artifact_gap_register_pre_restore_20260814.json"),
    )
    parser.add_argument("--seeds", default="1000-1019")
    parser.add_argument("--confirmation", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("archive/move_ledgers/parent_artifact_restoration_20260814.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.confirmation != CONFIRMATION:
        raise SystemExit(f"Refusing restore: --confirmation must equal {CONFIRMATION}")

    root = args.repo_root.resolve()
    canonical_root = (root / CANONICAL_ROOT).resolve()
    register_path = args.register if args.register.is_absolute() else root / args.register
    output_path = args.output if args.output.is_absolute() else root / args.output
    register_sha256 = sha256_file(register_path)
    if register_sha256 != PINNED_REGISTER_SHA256:
        raise RuntimeError(
            f"Pinned register SHA-256 mismatch: expected {PINNED_REGISTER_SHA256}, got {register_sha256}"
        )
    register = json.loads(register_path.read_text(encoding="utf-8-sig"))
    seeds = seed_list(args.seeds)
    if seeds != list(range(1000, 1020)):
        raise RuntimeError(f"This authorized restoration requires seeds 1000-1019; got {seeds}")
    if int(register.get("unique_missing_file_count", -1)) != 40 or len(register.get("missing_files", [])) != 40:
        raise RuntimeError("The pinned pre-restore register must contain exactly 40 identities")

    expected_by_path = {row["path"]: row for row in register["missing_files"]}
    preflight: list[dict[str, Any]] = []
    for seed in seeds:
        expected = expected_for_seed(register, seed)
        for key in sorted(TARGET_NAMES):
            identity = expected[key]
            target = root / identity["path"]
            if not path_within(target.parent, canonical_root):
                raise RuntimeError(f"Target escapes canonical root: {target}")
            if target.name != TARGET_NAMES[key]:
                raise RuntimeError(f"Unexpected target name: {target}")
            if not target.parent.is_dir():
                raise FileNotFoundError(f"Canonical target directory is missing: {target.parent}")
            state = check_existing(target, identity)
            preflight.append({"path": identity["path"], "state": state})

    missing_bytes = sum(
        int(expected_by_path[row["path"]]["expected_bytes"])
        for row in preflight
        if row["state"] == "missing"
    )
    free_bytes = shutil.disk_usage(canonical_root).free
    if free_bytes < missing_bytes + 512 * 1024 * 1024:
        raise RuntimeError(f"Insufficient free space: need at least {missing_bytes + 512 * 1024 * 1024}, have {free_bytes}")

    payload: dict[str, Any] = {
        "schema": "net_torch_parent_artifact_restoration_v1",
        "started_at": utc_now(),
        "completed_at": None,
        "status": "in_progress",
        "mode": "authorized_atomic_byte_identical_writeback",
        "authorization": {
            "user_request": "写回这 40 个已验证的 byte-identical parent CSV",
            "confirmation_token": CONFIRMATION,
        },
        "repo_root": str(root),
        "register_snapshot": register_path.relative_to(root).as_posix(),
        "register_snapshot_sha256": register_sha256,
        "canonical_root": CANONICAL_ROOT.as_posix(),
        "seeds_requested": seeds,
        "target_file_count": len(preflight),
        "target_bytes": sum(int(row["expected_bytes"]) for row in register["missing_files"]),
        "preflight": preflight,
        "free_bytes_before": free_bytes,
        "results": [],
        "safety": {
            "mismatched_existing_targets_overwritten": 0,
            "frozen_derived_bundle_files_modified": 0,
            "runtime_parent_gap_files_modified": 0,
            "protected_manuscripts_modified": 0,
            "atomic_promotion": "os.replace from a same-directory verified staging directory",
        },
    }
    atomic_json(output_path, payload)

    try:
        for seed in seeds:
            expected = expected_for_seed(register, seed)
            target_dir = (root / expected["panel_b"]["path"]).parent
            stage = Path(tempfile.mkdtemp(prefix=".fig5_parent_restore_", dir=target_dir))
            seed_result: dict[str, Any] | None = None
            try:
                seed_result = verify_seed(root, register, seed, stage)
                if seed_result["status"] != "byte_identical_reconstruction_verified":
                    raise RuntimeError(f"Seed {seed} reconstruction did not match the pinned identities")
                restoration: dict[str, Any] = {}
                for key in sorted(TARGET_NAMES):
                    output = seed_result["outputs"][key]
                    target = root / output["target_path"]
                    staged = stage / TARGET_NAMES[key]
                    state = check_existing(target, expected[key])
                    if state == "missing":
                        if staged.stat().st_size != int(output["expected_bytes"]):
                            raise RuntimeError(f"Staged size changed before promotion: {staged}")
                        if sha256_file(staged) != output["expected_sha256"]:
                            raise RuntimeError(f"Staged SHA-256 changed before promotion: {staged}")
                        os.replace(staged, target)
                        action = "restored"
                    else:
                        action = "already_byte_identical"
                    final_bytes = target.stat().st_size
                    final_sha256 = sha256_file(target)
                    if final_bytes != int(output["expected_bytes"]) or final_sha256 != output["expected_sha256"]:
                        raise RuntimeError(f"Post-write verification failed: {target}")
                    restoration[key] = {
                        "path": output["target_path"],
                        "action": action,
                        "bytes": final_bytes,
                        "sha256": final_sha256,
                        "post_write_verified": True,
                    }
                seed_result["restoration"] = restoration
                seed_result["restoration_status"] = "completed_verified"
                payload["results"].append(seed_result)
                payload["completed_seed_count"] = len(payload["results"])
                atomic_json(output_path, payload)
            finally:
                if stage.exists() and not any(stage.iterdir()):
                    stage.rmdir()
                elif stage.exists() and seed_result is not None and seed_result.get("restoration_status") == "completed_verified":
                    # An already-present target can leave its verified staged counterpart behind.
                    for child in stage.iterdir():
                        if child.is_file() and child.name in TARGET_NAMES.values():
                            child.unlink()
                    stage.rmdir()

        postcheck: list[dict[str, Any]] = []
        for row in register["missing_files"]:
            target = root / row["path"]
            state = check_existing(target, row)
            postcheck.append(
                {
                    "path": row["path"],
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                    "state": state,
                }
            )
        payload["postcheck"] = postcheck
        payload["restored_file_count"] = sum(
            item["action"] == "restored"
            for seed_result in payload["results"]
            for item in seed_result["restoration"].values()
        )
        payload["already_present_file_count"] = sum(
            item["action"] == "already_byte_identical"
            for seed_result in payload["results"]
            for item in seed_result["restoration"].values()
        )
        payload["verified_file_count"] = len(postcheck)
        payload["verified_bytes"] = sum(item["bytes"] for item in postcheck)
        payload["free_bytes_after"] = shutil.disk_usage(canonical_root).free
        payload["completed_at"] = utc_now()
        payload["status"] = "completed_verified"
        atomic_json(output_path, payload)
    except Exception as exc:
        payload["completed_at"] = utc_now()
        payload["status"] = "failed_partial_write_possible"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(output_path, payload)
        raise

    print(f"output={output_path}")
    print(f"status={payload['status']}")
    print(f"restored_file_count={payload['restored_file_count']}")
    print(f"already_present_file_count={payload['already_present_file_count']}")
    print(f"verified_file_count={payload['verified_file_count']}")
    print(f"verified_bytes={payload['verified_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
