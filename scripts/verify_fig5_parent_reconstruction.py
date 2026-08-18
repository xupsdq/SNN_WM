from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PANEL_B_COLUMNS = [
    "network_seed",
    "trial_id",
    "unit_id",
    "unit_group",
    "early_window_ms",
    "transition_type",
    "first_spike_dynamic",
    "first_spike_static",
    "delta_first_spike_latency",
    "early_spike_count_dynamic",
    "early_spike_count_static",
    "delta_early_spike_count",
]
PANEL_D_COLUMNS = [
    "network_seed",
    "trial_id",
    "condition",
    "condition_label",
    "unit_id",
    "unit_group",
    "layer_or_map",
    "row",
    "col",
    "included_in_main",
    "first_spike_static",
    "first_spike_condition",
    "transition_vs_static",
    "early_spike_count_static",
    "early_spike_count_condition",
    "delta_early_spike_count_vs_static",
    "perturbation_mode",
    "perturbed_layer",
    "perturbed_variables",
]
TARGET_NAMES = {
    "panel_b": "panel_b_early_firing_transition_metrics.csv",
    "panel_d": "panel_d_l1_stsp_perturbation_unit_transitions.csv",
}
CONDITIONS = ("dynamic_intact", "attenuate_l1_stsp", "reset_l1_stsp")
CONDITION_LABELS = {
    "dynamic_intact": "Dynamic",
    "attenuate_l1_stsp": "Attenuate L1 STSP",
    "reset_l1_stsp": "Reset L1 STSP",
}
INCLUDED_GROUPS = {"overlap_dominant", "probe_only_dominant", "random_matched"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_spike_map(spikes: np.ndarray) -> np.ndarray:
    array = np.asarray(spikes)
    first = np.full(array.shape[1:], -1, dtype=np.int64)
    fired = array > 0
    any_fire = fired.any(axis=0)
    if np.any(any_fire):
        first[any_fire] = np.argmax(fired, axis=0)[any_fire]
    return first


def transition_type(dynamic_first: np.ndarray, static_first: np.ndarray) -> np.ndarray:
    return np.select(
        [
            (dynamic_first >= 0) & (static_first >= 0) & (dynamic_first < static_first),
            (dynamic_first >= 0) & (static_first < 0),
            (dynamic_first < 0) & (static_first >= 0),
        ],
        ["advance", "recruit", "loss"],
        default="unchanged",
    )


def latency_delta(dynamic_first: np.ndarray, static_first: np.ndarray) -> np.ndarray:
    output = np.full(dynamic_first.shape, np.nan, dtype=np.float64)
    both = (dynamic_first >= 0) & (static_first >= 0)
    recruited = (dynamic_first >= 0) & (static_first < 0)
    lost = (dynamic_first < 0) & (static_first >= 0)
    output[both] = dynamic_first[both] - static_first[both]
    output[recruited] = -dynamic_first[recruited]
    output[lost] = static_first[lost]
    return output


def condition_metadata(condition: str) -> tuple[str, str, str]:
    if condition == "attenuate_l1_stsp":
        return "attenuate", "layer1", "u_pre;x_pre"
    if condition == "reset_l1_stsp":
        return "reset", "layer1", "u_pre;x_pre"
    return "none", "none", "none"


def append_frame(path: Path, frame: pd.DataFrame, *, first: bool) -> None:
    frame.to_csv(
        path,
        mode="w" if first else "a",
        header=first,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


def build_panel_b_frame(
    *,
    seed: int,
    trial_id: int,
    groups: pd.DataFrame,
    dynamic_spikes: np.ndarray,
    static_spikes: np.ndarray,
    early_window_ms: int,
) -> pd.DataFrame:
    rows = groups["row"].to_numpy(dtype=np.int64)
    cols = groups["col"].to_numpy(dtype=np.int64)
    first_dynamic_map = first_spike_map(dynamic_spikes)
    first_static_map = first_spike_map(static_spikes)
    first_dynamic = first_dynamic_map[rows, cols]
    first_static = first_static_map[rows, cols]
    early_dynamic_map = np.asarray(dynamic_spikes[:early_window_ms]).sum(axis=0)
    early_static_map = np.asarray(static_spikes[:early_window_ms]).sum(axis=0)
    early_dynamic = early_dynamic_map[rows, cols].astype(np.float64)
    early_static = early_static_map[rows, cols].astype(np.float64)
    return pd.DataFrame(
        {
            "network_seed": int(seed),
            "trial_id": int(trial_id),
            "unit_id": groups["unit_id"].to_numpy(dtype=np.int64),
            "unit_group": groups["unit_group"].astype(str).to_numpy(),
            "early_window_ms": int(early_window_ms),
            "transition_type": transition_type(first_dynamic, first_static),
            "first_spike_dynamic": first_dynamic,
            "first_spike_static": first_static,
            "delta_first_spike_latency": latency_delta(first_dynamic, first_static),
            "early_spike_count_dynamic": early_dynamic,
            "early_spike_count_static": early_static,
            "delta_early_spike_count": early_dynamic - early_static,
        },
        columns=PANEL_B_COLUMNS,
    )


def build_panel_d_frame(
    *,
    seed: int,
    trial_id: int,
    condition: str,
    groups: pd.DataFrame,
    condition_spikes: np.ndarray,
    static_spikes: np.ndarray,
    early_window_ms: int,
) -> pd.DataFrame:
    rows = groups["row"].to_numpy(dtype=np.int64)
    cols = groups["col"].to_numpy(dtype=np.int64)
    first_static_map = first_spike_map(static_spikes)
    first_condition_map = first_spike_map(condition_spikes)
    first_static = first_static_map[rows, cols]
    first_condition = first_condition_map[rows, cols]
    early_static_map = np.asarray(static_spikes[:early_window_ms]).sum(axis=0)
    early_condition_map = np.asarray(condition_spikes[:early_window_ms]).sum(axis=0)
    early_static = early_static_map[rows, cols].astype(np.float64)
    early_condition = early_condition_map[rows, cols].astype(np.float64)
    mode, layer, variables = condition_metadata(condition)
    return pd.DataFrame(
        {
            "network_seed": int(seed),
            "trial_id": int(trial_id),
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "unit_id": groups["unit_id"].to_numpy(dtype=np.int64),
            "unit_group": groups["unit_group"].astype(str).to_numpy(),
            "layer_or_map": "layer1",
            "row": rows,
            "col": cols,
            "included_in_main": True,
            "first_spike_static": first_static,
            "first_spike_condition": first_condition,
            "transition_vs_static": transition_type(first_condition, first_static),
            "early_spike_count_static": early_static,
            "early_spike_count_condition": early_condition,
            "delta_early_spike_count_vs_static": early_condition - early_static,
            "perturbation_mode": mode,
            "perturbed_layer": layer,
            "perturbed_variables": variables,
        },
        columns=PANEL_D_COLUMNS,
    )


def seed_list(value: str) -> list[int]:
    output: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            output.extend(range(int(start), int(end) + 1))
        else:
            output.append(int(token))
    return sorted(set(output))


def expected_for_seed(register: dict[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    marker = f"/seed_{seed}/"
    found: dict[str, dict[str, Any]] = {}
    for row in register["missing_files"]:
        if marker not in f"/{row['path']}":
            continue
        basename = Path(row["path"]).name
        for key, target in TARGET_NAMES.items():
            if basename == target:
                found[key] = row
    if set(found) != set(TARGET_NAMES):
        raise RuntimeError(f"Missing expected identities for seed {seed}: {sorted(found)}")
    return found


def verify_seed(root: Path, register: dict[str, Any], seed: int, scratch: Path) -> dict[str, Any]:
    expected = expected_for_seed(register, seed)
    source_seed = root / ".codex/tmp/20260713_data_adjustment/fig5_local_support_competition" / f"seed_{seed}"
    artifact_seed = root / "results/multi_seed_rollout/fig5/fig5_local_support_competition" / f"seed_{seed}"
    group_path = source_seed / "data/trial_specs/unit_group_definitions.csv"
    branch_path = artifact_seed / "data/intermediates/preprobe_support_bank/branch_traces.npz"
    config_path = source_seed / "config/run_config.json"
    for path in (group_path, branch_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    early_window_ms = int(config["early_window_ms"])
    unit_groups = pd.read_csv(group_path)
    if len(unit_groups) != 392000:
        raise RuntimeError(f"Unexpected unit-group row count for seed {seed}: {len(unit_groups)}")
    trial_ids = sorted(int(value) for value in unit_groups["trial_id"].unique())
    panel_b_path = scratch / TARGET_NAMES["panel_b"]
    panel_d_path = scratch / TARGET_NAMES["panel_d"]
    panel_b_rows = panel_d_rows = 0
    first_b = first_d = True

    with np.load(branch_path, allow_pickle=False) as arrays:
        names = set(arrays.files)
        for trial_id in trial_ids:
            trial_groups = unit_groups[unit_groups["trial_id"].eq(trial_id)]
            main_groups = trial_groups[trial_groups["unit_group"].isin(INCLUDED_GROUPS)]
            static_key = f"trial_{trial_id}__static_frozen__spikes"
            dynamic_key = f"trial_{trial_id}__dynamic_intact__spikes"
            if static_key not in names or dynamic_key not in names:
                raise RuntimeError(f"Missing branch arrays for seed {seed}, trial {trial_id}")
            static_spikes = arrays[static_key]
            dynamic_spikes = arrays[dynamic_key]
            frame_b = build_panel_b_frame(
                seed=seed,
                trial_id=trial_id,
                groups=trial_groups,
                dynamic_spikes=dynamic_spikes,
                static_spikes=static_spikes,
                early_window_ms=early_window_ms,
            )
            append_frame(panel_b_path, frame_b, first=first_b)
            first_b = False
            panel_b_rows += len(frame_b)
            for condition in CONDITIONS:
                condition_key = f"trial_{trial_id}__{condition}__spikes"
                if condition_key not in names:
                    raise RuntimeError(f"Missing {condition_key} for seed {seed}")
                condition_spikes = dynamic_spikes if condition == "dynamic_intact" else arrays[condition_key]
                frame_d = build_panel_d_frame(
                    seed=seed,
                    trial_id=trial_id,
                    condition=condition,
                    groups=main_groups,
                    condition_spikes=condition_spikes,
                    static_spikes=static_spikes,
                    early_window_ms=early_window_ms,
                )
                append_frame(panel_d_path, frame_d, first=first_d)
                first_d = False
                panel_d_rows += len(frame_d)

    result: dict[str, Any] = {
        "seed": seed,
        "sources": {
            "unit_groups": group_path.relative_to(root).as_posix(),
            "branch_traces": branch_path.relative_to(root).as_posix(),
            "run_config": config_path.relative_to(root).as_posix(),
        },
        "source_sha256": {
            "unit_groups": sha256_file(group_path),
            "branch_traces": sha256_file(branch_path),
            "run_config": sha256_file(config_path),
        },
        "unit_group_rows": int(len(unit_groups)),
        "trial_count": len(trial_ids),
        "outputs": {},
    }
    for key, path, rows in (("panel_b", panel_b_path, panel_b_rows), ("panel_d", panel_d_path, panel_d_rows)):
        identity = expected[key]
        actual_hash = sha256_file(path)
        actual_bytes = path.stat().st_size
        result["outputs"][key] = {
            "target_path": identity["path"],
            "rows": int(rows),
            "actual_bytes": int(actual_bytes),
            "expected_bytes": int(identity["expected_bytes"]),
            "actual_sha256": actual_hash,
            "expected_sha256": identity["expected_sha256"],
            "bytes_match": int(actual_bytes) == int(identity["expected_bytes"]),
            "sha256_match": actual_hash == identity["expected_sha256"],
        }
    result["status"] = (
        "byte_identical_reconstruction_verified"
        if all(row["bytes_match"] and row["sha256_match"] for row in result["outputs"].values())
        else "reconstruction_mismatch"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct missing Fig.5 parent CSVs in disposable scratch and compare SHA-256.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--register",
        type=Path,
        default=Path("archive/move_ledgers/parent_artifact_gap_register_pre_restore_20260814.json"),
    )
    parser.add_argument("--seeds", default="1000", help="Comma/range syntax, e.g. 1000 or 1000-1019")
    parser.add_argument("--output", type=Path, default=Path("archive/move_ledgers/parent_artifact_reconstruction_validation_20260814.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    register_path = args.register if args.register.is_absolute() else root / args.register
    output_path = args.output if args.output.is_absolute() else root / args.output
    register = json.loads(register_path.read_text(encoding="utf-8-sig"))
    seeds = seed_list(args.seeds)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="net_torch_parent_reconstruction_") as temp_dir:
        temp_root = Path(temp_dir)
        for seed in seeds:
            seed_scratch = temp_root / f"seed_{seed}"
            seed_scratch.mkdir(parents=True, exist_ok=True)
            results.append(verify_seed(root, register, seed, seed_scratch))
    verified = sum(row["status"] == "byte_identical_reconstruction_verified" for row in results)
    payload = {
        "schema": "net_torch_parent_artifact_reconstruction_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "check_only_disposable_scratch",
        "repo_root": str(root),
        "register": register_path.relative_to(root).as_posix(),
        "register_sha256": sha256_file(register_path),
        "seeds_requested": seeds,
        "seed_count": len(results),
        "verified_seed_count": verified,
        "verified_output_count": sum(
            output["sha256_match"] and output["bytes_match"]
            for row in results
            for output in row["outputs"].values()
        ),
        "status": "completed_verified" if verified == len(results) else "completed_with_mismatch",
        "safety": {
            "canonical_parent_files_written": 0,
            "canonical_parent_files_deleted": 0,
            "protected_manuscripts_modified": 0,
            "scratch_policy": "system temporary directory removed after hashes were recorded",
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output={output_path}")
    print(f"status={payload['status']}")
    print(f"verified_seed_count={verified}/{len(results)}")
    print(f"verified_output_count={payload['verified_output_count']}/{2 * len(results)}")
    return 0 if payload["status"] == "completed_verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
