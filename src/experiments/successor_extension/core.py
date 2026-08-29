from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.experiments.common.inference import crossed_bootstrap_mean_ci, stable_seed
from src.experiments.c5_l2_successor_closure import (
    PRIMARY_ENDPOINTS,
    build_c_anchor_mapping,
    donor_transfer,
    summarize_c5_endpoints,
    _load_parent,
    _paired_history_indices,
    _run_prefix,
    _validate_history_pairs,
)
from src.experiments.paper_figures.fig2.artifacts import write_cache_key
from src.experiments.paper_figures.fig2.fixed_b_artifacts import (
    FixedBArtifact,
    load_fixed_b_artifact,
    save_fixed_b_artifact,
)
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_runtime import (
    _encode_source_rows,
    _history_rows_at_k,
    _load_boundary,
    _run_branch,
    _simulate_history_rows,
)
from src.experiments.paper_figures.fig2.run_task import _build_context, _resolve_model_path
from src.experiments.paper_figures.fig2.successor_replay import (
    FAST_STATE_KEYS,
    STSP_STATE_KEYS,
    audit_stsp_only_restore,
    capture_successor_transition,
    continue_successor_transition,
    correct_passive_successor_effects,
    prepare_layer2_stsp_transplant,
    repeat_boundary,
    snapshot_boundary_numpy,
)
from src.experiments.paper_figures.fig2.types import Fig2Config
from src.experiments.paper_figures.run_paper_figures import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH_GLOB,
)
from src.experiments.common.dataset import build_class_index
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.ping_common import LAYER_KEYS

EXPERIMENT_ID = "successor_extension"
SCHEMA_NAME = "successor_extension"
SCHEMA_VERSION = 1

FROZEN_CONFIRMATORY_ROOT = "results/multi_seed_rollout/fig2/fixed_b_mechanism_confirmatory"
FROZEN_PROTOCOL_DIR = f"{FROZEN_CONFIRMATORY_ROOT}/frozen_protocol"
FROZEN_PROTOCOL_TASK_ID = "fixed_b_frozen_protocol"

TASK_K10_SPECS = "k10_extension_specs"
TASK_K10_INPUT = "k10_extension_input_bank"
TASK_K10_HISTORY = "k10_history_bank"
TASK_EXP_A = "exp_a_c5_k10_successor"
TASK_EXP_B = "exp_b_k10_l1_overlap_intervention"
TASK_EXP_C = "exp_c_c5_twohop_cd"

K10 = 10
NUM_CLASSES = 10
NEAR_ZERO = 1e-12
OVERLAP_EPS = 1e-4
FROZEN_PROTOCOL_SEED = 20260724  # sealed fixed-B v4 protocol seed
HISTORY_CONDITIONS = ("A", "C")
EXTENSION_HISTORY_ROW_ID_BASE = 1000
EXTENSION_SEED_OFFSETS = {
    # new deterministic RNG namespaces; never reused by the frozen protocol
    "history_sim": 300_000,
    "exp_b_random_mask": 916_000,
    "exp_c_d_hop": 830_000,
}


@dataclass(frozen=True)
class ExtensionConfig:
    output_root: str = "results/successor_extension_v1_medium"
    frozen_root: str = FROZEN_CONFIRMATORY_ROOT
    frozen_protocol_dir: str = FROZEN_PROTOCOL_DIR
    dataset_root: str = DEFAULT_DATASET_ROOT
    model_path_glob: str = DEFAULT_MODEL_PATH_GLOB
    device: str = "auto"
    network_seed: int = 1000
    families: int = 6
    anchors: int = 20
    anchors_per_chunk: int = 5
    bootstrap_draws: int = 5000
    minimum_valid_coverage: float = 0.80
    minimum_positive_fraction: float = 0.55
    minimum_mean_transfer: float = 0.10
    smoke: bool = False


# --------------------------------------------------------------------------- #
# path / artifact helpers
# --------------------------------------------------------------------------- #

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_artifact_from_dir(task_dir: Path, *, task_id: str) -> FixedBArtifact:
    cache_path = Path(task_dir) / "cache_key.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Required parent cache key is missing: {cache_path}")
    wrapper = json.loads(cache_path.read_text(encoding="utf-8"))
    expected = wrapper.get("cache_key") if isinstance(wrapper, dict) else None
    if not isinstance(expected, dict):
        raise RuntimeError(f"Unreadable cache key at {task_dir}")
    return load_fixed_b_artifact(Path(task_dir), expected, task_id=task_id)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _parent_entry(task_dir: Path) -> dict[str, Any]:
    cache_path = Path(task_dir) / "cache_key.json"
    return {
        "path": str(Path(task_dir).resolve()),
        "cache_key_sha256": _sha256_file(cache_path) if cache_path.exists() else "missing",
    }


def _write_task_manifest(
    task_dir: Path,
    *,
    task_id: str,
    parents: Mapping[str, Mapping[str, Any]],
    params: Mapping[str, Any],
    inference_scope: str = "single_seed_cohort_unit",
) -> None:
    _write_json(
        Path(task_dir) / "task_manifest.json",
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "parents": {name: dict(entry) for name, entry in sorted(parents.items())},
            "params": dict(params),
            "inference_scope": str(inference_scope),
        },
    )


def _cache_key(
    task_id: str,
    *,
    network_seed: int,
    parents: Mapping[str, Mapping[str, Any]],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "network_seed": int(network_seed),
        "parents": {name: entry["cache_key_sha256"] for name, entry in sorted(parents.items())},
        "params": dict(params),
    }


def _build_ctx(cfg: ExtensionConfig, *, load_model: bool) -> Any:
    repo_root = _repo_root()
    model_path = _resolve_model_path(
        None, str(cfg.model_path_glob), int(cfg.network_seed), smoke=bool(cfg.smoke)
    )
    fig_cfg = Fig2Config(
        model_path=str(model_path),
        dataset_root=str(_resolve(repo_root, cfg.dataset_root)),
        output_root=str(_resolve(repo_root, cfg.output_root)),
        network_seed=int(cfg.network_seed),
        device=str(cfg.device),
        smoke=False,
    )
    return _build_context(fig_cfg, load_model=load_model)


def _load_frozen_protocol(cfg: ExtensionConfig) -> FixedBArtifact:
    return _load_artifact_from_dir(
        _resolve(_repo_root(), cfg.frozen_protocol_dir), task_id=FROZEN_PROTOCOL_TASK_ID
    )


def _load_frozen_seed_artifact(cfg: ExtensionConfig, task_name: str) -> FixedBArtifact:
    return _load_artifact_from_dir(
        _resolve(_repo_root(), cfg.frozen_root)
        / f"seed_{int(cfg.network_seed)}"
        / "data"
        / "intermediates"
        / task_name,
        task_id=task_name,
    )


def _seed_root(cfg: ExtensionConfig) -> Path:
    return _resolve(_repo_root(), cfg.output_root) / f"seed_{int(cfg.network_seed)}"


def _metric_summary_payload(
    cfg: ExtensionConfig, *, experiment: str, endpoints: Mapping[str, Any], extra: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "experiment_id": f"{EXPERIMENT_ID}.{experiment}",
        "status": "completed",
        "promotion_status": "single_seed_not_manuscript_evidence",
        "network_seed": int(cfg.network_seed),
        "seed_role": "confirmatory_cohort_seed",
        "families": int(cfg.families),
        "anchors": int(cfg.anchors),
        "endpoints": dict(endpoints),
        "claim_boundary": (
            f"Single network within the 20-seed confirmatory cohort (seed {int(cfg.network_seed)}). "
            "Within-network cell variation quantifies stability only; network-level inference is "
            "produced by the cohort aggregate, not by this per-seed summary. "
            "Sufficiency, not necessity, uniqueness, or complete mediation."
        ),
        **dict(extra),
    }


# --------------------------------------------------------------------------- #
# K=10 extension specs (portable, model-independent)
# --------------------------------------------------------------------------- #

def _extension_label(candidate_id: int, condition: str, position: int) -> int:
    """Deterministic suffix-label rule that continues the frozen family formula."""
    candidate_id = int(candidate_id)
    position = int(position)
    if condition == "A":
        return (candidate_id + 3 * position) % NUM_CLASSES
    if condition == "C":
        return (candidate_id + 5 + 7 * position) % NUM_CLASSES
    raise ValueError(f"Unknown history condition: {condition!r}")


def pick_suffix_images_for_families(
    families: pd.DataFrame,
    used_image_ids: set[int],
    class_pools: Mapping[int, Sequence[int]],
) -> dict[tuple[int, str], list[int]]:
    """Deterministic, collision-free image assignment for positions 6..10.

    Frozen first-5 images and all B anchors are excluded. Each class pool is
    consumed in ascending image order so the assignment is reproducible without
    extending the frozen protocol's take_image cursor.
    """
    pools = {int(label): [int(value) for value in values if int(value) not in used_image_ids]
             for label, values in class_pools.items()}
    output: dict[tuple[int, str], list[int]] = {}
    for family in families.sort_values("history_family_id").itertuples(index=False):
        for condition in HISTORY_CONDITIONS:
            key = (int(family.history_family_id), condition)
            suffix: list[int] = []
            for position in range(5, K10):
                label = _extension_label(int(family.candidate_family_id), condition, position)
                pool = pools[label]
                if not pool:
                    raise RuntimeError(f"Exhausted class-{label} pool for family {key} position {position}")
                image_id = pool.pop(0)
                suffix.append(int(image_id))
            output[key] = suffix
    return output


def build_k10_extension_specs(cfg: ExtensionConfig, dataset: Any) -> pd.DataFrame:
    """K=10 A/C history specs: frozen positions 1-5 + new deterministic positions 6-10."""
    protocol = _load_frozen_protocol(cfg)
    families = protocol.tables["history_families"].copy()
    b_specs = protocol.tables["b_anchor_specs"]
    used_ids: set[int] = set()
    for encoded in protocol.tables["history_specs"]["sequence_image_ids"]:
        used_ids.update(int(value) for value in json.loads(str(encoded)))
    used_ids.update(int(value) for value in b_specs["B_image_id"])

    class_pools = {label: list(class_index) for label, class_index in build_class_index(dataset, NUM_CLASSES).items()}
    suffix_by_family = pick_suffix_images_for_families(families, used_ids, class_pools)

    rows: list[dict[str, Any]] = []
    for family in families.sort_values("history_family_id").itertuples(index=False):
        for condition in HISTORY_CONDITIONS:
            frozen_ids = [int(value) for value in json.loads(str(getattr(family, f"{condition}_full_image_ids")))]
            frozen_labels = [int(value) for value in json.loads(str(getattr(family, f"{condition}_full_labels")))]
            suffix_ids = suffix_by_family[(int(family.history_family_id), condition)]
            sequence = frozen_ids + suffix_ids
            labels = [_extension_label(int(family.candidate_family_id), condition, pos) for pos in range(K10)]
            if labels[:5] != frozen_labels:
                raise RuntimeError(f"Suffix label formula disagrees with frozen prefix for family {family.history_family_id} {condition}")
            rows.append(
                {
                    "protocol_seed": FROZEN_PROTOCOL_SEED,
                    "history_row_id": EXTENSION_HISTORY_ROW_ID_BASE + len(rows),
                    "history_family_id": int(family.history_family_id),
                    "candidate_family_id": int(family.candidate_family_id),
                    "history_condition": condition,
                    "prefix_k": K10,
                    "sequence_image_ids": json.dumps(sequence, separators=(",", ":")),
                    "sequence_labels": json.dumps(labels, separators=(",", ":")),
                    "sequence_encoding_seeds": json.dumps(
                        [FROZEN_PROTOCOL_SEED + 100_000 + int(value) for value in sequence],
                        separators=(",", ":"),
                    ),
                    "elapsed_steps": K10 * (200 + 200),
                    "history_fold": int(family.balance_stratum),
                }
            )
    return pd.DataFrame(rows)


SPECS_TASK_PARAMS = {
    "frozen_families_reused": 10,
    "suffix_positions": [6, 7, 8, 9, 10],
}


def save_k10_extension_specs(cfg: ExtensionConfig, dataset: Any) -> pd.DataFrame:
    """Persist the portable K=10 A/C history specs with a content-addressed cache key.

    The input bank declares these specs as a parent, so the specs task must be
    addressable like every other successor-extension artifact.
    """
    specs = build_k10_extension_specs(cfg, dataset)
    task_dir = _resolve(_repo_root(), cfg.output_root) / TASK_K10_SPECS
    task_dir.mkdir(parents=True, exist_ok=True)
    specs.to_csv(task_dir / "history_specs.csv", index=False, lineterminator="\n")
    key = _cache_key(
        TASK_K10_SPECS,
        network_seed=0,
        parents={},
        params={
            **SPECS_TASK_PARAMS,
            "history_specs_sha256": _sha256_file(task_dir / "history_specs.csv"),
        },
    )
    write_cache_key(task_dir, key)
    _write_task_manifest(
        task_dir,
        task_id=TASK_K10_SPECS,
        parents={},
        params=key["params"],
        inference_scope="portable_input_specs",
    )
    return specs


# --------------------------------------------------------------------------- #
# K=10 extension input bank (portable; encoding depends only on frozen inputs)
# --------------------------------------------------------------------------- #

def build_k10_extension_input_bank(cfg: ExtensionConfig, ctx: Any) -> FixedBArtifact:
    protocol = _load_frozen_protocol(cfg)
    specs = pd.read_csv(_resolve(_repo_root(), cfg.output_root) / TASK_K10_SPECS / "history_specs.csv")
    used = set(int(value) for value in protocol.tables["history_input_manifest"]["image_id"])

    encode_rows: list[dict[str, Any]] = []
    suffix_seen: set[int] = set()
    for row in specs.sort_values("history_row_id").itertuples(index=False):
        sequence = [int(value) for value in json.loads(str(row.sequence_image_ids))]
        seeds = [int(value) for value in json.loads(str(row.sequence_encoding_seeds))]
        for image_id, encoding_seed in zip(sequence[5:], seeds[5:]):
            if int(image_id) in used or int(image_id) in suffix_seen:
                continue
            suffix_seen.add(int(image_id))
            encode_rows.append(
                {
                    "image_id": int(image_id),
                    "label": _extension_label(int(row.candidate_family_id), str(row.history_condition), sequence.index(int(image_id))),
                    "encoding_seed": int(encoding_seed),
                }
            )
    encode_frame = pd.DataFrame(encode_rows).sort_values("image_id").reset_index(drop=True)

    item_steps = int(ctx.cfg.fixed_b_item_steps)
    new_spikes = _encode_source_rows(
        ctx, encode_frame, image_column="image_id", seed_column="encoding_seed", steps=item_steps
    ).astype(np.bool_, copy=False)

    frozen_manifest = protocol.tables["history_input_manifest"].copy()
    frozen_manifest["row_index"] = frozen_manifest["row_index"].astype(int)
    base_index = int(len(frozen_manifest))
    new_rows: list[dict[str, Any]] = []
    for local, row in enumerate(encode_frame.itertuples(index=False)):
        new_rows.append(
            {
                "protocol_seed": FROZEN_PROTOCOL_SEED,
                "history_input_id": int(row.image_id),
                "image_id": int(row.image_id),
                "label": int(row.label),
                "image_sha256": "",
                "encoding_seed": int(row.encoding_seed),
                "storage_key": "history_spikes",
                "row_index": int(base_index + local),
                "shape": "x".join(str(value) for value in new_spikes[local].shape),
                "dtype": str(new_spikes[local].dtype),
                "tensor_sha256": "",
                "spike_count": int(new_spikes[local].sum()),
            }
        )
    combined_manifest = pd.concat([frozen_manifest, pd.DataFrame(new_rows)], ignore_index=True)
    combined_spikes = np.concatenate(
        [protocol.arrays["history_spikes"], np.asarray(new_spikes)], axis=0
    ).astype(np.bool_, copy=False)

    task_dir = _resolve(_repo_root(), cfg.output_root) / TASK_K10_INPUT
    specs_dir = _resolve(_repo_root(), cfg.output_root) / TASK_K10_SPECS
    if not (specs_dir / "cache_key.json").exists():
        raise FileNotFoundError(
            "K10 extension specs parent has no cache key; run the "
            f"{TASK_K10_SPECS} task first: {specs_dir / 'cache_key.json'}"
        )
    parents = {
        TASK_K10_SPECS: _parent_entry(specs_dir),
        "fixed_b_frozen_protocol": _parent_entry(_resolve(_repo_root(), cfg.frozen_protocol_dir)),
    }
    key = _cache_key(
        TASK_K10_INPUT,
        network_seed=0,
        parents=parents,
        params={"suffix_images": int(len(new_rows)), "encoding_rule": "protocol_seed_plus_100000_plus_image_id"},
    )
    artifact = save_fixed_b_artifact(
        task_dir,
        key,
        tables={
            "input_manifest": protocol.tables["input_manifest"].copy(),
            "history_input_manifest": combined_manifest,
        },
        arrays={
            "exact_b_spikes": np.asarray(protocol.arrays["exact_b_spikes"]),
            "history_spikes": combined_spikes,
        },
        payloads={
            "extension": {
                "suffix_image_count": int(len(new_rows)),
                "frozen_prefix_images_reused_byte_for_byte": True,
                "new_images_encoded_once": True,
            }
        },
    )
    _write_task_manifest(task_dir, task_id=TASK_K10_INPUT, parents=parents, params=key["params"])
    return artifact


# --------------------------------------------------------------------------- #
# K=10 history bank (per seed; K5 checkpoint identity audit against frozen bank)
# --------------------------------------------------------------------------- #

def _bitwise_compare_rows(first: np.ndarray, second: np.ndarray) -> tuple[bool, float]:
    left = np.ascontiguousarray(first)
    right = np.ascontiguousarray(second)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False, float("inf")
    equal = left.tobytes(order="C") == right.tobytes(order="C")
    diff = 0.0 if equal else float(np.abs(left.astype(np.float64) - right.astype(np.float64)).max())
    return equal, diff


def build_k10_history_bank(cfg: ExtensionConfig, ctx: Any) -> FixedBArtifact:
    """Simulate the 10-item A/C histories from t=0; audit the 5-item checkpoint
    against the frozen per-seed K=5 history bank (bitwise identity gate)."""
    ext_inputs = _load_artifact_from_dir(
        _resolve(_repo_root(), cfg.output_root) / TASK_K10_INPUT, task_id=TASK_K10_INPUT
    )
    frozen_k5_bank = _load_frozen_seed_artifact(cfg, "fixed_b_history_bank")
    specs = pd.read_csv(_resolve(_repo_root(), cfg.output_root) / TASK_K10_SPECS / "history_specs.csv")

    k5_specs = specs.copy()
    k5_specs["prefix_k"] = 5
    k5_specs["elapsed_steps"] = 5 * (200 + 200)
    k5_specs["sequence_image_ids"] = [
        json.dumps([int(v) for v in json.loads(str(encoded))[:5]], separators=(",", ":"))
        for encoded in specs["sequence_image_ids"]
    ]
    k5_specs["sequence_labels"] = [
        json.dumps([int(v) for v in json.loads(str(encoded))[:5]], separators=(",", ":"))
        for encoded in specs["sequence_labels"]
    ]
    k5_specs["sequence_encoding_seeds"] = [
        json.dumps([int(v) for v in json.loads(str(encoded))[:5]], separators=(",", ":"))
        for encoded in specs["sequence_encoding_seeds"]
    ]

    audit_arrays, _ = _simulate_history_rows(ctx, k5_specs, ext_inputs)
    k10_arrays, prestate_features = _simulate_history_rows(ctx, specs, ext_inputs)

    frozen_k5_specs = (
        frozen_k5_bank.tables["history_specs"]
        .loc[frozen_k5_bank.tables["history_specs"]["prefix_k"].eq(5)]
        .sort_values("history_row_id")
        .reset_index(drop=True)
    )
    frozen_index = {
        (int(row.history_family_id), str(row.history_condition)): int(index)
        for index, row in frozen_k5_specs.iterrows()
    }
    audit_rows: list[dict[str, Any]] = []
    my_rows = k5_specs.sort_values("history_row_id").reset_index(drop=True)
    all_exact = True
    for position, row in my_rows.iterrows():
        frozen_pos = frozen_index[(int(row.history_family_id), str(row.history_condition))]
        for layer in LAYER_KEYS:
            for state in FAST_STATE_KEYS + STSP_STATE_KEYS:
                mine = audit_arrays[f"k5__{layer}__{state}"][position]
                frozen = frozen_k5_bank.arrays[f"k5__{layer}__{state}"][frozen_pos]
                exact, max_diff = _bitwise_compare_rows(mine, frozen)
                all_exact = all_exact and exact
                audit_rows.append(
                    {
                        "network_seed": int(cfg.network_seed),
                        "history_family_id": int(row.history_family_id),
                        "history_condition": str(row.history_condition),
                        "layer": layer,
                        "state": state,
                        "bitwise_equal": int(exact),
                        "max_abs_diff": float(max_diff),
                    }
                )
    audit = pd.DataFrame(audit_rows)
    if not all_exact:
        raise RuntimeError(
            "K5 checkpoint identity audit failed: the 5-item checkpoint of the K=10 simulation "
            "does not reproduce the frozen K=5 history bank bitwise; K=10 parent bank rejected."
        )

    task_dir = _seed_root(cfg) / "data" / "intermediates" / TASK_K10_HISTORY
    parents = {
        TASK_K10_SPECS: _parent_entry(_resolve(_repo_root(), cfg.output_root) / TASK_K10_SPECS),
        TASK_K10_INPUT: _parent_entry(_resolve(_repo_root(), cfg.output_root) / TASK_K10_INPUT),
        "frozen_k5_history_bank": _parent_entry(
            _resolve(_repo_root(), cfg.frozen_root)
            / f"seed_{int(cfg.network_seed)}" / "data" / "intermediates" / "fixed_b_history_bank"
        ),
    }
    key = _cache_key(
        TASK_K10_HISTORY,
        network_seed=int(cfg.network_seed),
        parents=parents,
        params={"prefix_k": K10, "simulated_from_t0": True, "k5_checkpoint_bitwise_audit": "all_pass"},
    )
    artifact = save_fixed_b_artifact(
        task_dir,
        key,
        tables={
            "history_specs": specs.reset_index(drop=True),
            "b_anchor_specs": _load_frozen_protocol(cfg).tables["b_anchor_specs"].copy(),
            "prestate_features": prestate_features,
        },
        arrays={key_name: value for key_name, value in k10_arrays.items() if key_name.startswith("k10__")},
        payloads={
            "extension": {
                "k5_checkpoint_identity_audit": "all_bitwise_pass",
                "history_timing": "each item stimulus followed by matched zero-input delay",
            }
        },
    )
    _write_csv(_seed_root(cfg) / "data" / "metrics" / "k10_history_bank_k5_identity_audit.csv", audit)
    _write_task_manifest(task_dir, task_id=TASK_K10_HISTORY, parents=parents, params=key["params"])
    return artifact


# --------------------------------------------------------------------------- #
# Experiment A: C5 successor transfer at K=10 (K=1/5 not rerun)
# --------------------------------------------------------------------------- #

def _extension_screening_verdict(
    endpoint_summary: pd.DataFrame, identity: pd.DataFrame, prefixes: Sequence[int]
) -> dict[str, Any]:
    identity_pass = bool(not identity.empty and identity["identity_pass"].eq(1).all())
    complete = bool(
        len(endpoint_summary) == 2 * len(prefixes)
        and set(endpoint_summary["endpoint"]) == set(PRIMARY_ENDPOINTS)
        and set(endpoint_summary["prefix_k"].astype(int)) == set(int(value) for value in prefixes)
    )
    strong = bool(complete and identity_pass and endpoint_summary["screening_pass"].eq(1).all())
    directionally_positive = bool(
        complete
        and identity_pass
        and endpoint_summary["mean_transfer"].gt(0.0).all()
        and endpoint_summary["crossed_bootstrap_ci95_low"].gt(0.0).all()
    )
    if strong:
        verdict = "supported_in_development_seed"
    elif directionally_positive:
        verdict = "directionally_supported_below_prespecified_strength_gate"
    elif complete and identity_pass and endpoint_summary["mean_transfer"].gt(0.0).any():
        verdict = "mixed_single_seed_evidence"
    else:
        verdict = "not_supported_in_development_seed"
    return {
        "verdict": verdict,
        "all_identity_gates_pass": identity_pass,
        "all_endpoint_depth_cells_present": complete,
        "all_prespecified_screening_gates_pass": strong,
        "all_crossed_bootstrap_intervals_above_zero": directionally_positive,
        "inference_unit_warning": (
            "History-family and anchor resampling quantifies within-network stability only; "
            "independently trained networks are required for manuscript-level inference."
        ),
    }


def run_experiment_a(cfg: ExtensionConfig, ctx: Any) -> dict[str, Any]:
    ext_inputs = _load_artifact_from_dir(
        _resolve(_repo_root(), cfg.output_root) / TASK_K10_INPUT, task_id=TASK_K10_INPUT
    )
    k10_bank = _load_artifact_from_dir(
        _seed_root(cfg) / "data" / "intermediates" / TASK_K10_HISTORY, task_id=TASK_K10_HISTORY
    )
    c_map = build_c_anchor_mapping(k10_bank.tables["b_anchor_specs"])
    out_dir = _seed_root(cfg) / "data" / "metrics" / TASK_EXP_A
    _write_json(
        out_dir / "protocol_freeze.json",
        {
            "experiment_id": TASK_EXP_A,
            "prefix_k": K10,
            "intervention": (
                "Transplant only post-B Layer-2 u/x from the paired donor within each A/C history pair; "
                "preserve receiver Layer-1/3 u/x, equalize fast variables via stsp_only restore, "
                "present an identical C spike tensor (cyclic next class, same replicate)."
            ),
            "primary_endpoints": list(PRIMARY_ENDPOINTS),
            "passive_correction": "active C displacement minus duration-matched zero-input displacement",
            "k1_k5_published_results_reused_not_rerun": True,
        },
    )
    cells, audit = _run_prefix(
        ctx,
        inputs=ext_inputs,
        histories=k10_bank,
        c_map=c_map,
        prefix_k=K10,
        anchors_per_chunk=max(1, int(cfg.anchors_per_chunk)),
        max_anchors=max(1, int(cfg.anchors)),
        max_history_families=max(1, int(cfg.families)),
    )
    endpoint_summary = summarize_c5_endpoints(cells, cfg)
    verdict = _extension_screening_verdict(endpoint_summary, audit, prefixes=(K10,))
    _write_csv(out_dir / "c5_k10_cell_metrics.csv", cells)
    _write_csv(out_dir / "c5_k10_endpoint_summary.csv", endpoint_summary)
    _write_csv(out_dir / "c5_k10_identity_audit.csv", audit)
    _write_json(out_dir / "c5_k10_screening_verdict.json", verdict)
    parents = {
        TASK_K10_HISTORY: _parent_entry(_seed_root(cfg) / "data" / "intermediates" / TASK_K10_HISTORY),
        TASK_K10_INPUT: _parent_entry(_resolve(_repo_root(), cfg.output_root) / TASK_K10_INPUT),
    }
    _write_task_manifest(
        out_dir, task_id=TASK_EXP_A, parents=parents,
        params={"prefix_k": K10, "families": cfg.families, "anchors": cfg.anchors},
    )
    summary = _metric_summary_payload(
        cfg,
        experiment=TASK_EXP_A,
        endpoints={
            row.endpoint: {
                "mean_transfer": float(row.mean_transfer),
                "ci95_low": float(row.crossed_bootstrap_ci95_low),
                "ci95_high": float(row.crossed_bootstrap_ci95_high),
                "positive_fraction": float(row.positive_fraction),
                "valid_coverage": float(row.valid_coverage),
                "n_valid_cells": int(row.n_valid_cells),
                "screening_pass": int(row.screening_pass),
            }
            for row in endpoint_summary.itertuples(index=False)
        },
        extra={"verdict": verdict, "n_cells": int(len(cells)), "n_chunks": int(audit["chunk_id"].nunique())},
    )
    _write_json(out_dir / "summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# Experiment B: K=10 Layer-1 overlap-targeted u/x reset (pre-B only)
# --------------------------------------------------------------------------- #

def build_overlap_masks(
    u: np.ndarray,
    x: np.ndarray,
    support: np.ndarray,
    rng: np.random.Generator,
    *,
    baseline: float = 0.2,
    eps: float = OVERLAP_EPS,
) -> dict[str, np.ndarray | int]:
    """Outcome-blind L1 intervention masks.

    overlap    = history-deviated STSP sites (|u*x - U| > eps) that lie under the
                 incoming input support.
    nonoverlap = history-deviated STSP sites outside the input support.
    random     = |overlap|-sized disjoint sample from the remaining deviated sites.
    """
    u = np.asarray(u, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    support = np.asarray(support, dtype=bool)
    if u.shape != support.shape or x.shape != support.shape:
        raise ValueError("u/x and support must share spatial shape")
    deviated = np.abs(u * x - float(baseline)) > float(eps)
    overlap = deviated & support
    nonoverlap = deviated & ~support
    pool = np.argwhere(deviated & ~overlap)
    target = int(overlap.sum())
    random_mask = np.zeros_like(support, dtype=bool)
    insufficient = 0
    if target > 0:
        if len(pool) < target:
            insufficient = 1
        else:
            chosen = rng.choice(len(pool), size=target, replace=False)
            random_mask[tuple(pool[chosen].T)] = True
    return {
        "overlap": overlap,
        "nonoverlap": nonoverlap,
        "random": random_mask,
        "overlap_units": int(overlap.sum()),
        "nonoverlap_units": int(nonoverlap.sum()),
        "random_units": int(random_mask.sum()),
        "deviated_units": int(deviated.sum()),
        "insufficient_random": insufficient,
        "overlap_empty": int(target == 0),
    }


def _perturb_boundary_row(
    boundary: Mapping[str, Mapping[str, np.ndarray]],
    row_index: int,
    mask: np.ndarray | None,
    *,
    baseline_u: float = 0.2,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for layer in LAYER_KEYS:
        out[layer] = {}
        for state in FAST_STATE_KEYS + STSP_STATE_KEYS:
            value = boundary[layer][state][row_index]
            if layer == "layer1" and state in STSP_STATE_KEYS and mask is not None:
                value = value.copy()
                if state == "u":
                    value[mask] = baseline_u
                else:
                    value[mask] = 1.0
            out[layer][state] = value
    return out


def _history_contrast_attenuation(
    vector_a: np.ndarray, vector_c: np.ndarray, d0: np.ndarray
) -> tuple[float, float, float]:
    d = (vector_a - vector_c).astype(np.float64)
    denominator = float(np.dot(d0, d0))
    if denominator <= NEAR_ZERO:
        return float("nan"), float("nan"), denominator
    retention = float(np.dot(d, d0)) / denominator
    return 1.0 - retention, retention, denominator


def run_experiment_b(cfg: ExtensionConfig, ctx: Any) -> dict[str, Any]:
    ext_inputs = _load_artifact_from_dir(
        _resolve(_repo_root(), cfg.output_root) / TASK_K10_INPUT, task_id=TASK_K10_INPUT
    )
    k10_bank = _load_artifact_from_dir(
        _seed_root(cfg) / "data" / "intermediates" / TASK_K10_HISTORY, task_id=TASK_K10_HISTORY
    )
    out_dir = _seed_root(cfg) / "data" / "metrics" / TASK_EXP_B
    _write_json(
        out_dir / "protocol_freeze.json",
        {
            "experiment_id": TASK_EXP_B,
            "prefix_k": K10,
            "intervention": (
                "Pre-B only: reset Layer-1 u/x to baseline on outcome-blind masks derived from "
                "pre-B |u*x-U| and the frozen B spike support. L1/L2/L3 STSP stay dynamic during B. "
                "No Fig.4 frozen L2/L3 probe semantics."
            ),
            "conditions": ["intact", "overlap_reset", "nonoverlap_reset", "random_matched_reset"],
            "mask_baseline": 0.2,
            "mask_eps": OVERLAP_EPS,
            "primary_endpoints": [
                "early_layer2_b_history_contrast_attenuation",
                "post_b_layer2_ux_history_contrast_attenuation",
            ],
        },
    )
    rows = _history_rows_at_k(k10_bank.tables["history_specs"], K10)
    selected = rows.loc[rows["history_condition"].isin(HISTORY_CONDITIONS)].copy()
    families = sorted(int(value) for value in selected["history_family_id"].unique())[: int(cfg.families)]
    selected = selected.loc[selected["history_family_id"].isin(families)].reset_index(drop=True)
    _validate_history_pairs(selected)
    k10_specs_sorted = k10_bank.tables["history_specs"].sort_values("history_row_id").reset_index(drop=True)
    position_map = {int(row.history_row_id): int(position) for position, row in k10_specs_sorted.iterrows()}
    row_indices = [position_map[int(value)] for value in selected["history_row_id"]]
    boundary = _load_boundary(k10_bank, K10, row_indices=row_indices)
    elapsed = sorted(int(value) for value in selected["elapsed_steps"].unique())
    if len(elapsed) != 1:
        raise RuntimeError(f"Non-unique elapsed_steps for K=10: {elapsed}")
    current_time = int(elapsed[0])

    exact_inputs = np.asarray(ext_inputs.arrays["exact_b_spikes"], dtype=np.bool_)
    anchor_ids = sorted(int(value) for value in k10_bank.tables["b_anchor_specs"]["b_anchor_id"])[: int(cfg.anchors)]
    spatial_shape = tuple(int(value) for value in exact_inputs.shape[2:])
    n_hist = int(len(selected))
    conditions = ("intact", "overlap_reset", "nonoverlap_reset", "random_matched_reset")

    cell_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    for chunk_id, start in enumerate(range(0, len(anchor_ids), int(cfg.anchors_per_chunk))):
        chunk_anchor_ids = anchor_ids[start : start + int(cfg.anchors_per_chunk)]
        batch_rows: list[dict[str, Any]] = []
        batch_boundaries: list[dict[str, dict[str, np.ndarray]]] = []
        batch_inputs: list[np.ndarray] = []
        for anchor_id in chunk_anchor_ids:
            b_row = exact_inputs[int(anchor_id)]
            support = b_row.any(axis=0)
            for history_index in range(n_hist):
                history = selected.iloc[history_index]
                u = boundary["layer1"]["u"][history_index]
                x = boundary["layer1"]["x"][history_index]
                rng = np.random.default_rng(
                    int(cfg.network_seed) + EXTENSION_SEED_OFFSETS["exp_b_random_mask"]
                    + 10_000 * int(anchor_id) + history_index
                )
                masks = build_overlap_masks(u, x, support, rng)
                mask_rows.append(
                    {
                        "network_seed": int(cfg.network_seed),
                        "history_family_id": int(history["history_family_id"]),
                        "history_condition": str(history["history_condition"]),
                        "b_anchor_id": int(anchor_id),
                        "overlap_units": int(masks["overlap_units"]),
                        "nonoverlap_units": int(masks["nonoverlap_units"]),
                        "random_units": int(masks["random_units"]),
                        "deviated_units": int(masks["deviated_units"]),
                        "insufficient_random": int(masks["insufficient_random"]),
                        "overlap_empty": int(masks["overlap_empty"]),
                    }
                )
                for condition in conditions:
                    mask = {
                        "intact": None,
                        "overlap_reset": masks["overlap"],
                        "nonoverlap_reset": masks["nonoverlap"],
                        "random_matched_reset": masks["random"],
                    }[condition]
                    batch_boundaries.append(_perturb_boundary_row(boundary, history_index, mask))
                    batch_inputs.append(b_row)
                    batch_rows.append(
                        {
                            "history_family_id": int(history["history_family_id"]),
                            "history_condition": str(history["history_condition"]),
                            "b_anchor_id": int(anchor_id),
                            "condition": condition,
                        }
                    )
        batch_boundary = {
            layer: {state: np.stack([row[layer][state] for row in batch_boundaries], axis=0) for state in batch_boundaries[0][layer]}
            for layer in batch_boundaries[0]
        }
        b_input = torch.as_tensor(np.stack(batch_inputs, axis=0), device=ctx.device)
        base_seed = int(cfg.network_seed) + 914_000 + 10_000 * K10 + chunk_id
        early_active = capture_successor_transition(
            ctx, boundary=batch_boundary, input_seq=b_input, current_time=current_time,
            passive=False, random_seed=base_seed + 0,
        )
        early_passive = capture_successor_transition(
            ctx, boundary=batch_boundary, input_seq=b_input, current_time=current_time,
            passive=True, random_seed=base_seed + 1,
        )
        early_corrected = correct_passive_successor_effects(early_active, early_passive)[
            "early_layer2_event_map"
        ]
        free = _run_branch(
            ctx, boundary=batch_boundary, input_seq=b_input, current_time=current_time,
            restore_mode="stsp_only", branch="free", replay_l1_pooled=None,
            capture_l1_pooled=False, capture_strong_path=False, random_seed=base_seed + 2,
        )
        passive = _run_branch(
            ctx, boundary=batch_boundary, input_seq=b_input, current_time=current_time,
            restore_mode="stsp_only", branch="passive", replay_l1_pooled=None,
            capture_l1_pooled=False, capture_strong_path=False, random_seed=base_seed + 3,
        )
        l2_corrected = (free["layer2_ux"] - free["layer2_ux_pre"]) - (passive["layer2_ux"] - passive["layer2_ux_pre"])

        index_lookup: dict[tuple[int, str, str], int] = {}
        for local, meta in enumerate(batch_rows):
            index_lookup[(meta["history_family_id"], meta["history_condition"], meta["condition"])] = local
        endpoints = {
            "early_layer2_b_history_contrast_attenuation": early_corrected.reshape(len(batch_rows), -1),
            "post_b_layer2_ux_history_contrast_attenuation": l2_corrected.reshape(len(batch_rows), -1),
        }
        for family_id in families:
            for anchor_id in chunk_anchor_ids:
                d0_by_endpoint: dict[str, np.ndarray] = {}
                atten: dict[str, dict[str, float]] = {}
                for endpoint, vectors in endpoints.items():
                    y_a_intact = vectors[index_lookup[(family_id, "A", "intact")]].astype(np.float64)
                    y_c_intact = vectors[index_lookup[(family_id, "C", "intact")]].astype(np.float64)
                    d0 = y_a_intact - y_c_intact
                    d0_by_endpoint[endpoint] = d0
                    atten[endpoint] = {}
                    for condition in ("overlap_reset", "nonoverlap_reset", "random_matched_reset"):
                        y_a = vectors[index_lookup[(family_id, "A", condition)]].astype(np.float64)
                        y_c = vectors[index_lookup[(family_id, "C", condition)]].astype(np.float64)
                        value, retention, _ = _history_contrast_attenuation(y_a, y_c, d0)
                        atten[endpoint][condition] = value
                for endpoint in endpoints:
                    atten_values = atten[endpoint]
                    margin = float(atten_values["overlap_reset"] - max(atten_values["nonoverlap_reset"], atten_values["random_matched_reset"]))
                    cell_rows.append(
                        {
                            "network_seed": int(cfg.network_seed),
                            "prefix_k": K10,
                            "endpoint": endpoint,
                            "history_family_id": int(family_id),
                            "b_anchor_id": int(anchor_id),
                            "attenuation_intact": 0.0,
                            "attenuation_overlap_reset": atten_values["overlap_reset"],
                            "attenuation_nonoverlap_reset": atten_values["nonoverlap_reset"],
                            "attenuation_random_matched_reset": atten_values["random_matched_reset"],
                            "overlap_specific_margin": margin,
                            "d0_norm_sq": float(np.dot(d0_by_endpoint[endpoint], d0_by_endpoint[endpoint])),
                            "valid": int(np.isfinite(margin)),
                        }
                    )

    cells = pd.DataFrame(cell_rows)
    mask_audit = pd.DataFrame(mask_rows)
    summary_rows: list[dict[str, Any]] = []
    for endpoint, part in cells.groupby("endpoint"):
        valid = part.loc[part["valid"].eq(1)].copy()
        margins = valid["overlap_specific_margin"].to_numpy(dtype=np.float64)
        ci_low, ci_high = (
            crossed_bootstrap_mean_ci(
                margins,
                valid["history_family_id"].to_numpy(dtype=np.int64),
                valid["b_anchor_id"].to_numpy(dtype=np.int64),
                draws=int(cfg.bootstrap_draws),
                seed=stable_seed(endpoint),
            )
            if len(valid) else (float("nan"), float("nan"))
        )
        summary_rows.append(
            {
                "network_seed": int(cfg.network_seed),
                "endpoint": endpoint,
                "n_cells": int(len(part)),
                "n_valid_cells": int(len(valid)),
                "valid_coverage": float(len(valid) / max(1, len(part))),
                "mean_overlap_attenuation": float(valid["attenuation_overlap_reset"].mean()) if len(valid) else float("nan"),
                "mean_nonoverlap_attenuation": float(valid["attenuation_nonoverlap_reset"].mean()) if len(valid) else float("nan"),
                "mean_random_attenuation": float(valid["attenuation_random_matched_reset"].mean()) if len(valid) else float("nan"),
                "mean_overlap_specific_margin": float(margins.mean()) if len(margins) else float("nan"),
                "positive_margin_fraction": float(np.mean(margins > 0)) if len(margins) else float("nan"),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "inference_scope": "within_network_crossed_history_family_by_anchor_stability_only",
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    _write_csv(out_dir / "exp_b_cell_metrics.csv", cells)
    _write_csv(out_dir / "exp_b_mask_audit.csv", mask_audit)
    _write_csv(out_dir / "exp_b_network_summary.csv", summary_frame)
    parents = {
        TASK_K10_HISTORY: _parent_entry(_seed_root(cfg) / "data" / "intermediates" / TASK_K10_HISTORY),
        TASK_K10_INPUT: _parent_entry(_resolve(_repo_root(), cfg.output_root) / TASK_K10_INPUT),
    }
    _write_task_manifest(out_dir, task_id=TASK_EXP_B, parents=parents, params={"prefix_k": K10, "families": cfg.families, "anchors": cfg.anchors})
    summary = _metric_summary_payload(
        cfg,
        experiment=TASK_EXP_B,
        endpoints={
            row.endpoint: {
                "mean_overlap_specific_margin": float(row.mean_overlap_specific_margin),
                "positive_margin_fraction": float(row.positive_margin_fraction),
                "mean_overlap_attenuation": float(row.mean_overlap_attenuation),
                "mean_nonoverlap_attenuation": float(row.mean_nonoverlap_attenuation),
                "mean_random_attenuation": float(row.mean_random_attenuation),
                "ci95_low": float(row.bootstrap_ci95_low),
                "ci95_high": float(row.bootstrap_ci95_high),
                "valid_coverage": float(row.valid_coverage),
                "n_valid_cells": int(row.n_valid_cells),
            }
            for row in summary_frame.itertuples(index=False)
        },
        extra={"n_cells": int(len(cells)), "mask_unit_counts": {
            "mean_overlap_units": float(mask_audit["overlap_units"].mean()),
            "mean_nonoverlap_units": float(mask_audit["nonoverlap_units"].mean()),
            "insufficient_random_cells": int(mask_audit["insufficient_random"].sum()),
            "overlap_empty_cells": int(mask_audit["overlap_empty"].sum()),
        }},
    )
    _write_json(out_dir / "summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# Experiment C: two-hop propagation at K=5 (B -> C -> D, single transplant)
# --------------------------------------------------------------------------- #

def build_d_anchor_mapping(b_specs: pd.DataFrame) -> pd.DataFrame:
    required = {"b_anchor_id", "B_image_id", "B_label", "B_replicate_id"}
    missing = sorted(required.difference(b_specs.columns))
    if missing:
        raise ValueError(f"b_anchor_specs missing columns: {missing}")
    lookup = {
        (int(row.B_label), int(row.B_replicate_id)): row
        for row in b_specs.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for row in b_specs.sort_values("b_anchor_id").itertuples(index=False):
        d_key = ((int(row.B_label) + 2) % NUM_CLASSES, int(row.B_replicate_id))
        if d_key not in lookup:
            raise ValueError(f"Missing deterministic D anchor for key={d_key}")
        target = lookup[d_key]
        rows.append(
            {
                "b_anchor_id": int(row.b_anchor_id),
                "B_label": int(row.B_label),
                "d_anchor_id": int(target.b_anchor_id),
                "D_image_id": int(target.B_image_id),
                "D_label": int(target.B_label),
                "mapping_rule": "cyclic_plus_two_same_replicate",
            }
        )
    mapping = pd.DataFrame(rows)
    if mapping["d_anchor_id"].nunique() != len(mapping):
        raise RuntimeError("D anchor mapping is not one-to-one")
    if mapping["B_label"].eq(mapping["D_label"]).any():
        raise RuntimeError("D anchor mapping contains same-label pairs")
    return mapping


def run_experiment_c(cfg: ExtensionConfig, ctx: Any) -> dict[str, Any]:
    frozen_k5_bank = _load_frozen_seed_artifact(cfg, "fixed_b_history_bank")
    frozen_input_bank = _load_frozen_seed_artifact(cfg, "fixed_b_input_bank")
    out_dir = _seed_root(cfg) / "data" / "metrics" / TASK_EXP_C
    b_specs = frozen_k5_bank.tables["b_anchor_specs"]
    c_map = build_c_anchor_mapping(b_specs)
    d_map = build_d_anchor_mapping(b_specs)
    _write_json(
        out_dir / "protocol_freeze.json",
        {
            "experiment_id": TASK_EXP_C,
            "prefix_k": 5,
            "intervention": (
                "Single post-B Layer-2 u/x donor transplant per A/C pair; then identical C and D "
                "without any further transplant, stsp_only restore or fast-state equalization between C and D. "
                "Active-D and passive-D branches are exact clones of the full post-C boundary. "
                "Only the Layer-3 per-item decision timer is re-initialized before D, as in every probe. "
                "D = (B_label + 2) mod 10, same replicate."
            ),
            "primary_endpoints": [
                "early_layer2_event_map_donor_transfer_at_D",
                "layer3_post_D_ux_donor_transfer",
            ],
            "secondary_attribution_branch": "stsp_only continuation from post-C boundary (STSP-only carry-forward)",
            "post_c_endpoints_are_gates_not_primary": True,
        },
    )
    rows = _history_rows_at_k(frozen_k5_bank.tables["history_specs"], 5)
    selected = rows.loc[rows["history_condition"].isin(HISTORY_CONDITIONS)].copy()
    families = sorted(int(value) for value in selected["history_family_id"].unique())[: int(cfg.families)]
    selected = selected.loc[selected["history_family_id"].isin(families)].reset_index(drop=True)
    _validate_history_pairs(selected)
    frozen_k5_specs = (
        frozen_k5_bank.tables["history_specs"]
        .loc[frozen_k5_bank.tables["history_specs"]["prefix_k"].eq(5)]
        .sort_values("history_row_id")
        .reset_index(drop=True)
    )
    position_map = {int(row.history_row_id): int(position) for position, row in frozen_k5_specs.iterrows()}
    row_indices = [position_map[int(value)] for value in selected["history_row_id"]]
    history_boundary = _load_boundary(frozen_k5_bank, 5, row_indices=row_indices)
    elapsed = sorted(int(value) for value in selected["elapsed_steps"].unique())
    if len(elapsed) != 1:
        raise RuntimeError(f"Non-unique elapsed_steps for K=5: {elapsed}")
    current_time = int(elapsed[0])

    exact_inputs = np.asarray(frozen_input_bank.arrays["exact_b_spikes"], dtype=np.bool_)
    spatial_shape = tuple(int(value) for value in exact_inputs.shape[2:])
    mapping = c_map.sort_values("b_anchor_id").reset_index(drop=True)
    anchor_ids = [int(value) for value in mapping["b_anchor_id"]][: int(cfg.anchors)]
    mapping_by_anchor = mapping.set_index("b_anchor_id", drop=False)
    d_by_anchor = d_map.set_index("b_anchor_id", drop=False)
    history_count = int(len(selected))
    local_donor_indices = _paired_history_indices(selected)

    cell_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for chunk_id, start in enumerate(range(0, len(anchor_ids), int(cfg.anchors_per_chunk))):
        chunk_anchor_ids = anchor_ids[start : start + int(cfg.anchors_per_chunk)]
        anchor_count = int(len(chunk_anchor_ids))
        cell_count = int(anchor_count * history_count)
        repeated_history = repeat_boundary(history_boundary, anchor_count)
        b_input = np.repeat(
            exact_inputs[np.asarray(chunk_anchor_ids, dtype=np.int64)], history_count, axis=0
        )
        _run_branch(
            ctx,
            boundary=repeated_history,
            input_seq=torch.as_tensor(b_input, device=ctx.device),
            current_time=current_time,
            restore_mode="full_boundary",
            branch="free",
            replay_l1_pooled=None,
            capture_l1_pooled=False,
            capture_strong_path=False,
            random_seed=int(ctx.cfg.network_seed) + 810_000 + 10_000 * 5 + chunk_id,
        )
        post_b = snapshot_boundary_numpy(ctx.net)
        donor_indices = np.concatenate(
            [local_donor_indices + anchor_index * history_count for anchor_index in range(anchor_count)]
        ).astype(np.int64, copy=False)
        conditions, slices, transplant_audit = prepare_layer2_stsp_transplant(
            post_b,
            donor_indices,
        )
        mix_exact = transplant_audit["layer2_only_mix_exact"]
        sham_boundary_exact = transplant_audit["own_sham_boundary_exact"]
        restore_audit = audit_stsp_only_restore(ctx, conditions, input_shape=spatial_shape)
        c_anchor_ids = [int(mapping_by_anchor.loc[anchor_id, "c_anchor_id"]) for anchor_id in chunk_anchor_ids]
        c_input = np.repeat(
            exact_inputs[np.asarray(c_anchor_ids, dtype=np.int64)], history_count, axis=0
        )
        combined_c = np.concatenate([c_input, c_input, c_input], axis=0)
        c_tensor_identical = len(
            {_array_sha256(combined_c[index * cell_count : (index + 1) * cell_count]) for index in range(3)}
        ) == 1
        probe_time = current_time + int(ctx.cfg.fixed_b_stimulus_steps) + int(ctx.cfg.fixed_b_post_steps)
        c_result = capture_successor_transition(
            ctx,
            boundary=conditions,
            input_seq=torch.as_tensor(combined_c, device=ctx.device),
            current_time=probe_time,
            passive=False,
            random_seed=int(ctx.cfg.network_seed) + 820_000 + 10_000 * 5 + chunk_id,
        )
        post_c_full = snapshot_boundary_numpy(ctx.net)
        c_zero = capture_successor_transition(
            ctx,
            boundary=conditions,
            input_seq=torch.as_tensor(combined_c, device=ctx.device),
            current_time=probe_time,
            passive=True,
            random_seed=int(ctx.cfg.network_seed) + 821_000 + 10_000 * 5 + chunk_id,
        )
        corrected = correct_passive_successor_effects(c_result, c_zero)
        native_l2 = corrected["early_layer2_event_map"][slices["native"]]
        swap_l2 = corrected["early_layer2_event_map"][slices["layer2_swap"]]
        sham_l2 = corrected["early_layer2_event_map"][slices["own_sham"]]
        native_l3 = corrected["layer3_successor_ux"][slices["native"]]
        swap_l3 = corrected["layer3_successor_ux"][slices["layer2_swap"]]
        sham_l3 = corrected["layer3_successor_ux"][slices["own_sham"]]
        donor_l2 = native_l2[donor_indices]
        donor_l3 = native_l3[donor_indices]
        gate_l2_transfer, gate_l2_valid = donor_transfer(swap_l2, native_l2, donor_l2)
        gate_l3_transfer, gate_l3_valid = donor_transfer(swap_l3, native_l3, donor_l3)
        sham_c_max = float(max(np.abs(sham_l2 - native_l2).max(), np.abs(sham_l3 - native_l3).max()))

        # --- D phase: exact clone of the full post-C boundary; no re-transplant ---
        d_anchor_ids = [int(d_by_anchor.loc[anchor_id, "d_anchor_id"]) for anchor_id in chunk_anchor_ids]
        d_input = np.repeat(
            exact_inputs[np.asarray(d_anchor_ids, dtype=np.int64)], history_count, axis=0
        )
        combined_d = np.concatenate([d_input, d_input, d_input], axis=0)
        d_tensor_identical = len(
            {_array_sha256(combined_d[index * cell_count : (index + 1) * cell_count]) for index in range(3)}
        ) == 1
        d_time = probe_time + int(ctx.cfg.fixed_b_stimulus_steps) + int(ctx.cfg.fixed_b_post_steps)
        base_seed = int(ctx.cfg.network_seed) + EXTENSION_SEED_OFFSETS["exp_c_d_hop"] + 10_000 * 5 + chunk_id
        d_result = continue_successor_transition(
            ctx, boundary=post_c_full, input_seq=torch.as_tensor(combined_d, device=ctx.device),
            current_time=d_time, passive=False, random_seed=base_seed + 0, restore_mode="full_boundary",
        )
        d_zero = continue_successor_transition(
            ctx, boundary=post_c_full, input_seq=torch.as_tensor(combined_d, device=ctx.device),
            current_time=d_time, passive=True, random_seed=base_seed + 1, restore_mode="full_boundary",
        )
        d_secondary = continue_successor_transition(
            ctx, boundary=post_c_full, input_seq=torch.as_tensor(combined_d, device=ctx.device),
            current_time=d_time, passive=False, random_seed=base_seed + 2, restore_mode="stsp_only",
        )
        d_secondary_zero = continue_successor_transition(
            ctx, boundary=post_c_full, input_seq=torch.as_tensor(combined_d, device=ctx.device),
            current_time=d_time, passive=True, random_seed=base_seed + 3, restore_mode="stsp_only",
        )
        d_corrected = correct_passive_successor_effects(d_result, d_zero)
        d_secondary_corrected = correct_passive_successor_effects(d_secondary, d_secondary_zero)

        d_native_l2 = d_corrected["early_layer2_event_map"][slices["native"]]
        d_swap_l2 = d_corrected["early_layer2_event_map"][slices["layer2_swap"]]
        d_sham_l2 = d_corrected["early_layer2_event_map"][slices["own_sham"]]
        d_native_l3 = d_corrected["layer3_successor_ux"][slices["native"]]
        d_swap_l3 = d_corrected["layer3_successor_ux"][slices["layer2_swap"]]
        d_sham_l3 = d_corrected["layer3_successor_ux"][slices["own_sham"]]
        sham_d_max = float(max(np.abs(d_sham_l2 - d_native_l2).max(), np.abs(d_sham_l3 - d_native_l3).max()))

        d_l2_transfer, d_l2_valid = donor_transfer(d_swap_l2, d_native_l2, d_native_l2[donor_indices])
        d_l3_transfer, d_l3_valid = donor_transfer(d_swap_l3, d_native_l3, d_native_l3[donor_indices])
        s_l2 = d_secondary_corrected["early_layer2_event_map"]
        s_l3 = d_secondary_corrected["layer3_successor_ux"]
        s_native_l2, s_swap_l2v = s_l2[slices["native"]], s_l2[slices["layer2_swap"]]
        s_native_l3, s_swap_l3v = s_l3[slices["native"]], s_l3[slices["layer2_swap"]]
        sec_l2_transfer, sec_l2_valid = donor_transfer(s_swap_l2v, s_native_l2, s_native_l2[donor_indices])
        sec_l3_transfer, sec_l3_valid = donor_transfer(s_swap_l3v, s_native_l3, s_native_l3[donor_indices])

        fast_residual = _fast_state_residual(post_c_full, slices, donor_indices)
        identity_pass = bool(
            mix_exact
            and sham_boundary_exact
            and restore_audit["all_stsp_exact"]
            and restore_audit["fast_state_uniform"]
            and c_tensor_identical
            and d_tensor_identical
            and sham_c_max == 0.0
            and sham_d_max == 0.0
        )
        audit_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "chunk_id": int(chunk_id),
                "layer2_only_mix_exact": int(mix_exact),
                "own_sham_boundary_exact": int(sham_boundary_exact),
                "stsp_restore_exact": int(restore_audit["all_stsp_exact"]),
                "fast_state_uniform_after_reset": int(restore_audit["fast_state_uniform"]),
                "C_tensor_identical_across_conditions": int(c_tensor_identical),
                "D_tensor_identical_across_conditions": int(d_tensor_identical),
                "own_sham_output_exact_at_C": int(sham_c_max == 0.0),
                "own_sham_output_exact_at_D": int(sham_d_max == 0.0),
                "sham_c_max_abs": sham_c_max,
                "sham_d_max_abs": sham_d_max,
                "postC_fast_residual_l2_vmem_max_abs": fast_residual["layer2_vmem"],
                "postC_fast_residual_l1_vmem_max_abs": fast_residual["layer1_vmem"],
                "identity_pass": int(identity_pass),
            }
        )
        for local_anchor_index, b_anchor_id in enumerate(chunk_anchor_ids):
            map_row = mapping_by_anchor.loc[int(b_anchor_id)]
            d_row = d_by_anchor.loc[int(b_anchor_id)]
            for history_index, history in selected.iterrows():
                index = int(local_anchor_index * history_count + int(history_index))
                donor_history = selected.iloc[int(local_donor_indices[int(history_index)])]
                cell_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "prefix_k": 5,
                        "history_family_id": int(history["history_family_id"]),
                        "receiver_history_condition": str(history["history_condition"]),
                        "donor_history_condition": str(donor_history["history_condition"]),
                        "b_anchor_id": int(b_anchor_id),
                        "B_label": int(map_row["B_label"]),
                        "c_anchor_id": int(map_row["c_anchor_id"]),
                        "C_label": int(map_row["C_label"]),
                        "d_anchor_id": int(d_row["d_anchor_id"]),
                        "D_label": int(d_row["D_label"]),
                        "gate_early_layer2_C_donor_transfer": float(gate_l2_transfer[index]),
                        "gate_layer3_postC_donor_transfer": float(gate_l3_transfer[index]),
                        "gate_transfer_valid": int(gate_l2_valid[index] and gate_l3_valid[index]),
                        "early_layer2_D_donor_transfer": float(d_l2_transfer[index]),
                        "early_layer2_D_transfer_valid": int(d_l2_valid[index]),
                        "layer3_postD_ux_donor_transfer": float(d_l3_transfer[index]),
                        "layer3_postD_ux_transfer_valid": int(d_l3_valid[index]),
                        "secondary_stsp_only_early_layer2_D_donor_transfer": float(sec_l2_transfer[index]),
                        "secondary_stsp_only_layer3_postD_ux_donor_transfer": float(sec_l3_transfer[index]),
                        "secondary_transfer_valid": int(sec_l2_valid[index] and sec_l3_valid[index]),
                    }
                )
    cells = pd.DataFrame(cell_rows)
    audit = pd.DataFrame(audit_rows)
    summary_rows: list[dict[str, Any]] = []
    for endpoint, valid_column in (
        ("early_layer2_D_donor_transfer", "early_layer2_D_transfer_valid"),
        ("layer3_postD_ux_donor_transfer", "layer3_postD_ux_transfer_valid"),
        ("secondary_stsp_only_early_layer2_D_donor_transfer", "secondary_transfer_valid"),
        ("secondary_stsp_only_layer3_postD_ux_donor_transfer", "secondary_transfer_valid"),
    ):
        valid = cells.loc[cells[valid_column].eq(1) & np.isfinite(cells[endpoint])].copy()
        values = valid[endpoint].to_numpy(dtype=np.float64)
        ci_low, ci_high = (
            crossed_bootstrap_mean_ci(
                values,
                valid["history_family_id"].to_numpy(dtype=np.int64),
                valid["b_anchor_id"].to_numpy(dtype=np.int64),
                draws=int(cfg.bootstrap_draws),
                seed=stable_seed(endpoint),
            )
            if len(valid) else (float("nan"), float("nan"))
        )
        summary_rows.append(
            {
                "network_seed": int(cfg.network_seed),
                "endpoint": endpoint,
                "role": "primary" if "secondary" not in endpoint else "secondary_attribution",
                "n_cells": int(len(valid)),
                "valid_coverage": float(len(valid) / max(1, len(cells))),
                "mean_donor_transfer": float(values.mean()) if len(values) else float("nan"),
                "positive_fraction": float(np.mean(values > 0)) if len(values) else float("nan"),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "inference_scope": "within_network_crossed_history_family_by_anchor_stability_only",
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    gate_mean = float(cells.loc[cells["gate_transfer_valid"].eq(1), "gate_early_layer2_C_donor_transfer"].mean())
    _write_csv(out_dir / "exp_c_cell_metrics.csv", cells)
    _write_csv(out_dir / "exp_c_identity_audit.csv", audit)
    _write_csv(out_dir / "exp_c_network_summary.csv", summary_frame)
    parents = {
        "frozen_k5_history_bank": _parent_entry(
            _resolve(_repo_root(), cfg.frozen_root)
            / f"seed_{int(cfg.network_seed)}" / "data" / "intermediates" / "fixed_b_history_bank"
        ),
        "frozen_input_bank": _parent_entry(
            _resolve(_repo_root(), cfg.frozen_root)
            / f"seed_{int(cfg.network_seed)}" / "data" / "intermediates" / "fixed_b_input_bank"
        ),
    }
    _write_task_manifest(out_dir, task_id=TASK_EXP_C, parents=parents, params={"prefix_k": 5, "families": cfg.families, "anchors": cfg.anchors})
    summary = _metric_summary_payload(
        cfg,
        experiment=TASK_EXP_C,
        endpoints={
            row.endpoint: {
                "role": row.role,
                "mean_donor_transfer": float(row.mean_donor_transfer),
                "positive_fraction": float(row.positive_fraction),
                "ci95_low": float(row.bootstrap_ci95_low),
                "ci95_high": float(row.bootstrap_ci95_high),
                "valid_coverage": float(row.valid_coverage),
                "n_cells": int(row.n_cells),
            }
            for row in summary_frame.itertuples(index=False)
        },
        extra={
            "all_identity_gates_pass": bool(audit["identity_pass"].eq(1).all()),
            "n_chunks": int(audit["chunk_id"].nunique()),
            "gate_mean_early_layer2_C_donor_transfer": gate_mean,
            "postC_fast_residual_l2_vmem_max_abs": float(audit["postC_fast_residual_l2_vmem_max_abs"].max()),
        },
    )
    _write_json(out_dir / "summary.json", summary)
    return summary


def _fast_state_residual(
    boundary: Mapping[str, Mapping[str, np.ndarray]],
    slices: Mapping[str, slice],
    donor_indices: np.ndarray,
) -> dict[str, float]:
    """Max |swap - native| fast-state deviation at the post-C boundary (audit only)."""
    output: dict[str, float] = {}
    for layer, state in (("layer1", "v_mem"), ("layer2", "v_mem")):
        native = boundary[layer][state][slices["native"]]
        swap = boundary[layer][state][slices["layer2_swap"]]
        output[f"{layer}_vmem"] = float(np.abs(swap - native).max()) if native.size else float("nan")
    return output


def _array_sha256(value: np.ndarray | torch.Tensor) -> str:
    array = np.ascontiguousarray(
        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    )
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()
