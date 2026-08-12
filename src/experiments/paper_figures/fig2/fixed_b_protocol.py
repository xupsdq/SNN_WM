from __future__ import annotations

import hashlib
import json
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.experiments.paper_figures.fig2.artifacts import read_json
from src.experiments.paper_figures.fig2.cache_keys import dataframe_hash, model_fingerprint
from src.experiments.paper_figures.fig2.fixed_b_artifacts import (
    FixedBArtifact,
    array_hash,
    load_fixed_b_artifact,
    save_fixed_b_artifact,
)
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_specs import FIXED_B_SCHEMA_VERSION
from src.experiments.paper_figures.fig2.types import ExperimentContext


PROTOCOL_TASK_ID = "fixed_b_frozen_protocol"
CONFIRMATORY_AUTHORIZATION = (
    "Run the frozen fixed-B v4 mechanism protocol on all untouched networks 1001 through 1019."
)
DEVELOPMENT_SEEDS = (1000,)
CONFIRMATORY_SEEDS = tuple(range(1001, 1020))
FULL_COHORT_SEEDS = DEVELOPMENT_SEEDS + CONFIRMATORY_SEEDS

_METADATA_COLUMNS = {
    "network_seed",
    "protocol_seed",
    "candidate_history_row_id",
    "history_row_id",
    "candidate_family_id",
    "history_family_id",
    "balance_stratum",
    "history_condition",
    "prefix_k",
}


def select_history_families(
    ctx: ExperimentContext,
    candidate_features: pd.DataFrame,
    candidate_overlap: pd.DataFrame,
    candidate_families: pd.DataFrame,
) -> tuple[list[int], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select histories using only pre-B summaries and predeclared exact-B overlap."""
    required = {"candidate_family_id", "history_condition", "prefix_k"}
    if not required.issubset(candidate_features.columns):
        raise ValueError(f"Candidate feature table is missing {sorted(required - set(candidate_features.columns))}")
    feature_columns = [
        column
        for column in candidate_features.columns
        if column not in _METADATA_COLUMNS and pd.api.types.is_numeric_dtype(candidate_features[column])
    ]
    allowed_suffixes = {
        "u_mean",
        "x_mean",
        "g_mean",
        "u_norm",
        "x_norm",
        "g_norm",
        "headroom_mean",
        "distance_to_bounds",
        "support_mass",
        "prior_spike_count",
    }
    forbidden = [
        column
        for column in feature_columns
        if not (
            column.startswith(("layer1_", "layer2_", "layer3_"))
            and column.split("_", 1)[1] in allowed_suffixes
        )
    ]
    if forbidden:
        raise ValueError(
            "Outcome-blind source selection received non-prestate covariates: "
            f"{sorted(forbidden)}"
        )
    feature_columns = [column for column in feature_columns if not column.startswith("overlap_")]
    if not feature_columns:
        raise ValueError("Candidate feature table has no scalar matching covariates")

    keys = ["candidate_family_id", "prefix_k"]
    duplicates = candidate_features.duplicated(keys + ["history_condition"])
    if bool(duplicates.any()):
        raise ValueError("Candidate pre-B features are not unique per family/condition/K")
    a = candidate_features.loc[candidate_features["history_condition"].eq("A")]
    c = candidate_features.loc[candidate_features["history_condition"].eq("C")]
    merged = a.merge(c, on=keys, suffixes=("__A", "__C"), validate="one_to_one")
    prefix_depths = sorted(int(value) for value in merged["prefix_k"].unique())
    candidate_ids = sorted(int(value) for value in merged["candidate_family_id"].unique())
    a_vectors: dict[int, np.ndarray] = {}
    c_vectors: dict[int, np.ndarray] = {}
    expanded_names: list[str] = []
    for prefix_k in prefix_depths:
        expanded_names.extend(f"K{prefix_k}:{column}" for column in feature_columns)
    by_key = merged.set_index(keys)
    for candidate_id in candidate_ids:
        a_values: list[float] = []
        c_values: list[float] = []
        for prefix_k in prefix_depths:
            row = by_key.loc[(candidate_id, prefix_k)]
            a_values.extend(float(row[f"{column}__A"]) for column in feature_columns)
            c_values.extend(float(row[f"{column}__C"]) for column in feature_columns)
        a_vectors[candidate_id] = np.asarray(a_values, dtype=np.float64)
        c_vectors[candidate_id] = np.asarray(c_values, dtype=np.float64)

    strata_by_candidate = candidate_families.set_index("candidate_family_id")["balance_stratum"].astype(int).to_dict()
    n_selected = int(ctx.cfg.fixed_b_history_families)
    unique_strata = sorted(set(strata_by_candidate.values()))
    if n_selected <= len(unique_strata):
        selected_strata = unique_strata[:n_selected]
        groups = [sorted(candidate for candidate in candidate_ids if strata_by_candidate[candidate] == stratum) for stratum in selected_strata]
    else:
        groups = [[candidate] for candidate in candidate_ids]
    if len(groups) != n_selected or any(not group for group in groups):
        raise ValueError("Candidate balance strata cannot supply the locked selected-family count")

    rng = np.random.default_rng(int(ctx.cfg.fixed_b_protocol_seed) + 911)
    candidate_sets: list[tuple[float, float, tuple[int, ...]]] = []

    def scalar_score(ids: Sequence[int]) -> tuple[float, float]:
        smd = _standardized_mean_differences(
            np.stack([a_vectors[int(value)] for value in ids]),
            np.stack([c_vectors[int(value)] for value in ids]),
        )
        finite = np.where(np.isfinite(smd), np.abs(smd), np.inf)
        return float(np.max(finite)), float(np.mean(finite))

    def consider(ids: Sequence[int]) -> None:
        ordered = tuple(int(value) for value in ids)
        maximum, mean_value = scalar_score(ordered)
        candidate_sets.append((maximum, mean_value, ordered))
        candidate_sets.sort(key=lambda item: (item[0], item[1], item[2]))
        del candidate_sets[128:]

    consider([group[0] for group in groups])
    n_draws = 4_000 if bool(ctx.cfg.smoke) else 80_000
    for _ in range(n_draws):
        consider([group[int(rng.integers(0, len(group)))] for group in groups])

    current = list(candidate_sets[0][2])
    improved = True
    while improved:
        improved = False
        baseline = scalar_score(current)
        for position, group in enumerate(groups):
            best = (baseline[0], baseline[1], current[position])
            for candidate_id in group:
                proposal = list(current)
                proposal[position] = int(candidate_id)
                maximum, mean_value = scalar_score(proposal)
                score = (maximum, mean_value, int(candidate_id))
                if score < best:
                    best = score
            if best[2] != current[position]:
                current[position] = int(best[2])
                baseline = (best[0], best[1])
                improved = True
        consider(current)

    max_allowed = float(ctx.cfg.fixed_b_source_match_max_smd)
    feasible = [item for item in candidate_sets if item[0] <= max_allowed + 1e-12]
    if not feasible:
        best = candidate_sets[0]
        raise RuntimeError(
            f"Outcome-blind source matching failed: best max_abs_smd={best[0]:.6g} exceeds {max_allowed:.6g}"
        )
    overlap_lookup = _overlap_variation_lookup(candidate_overlap)
    chosen = max(
        feasible,
        key=lambda item: (
            _selection_overlap_variation(item[2], overlap_lookup),
            -item[0],
            -item[1],
            tuple(-value for value in item[2]),
        ),
    )
    selected_ids = list(chosen[2])
    a_selected = np.stack([a_vectors[value] for value in selected_ids])
    c_selected = np.stack([c_vectors[value] for value in selected_ids])
    smd = _standardized_mean_differences(a_selected, c_selected)
    balance_rows = [
        {
            "protocol_seed": int(ctx.cfg.fixed_b_protocol_seed),
            "covariate": name,
            "A_mean": float(a_selected[:, index].mean()),
            "C_mean": float(c_selected[:, index].mean()),
            "pooled_sd": float(_pooled_sd(a_selected[:, index], c_selected[:, index])),
            "standardized_mean_difference": float(smd[index]),
            "abs_standardized_mean_difference": float(abs(smd[index])),
            "threshold": max_allowed,
            "passed": int(abs(float(smd[index])) <= max_allowed + 1e-12),
        }
        for index, name in enumerate(expanded_names)
    ]
    selection_rows = []
    for rank, candidate_id in enumerate(selected_ids):
        selection_rows.append(
            {
                "protocol_seed": int(ctx.cfg.fixed_b_protocol_seed),
                "selection_rank": int(rank),
                "candidate_family_id": int(candidate_id),
                "balance_stratum": int(strata_by_candidate[candidate_id]),
                "selection_uses_outcomes": 0,
                "candidate_overlap_variation": float(overlap_lookup.get(candidate_id, np.nan)),
            }
        )
    summary = {
        "fixed_b_schema_version": FIXED_B_SCHEMA_VERSION,
        "protocol_seed": int(ctx.cfg.fixed_b_protocol_seed),
        "candidate_count": int(len(candidate_ids)),
        "selected_count": int(len(selected_ids)),
        "selected_candidate_ids": selected_ids,
        "max_abs_standardized_mean_difference": float(np.max(np.abs(smd))),
        "mean_abs_standardized_mean_difference": float(np.mean(np.abs(smd))),
        "source_match_threshold": max_allowed,
        "overlap_variation": float(_selection_overlap_variation(selected_ids, overlap_lookup)),
        "selection_uses_outcomes": False,
        "outcome_columns_accessed": [],
    }
    return selected_ids, pd.DataFrame(selection_rows), pd.DataFrame(balance_rows), summary


def frozen_protocol_dir(ctx: ExperimentContext) -> Path:
    configured = str(ctx.cfg.fixed_b_protocol_dir).strip()
    if configured:
        return Path(configured).resolve()
    return (Path(ctx.cfg.output_root).resolve().parent / "frozen_protocol").resolve()


def seal_frozen_protocol(
    ctx: ExperimentContext,
    *,
    specs: FixedBArtifact,
    inputs: FixedBArtifact,
    histories: FixedBArtifact,
    protocol_dir: Path | None = None,
) -> FixedBArtifact:
    if int(ctx.cfg.network_seed) != 1000:
        raise RuntimeError("Only development seed 1000 may select and seal the fixed-B protocol")
    protocol_dir = frozen_protocol_dir(ctx) if protocol_dir is None else Path(protocol_dir).resolve()
    selected_tables = {
        name: histories.tables[name].copy()
        for name in (
            "history_families",
            "history_specs",
            "b_anchor_specs",
            "cell_specs",
            "fold_specs",
            "branch_specs",
            "swap_specs",
            "null_specs",
            "selection_audit",
            "source_balance",
            "candidate_overlap",
            "prestate_features",
        )
    }
    selected_image_ids = _selected_history_image_ids(selected_tables["history_specs"])
    history_manifest = inputs.tables["history_input_manifest"].copy()
    history_manifest = history_manifest.loc[history_manifest["image_id"].astype(int).isin(selected_image_ids)].copy()
    history_manifest = history_manifest.sort_values("image_id").reset_index(drop=True)
    source_indices = history_manifest["row_index"].to_numpy(dtype=np.int64)
    history_manifest["row_index"] = np.arange(len(history_manifest), dtype=np.int64)
    selected_tables["history_input_manifest"] = history_manifest
    selected_tables["input_manifest"] = inputs.tables["input_manifest"].copy()
    selected_arrays = {
        "exact_b_spikes": np.asarray(inputs.arrays["exact_b_spikes"]),
        "history_spikes": np.asarray(inputs.arrays["history_spikes"])[source_indices],
    }
    endpoint_spec = dict(specs.payloads["endpoint_spec"])
    source_selection = dict(histories.payloads["source_selection"])
    identity = _protocol_identity(
        ctx,
        tables=selected_tables,
        arrays=selected_arrays,
        endpoint_spec=endpoint_spec,
        source_selection=source_selection,
        parent_digests={
            "fixed_b_specs": specs.digest,
            "fixed_b_input_bank": inputs.digest,
            "fixed_b_history_bank": histories.digest,
        },
    )
    protocol_digest = _canonical_digest(identity)
    cache_key = {
        "schema_name": "fig2_runtime_artifacts",
        "schema_version": 1,
        "fixed_b_schema_version": FIXED_B_SCHEMA_VERSION,
        "task_id": PROTOCOL_TASK_ID,
        "protocol_digest": protocol_digest,
        "identity": identity,
    }
    payloads = {
        "endpoint_spec": endpoint_spec,
        "source_selection": source_selection,
        "protocol": {
            "fixed_b_schema_version": FIXED_B_SCHEMA_VERSION,
            "protocol_digest": protocol_digest,
            "protocol_seed": int(ctx.cfg.fixed_b_protocol_seed),
            "selector_network_seed": 1000,
            "selector_model": model_fingerprint(ctx.cfg.model_path),
            "dataset_root": str(Path(ctx.cfg.dataset_root).resolve()),
            "dataset_split": str(ctx.cfg.split),
            "selected_candidate_ids": [int(value) for value in selected_tables["history_families"]["candidate_family_id"]],
            "prefix_depths": [int(value) for value in ctx.cfg.fixed_b_prefix_depths],
            "folds": int(ctx.cfg.fixed_b_folds),
            "null_replicates": int(ctx.cfg.fixed_b_null_replicates),
            "outcome_access_before_seal": False,
            "outcome_artifacts_accessed": [],
            "sealed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "remaining_seeds_allowed": False,
        },
    }
    if (protocol_dir / "cache_key.json").exists():
        existing = load_frozen_protocol(protocol_dir)
        existing_digest = str(existing.payloads["protocol"]["protocol_digest"])
        if existing_digest != protocol_digest:
            raise RuntimeError(
                f"Frozen fixed-B protocol is immutable: existing={existing_digest}, proposed={protocol_digest}"
            )
        return existing
    protocol_dir.parent.mkdir(parents=True, exist_ok=True)
    return save_fixed_b_artifact(
        protocol_dir,
        cache_key,
        tables=selected_tables,
        arrays=selected_arrays,
        payloads=payloads,
    )


def load_frozen_protocol(protocol_dir: str | Path) -> FixedBArtifact:
    protocol_dir = Path(protocol_dir).resolve()
    cache_path = protocol_dir / "cache_key.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing frozen fixed-B protocol cache key: {cache_path}")
    stored_key = read_json(cache_path)
    cache_key = dict(stored_key.get("cache_key", stored_key))
    if str(cache_key.get("task_id")) != PROTOCOL_TASK_ID:
        raise ValueError(f"Unexpected frozen protocol task id in {cache_path}")
    artifact = load_fixed_b_artifact(protocol_dir, cache_key, task_id=PROTOCOL_TASK_ID)
    recomputed_identity = _protocol_identity_from_artifact(artifact)
    recomputed_digest = _canonical_digest(recomputed_identity)
    recorded = str(artifact.payloads["protocol"].get("protocol_digest", ""))
    if recomputed_digest != recorded or str(cache_key.get("protocol_digest", "")) != recorded:
        raise RuntimeError("Frozen fixed-B protocol identity digest mismatch")
    return artifact


def validate_seed_permission(
    network_seed: int,
    *,
    task_state_path: str | Path | None,
    protocol: FixedBArtifact | None,
) -> str:
    seed = int(network_seed)
    if seed == 1000:
        return "development_seed_1000"
    if seed not in set(CONFIRMATORY_SEEDS):
        raise RuntimeError(f"Corrected fixed-B seed {seed} is outside the locked 1000..1019 cohort")
    if protocol is None:
        raise RuntimeError("Later corrected fixed-B seeds require the frozen seed-1000 protocol")
    if not task_state_path:
        raise RuntimeError("Later corrected fixed-B seeds require --fixed-b-task-state")
    state = json.loads(Path(task_state_path).read_text(encoding="utf-8"))
    if state.get("schema_version") != "4.0" or state.get("remaining_seeds_allowed") is not True:
        raise RuntimeError("Remaining corrected fixed-B seeds are still forbidden by task state")
    authorization = state.get("runtime_authorization") or {}
    text = str(authorization.get("text", ""))
    track = str(authorization.get("track", ""))
    expected_digest = protocol_digest(protocol)
    if str(state.get("protocol_digest", "")) != expected_digest:
        raise RuntimeError("Task-state protocol digest does not match the frozen fixed-B protocol")
    if [int(value) for value in state.get("confirmatory_networks", [])] != list(CONFIRMATORY_SEEDS):
        raise RuntimeError("Task state does not contain the exact frozen confirmatory cohort")
    if text != CONFIRMATORY_AUTHORIZATION or track != "confirmatory_v4":
        raise RuntimeError("Task state does not contain the exact fixed-B v4 runtime authorization")
    return track


def protocol_digest(protocol: FixedBArtifact) -> str:
    return str(protocol.payloads["protocol"]["protocol_digest"])


def _protocol_identity(
    ctx: ExperimentContext,
    *,
    tables: Mapping[str, pd.DataFrame],
    arrays: Mapping[str, np.ndarray],
    endpoint_spec: Mapping[str, Any],
    source_selection: Mapping[str, Any],
    parent_digests: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "fixed_b_schema_version": FIXED_B_SCHEMA_VERSION,
        "protocol_seed": int(ctx.cfg.fixed_b_protocol_seed),
        "dataset_root": str(Path(ctx.cfg.dataset_root).resolve()),
        "dataset_split": str(ctx.cfg.split),
        "selector_network_seed": 1000,
        "selector_model": model_fingerprint(ctx.cfg.model_path),
        "table_hashes": {
            name: _persisted_dataframe_hash(table)
            for name, table in sorted(tables.items())
        },
        "array_hashes": {name: array_hash(array) for name, array in sorted(arrays.items())},
        "endpoint_spec_hash": _canonical_digest(endpoint_spec),
        "source_selection_hash": _canonical_digest(source_selection),
        "parent_digests": dict(sorted((str(key), str(value)) for key, value in parent_digests.items())),
    }


def _protocol_identity_from_artifact(artifact: FixedBArtifact) -> dict[str, Any]:
    recorded = artifact.payloads["protocol"]
    stored_key = read_json(artifact.root / "cache_key.json")
    cache_key = dict(stored_key.get("cache_key", stored_key))
    table_hashes = {
        str(row.name): str(row.content_sha256)
        for row in artifact.manifest.loc[
            artifact.manifest["kind"].eq("table")
        ].itertuples(index=False)
    }
    return {
        "fixed_b_schema_version": int(recorded["fixed_b_schema_version"]),
        "protocol_seed": int(recorded["protocol_seed"]),
        "dataset_root": str(recorded["dataset_root"]),
        "dataset_split": str(recorded["dataset_split"]),
        "selector_network_seed": int(recorded["selector_network_seed"]),
        "selector_model": dict(recorded["selector_model"]),
        "table_hashes": dict(sorted(table_hashes.items())),
        "array_hashes": {name: array_hash(array) for name, array in sorted(artifact.arrays.items())},
        "endpoint_spec_hash": _canonical_digest(artifact.payloads["endpoint_spec"]),
        "source_selection_hash": _canonical_digest(artifact.payloads["source_selection"]),
        "parent_digests": dict(cache_key["identity"]["parent_digests"]),
    }


def _persisted_dataframe_hash(table: pd.DataFrame) -> str:
    encoded = table.to_csv(index=False, lineterminator="\n")
    persisted = pd.read_csv(StringIO(encoded), keep_default_na=False)
    return dataframe_hash(persisted)


def _selected_history_image_ids(history_specs: pd.DataFrame) -> set[int]:
    image_ids: set[int] = set()
    for encoded in history_specs.loc[history_specs["history_condition"].isin(["A", "C"]), "sequence_image_ids"]:
        image_ids.update(int(value) for value in json.loads(str(encoded)))
    return image_ids


def _standardized_mean_differences(a: np.ndarray, c: np.ndarray) -> np.ndarray:
    differences = a.mean(axis=0) - c.mean(axis=0)
    pooled = np.asarray([_pooled_sd(a[:, index], c[:, index]) for index in range(a.shape[1])])
    output = np.zeros_like(differences, dtype=np.float64)
    supported = pooled > 1e-12
    output[supported] = differences[supported] / pooled[supported]
    output[~supported & (np.abs(differences) > 1e-12)] = np.inf
    return output


def _pooled_sd(a: np.ndarray, c: np.ndarray) -> float:
    if len(a) < 2 or len(c) < 2:
        return float(np.std(np.concatenate([a, c]), ddof=0))
    return float(np.sqrt(0.5 * (np.var(a, ddof=1) + np.var(c, ddof=1))))


def _overlap_variation_lookup(candidate_overlap: pd.DataFrame) -> dict[int, float]:
    required = {"candidate_family_id", "history_condition", "prefix_k", "b_anchor_id", "projected_overlap"}
    if not required.issubset(candidate_overlap.columns):
        raise ValueError(f"Candidate overlap table is missing {sorted(required - set(candidate_overlap.columns))}")
    pivot = candidate_overlap.pivot_table(
        index=["candidate_family_id", "prefix_k", "b_anchor_id"],
        columns="history_condition",
        values="projected_overlap",
        aggfunc="first",
    ).dropna(subset=["A", "C"])
    contrast = (pivot["A"] - pivot["C"]).abs()
    return {
        int(candidate_id): float(values.std(ddof=0))
        for candidate_id, values in contrast.groupby(level="candidate_family_id")
    }


def _selection_overlap_variation(selected_ids: Sequence[int], lookup: Mapping[int, float]) -> float:
    values = np.asarray([float(lookup.get(int(value), 0.0)) for value in selected_ids], dtype=np.float64)
    return float(values.mean() + values.std(ddof=0))


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "CONFIRMATORY_SEEDS",
    "CONFIRMATORY_AUTHORIZATION",
    "DEVELOPMENT_SEEDS",
    "FULL_COHORT_SEEDS",
    "PROTOCOL_TASK_ID",
    "frozen_protocol_dir",
    "load_frozen_protocol",
    "protocol_digest",
    "seal_frozen_protocol",
    "select_history_families",
    "validate_seed_permission",
]
