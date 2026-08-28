from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from src.experiments.paper_figures.common.bundle_io import relative_to_root, save_csv_with_registry
from src.experiments.paper_figures.fig2.artifacts import task_artifact_dir, validate_cache_key_integrity, write_json
from src.experiments.paper_figures.fig2.cache_keys import (
    build_fixed_b_cache_key,
    sha256_file,
)
from src.experiments.paper_figures.fig2.fixed_b_artifacts import (
    FixedBArtifact,
    artifact_exists_and_matches,
    load_fixed_b_artifact,
    save_fixed_b_artifact,
)
from src.experiments.paper_figures.fig2.fixed_b_protocol import (
    frozen_protocol_dir,
    load_frozen_protocol,
    protocol_digest,
    seal_frozen_protocol,
    validate_seed_permission,
)
from src.experiments.paper_figures.fig2.schemas import (
    TASK_FIXED_B_ANALYSIS,
    TASK_FIXED_B_COHORT_AGGREGATE,
    TASK_FIXED_B_HISTORY_BANK,
    TASK_FIXED_B_INPUT_BANK,
    TASK_FIXED_B_PROTOCOL,
    TASK_FIXED_B_REPLAY_BANK,
    TASK_FIXED_B_ROLLOUT_BANK,
    TASK_FIXED_B_SPECS,
    TASK_FIXED_B_SWAP_BANK,
)
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_mechanism_analysis import (
    analyze_fixed_b_mechanism_single_seed,
)
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_cohort import aggregate_fixed_b_cohort
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_runtime import (
    build_exact_b_input_bank,
    build_history_boundary_bank,
    build_replay_bank,
    build_rollout_bank,
    build_swap_bank,
)
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_specs import (
    FIXED_B_SCHEMA_VERSION,
    build_fixed_b_specs,
)
from src.experiments.paper_figures.fig2.types import ExperimentContext


Builder = Callable[[], tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]]


def run_fixed_b_task(
    ctx: ExperimentContext,
    *,
    task_id: str,
    mode: str,
    artifact_root: Path,
) -> None:
    protocol_path = frozen_protocol_dir(ctx)
    protocol = load_frozen_protocol(protocol_path) if (protocol_path / "cache_key.json").exists() else None
    validate_seed_permission(
        int(ctx.cfg.network_seed),
        task_state_path=str(ctx.cfg.fixed_b_task_state_path).strip() or None,
        protocol=protocol,
    )
    if task_id == TASK_FIXED_B_COHORT_AGGREGATE:
        if not str(ctx.cfg.fixed_b_task_state_path).strip():
            raise RuntimeError("fixed_b_cohort_aggregate requires --fixed-b-task-state")
        paths = aggregate_fixed_b_cohort(
            figure_root=Path(ctx.cfg.output_root).resolve(),
            protocol_dir=protocol_path,
            task_state_path=ctx.cfg.fixed_b_task_state_path,
        )
        for name, path in paths.items():
            ctx.output_files[f"fixed_b_{name}"] = relative_to_root(
                path,
                Path(ctx.cfg.output_root).resolve(),
            )
        ctx.completed_modules[task_id] = True
        return

    cache: dict[str, FixedBArtifact] = {}

    def get_specs(parent_mode: str) -> FixedBArtifact:
        later_seed = int(ctx.cfg.network_seed) != 1000
        extra = {
            "protocol_version": FIXED_B_SCHEMA_VERSION,
            "selection_uses_outcomes": False,
            "source_stage": "frozen_selected" if later_seed else "candidate",
        }
        if later_seed:
            assert protocol is not None
            extra["protocol_digest"] = protocol_digest(protocol)
            extra["frozen_audit_tables"] = [
                "selection_audit",
                "source_balance",
                "candidate_overlap",
            ]
        key = build_fixed_b_cache_key(
            ctx.cfg,
            task_id=TASK_FIXED_B_SPECS,
            model_dependent=False,
            extra=extra,
        )

        def build() -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
            if later_seed:
                assert protocol is not None
                names = (
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
                )
                return (
                    {name: protocol.tables[name].copy() for name in names},
                    {},
                    {
                        "endpoint_spec": dict(protocol.payloads["endpoint_spec"]),
                        "protocol_reference": {"protocol_digest": protocol_digest(protocol)},
                    },
                )
            tables, endpoint_spec = build_fixed_b_specs(ctx)
            return tables, {}, {"endpoint_spec": endpoint_spec}

        artifact = _get_or_build(
            task_id=TASK_FIXED_B_SPECS,
            mode=parent_mode,
            artifact_root=artifact_root,
            cache_key=key,
            builder=build,
        )
        _validate_fixed_b_artifact(TASK_FIXED_B_SPECS, artifact, ctx=ctx)
        cache[TASK_FIXED_B_SPECS] = artifact
        return artifact

    def get_input(parent_mode: str) -> FixedBArtifact:
        specs = cache.get(TASK_FIXED_B_SPECS) or get_specs(parent_mode)
        later_seed = int(ctx.cfg.network_seed) != 1000
        extra = {
            "protocol_version": FIXED_B_SCHEMA_VERSION,
            "encoding_rule": "one_frozen_tensor_per_source_image",
        }
        if later_seed:
            assert protocol is not None
            extra["protocol_digest"] = protocol_digest(protocol)
        key = build_fixed_b_cache_key(
            ctx.cfg,
            task_id=TASK_FIXED_B_INPUT_BANK,
            parent_digests={TASK_FIXED_B_SPECS: specs.digest},
            model_dependent=False,
            extra=extra,
        )

        def build() -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
            if later_seed:
                assert protocol is not None
                return (
                    {
                        "input_manifest": protocol.tables["input_manifest"].copy(),
                        "history_input_manifest": protocol.tables["history_input_manifest"].copy(),
                    },
                    {
                        "exact_b_spikes": protocol.arrays["exact_b_spikes"],
                        "history_spikes": protocol.arrays["history_spikes"],
                    },
                    {
                        "protocol": {
                            "protocol_digest": protocol_digest(protocol),
                            "source": "frozen_protocol",
                        }
                    },
                )
            tables, arrays, input_protocol = build_exact_b_input_bank(ctx, specs)
            return tables, arrays, {"protocol": input_protocol}

        artifact = _get_or_build(
            task_id=TASK_FIXED_B_INPUT_BANK,
            mode=parent_mode,
            artifact_root=artifact_root,
            cache_key=key,
            builder=build,
        )
        _validate_fixed_b_artifact(
            TASK_FIXED_B_INPUT_BANK,
            artifact,
            ctx=ctx,
            parents={TASK_FIXED_B_SPECS: specs},
        )
        cache[TASK_FIXED_B_INPUT_BANK] = artifact
        return artifact

    def get_history(parent_mode: str) -> FixedBArtifact:
        specs = cache.get(TASK_FIXED_B_SPECS) or get_specs(parent_mode)
        inputs = cache.get(TASK_FIXED_B_INPUT_BANK) or get_input(parent_mode)
        extra = {
            "protocol_version": FIXED_B_SCHEMA_VERSION,
            "boundary": "selected_full_fast_state_plus_all_layer_u_x",
            "source_selection": "seed1000_pre_B_only" if int(ctx.cfg.network_seed) == 1000 else "frozen_ids",
        }
        if protocol is not None and int(ctx.cfg.network_seed) != 1000:
            extra["protocol_digest"] = protocol_digest(protocol)
        key = build_fixed_b_cache_key(
            ctx.cfg,
            task_id=TASK_FIXED_B_HISTORY_BANK,
            parent_digests={TASK_FIXED_B_SPECS: specs.digest, TASK_FIXED_B_INPUT_BANK: inputs.digest},
            extra=extra,
        )

        def build() -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
            tables, arrays, history_protocol = build_history_boundary_bank(ctx, specs, inputs)
            return tables, arrays, history_protocol

        artifact = _get_or_build(
            task_id=TASK_FIXED_B_HISTORY_BANK,
            mode=parent_mode,
            artifact_root=artifact_root,
            cache_key=key,
            builder=build,
        )
        _validate_fixed_b_artifact(
            TASK_FIXED_B_HISTORY_BANK,
            artifact,
            ctx=ctx,
            parents={TASK_FIXED_B_SPECS: specs, TASK_FIXED_B_INPUT_BANK: inputs},
        )
        cache[TASK_FIXED_B_HISTORY_BANK] = artifact
        return artifact

    def get_protocol(parent_mode: str) -> FixedBArtifact:
        nonlocal protocol
        if protocol is not None:
            return protocol
        if parent_mode == "require":
            protocol = load_frozen_protocol(protocol_path)
            return protocol
        specs = cache.get(TASK_FIXED_B_SPECS) or get_specs(parent_mode)
        inputs = cache.get(TASK_FIXED_B_INPUT_BANK) or get_input(parent_mode)
        histories = cache.get(TASK_FIXED_B_HISTORY_BANK) or get_history(parent_mode)
        protocol = seal_frozen_protocol(
            ctx,
            specs=specs,
            inputs=inputs,
            histories=histories,
            protocol_dir=protocol_path,
        )
        return protocol

    def get_replay(parent_mode: str) -> FixedBArtifact:
        frozen = get_protocol(parent_mode)
        specs = cache.get(TASK_FIXED_B_SPECS) or get_specs(parent_mode)
        inputs = cache.get(TASK_FIXED_B_INPUT_BANK) or get_input(parent_mode)
        histories = cache.get(TASK_FIXED_B_HISTORY_BANK) or get_history(parent_mode)
        key = build_fixed_b_cache_key(
            ctx.cfg,
            task_id=TASK_FIXED_B_REPLAY_BANK,
            parent_digests={
                TASK_FIXED_B_SPECS: specs.digest,
                TASK_FIXED_B_INPUT_BANK: inputs.digest,
                TASK_FIXED_B_HISTORY_BANK: histories.digest,
            },
            extra={
                "protocol_version": FIXED_B_SCHEMA_VERSION,
                "protocol_digest": protocol_digest(frozen),
                "replay_source": "S0_plus_exact_B_L1_pooled_events",
            },
        )

        def build() -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
            tables, arrays, replay_protocol = build_replay_bank(ctx, specs, inputs, histories)
            return tables, arrays, {"protocol": replay_protocol}

        artifact = _get_or_build(
            task_id=TASK_FIXED_B_REPLAY_BANK,
            mode=parent_mode,
            artifact_root=artifact_root,
            cache_key=key,
            builder=build,
        )
        _validate_fixed_b_artifact(TASK_FIXED_B_REPLAY_BANK, artifact, ctx=ctx)
        cache[TASK_FIXED_B_REPLAY_BANK] = artifact
        return artifact

    def get_rollouts(parent_mode: str) -> FixedBArtifact:
        frozen = get_protocol(parent_mode)
        specs = cache.get(TASK_FIXED_B_SPECS) or get_specs(parent_mode)
        inputs = cache.get(TASK_FIXED_B_INPUT_BANK) or get_input(parent_mode)
        histories = cache.get(TASK_FIXED_B_HISTORY_BANK) or get_history(parent_mode)
        replay = cache.get(TASK_FIXED_B_REPLAY_BANK) or get_replay(parent_mode)
        key = build_fixed_b_cache_key(
            ctx.cfg,
            task_id=TASK_FIXED_B_ROLLOUT_BANK,
            parent_digests={
                TASK_FIXED_B_SPECS: specs.digest,
                TASK_FIXED_B_INPUT_BANK: inputs.digest,
                TASK_FIXED_B_HISTORY_BANK: histories.digest,
                TASK_FIXED_B_REPLAY_BANK: replay.digest,
            },
            extra={
                "protocol_version": FIXED_B_SCHEMA_VERSION,
                "protocol_digest": protocol_digest(frozen),
                "tracks": ["natural", "stsp_isolated"],
                "branches": ["passive", "free", "replay"],
                "actual_events": "packed_full_layer2_presynaptic",
                "voltage_storage": "online_bounded_projections_only",
            },
        )

        def build() -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
            tables, arrays, rollout_protocol = build_rollout_bank(ctx, specs, inputs, histories, replay)
            return tables, arrays, {"protocol": rollout_protocol}

        artifact = _get_or_build(
            task_id=TASK_FIXED_B_ROLLOUT_BANK,
            mode=parent_mode,
            artifact_root=artifact_root,
            cache_key=key,
            builder=build,
        )
        _validate_fixed_b_artifact(
            TASK_FIXED_B_ROLLOUT_BANK,
            artifact,
            ctx=ctx,
            parents={TASK_FIXED_B_INPUT_BANK: inputs},
        )
        cache[TASK_FIXED_B_ROLLOUT_BANK] = artifact
        return artifact

    def get_swaps(parent_mode: str) -> FixedBArtifact:
        frozen = get_protocol(parent_mode)
        specs = cache.get(TASK_FIXED_B_SPECS) or get_specs(parent_mode)
        inputs = cache.get(TASK_FIXED_B_INPUT_BANK) or get_input(parent_mode)
        histories = cache.get(TASK_FIXED_B_HISTORY_BANK) or get_history(parent_mode)
        key = build_fixed_b_cache_key(
            ctx.cfg,
            task_id=TASK_FIXED_B_SWAP_BANK,
            parent_digests={
                TASK_FIXED_B_SPECS: specs.digest,
                TASK_FIXED_B_INPUT_BANK: inputs.digest,
                TASK_FIXED_B_HISTORY_BANK: histories.digest,
            },
            extra={
                "protocol_version": FIXED_B_SCHEMA_VERSION,
                "protocol_digest": protocol_digest(frozen),
                "fast_state": "identical_baseline",
                "scopes": ["all_layers", "layer1_only"],
            },
        )

        def build() -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
            tables, arrays, swap_protocol = build_swap_bank(ctx, specs, inputs, histories)
            return tables, arrays, {"protocol": swap_protocol}

        artifact = _get_or_build(
            task_id=TASK_FIXED_B_SWAP_BANK,
            mode=parent_mode,
            artifact_root=artifact_root,
            cache_key=key,
            builder=build,
        )
        _validate_fixed_b_artifact(TASK_FIXED_B_SWAP_BANK, artifact, ctx=ctx)
        cache[TASK_FIXED_B_SWAP_BANK] = artifact
        return artifact

    getters = {
        TASK_FIXED_B_SPECS: get_specs,
        TASK_FIXED_B_INPUT_BANK: get_input,
        TASK_FIXED_B_HISTORY_BANK: get_history,
        TASK_FIXED_B_PROTOCOL: get_protocol,
        TASK_FIXED_B_REPLAY_BANK: get_replay,
        TASK_FIXED_B_ROLLOUT_BANK: get_rollouts,
        TASK_FIXED_B_SWAP_BANK: get_swaps,
    }
    if task_id in getters:
        artifact = getters[task_id](mode)
        if task_id == TASK_FIXED_B_SPECS:
            _write_specs_to_bundle(ctx, artifact.tables, artifact.payloads["endpoint_spec"])
        elif task_id == TASK_FIXED_B_HISTORY_BANK:
            _write_specs_to_bundle(ctx, artifact.tables, specs_payload=get_specs(mode).payloads["endpoint_spec"])
        elif task_id == TASK_FIXED_B_PROTOCOL:
            _write_protocol_reference(ctx, artifact)
        ctx.completed_modules[task_id] = True
        return
    if task_id != TASK_FIXED_B_ANALYSIS:
        raise ValueError(f"Unsupported fixed-B task: {task_id}")

    frozen = get_protocol(mode)
    specs = get_specs(mode)
    inputs = get_input(mode)
    histories = get_history(mode)
    replay = get_replay(mode)
    rollouts = get_rollouts(mode)
    swaps = get_swaps(mode)
    _write_specs_to_bundle(ctx, histories.tables, specs_payload=frozen.payloads["endpoint_spec"])
    _write_protocol_reference(ctx, frozen)
    result_tables, decision = analyze_fixed_b_mechanism_single_seed(
        ctx,
        specs,
        inputs,
        histories,
        replay,
        rollouts,
        swaps,
        protocol=frozen,
    )
    trajectory_path = ctx.raw_dir / "fixed_b_state_trajectory_rows.csv"
    save_csv_with_registry(ctx, rollouts.tables["state_trajectory_rows"], trajectory_path)
    ctx.output_files["fixed_b_state_trajectory_rows"] = relative_to_root(trajectory_path, ctx.seed_dir)
    for name, table in result_tables.items():
        save_csv_with_registry(ctx, table, ctx.metrics_dir / f"{name}.csv")
    decision_path = ctx.metrics_dir / "fixed_b_single_seed_decision.json"
    write_json(decision, decision_path)
    ctx.output_files["fixed_b_single_seed_decision"] = relative_to_root(decision_path, ctx.seed_dir)
    if int(ctx.cfg.network_seed) == 1000:
        _write_seed_1000_alignment(
            ctx,
            result_tables=result_tables,
            decision=decision,
            protocol=frozen,
        )
    ctx.completed_modules[TASK_FIXED_B_ANALYSIS] = True


def _get_or_build(
    *,
    task_id: str,
    mode: str,
    artifact_root: Path,
    cache_key: Mapping[str, Any],
    builder: Builder,
) -> FixedBArtifact:
    task_dir = task_artifact_dir(artifact_root, task_id)
    if mode == "require":
        return load_fixed_b_artifact(task_dir, cache_key, task_id=task_id)
    cache_file = task_dir / "cache_key.json"
    if mode == "auto" and cache_file.exists():
        validate_cache_key_integrity(task_dir, task_id=task_id)
        if artifact_exists_and_matches(task_dir, cache_key):
            return load_fixed_b_artifact(task_dir, cache_key, task_id=task_id)
    tables, arrays, payloads = builder()
    return save_fixed_b_artifact(task_dir, cache_key, tables=tables, arrays=arrays, payloads=payloads)


def _write_specs_to_bundle(
    ctx: ExperimentContext,
    tables: Mapping[str, pd.DataFrame],
    specs_payload: Mapping[str, Any],
) -> None:
    selected_names = (
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
    )
    for name in selected_names:
        if name in tables:
            save_csv_with_registry(ctx, tables[name], ctx.trial_specs_dir / f"fixed_b_{name}.csv")
    write_json(dict(specs_payload), ctx.trial_specs_dir / "fixed_b_endpoint_spec.json")
    ctx.output_files["fixed_b_endpoint_spec"] = relative_to_root(
        ctx.trial_specs_dir / "fixed_b_endpoint_spec.json",
        ctx.seed_dir,
    )


def _write_protocol_reference(ctx: ExperimentContext, protocol: FixedBArtifact) -> None:
    path = ctx.trial_specs_dir / "fixed_b_protocol_reference.json"
    write_json(
        {
            "fixed_b_schema_version": FIXED_B_SCHEMA_VERSION,
            "protocol_digest": protocol_digest(protocol),
            "protocol_dir": str(protocol.root.resolve()),
            "remaining_seeds_allowed": False,
        },
        path,
    )
    ctx.output_files["fixed_b_protocol_reference"] = relative_to_root(path, ctx.seed_dir)
def _write_seed_1000_alignment(
    ctx: ExperimentContext,
    *,
    result_tables: Mapping[str, pd.DataFrame],
    decision: Mapping[str, Any],
    protocol: FixedBArtifact,
) -> None:
    transition_root = Path(ctx.cfg.output_root).resolve()
    aggregate_dir = transition_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    checklist = result_tables["fixed_b_prediction_checklist"].copy()
    claim_ledger = result_tables["fixed_b_claim_ledger"].copy()
    checklist_path = aggregate_dir / "fixed_b_seed_1000_checklist.csv"
    claim_path = aggregate_dir / "fixed_b_seed_1000_claim_ledger.csv"
    checklist.to_csv(
        checklist_path,
        index=False,
        lineterminator="\n",
    )
    claim_ledger.to_csv(
        claim_path,
        index=False,
        lineterminator="\n",
    )
    evidence_paths = {
        "single_seed_decision": (
            ctx.metrics_dir / "fixed_b_single_seed_decision.json"
        ),
        "prediction_checklist": (
            ctx.metrics_dir / "fixed_b_prediction_checklist.csv"
        ),
        "claim_ledger": (
            ctx.metrics_dir / "fixed_b_claim_ledger.csv"
        ),
        "engineering_gates": (
            ctx.metrics_dir / "fixed_b_engineering_gates.csv"
        ),
        "network_scalars": (
            ctx.metrics_dir / "fixed_b_network_scalars.csv"
        ),
        "frozen_protocol_cache_key": (
            protocol.root / "cache_key.json"
        ),
    }
    missing = [
        name
        for name, path in evidence_paths.items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Seed-1000 alignment evidence is missing: {missing}"
        )
    engineering_failures = checklist.loc[
        checklist["tier"].eq("engineering")
        & checklist["passed"].eq(0)
    ]
    seed_dirs = sorted(
        path.name
        for path in transition_root.glob("seed_*")
        if path.is_dir()
    )
    alignment = {
        "fixed_b_schema_version": FIXED_B_SCHEMA_VERSION,
        "network_seed": 1000,
        "protocol_digest": protocol_digest(protocol),
        "verdict": str(decision["verdict"]),
        "engineering_valid": bool(decision["engineering_valid"]),
        "existing_chain_valid": bool(
            decision["existing_chain_valid"]
        ),
        "core_development_pass": bool(
            decision["core_development_pass"]
        ),
        "strong_development_pass": bool(
            decision["strong_development_pass"]
        ),
        "continuation_eligible": bool(
            decision["continuation_eligible"]
        ),
        "eligible_tracks": list(decision["eligible_tracks"]),
        "remaining_seeds_allowed": False,
        "confirmatory_inference_performed": False,
        "inference_role": "development_only",
        "claim_boundary": str(decision["claim_boundary"]),
        "seed_directories": seed_dirs,
        "later_seed_directories": [
            value for value in seed_dirs if value != "seed_1000"
        ],
        "gap_checklist_rows": int(len(checklist)),
        "failed_engineering_items": [
            f"{row.endpoint}@K={row.prefix_k}"
            for row in engineering_failures.itertuples(index=False)
        ],
        "checklist_csv": checklist_path.relative_to(
            transition_root
        ).as_posix(),
        "claim_ledger_csv": claim_path.relative_to(
            transition_root
        ).as_posix(),
        "evidence_sha256": {
            name: sha256_file(path)
            for name, path in sorted(evidence_paths.items())
        },
    }
    alignment_path = (
        aggregate_dir / "fixed_b_seed_1000_alignment.json"
    )
    write_json(alignment, alignment_path)




def _validate_fixed_b_artifact(
    task_id: str,
    artifact: FixedBArtifact,
    *,
    ctx: ExperimentContext,
    parents: Mapping[str, FixedBArtifact] | None = None,
) -> None:
    parents = dict(parents or {})
    n_selected = int(ctx.cfg.fixed_b_history_families)
    n_candidates = int(ctx.cfg.fixed_b_candidate_families)
    n_anchors = int(ctx.cfg.fixed_b_anchors)
    n_depths = len(tuple(ctx.cfg.fixed_b_prefix_depths))
    if task_id == TASK_FIXED_B_SPECS:
        if "candidate_history_specs" in artifact.tables:
            _require_members(
                artifact,
                tables=(
                    "candidate_history_families",
                    "candidate_history_specs",
                    "history_input_specs",
                    "b_anchor_specs",
                    "branch_specs",
                    "null_specs",
                ),
                payloads=("endpoint_spec",),
            )
            if len(artifact.tables["candidate_history_specs"]) != 2 * n_candidates * n_depths:
                raise ValueError("fixed-B candidate history row-count mismatch")
        else:
            _require_members(
                artifact,
                tables=("history_families", "history_specs", "b_anchor_specs", "cell_specs", "fold_specs", "branch_specs", "swap_specs", "null_specs"),
                payloads=("endpoint_spec",),
            )
            if len(artifact.tables["history_specs"]) != 3 * n_selected * n_depths:
                raise ValueError("fixed-B selected history row-count mismatch")
        if len(artifact.tables["b_anchor_specs"]) != n_anchors:
            raise ValueError("fixed-B B-anchor row-count mismatch")
    elif task_id == TASK_FIXED_B_INPUT_BANK:
        _require_members(
            artifact,
            tables=("input_manifest", "history_input_manifest"),
            arrays=("exact_b_spikes", "history_spikes"),
        )
        if artifact.arrays["exact_b_spikes"].shape[0] != n_anchors:
            raise ValueError("fixed-B exact-B input-bank membership mismatch")
        if artifact.tables["input_manifest"]["tensor_sha256"].nunique() != n_anchors:
            raise ValueError("fixed-B exact-B tensors are not uniquely identified")
    elif task_id == TASK_FIXED_B_HISTORY_BANK:
        _require_members(
            artifact,
            tables=(
                "history_families",
                "history_specs",
                "b_anchor_specs",
                "cell_specs",
                "fold_specs",
                "branch_specs",
                "swap_specs",
                "null_specs",
                "prestate_features",
                "restoration_audit",
            ),
            payloads=("source_selection",),
        )
        required = {
            f"k{k}__{layer}__{state}"
            for k in ctx.cfg.fixed_b_prefix_depths
            for layer in ("layer1", "layer2", "layer3")
            for state in ("v_mem", "g_e", "res", "inh_trace", "u", "x")
        }
        missing = sorted(required - set(artifact.arrays))
        if missing:
            raise KeyError(f"fixed-B history bank missing arrays: {missing}")
        if len(artifact.tables["history_specs"]) != 3 * n_selected * n_depths:
            raise ValueError("fixed-B history bank selected membership mismatch")
    elif task_id == TASK_FIXED_B_REPLAY_BANK:
        _require_members(artifact, tables=("replay_manifest", "b_event_features"))
        required = {f"replay_k{k}" for k in ctx.cfg.fixed_b_prefix_depths}
        if required - set(artifact.arrays):
            raise KeyError("fixed-B replay bank is missing a prefix-depth array")
        if len(artifact.tables["replay_manifest"]) != n_anchors * n_depths:
            raise ValueError("fixed-B replay manifest row-count mismatch")
    elif task_id == TASK_FIXED_B_ROLLOUT_BANK:
        _require_members(
            artifact,
            tables=(
                "rollout_rows",
                "state_trajectory_rows",
                "layer2_event_manifest",
            ),
            arrays=(
                "delta_layer1_ux",
                "delta_layer2_ux",
                "delta_layer2_g",
                "delta_layer3_ux",
                "class_scores",
                "layer1_drive_features",
                "layer1_voltage_features",
                "layer1_inhibition_features",
                "layer1_event_features",
                "layer2_presynaptic_event_bits",
            ),
        )
        expected_rows = 9 * n_selected * n_anchors * n_depths
        if len(artifact.tables["rollout_rows"]) != expected_rows:
            raise ValueError(f"fixed-B rollout row-count mismatch: expected={expected_rows}, found={len(artifact.tables['rollout_rows'])}")
        if len(artifact.tables["state_trajectory_rows"]) != 4 * expected_rows:
            raise ValueError("fixed-B state-trajectory row-count mismatch")
        for array_name in (
            "delta_layer1_ux",
            "delta_layer2_ux",
            "delta_layer2_g",
            "delta_layer3_ux",
            "class_scores",
            "layer1_drive_features",
            "layer1_voltage_features",
            "layer1_inhibition_features",
            "layer1_event_features",
        ):
            _validate_row_alignment(artifact, "rollout_rows", "rollout_row_id", array_name)
        expected_hashes = parents[TASK_FIXED_B_INPUT_BANK].tables["input_manifest"].set_index("b_anchor_id")["tensor_sha256"]
        found = artifact.tables["rollout_rows"]["b_anchor_id"].map(expected_hashes)
        if not artifact.tables["rollout_rows"]["exact_b_tensor_sha256"].eq(found).all():
            raise ValueError("fixed-B rollout exact-B hash identity mismatch")
        expected_event_rows = 2 * n_selected * n_anchors * n_depths
        if len(artifact.tables["layer2_event_manifest"]) != expected_event_rows:
            raise ValueError(
                "fixed-B actual Layer2-presynaptic event membership mismatch"
            )
    elif task_id == TASK_FIXED_B_SWAP_BANK:
        _require_members(
            artifact,
            tables=("swap_rows", "swap_isolation_audit"),
            arrays=(
                "delta_layer2_ux",
                "class_scores",
                "class_scores_early",
                "class_scores_b_end",
                "class_scores_post",
                "layer1_voltage_features",
                "layer1_event_features",
                "layer1_drive_features",
            ),
        )
        expected_rows = n_depths * n_selected * 2 * 4 * n_anchors
        if len(artifact.tables["swap_rows"]) != expected_rows:
            raise ValueError("fixed-B swap row-count mismatch")
        for array_name in (
            "delta_layer2_ux",
            "class_scores",
            "class_scores_early",
            "class_scores_b_end",
            "class_scores_post",
            "layer1_voltage_features",
            "layer1_event_features",
            "layer1_drive_features",
        ):
            _validate_row_alignment(artifact, "swap_rows", "swap_row_id", array_name)
        if not artifact.tables["swap_isolation_audit"]["fast_state_equalized"].eq(1).all():
            raise RuntimeError("fixed-B swap fast-state isolation failed")
        if not artifact.tables["swap_isolation_audit"]["layer1_donor_stsp_applied"].eq(1).all():
            raise RuntimeError("fixed-B Layer1 donor STSP application failed")
        layer1_only = artifact.tables["swap_isolation_audit"].loc[
            artifact.tables["swap_isolation_audit"]["swap_scope"].eq(
                "layer1_only"
            )
        ]
        if not layer1_only["receiver_layer2_3_stsp_preserved"].eq(1).all():
            raise RuntimeError(
                "fixed-B Layer1-only receiver Layer2/3 STSP isolation failed"
            )
    else:
        raise ValueError(f"No validator for fixed-B task {task_id}")


def _require_members(
    artifact: FixedBArtifact,
    *,
    tables: tuple[str, ...] = (),
    arrays: tuple[str, ...] = (),
    payloads: tuple[str, ...] = (),
) -> None:
    for name in tables:
        if name not in artifact.tables:
            raise KeyError(f"Missing fixed-B table {name!r} in {artifact.root}")
    for name in arrays:
        if name not in artifact.arrays:
            raise KeyError(f"Missing fixed-B array {name!r} in {artifact.root}")
    for name in payloads:
        if name not in artifact.payloads:
            raise KeyError(f"Missing fixed-B JSON payload {name!r} in {artifact.root}")


def _validate_row_alignment(
    artifact: FixedBArtifact,
    table_name: str,
    id_column: str,
    array_name: str,
) -> None:
    table = artifact.tables[table_name]
    if table[id_column].astype(int).tolist() != list(range(len(table))):
        raise ValueError(f"fixed-B row ids are not contiguous for {table_name}")
    if artifact.arrays[array_name].shape[0] != len(table):
        raise ValueError(f"fixed-B table/array row mismatch for {table_name}/{array_name}")


__all__ = ["run_fixed_b_task"]
