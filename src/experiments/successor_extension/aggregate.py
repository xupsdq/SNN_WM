"""Network-level aggregate for the 20-seed confirmatory successor-extension cohort.

The network is the inference unit. Each experiment has exactly two primary
endpoints; per experiment we run a one-sided exact sign-flip test against zero,
a deterministic 20k-draw bootstrap 95% CI, and Holm correction within the
experiment (the two endpoints form one correction family per cohort scope).

Statistics use the shared reproducible-inference seam.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.experiments.common.inference import (
    bootstrap_mean_ci,
    exact_sign_flip_p,
    holm_adjust,
    stable_seed,
)
from src.experiments.successor_extension.core import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TASK_EXP_A,
    TASK_EXP_B,
    TASK_EXP_C,
    TASK_K10_INPUT,
    TASK_K10_SPECS,
)
from src.experiments.successor_extension.runtime import (
    parent_entry,
    resolve_repo_path,
    sha256_file,
    write_json,
)

AGGREGATE_EXPERIMENT_ID = "successor_extension_v1_confirmatory_20seed_aggregate"
AGGREGATE_TASK_ID = "successor_extension_v1_confirmatory_20seed_aggregate"
RANDOM_SEED = 20260726
BOOTSTRAP_DRAWS = 20_000
ALPHA = 0.05
NULL_VALUE = 0.0
ALTERNATIVE = "greater"

SCOPE_FULL20 = "full20"
SCOPE_SENSITIVITY = "sensitivity_1001_1019"

# Exactly two primary endpoints per experiment. value_field names the key in
# each per-seed summary.json endpoint payload that carries the per-network value.
EXPERIMENT_ENDPOINTS: dict[str, dict[str, Any]] = {
    TASK_EXP_A: {
        "label": "c5_k10_successor",
        "value_field": "mean_transfer",
        "endpoints": (
            "early_layer2_event_map_donor_transfer",
            "layer3_successor_ux_donor_transfer",
        ),
    },
    TASK_EXP_B: {
        "label": "k10_l1_overlap_intervention",
        "value_field": "mean_overlap_specific_margin",
        "endpoints": (
            "early_layer2_b_history_contrast_attenuation",
            "post_b_layer2_ux_history_contrast_attenuation",
        ),
    },
    TASK_EXP_C: {
        "label": "c5_twohop_cd",
        "value_field": "mean_donor_transfer",
        "endpoints": (
            "early_layer2_D_donor_transfer",
            "layer3_postD_ux_donor_transfer",
        ),
    },
}

INFERENCE_COLUMNS = (
    "cohort",
    "experiment",
    "endpoint",
    "value_kind",
    "n_networks",
    "network_seeds",
    "mean",
    "median",
    "sd_across_networks",
    "min_network_value",
    "max_network_value",
    "positive_network_fraction",
    "bootstrap_ci95_low",
    "bootstrap_ci95_high",
    "p_one_sided_exact_sign_flip",
    "holm_adjusted_p",
    "independent_unit",
    "primary_pass",
)


def _load_seed_summary(root: Path, seed: int, task: str) -> dict[str, Any]:
    path = root / f"seed_{int(seed)}" / "data" / "metrics" / task / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-seed summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise RuntimeError(f"{path}: status is {summary.get('status')!r}, not 'completed'")
    if int(summary.get("network_seed", -1)) != int(seed):
        raise RuntimeError(f"{path}: network_seed does not match seed_{int(seed)}")
    return summary


def _primary_endpoint_values(
    root: Path, seeds: Sequence[int], task: str
) -> dict[str, np.ndarray]:
    spec = EXPERIMENT_ENDPOINTS[task]
    expected = tuple(spec["endpoints"])
    value_field = str(spec["value_field"])
    collected: dict[str, list[float]] = {endpoint: [] for endpoint in expected}
    for seed in seeds:
        summary = _load_seed_summary(root, int(seed), task)
        endpoints = summary.get("endpoints")
        if not isinstance(endpoints, dict) or not endpoints:
            raise RuntimeError(f"seed {seed} {task}: summary has no endpoint payload")
        primaries: dict[str, Any] = {}
        for name, payload in endpoints.items():
            if task == TASK_EXP_C:
                if payload.get("role") == "primary":
                    primaries[str(name)] = payload
            else:
                primaries[str(name)] = payload
        if set(primaries) != set(expected):
            raise RuntimeError(
                f"seed {seed} {task}: expected exactly the two primary endpoints "
                f"{sorted(expected)}, got {sorted(primaries)}"
            )
        for endpoint in expected:
            value = primaries[endpoint].get(value_field)
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(
                    f"seed {seed} {task}/{endpoint}: {value_field} missing or non-finite: {value!r}"
                )
            collected[endpoint].append(float(value))
    return {endpoint: np.asarray(values, dtype=np.float64) for endpoint, values in collected.items()}


def _verify_coverage(root: Path, seeds: Sequence[int]) -> None:
    missing: list[str] = []
    for seed in seeds:
        for task in EXPERIMENT_ENDPOINTS:
            path = root / f"seed_{int(seed)}" / "data" / "metrics" / task / "summary.json"
            if not path.exists():
                missing.append(f"seed_{seed}/{task}")
    if missing:
        raise RuntimeError(
            f"Aggregate requires exact {len(seeds)}-seed coverage; missing: {missing}"
        )


def _scope_inference(
    root: Path,
    *,
    scope: str,
    seeds: Sequence[int],
    bootstrap_draws: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    effect_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    verdicts: dict[str, Any] = {}
    for task, spec in EXPERIMENT_ENDPOINTS.items():
        values_by_endpoint = _primary_endpoint_values(root, seeds, task)
        for endpoint in spec["endpoints"]:
            values = values_by_endpoint[endpoint]
            for seed, value in zip(seeds, values):
                effect_rows.append(
                    {
                        "cohort": scope,
                        "experiment": task,
                        "network_seed": int(seed),
                        "endpoint": endpoint,
                        "value_kind": spec["value_field"],
                        "value": float(value),
                    }
                )
        endpoint_stats: dict[str, dict[str, float]] = {}
        p_values: dict[str, float] = {}
        for endpoint in spec["endpoints"]:
            values = values_by_endpoint[endpoint]
            low, high = bootstrap_mean_ci(
                values,
                draws=int(bootstrap_draws),
                seed=stable_seed(int(random_seed), scope, task, endpoint),
            )
            p_one = float(exact_sign_flip_p(values - NULL_VALUE, alternative=ALTERNATIVE))
            endpoint_stats[endpoint] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "sd": float(values.std(ddof=1)),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "p_one_sided": p_one,
                "positive_network_fraction": float(np.mean(values > NULL_VALUE)),
            }
            p_values[endpoint] = p_one
        holm = holm_adjust(np.asarray([p_values[endpoint] for endpoint in spec["endpoints"]]))
        for index, endpoint in enumerate(spec["endpoints"]):
            stats = endpoint_stats[endpoint]
            values = values_by_endpoint[endpoint]
            holm_p = float(holm[index])
            inference_rows.append(
                {
                    "cohort": scope,
                    "experiment": task,
                    "endpoint": endpoint,
                    "value_kind": spec["value_field"],
                    "n_networks": int(len(values)),
                    "network_seeds": "|".join(str(int(seed)) for seed in seeds),
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "sd_across_networks": stats["sd"],
                    "min_network_value": float(values.min()),
                    "max_network_value": float(values.max()),
                    "positive_network_fraction": stats["positive_network_fraction"],
                    "bootstrap_ci95_low": stats["ci95_low"],
                    "bootstrap_ci95_high": stats["ci95_high"],
                    "p_one_sided_exact_sign_flip": stats["p_one_sided"],
                    "holm_adjusted_p": holm_p,
                    "independent_unit": "independently_trained_network",
                    "primary_pass": int(holm_p < ALPHA),
                }
            )
        all_holm_significant = bool(all(holm_p < ALPHA for holm_p in holm))
        all_ci_above_zero = bool(
            all(endpoint_stats[endpoint]["ci95_low"] > NULL_VALUE for endpoint in spec["endpoints"])
        )
        verdicts[task] = {
            "scope": scope,
            "experiment": task,
            "n_networks": int(len(seeds)),
            "network_seeds": [int(seed) for seed in seeds],
            "endpoints": endpoint_stats,
            "all_primary_endpoints_holm_significant": all_holm_significant,
            "all_primary_ci95_above_zero": all_ci_above_zero,
            "verdict": (
                "supported"
                if all_holm_significant and all_ci_above_zero
                else "not_supported"
            ),
        }
    effects = pd.DataFrame(effect_rows)
    inference = pd.DataFrame(inference_rows, columns=list(INFERENCE_COLUMNS))
    return effects, inference, verdicts


def run_aggregate(
    *,
    output_root: str | Path,
    seeds: Sequence[int],
    sensitivity_seeds: Sequence[int],
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    root = resolve_repo_path(output_root)
    seeds = tuple(int(value) for value in seeds)
    sensitivity_seeds = tuple(int(value) for value in sensitivity_seeds)
    _verify_coverage(root, seeds)
    _verify_coverage(root, sensitivity_seeds)

    scopes = (
        (SCOPE_FULL20, seeds),
        (SCOPE_SENSITIVITY, sensitivity_seeds),
    )
    for scope, scope_seeds in scopes:
        if len(scope_seeds) > 24:
            raise ValueError("Exact sign-flip is bounded to 24 networks per scope")

    out_dir = root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_verdicts: dict[str, Any] = {}
    saved: dict[str, str] = {}
    for scope, scope_seeds in scopes:
        effects, inference, verdicts = _scope_inference(
            root,
            scope=scope,
            seeds=scope_seeds,
            bootstrap_draws=int(bootstrap_draws),
            random_seed=int(random_seed),
        )
        effects_path = (
            out_dir / "network_effects.csv"
            if scope == SCOPE_FULL20
            else out_dir / f"network_effects_{scope}.csv"
        )
        inference_path = (
            out_dir / "population_inference.csv"
            if scope == SCOPE_FULL20
            else out_dir / f"population_inference_{scope}.csv"
        )
        effects.to_csv(effects_path, index=False, lineterminator="\n")
        inference.to_csv(inference_path, index=False, lineterminator="\n")
        saved[effects_path.name] = str(effects_path)
        saved[inference_path.name] = str(inference_path)
        all_verdicts[scope] = verdicts

    verdict_path = out_dir / "verdict.json"
    write_json(
        verdict_path,
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "experiment_id": AGGREGATE_EXPERIMENT_ID,
            "inference_unit": "independently_trained_network",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "bootstrap_draws": int(bootstrap_draws),
            "random_seed": int(random_seed),
            "null_value": NULL_VALUE,
            "alternative": ALTERNATIVE,
            "scopes": {
                SCOPE_FULL20: [int(seed) for seed in seeds],
                SCOPE_SENSITIVITY: [int(seed) for seed in sensitivity_seeds],
            },
            "verdicts": all_verdicts,
            "claim_boundary": (
                "Network-level inference over the confirmatory successor-extension cohort. "
                "Primary endpoints only; experiment-C first-hop gate and STSP-only endpoints "
                "remain descriptive audit material in the per-seed summaries."
            ),
        },
    )
    saved[verdict_path.name] = str(verdict_path)

    parents: dict[str, Mapping[str, Any]] = {
        TASK_K10_SPECS: parent_entry(root / TASK_K10_SPECS),
        TASK_K10_INPUT: parent_entry(root / TASK_K10_INPUT),
    }
    for seed in seeds:
        for task in EXPERIMENT_ENDPOINTS:
            summary_path = root / f"seed_{int(seed)}" / "data" / "metrics" / task / "summary.json"
            parents[f"seed_{int(seed)}_{task}"] = {
                "path": str(summary_path.resolve()),
                "cache_key_sha256": sha256_file(summary_path),
            }
    manifest_path = out_dir / "task_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "task_id": AGGREGATE_TASK_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "parents": {name: dict(entry) for name, entry in sorted(parents.items())},
            "params": {
                "seeds": [int(seed) for seed in seeds],
                "sensitivity_seeds": [int(seed) for seed in sensitivity_seeds],
                "bootstrap_draws": int(bootstrap_draws),
                "random_seed": int(random_seed),
            },
            "inference_scope": "network_level_cohort_primary_endpoints",
        },
    )
    saved[manifest_path.name] = str(manifest_path)

    summary = {
        "experiment_id": AGGREGATE_EXPERIMENT_ID,
        "status": "completed",
        "inference_unit": "independently_trained_network",
        "seeds": [int(seed) for seed in seeds],
        "sensitivity_seeds": [int(seed) for seed in sensitivity_seeds],
        "experiments": sorted(EXPERIMENT_ENDPOINTS),
        "primary_endpoints_per_experiment": {
            task: list(spec["endpoints"]) for task, spec in EXPERIMENT_ENDPOINTS.items()
        },
        "bootstrap_draws": int(bootstrap_draws),
        "random_seed": int(random_seed),
        "saved_files": dict(sorted(saved.items())),
    }
    summary_path = out_dir / "summary.json"
    write_json(summary_path, summary)
    saved[summary_path.name] = str(summary_path)

    artifact_manifest = {
        "experiment_id": AGGREGATE_EXPERIMENT_ID,
        "title": "20-seed confirmatory successor-extension population aggregate",
        "files": {
            path.name: sha256_file(path)
            for path in sorted(out_dir.rglob("*"))
            if path.is_file() and path.name != "artifact_manifest.json"
        },
    }
    artifact_manifest_path = out_dir / "artifact_manifest.json"
    write_json(artifact_manifest_path, artifact_manifest)
    saved[artifact_manifest_path.name] = str(artifact_manifest_path)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Successor-extension 20-seed population aggregate.")
    parser.add_argument("--output-root", default="results/successor_extension_v1_confirmatory_20seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1020)))
    parser.add_argument("--sensitivity-seeds", type=int, nargs="+", default=list(range(1001, 1020)))
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    payload = run_aggregate(
        output_root=args.output_root,
        seeds=args.seeds,
        sensitivity_seeds=args.sensitivity_seeds,
        bootstrap_draws=args.bootstrap_draws,
        random_seed=RANDOM_SEED,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
