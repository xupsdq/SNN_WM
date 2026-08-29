from __future__ import annotations

"""Canonical runtime entrypoint for the Fig.6b order-specificity pilot.

DAG: persisted stimulus specs -> validated reusable state-bank artifacts ->
downstream analysis -> plot-only panels (src/plotting/experiments/
manuscript_fig6b_order_specificity_plot.py).

Tasks:
- sequence_specs: build + validate + persist data/sequence_specs.csv and
  data/singleton_reference_specs.csv (shared by all networks).
- state_bank: simulate one network; persist data/intermediates/seed_<n>/.
- analysis: load-only for parents; runs the pre-registered leave-one-set-out
  candidate matching; writes metrics, caption, and summary.
- all: sequence_specs + state_bank (with --network-seed) or + analysis
  (without --network-seed).
"""


import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.common.results import (
    prepare_result_layout,
    save_log_lines,
    save_run_config,
    save_summary_json,
)
from src.experiments.paper_figures.fig6b_order_specificity.analysis import run_analysis
from src.experiments.paper_figures.fig6b_order_specificity.artifacts import (
    ArtifactState,
    artifact_dir_for,
    bank_digest,
    cache_key_digest,
    copy_bank_to_bundle,
    load_bank_artifact,
    load_specs_artifact,
    save_bank_artifact,
    save_specs_artifact,
)
from src.experiments.paper_figures.fig6b_order_specificity.formal_spec import (
    FORMAL_SPEC_PATH,
    FORMAL_SPEC_SHA256_PATH,
    load_frozen_formal_spec,
)
from src.experiments.paper_figures.fig6b_order_specificity.simulation import (
    build_dataset_index,
    build_simulation_context,
    capture_network_state_bank,
)
from src.experiments.paper_figures.fig6b_order_specificity.specs import (
    build_stimulus_specs,
    specs_cache_key,
    specs_digest,
    validate_stimulus_specs,
)
from src.experiments.paper_figures.fig6b_order_specificity.types import (
    ANALYSIS_SCOPES,
    DEFAULT_DATASET_ROOT,
    EXPERIMENT_ID,
    OrderSpecificityConfig,
)

TASK_SEQUENCE_SPECS = "sequence_specs"
TASK_STATE_BANK = "state_bank"
TASK_ANALYSIS = "analysis"
TASK_ALL = "all"
TASK_IDS = (TASK_SEQUENCE_SPECS, TASK_STATE_BANK, TASK_ANALYSIS, TASK_ALL)
REUSE_MODES = ("off", "auto", "require")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _config_from_args(args)
    layout = prepare_result_layout(cfg.output_dir)
    artifact_state = ArtifactState()
    logs: list[str] = []
    run_info = {
        "experiment_id": EXPERIMENT_ID,
        "analysis_scope": str(cfg.analysis_scope),
        "task": str(cfg.task),
        "reuse_artifacts": str(cfg.reuse_artifacts),
        "network_seed": cfg.network_seed,
        "output_dir": str(Path(cfg.output_dir).resolve()),
        "model_path": str(cfg.model_path),
        "smoke": bool(cfg.smoke),
    }
    if cfg.analysis_scope == "formal":
        frozen_spec = load_frozen_formal_spec()
        formal_spec_target = layout.meta_dir / "formal_analysis_spec.json"
        formal_digest_target = layout.meta_dir / "formal_analysis_spec.sha256"
        formal_spec_target.write_bytes(FORMAL_SPEC_PATH.read_bytes())
        formal_digest_target.write_bytes(FORMAL_SPEC_SHA256_PATH.read_bytes())
        run_info["formal_spec_schema"] = str(frozen_spec["schema"])
        run_info["formal_spec_sha256"] = FORMAL_SPEC_SHA256_PATH.read_text(
            encoding="utf-8"
        ).split()[0]
    (layout.meta_dir / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    try:
        sequence_specs: pd.DataFrame | None = None
        reference_specs: pd.DataFrame | None = None
        if cfg.task in (TASK_SEQUENCE_SPECS, TASK_STATE_BANK, TASK_ANALYSIS, TASK_ALL):
            sequence_specs, reference_specs, spec_digest = _get_sequence_specs(cfg, layout, artifact_state, logs)
        if cfg.task in (TASK_STATE_BANK, TASK_ALL) and cfg.network_seed is not None:
            _get_state_bank(cfg, layout, artifact_state, logs)
        if cfg.task in (TASK_ANALYSIS, TASK_ALL) and cfg.network_seed is None:
            bank_dirs = _require_network_banks(cfg, layout)
            summary = run_analysis(
                cfg,
                layout,
                sequence_specs,
                reference_specs,
                bank_dirs,
                logs=logs,
            )
            save_run_config(asdict(cfg), layout.root)
            save_summary_json(summary, layout.root)
        _finalize_bundle(cfg, layout, artifact_state, logs)
        save_log_lines(logs, layout.logs_dir)
        return 0
    except Exception as exc:
        logs.append(f"FAILED: {type(exc).__name__}: {exc}")
        save_log_lines(logs, layout.logs_dir)
        raise


def _get_sequence_specs(
    cfg: OrderSpecificityConfig,
    layout: Any,
    artifact_state: ArtifactState,
    logs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    artifact_dir = artifact_dir_for(layout.root, TASK_SEQUENCE_SPECS)
    expected_key = specs_cache_key(cfg)
    if cfg.reuse_artifacts in ("auto", "require"):
        try:
            sequence_specs, reference_specs = load_specs_artifact(artifact_dir, expected_key, expected_digest="")
            digest = specs_digest(cfg, sequence_specs, reference_specs)
            _verify_stored_digest(artifact_dir, digest, "sequence-specs")
            audit = validate_stimulus_specs(cfg, sequence_specs, reference_specs)
            audit.to_csv(layout.metrics_dir / "stimulus_spec_audit.csv", index=False, encoding="utf-8")
            _write_specs_to_bundle(layout, sequence_specs, reference_specs)
            artifact_state.record(TASK_SEQUENCE_SPECS, "loaded", artifact_dir, digest, expected_key)
            logs.append(f"sequence_specs source=loaded digest={digest[:16]}")
            return sequence_specs, reference_specs, digest
        except Exception as exc:
            if cfg.reuse_artifacts == "require":
                raise RuntimeError(
                    f"require mode: sequence-specs artifact unavailable or invalid in {artifact_dir}: {exc}"
                )
            logs.append(f"sequence_specs cache miss ({type(exc).__name__}); rebuilding")
    _, class_index = build_dataset_index(cfg)
    sequence_specs, reference_specs = build_stimulus_specs(cfg, class_index)
    digest = specs_digest(cfg, sequence_specs, reference_specs)
    audit = validate_stimulus_specs(cfg, sequence_specs, reference_specs)
    audit.to_csv(layout.metrics_dir / "stimulus_spec_audit.csv", index=False, encoding="utf-8")
    _write_specs_to_bundle(layout, sequence_specs, reference_specs)
    if cfg.reuse_artifacts != "off":
        save_specs_artifact(artifact_dir, sequence_specs, reference_specs, expected_key, digest=digest)
        artifact_state.record(TASK_SEQUENCE_SPECS, "built", artifact_dir, digest, expected_key)
    logs.append(f"sequence_specs source=built digest={digest[:16]}")
    return sequence_specs, reference_specs, digest


def _verify_stored_digest(artifact_dir: Path, digest: str, label: str) -> None:
    import json as _json

    digest_path = artifact_dir / "digest.json"
    if digest_path.exists():
        stored = str(_json.loads(digest_path.read_text(encoding="utf-8")).get("digest", ""))
        if stored != digest:
            raise RuntimeError(f"{label} artifact digest mismatch in {artifact_dir}: expected {digest}, found {stored}")


def _write_specs_to_bundle(layout: Any, sequence_specs: pd.DataFrame, reference_specs: pd.DataFrame) -> None:
    sequence_specs.to_csv(layout.data_file("sequence_specs.csv"), index=False, encoding="utf-8")
    reference_specs.to_csv(layout.data_file("singleton_reference_specs.csv"), index=False, encoding="utf-8")


def _get_state_bank(
    cfg: OrderSpecificityConfig,
    layout: Any,
    artifact_state: ArtifactState,
    logs: list[str],
) -> Path:
    if cfg.network_seed is None:
        raise RuntimeError("state_bank task requires --network-seed")
    network_seed = int(cfg.network_seed)
    artifact_dir = artifact_dir_for(layout.root, TASK_STATE_BANK, network_seed)
    spec_digest = specs_digest(
        cfg,
        pd.read_csv(layout.data_file("sequence_specs.csv")),
        pd.read_csv(layout.data_file("singleton_reference_specs.csv")),
    )
    expected_key = {
        "network_seed": network_seed,
        "model_path": str(cfg.model_path),
        "device": str(cfg.device),
        "sample_ms": int(cfg.sample_ms),
        "delay_ms": int(cfg.delay_ms),
        "dt": float(cfg.dt),
        "sequence_length": int(cfg.sequence_length),
        "split": str(cfg.split),
        "specs_digest": spec_digest,
    }
    bundle_bank_dir = layout.root / "data" / "intermediates" / f"seed_{network_seed}"
    if cfg.reuse_artifacts in ("auto", "require"):
        try:
            load_bank_artifact(artifact_dir, expected_key, expected_digest="")
            digest = bank_digest(artifact_dir)
            _verify_stored_digest(artifact_dir, digest, f"state-bank seed {network_seed}")
            copy_bank_to_bundle(artifact_dir, bundle_bank_dir)
            artifact_state.record(TASK_STATE_BANK, "loaded", artifact_dir, digest, expected_key)
            logs.append(f"state_bank seed={network_seed} source=loaded digest={digest[:16]}")
            return artifact_dir
        except Exception as exc:
            if cfg.reuse_artifacts == "require":
                raise RuntimeError(
                    f"require mode: state-bank artifact unavailable or invalid for seed {network_seed} in {artifact_dir}: {exc}"
                )
            logs.append(f"state_bank seed={network_seed} cache miss ({type(exc).__name__}); rebuilding")

    sequence_specs = pd.read_csv(layout.data_file("sequence_specs.csv"))
    reference_specs = pd.read_csv(layout.data_file("singleton_reference_specs.csv"))
    ctx = build_simulation_context(cfg)
    bank_dir = Path("tmp") / "manuscript_fig6b_order_specificity" / f"seed_{network_seed}"
    capture_network_state_bank(ctx, sequence_specs, reference_specs, bank_dir)
    digest = bank_digest(bank_dir)
    copy_bank_to_bundle(bank_dir, bundle_bank_dir)
    if cfg.reuse_artifacts != "off":
        save_bank_artifact(artifact_dir, bank_dir, expected_key, digest=digest)
        artifact_state.record(TASK_STATE_BANK, "built", artifact_dir, digest, expected_key)
    logs.append(f"state_bank seed={network_seed} source=built digest={digest[:16]}")
    return bank_dir


def _require_network_banks(cfg: OrderSpecificityConfig, layout: Any) -> dict[int, Path]:
    """Load-only parent resolution: missing/stale banks fail loudly."""
    seeds = cfg.expected_network_seeds if not cfg.smoke else (1000,)
    bank_dirs: dict[int, Path] = {}
    for network_seed in seeds:
        bundle_bank_dir = layout.root / "data" / "intermediates" / f"seed_{network_seed}"
        bank_path = bundle_bank_dir / "state_bank_layer2.npz"
        meta_path = bundle_bank_dir / "sequence_meta.csv"
        if not bank_path.exists() or not meta_path.exists():
            raise RuntimeError(
                f"analysis requires network bank for seed {network_seed}; missing in {bundle_bank_dir}"
            )
        bank_dirs[int(network_seed)] = bundle_bank_dir
    return bank_dirs


def _finalize_bundle(
    cfg: OrderSpecificityConfig,
    layout: Any,
    artifact_state: ArtifactState,
    logs: list[str],
) -> None:
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "task": str(cfg.task),
        "network_seed": cfg.network_seed,
        "reuse_artifacts": str(cfg.reuse_artifacts),
        "files": {
            path.relative_to(layout.root).as_posix(): _sha256_file(path)
            for path in sorted(layout.root.rglob("*"))
            if path.is_file() and path.name != "artifact_manifest.json"
        },
    }
    (layout.root / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logs.append(f"artifact_manifest written: {len(manifest['files'])} files")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _config_from_args(args: argparse.Namespace) -> OrderSpecificityConfig:
    model_path = str(args.model_path) if args.model_path else ""
    if not model_path and args.network_seed is not None and not args.smoke:
        from src.experiments.paper_figures.run_paper_figures import discover_checkpoints

        checkpoints = discover_checkpoints(str(args.model_path_glob))
        by_seed = {int(item.seed): str(item.model_path) for item in checkpoints}
        if int(args.network_seed) not in by_seed:
            raise FileNotFoundError(
                f"No checkpoint for network seed {args.network_seed} matched {args.model_path_glob}. "
                f"Known seeds: {sorted(by_seed)}"
            )
        model_path = by_seed[int(args.network_seed)]
    if args.smoke:
        num_sets = min(int(args.num_sets), 2)
    else:
        num_sets = int(args.num_sets)
    return OrderSpecificityConfig(
        output_dir=str(Path(args.output_dir).resolve()),
        task=str(args.task),
        analysis_scope=str(args.analysis_scope),
        reuse_artifacts=str(args.reuse_artifacts),
        network_seed=int(args.network_seed) if args.network_seed is not None else None,
        model_path=model_path,
        model_path_glob=str(args.model_path_glob),
        dataset_root=str(args.dataset_root),
        device=str(args.device),
        split=str(args.split),
        sample_ms=int(args.sample_ms),
        delay_ms=int(args.delay_ms),
        num_sets=num_sets,
        batch_size=max(1, int(args.batch_size)),
        n_permutation_draws=max(10, int(args.n_permutation_draws)),
        n_tiebreak_draws=max(10, int(args.n_tiebreak_draws)),
        smoke=bool(args.smoke),
        show_progress=not bool(args.no_progress),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fig.6b fixed-set, fixed-latest temporal-order identification.",
        allow_abbrev=False,
    )
    parser.add_argument("--task", required=True, choices=TASK_IDS)
    parser.add_argument("--analysis-scope", default="pilot", choices=ANALYSIS_SCOPES)
    parser.add_argument("--reuse-artifacts", default="auto", choices=REUSE_MODES)
    parser.add_argument("--output-dir", required=True, help="Candidate bundle root (e.g. results/paper_figure_candidates/manuscript_fig6b_order_specificity_pilot)")
    parser.add_argument("--network-seed", type=int, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-path-glob", default="results/multi_snn/sdnn_ensemble_20/sdnn_ensemble_20/seed_*/net_final.pth")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay-ms", type=int, default=200)
    parser.add_argument("--num-sets", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--n-permutation-draws", type=int, default=200)
    parser.add_argument("--n-tiebreak-draws", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.task == TASK_STATE_BANK and args.network_seed is None:
        parser.error("--network-seed is required for the state_bank task")
    if args.task == TASK_ANALYSIS and args.network_seed is not None:
        parser.error("--network-seed must be omitted for the analysis task")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
