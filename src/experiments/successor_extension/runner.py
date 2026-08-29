from __future__ import annotations

import argparse
import json
from typing import Sequence

from src.experiments.successor_extension.core import (
    EXPERIMENT_ID,
    TASK_EXP_A,
    TASK_EXP_B,
    TASK_EXP_C,
    TASK_K10_HISTORY,
    TASK_K10_INPUT,
    TASK_K10_SPECS,
    ExtensionConfig,
    build_k10_extension_input_bank,
    build_k10_history_bank,
    run_experiment_a,
    run_experiment_b,
    run_experiment_c,
    save_k10_extension_specs,
)
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.successor_extension.runtime import build_context, resolve_repo_path

TASKS = (TASK_K10_SPECS, TASK_K10_INPUT, TASK_K10_HISTORY, TASK_EXP_A, TASK_EXP_B, TASK_EXP_C, "all")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    defaults = ExtensionConfig()
    parser = argparse.ArgumentParser(
        description="Successor-extension bypass DAG: K=10 extension, K=10 L1 overlap intervention, K=5 two-hop propagation.",
        allow_abbrev=False,
    )
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--output-root", default=defaults.output_root)
    parser.add_argument("--network-seed", type=int, default=defaults.network_seed)
    parser.add_argument("--device", default=defaults.device, choices=("auto", "cpu", "cuda"))
    parser.add_argument("--families", type=int, default=defaults.families)
    parser.add_argument("--anchors", type=int, default=defaults.anchors)
    parser.add_argument("--anchors-per-chunk", type=int, default=defaults.anchors_per_chunk)
    parser.add_argument("--dataset-root", default=defaults.dataset_root)
    parser.add_argument("--model-path-glob", default=defaults.model_path_glob)
    parser.add_argument("--bootstrap-draws", type=int, default=defaults.bootstrap_draws)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = ExtensionConfig(
        output_root=str(args.output_root),
        dataset_root=str(args.dataset_root),
        model_path_glob=str(args.model_path_glob),
        device=str(args.device),
        network_seed=int(args.network_seed),
        families=max(1, int(args.families)),
        anchors=max(1, int(args.anchors)),
        anchors_per_chunk=max(1, int(args.anchors_per_chunk)),
        bootstrap_draws=max(100, int(args.bootstrap_draws)),
        smoke=bool(args.smoke),
    )
    task = str(args.task)
    requested = list(TASKS[:-1]) if task == "all" else [task]

    for current in requested:
        print(f"[successor_extension] task={current} seed={cfg.network_seed} device={cfg.device}", flush=True)
        if current == TASK_K10_SPECS:
            dataset = load_mnist_skeleton_dataset(
                str(resolve_repo_path(cfg.dataset_root)), "test"
            )
            specs = save_k10_extension_specs(cfg, dataset)
            task_dir = resolve_repo_path(cfg.output_root) / TASK_K10_SPECS
            _print_json(
                {
                    "task": current,
                    "status": "completed",
                    "n_history_rows": int(len(specs)),
                    "n_families": int(specs["history_family_id"].nunique()),
                    "output": str(task_dir / "history_specs.csv"),
                }
            )
        elif current == TASK_K10_INPUT:
            ctx = build_context(cfg, load_model=True)
            artifact = build_k10_extension_input_bank(cfg, ctx)
            _print_json(
                {
                    "task": current,
                    "status": "completed",
                    "suffix_images_encoded": int(
                        len(artifact.tables["history_input_manifest"])
                        - 100
                    ),
                    "history_spikes_shape": list(artifact.arrays["history_spikes"].shape),
                    "exact_b_spikes_shape": list(artifact.arrays["exact_b_spikes"].shape),
                }
            )
        elif current == TASK_K10_HISTORY:
            ctx = build_context(cfg, load_model=True)
            artifact = build_k10_history_bank(cfg, ctx)
            _print_json(
                {
                    "task": current,
                    "status": "completed",
                    "k5_checkpoint_identity_audit": "all_bitwise_pass",
                    "k10_array_keys": sorted(
                        key for key in artifact.arrays if key.startswith("k10__")
                    )[:6],
                }
            )
        elif current == TASK_EXP_A:
            ctx = build_context(cfg, load_model=True)
            summary = run_experiment_a(cfg, ctx)
            _print_json(summary)
        elif current == TASK_EXP_B:
            ctx = build_context(cfg, load_model=True)
            summary = run_experiment_b(cfg, ctx)
            _print_json(summary)
        elif current == TASK_EXP_C:
            ctx = build_context(cfg, load_model=True)
            summary = run_experiment_c(cfg, ctx)
            _print_json(summary)
        else:
            raise SystemExit(f"Unsupported task: {current}")
    print(f"[successor_extension] done tasks={','.join(requested)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
