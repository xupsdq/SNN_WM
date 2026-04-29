from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from src.training.train_sdnn import TrainingConfig, train_single_network


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_summary_csv(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "seed",
        "status",
        "output_dir",
        "final_accuracy",
        "elapsed_seconds",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})
    return path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train multiple SDNN networks from scratch with different seeds.")
    parser.add_argument("--output-dir", type=str, default="results/sdnn_ensemble")
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-networks", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=28)
    parser.add_argument("--l1-epochs", type=int, default=2)
    parser.add_argument("--l2-epochs", type=int, default=10)
    parser.add_argument("--l3-epochs", type=int, default=500)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--l3-eval-every", type=int, default=100, help="Use 0 to disable periodic L3 evaluation.")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--enable-stsp", action="store_true", help="Enable dynamic STSP during training. Default is disabled.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=1)
    parser.add_argument("--smoke-eval-batches", type=int, default=1)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_started = time.time()
    records: list[dict[str, Any]] = []

    l3_eval_every = None if args.l3_eval_every <= 0 else args.l3_eval_every
    effective_l1_epochs = 1 if args.smoke else args.l1_epochs
    effective_l2_epochs = 1 if args.smoke else args.l2_epochs
    effective_l3_epochs = 1 if args.smoke else args.l3_epochs
    effective_max_batches = args.smoke_batches if args.smoke else args.max_batches
    effective_eval_max_batches = args.smoke_eval_batches if args.smoke else args.eval_max_batches
    effective_l3_eval_every = 1 if args.smoke else l3_eval_every

    ensemble_config = {
        "output_dir": str(root),
        "dataset_root": args.dataset_root,
        "device": args.device,
        "num_networks": int(args.num_networks),
        "seed_start": int(args.seed_start),
        "batch_size": int(args.batch_size),
        "input_size": int(args.input_size),
        "l1_epochs": int(effective_l1_epochs),
        "l2_epochs": int(effective_l2_epochs),
        "l3_epochs": int(effective_l3_epochs),
        "max_batches": effective_max_batches,
        "eval_max_batches": effective_eval_max_batches,
        "l3_eval_every": effective_l3_eval_every,
        "skip_final_eval": bool(args.skip_final_eval),
        "enable_stsp": bool(args.enable_stsp),
        "smoke": bool(args.smoke),
    }
    _write_json(root / "ensemble_run_config.json", ensemble_config)

    for index in range(args.num_networks):
        seed = int(args.seed_start + index)
        run_dir = root / f"seed_{seed:04d}"
        print(f"\n=== [Ensemble] Network {index + 1}/{args.num_networks} | seed={seed} ===")
        started = time.time()
        record: dict[str, Any] = {
            "index": index,
            "seed": seed,
            "status": "failed",
            "output_dir": str(run_dir),
            "final_accuracy": None,
            "elapsed_seconds": None,
            "error": "",
        }
        try:
            summary = train_single_network(
                TrainingConfig(
                    output_dir=str(run_dir),
                    dataset_root=args.dataset_root,
                    device=args.device,
                    seed=seed,
                    batch_size=args.batch_size,
                    input_size=args.input_size,
                    l1_epochs=effective_l1_epochs,
                    l2_epochs=effective_l2_epochs,
                    l3_epochs=effective_l3_epochs,
                    max_batches=effective_max_batches,
                    eval_max_batches=effective_eval_max_batches,
                    l3_eval_every=effective_l3_eval_every,
                    skip_final_eval=args.skip_final_eval,
                    enable_stsp=args.enable_stsp,
                    smoke=args.smoke,
                )
            )
            record["status"] = "success"
            record["final_accuracy"] = summary.get("final_accuracy")
            record["elapsed_seconds"] = summary.get("elapsed_seconds")
        except Exception as exc:
            record["error"] = repr(exc)
            record["elapsed_seconds"] = time.time() - started
            records.append(record)
            _write_summary_csv(root / "ensemble_summary.csv", records)
            _write_json(root / "ensemble_summary.json", {"records": records})
            if not args.continue_on_error:
                raise
        else:
            records.append(record)
            _write_summary_csv(root / "ensemble_summary.csv", records)
            _write_json(root / "ensemble_summary.json", {"records": records})

    total_elapsed = time.time() - run_started
    payload = {
        "status": "success" if all(item["status"] == "success" for item in records) else "failed",
        "elapsed_seconds": total_elapsed,
        "records": records,
    }
    _write_json(root / "ensemble_summary.json", payload)
    _write_summary_csv(root / "ensemble_summary.csv", records)
    print(f"\n[Done] Ensemble finished in {total_elapsed:.1f}s. Summary: {root / 'ensemble_summary.csv'}")


if __name__ == "__main__":
    main()
