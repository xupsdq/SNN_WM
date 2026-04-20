from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch

from src.experiments.common.runtime import seed_everything

DEFAULT_MODEL_PATH = str(Path("results") / "sdnn_deep_final" / "net_final.pth")
DEFAULT_DATASET_ROOT = "./MNIST"


def resolve_device_strict(device_arg: str = "auto") -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def build_common_parser(description: str, default_output_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--smoke", action="store_true")
    return parser


def setup_logger(log_path: str | Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def format_smoke_command(module_name: str, output_dir: str | Path, extra_args: Sequence[str] | None = None) -> str:
    cmd = ["conda", "run", "-n", "torch_env", "python", "-m", module_name, "--device", "cuda", "--smoke", "--output-dir", str(output_dir)]
    if extra_args:
        cmd.extend(str(item) for item in extra_args)
    return subprocess.list2cmdline(cmd)


def run_python_module(
    module_name: str,
    args: Iterable[str],
    logger: logging.Logger,
    *,
    cwd: str | Path | None = None,
) -> None:
    command = [sys.executable, "-m", module_name, *[str(item) for item in args]]
    logger.info("[Stage] Running module: %s", subprocess.list2cmdline(command))
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        logger.info("[Stage stdout]\n%s", completed.stdout.rstrip())
    if completed.stderr:
        logger.info("[Stage stderr]\n%s", completed.stderr.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(f"Module {module_name} failed with exit code {completed.returncode}")


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_MODEL_PATH",
    "build_common_parser",
    "format_smoke_command",
    "resolve_device_strict",
    "run_python_module",
    "seed_everything",
    "setup_logger",
]

