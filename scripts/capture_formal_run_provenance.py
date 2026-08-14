from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL_PATH_GLOB = "results/multi_snn/sdnn_ensemble_20/sdnn_ensemble_20/seed_*/net_final.pth"
DEFAULT_OUTPUT_ROOT = "results/paper_figure_multi_seed"
DEFAULT_DATASET_ROOT = "MNIST"
DEFAULT_HASH_ROOTS = (
    "src/experiments/paper_figures",
    "src/experiments/common",
    "src/plotting/paper_fig",
    "scripts",
    "configs",
)
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}


def _run(cmd: list[str], *, cwd: Path) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    except Exception as exc:
        return 127, "", repr(exc)
    return completed.returncode, completed.stdout, completed.stderr


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def _write_hash_csv(path: Path, files: Iterable[Path], *, repo_root: Path) -> int:
    rows = []
    for item in sorted({file.resolve() for file in files}, key=lambda value: str(value).lower()):
        if not item.exists() or not item.is_file():
            continue
        stat = item.stat()
        try:
            rel = item.relative_to(repo_root)
        except ValueError:
            rel = item
        rows.append(
            {
                "path": str(rel).replace("\\", "/"),
                "sha256": _sha256(item),
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "size_bytes", "mtime_utc"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _default_formal_command(python_exe: str, output_root: str, model_glob: str, dataset_root: str, device: str) -> str:
    return (
        f'& "{python_exe}" -m src.experiments.paper_figures.run_paper_figures '
        f'--figs all --scope both --all-seeds --model-path-glob "{model_glob}" '
        f'--dataset-root "{dataset_root}" --output-root "{output_root}" '
        f"--device {device} --force --no-progress"
    )


def _default_statistics_command(python_exe: str, paper_fig_root: str, output_dir: str) -> str:
    return (
        f'& "{python_exe}" scripts\\compute_manuscript_statistics.py '
        f'--paper-fig-root "{paper_fig_root}" --output-dir "{output_dir}" --min-networks 20'
    )


def capture(args: argparse.Namespace) -> dict[str, object]:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root)
    provenance_dir = Path(args.provenance_dir) if args.provenance_dir else output_root / "_provenance"
    provenance_dir = provenance_dir.resolve()
    provenance_dir.mkdir(parents=True, exist_ok=True)

    git_commit_code, git_commit, git_commit_err = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    git_status_code, git_status, git_status_err = _run(["git", "status", "--porcelain=v1"], cwd=repo_root)
    git_branch_code, git_branch, git_branch_err = _run(["git", "branch", "--show-current"], cwd=repo_root)
    diff_code, diff_stdout, diff_stderr = _run(["git", "diff", "--", "src", "scripts", "configs", "docs"], cwd=repo_root)

    _write_text(provenance_dir / "git_head.txt", git_commit)
    _write_text(provenance_dir / "git_branch.txt", git_branch)
    _write_text(provenance_dir / "git_status_porcelain.txt", git_status)
    _write_text(provenance_dir / "git_diff_src_scripts_configs_docs.patch", diff_stdout)

    pip_freeze_status = None
    if not args.no_pip_freeze:
        pip_code, pip_stdout, pip_stderr = _run([sys.executable, "-m", "pip", "freeze"], cwd=repo_root)
        pip_freeze_status = {"returncode": pip_code, "stderr": pip_stderr.strip()}
        _write_text(provenance_dir / "pip_freeze.txt", pip_stdout)

    checkpoint_files = [Path(item) for item in glob.glob(str(repo_root / args.model_path_glob))]
    checkpoint_count = _write_hash_csv(provenance_dir / "checkpoint_sha256.csv", checkpoint_files, repo_root=repo_root)

    hash_roots = [repo_root / value for value in args.hash_root]
    source_count = _write_hash_csv(
        provenance_dir / "source_config_sha256.csv",
        (file for root in hash_roots for file in _iter_files(root)),
        repo_root=repo_root,
    )

    dataset_count = None
    if args.include_dataset_hashes:
        dataset_count = _write_hash_csv(
            provenance_dir / "dataset_sha256.csv",
            _iter_files(repo_root / args.dataset_root),
            repo_root=repo_root,
        )

    formal_command = args.formal_run_command or _default_formal_command(
        args.python_executable,
        args.output_root,
        args.model_path_glob,
        args.dataset_root,
        args.device,
    )
    statistics_command = args.statistics_command or _default_statistics_command(
        args.python_executable,
        args.paper_fig_root,
        args.statistics_output_dir,
    )
    _write_text(provenance_dir / "formal_run_command.txt", formal_command + "\n")
    _write_text(provenance_dir / "statistics_command.txt", statistics_command + "\n")

    payload: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "output_root": str((repo_root / output_root).resolve()),
        "provenance_dir": str(provenance_dir),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable_current": sys.executable,
        "python_executable_intended": args.python_executable,
        "python_version": sys.version,
        "git": {
            "commit": git_commit.strip() if git_commit_code == 0 else None,
            "branch": git_branch.strip() if git_branch_code == 0 else None,
            "status_porcelain": git_status,
            "is_dirty": bool(git_status.strip()),
            "errors": {
                "commit": git_commit_err.strip(),
                "branch": git_branch_err.strip(),
                "status": git_status_err.strip(),
                "diff": diff_stderr.strip(),
            },
            "returncodes": {
                "commit": git_commit_code,
                "branch": git_branch_code,
                "status": git_status_code,
                "diff": diff_code,
            },
        },
        "inputs": {
            "model_path_glob": args.model_path_glob,
            "checkpoint_count": checkpoint_count,
            "dataset_root": args.dataset_root,
            "dataset_hash_count": dataset_count,
            "source_config_hash_count": source_count,
            "hash_roots": list(args.hash_root),
        },
        "commands": {
            "formal_run": formal_command,
            "statistics": statistics_command,
            "dry_run": formal_command.replace(" --force ", " --dry-run "),
        },
        "pip_freeze": pip_freeze_status,
    }
    (provenance_dir / "formal_run_provenance.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture provenance files before a formal paper-figure run.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--provenance-dir", default=None)
    parser.add_argument("--model-path-glob", default=DEFAULT_MODEL_PATH_GLOB)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--paper-fig-root", default="results/paper_figures/outputs")
    parser.add_argument("--statistics-output-dir", default="results/paper_figures/statistics_final")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hash-root", action="append", default=list(DEFAULT_HASH_ROOTS))
    parser.add_argument("--include-dataset-hashes", action="store_true")
    parser.add_argument("--no-pip-freeze", action="store_true")
    parser.add_argument("--formal-run-command", default=None)
    parser.add_argument("--statistics-command", default=None)
    payload = capture(parser.parse_args())
    print(f"Wrote formal run provenance to {payload['provenance_dir']}")
    print(f"Checkpoint files hashed: {payload['inputs']['checkpoint_count']}")
    print(f"Source/config files hashed: {payload['inputs']['source_config_hash_count']}")


if __name__ == "__main__":
    main()
