from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    repo_root: Path
    dataset_root: Path
    results_root: Path
    model_path: Path

    def to_json_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_default_path_config() -> PathConfig:
    repo_root = get_repo_root()
    results_root = repo_root / "results"
    return PathConfig(
        repo_root=repo_root,
        dataset_root=repo_root / "MNIST",
        results_root=results_root,
        model_path=results_root / "sdnn_deep_final" / "net_final.pth",
    )


DEFAULT_PATH_CONFIG = build_default_path_config()

