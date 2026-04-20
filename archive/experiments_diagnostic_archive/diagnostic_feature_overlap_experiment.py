from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
    archive_root = Path(__file__).resolve().parent
    repo_root = archive_root.parents[1]
    experiment_dir = archive_root / "src" / "experiments"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(experiment_dir) not in sys.path:
        sys.path.insert(0, str(experiment_dir))
    target = experiment_dir / "diagnostic_feature_overlap_experiment.py"
    spec = importlib.util.spec_from_file_location("archive_diagnostic_feature_overlap_experiment", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load archive module: {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
