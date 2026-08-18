from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config.defaults import DEFAULT_PROJECT_DEFAULTS

# --- Experiment identity -------------------------------------------------
# Manuscript figure numbering is the only user-visible numbering. The runtime
# id below is code-traceability only.
EXPERIMENT_ID = "manuscript_fig6b_order_specificity"
MANUSCRIPT_FIGURE = "Fig.6b"

NUM_CLASSES = 10
ITEM_ROLES = ("A", "B", "C", "D")
SEQUENCE_LENGTH = 4

# The six historical orders: the fixed item set {A, B, C} + latest D. Only the
# A/B/C -> slot assignment varies; D always occupies slot 4.
ORDER_NAMES = (
    "A-B-C-D",
    "A-C-B-D",
    "B-A-C-D",
    "B-C-A-D",
    "C-A-B-D",
    "C-B-A-D",
)
ORDER_PERMUTATIONS = (
    ("A", "B", "C"),
    ("A", "C", "B"),
    ("B", "A", "C"),
    ("B", "C", "A"),
    ("C", "A", "B"),
    ("C", "B", "A"),
)
N_ORDERS = len(ORDER_NAMES)
CHANCE_ACCURACY = 1.0 / float(N_ORDERS)

# Fixed stimulus-spec seed: identical controlled stimulus specifications for
# every network, independent of network seed.
STIMULUS_SPEC_SEED = 20260814

PILOT_NETWORK_SEEDS = (1000, 1001, 1002)
FORMAL_NETWORK_SEEDS = tuple(range(1000, 1020))
ANALYSIS_SCOPES = ("pilot", "formal")
DEFAULT_DATASET_ROOT = str(DEFAULT_PROJECT_DEFAULTS.paths.dataset_root)


@dataclass(frozen=True)
class OrderSpecificityConfig:
    """Configuration for fixed-set, fixed-latest temporal-order analysis."""

    output_dir: str
    task: str = "all"
    analysis_scope: str = "pilot"
    reuse_artifacts: str = "auto"  # off | auto | require
    network_seed: int | None = None
    model_path: str = ""
    model_path_glob: str = "results/multi_snn/sdnn_ensemble_20/sdnn_ensemble_20/seed_*/net_final.pth"
    dataset_root: str = DEFAULT_DATASET_ROOT
    device: str = "auto"
    split: str = "test"
    dt: float = 0.001
    sample_ms: int = 200
    delay_ms: int = 200
    sequence_length: int = SEQUENCE_LENGTH
    num_sets: int = 12
    num_orders: int = N_ORDERS
    stimulus_spec_seed: int = STIMULUS_SPEC_SEED
    batch_size: int = 6
    n_permutation_draws: int = 200
    n_tiebreak_draws: int = 1000
    smoke: bool = False
    show_progress: bool = True
    # --- Pre-registered GO/BORDERLINE/STOP thresholds --------------------
    gate_mean_accuracy_go: float = 0.50
    gate_borderline_low: float = 0.33
    gate_confusion_diagonal_ratio: float = 2.0

    @property
    def sample_steps(self) -> int:
        return max(1, int(round((float(self.sample_ms) * 0.001) / float(self.dt))))

    @property
    def delay_steps(self) -> int:
        return max(1, int(round((float(self.delay_ms) * 0.001) / float(self.dt))))

    @property
    def expected_network_seeds(self) -> tuple[int, ...]:
        if self.analysis_scope == "pilot":
            return PILOT_NETWORK_SEEDS
        if self.analysis_scope == "formal":
            return FORMAL_NETWORK_SEEDS
        raise ValueError(
            f"Unknown analysis_scope={self.analysis_scope!r}; expected one of {ANALYSIS_SCOPES}"
        )


@dataclass
class SimulationContext:
    cfg: OrderSpecificityConfig
    device: object
    net: object
    encoder: object
    dataset: object
    class_index: dict[int, list[int]]
    warnings: list[str] = field(default_factory=list)


__all__ = [
    "ANALYSIS_SCOPES",
    "CHANCE_ACCURACY",
    "EXPERIMENT_ID",
    "FORMAL_NETWORK_SEEDS",
    "ITEM_ROLES",
    "MANUSCRIPT_FIGURE",
    "N_ORDERS",
    "NUM_CLASSES",
    "ORDER_NAMES",
    "ORDER_PERMUTATIONS",
    "PILOT_NETWORK_SEEDS",
    "SEQUENCE_LENGTH",
    "STIMULUS_SPEC_SEED",
    "OrderSpecificityConfig",
    "SimulationContext",
]
