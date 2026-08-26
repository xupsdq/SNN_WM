"""Dataclass defaults for the Masse delayed-cue LIF experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping


CLASS_FIXATION = 0
CLASS_NONMATCH = 1
CLASS_MATCH = 2

RULE_DMS = "DMS"
RULE_DMRS90 = "DMRS90"
RULES = (RULE_DMS, RULE_DMRS90)

N_DIRECTIONS = 8
N_MOTION_CHANNELS = 24
N_RULE_CHANNELS = 6
N_INPUT = N_MOTION_CHANNELS + N_RULE_CHANNELS
N_OUTPUT = 3

FIXATION_MS = 500.0
SAMPLE_MS = 500.0
PRE_RULE_DELAY_MS = 500.0
RULE_CUE_MS = 250.0
POST_RULE_DELAY_MS = 250.0
TEST_MS = 500.0
GRACE_MS = 50.0
TRIAL_MS = 2500.0

SAMPLE_START_MS = FIXATION_MS
SAMPLE_STOP_MS = SAMPLE_START_MS + SAMPLE_MS
RULE_START_MS = SAMPLE_STOP_MS + PRE_RULE_DELAY_MS
RULE_STOP_MS = RULE_START_MS + RULE_CUE_MS
TEST_START_MS = RULE_STOP_MS + POST_RULE_DELAY_MS
TEST_STOP_MS = TEST_START_MS + TEST_MS
DELAY_DECODE_MS = 100.0
DELAY_DECODE_START_MS = TEST_START_MS - DELAY_DECODE_MS
DELAY_DECODE_STOP_MS = TEST_START_MS

KAPPA = 2.0
TUNING_HEIGHT = 4.0
SFA_RATIO = 0.25
DT_MS = 2.0
TAU_MEM_MS = 20.0
TAU_SYN_MS = 800.0
TAU_SFA_MS = 400.0
TAU_READOUT_MS = 20.0
V_THRESHOLD = 1.0
V_RESET = 0.0
SFA_BETA = 1.6
SURROGATE_ALPHA = 10.0
REFRACTORY_STEPS = 3
TEST_LOSS_WEIGHT = 2.0
ADAM_LR = 3e-4
OVERFIT_LR = 3e-3
TRIAL_TABLE_SEED = 0
MODEL_INIT_SEED = 1
TRAIN_ORDER_SEED = 2
SPIKE_COST = 2e-2
NO_STSP_RECURRENT_SCALE = 1.0 / 3.0
STSP_U_FAC = 0.15
STSP_U_DEP = 0.45
STSP_TAU_X_FAC_MS = 200.0
STSP_TAU_U_FAC_MS = 1500.0
STSP_TAU_X_DEP_MS = 1500.0
STSP_TAU_U_DEP_MS = 200.0
PROFILE_CHOICES = (
    "smoke",
    "formal",
    "overfit",
    "stripped_no_stsp",
    "stripped_stsp",
)


@dataclass(frozen=True)
class MasseDelayedCueConfig:
    profile: str
    n_hidden: int
    n_train: int
    n_val: int
    n_test: int
    batch_size: int
    max_epochs: int
    learning_rate: float
    device: str = "cuda"
    dt_ms: float = DT_MS
    trial_ms: float = TRIAL_MS
    sfa_ratio: float = SFA_RATIO
    tau_mem_ms: float = TAU_MEM_MS
    tau_syn_ms: float = TAU_SYN_MS
    tau_sfa_ms: float = TAU_SFA_MS
    tau_readout_ms: float = TAU_READOUT_MS
    input_gain: float = 1.0
    early_stopping_patience: int | None = None
    trial_table_seed: int = TRIAL_TABLE_SEED
    model_init_seed: int = MODEL_INIT_SEED
    train_order_seed: int = TRAIN_ORDER_SEED
    input_noise: bool = False
    use_synaptic_current: bool = True
    use_stsp: bool = False
    recurrent_weight_scale: float = 1.0
    spike_cost: float = 0.0

    @property
    def stripped(self) -> bool:
        return not self.use_synaptic_current

    @property
    def n_sfa(self) -> int:
        return int(round(self.n_hidden * self.sfa_ratio))

    @property
    def n_steps(self) -> int:
        return int(round(self.trial_ms / self.dt_ms))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["n_sfa"] = self.n_sfa
        payload["n_steps"] = self.n_steps
        payload["n_input"] = N_INPUT
        payload["n_output"] = N_OUTPUT
        return payload


def smoke_config(**overrides: Any) -> MasseDelayedCueConfig:
    values: dict[str, Any] = dict(
        profile="smoke",
        n_hidden=64,
        n_train=128,
        n_val=64,
        n_test=64,
        batch_size=64,
        max_epochs=5,
        learning_rate=ADAM_LR,
        device="cuda",
    )
    values.update(overrides)
    return MasseDelayedCueConfig(**values)


def formal_config(**overrides: Any) -> MasseDelayedCueConfig:
    values: dict[str, Any] = dict(
        profile="formal",
        n_hidden=500,
        n_train=1024,
        n_val=256,
        n_test=256,
        batch_size=64,
        max_epochs=100,
        learning_rate=ADAM_LR,
        device="cuda",
        early_stopping_patience=20,
    )
    values.update(overrides)
    return MasseDelayedCueConfig(**values)


def _stripped_shared(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = dict(
        n_hidden=500,
        n_train=1024,
        n_val=256,
        n_test=256,
        batch_size=64,
        max_epochs=100,
        learning_rate=ADAM_LR,
        device="cuda",
        early_stopping_patience=20,
        sfa_ratio=0.0,
        use_synaptic_current=False,
        spike_cost=SPIKE_COST,
    )
    values.update(overrides)
    return values


def stripped_no_stsp_config(**overrides: Any) -> MasseDelayedCueConfig:
    values = _stripped_shared(
        profile="stripped_no_stsp",
        use_stsp=False,
        recurrent_weight_scale=NO_STSP_RECURRENT_SCALE,
        **overrides,
    )
    return MasseDelayedCueConfig(**values)


def stripped_stsp_config(**overrides: Any) -> MasseDelayedCueConfig:
    values = _stripped_shared(
        profile="stripped_stsp",
        use_stsp=True,
        recurrent_weight_scale=1.0,
        **overrides,
    )
    return MasseDelayedCueConfig(**values)


def overfit_config(**overrides: Any) -> MasseDelayedCueConfig:
    values: dict[str, Any] = dict(
        profile="overfit",
        n_hidden=32,
        n_train=32,
        n_val=32,
        n_test=32,
        batch_size=32,
        max_epochs=150,
        learning_rate=OVERFIT_LR,
        device="cuda",
        input_gain=2.0,
        early_stopping_patience=None,
    )
    values.update(overrides)
    return MasseDelayedCueConfig(**values)


def profile_config(profile: str, **overrides: Any) -> MasseDelayedCueConfig:
    builders = {
        "smoke": smoke_config,
        "formal": formal_config,
        "overfit": overfit_config,
        "stripped_no_stsp": stripped_no_stsp_config,
        "stripped_stsp": stripped_stsp_config,
    }
    if profile not in builders:
        raise ValueError(f"Unknown profile {profile!r}; expected {', '.join(PROFILE_CHOICES)}")
    return builders[profile](**overrides)


def config_from_mapping(payload: Mapping[str, Any]) -> MasseDelayedCueConfig:
    field_names = MasseDelayedCueConfig.__dataclass_fields__.keys()
    values = {key: payload[key] for key in field_names if key in payload}
    return MasseDelayedCueConfig(**values)


def with_overrides(config: MasseDelayedCueConfig, **overrides: Any) -> MasseDelayedCueConfig:
    return replace(config, **overrides)
