"""Configuration for the sparse Tiddia recurrent network backend."""

from dataclasses import dataclass, replace
import math
from typing import Optional


@dataclass(frozen=True)
class TiddiaNetworkConfig:
    """Scientific and storage parameters of the recurrent network graph."""

    n_exc: int = 8_000
    n_inh: int = 2_000
    coding_fraction: float = 0.10
    n_memories: int = 5
    connection_probability: float = 0.20
    gamma_0: float = 0.10
    facilitated_fraction: float = 1.0
    allow_autapses: bool = True
    allow_multapses: bool = True
    seed: int = 143_202_461

    dt_ms: float = 0.05
    delay_min_ms: float = 0.10
    delay_max_ms: float = 1.00
    capacitance_pf: float = 250.0
    tau_syn_ms: float = 2.0

    tau_m_exc_ms: float = 15.0
    tau_m_inh_ms: float = 10.0
    refractory_exc_ms: float = 2.0
    refractory_inh_ms: float = 2.0
    threshold_exc_mv: float = 20.0
    threshold_inh_mv: float = 20.0
    reset_exc_mv: float = 0.0
    reset_inh_mv: float = 0.0
    resting_exc_mv: float = 0.0
    resting_inh_mv: float = 0.0

    j_e_to_i_mv: float = 0.135
    j_i_to_e_mv: float = 0.25
    j_i_to_i_mv: float = 0.20
    j_baseline_mv: float = 0.10
    j_potentiated_mv: float = 0.45
    j_baseline_lognormal_std_mv: Optional[float] = None
    j_potentiated_lognormal_std_mv: Optional[float] = None

    stsp_u: float = 0.19
    stsp_initial_u: float = 0.19
    stsp_initial_x: float = 1.0
    stsp_tau_fac_ms: float = 1_500.0
    stsp_tau_rec_ms: float = 200.0
    stsp_initial_u_truncnorm_std: Optional[float] = None
    stsp_initial_x_uniform: bool = False
    stsp_tau_fac_truncnorm_std_ms: Optional[float] = None
    stsp_tau_rec_truncnorm_std_ms: Optional[float] = None
    nonfacilitating_tau_ms: float = 325.0
    nonfacilitating_tau_std_ms: float = 50.0

    sampling_target_chunk: int = 128

    def __post_init__(self) -> None:
        if self.n_exc <= 0 or self.n_inh <= 0:
            raise ValueError("n_exc and n_inh must be positive.")
        if self.n_memories <= 0:
            raise ValueError("n_memories must be positive.")
        for name in (
            "coding_fraction",
            "connection_probability",
            "facilitated_fraction",
            "gamma_0",
            "stsp_u",
            "stsp_initial_u",
            "stsp_initial_x",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("{} must lie in [0, 1].".format(name))
        if self.n_memories * self.coding_fraction > 1.0:
            raise ValueError("Selective populations exceed the excitatory population.")
        for name in (
            "dt_ms",
            "delay_min_ms",
            "delay_max_ms",
            "capacitance_pf",
            "tau_syn_ms",
            "tau_m_exc_ms",
            "tau_m_inh_ms",
            "stsp_tau_fac_ms",
            "stsp_tau_rec_ms",
            "nonfacilitating_tau_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError("{} must be finite and positive.".format(name))
        if self.delay_min_ms > self.delay_max_ms:
            raise ValueError("delay_min_ms must not exceed delay_max_ms.")
        if self.min_delay_steps < 1:
            raise ValueError("Synaptic delay must be at least one simulation step.")
        if self.max_delay_steps > 32_767:
            raise ValueError("Delay exceeds the int16 storage range.")
        if self.selective_population_size <= 0 or self.nonselective_population_size <= 0:
            raise ValueError("Both selective and non-selective populations must be non-empty.")
        if self.sampling_target_chunk <= 0:
            raise ValueError("sampling_target_chunk must be positive.")
        for name in (
            "j_e_to_i_mv",
            "j_i_to_e_mv",
            "j_i_to_i_mv",
            "j_baseline_mv",
            "j_potentiated_mv",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} must be finite and non-negative.".format(name))
        for name in (
            "j_baseline_lognormal_std_mv",
            "j_potentiated_lognormal_std_mv",
            "stsp_initial_u_truncnorm_std",
            "stsp_tau_fac_truncnorm_std_ms",
            "stsp_tau_rec_truncnorm_std_ms",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("{} must be None or finite and positive.".format(name))

    @property
    def n_neurons(self) -> int:
        return self.n_exc + self.n_inh

    @property
    def selective_population_size(self) -> int:
        return int(self.coding_fraction * self.n_exc)

    @property
    def nonselective_population_size(self) -> int:
        return self.n_exc - self.n_memories * self.selective_population_size

    @property
    def min_delay_steps(self) -> int:
        return int(math.floor(self.delay_min_ms / self.dt_ms + 0.5))

    @property
    def max_delay_steps(self) -> int:
        return int(math.floor(self.delay_max_ms / self.dt_ms + 0.5))

    @classmethod
    def heterogeneous_run_config(cls, **overrides) -> "TiddiaNetworkConfig":
        """Return the effective settings hard-coded by upstream ``run_model.py``."""

        config = cls(
            facilitated_fraction=0.9,
            allow_autapses=True,
            allow_multapses=False,
            j_baseline_lognormal_std_mv=0.08,
            stsp_initial_u_truncnorm_std=0.04,
            stsp_initial_x_uniform=True,
            stsp_tau_fac_truncnorm_std_ms=None,
            stsp_tau_rec_truncnorm_std_ms=None,
        )
        return replace(config, **overrides)
