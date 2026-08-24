"""PyTorch implementation of the NEST 3.1 neuron and STSP event kernels.

The functions in this module reproduce the update order used by the Tiddia
working-memory model. State is explicit and per synapse, so the same kernels
can later be used by a sparse recurrent event scheduler without changing the
scientific equations.
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple, Union

import torch


TensorOrScalar = Union[torch.Tensor, float]


def _require_floating_dtype(dtype: torch.dtype) -> None:
    if not dtype.is_floating_point:
        raise TypeError("The equivalent kernels require a floating-point dtype.")


def _validate_scalar_or_tensor(
    name: str,
    value: TensorOrScalar,
    *,
    positive: bool = False,
    unit_interval: bool = False,
) -> None:
    tensor = torch.as_tensor(value)
    if tensor.numel() == 0:
        raise ValueError("{} must not be empty.".format(name))
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("{} must contain only finite values.".format(name))
    if positive and bool((tensor <= 0).any().item()):
        raise ValueError("{} must be strictly positive.".format(name))
    if unit_interval and bool(((tensor < 0) | (tensor > 1)).any().item()):
        raise ValueError("{} must lie in [0, 1].".format(name))


def _as_like(value: TensorOrScalar, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _broadcast_like(value: TensorOrScalar, reference: torch.Tensor) -> torch.Tensor:
    tensor = _as_like(value, reference)
    try:
        return torch.broadcast_to(tensor, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            "A supplied value is not broadcastable to state shape {}.".format(
                tuple(reference.shape)
            )
        ) from exc


@dataclass(frozen=True)
class IafPscExpParameters:
    """Parameters of deterministic NEST 3.1 ``iaf_psc_exp``.

    Voltages are expressed in mV, currents in pA, capacitance in pF, and time
    values in ms. ``delta`` escape noise is outside this kernel because the
    upstream working-memory model uses deterministic thresholding.
    """

    tau_m: float = 10.0
    c_m: float = 250.0
    t_ref: float = 2.0
    e_l: float = -70.0
    i_e: float = 0.0
    v_th: float = -55.0
    v_reset: float = -70.0
    tau_syn_ex: float = 2.0
    tau_syn_in: float = 2.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        for name in ("tau_m", "c_m", "tau_syn_ex", "tau_syn_in", "dt"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError("{} must be finite and strictly positive.".format(name))
        if not math.isfinite(self.t_ref) or self.t_ref < 0:
            raise ValueError("t_ref must be finite and non-negative.")
        for name in ("e_l", "i_e", "v_th", "v_reset"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError("{} must be finite.".format(name))
        if self.v_reset >= self.v_th:
            raise ValueError("v_reset must be smaller than v_th.")


def _propagator_32(tau_syn: float, tau_m: float, c_m: float, dt: float) -> float:
    """Return NEST's numerically stable current-to-voltage propagator."""

    singular = dt / c_m * math.exp(-dt / tau_m)
    if tau_m == tau_syn:
        regular = singular
    else:
        regular = (
            -tau_m
            / (c_m * (1.0 - tau_m / tau_syn))
            * math.exp(-dt / tau_syn)
            * math.expm1(dt * (1.0 / tau_syn - 1.0 / tau_m))
        )
    linear = (
        dt
        * dt
        * (tau_syn - tau_m)
        * math.exp(-dt / tau_m)
        / (2.0 * c_m * tau_m * tau_m)
    )
    if tau_m == tau_syn or (
        abs(tau_m - tau_syn) < 0.1
        and abs(regular - singular) > 2.0 * abs(linear)
    ):
        return singular
    return regular


@dataclass(frozen=True)
class IafPscExpPropagators:
    """Precomputed one-step factors used by ``iaf_psc_exp_step``."""

    p11_ex: float
    p11_in: float
    p22: float
    p21_ex: float
    p21_in: float
    p20: float
    refractory_steps: int

    @classmethod
    def from_parameters(cls, params: IafPscExpParameters) -> "IafPscExpPropagators":
        refractory_steps = int(math.floor(params.t_ref / params.dt + 0.5))
        return cls(
            p11_ex=math.exp(-params.dt / params.tau_syn_ex),
            p11_in=math.exp(-params.dt / params.tau_syn_in),
            p22=math.exp(-params.dt / params.tau_m),
            p21_ex=_propagator_32(
                params.tau_syn_ex, params.tau_m, params.c_m, params.dt
            ),
            p21_in=_propagator_32(
                params.tau_syn_in, params.tau_m, params.c_m, params.dt
            ),
            p20=params.tau_m
            / params.c_m
            * (1.0 - math.exp(-params.dt / params.tau_m)),
            refractory_steps=refractory_steps,
        )


@dataclass
class IafPscExpState:
    """Tensor state matching the internal NEST representation.

    ``v_m_relative`` is relative to ``E_L`` exactly as in NEST 3.1. ``i_0``
    is an unfiltered step-current state and ``i_1`` is filtered with the
    excitatory synaptic time constant.
    """

    v_m_relative: torch.Tensor
    i_syn_ex: torch.Tensor
    i_syn_in: torch.Tensor
    i_0: torch.Tensor
    i_1: torch.Tensor
    refractory_count: torch.Tensor

    def absolute_voltage(self, params: IafPscExpParameters) -> torch.Tensor:
        return self.v_m_relative + params.e_l


def make_iaf_psc_exp_state(
    shape: Sequence[int],
    params: IafPscExpParameters,
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float64,
    initial_voltage_mv: Optional[float] = None,
) -> IafPscExpState:
    """Create a zero-current neuron state on the requested device."""

    _require_floating_dtype(dtype)
    state_shape = tuple(shape)
    voltage = params.e_l if initial_voltage_mv is None else float(initial_voltage_mv)
    if not math.isfinite(voltage):
        raise ValueError("initial_voltage_mv must be finite.")
    zeros = torch.zeros(state_shape, dtype=dtype, device=device)
    return IafPscExpState(
        v_m_relative=torch.full(
            state_shape, voltage - params.e_l, dtype=dtype, device=device
        ),
        i_syn_ex=zeros.clone(),
        i_syn_in=zeros.clone(),
        i_0=zeros.clone(),
        i_1=zeros.clone(),
        refractory_count=torch.zeros(state_shape, dtype=torch.int64, device=device),
    )


def iaf_psc_exp_step(
    state: IafPscExpState,
    params: IafPscExpParameters,
    *,
    propagators: Optional[IafPscExpPropagators] = None,
    incoming_spikes_ex: TensorOrScalar = 0.0,
    incoming_spikes_in: TensorOrScalar = 0.0,
    incoming_current_0: TensorOrScalar = 0.0,
    incoming_current_1: TensorOrScalar = 0.0,
    constant_current: Optional[TensorOrScalar] = None,
    spike_override: Optional[TensorOrScalar] = None,
) -> Tuple[IafPscExpState, torch.Tensor]:
    """Advance deterministic ``iaf_psc_exp`` by one grid step.

    The operation order follows NEST 3.1: propagate voltage, decay currents,
    add events arriving in the current slot, test threshold, then install the
    step-current values for the following step. An arriving spike therefore
    changes the PSC immediately and membrane voltage on the next grid step.
    ``constant_current`` replaces ``params.i_e`` for this step when supplied.
    """

    props = propagators or IafPscExpPropagators.from_parameters(params)
    drive = params.i_e if constant_current is None else constant_current
    drive_tensor = _as_like(drive, state.v_m_relative)

    active = state.refractory_count == 0
    evolved_voltage = (
        state.v_m_relative * props.p22
        + state.i_syn_ex * props.p21_ex
        + state.i_syn_in * props.p21_in
        + (drive_tensor + state.i_0) * props.p20
    )
    voltage = torch.where(active, evolved_voltage, state.v_m_relative)
    refractory_count = torch.clamp(state.refractory_count - 1, min=0)

    i_syn_ex = (
        state.i_syn_ex * props.p11_ex
        + (1.0 - props.p11_ex) * state.i_1
        + _as_like(incoming_spikes_ex, state.i_syn_ex)
    )
    i_syn_in = (
        state.i_syn_in * props.p11_in
        + _as_like(incoming_spikes_in, state.i_syn_in)
    )

    threshold_spikes = voltage >= (params.v_th - params.e_l)
    if spike_override is None:
        spiked = threshold_spikes
    else:
        spiked = _broadcast_like(spike_override, voltage).to(dtype=torch.bool)
    voltage = torch.where(
        spiked,
        torch.as_tensor(
            params.v_reset - params.e_l,
            dtype=voltage.dtype,
            device=voltage.device,
        ),
        voltage,
    )
    refractory_count = torch.where(
        spiked,
        torch.as_tensor(
            props.refractory_steps,
            dtype=refractory_count.dtype,
            device=refractory_count.device,
        ),
        refractory_count,
    )

    return (
        IafPscExpState(
            v_m_relative=voltage,
            i_syn_ex=i_syn_ex,
            i_syn_in=i_syn_in,
            i_0=_broadcast_like(incoming_current_0, state.i_0).clone(),
            i_1=_broadcast_like(incoming_current_1, state.i_1).clone(),
            refractory_count=refractory_count,
        ),
        spiked,
    )


@dataclass(frozen=True)
class Tsodyks3Parameters:
    """Per-synapse parameters of the Tiddia/NEST ``tsodyks3`` rule.

    The static ``u`` field corresponds to the NEST/NESTML parameter ``U``;
    the event-dependent utilization remains in ``Tsodyks3State.u``.
    """

    u: TensorOrScalar = 0.19
    tau_rec_ms: TensorOrScalar = 200.0
    tau_fac_ms: TensorOrScalar = 1500.0
    weight: TensorOrScalar = 1.0

    def __post_init__(self) -> None:
        _validate_scalar_or_tensor("u", self.u, unit_interval=True)
        _validate_scalar_or_tensor("tau_rec_ms", self.tau_rec_ms, positive=True)
        _validate_scalar_or_tensor("tau_fac_ms", self.tau_fac_ms, positive=True)
        _validate_scalar_or_tensor("weight", self.weight)


@dataclass
class Tsodyks3State:
    """Dynamic state stored independently for every plastic connection."""

    u: torch.Tensor
    x: torch.Tensor
    last_spike_time_ms: torch.Tensor


def make_tsodyks3_state(
    shape: Sequence[int],
    params: Tsodyks3Parameters,
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float64,
    initial_u: Optional[TensorOrScalar] = None,
    initial_x: TensorOrScalar = 1.0,
    initial_last_spike_time_ms: TensorOrScalar = 0.0,
) -> Tsodyks3State:
    """Create independent dynamic variables for a tensor of synapses."""

    _require_floating_dtype(dtype)
    state_shape = tuple(shape)
    u_value = params.u if initial_u is None else initial_u
    _validate_scalar_or_tensor("initial_u", u_value, unit_interval=True)
    _validate_scalar_or_tensor("initial_x", initial_x, unit_interval=True)
    _validate_scalar_or_tensor("initial_last_spike_time_ms", initial_last_spike_time_ms)

    def expanded(value: TensorOrScalar) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
        try:
            return torch.broadcast_to(tensor, state_shape).clone()
        except RuntimeError as exc:
            raise ValueError(
                "Initial STSP values must be broadcastable to shape {}.".format(
                    state_shape
                )
            ) from exc

    return Tsodyks3State(
        u=expanded(u_value),
        x=expanded(initial_x),
        last_spike_time_ms=expanded(initial_last_spike_time_ms),
    )


def tsodyks3_on_pre_spike(
    state: Tsodyks3State,
    params: Tsodyks3Parameters,
    spike_time_ms: TensorOrScalar,
) -> Tuple[Tsodyks3State, torch.Tensor]:
    """Apply one presynaptic event with exact ``tsodyks3`` update order.

    The emitted current is ``weight * u(t+) * x(t-)``: utilization is
    facilitated before release and resources are depleted after release.
    """

    event_time = _broadcast_like(spike_time_ms, state.last_spike_time_ms)
    elapsed = event_time - state.last_spike_time_ms
    if bool((elapsed < 0).any().item()):
        raise ValueError("spike_time_ms must not precede the previous spike time.")

    u_after_facilitation, x_after_release, released_current = tsodyks3_event_values(
        u=state.u,
        x=state.x,
        elapsed_ms=elapsed,
        baseline_u=params.u,
        tau_rec_ms=params.tau_rec_ms,
        tau_fac_ms=params.tau_fac_ms,
        weight=params.weight,
    )

    return (
        Tsodyks3State(
            u=u_after_facilitation,
            x=x_after_release,
            last_spike_time_ms=event_time.expand_as(state.last_spike_time_ms).clone(),
        ),
        released_current,
    )


def tsodyks3_event_values(
    *,
    u: torch.Tensor,
    x: torch.Tensor,
    elapsed_ms: TensorOrScalar,
    baseline_u: TensorOrScalar,
    tau_rec_ms: TensorOrScalar,
    tau_fac_ms: TensorOrScalar,
    weight: TensorOrScalar,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized tsodyks3 algebra shared by scalar and sparse schedulers."""

    elapsed = _broadcast_like(elapsed_ms, u)
    baseline = _as_like(baseline_u, u)
    tau_rec = _as_like(tau_rec_ms, x)
    tau_fac = _as_like(tau_fac_ms, u)
    edge_weight = _as_like(weight, u)
    x_before_release = 1.0 + (x - 1.0) * torch.exp(-elapsed / tau_rec)
    u_before_facilitation = baseline + (u - baseline) * torch.exp(
        -elapsed / tau_fac
    )
    u_after_facilitation = u_before_facilitation + baseline * (
        1.0 - u_before_facilitation
    )
    released_current = edge_weight * u_after_facilitation * x_before_release
    x_after_release = x_before_release - u_after_facilitation * x_before_release
    return u_after_facilitation, x_after_release, released_current


class DelayRingBuffer:
    """Fixed-grid current buffer with NEST-style positive delays."""

    def __init__(
        self,
        shape: Sequence[int],
        max_delay_steps: int,
        *,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        _require_floating_dtype(dtype)
        if not isinstance(max_delay_steps, int) or max_delay_steps < 1:
            raise ValueError("max_delay_steps must be a positive integer.")
        self.max_delay_steps = max_delay_steps
        self._buffer = torch.zeros(
            (max_delay_steps + 1,) + tuple(shape), dtype=dtype, device=device
        )
        self._cursor = 0

    @property
    def cursor(self) -> int:
        return self._cursor

    def pop_current(self) -> torch.Tensor:
        value = self._buffer[self._cursor].clone()
        self._buffer[self._cursor].zero_()
        return value

    def schedule(self, value: TensorOrScalar, delay_steps: int) -> None:
        if not isinstance(delay_steps, int) or not 1 <= delay_steps <= self.max_delay_steps:
            raise ValueError(
                "delay_steps must be an integer in [1, {}].".format(
                    self.max_delay_steps
                )
            )
        slot = (self._cursor + delay_steps) % self._buffer.shape[0]
        self._buffer[slot].add_(_as_like(value, self._buffer[slot]))

    def advance(self) -> None:
        self._cursor = (self._cursor + 1) % self._buffer.shape[0]

    def reset(self) -> None:
        self._buffer.zero_()
        self._cursor = 0
