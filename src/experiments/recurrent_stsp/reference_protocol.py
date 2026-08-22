"""Reproduce the upstream NEST 3.1 single-synapse validation protocol."""

from dataclasses import dataclass
from typing import Union

import torch

from .nest_equivalent import (
    DelayRingBuffer,
    IafPscExpParameters,
    IafPscExpPropagators,
    Tsodyks3Parameters,
    iaf_psc_exp_step,
    make_iaf_psc_exp_state,
    make_tsodyks3_state,
    tsodyks3_on_pre_spike,
)


@dataclass(frozen=True)
class ReferenceProtocolResult:
    """Trace and event outputs produced by the single-synapse protocol."""

    times_ms: torch.Tensor
    target_voltage_mv: torch.Tensor
    source_spike_times_ms: torch.Tensor
    released_currents_pa: torch.Tensor


@torch.no_grad()
def run_tsodyks3_reference_protocol(
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ReferenceProtocolResult:
    """Run the 500 ms burst, 1000 ms pause, 500 ms recovery protocol.

    This mirrors ``comparison_tsodyks3_NESTML/evaluate_...py`` from the pinned
    upstream source: NEST's default ``iaf_psc_exp`` parameters, 0.1 ms grid,
    100 pA absolute weight, and a 1 ms synaptic delay.
    """

    neuron_params = IafPscExpParameters(dt=0.1, tau_syn_ex=2.0)
    propagators = IafPscExpPropagators.from_parameters(neuron_params)
    source_state = make_iaf_psc_exp_state(
        (), neuron_params, device=device, dtype=dtype
    )
    target_state = make_iaf_psc_exp_state(
        (), neuron_params, device=device, dtype=dtype
    )

    stsp_params = Tsodyks3Parameters(
        u=0.2,
        tau_rec_ms=200.0,
        tau_fac_ms=1500.0,
        weight=100.0,
    )
    stsp_state = make_tsodyks3_state(
        (), stsp_params, device=device, dtype=dtype, initial_u=0.2, initial_x=1.0
    )
    delay_steps = 10
    delayed_current = DelayRingBuffer(
        (), delay_steps, device=device, dtype=dtype
    )

    number_of_steps = 20_000
    voltage_trace = torch.empty(number_of_steps, dtype=dtype, device=device)
    spike_times = []
    released_currents = []

    for step in range(number_of_steps):
        incoming_current = delayed_current.pop_current()
        source_current = 376.0 if step < 5_000 or step >= 15_000 else 0.0
        source_state, source_spiked = iaf_psc_exp_step(
            source_state,
            neuron_params,
            propagators=propagators,
            constant_current=source_current,
        )
        if bool(source_spiked.item()):
            spike_time = (step + 1) * neuron_params.dt
            stsp_state, release = tsodyks3_on_pre_spike(
                stsp_state, stsp_params, spike_time
            )
            delayed_current.schedule(release, delay_steps)
            spike_times.append(spike_time)
            released_currents.append(release)

        target_state, _ = iaf_psc_exp_step(
            target_state,
            neuron_params,
            propagators=propagators,
            incoming_spikes_ex=incoming_current,
        )
        voltage_trace[step] = target_state.absolute_voltage(neuron_params)
        delayed_current.advance()

    times = torch.arange(
        1, number_of_steps + 1, dtype=dtype, device=device
    ) * neuron_params.dt
    return ReferenceProtocolResult(
        times_ms=times,
        target_voltage_mv=voltage_trace,
        source_spike_times_ms=torch.as_tensor(
            spike_times, dtype=dtype, device=device
        ),
        released_currents_pa=torch.stack(released_currents),
    )
