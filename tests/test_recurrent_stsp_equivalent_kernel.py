from pathlib import Path

import numpy as np
import pytest
import torch

from src.experiments.recurrent_stsp import (
    DelayRingBuffer,
    IafPscExpParameters,
    IafPscExpPropagators,
    Tsodyks3Parameters,
    iaf_psc_exp_step,
    make_iaf_psc_exp_state,
    make_tsodyks3_state,
    run_tsodyks3_reference_protocol,
    tsodyks3_on_pre_spike,
)


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_VOLTAGE = (
    ROOT
    / "third_party"
    / "working_memory_spiking_network"
    / "test_synapse_model"
    / "comparison_tsodyks3_NESTML"
    / "voltage_data.dat"
)


def test_tsodyks3_facilitates_before_release_and_depletes_after_release():
    params = Tsodyks3Parameters(
        u=0.2, tau_rec_ms=200.0, tau_fac_ms=1500.0, weight=100.0
    )
    state = make_tsodyks3_state((), params, initial_u=0.2, initial_x=1.0)

    state, released_current = tsodyks3_on_pre_spike(state, params, 59.3)

    assert released_current.item() == pytest.approx(36.0, abs=1e-14)
    assert state.u.item() == pytest.approx(0.36, abs=1e-14)
    assert state.x.item() == pytest.approx(0.64, abs=1e-14)
    assert state.last_spike_time_ms.item() == pytest.approx(59.3, abs=1e-14)


def test_tsodyks3_keeps_independent_per_connection_state():
    params = Tsodyks3Parameters(
        u=torch.tensor([0.1, 0.2], dtype=torch.float64),
        tau_rec_ms=torch.tensor([100.0, 200.0], dtype=torch.float64),
        tau_fac_ms=torch.tensor([500.0, 1500.0], dtype=torch.float64),
        weight=torch.tensor([2.0, 3.0], dtype=torch.float64),
    )
    state = make_tsodyks3_state((2,), params, initial_u=params.u)

    next_state, released_current = tsodyks3_on_pre_spike(
        state, params, torch.tensor([10.0, 20.0], dtype=torch.float64)
    )

    assert next_state.u.shape == (2,)
    assert next_state.x.shape == (2,)
    assert not torch.equal(next_state.u[0], next_state.u[1])
    assert not torch.equal(released_current[0], released_current[1])


def test_delay_ring_buffer_delivers_after_exact_positive_delay():
    buffer = DelayRingBuffer((2,), max_delay_steps=3, dtype=torch.float64)
    buffer.schedule(torch.tensor([1.0, 2.0], dtype=torch.float64), delay_steps=2)

    assert torch.equal(buffer.pop_current(), torch.zeros(2, dtype=torch.float64))
    buffer.advance()
    assert torch.equal(buffer.pop_current(), torch.zeros(2, dtype=torch.float64))
    buffer.advance()
    assert torch.equal(
        buffer.pop_current(), torch.tensor([1.0, 2.0], dtype=torch.float64)
    )


def test_reference_protocol_matches_pinned_nest_tsodyks3_voltage_trace():
    upstream = np.loadtxt(UPSTREAM_VOLTAGE)
    result = run_tsodyks3_reference_protocol(dtype=torch.float64)

    number_of_recorded_steps = upstream.shape[1]
    np.testing.assert_allclose(
        result.times_ms[:number_of_recorded_steps].cpu().numpy(),
        upstream[0],
        rtol=0.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        result.target_voltage_mv[:number_of_recorded_steps].cpu().numpy(),
        upstream[3],
        rtol=0.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        result.source_spike_times_ms[:8].cpu().numpy(),
        np.array([59.3, 120.6, 181.9, 243.2, 304.5, 365.8, 427.1, 488.4]),
        rtol=0.0,
        atol=2e-12,
    )


def _run_short_kernel_sequence(device):
    params = IafPscExpParameters(
        tau_m=15.0,
        c_m=250.0,
        t_ref=2.0,
        e_l=0.0,
        i_e=0.0,
        v_th=20.0,
        v_reset=0.0,
        tau_syn_ex=2.0,
        tau_syn_in=2.0,
        dt=0.05,
    )
    props = IafPscExpPropagators.from_parameters(params)
    state = make_iaf_psc_exp_state(
        (32,), params, device=device, dtype=torch.float64
    )
    drive = torch.linspace(300.0, 500.0, 32, dtype=torch.float64, device=device)
    for step in range(128):
        incoming = torch.zeros(32, dtype=torch.float64, device=device)
        if step in (4, 17, 65):
            incoming = torch.linspace(
                1.0, 20.0, 32, dtype=torch.float64, device=device
            )
        state, _ = iaf_psc_exp_step(
            state,
            params,
            propagators=props,
            incoming_spikes_ex=incoming,
            constant_current=drive,
        )
    assert state.i_0.shape == (32,)
    assert state.i_1.shape == (32,)
    return state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_iaf_psc_exp_cpu_and_cuda_paths_agree():
    cpu_state = _run_short_kernel_sequence("cpu")
    cuda_state = _run_short_kernel_sequence("cuda")

    torch.testing.assert_close(
        cuda_state.v_m_relative.cpu(), cpu_state.v_m_relative, rtol=0.0, atol=2e-12
    )
    torch.testing.assert_close(
        cuda_state.i_syn_ex.cpu(), cpu_state.i_syn_ex, rtol=0.0, atol=2e-12
    )
    assert torch.equal(cuda_state.refractory_count.cpu(), cpu_state.refractory_count)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tsodyks3_cpu_and_cuda_paths_agree():
    def run(device):
        params = Tsodyks3Parameters(
            u=torch.linspace(0.1, 0.3, 64, dtype=torch.float64, device=device),
            tau_rec_ms=torch.linspace(
                100.0, 300.0, 64, dtype=torch.float64, device=device
            ),
            tau_fac_ms=torch.linspace(
                500.0, 1500.0, 64, dtype=torch.float64, device=device
            ),
            weight=torch.linspace(1.0, 5.0, 64, dtype=torch.float64, device=device),
        )
        state = make_tsodyks3_state((64,), params, device=device, initial_u=params.u)
        releases = []
        for spike_time in (10.0, 70.0, 120.0):
            state, released = tsodyks3_on_pre_spike(state, params, spike_time)
            releases.append(released)
        return state, torch.stack(releases)

    cpu_state, cpu_releases = run("cpu")
    cuda_state, cuda_releases = run("cuda")
    torch.testing.assert_close(cuda_state.u.cpu(), cpu_state.u, rtol=0.0, atol=2e-12)
    torch.testing.assert_close(cuda_state.x.cpu(), cpu_state.x, rtol=0.0, atol=2e-12)
    torch.testing.assert_close(
        cuda_releases.cpu(), cpu_releases, rtol=0.0, atol=2e-12
    )


def test_equivalent_kernel_rejects_invalid_time_and_parameters():
    with pytest.raises(ValueError, match="tau_m"):
        IafPscExpParameters(tau_m=0.0)
    with pytest.raises(ValueError, match="tau_rec_ms"):
        Tsodyks3Parameters(tau_rec_ms=0.0)

    params = Tsodyks3Parameters()
    state = make_tsodyks3_state((), params, initial_last_spike_time_ms=10.0)
    with pytest.raises(ValueError, match="must not precede"):
        tsodyks3_on_pre_spike(state, params, 9.0)
