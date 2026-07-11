from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    ConservativeScales,
    PaczynskiWiitaPotential,
    PhysicalTransportClosure,
    wind_escape_diagnostics,
    wind_launch_energy,
    conservative_source_terms,
    enthalpy_energy_identity_audit,
    integrate_interval_transport,
    integrate_sampled_interval_transport,
    local_gradient,
    legacy_energy_identity_audit,
    reconstruct_conservative_state,
    simpson_interval_residual,
    stream_source_interval_integral,
    stream_source_prime,
)
from imri_qpe.scales import eddington_mdot
from imri_qpe.units import solar_masses_to_g


@pytest.fixture
def params() -> SimpleNamespace:
    mass = solar_masses_to_g(1.0e4)
    return SimpleNamespace(
        M2_g=mass,
        Mdot_g_s=0.1 * eddington_mdot(mass),
        alpha=0.01,
        mu_stress=0.0,
        stress_factor=1.0,
        mu_mol=0.62,
        kappa=0.34,
        gamma_gas=5.0 / 3.0,
        partial_eps=1.0e-5,
    )


def test_conservative_state_uses_angular_flux_as_primary(params: SimpleNamespace) -> None:
    potential = PaczynskiWiitaPotential(params.M2_g)
    scales = ConservativeScales(params.Mdot_g_s, potential.l_k(potential.r_isco))
    state = reconstruct_conservative_state(
        np.log(30.0 * potential.r_g),
        np.log(1.0e7),
        np.log(2.0e6),
        1.2,
        0.93,
        params,
        scales,
    )

    assert state.mdot == pytest.approx(1.2 * scales.mdot)
    assert state.J == pytest.approx(0.93 * scales.angular_flux)
    assert state.mdot * state.l - state.G == pytest.approx(state.J)
    assert state.mechanical_energy_flux == pytest.approx(
        state.mdot * state.bernoulli - state.Omega * state.G
    )


def test_physical_closure_keeps_mass_torque_and_power_separate(params: SimpleNamespace) -> None:
    potential = PaczynskiWiitaPotential(params.M2_g)
    scales = ConservativeScales(params.Mdot_g_s, potential.l_k(potential.r_isco))
    state = reconstruct_conservative_state(
        np.log(40.0 * potential.r_g),
        np.log(8.0e6),
        np.log(1.8e6),
        1.0,
        1.0,
        params,
        scales,
    )
    closure = PhysicalTransportClosure(
        stream_circularization_radius=80.0 * potential.r_g,
        wind_angular_momentum_factor=1.25,
        wind_launch_energy_multiplier=2.0,
        external_torque_prime=3.0e40,
        external_power_prime=4.0e48,
    )
    source = conservative_source_terms(
        state,
        stream_prime=3.0e20,
        wind_prime=1.0e20,
        closure=closure,
        params=params,
        radiative_loss_prime=5.0e48,
    )

    assert source.mass_rhs == pytest.approx(-2.0e20)
    assert source.carried.l_wind == pytest.approx(1.25 * state.l)
    assert source.carried.wind_torque_work == pytest.approx(
        state.Omega * (source.carried.l_wind - state.l)
    )
    assert source.angular_rhs == pytest.approx(
        1.0e20 * source.carried.l_wind
        - 3.0e20 * source.carried.l_stream
        + closure.external_torque_prime
    )
    assert source.energy_rhs == pytest.approx(
        5.0e48
        + 1.0e20 * source.carried.B_wind
        - 3.0e20 * source.carried.B_stream
        + closure.external_power_prime
    )


def test_power_primary_wind_energy_is_algebraically_equivalent(
    params: SimpleNamespace,
) -> None:
    potential = PaczynskiWiitaPotential(params.M2_g)
    scales = ConservativeScales(params.Mdot_g_s, potential.l_k(potential.r_isco))
    state = reconstruct_conservative_state(
        np.log(35.0 * potential.r_g),
        np.log(9.0e6),
        np.log(2.1e6),
        1.0,
        0.98,
        params,
        scales,
    )
    closure = PhysicalTransportClosure(
        stream_circularization_radius=80.0 * potential.r_g,
        wind_angular_momentum_factor=1.15,
        wind_launch_energy_multiplier=8.0,
    )
    carried = conservative_source_terms(
        state,
        stream_prime=2.0e20,
        wind_prime=3.0e19,
        closure=closure,
        params=params,
        radiative_loss_prime=4.0e47,
    )
    power = conservative_source_terms(
        state,
        stream_prime=2.0e20,
        wind_prime=3.0e19,
        closure=closure,
        params=params,
        radiative_loss_prime=4.0e47,
        wind_launch_power_prime=(
            3.0e19 * carried.carried.wind_launch_energy
        ),
    )

    assert power.wind_base_energy_prime + power.wind_launch_power_prime == pytest.approx(
        power.wind_prime * power.carried.B_wind, rel=2.0e-15
    )
    assert power.energy_rhs == pytest.approx(carried.energy_rhs, rel=2.0e-15)

    escape = wind_escape_diagnostics(state, closure, params)
    assert escape.terminal_margin == pytest.approx(escape.wind_bernoulli)
    assert escape.prescribed_launch_energy - escape.required_launch_energy == pytest.approx(
        escape.terminal_margin
    )
    assert escape.terminal_speed == pytest.approx(
        np.sqrt(2.0 * max(escape.wind_bernoulli, 0.0))
    )

    with pytest.raises(ValueError, match="wind_launch_power_prime"):
        conservative_source_terms(
            state,
            stream_prime=0.0,
            wind_prime=1.0,
            closure=closure,
            params=params,
            wind_launch_power_prime=-1.0,
        )


def test_terminal_bernoulli_launch_is_exact_and_rejects_nonpositive_energy(
    params: SimpleNamespace,
) -> None:
    potential = PaczynskiWiitaPotential(params.M2_g)
    scales = ConservativeScales(params.Mdot_g_s, potential.l_k(potential.r_isco))
    state = reconstruct_conservative_state(
        np.log(50.0 * potential.r_g),
        np.log(7.0e6),
        np.log(1.7e6),
        1.0,
        1.0,
        params,
        scales,
    )
    target = float(state.bernoulli + 2.0e18)
    closure = PhysicalTransportClosure(
        stream_circularization_radius=80.0 * potential.r_g,
        wind_launch_mode="terminal_bernoulli",
        wind_terminal_bernoulli=target,
        wind_mass_loading_cap_per_log_radius=0.5,
    )
    launch = wind_launch_energy(state, closure, params)
    source = conservative_source_terms(
        state,
        stream_prime=0.0,
        wind_prime=1.0e18,
        closure=closure,
        params=params,
        wind_launch_power_prime=1.0e18 * launch,
    )

    assert launch == pytest.approx(2.0e18)
    assert source.carried.B_wind == pytest.approx(target)
    assert wind_escape_diagnostics(
        state, closure, params, target_terminal_bernoulli=target
    ).terminal_margin == pytest.approx(0.0, abs=1.0e-10 * abs(target))

    invalid = PhysicalTransportClosure(
        stream_circularization_radius=80.0 * potential.r_g,
        wind_launch_mode="terminal_bernoulli",
        wind_terminal_bernoulli=state.bernoulli,
    )
    with pytest.raises(ValueError, match="positive launch energy"):
        wind_launch_energy(state, invalid, params)

    with pytest.raises(ValueError, match="wind_launch_mode"):
        PhysicalTransportClosure(wind_launch_mode="invalid")
    with pytest.raises(ValueError, match="wind_mass_loading_cap"):
        PhysicalTransportClosure(wind_mass_loading_cap_per_log_radius=0.0)


def test_manufactured_simpson_transport_closes(params: SimpleNamespace) -> None:
    potential = PaczynskiWiitaPotential(params.M2_g)
    scales = ConservativeScales(params.Mdot_g_s, potential.l_k(potential.r_isco))
    closure = PhysicalTransportClosure(stream_circularization_radius=60.0 * potential.r_g)
    dx = 0.04
    left = reconstruct_conservative_state(
        np.log(30.0 * potential.r_g), np.log(1.0e7), np.log(2.0e6), 1.0, 1.0, params, scales
    )
    midpoint = reconstruct_conservative_state(
        left.logR + 0.5 * dx, np.log(1.0e7), np.log(2.0e6), 1.0, 1.0, params, scales
    )
    right_base = reconstruct_conservative_state(
        left.logR + dx, np.log(1.0e7), np.log(2.0e6), 1.0, 1.0, params, scales
    )
    zero_left = conservative_source_terms(
        left, stream_prime=0.0, wind_prime=0.0, closure=closure, params=params, radiative_loss_prime=0.0
    )
    zero_mid = conservative_source_terms(
        midpoint, stream_prime=0.0, wind_prime=0.0, closure=closure, params=params, radiative_loss_prime=0.0
    )
    zero_right = conservative_source_terms(
        right_base, stream_prime=0.0, wind_prime=0.0, closure=closure, params=params, radiative_loss_prime=0.0
    )

    rows = simpson_interval_residual(
        dx,
        left,
        midpoint,
        right_base,
        zero_left,
        zero_mid,
        zero_right,
        scales,
        energy_flux_left=1.25 * scales.energy_flux,
        energy_flux_right=1.25 * scales.energy_flux,
    )
    assert rows.mass == pytest.approx(0.0)
    assert rows.angular_momentum == pytest.approx(0.0)
    assert rows.energy == pytest.approx(0.0)


def test_exact_compact_stream_interval_moments_replace_source_quadrature(
    params: SimpleNamespace,
) -> None:
    potential = PaczynskiWiitaPotential(params.M2_g)
    source_params = SimpleNamespace(
        **params.__dict__,
        R_out=300.0 * potential.r_g,
        stream_source_fraction=0.30,
        stream_source_center_fraction=0.8,
        stream_source_log_width=0.08,
        stream_source_shape="compact_c2",
        stream_source_shape_blend=1.0,
    )
    closure = PhysicalTransportClosure(
        stream_circularization_radius=240.0 * potential.r_g,
        external_torque_prime=2.0e39,
        external_power_prime=3.0e47,
    )
    scales = ConservativeScales(params.Mdot_g_s, potential.l_k(potential.r_isco))
    center = np.log(0.8 * source_params.R_out)
    width = source_params.stream_source_log_width
    log_radii = (center - width, center, center + width)
    states = tuple(
        reconstruct_conservative_state(
            log_radius,
            np.log(8.0e6),
            np.log(1.8e6),
            1.0,
            1.0,
            source_params,
            scales,
        )
        for log_radius in log_radii
    )
    wind_primes = (1.0e19, 2.0e19, 4.0e19)
    radiative_primes = (2.0e47, 3.0e47, 5.0e47)
    sources = tuple(
        conservative_source_terms(
            state,
            stream_prime=stream_source_prime(state.logR, source_params),
            wind_prime=wind_prime,
            closure=closure,
            params=source_params,
            radiative_loss_prime=radiative_prime,
        )
        for state, wind_prime, radiative_prime in zip(
            states, wind_primes, radiative_primes
        )
    )
    dx = 2.0 * width
    exact_stream = stream_source_interval_integral(
        log_radii[0], log_radii[-1], source_params
    )
    transport = integrate_interval_transport(
        dx,
        *sources,
        exact_stream_mass=exact_stream,
    )

    assert exact_stream == pytest.approx(0.30 * params.Mdot_g_s, rel=1.0e-14)
    assert transport.stream_mass == pytest.approx(exact_stream)
    assert abs(transport.stream_mass_quadrature_error) > 1.0e-6 * exact_stream
    assert transport.stream_angular_momentum == pytest.approx(
        exact_stream * sources[1].carried.l_stream
    )
    assert transport.stream_energy == pytest.approx(
        exact_stream * sources[1].carried.B_stream
    )
    assert transport.mass_rhs == pytest.approx(transport.wind_mass - exact_stream)
    assert transport.angular_rhs == pytest.approx(
        transport.wind_angular_momentum
        - transport.stream_angular_momentum
        + transport.external_angular_momentum
    )
    assert transport.energy_rhs == pytest.approx(
        transport.radiative_energy
        + transport.wind_energy
        - transport.stream_energy
        + transport.external_energy
    )

    rows = simpson_interval_residual(
        dx,
        *states,
        *sources,
        scales,
        exact_stream_mass=exact_stream,
    )
    assert rows.mass == pytest.approx(
        (states[-1].mdot - states[0].mdot - transport.mass_rhs) / scales.mdot
    )
    assert rows.angular_momentum == pytest.approx(
        (states[-1].J - states[0].J - transport.angular_rhs) / scales.angular_flux
    )


def test_exact_stream_integrals_are_additive_across_source_cells(
    params: SimpleNamespace,
) -> None:
    potential = PaczynskiWiitaPotential(params.M2_g)
    source_params = SimpleNamespace(
        **params.__dict__,
        R_out=300.0 * potential.r_g,
        stream_source_fraction=0.17,
        stream_source_center_fraction=0.8,
        stream_source_log_width=0.08,
        stream_source_shape="compact_c2",
        stream_source_shape_blend=1.0,
    )
    center = np.log(0.8 * source_params.R_out)
    edges = np.linspace(center - 0.08, center + 0.08, 17)
    cell_integrals = [
        stream_source_interval_integral(left, right, source_params)
        for left, right in zip(edges[:-1], edges[1:])
    ]
    assert sum(cell_integrals) == pytest.approx(0.17 * params.Mdot_g_s, rel=2.0e-14)
    assert min(cell_integrals) >= 0.0


def test_sampled_transport_is_quadrature_order_independent_for_constant_terms(
    params: SimpleNamespace,
) -> None:
    potential = PaczynskiWiitaPotential(params.M2_g)
    scales = ConservativeScales(params.Mdot_g_s, potential.l_k(potential.r_isco))
    state = reconstruct_conservative_state(
        np.log(40.0 * potential.r_g),
        np.log(8.0e6),
        np.log(1.8e6),
        1.0,
        1.0,
        params,
        scales,
    )
    closure = PhysicalTransportClosure(
        stream_circularization_radius=80.0 * potential.r_g,
        external_torque_prime=2.0e39,
        external_power_prime=3.0e47,
    )
    source = conservative_source_terms(
        state,
        stream_prime=4.0e20,
        wind_prime=2.0e20,
        closure=closure,
        params=params,
        radiative_loss_prime=5.0e47,
    )
    _nodes8, weights8 = np.polynomial.legendre.leggauss(8)
    _nodes16, weights16 = np.polynomial.legendre.leggauss(16)
    result8 = integrate_sampled_interval_transport(
        0.07, [source] * 8, 0.5 * weights8
    )
    result16 = integrate_sampled_interval_transport(
        0.07, [source] * 16, 0.5 * weights16
    )
    for field in result8.__dataclass_fields__:
        assert getattr(result8, field) == pytest.approx(getattr(result16, field))


def test_vertical_work_correction_closes_legacy_energy_identity(params: SimpleNamespace) -> None:
    potential = PaczynskiWiitaPotential(params.M2_g)
    logR = float(np.log(30.0 * potential.r_g))
    y = np.log(np.asarray([1.0e6, 2.0e6]))
    g = np.asarray([0.17, -0.44])
    lambda0 = float(potential.l_k(potential.r_isco) / (potential.r_g * 2.99792458e10))

    audit = legacy_energy_identity_audit(logR, y, g, lambda0, params)

    assert abs(audit.normalized_corrected_defect) < 1.0e-12
    assert audit.raw_identity_defect == pytest.approx(-audit.vertical_work_derivative)
    assert abs(audit.normalized_raw_defect) > 1.0e-8


def test_enthalpy_work_closes_actual_flux_derivative(params: SimpleNamespace) -> None:
    potential = PaczynskiWiitaPotential(params.M2_g)
    logR = float(np.log(30.0 * potential.r_g))
    y = np.log(np.asarray([1.0e6, 2.0e6]))
    lambda0 = float(
        potential.l_k(potential.r_isco) / (potential.r_g * 2.99792458e10)
    )
    g = local_gradient(logR, y, lambda0, params)

    audit = enthalpy_energy_identity_audit(logR, y, g, lambda0, params)

    assert abs(audit.normalized_corrected_defect) < 1.0e-12
    assert abs(audit.normalized_raw_defect) > 1.0e-8


def test_physical_stream_closure_requires_capture_data(params: SimpleNamespace) -> None:
    potential = PaczynskiWiitaPotential(params.M2_g)
    scales = ConservativeScales(params.Mdot_g_s, potential.l_k(potential.r_isco))
    state = reconstruct_conservative_state(
        np.log(30.0 * potential.r_g), np.log(1.0e7), np.log(2.0e6), 1.0, 1.0, params, scales
    )
    with pytest.raises(ValueError, match="physical stream closure"):
        conservative_source_terms(
            state,
            stream_prime=0.0,
            wind_prime=0.0,
            closure=PhysicalTransportClosure(),
            params=params,
        )
