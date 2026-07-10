from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    ConservativeScales,
    PaczynskiWiitaPotential,
    PhysicalTransportClosure,
    conservative_source_terms,
    legacy_energy_identity_audit,
    reconstruct_conservative_state,
    simpson_interval_residual,
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
