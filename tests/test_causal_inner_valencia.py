from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    SchwarzschildKerrSchildGeometry,
    audit_ideal_gas_valencia_eigensystem,
    schwarzschild_kerr_schild_geometry,
    valencia_column_state,
    valencia_flux_primary_count,
    valencia_radial_characteristic_speeds_over_c,
)
from imri_qpe.parameters import FiducialParams


def _flat_geometry(radius: float = 10.0) -> SchwarzschildKerrSchildGeometry:
    return SchwarzschildKerrSchildGeometry(
        radius=radius,
        gravitational_radius=0.0,
        lapse=1.0,
        radial_shift_over_c=0.0,
        gamma_rr=1.0,
        gamma_phiphi=radius**2,
        inverse_gamma_rr=1.0,
        sqrt_spatial_metric=radius**2,
    )


def test_kerr_schild_light_cones_cross_the_horizon_smoothly() -> None:
    mass = FiducialParams().M2_g
    reference = schwarzschild_kerr_schild_geometry(1.0, mass)
    gravitational_radius = reference.gravitational_radius
    outside = schwarzschild_kerr_schild_geometry(
        3.0 * gravitational_radius, mass
    )
    horizon = schwarzschild_kerr_schild_geometry(
        2.0 * gravitational_radius, mass
    )
    inside = schwarzschild_kerr_schild_geometry(
        1.5 * gravitational_radius, mass
    )

    assert outside.outgoing_light_speed_over_c > 0.0
    assert horizon.outgoing_light_speed_over_c == pytest.approx(0.0)
    assert inside.outgoing_light_speed_over_c < 0.0
    assert inside.ingoing_light_speed_over_c == pytest.approx(-1.0)


def test_flat_radial_characteristics_recover_velocity_addition() -> None:
    geometry = _flat_geometry()
    velocity = -0.3
    sound = 0.2
    speeds = valencia_radial_characteristic_speeds_over_c(
        geometry,
        radial_velocity_over_c=velocity,
        azimuthal_velocity_over_c=0.0,
        sound_speed_over_c=sound,
    )

    expected_minus = (velocity - sound) / (1.0 - velocity * sound)
    expected_plus = (velocity + sound) / (1.0 + velocity * sound)
    assert speeds == pytest.approx(
        (expected_minus, velocity, velocity, expected_plus)
    )


def test_inside_horizon_all_fluid_characteristics_leave_inner_domain() -> None:
    mass = FiducialParams().M2_g
    base = schwarzschild_kerr_schild_geometry(1.0, mass)
    geometry = schwarzschild_kerr_schild_geometry(
        1.8 * base.gravitational_radius, mass
    )

    for beta_r in (-0.8, -0.2, 0.0, 0.2, 0.8):
        maximum_phi = np.sqrt(max(1.0 - beta_r**2, 0.0))
        beta_phi = 0.5 * maximum_phi
        speeds = valencia_radial_characteristic_speeds_over_c(
            geometry,
            radial_velocity_over_c=beta_r,
            azimuthal_velocity_over_c=beta_phi,
            sound_speed_over_c=1.0 / np.sqrt(3.0),
        )
        assert max(speeds) < 0.0


def test_valencia_column_state_recovers_cold_newtonian_limit() -> None:
    geometry = _flat_geometry(radius=20.0)
    sigma = 3.0
    beta_r = -1.0e-5
    beta_phi = 2.0e-5
    internal = 1.0e-10 * C**2
    pressure = 1.0e-10 * sigma * C**2
    state = valencia_column_state(
        geometry,
        surface_density=sigma,
        radial_velocity_over_c=beta_r,
        azimuthal_velocity_over_c=beta_phi,
        specific_internal_energy=internal,
        integrated_pressure=pressure,
    )

    assert state.conserved[0] == pytest.approx(sigma, rel=3.0e-10)
    assert state.conserved[1] == pytest.approx(
        sigma * beta_r, rel=1.0e-8
    )
    assert state.conserved[2] == pytest.approx(
        sigma * geometry.radius * beta_phi, rel=1.0e-8
    )
    expected_tau = sigma * (
        0.5 * (beta_r**2 + beta_phi**2) + internal / C**2
    )
    assert state.conserved[3] == pytest.approx(expected_tau, rel=2.0e-6)


def test_analytic_characteristics_match_conservative_flux_jacobian() -> None:
    mass = FiducialParams().M2_g
    base = schwarzschild_kerr_schild_geometry(1.0, mass)
    geometry = schwarzschild_kerr_schild_geometry(
        4.5 * base.gravitational_radius, mass
    )
    audit = audit_ideal_gas_valencia_eigensystem(
        geometry,
        surface_density=2.0,
        radial_velocity_over_c=-0.2,
        azimuthal_velocity_over_c=0.55,
        integrated_pressure=0.03 * 2.0 * C**2,
    )

    assert audit.maximum_eigenvalue_defect < 2.0e-8
    assert audit.stationary_flux_rank == 4


def test_stationary_flux_loses_one_rank_at_acoustic_critical_point() -> None:
    geometry = _flat_geometry()
    sigma = 1.0
    pressure = 0.02 * sigma * C**2

    def outgoing_speed(beta_r: float) -> float:
        audit = audit_ideal_gas_valencia_eigensystem(
            geometry,
            surface_density=sigma,
            radial_velocity_over_c=beta_r,
            azimuthal_velocity_over_c=0.25,
            integrated_pressure=pressure,
        )
        return max(audit.analytic_speeds_over_c)

    critical_velocity = brentq(outgoing_speed, -0.9, -1.0e-4)
    critical = audit_ideal_gas_valencia_eigensystem(
        geometry,
        surface_density=sigma,
        radial_velocity_over_c=critical_velocity,
        azimuthal_velocity_over_c=0.25,
        integrated_pressure=pressure,
    )

    assert max(abs(value) for value in critical.analytic_speeds_over_c) > 0.0
    assert critical.analytic_speeds_over_c[-1] == pytest.approx(
        0.0, abs=2.0e-12
    )
    assert critical.stationary_flux_rank == 3


def test_valencia_prototype_rejects_superluminal_rotation() -> None:
    with pytest.raises(ValueError, match="subluminal"):
        valencia_column_state(
            _flat_geometry(),
            surface_density=1.0,
            radial_velocity_over_c=0.8,
            azimuthal_velocity_over_c=0.8,
            specific_internal_energy=0.0,
            integrated_pressure=1.0,
        )


def test_flux_primary_dae_count_is_exactly_square() -> None:
    count = valencia_flux_primary_count(16)

    assert count.total_unknowns == 196
    assert count.total_rows == count.total_unknowns
    assert count.physical_inner_boundary_rows == 0
