from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from imri_qpe.constants import G
from imri_qpe.layer3_minidisk_1d.hill_roche_nozzle import (
    HillRocheNozzleProvider,
    HillRocheNozzleReservoir,
    OverflowBoundaryProvider,
    audit_hill_roche_nozzle_transverse_quadrature,
    fiducial_hill_roche_nozzle_geometry,
    hill_roche_midplane_force_derivative,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.layer3_minidisk_1d.transonic_potential import (
    PaczynskiWiitaPotential,
)


def _fiducial_reservoir():
    params = FiducialParams()
    potential = PaczynskiWiitaPotential(params.M2_g)
    geometry = fiducial_hill_roche_nozzle_geometry()
    radius = 335.0 * potential.r_g
    density = 1.0e-8
    gamma = 5.0 / 3.0
    sound_speed = 4.0e8
    pressure = density * sound_speed**2 / gamma
    reservoir = HillRocheNozzleReservoir(
        radius=radius,
        density=density,
        pressure=pressure,
        radial_velocity=1.0e7,
        specific_angular_momentum=float(potential.l_k(radius)),
    )
    return params, geometry, reservoir


def test_fiducial_pw_hill_saddle_is_regular_and_near_nominal_radius() -> None:
    params, geometry, _reservoir = _fiducial_reservoir()
    assert geometry.saddle_radius > geometry.nominal_hill_radius
    assert geometry.saddle_radius / geometry.nominal_hill_radius < 1.01
    force = float(
        hill_roche_midplane_force_derivative(
            geometry.saddle_radius,
            geometry.secondary_mass,
            geometry.pattern_omega,
        )
    )
    force_scale = G * params.M2_g / geometry.saddle_radius**2
    assert abs(force / force_scale) < 1.0e-12
    assert geometry.transverse_curvature_z > geometry.transverse_curvature_y
    assert geometry.channel_count == 2


def test_adiabatic_nozzle_closes_sonic_jacobi_and_pattern_power_ledgers() -> None:
    _params, geometry, reservoir = _fiducial_reservoir()
    provider = HillRocheNozzleProvider(geometry, gamma=5.0 / 3.0)
    assert isinstance(provider, OverflowBoundaryProvider)
    gate = provider.evaluate(reservoir)
    assert gate.choked
    assert gate.solution is not None
    assert gate.available_specific_energy > 0.0
    assert gate.required_enthalpy_multiplier < 1.0
    solution = gate.solution
    assert solution.choked
    assert solution.saddle_flux.mass > 0.0
    assert solution.sonic_sound_speed > 0.0
    assert solution.sonic_density > 0.0
    assert solution.sonic_pressure > 0.0
    assert abs(solution.sonic_residual) < 1.0e-12
    assert abs(solution.jacobi_residual) < 1.0e-13
    assert abs(solution.energy_pairing_residual) < 1.0e-13
    entropy_edge = reservoir.pressure / reservoir.density**solution.gamma
    entropy_sonic = (
        solution.sonic_pressure / solution.sonic_density**solution.gamma
    )
    assert entropy_sonic == pytest.approx(entropy_edge, rel=2.0e-15)
    assert solution.saddle_flux.rotating_energy == pytest.approx(
        solution.saddle_flux.mass * solution.rotating_bernoulli,
        rel=2.0e-15,
    )
    assert solution.saddle_flux.total_energy == pytest.approx(
        solution.saddle_flux.rotating_energy
        + geometry.pattern_omega * solution.saddle_flux.angular_momentum,
        rel=2.0e-15,
    )
    assert solution.edge_total_energy_flux == pytest.approx(
        solution.saddle_flux.total_energy + solution.binary_power_gain,
        rel=2.0e-15,
    )


def test_nozzle_flux_scales_only_with_declared_channel_geometry() -> None:
    _params, geometry, reservoir = _fiducial_reservoir()
    quarter = HillRocheNozzleProvider(
        replace(geometry, channel_count=1, filling_factor=0.25),
        gamma=5.0 / 3.0,
    ).solve(reservoir)
    full = HillRocheNozzleProvider(
        replace(geometry, channel_count=2, filling_factor=1.0),
        gamma=5.0 / 3.0,
    ).solve(reservoir)
    assert full.saddle_flux.mass / quarter.saddle_flux.mass == pytest.approx(8.0)
    assert (
        full.saddle_flux.radial_momentum
        / quarter.saddle_flux.radial_momentum
    ) == pytest.approx(8.0)
    assert (
        full.saddle_flux.angular_momentum
        / quarter.saddle_flux.angular_momentum
    ) == pytest.approx(8.0)
    assert (
        full.saddle_flux.total_energy / quarter.saddle_flux.total_energy
    ) == pytest.approx(8.0)
    assert full.sonic_sound_speed == quarter.sonic_sound_speed
    assert full.sonic_density == quarter.sonic_density


def test_transverse_quadrature_converges_to_analytic_polytropic_throat() -> None:
    _params, geometry, reservoir = _fiducial_reservoir()
    solution = HillRocheNozzleProvider(
        geometry, gamma=5.0 / 3.0
    ).solve(reservoir)
    audits = [
        audit_hill_roche_nozzle_transverse_quadrature(solution, zones)
        for zones in (16, 32, 64, 128, 256)
    ]
    mass_errors = np.asarray([audit.mass_relative_error for audit in audits])
    pressure_errors = np.asarray(
        [audit.pressure_relative_error for audit in audits]
    )
    assert np.all(mass_errors[1:] < mass_errors[:-1])
    assert np.all(pressure_errors[1:] < pressure_errors[:-1])
    assert mass_errors[-1] < 2.0e-5
    assert pressure_errors[-1] < 2.0e-5


def test_nozzle_rejects_reservoir_that_cannot_reach_saddle() -> None:
    _params, geometry, reservoir = _fiducial_reservoir()
    cold = replace(
        reservoir,
        pressure=1.0e-4 * reservoir.pressure,
        radial_velocity=0.0,
        specific_angular_momentum=geometry.pattern_omega * reservoir.radius**2,
    )
    provider = HillRocheNozzleProvider(geometry, gamma=5.0 / 3.0)
    gate = provider.evaluate(cold)
    assert not gate.choked
    assert gate.solution is None
    assert gate.available_specific_energy < 0.0
    assert gate.required_enthalpy_multiplier > 1.0
    with pytest.raises(ValueError, match="does not reach the Hill saddle"):
        provider.solve(cold)


def test_nozzle_validation_rejects_hidden_geometry_freedom() -> None:
    _params, geometry, reservoir = _fiducial_reservoir()
    with pytest.raises(ValueError, match="filling_factor"):
        HillRocheNozzleProvider(
            replace(geometry, filling_factor=0.0), gamma=5.0 / 3.0
        )
    with pytest.raises(ValueError, match="channel_count"):
        HillRocheNozzleProvider(
            replace(geometry, channel_count=3), gamma=5.0 / 3.0
        )
    with pytest.raises(ValueError, match="gamma"):
        HillRocheNozzleProvider(geometry, gamma=1.0)
    with pytest.raises(ValueError, match="inside the Hill saddle"):
        HillRocheNozzleProvider(geometry, gamma=5.0 / 3.0).solve(
            replace(reservoir, radius=geometry.saddle_radius)
        )
