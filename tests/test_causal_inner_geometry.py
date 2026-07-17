from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    ValenciaPerfectFluidPrimitive,
    audit_kerr_schild_column_sources,
    audit_stationary_kerr_schild_finite_volume_profile,
    kerr_schild_column_geometry,
    kerr_schild_column_measure_antiderivative,
    make_kerr_schild_column_grid,
    valencia_conserved_from_killing,
)
from imri_qpe.parameters import FiducialParams


def _gravitational_radius() -> float:
    return G * FiducialParams().M2_g / C**2


def _generic_primitive(radius: float) -> ValenciaPerfectFluidPrimitive:
    del radius
    return ValenciaPerfectFluidPrimitive(
        surface_density=2.0,
        radial_velocity_over_c=-0.2,
        azimuthal_velocity_over_c=0.5,
        specific_internal_energy=0.03 * C**2,
        integrated_pressure=0.02 * 2.0 * C**2,
    )


def _circular_dust_primitive(
    radius: float,
    gravitational_radius: float,
) -> ValenciaPerfectFluidPrimitive:
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    metric_ratio = 2.0 * gravitational_radius / radius
    radial_velocity = metric_ratio
    azimuthal_velocity = (
        np.sqrt(gravitational_radius / radius) / geometry.base.lapse
    )
    return ValenciaPerfectFluidPrimitive(
        surface_density=1.0,
        radial_velocity_over_c=float(radial_velocity),
        azimuthal_velocity_over_c=float(azimuthal_velocity),
        specific_internal_energy=0.0,
        integrated_pressure=0.0,
    )


def _radial_dust_primitive(
    radius: float,
    gravitational_radius: float,
) -> ValenciaPerfectFluidPrimitive:
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    free_fall_speed = np.sqrt(2.0 * gravitational_radius / radius)
    coordinate_time_velocity = (
        1.0 + free_fall_speed + free_fall_speed**2
    ) / (1.0 + free_fall_speed)
    lorentz_factor = geometry.base.lapse * coordinate_time_velocity
    coordinate_velocity = (
        -free_fall_speed / lorentz_factor
        + geometry.base.radial_shift_over_c / geometry.base.lapse
    )
    radial_velocity = (
        np.sqrt(geometry.base.gamma_rr) * coordinate_velocity
    )
    return ValenciaPerfectFluidPrimitive(
        surface_density=float(
            np.sqrt(10.0 * gravitational_radius / radius)
        ),
        radial_velocity_over_c=float(radial_velocity),
        azimuthal_velocity_over_c=0.0,
        specific_internal_energy=0.0,
        integrated_pressure=0.0,
    )


def test_metric_inverse_and_radial_derivative_are_consistent() -> None:
    gravitational_radius = _gravitational_radius()
    radius = 4.5 * gravitational_radius
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    step = 1.0e-5 * radius
    upper = kerr_schild_column_geometry(
        radius + step,
        gravitational_radius,
    ).spacetime_metric
    lower = kerr_schild_column_geometry(
        radius - step,
        gravitational_radius,
    ).spacetime_metric
    numerical_derivative = (upper - lower) / (2.0 * step)

    assert geometry.spacetime_metric @ geometry.inverse_spacetime_metric == (
        pytest.approx(np.eye(3), abs=2.0e-15)
    )
    assert geometry.radial_spacetime_metric_derivative == pytest.approx(
        numerical_derivative,
        rel=2.0e-10,
        abs=2.0e-13 / gravitational_radius,
    )


def test_exact_column_measure_matches_quadrature_and_flat_limit() -> None:
    gravitational_radius = _gravitational_radius()
    left = 1.7 * gravitational_radius
    right = 30.0 * gravitational_radius
    exact = (
        kerr_schild_column_measure_antiderivative(
            right,
            gravitational_radius,
        )
        - kerr_schild_column_measure_antiderivative(
            left,
            gravitational_radius,
        )
    )
    numerical = quad(
        lambda radius: (
            2.0
            * np.pi
            * radius
            * np.sqrt(1.0 + 2.0 * gravitational_radius / radius)
        ),
        left,
        right,
        epsabs=0.0,
        epsrel=2.0e-13,
    )[0]
    flat = (
        kerr_schild_column_measure_antiderivative(right, 0.0)
        - kerr_schild_column_measure_antiderivative(left, 0.0)
    )

    assert exact == pytest.approx(numerical, rel=3.0e-15)
    assert flat == pytest.approx(
        np.pi * (right**2 - left**2),
        rel=2.0e-15,
    )


@pytest.mark.parametrize("radius_over_rg", (20.0, 4.5, 2.0, 1.8))
def test_covariant_and_three_plus_one_sources_agree(
    radius_over_rg: float,
) -> None:
    gravitational_radius = _gravitational_radius()
    geometry = kerr_schild_column_geometry(
        radius_over_rg * gravitational_radius,
        gravitational_radius,
    )
    audit = audit_kerr_schild_column_sources(
        geometry,
        _generic_primitive(geometry.radius),
    )
    source_scale = max(
        abs(audit.radial_momentum_source),
        abs(audit.tau_source),
        1.0 / gravitational_radius,
    )

    assert abs(audit.momentum_source_identity_defect) < (
        3.0e-15 * source_scale
    )
    assert abs(audit.tau_source_identity_defect) < (
        3.0e-15 * source_scale
    )
    assert abs(audit.killing_density_identity_defect) < 2.0e-15
    assert abs(audit.killing_flux_identity_defect) < 2.0e-15
    recovered = valencia_conserved_from_killing(
        geometry,
        audit.killing_conserved,
    )
    assert recovered == pytest.approx(
        audit.valencia_state.conserved,
        rel=2.0e-15,
        abs=2.0e-15,
    )


def test_flat_constant_pressure_balances_cylindrical_source() -> None:
    grid = make_kerr_schild_column_grid(2.0, 20.0, 24, 0.0)
    pressure = 0.04 * C**2

    def primitive(radius: float) -> ValenciaPerfectFluidPrimitive:
        del radius
        return ValenciaPerfectFluidPrimitive(
            surface_density=1.0,
            radial_velocity_over_c=0.0,
            azimuthal_velocity_over_c=0.0,
            specific_internal_energy=0.0,
            integrated_pressure=pressure,
        )

    audit = audit_stationary_kerr_schild_finite_volume_profile(
        grid,
        primitive,
        quadrature_order=2,
    )

    assert np.max(np.abs(audit.integrated_residuals)) < 5.0e-14
    assert np.max(np.abs(audit.telescoping_defect)) < 5.0e-14


@pytest.mark.parametrize("radius_over_rg", (6.1, 10.0, 20.0))
def test_schwarzschild_circular_dust_orbit_has_zero_radial_source(
    radius_over_rg: float,
) -> None:
    gravitational_radius = _gravitational_radius()
    radius = radius_over_rg * gravitational_radius
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    audit = audit_kerr_schild_column_sources(
        geometry,
        _circular_dust_primitive(radius, gravitational_radius),
    )

    assert abs(audit.radial_momentum_source) < (
        3.0e-14 / gravitational_radius
    )
    assert abs(audit.killing_flux_over_c[0]) < 2.0e-15
    assert abs(audit.killing_flux_over_c[3]) < 2.0e-15


def test_radial_free_fall_has_constant_mass_and_killing_energy_flux() -> None:
    gravitational_radius = _gravitational_radius()
    grid = make_kerr_schild_column_grid(
        1.5 * gravitational_radius,
        20.0 * gravitational_radius,
        96,
        gravitational_radius,
    )
    audit = audit_stationary_kerr_schild_finite_volume_profile(
        grid,
        lambda radius: _radial_dust_primitive(
            radius,
            gravitational_radius,
        ),
        quadrature_order=8,
    )
    fluxes = audit.weighted_face_fluxes_over_c
    mass_scale = abs(fluxes[0, 0])

    assert np.ptp(fluxes[:, 0]) < 3.0e-14 * mass_scale
    assert np.ptp(fluxes[:, 3]) < 6.0e-14 * mass_scale
    momentum_scale = np.max(np.abs(np.diff(fluxes[:, 1])))
    assert np.max(np.abs(audit.integrated_residuals[:, 1])) < (
        1.0e-11 * momentum_scale
    )
    assert np.max(np.abs(audit.telescoping_defect)) < (
        2.0e-14 * mass_scale
    )


def test_radial_free_fall_geometric_source_converges_spatially() -> None:
    gravitational_radius = _gravitational_radius()
    errors = []
    for n_cells in (16, 32, 64):
        grid = make_kerr_schild_column_grid(
            1.5 * gravitational_radius,
            20.0 * gravitational_radius,
            n_cells,
            gravitational_radius,
        )
        audit = audit_stationary_kerr_schild_finite_volume_profile(
            grid,
            lambda radius: _radial_dust_primitive(
                radius,
                gravitational_radius,
            ),
            quadrature_order=1,
        )
        momentum_scale = np.max(
            np.abs(np.diff(audit.weighted_face_fluxes_over_c[:, 1]))
        )
        errors.append(
            np.max(np.abs(audit.integrated_residuals[:, 1]))
            / momentum_scale
        )

    assert errors[0] / errors[1] > 3.3
    assert errors[1] / errors[2] > 3.3


def test_geometry_contract_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        kerr_schild_column_geometry(0.0, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        kerr_schild_column_geometry(1.0, -1.0)
    with pytest.raises(ValueError, match="positive integer"):
        make_kerr_schild_column_grid(1.0, 2.0, 0, 1.0)
    with pytest.raises(ValueError, match="length four"):
        valencia_conserved_from_killing(
            kerr_schild_column_geometry(2.0, 1.0),
            np.ones(3),
        )
