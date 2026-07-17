from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    SchwarzschildCurvatureVerticalFrequency,
    QuasiHydrostaticGasRadiationColumnEOS,
    ValenciaPerfectFluidPrimitive,
    audit_causal_five_field_boundaries,
    audit_causal_five_field_principal,
    calibrate_causal_alpha_shear,
    causal_five_field_dae_count,
    causal_comoving_energy_source,
    causal_rest_frame_shear_rate,
    causal_temporal_vertical_work_storage,
    kerr_schild_column_four_velocity,
    kerr_schild_column_geometry,
)
from imri_qpe.parameters import FiducialParams


def _scales() -> tuple[float, float]:
    mass = FiducialParams().M2_g
    return mass, G * mass / C**2


def _column(
    radius: float,
    gravitational_radius: float,
    *,
    surface_density: float,
    temperature: float,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
):
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    eos = SchwarzschildCurvatureVerticalFrequency(
        gravitational_radius
    ).eos(radius)
    thermodynamics = eos.from_surface_density_temperature(
        surface_density,
        temperature,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=surface_density,
        radial_velocity_over_c=radial_velocity_over_c,
        azimuthal_velocity_over_c=azimuthal_velocity_over_c,
        specific_internal_energy=(
            thermodynamics.specific_internal_energy
        ),
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    closure = calibrate_causal_alpha_shear(
        primitive,
        alpha=0.1,
        reference_positive_shear_rate=(
            1.5 * C * np.sqrt(gravitational_radius / radius**3)
        ),
        viscous_signal_speed_over_c=(
            np.sqrt(0.1) * thermodynamics.sound_speed / C
        ),
    )
    return geometry, eos, primitive, closure


@pytest.mark.parametrize("n_cells", [1, 16, 64, 96])
def test_five_field_flux_primary_count_is_exactly_square(
    n_cells: int,
) -> None:
    count = causal_five_field_dae_count(n_cells)

    assert count.conserved_unknowns == 5 * n_cells
    assert count.primitive_unknowns == 5 * n_cells
    assert count.face_flux_unknowns == 5 * (n_cells + 1)
    assert count.total_unknowns == 15 * n_cells + 5
    assert count.total_rows == count.total_unknowns
    assert count.nonconservative_shear_rows == n_cells
    assert count.physical_inner_boundary_conditions == 0
    assert count.physical_outer_boundary_conditions == 2
    assert count.square


def test_covariant_shear_rate_recovers_newtonian_circular_limit() -> None:
    mass, gravitational_radius = _scales()
    radius = 1.0e5 * gravitational_radius
    relative_step = 2.0e-5

    def lower_velocity(local_radius: float) -> np.ndarray:
        geometry, _eos, primitive, _closure = _column(
            local_radius,
            gravitational_radius,
            surface_density=1.0e3,
            temperature=1.0e4,
            radial_velocity_over_c=(
                2.0 * gravitational_radius / local_radius
            ),
            azimuthal_velocity_over_c=(
                np.sqrt(gravitational_radius / local_radius)
                / kerr_schild_column_geometry(
                    local_radius,
                    gravitational_radius,
                ).base.lapse
            ),
        )
        return (
            geometry.spacetime_metric
            @ kerr_schild_column_four_velocity(
                geometry,
                primitive,
            )
        )

    left = radius * (1.0 - relative_step)
    right = radius * (1.0 + relative_step)
    derivative = (
        lower_velocity(right) - lower_velocity(left)
    ) / (right - left)
    geometry, _eos, primitive, _closure = _column(
        radius,
        gravitational_radius,
        surface_density=1.0e3,
        temperature=1.0e4,
        radial_velocity_over_c=2.0 * gravitational_radius / radius,
        azimuthal_velocity_over_c=(
            np.sqrt(gravitational_radius / radius)
            / kerr_schild_column_geometry(
                radius,
                gravitational_radius,
            ).base.lapse
        ),
    )
    shear = causal_rest_frame_shear_rate(
        geometry,
        primitive,
        radial_lower_four_velocity_derivative=derivative,
    )
    expected = 1.5 * np.sqrt(G * mass / radius**3)

    assert shear > 0.0
    assert shear / expected == pytest.approx(1.0, rel=8.0e-5)


def test_responsive_five_field_principal_has_real_causal_modes() -> None:
    _mass, gravitational_radius = _scales()
    radius = 20.0 * gravitational_radius
    geometry, eos, _primitive, closure = _column(
        radius,
        gravitational_radius,
        surface_density=1.0e7,
        temperature=1.0e7,
        radial_velocity_over_c=-0.01,
        azimuthal_velocity_over_c=0.20,
    )
    audit = audit_causal_five_field_principal(
        geometry,
        eos,
        closure,
        surface_density=1.0e7,
        radial_velocity_over_c=-0.01,
        azimuthal_velocity_over_c=0.20,
        temperature=1.0e7,
    )

    assert audit.causal_and_hyperbolic
    assert audit.maximum_local_rest_eigenvalue_defect < 1.0e-13
    assert audit.maximum_imaginary_eigenvalue < 1.0e-14
    assert audit.incoming_mode_response_rank == 2
    assert audit.incoming_mode_response_smallest_singular_value > 0.1


def test_inner_excision_and_outer_roche_shear_count_pass() -> None:
    _mass, gravitational_radius = _scales()
    inner_radius = 1.8 * gravitational_radius
    inner_geometry, inner_eos, _primitive, inner_closure = _column(
        inner_radius,
        gravitational_radius,
        surface_density=1.0e5,
        temperature=3.0e7,
        radial_velocity_over_c=-0.40,
        azimuthal_velocity_over_c=0.60,
    )
    inner = audit_causal_five_field_principal(
        inner_geometry,
        inner_eos,
        inner_closure,
        surface_density=1.0e5,
        radial_velocity_over_c=-0.40,
        azimuthal_velocity_over_c=0.60,
        temperature=3.0e7,
    )

    outer_radius = 335.0 * gravitational_radius
    outer_geometry = kerr_schild_column_geometry(
        outer_radius,
        gravitational_radius,
    )
    outer_beta_r = 2.0 * gravitational_radius / outer_radius
    outer_beta_phi = (
        np.sqrt(gravitational_radius / outer_radius)
        / outer_geometry.base.lapse
    )
    _, outer_eos, _primitive, outer_closure = _column(
        outer_radius,
        gravitational_radius,
        surface_density=1.0e4,
        temperature=8.0e5,
        radial_velocity_over_c=outer_beta_r,
        azimuthal_velocity_over_c=outer_beta_phi,
    )
    outer = audit_causal_five_field_principal(
        outer_geometry,
        outer_eos,
        outer_closure,
        surface_density=1.0e4,
        radial_velocity_over_c=outer_beta_r,
        azimuthal_velocity_over_c=outer_beta_phi,
        temperature=8.0e5,
    )
    boundary = audit_causal_five_field_boundaries(inner, outer)

    assert inner.incoming_inner_characteristics == 0
    assert outer.incoming_outer_characteristics == 2
    assert outer.stationary_coordinate_rank == 4
    assert boundary.outer_face_jacobian_rank == 5
    assert boundary.outer_physical_boundary_conditions == 2
    assert boundary.passed


def test_temporal_vertical_work_is_exact_killing_storage_increment() -> None:
    _mass, gravitational_radius = _scales()
    radius = 20.0 * gravitational_radius
    geometry, eos, primitive, _closure = _column(
        radius,
        gravitational_radius,
        surface_density=1.0e6,
        temperature=2.0e7,
        radial_velocity_over_c=-0.08,
        azimuthal_velocity_over_c=0.35,
    )
    old = eos.from_surface_density_temperature(1.0e6, 1.9e7)
    new = eos.from_surface_density_temperature(1.0e6, 2.1e7)
    storage = causal_temporal_vertical_work_storage(
        geometry,
        primitive,
        old,
        new,
    )
    four_velocity = kerr_schild_column_four_velocity(
        geometry,
        primitive,
    )
    coordinate_time = 3.25
    proper_rate = (
        -storage.work_per_area
        * four_velocity[0]
        / coordinate_time
    )
    source = causal_comoving_energy_source(
        geometry,
        primitive,
        comoving_energy_rate=proper_rate,
    )
    integrated_source = (
        source.killing_source_per_ct * C * coordinate_time
    )

    assert storage.killing_storage_increment == pytest.approx(
        -integrated_source,
        rel=3.0e-15,
        abs=1.0e-28,
    )


def test_temporal_vertical_work_reduces_to_pi_dlnh_at_rest() -> None:
    radius = 1.0e8
    geometry = kerr_schild_column_geometry(radius, 0.0)
    eos = QuasiHydrostaticGasRadiationColumnEOS(
        proper_vertical_frequency=1.0e-2
    )
    old = eos.from_surface_density_temperature(1.0e5, 1.0e6)
    new = eos.from_surface_density_temperature(1.0e5, 1.1e6)
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=1.0e5,
        radial_velocity_over_c=0.0,
        azimuthal_velocity_over_c=0.0,
        specific_internal_energy=new.specific_internal_energy,
        integrated_pressure=new.integrated_pressure,
    )
    storage = causal_temporal_vertical_work_storage(
        geometry,
        primitive,
        old,
        new,
    )

    assert storage.killing_storage_increment[:3] == pytest.approx(
        np.zeros(3),
        abs=1.0e-30,
    )
    assert storage.killing_storage_increment[3] == pytest.approx(
        storage.work_per_area / C**2,
        rel=2.0e-15,
    )
