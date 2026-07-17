from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import brentq

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    KERR_SCHILD_HILL_ENERGY_ZERO,
    GasRadiationHillRocheNozzleProvider,
    SchwarzschildCurvatureVerticalFrequency,
    ValenciaPerfectFluidPrimitive,
    apply_kerr_schild_hill_roche_boundary,
    audit_kerr_schild_migration_rank,
    exact_kerr_schild_compact_stream_sources,
    fiducial_hill_roche_nozzle_geometry,
    kerr_schild_column_four_velocity,
    kerr_schild_column_geometry,
    kerr_schild_hill_roche_reservoir,
    kerr_schild_specific_injection_moments,
    kerr_schild_stream_injection,
    make_kerr_schild_column_grid,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


def _scales() -> tuple[float, float]:
    mass = FiducialParams().M2_g
    return mass, G * mass / C**2


def _circular_column(
    radius: float,
    gravitational_radius: float,
    *,
    surface_density: float,
    temperature: float,
):
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    frequency = SchwarzschildCurvatureVerticalFrequency(
        gravitational_radius
    )
    eos = frequency.eos(radius)
    thermodynamics = eos.from_surface_density_temperature(
        surface_density,
        temperature,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=surface_density,
        radial_velocity_over_c=(
            2.0 * gravitational_radius / radius
        ),
        azimuthal_velocity_over_c=float(
            np.sqrt(gravitational_radius / radius)
            / geometry.base.lapse
        ),
        specific_internal_energy=(
            thermodynamics.specific_internal_energy
        ),
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    return geometry, eos, thermodynamics, primitive


def _roche_provider() -> GasRadiationHillRocheNozzleProvider:
    geometry = replace(
        fiducial_hill_roche_nozzle_geometry(),
        energy_zero=KERR_SCHILD_HILL_ENERGY_ZERO,
    )
    return GasRadiationHillRocheNozzleProvider(
        geometry,
        transverse_quadrature_zones=24,
    )


def _roche_boundary(
    temperature: float,
    *,
    surface_density: float = 1.0e4,
):
    _mass, gravitational_radius = _scales()
    radius = 335.0 * gravitational_radius
    geometry, eos, _thermodynamics, primitive = _circular_column(
        radius,
        gravitational_radius,
        surface_density=surface_density,
        temperature=temperature,
    )
    return apply_kerr_schild_hill_roche_boundary(
        geometry,
        eos,
        primitive,
        temperature=temperature,
        provider=_roche_provider(),
    )


def test_vertical_frequency_provider_is_regular_and_has_declared_slope() -> None:
    _mass, gravitational_radius = _scales()
    provider = SchwarzschildCurvatureVerticalFrequency(
        gravitational_radius
    )
    for radius_over_rg in (1.5, 2.0, 20.0, 335.0):
        radius = radius_over_rg * gravitational_radius
        assert np.isfinite(provider.frequency(radius))
        assert provider.frequency(radius) > 0.0
        assert provider.logarithmic_radial_derivative(radius) == -1.5
        step = 1.0e-5
        numerical = (
            np.log(provider.frequency(radius * np.exp(step)))
            - np.log(provider.frequency(radius * np.exp(-step)))
        ) / (2.0 * step)
        assert numerical == pytest.approx(-1.5, rel=2.0e-10)


def test_stream_moments_are_the_covariant_four_state_moments() -> None:
    _mass, gravitational_radius = _scales()
    radius = 240.0 * gravitational_radius
    geometry, _eos, thermodynamics, primitive = _circular_column(
        radius,
        gravitational_radius,
        surface_density=1.0e5,
        temperature=1.0e6,
    )
    moments = kerr_schild_specific_injection_moments(
        geometry,
        primitive,
    )
    four_velocity = kerr_schild_column_four_velocity(
        geometry,
        primitive,
    )
    lower_velocity = geometry.spacetime_metric @ four_velocity
    enthalpy_over_c2 = (
        1.0 + thermodynamics.specific_enthalpy / C**2
    )

    assert moments.radial_momentum_over_c == pytest.approx(
        enthalpy_over_c2 * lower_velocity[1],
        rel=2.0e-15,
    )
    assert moments.angular_momentum_over_c == pytest.approx(
        enthalpy_over_c2 * lower_velocity[2],
        rel=2.0e-15,
    )
    assert moments.killing_energy_over_c2 == pytest.approx(
        -enthalpy_over_c2 * lower_velocity[0],
        rel=2.0e-15,
    )
    assert moments.specific_killing_energy > 0.0


def test_stream_moments_recover_the_newtonian_weak_field_orbit() -> None:
    _mass, gravitational_radius = _scales()
    radius = 1.0e4 * gravitational_radius
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=1.0,
        radial_velocity_over_c=2.0 * gravitational_radius / radius,
        azimuthal_velocity_over_c=float(
            np.sqrt(gravitational_radius / radius)
            / geometry.base.lapse
        ),
        specific_internal_energy=0.0,
        integrated_pressure=0.0,
    )
    moments = kerr_schild_specific_injection_moments(
        geometry,
        primitive,
    )
    newtonian_angular_momentum = (
        C * np.sqrt(gravitational_radius * radius)
    )
    newtonian_binding_energy = (
        -0.5 * C**2 * gravitational_radius / radius
    )
    assert abs(moments.transport_radial_velocity / C) < 1.0e-15
    assert (
        moments.kinematic_specific_angular_momentum
        / newtonian_angular_momentum
        - 1.0
    ) == pytest.approx(1.5003375844e-4, rel=2.0e-9)
    assert (
        (moments.specific_killing_energy - C**2)
        / newtonian_binding_energy
        - 1.0
    ) == pytest.approx(-7.5033756788e-5, rel=2.0e-8)


@pytest.mark.parametrize("shape", ["compact_c2", "compact_c4"])
def test_compact_stream_sources_are_exact_conservative_cell_moments(
    shape: str,
) -> None:
    mass, gravitational_radius = _scales()
    radius = 240.0 * gravitational_radius
    geometry, _eos, _thermodynamics, primitive = _circular_column(
        radius,
        gravitational_radius,
        surface_density=1.0e5,
        temperature=1.0e6,
    )
    injection = kerr_schild_stream_injection(
        geometry,
        primitive,
        rest_mass_rate=5.0 * eddington_mdot(mass),
    )
    active_counts = []
    for n_cells in (32, 64, 128):
        grid = make_kerr_schild_column_grid(
            1.8 * gravitational_radius,
            335.0 * gravitational_radius,
            n_cells,
            gravitational_radius,
        )
        source = exact_kerr_schild_compact_stream_sources(
            grid,
            injection,
            center=radius,
            log_width=0.08,
            shape=shape,
        )
        active_counts.append(int(np.count_nonzero(source.rest_mass)))
        assert np.sum(source.rest_mass) == pytest.approx(
            injection.rest_mass_rate,
            rel=3.0e-16,
        )
        assert np.sum(source.radial_momentum_over_c) == pytest.approx(
            injection.rest_mass_rate
            * injection.moments.radial_momentum_over_c,
            rel=3.0e-16,
        )
        assert np.sum(source.angular_momentum_over_c) == pytest.approx(
            injection.rest_mass_rate
            * injection.moments.angular_momentum_over_c,
            rel=3.0e-16,
        )
        assert np.sum(source.killing_energy_over_c2) == pytest.approx(
            injection.rest_mass_rate
            * injection.moments.killing_energy_over_c2,
            rel=3.0e-16,
        )
        assert source.weighted_killing_source_per_ct == pytest.approx(
            source.matrix / C,
            rel=2.0e-15,
        )
    assert active_counts[-1] > active_counts[0]


def test_stream_source_rejects_support_outside_the_kerr_schild_grid() -> None:
    mass, gravitational_radius = _scales()
    radius = 330.0 * gravitational_radius
    geometry, _eos, _thermodynamics, primitive = _circular_column(
        radius,
        gravitational_radius,
        surface_density=1.0e5,
        temperature=1.0e6,
    )
    injection = kerr_schild_stream_injection(
        geometry,
        primitive,
        rest_mass_rate=eddington_mdot(mass),
    )
    grid = make_kerr_schild_column_grid(
        1.8 * gravitational_radius,
        335.0 * gravitational_radius,
        64,
        gravitational_radius,
    )
    with pytest.raises(ValueError, match="support must lie inside"):
        exact_kerr_schild_compact_stream_sources(
            grid,
            injection,
            center=radius,
            log_width=0.08,
        )


def test_roche_boundary_has_closed_and_choked_characteristic_contracts() -> None:
    closed = _roche_boundary(8.0e5)
    assert not closed.gate.choked
    assert closed.rest_mass_rate == 0.0
    assert closed.angular_momentum_rate == 0.0
    assert closed.killing_energy_rate == 0.0
    assert closed.radial_momentum_rate == pytest.approx(
        closed.pressure_traction
    )
    assert closed.incoming_outer_characteristics == 1
    assert closed.no_inward_mass
    assert closed.zero_outer_stress

    choked = _roche_boundary(1.0e6)
    assert choked.gate.choked
    assert choked.rest_mass_rate > 0.0
    assert choked.radial_momentum_rate > choked.pressure_traction
    assert choked.angular_momentum_rate == pytest.approx(
        choked.rest_mass_rate
        * choked.edge_state.moments.specific_angular_momentum,
        rel=2.0e-13,
    )
    assert choked.killing_energy_rate == pytest.approx(
        choked.rest_mass_rate
        * choked.edge_state.moments.specific_killing_energy,
        rel=2.0e-13,
    )
    assert max(
        abs(choked.angular_momentum_relative_defect),
        abs(choked.killing_energy_relative_defect),
        abs(choked.binary_pattern_power_relative_defect),
    ) < 2.0e-12
    assert choked.weighted_killing_flux_over_c == pytest.approx(
        np.asarray(
            [
                choked.rest_mass_rate / C,
                choked.radial_momentum_rate / C**2,
                choked.angular_momentum_rate / C**2,
                choked.killing_energy_rate / C**3,
            ]
        ),
        rel=2.0e-15,
    )


def test_roche_opening_is_continuous_in_the_edge_temperature() -> None:
    _mass, gravitational_radius = _scales()
    radius = 335.0 * gravitational_radius
    provider = _roche_provider()

    def available(temperature: float) -> float:
        geometry, eos, _thermodynamics, primitive = _circular_column(
            radius,
            gravitational_radius,
            surface_density=1.0e4,
            temperature=temperature,
        )
        _edge, reservoir = kerr_schild_hill_roche_reservoir(
            geometry,
            eos,
            primitive,
            temperature=temperature,
        )
        return provider.available_specific_energy(reservoir)

    threshold = brentq(
        available,
        8.0e5,
        1.0e6,
        xtol=1.0e-7,
        rtol=1.0e-13,
    )
    below = _roche_boundary(threshold * (1.0 - 1.0e-6))
    above = _roche_boundary(threshold * (1.0 + 1.0e-6))
    assert not below.gate.choked
    assert above.gate.choked
    assert below.gate.available_specific_energy < 0.0
    assert above.gate.available_specific_energy > 0.0
    assert above.rest_mass_rate > 0.0
    assert above.rest_mass_rate < _roche_boundary(1.0e6).rest_mass_rate


def test_roche_boundary_rejects_old_energy_zero_and_nonzero_stress() -> None:
    _mass, gravitational_radius = _scales()
    radius = 335.0 * gravitational_radius
    geometry, eos, _thermodynamics, primitive = _circular_column(
        radius,
        gravitational_radius,
        surface_density=1.0e4,
        temperature=8.0e5,
    )
    old_provider = GasRadiationHillRocheNozzleProvider(
        fiducial_hill_roche_nozzle_geometry()
    )
    with pytest.raises(ValueError, match="Killing-energy zero"):
        apply_kerr_schild_hill_roche_boundary(
            geometry,
            eos,
            primitive,
            temperature=8.0e5,
            provider=old_provider,
        )
    with pytest.raises(ValueError, match="zero outer shear stress"):
        apply_kerr_schild_hill_roche_boundary(
            geometry,
            eos,
            primitive,
            temperature=8.0e5,
            provider=_roche_provider(),
            outer_specific_stress=1.0,
        )


def test_migrated_source_and_roche_boundary_preserve_the_square_dae_count() -> None:
    boundary = _roche_boundary(1.0e6)
    for n_cells in (16, 64):
        audit = audit_kerr_schild_migration_rank(
            n_cells,
            boundary,
        )
        assert audit.total_unknowns == 12 * n_cells + 4
        assert audit.total_rows == audit.total_unknowns
        assert audit.source_unknowns == 0
        assert audit.source_rows == 0
        assert audit.boundary_face_rows == 4
        assert audit.boundary_face_jacobian_rank == 4
        assert audit.physical_outer_boundary_conditions == 1
        assert audit.square
