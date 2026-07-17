from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    QuasiHydrostaticGasRadiationColumnEOS,
    ValenciaPerfectFluidPrimitive,
    audit_causal_stress_characteristics,
    audit_quasi_hydrostatic_characteristics,
    calibrate_causal_alpha_shear,
    causal_comoving_energy_source,
    causal_diffusion_cooling_rate,
    causal_stress_work_partition,
    causal_thermal_column_source,
    hydrostatic_vertical_work_identity_defect,
    kerr_schild_column_geometry,
    make_kerr_schild_column_grid,
    recover_valencia_gas_radiation_primitives,
    temporal_vertical_work_per_area,
    valencia_gas_radiation_column_state,
)
from imri_qpe.parameters import FiducialParams


def _scales() -> tuple[float, float]:
    mass = FiducialParams().M2_g
    return mass, G * mass / C**2


def _eos(radius_over_rg: float):
    mass, gravitational_radius = _scales()
    radius = radius_over_rg * gravitational_radius
    return QuasiHydrostaticGasRadiationColumnEOS(
        proper_vertical_frequency=np.sqrt(G * mass / radius**3)
    )


def test_dynamic_height_eos_closes_hydrostatic_and_thermal_identities() -> None:
    eos = _eos(20.0)
    state = eos.from_surface_density_temperature(1.0e7, 1.0e7)
    pressure = (
        state.integrated_pressure
        / (2.0 * state.proper_half_thickness)
    )

    assert state.integrated_pressure / state.surface_density == pytest.approx(
        eos.proper_vertical_frequency**2
        * state.proper_half_thickness**2,
        rel=3.0e-15,
    )
    assert state.integrated_pressure / state.surface_density == pytest.approx(
        pressure / state.density,
        rel=2.0e-15,
    )
    recovered = eos.from_surface_density_internal_energy(
        state.surface_density,
        state.specific_internal_energy,
    )
    assert recovered.temperature == pytest.approx(
        state.temperature,
        rel=5.0e-12,
    )
    assert recovered.proper_half_thickness == pytest.approx(
        state.proper_half_thickness,
        rel=5.0e-12,
    )
    assert hydrostatic_vertical_work_identity_defect(
        state,
        surface_density_derivative=2.3e-4 * state.surface_density,
        height_derivative=-1.7e-4 * state.proper_half_thickness,
    ) < 2.0e-15


def test_dynamic_height_derivatives_match_independent_finite_differences() -> None:
    eos = _eos(20.0)
    sigma = 1.0e7
    temperature = 1.0e7
    derivatives = eos.derivatives(sigma, temperature)
    step = 1.0e-5

    sigma_minus = eos.from_surface_density_temperature(
        sigma * np.exp(-step),
        temperature,
    )
    sigma_plus = eos.from_surface_density_temperature(
        sigma * np.exp(step),
        temperature,
    )
    temperature_minus = eos.from_surface_density_temperature(
        sigma,
        temperature * np.exp(-step),
    )
    temperature_plus = eos.from_surface_density_temperature(
        sigma,
        temperature * np.exp(step),
    )
    frequency_minus = QuasiHydrostaticGasRadiationColumnEOS(
        eos.proper_vertical_frequency * np.exp(-step)
    ).from_surface_density_temperature(sigma, temperature)
    frequency_plus = QuasiHydrostaticGasRadiationColumnEOS(
        eos.proper_vertical_frequency * np.exp(step)
    ).from_surface_density_temperature(sigma, temperature)

    assert derivatives.height_log_surface_density == pytest.approx(
        (
            np.log(sigma_plus.proper_half_thickness)
            - np.log(sigma_minus.proper_half_thickness)
        )
        / (2.0 * step),
        rel=2.0e-9,
    )
    assert derivatives.height_log_temperature == pytest.approx(
        (
            np.log(temperature_plus.proper_half_thickness)
            - np.log(temperature_minus.proper_half_thickness)
        )
        / (2.0 * step),
        rel=2.0e-9,
    )
    assert derivatives.height_log_vertical_frequency == pytest.approx(
        (
            np.log(frequency_plus.proper_half_thickness)
            - np.log(frequency_minus.proper_half_thickness)
        )
        / (2.0 * step),
        rel=2.0e-9,
    )
    assert derivatives.internal_energy_log_surface_density == pytest.approx(
        (
            sigma_plus.specific_internal_energy
            - sigma_minus.specific_internal_energy
        )
        / (2.0 * step),
        rel=2.0e-9,
    )
    assert derivatives.pressure_log_temperature == pytest.approx(
        (
            temperature_plus.integrated_pressure
            - temperature_minus.integrated_pressure
        )
        / (2.0 * step),
        rel=2.0e-9,
    )


@pytest.mark.parametrize(
    ("radius_over_rg", "surface_density", "temperature"),
    (
        (20.0, 1.0e5, 1.0e7),
        (4.5, 1.0e7, 3.0e7),
        (1.8, 1.0e9, 3.0e8),
    ),
)
def test_dynamic_height_characteristics_include_vertical_pressure_work(
    radius_over_rg: float,
    surface_density: float,
    temperature: float,
) -> None:
    audit = audit_quasi_hydrostatic_characteristics(
        _eos(radius_over_rg),
        surface_density=surface_density,
        temperature=temperature,
    )

    assert audit.maximum_eigenvalue_defect < 2.0e-15
    assert audit.maximum_imaginary_eigenvalue == 0.0
    assert audit.numerical_speeds_over_c[0] < 0.0
    assert audit.numerical_speeds_over_c[1] == pytest.approx(
        0.0,
        abs=2.0e-15,
    )
    assert audit.numerical_speeds_over_c[2] > 0.0


def test_dynamic_height_valencia_recovery_round_trip() -> None:
    _, gravitational_radius = _scales()
    geometry = kerr_schild_column_geometry(
        4.5 * gravitational_radius,
        gravitational_radius,
    )
    eos = _eos(4.5)
    state, thermodynamics = valencia_gas_radiation_column_state(
        geometry.base,
        eos,
        surface_density=1.0e7,
        radial_velocity_over_c=-0.20,
        azimuthal_velocity_over_c=0.55,
        temperature=3.0e7,
    )
    recovered = recover_valencia_gas_radiation_primitives(
        geometry.base,
        eos,
        state.conserved,
    )

    assert recovered.maximum_relative_conserved_defect < 2.0e-11
    assert recovered.primitive.temperature == pytest.approx(
        3.0e7,
        rel=2.0e-10,
    )
    assert (
        recovered.primitive.thermodynamics.proper_half_thickness
        == pytest.approx(
            thermodynamics.proper_half_thickness,
            rel=2.0e-10,
        )
    )


def test_dynamic_height_state_retains_causal_shear_spectrum() -> None:
    mass, gravitational_radius = _scales()
    radius_over_rg = 1.8
    radius = radius_over_rg * gravitational_radius
    geometry = kerr_schild_column_geometry(radius, gravitational_radius)
    eos = _eos(radius_over_rg)
    thermodynamics = eos.from_surface_density_temperature(
        1.0e9,
        3.0e8,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=1.0e9,
        radial_velocity_over_c=-0.40,
        azimuthal_velocity_over_c=0.60,
        specific_internal_energy=thermodynamics.specific_internal_energy,
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    closure = calibrate_causal_alpha_shear(
        primitive,
        alpha=0.1,
        reference_positive_shear_rate=(
            1.5 * np.sqrt(G * mass / radius**3)
        ),
        viscous_signal_speed_over_c=(
            np.sqrt(0.1) * thermodynamics.sound_speed / C
        ),
    )
    audit = audit_causal_stress_characteristics(
        geometry,
        eos,
        closure,
        surface_density=1.0e9,
        radial_velocity_over_c=-0.40,
        azimuthal_velocity_over_c=0.60,
        temperature=3.0e8,
    )

    assert audit.causal_and_hyperbolic
    assert audit.causally_outgoing_inner_edge


def test_flat_static_cooling_is_a_pure_killing_energy_sink() -> None:
    geometry = kerr_schild_column_geometry(10.0, 0.0)
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=2.0,
        radial_velocity_over_c=0.0,
        azimuthal_velocity_over_c=0.0,
        specific_internal_energy=0.03 * C**2,
        integrated_pressure=0.01 * C**2,
    )
    source = causal_comoving_energy_source(
        geometry,
        primitive,
        comoving_energy_rate=-3.0e20,
    )

    assert source.killing_source_per_ct[:3] == pytest.approx(
        np.zeros(3),
        abs=0.0,
    )
    assert source.killing_source_per_ct[3] == pytest.approx(
        -3.0e20 / C**3,
        rel=2.0e-15,
    )
    assert source.relative_identity_defect < 2.0e-15
    assert source.comoving_momentum_relative_defect < 2.0e-15


def test_rotating_cooling_four_force_has_no_comoving_momentum() -> None:
    _, gravitational_radius = _scales()
    geometry = kerr_schild_column_geometry(
        4.5 * gravitational_radius,
        gravitational_radius,
    )
    state = _eos(4.5).from_surface_density_temperature(1.0e7, 3.0e7)
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=state.surface_density,
        radial_velocity_over_c=-0.2,
        azimuthal_velocity_over_c=0.55,
        specific_internal_energy=state.specific_internal_energy,
        integrated_pressure=state.integrated_pressure,
    )
    source = causal_comoving_energy_source(
        geometry,
        primitive,
        comoving_energy_rate=-1.0e22,
    )

    assert source.killing_source_per_ct[0] == 0.0
    assert source.killing_source_per_ct[1] != 0.0
    assert source.killing_source_per_ct[2] < 0.0
    assert source.killing_source_per_ct[3] < 0.0
    assert source.relative_identity_defect < 2.0e-15
    assert source.comoving_momentum_relative_defect < 2.0e-15


def test_vertical_compression_can_balance_diffusion_cooling_locally() -> None:
    _, gravitational_radius = _scales()
    geometry = kerr_schild_column_geometry(
        20.0 * gravitational_radius,
        gravitational_radius,
    )
    eos = _eos(20.0)
    thermodynamics = eos.from_surface_density_temperature(1.0e7, 1.0e7)
    cooling, optical_depth = causal_diffusion_cooling_rate(thermodynamics)
    expansion_rate = -cooling / thermodynamics.integrated_pressure
    source = causal_thermal_column_source(
        geometry,
        eos,
        surface_density=1.0e7,
        radial_velocity_over_c=-0.01,
        azimuthal_velocity_over_c=0.20,
        temperature=1.0e7,
        proper_log_height_rate=expansion_rate,
    )

    assert optical_depth > 1.0
    assert source.vertical_work_rate == pytest.approx(cooling)
    assert source.total_killing_source_per_ct == pytest.approx(
        np.zeros(4),
        abs=1.0e-20,
    )
    assert source.local_viscous_energy_source == 0.0


def test_temporal_vertical_work_is_antisymmetric() -> None:
    eos = _eos(20.0)
    old = eos.from_surface_density_temperature(1.0e7, 1.0e7)
    new = eos.from_surface_density_temperature(1.1e7, 1.2e7)

    forward = temporal_vertical_work_per_area(old, new)
    reverse = temporal_vertical_work_per_area(new, old)

    assert forward == pytest.approx(-reverse, rel=2.0e-15)


def test_stress_work_product_rule_has_no_total_energy_source() -> None:
    partition = causal_stress_work_partition(
        left_angular_velocity=4.0,
        right_angular_velocity=3.0,
        left_torque=7.0,
        right_torque=9.0,
    )

    assert partition.product_rule_defect == pytest.approx(0.0, abs=1.0e-14)
    assert (
        partition.torque_work_flux_difference
        == pytest.approx(
            partition.angular_exchange_work
            + partition.shear_conversion_work,
            abs=1.0e-14,
        )
    )
    assert partition.shear_conversion_work < 0.0
    assert partition.explicit_total_energy_heating_source == 0.0


def test_thermal_killing_source_midpoint_integration_converges() -> None:
    mass, gravitational_radius = _scales()
    left = 10.0 * gravitational_radius
    right = 30.0 * gravitational_radius
    reference_radius = 20.0 * gravitational_radius

    def weighted_energy_source(radius: float) -> float:
        geometry = kerr_schild_column_geometry(
            radius,
            gravitational_radius,
        )
        ratio = radius / reference_radius
        eos = QuasiHydrostaticGasRadiationColumnEOS(
            proper_vertical_frequency=np.sqrt(G * mass / radius**3)
        )
        source = causal_thermal_column_source(
            geometry,
            eos,
            surface_density=1.0e8 * ratio ** (-0.5),
            radial_velocity_over_c=-0.03,
            azimuthal_velocity_over_c=0.20,
            temperature=1.0e7 * ratio ** (-0.2),
            proper_log_height_rate=2.0e-3 * ratio ** (-1.0),
        )
        return float(
            geometry.face_measure
            * source.total_killing_source_per_ct[3]
        )

    reference = quad(
        weighted_energy_source,
        left,
        right,
        epsabs=0.0,
        epsrel=2.0e-11,
        limit=200,
    )[0]
    errors = []
    for cells in (16, 32, 64, 128):
        grid = make_kerr_schild_column_grid(
            left,
            right,
            cells,
            gravitational_radius,
        )
        midpoint = sum(
            weighted_energy_source(radius)
            / kerr_schild_column_geometry(
                radius,
                gravitational_radius,
            ).face_measure
            * measure
            for radius, measure in zip(
                grid.centers,
                grid.cell_measures,
                strict=True,
            )
        )
        errors.append(abs(midpoint / reference - 1.0))

    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.min(orders) > 1.95
    assert errors[-1] < 1.2e-5
