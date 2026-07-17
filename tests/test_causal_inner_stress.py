from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    FixedHeightGasRadiationColumnEOS,
    ValenciaPerfectFluidPrimitive,
    audit_advected_stress_flux_eigensystem,
    audit_causal_stress_characteristics,
    calibrate_causal_alpha_shear,
    causal_stress_column_state,
    causal_stress_relaxation_source,
    causal_stress_torque_and_power,
    equilibrium_alpha_specific_stress,
    kerr_schild_column_geometry,
)
from imri_qpe.parameters import FiducialParams


def _scales() -> tuple[float, float]:
    mass = FiducialParams().M2_g
    return mass, G * mass / C**2


def _eos() -> FixedHeightGasRadiationColumnEOS:
    return FixedHeightGasRadiationColumnEOS(
        proper_half_thickness=1.0e7
    )


def _primitive(
    surface_density: float,
    temperature: float,
    radial_velocity: float,
    azimuthal_velocity: float,
) -> ValenciaPerfectFluidPrimitive:
    thermodynamics = _eos().from_surface_density_temperature(
        surface_density,
        temperature,
    )
    return ValenciaPerfectFluidPrimitive(
        surface_density=surface_density,
        radial_velocity_over_c=radial_velocity,
        azimuthal_velocity_over_c=azimuthal_velocity,
        specific_internal_energy=thermodynamics.specific_internal_energy,
        integrated_pressure=thermodynamics.integrated_pressure,
    )


def _closure(
    radius_over_rg: float,
    primitive: ValenciaPerfectFluidPrimitive,
):
    mass, gravitational_radius = _scales()
    radius = radius_over_rg * gravitational_radius
    orbital_frequency = np.sqrt(G * mass / radius**3)
    thermodynamics = _eos().from_surface_density_internal_energy(
        primitive.surface_density,
        primitive.specific_internal_energy,
    )
    return calibrate_causal_alpha_shear(
        primitive,
        alpha=0.1,
        reference_positive_shear_rate=1.5 * orbital_frequency,
        viscous_signal_speed_over_c=(
            np.sqrt(0.1) * thermodynamics.sound_speed / C
        ),
    )


def test_causal_shear_calibration_recovers_common_alpha_stress() -> None:
    primitive = _primitive(1.0e5, 3.0e7, -0.2, 0.55)
    closure = _closure(4.5, primitive)

    assert closure.target_specific_stress(
        closure.reference_positive_shear_rate
    ) == pytest.approx(
        equilibrium_alpha_specific_stress(primitive, alpha=0.1),
        rel=2.0e-15,
    )
    assert (
        closure.specific_shear_viscosity_seconds
        / (
            closure.relaxation_time
            * closure.specific_enthalpy_over_c2
        )
    ) == pytest.approx(
        closure.viscous_signal_speed_over_c**2,
        rel=2.0e-15,
    )
    assert closure.relaxation_time > 0.0


@pytest.mark.parametrize(
    (
        "radius_over_rg",
        "surface_density",
        "temperature",
        "radial_velocity",
        "azimuthal_velocity",
    ),
    (
        (20.0, 1.0e7, 1.0e7, -0.01, 0.20),
        (4.5, 1.0e5, 3.0e7, -0.20, 0.55),
        (1.8, 1.0e3, 3.0e8, -0.40, 0.60),
    ),
)
def test_causal_stress_tensor_and_characteristics_pass_local_gates(
    radius_over_rg: float,
    surface_density: float,
    temperature: float,
    radial_velocity: float,
    azimuthal_velocity: float,
) -> None:
    _, gravitational_radius = _scales()
    geometry = kerr_schild_column_geometry(
        radius_over_rg * gravitational_radius,
        gravitational_radius,
    )
    primitive = _primitive(
        surface_density,
        temperature,
        radial_velocity,
        azimuthal_velocity,
    )
    closure = _closure(radius_over_rg, primitive)
    state = causal_stress_column_state(
        geometry,
        primitive,
        specific_stress=closure.equilibrium_specific_stress,
    )
    audit = audit_causal_stress_characteristics(
        geometry,
        _eos(),
        closure,
        surface_density=surface_density,
        radial_velocity_over_c=radial_velocity,
        azimuthal_velocity_over_c=azimuthal_velocity,
        temperature=temperature,
    )

    assert state.tensor_trace_relative_defect < 5.0e-15
    assert state.tensor_orthogonality_relative_defect < 5.0e-15
    assert state.radial_work_relative_defect < 5.0e-15
    assert audit.causal_and_hyperbolic
    assert audit.stationary_flux_rank == 5
    if radius_over_rg < 2.0:
        assert audit.causally_outgoing_inner_edge


def test_stationary_circular_stress_pairs_torque_and_killing_power() -> None:
    mass, gravitational_radius = _scales()
    radius_over_rg = 20.0
    radius = radius_over_rg * gravitational_radius
    geometry = kerr_schild_column_geometry(radius, gravitational_radius)
    thermodynamics = _eos().from_surface_density_temperature(
        1.0e5,
        3.0e7,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=1.0e5,
        radial_velocity_over_c=2.0 / radius_over_rg,
        azimuthal_velocity_over_c=(
            np.sqrt(gravitational_radius / radius)
            / geometry.base.lapse
        ),
        specific_internal_energy=thermodynamics.specific_internal_energy,
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    state = causal_stress_column_state(
        geometry,
        primitive,
        specific_stress=equilibrium_alpha_specific_stress(
            primitive,
            alpha=0.1,
        ),
    )
    torque, power = causal_stress_torque_and_power(geometry, state)

    assert torque > 0.0
    assert state.coordinate_angular_velocity == pytest.approx(
        np.sqrt(G * mass / radius**3),
        rel=2.0e-15,
    )
    assert power == pytest.approx(
        state.coordinate_angular_velocity * torque,
        rel=8.0e-15,
    )


def test_relativistic_torque_converges_to_common_weak_field_limit() -> None:
    _, gravitational_radius = _scales()
    thermodynamics = _eos().from_surface_density_temperature(
        1.0e5,
        3.0e7,
    )
    relative_errors = []
    for radius_over_rg in (100.0, 1000.0, 10000.0):
        radius = radius_over_rg * gravitational_radius
        geometry = kerr_schild_column_geometry(
            radius,
            gravitational_radius,
        )
        primitive = ValenciaPerfectFluidPrimitive(
            surface_density=1.0e5,
            radial_velocity_over_c=2.0 / radius_over_rg,
            azimuthal_velocity_over_c=(
                np.sqrt(gravitational_radius / radius)
                / geometry.base.lapse
            ),
            specific_internal_energy=(
                thermodynamics.specific_internal_energy
            ),
            integrated_pressure=thermodynamics.integrated_pressure,
        )
        state = causal_stress_column_state(
            geometry,
            primitive,
            specific_stress=equilibrium_alpha_specific_stress(
                primitive,
                alpha=0.1,
            ),
        )
        torque, _ = causal_stress_torque_and_power(geometry, state)
        common_torque = (
            2.0
            * np.pi
            * radius**2
            * 0.1
            * thermodynamics.integrated_pressure
        )
        relative_errors.append(abs(torque / common_torque - 1.0))

    assert relative_errors[1] < 0.11 * relative_errors[0]
    assert relative_errors[2] < 0.11 * relative_errors[1]
    assert relative_errors[2] < 6.0e-5


def test_relaxation_source_has_equilibrium_and_restoring_signs() -> None:
    _, gravitational_radius = _scales()
    geometry = kerr_schild_column_geometry(
        4.5 * gravitational_radius,
        gravitational_radius,
    )
    primitive = _primitive(1.0e5, 3.0e7, -0.2, 0.55)
    closure = _closure(4.5, primitive)
    equilibrium = causal_stress_column_state(
        geometry,
        primitive,
        specific_stress=closure.equilibrium_specific_stress,
    )
    low = causal_stress_column_state(
        geometry,
        primitive,
        specific_stress=0.5 * closure.equilibrium_specific_stress,
    )
    high = causal_stress_column_state(
        geometry,
        primitive,
        specific_stress=1.5 * closure.equilibrium_specific_stress,
    )
    arguments = {
        "positive_shear_rate": closure.reference_positive_shear_rate
    }

    assert causal_stress_relaxation_source(
        geometry,
        equilibrium,
        closure,
        **arguments,
    ) == pytest.approx(0.0, abs=1.0e-20)
    assert causal_stress_relaxation_source(
        geometry,
        low,
        closure,
        **arguments,
    ) > 0.0
    assert causal_stress_relaxation_source(
        geometry,
        high,
        closure,
        **arguments,
    ) < 0.0


def test_advected_stress_without_shear_principal_term_is_rejected() -> None:
    _, gravitational_radius = _scales()
    geometry = kerr_schild_column_geometry(
        20.0 * gravitational_radius,
        gravitational_radius,
    )
    primitive = _primitive(1.0e7, 1.0e7, -0.01, 0.20)
    closure = _closure(20.0, primitive)
    defects = []
    for step in (1.0e-3, 2.0e-4, 1.0e-4):
        audit = audit_advected_stress_flux_eigensystem(
            geometry,
            _eos(),
            closure,
            surface_density=1.0e7,
            radial_velocity_over_c=-0.01,
            azimuthal_velocity_over_c=0.20,
            temperature=1.0e7,
            finite_difference_step=step,
        )
        defects.append(audit.maximum_imaginary_eigenvalue)
        assert not audit.hyperbolic

    assert min(defects) > 6.0e-5
    assert max(defects) / min(defects) < 1.001
