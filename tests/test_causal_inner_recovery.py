from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    FixedHeightGasRadiationColumnEOS,
    audit_gas_radiation_valencia_eigensystem,
    recover_valencia_gas_radiation_primitives,
    schwarzschild_kerr_schild_geometry,
    valencia_gas_radiation_column_state,
)
from imri_qpe.parameters import FiducialParams


def _geometry(radius_over_rg: float):
    parameters = FiducialParams()
    reference = schwarzschild_kerr_schild_geometry(
        1.0,
        parameters.M2_g,
    )
    return schwarzschild_kerr_schild_geometry(
        radius_over_rg * reference.gravitational_radius,
        parameters.M2_g,
    )


def _column_eos() -> FixedHeightGasRadiationColumnEOS:
    return FixedHeightGasRadiationColumnEOS(proper_half_thickness=1.0e7)


def test_fixed_height_eos_inverts_gas_and_radiation_states() -> None:
    eos = _column_eos()

    for surface_density, temperature in (
        (1.0e7, 1.0e6),
        (1.0e5, 3.0e7),
        (1.0e3, 3.0e8),
    ):
        state = eos.from_surface_density_temperature(
            surface_density,
            temperature,
        )
        recovered = eos.from_surface_density_internal_energy(
            surface_density,
            state.specific_internal_energy,
        )

        assert recovered.temperature == pytest.approx(
            temperature,
            rel=2.0e-13,
        )
        assert recovered.integrated_pressure == pytest.approx(
            state.integrated_pressure,
            rel=8.0e-13,
        )


def test_radiation_dominated_sound_speed_approaches_c_over_sqrt_three() -> None:
    state = _column_eos().from_surface_density_temperature(
        1.0e5,
        1.0e9,
    )

    assert state.sound_speed / C == pytest.approx(
        1.0 / np.sqrt(3.0),
        rel=3.0e-4,
    )
    assert state.sound_speed < C


@pytest.mark.parametrize(
    (
        "radius_over_rg",
        "surface_density",
        "temperature",
        "radial_velocity",
        "azimuthal_velocity",
    ),
    (
        (20.0, 1.0e5, 1.0e6, -0.01, 0.20),
        (4.5, 1.0e5, 3.0e7, -0.20, 0.55),
        (1.8, 1.0e5, 1.0e8, -0.40, 0.60),
        (1.8, 1.0e3, 3.0e8, 0.20, 0.75),
    ),
)
def test_valencia_gas_radiation_round_trip(
    radius_over_rg: float,
    surface_density: float,
    temperature: float,
    radial_velocity: float,
    azimuthal_velocity: float,
) -> None:
    geometry = _geometry(radius_over_rg)
    eos = _column_eos()
    state, _ = valencia_gas_radiation_column_state(
        geometry,
        eos,
        surface_density=surface_density,
        radial_velocity_over_c=radial_velocity,
        azimuthal_velocity_over_c=azimuthal_velocity,
        temperature=temperature,
    )
    recovered = recover_valencia_gas_radiation_primitives(
        geometry,
        eos,
        state.conserved,
    )

    assert recovered.maximum_relative_conserved_defect < 2.0e-11
    assert recovered.primitive.surface_density == pytest.approx(
        surface_density,
        rel=2.0e-11,
    )
    assert recovered.primitive.radial_velocity_over_c == pytest.approx(
        radial_velocity,
        rel=2.0e-11,
        abs=2.0e-13,
    )
    assert recovered.primitive.azimuthal_velocity_over_c == pytest.approx(
        azimuthal_velocity,
        rel=2.0e-11,
        abs=2.0e-13,
    )
    assert recovered.primitive.temperature == pytest.approx(
        temperature,
        rel=2.0e-8,
    )
    assert recovered.pressure_root_iterations < 20


def test_declared_wp10c1_matrix_passes_primitive_gate() -> None:
    eos = _column_eos()
    maximum_defect = 0.0

    for radius, radial_velocity, azimuthal_velocity in (
        (20.0, -0.01, 0.20),
        (4.5, -0.20, 0.55),
        (1.8, -0.40, 0.60),
    ):
        geometry = _geometry(radius)
        for surface_density, temperature in (
            (1.0e7, 1.0e7),
            (1.0e5, 3.0e7),
            (1.0e3, 3.0e8),
        ):
            state, _ = valencia_gas_radiation_column_state(
                geometry,
                eos,
                surface_density=surface_density,
                radial_velocity_over_c=radial_velocity,
                azimuthal_velocity_over_c=azimuthal_velocity,
                temperature=temperature,
            )
            recovered = recover_valencia_gas_radiation_primitives(
                geometry,
                eos,
                state.conserved,
            ).primitive
            defects = (
                abs(recovered.surface_density / surface_density - 1.0),
                abs(
                    recovered.radial_velocity_over_c / radial_velocity
                    - 1.0
                ),
                abs(
                    recovered.azimuthal_velocity_over_c
                    / azimuthal_velocity
                    - 1.0
                ),
                abs(recovered.temperature / temperature - 1.0),
            )
            maximum_defect = max(maximum_defect, *defects)

    assert maximum_defect < 1.0e-10


@pytest.mark.parametrize(
    (
        "radius_over_rg",
        "temperature",
        "radial_velocity",
        "azimuthal_velocity",
    ),
    (
        (20.0, 1.0e6, -0.01, 0.20),
        (4.5, 3.0e7, -0.20, 0.55),
        (1.8, 1.0e8, -0.40, 0.60),
    ),
)
def test_gas_radiation_characteristics_match_conservative_jacobian(
    radius_over_rg: float,
    temperature: float,
    radial_velocity: float,
    azimuthal_velocity: float,
) -> None:
    audit = audit_gas_radiation_valencia_eigensystem(
        _geometry(radius_over_rg),
        _column_eos(),
        surface_density=1.0e5,
        radial_velocity_over_c=radial_velocity,
        azimuthal_velocity_over_c=azimuthal_velocity,
        temperature=temperature,
    )

    assert audit.maximum_eigenvalue_defect < 1.0e-7
    assert audit.stationary_flux_rank == 4
    if radius_over_rg < 2.0:
        assert audit.causally_outgoing_inner_edge


def test_recovery_rejects_nonphysical_conserved_state() -> None:
    with pytest.raises(ValueError, match="pressure root"):
        recover_valencia_gas_radiation_primitives(
            _geometry(4.5),
            _column_eos(),
            np.asarray([1.0, 100.0, 0.0, 0.0]),
        )


def test_recovery_rejects_invalid_conserved_shape() -> None:
    with pytest.raises(ValueError, match="length-four"):
        recover_valencia_gas_radiation_primitives(
            _geometry(4.5),
            _column_eos(),
            np.ones(3),
        )
