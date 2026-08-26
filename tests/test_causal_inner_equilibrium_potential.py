import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.causal_inner_equilibrium_potential import (
    analytic_potential_current_jacobian,
    audit_equilibrium_column_potential,
    density_from_mass_affinity,
    entropy_variables_from_primitive,
    equilibrium_column_potential_state,
    gas_radiation_specific_chemical_potential,
)


def _minkowski_velocity(beta=(0.06, 0.87, 0.015)):
    values = np.asarray(beta, dtype=float)
    gamma = 1.0 / np.sqrt(1.0 - values @ values)
    return np.concatenate(([gamma], gamma * values))


def test_chemical_affinity_density_roundtrip():
    for rho, temperature in ((4.0e-8, 8.0e5), (5.0e-7, 3.4e6), (2.0e-6, 9.0e6)):
        chemical = gas_radiation_specific_chemical_potential(rho, temperature)
        recovered = density_from_mass_affinity(chemical / temperature, temperature)
        assert abs(recovered - rho) / rho <= 2.0e-9


def test_potential_derivatives_generate_mass_and_stress_energy():
    metric = np.diag((-1.0, 1.0, 1.0, 1.0))
    velocity = _minkowski_velocity()
    rho = 5.0e-7
    temperature = 3.4e6
    alpha, beta = entropy_variables_from_primitive(
        metric, velocity, density=rho, temperature=temperature
    )
    state = equilibrium_column_potential_state(
        metric,
        alpha,
        beta,
        proper_half_thickness=1.2e8,
    )
    jacobian = analytic_potential_current_jacobian(state)
    np.testing.assert_allclose(jacobian[:, 0], state.surface_mass_current)
    np.testing.assert_allclose(jacobian[:, 1:], state.column_stress_energy)


def test_complete_equilibrium_audit_passes_on_disk_like_states():
    metric = np.diag((-1.0, 1.0, 1.0, 1.0))
    for rho, temperature, height, velocity in (
        (4.0e-8, 8.0e5, 8.0e7, _minkowski_velocity((0.02, 0.75, 0.0))),
        (5.0e-7, 3.4e6, 1.2e8, _minkowski_velocity()),
        (2.0e-6, 9.0e6, 2.0e8, _minkowski_velocity((0.08, 0.6, -0.02))),
    ):
        audit = audit_equilibrium_column_potential(
            metric,
            velocity,
            density=rho,
            temperature=temperature,
            proper_half_thickness=height,
        )
        assert audit.passed, audit


def test_invalid_thermodynamic_and_timelike_inputs_fail_closed():
    with pytest.raises(ValueError, match="density"):
        gas_radiation_specific_chemical_potential(-1.0, 1.0e6)
    metric = np.diag((-1.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="timelike"):
        equilibrium_column_potential_state(
            metric,
            1.0,
            np.asarray((0.0, 1.0, 0.0, 0.0)),
            proper_half_thickness=1.0,
        )
