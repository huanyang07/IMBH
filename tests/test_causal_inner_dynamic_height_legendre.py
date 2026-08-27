import numpy as np
import pytest

from imri_qpe.constants import A_RAD, DEFAULT_MU_MOL
from imri_qpe.layer3_minidisk_1d.causal_inner_dynamic_height_legendre import (
    centered_dynamic_height_entropy_hessian,
    dynamic_height_entropy_state,
    equilibrium_dynamic_height_conserved,
    height_force_identity_defect,
)
from imri_qpe.scales import gas_constant_per_gram


def _fixture():
    sigma = 115.9571604613971
    temperature = 4436398.409641123
    height = 245860382.301911
    omega = 8.279018646718441
    conserved = equilibrium_dynamic_height_conserved(
        surface_mass=sigma,
        temperature=temperature,
        proper_half_thickness=height,
        proper_vertical_frequency=omega,
    )
    return conserved, omega, temperature


def test_equilibrium_state_recovers_exact_gas_radiation_column():
    conserved, omega, temperature = _fixture()
    state = dynamic_height_entropy_state(
        conserved,
        proper_vertical_frequency=omega,
        temperature_seed=temperature,
    )
    gas_constant = gas_constant_per_gram(DEFAULT_MU_MOL)
    expected_internal = (
        gas_constant * temperature / (5.0 / 3.0 - 1.0)
        + A_RAD * temperature**4 / state.density
    )
    assert state.temperature == pytest.approx(temperature, rel=3.0e-14)
    assert state.specific_internal_energy == pytest.approx(
        expected_internal, rel=3.0e-15
    )
    assert height_force_identity_defect(
        state, proper_vertical_frequency=omega
    ) < 2.0e-15


def test_height_entropy_hessian_is_symmetric_and_exposes_physical_obstruction():
    conserved, omega, temperature = _fixture()
    audit = centered_dynamic_height_entropy_hessian(
        conserved,
        proper_vertical_frequency=omega,
        temperature_seed=temperature,
        step_factor=1.0e-3,
    )
    np.testing.assert_allclose(audit.scaled_hessian, audit.scaled_hessian.T)
    assert audit.symmetry_defect == 0.0
    assert audit.equilibrated_eigenvalues[0] < -1.0
    assert audit.equilibrated_eigenvalues[-1] > 1.0


def test_invalid_height_candidate_fails_closed():
    conserved, omega, temperature = _fixture()
    invalid = np.array(conserved, copy=True)
    invalid[2] = 0.0
    with pytest.raises(ValueError, match="positive"):
        dynamic_height_entropy_state(
            invalid,
            proper_vertical_frequency=omega,
            temperature_seed=temperature,
        )
