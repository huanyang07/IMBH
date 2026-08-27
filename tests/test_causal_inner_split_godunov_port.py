import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.causal_inner_split_godunov_port import (
    SPLIT_ELEVEN_FIELD_NAMES,
    audit_split_godunov_port_hamiltonian_form,
    build_split_godunov_port_hamiltonian_form,
    vertical_port_hamiltonian,
)


def _form(transport=-0.27):
    return build_split_godunov_port_hamiltonian_form(
        proper_half_thickness=2.45860382301911e8,
        temperature=4.436398409641123e6,
        proper_vertical_frequency=8.279018646718441,
        alpha=0.1,
        transport_speed_over_c=transport,
    )


def test_split_field_count_is_nine_plus_two():
    form = _form()
    assert len(SPLIT_ELEVEN_FIELD_NAMES) == 11
    assert form.transport_temporal_matrix.shape == (9, 9)
    assert form.vertical_port.entropy_metric.shape == (2, 2)
    assert form.temporal_matrix.shape == (11, 11)


def test_vertical_port_is_positive_skew_and_dissipative():
    port = vertical_port_hamiltonian(
        proper_half_thickness=2.0e8,
        temperature=5.0e6,
        proper_vertical_frequency=7.0,
        alpha=0.1,
        transport_speed_over_c=-0.2,
    )
    assert np.min(np.linalg.eigvalsh(port.entropy_metric)) > 0.0
    np.testing.assert_allclose(
        port.reversible_source_matrix + port.reversible_source_matrix.T,
        0.0,
        atol=1.0e-15,
    )
    assert np.max(
        np.linalg.eigvalsh(0.5 * (port.source_matrix + port.source_matrix.T))
    ) <= 0.0


@pytest.mark.parametrize("transport", (-0.40, -0.20, 0.0, 0.20))
def test_split_reference_form_is_symmetric_hyperbolic_and_causal(transport):
    audit = audit_split_godunov_port_hamiltonian_form(_form(transport))
    assert audit.passed
    assert audit.maximum_absolute_characteristic_speed_over_c <= 0.999
    assert audit.port_skew_relative_defect <= 1.0e-12
    assert audit.source_entropy_positive_part <= 1.0e-12


def test_invalid_port_fails_closed():
    with pytest.raises(ValueError, match="causal"):
        vertical_port_hamiltonian(
            proper_half_thickness=2.0e8,
            temperature=5.0e6,
            proper_vertical_frequency=7.0,
            alpha=0.1,
            transport_speed_over_c=1.0,
        )
