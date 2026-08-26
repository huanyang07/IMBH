from dataclasses import replace

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.causal_inner_eleven_field_convex import (
    ELEVEN_FIELD_PERTURBATION_NAMES,
    N_ELEVEN_FIELDS,
    N_FULL_SHEAR_AMPLITUDES,
    N_VERTICAL_EQUILIBRIUM_FIELDS,
    ElevenFieldConvexNormalFormParameters,
    audit_eleven_field_convex_normal_form,
    audit_full_shear_rest_frame,
    build_eleven_field_convex_normal_form,
    full_shear_rest_frame,
    one_Rphi_amplitude_embedding,
    project_full_shear_amplitudes,
    reconstruct_full_shear_tensor,
    reference_eleven_field_parameters,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (
    kerr_schild_column_geometry,
)


def _frame(beta_z=0.021):
    geometry = kerr_schild_column_geometry(3.0351964349786267e9, 8.0e8)
    return full_shear_rest_frame(
        geometry,
        radial_velocity_over_c=0.059228133078858144,
        azimuthal_velocity_over_c=0.8785842814260398,
        vertical_velocity_over_c=beta_z,
    )


def test_field_count_preserves_all_five_shear_components():
    assert N_FULL_SHEAR_AMPLITUDES == 5
    assert N_ELEVEN_FIELDS == 11
    assert N_VERTICAL_EQUILIBRIUM_FIELDS == 9
    assert len(ELEVEN_FIELD_PERTURBATION_NAMES) == 11


def test_full_shear_basis_constraints_and_roundtrip():
    audit = audit_full_shear_rest_frame(_frame())
    assert audit.passed
    assert audit.basis_gram_defect <= 2.0e-13
    assert audit.basis_trace_defect <= 2.0e-13
    assert audit.basis_velocity_orthogonality_defect <= 2.0e-13
    assert audit.amplitude_roundtrip_defect <= 2.0e-13


def test_one_Rphi_embedding_recovers_unprojected_tensor():
    frame = _frame(beta_z=0.0)
    chi = 8.073892986809419e-4
    tensor = reconstruct_full_shear_tensor(
        frame, one_Rphi_amplitude_embedding(chi)
    )
    expected = chi * (
        np.outer(frame.rest_triad[0], frame.rest_triad[1])
        + np.outer(frame.rest_triad[1], frame.rest_triad[0])
    )
    np.testing.assert_allclose(tensor, expected, atol=2.0e-16, rtol=2.0e-14)


def test_projection_roundtrip_for_random_amplitudes():
    random = np.random.default_rng(20260826)
    frame = _frame()
    for _ in range(64):
        amplitudes = random.normal(scale=0.03, size=5)
        tensor = reconstruct_full_shear_tensor(frame, amplitudes, stress_scale=4.2)
        recovered = project_full_shear_amplitudes(frame, tensor, stress_scale=4.2)
        np.testing.assert_allclose(recovered, amplitudes, atol=2.0e-14, rtol=0.0)


def test_invalid_velocity_and_coefficient_inputs_fail_closed():
    geometry = kerr_schild_column_geometry(3.1e9, 8.0e8)
    with pytest.raises(ValueError, match="subluminal"):
        full_shear_rest_frame(
            geometry,
            radial_velocity_over_c=0.9,
            azimuthal_velocity_over_c=0.8,
            vertical_velocity_over_c=0.0,
        )
    reference = reference_eleven_field_parameters()
    with pytest.raises(ValueError, match="positive"):
        replace(reference, shear_weights=(1.0, 1.0, 0.0, 1.0, 1.0))


def test_reference_common_potential_is_symmetric_hyperbolic_and_dissipative():
    parameters = reference_eleven_field_parameters()
    form = build_eleven_field_convex_normal_form(parameters)
    np.testing.assert_allclose(form.temporal_matrix, form.temporal_matrix.T)
    np.testing.assert_allclose(form.radial_matrix, form.radial_matrix.T)
    assert np.min(np.linalg.eigvalsh(form.temporal_matrix)) > 0.0
    assert np.max(
        np.linalg.eigvalsh(0.5 * (form.source_matrix + form.source_matrix.T))
    ) <= 1.0e-13
    audit = audit_eleven_field_convex_normal_form(parameters)
    assert audit.passed
    assert audit.maximum_absolute_characteristic_speed_over_c < 1.0
    assert audit.subcharacteristic_interlacing_violation <= 1.0e-12


def test_vertical_equilibrium_compresses_only_the_vertical_pair():
    form = build_eleven_field_convex_normal_form(reference_eleven_field_parameters())
    transformed = form.chart_to_relaxation @ form.vertical_equilibrium_embedding
    expected = np.zeros((11, 9))
    expected[:4, :4] = np.eye(4)
    expected[6:, 4:] = np.eye(5)
    np.testing.assert_allclose(transformed, expected, atol=1.0e-15, rtol=0.0)


def test_random_structural_fixtures_retain_the_algebraic_theorem():
    random = np.random.default_rng(31120260826)
    reference = reference_eleven_field_parameters()
    for _ in range(48):
        parameters = ElevenFieldConvexNormalFormParameters(
            transport_speed_over_c=float(random.uniform(-0.05, 0.05)),
            surface_density_weight=float(random.uniform(1.1, 2.0)),
            radial_velocity_weight=float(random.uniform(1.5, 2.5)),
            azimuthal_velocity_weight=float(random.uniform(1.4, 2.2)),
            thermal_weight=float(random.uniform(0.9, 1.5)),
            vertical_velocity_weight=float(random.uniform(1.0, 1.8)),
            shear_weights=tuple(random.uniform(0.8, 1.3, size=5)),
            vertical_frequency=float(random.uniform(0.65, 0.95)),
            vertical_damping=float(random.uniform(0.0, 0.3)),
            shear_relaxation_rates=tuple(random.uniform(0.2, 0.6, size=5)),
            height_log_surface_density=float(random.uniform(-0.25, -0.05)),
            height_log_temperature=float(random.uniform(0.9, 1.5)),
            mass_radial_coupling=reference.mass_radial_coupling,
            radial_thermal_coupling=reference.radial_thermal_coupling,
            radial_height_coupling=reference.radial_height_coupling,
            thermal_height_coupling=reference.thermal_height_coupling,
            radial_shear_diagonal_couplings=(0.06, 0.04),
            azimuthal_Rphi_coupling=0.13,
            vertical_Rz_coupling=0.11,
        )
        audit = audit_eleven_field_convex_normal_form(parameters)
        assert audit.passed
