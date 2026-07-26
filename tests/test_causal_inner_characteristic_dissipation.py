from __future__ import annotations

from dataclasses import replace

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_characteristic_dissipation,
    causal_five_field_coordinate_principal_basis,
    causal_five_field_descriptor_jump,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)


def _inputs(n_cells: int = 8):
    context = make_causal_five_field_regression_context(n_cells)
    primitives = make_causal_five_field_seed(context).primitives
    return context, primitives


def test_coordinate_principal_basis_is_complete_and_causal() -> None:
    context, primitives = _inputs()
    cell = 2
    basis = causal_five_field_coordinate_principal_basis(
        context,
        float(context.grid.centers[cell]),
        primitives[cell],
    )

    assert basis.passed
    assert basis.temporal_storage_matrix.shape == (5, 5)
    assert basis.spatial_principal_matrix.shape == (5, 5)
    assert np.all(np.diff(basis.numerical_speeds_over_c) > 0.0)
    assert basis.maximum_eigenpair_defect <= 1.0e-10
    assert basis.maximum_biorthogonality_defect <= 1.0e-10
    assert basis.maximum_imaginary_part == 0.0
    np.testing.assert_allclose(
        basis.descriptor_left_eigenvectors
        @ basis.descriptor_right_eigenvectors,
        np.eye(5),
        rtol=0.0,
        atol=1.0e-10,
    )


def test_descriptor_jump_and_matrix_penalty_are_symmetric_and_dissipative() -> None:
    context, primitives = _inputs()
    face = 3
    radius = float(context.grid.edges[face])
    left = np.array(primitives[face - 1], copy=True)
    right = np.array(primitives[face], copy=True)
    forward = causal_five_field_characteristic_dissipation(
        context,
        radius,
        left,
        right,
        face_measure=float(context.grid.face_measures[face]),
    )
    reverse_jump = causal_five_field_descriptor_jump(
        context,
        radius,
        right,
        left,
    )
    constant = causal_five_field_characteristic_dissipation(
        context,
        radius,
        left,
        left,
        face_measure=float(context.grid.face_measures[face]),
    )

    np.testing.assert_allclose(
        reverse_jump,
        -forward.descriptor_jump,
        rtol=3.0e-12,
        atol=1.0e-24,
    )
    assert forward.quadratic_dissipation >= 0.0
    assert forward.scalar_equal_speed_defect <= 1.0e-10
    assert np.array_equal(constant.descriptor_jump, np.zeros(5))
    assert np.array_equal(constant.dissipative_flux_over_c, np.zeros(5))
    assert constant.quadratic_dissipation == 0.0


def test_characteristic_matrix_is_audit_only_and_returns_one_face_flux() -> None:
    context, primitives = _inputs()
    production = causal_five_field_state_from_primitives(
        context,
        primitives,
    )
    audit_context = replace(
        context,
        interior_dissipation_mode="characteristic_matrix_audit",
    ).validated()
    candidate = causal_five_field_state_from_primitives(
        audit_context,
        primitives,
    )
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(candidate),
        audit_context,
    )

    assert context.interior_dissipation_mode == "scalar_rusanov"
    assert audit_context.interior_dissipation_mode == (
        "characteristic_matrix_audit"
    )
    assert evaluation.numerical_weighted_face_fluxes_over_c.shape == (
        primitives.shape[0] + 1,
        5,
    )
    assert np.all(
        np.isfinite(evaluation.numerical_weighted_face_fluxes_over_c)
    )
    assert not np.array_equal(
        production.weighted_face_fluxes_over_c[1:-1],
        candidate.weighted_face_fluxes_over_c[1:-1],
    )
