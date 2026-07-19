from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    causal_coincident_fine_faces,
    causal_five_field_constraint_manifold_jvp,
    causal_five_field_consistent_tangent_decomposition,
    causal_five_field_residual_terms,
    causal_five_field_state_from_primitives,
    causal_five_field_term_reconstruction_defect,
    causal_nested_refinement_ratio,
    causal_restrict_cell_averages,
    causal_restrict_cell_integrals,
    causal_spatial_contraction_order,
    causal_spatial_difference_metrics,
    evaluate_causal_five_field_dae,
    causal_five_field_regression_seed_parameters,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)


def _nested_contexts():
    return (
        make_causal_five_field_regression_context(4),
        make_causal_five_field_regression_context(8),
    )


def test_nested_restriction_preserves_integrals_and_averages() -> None:
    coarse, fine = _nested_contexts()
    ratio = causal_nested_refinement_ratio(coarse.grid, fine.grid)
    fine_integrals = np.arange(16, dtype=float).reshape(8, 2)
    expected_integrals = fine_integrals.reshape(4, 2, 2).sum(axis=1)

    assert ratio == 2
    assert np.array_equal(
        causal_restrict_cell_integrals(
            coarse.grid,
            fine.grid,
            fine_integrals,
        ),
        expected_integrals,
    )

    constant = np.column_stack(
        (
            np.full(8, 2.5),
            np.full(8, -4.0),
        )
    )
    restricted = causal_restrict_cell_averages(
        coarse.grid,
        fine.grid,
        constant,
    )
    assert np.allclose(restricted, constant[::2], rtol=2.0e-15)


def test_nested_face_selection_uses_exact_coincident_faces() -> None:
    coarse, fine = _nested_contexts()
    values = np.arange(18, dtype=float).reshape(9, 2)

    selected = causal_coincident_fine_faces(
        coarse.grid,
        fine.grid,
        values,
    )

    assert np.array_equal(selected, values[::2])
    assert np.array_equal(coarse.grid.edges, fine.grid.edges[::2])


def test_spatial_metrics_are_weighted_and_can_exclude_boundaries() -> None:
    left = np.asarray((8.0, 2.0, 3.0, 10.0))
    right = np.asarray((0.0, 1.0, 1.0, 0.0))
    measures = np.asarray((1.0, 2.0, 3.0, 4.0))
    radii = np.asarray((1.0, 2.0, 3.0, 4.0))

    full = causal_spatial_difference_metrics(
        left,
        right,
        measures,
        radii,
    )
    interior = causal_spatial_difference_metrics(
        left,
        right,
        measures,
        radii,
        exclude_boundary_cells=1,
    )

    assert full["maximum_absolute_difference"] == 10.0
    assert full["maximum_difference_radius"] == 4.0
    assert full["measure_weighted_l1_difference"] == pytest.approx(5.6)
    assert interior["maximum_absolute_difference"] == 2.0
    assert interior["maximum_difference_radius"] == 3.0
    assert interior["measure_weighted_l1_difference"] == pytest.approx(1.6)


def test_spatial_contraction_order_uses_successive_pair_differences() -> None:
    assert causal_spatial_contraction_order(0.8, 0.4) == pytest.approx(1.0)
    assert causal_spatial_contraction_order(0.8, 0.2) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="positive and finite"):
        causal_spatial_contraction_order(0.8, 0.0)


def test_residual_term_decomposition_reconstructs_conservation_rows() -> None:
    context = make_causal_five_field_regression_context(4)
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )
    evaluation = evaluate_causal_five_field_dae(vector, context)
    terms = causal_five_field_residual_terms(
        context,
        vector,
        evaluation,
    )
    defect = causal_five_field_term_reconstruction_defect(
        evaluation,
        terms,
    )

    assert set(terms) == {
        "temporal_conserved_storage",
        "temporal_vertical_storage",
        "central_face_transport",
        "rusanov_face_transport",
        "flux_primary_closure",
        "perfect_fluid_geometry",
        "stress_geometry",
        "radiative_cooling",
        "vertical_work",
        "stress_relaxation",
        "stream",
    }
    assert defect["maximum_relative_defect"] < 2.0e-15


def test_consistent_tangent_decomposition_closes_component_sum() -> None:
    context = make_causal_five_field_regression_context(4)
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )

    decomposition = causal_five_field_consistent_tangent_decomposition(
        context,
        vector,
    )
    component_sum = np.sum(
        np.asarray(
            [
                component["log_h_over_r_tangent_per_s"]
                for component in decomposition["components"].values()
            ]
        ),
        axis=0,
    )

    assert decomposition["maximum_scaled_consistency_defect"] < 1.0e-9
    assert (
        decomposition[
            "maximum_residual_reconstruction_relative_defect"
        ]
        < 1.0e-12
    )
    assert (
        decomposition[
            "maximum_tangent_reconstruction_relative_defect"
        ]
        < 1.0e-8
    )
    assert np.allclose(
        component_sum,
        decomposition["full"]["log_h_over_r_tangent_per_s"],
        rtol=1.0e-8,
        atol=1.0e-12,
    )


def test_constraint_manifold_jvp_reconstructs_term_derivative() -> None:
    context = make_causal_five_field_regression_context(
        4,
        spatial_reconstruction="plm_smooth",
        boundary_trace_reconstruction="plm_one_sided",
        cell_rate_scheme="quadratic_log_radius",
        cell_source_quadrature="gauss_legendre_4",
        cell_storage_quadrature="gauss_legendre_4",
    )
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )
    radius = context.grid.centers / context.grid.gravitational_radius
    direction = np.zeros((4, 5), dtype=float)
    direction[:, 3] = np.exp(
        -0.5 * (np.log(radius / 20.0) / 0.35) ** 2
    )
    audit = causal_five_field_constraint_manifold_jvp(
        context,
        vector,
        direction,
    )

    assert audit["maximum_reconstruction_relative_defect"] < 1.0e-7
    assert set(audit["term_jvps"]) == {
        "temporal_conserved_storage",
        "temporal_vertical_storage",
        "central_face_transport",
        "rusanov_face_transport",
        "flux_primary_closure",
        "perfect_fluid_geometry",
        "stress_geometry",
        "radiative_cooling",
        "vertical_work",
        "stress_relaxation",
        "stream",
    }
