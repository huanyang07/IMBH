from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    causal_coincident_fine_faces,
    causal_five_field_assemble_evolving_tangent,
    causal_five_field_branch_frozen_mapped_storage_derivatives,
    causal_five_field_branch_frozen_mapped_storage_matrix,
    causal_five_field_constraint_manifold_jvp,
    causal_five_field_consistent_tangent_decomposition,
    causal_five_field_dae_scaling,
    causal_five_field_evolving_tangent_matrices,
    causal_five_field_mapped_conserved_from_primitives,
    causal_five_field_path_temporal_storage_increment,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_reduced_backward_euler_residual,
    causal_five_field_reduced_descriptor_matrices,
    causal_five_field_reduced_stationary_jacobian,
    causal_five_field_reduced_storage_action,
    causal_five_field_reduced_storage_matrices,
    causal_five_field_reduced_storage_rate_derivatives,
    causal_five_field_reduced_storage_rate_directional_derivative,
    causal_five_field_reduced_stationary_residual,
    causal_five_field_residual_terms,
    causal_five_field_scaled_primitive_vector_field,
    causal_five_field_state_from_primitives,
    causal_five_field_term_reconstruction_defect,
    causal_five_field_unified_mapped_storage_derivatives,
    causal_five_field_unified_mapped_storage_matrix,
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


def _independent_storage_component_matrices(
    context,
    primitives: np.ndarray,
    primitive_scales: np.ndarray,
    row_scales: np.ndarray,
    difference_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dense mapped and height storage matrices without coloring."""

    n_cells = int(context.grid.centers.size)
    n_reduced = 5 * n_cells
    charts = np.asarray(primitives, dtype=float).reshape(n_cells, 5)
    scales = np.asarray(primitive_scales, dtype=float)
    rows = np.asarray(row_scales, dtype=float)
    measures = context.grid.cell_measures[:, None]
    conserved = np.empty((n_reduced, n_reduced), dtype=float)
    vertical = np.empty_like(conserved)
    for column in range(n_reduced):
        increment = np.zeros(n_reduced, dtype=float)
        increment[column] = difference_step * scales[column]
        plus = (charts.ravel() + increment).reshape(n_cells, 5)
        minus = (charts.ravel() - increment).reshape(n_cells, 5)
        plus_conserved = (
            causal_five_field_mapped_conserved_from_primitives(
                context,
                plus,
            )
        )
        minus_conserved = (
            causal_five_field_mapped_conserved_from_primitives(
                context,
                minus,
            )
        )
        plus_path = causal_five_field_path_temporal_storage_increment(
            context,
            charts,
            plus,
        )
        minus_path = causal_five_field_path_temporal_storage_increment(
            context,
            charts,
            minus,
        )
        conserved[:, column] = (
            measures
            * (plus_conserved - minus_conserved)
            / (2.0 * C * difference_step)
        ).ravel() / rows
        vertical_difference = np.zeros((n_cells, 5), dtype=float)
        vertical_difference[:, :4] = (
            plus_path.vertical_killing_increment
            - minus_path.vertical_killing_increment
        )
        vertical[:, column] = (
            measures
            * vertical_difference
            / (2.0 * C * difference_step)
        ).ravel() / rows
    return conserved, vertical


def _independent_complete_storage_rate_action(
    context,
    primitives: np.ndarray,
    primitive_rate_per_s: np.ndarray,
    primitive_scales: np.ndarray,
    row_scales: np.ndarray,
    difference_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the complete storage to one rate without a mass matrix."""

    n_cells = int(context.grid.centers.size)
    charts = np.asarray(primitives, dtype=float).reshape(n_cells, 5)
    rate = np.asarray(primitive_rate_per_s, dtype=float).reshape(
        n_cells,
        5,
    )
    scales = np.asarray(primitive_scales, dtype=float).reshape(
        n_cells,
        5,
    )
    rows = np.asarray(row_scales, dtype=float).reshape(n_cells, 5)
    maximum_scaled_rate = float(np.max(np.abs(rate / scales)))
    if maximum_scaled_rate == 0.0:
        zeros = np.zeros(5 * n_cells, dtype=float)
        return zeros, np.array(zeros, copy=True)
    timestep = float(difference_step) / maximum_scaled_rate
    increment = timestep * rate
    plus_conserved = causal_five_field_mapped_conserved_from_primitives(
        context,
        charts + increment,
    )
    minus_conserved = causal_five_field_mapped_conserved_from_primitives(
        context,
        charts - increment,
    )
    plus_path = causal_five_field_path_temporal_storage_increment(
        context,
        charts,
        charts + increment,
    )
    minus_path = causal_five_field_path_temporal_storage_increment(
        context,
        charts,
        charts - increment,
    )
    denominator = 2.0 * C * timestep
    measures = np.asarray(
        context.grid.cell_measures,
        dtype=float,
    )[:, None]
    conserved = (
        measures * (plus_conserved - minus_conserved) / denominator
    ) / rows
    vertical = np.zeros((n_cells, 5), dtype=float)
    vertical[:, :4] = (
        measures
        * (
            plus_path.vertical_killing_increment
            - minus_path.vertical_killing_increment
        )
        / denominator
    )[:, :4] / rows[:, :4]
    return conserved.ravel(), vertical.ravel()


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


def test_sparse_consistent_tangent_matches_dense_solver() -> None:
    context = make_causal_five_field_regression_context(4)
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )

    dense = causal_five_field_consistent_tangent_decomposition(
        context,
        vector,
        linear_solver="dense",
    )
    sparse = causal_five_field_consistent_tangent_decomposition(
        context,
        vector,
        linear_solver="sparse",
    )

    assert dense["linear_solver"] == "dense"
    assert sparse["linear_solver"] == "sparse"
    assert sparse["consistency_nonzeros"] > 0
    assert np.allclose(
        sparse["full"]["physical_tangent_per_s"],
        dense["full"]["physical_tangent_per_s"],
        rtol=2.0e-8,
        atol=1.0e-10,
    )
    for name in dense["components"]:
        assert np.allclose(
            sparse["components"][name][
                "log_h_over_r_tangent_per_s"
            ],
            dense["components"][name][
                "log_h_over_r_tangent_per_s"
            ],
            rtol=1.0e-6,
            atol=1.0e-10,
        )


def test_reduced_descriptor_matches_direct_primitive_differences() -> None:
    context = make_causal_five_field_regression_context(2)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    reduced = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
    )
    n_reduced = 10
    step = 2.0e-6
    primitive_scale = scaling.column_scales[
        n_reduced : 2 * n_reduced
    ]
    row_scale = scaling.row_scales[:n_reduced]
    base = state.primitives.ravel()
    direct_stationary = np.empty((n_reduced, n_reduced))
    direct_be = np.empty((n_reduced, n_reduced))
    for index in range(n_reduced):
        plus = np.array(base, copy=True)
        minus = np.array(base, copy=True)
        plus[index] += step * primitive_scale[index]
        minus[index] -= step * primitive_scale[index]
        direct_stationary[:, index] = (
            causal_five_field_reduced_stationary_residual(
                plus,
                context,
            )
            - causal_five_field_reduced_stationary_residual(
                minus,
                context,
            )
        ) / (2.0 * step * row_scale)
        direct_be[:, index] = (
            causal_five_field_reduced_backward_euler_residual(
                plus,
                context,
                old_vector=vector,
                timestep_seconds=1.0,
            )
            - causal_five_field_reduced_backward_euler_residual(
                minus,
                context,
                old_vector=vector,
                timestep_seconds=1.0,
            )
        ) / (2.0 * step * row_scale)

    assert reduced["dimensions"] == (n_reduced, n_reduced)
    assert reduced["algebraic_solve_relative_defect"] < 1.0e-10
    assert (
        reduced["maximum_scaled_algebraic_reconstruction_defect"]
        < 1.0e-10
    )
    assert reduced["maximum_scaled_descriptor_algebraic_row"] < 1.0e-9
    assert np.allclose(
        reduced["stationary_reduced_scaled_jacobian"],
        direct_stationary,
        rtol=3.0e-6,
        atol=3.0e-8,
    )
    assert np.allclose(
        reduced["descriptor_reduced_scaled_matrix"],
        direct_be - direct_stationary,
        rtol=3.0e-6,
        atol=3.0e-8,
    )
    assert np.linalg.matrix_rank(
        reduced["descriptor_reduced_scaled_matrix"]
    ) == n_reduced


def test_reduced_storage_action_matches_descriptor_and_is_vector_valued() -> None:
    context = make_causal_five_field_regression_context(2)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    reduced = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
    )
    scaled_rate = np.zeros(10, dtype=float)
    scaled_rate[3::5] = (0.2, -0.15)
    physical_rate = (
        reduced["primitive_column_scales"] * scaled_rate
    )

    action = causal_five_field_reduced_storage_action(
        context,
        state.primitives.ravel(),
        physical_rate,
    )
    scaled_action = (
        action["total_conservation_storage_per_ct"].ravel()
        / reduced["conservation_row_scales"]
    )
    matrix_action = (
        reduced["descriptor_reduced_scaled_matrix"] @ scaled_rate
    )
    vertical = action["vertical_storage_per_ct"]

    assert np.allclose(
        scaled_action,
        matrix_action,
        rtol=2.0e-5,
        atol=2.0e-8,
    )
    assert np.array_equal(vertical[:, 0], np.zeros(2))
    assert np.array_equal(vertical[:, 4], np.zeros(2))
    assert np.all(np.max(np.abs(vertical[:, 1:4]), axis=0) > 0.0)


def test_independent_smooth_tangent_blocks_match_dense_small_n() -> None:
    context = make_causal_five_field_regression_context(2)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    frozen = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
    )
    stationary = causal_five_field_reduced_stationary_jacobian(
        context,
        vector,
        finite_difference_step=2.0e-6,
    )
    storage = causal_five_field_reduced_storage_matrices(
        context,
        state.primitives.ravel(),
        primitive_column_scales=frozen["primitive_column_scales"],
        conservation_row_scales=frozen["conservation_row_scales"],
        finite_difference_step=2.0e-6,
    )
    scaled_residual = (
        causal_five_field_reduced_stationary_residual(
            state.primitives.ravel(),
            context,
        )
        / frozen["conservation_row_scales"]
    )
    scaled_rate = np.linalg.solve(
        storage["descriptor_reduced_scaled_matrix"],
        -scaled_residual,
    )
    physical_rate = frozen["primitive_column_scales"] * scaled_rate
    rate_derivative = (
        causal_five_field_reduced_storage_rate_derivatives(
            context,
            state.primitives.ravel(),
            physical_rate,
            primitive_column_scales=frozen["primitive_column_scales"],
            conservation_row_scales=frozen["conservation_row_scales"],
            storage_matrix_difference_step=2.0e-6,
            storage_rate_derivative_step=1.0e-3,
        )
    )
    assembled = causal_five_field_assemble_evolving_tangent(
        storage["descriptor_reduced_scaled_matrix"],
        stationary["stationary_reduced_scaled_jacobian"],
        rate_derivative["storage_rate_derivative_scaled_matrix"],
    )
    monolithic = causal_five_field_evolving_tangent_matrices(
        context,
        vector,
        reduced_descriptor=frozen,
        storage_rate_derivative_step=1.0e-3,
    )

    assert stationary["stationary_jacobian_source"] == (
        "independent_full_dae_colored_schur"
    )
    assert storage["mass_matrix_source"] == (
        "independent_gauss_mapped_vector_storage_one_form"
    )
    assert rate_derivative["storage_rate_derivative_source"] == (
        "independent_nested_colored_mapped_plus_vertical_rate_action"
    )
    assert np.array_equal(
        stationary["stationary_reduced_scaled_jacobian"],
        frozen["stationary_reduced_scaled_jacobian"],
    )
    assert np.array_equal(
        storage["descriptor_reduced_scaled_matrix"],
        monolithic["descriptor_reduced_scaled_matrix"],
    )
    assert np.array_equal(
        storage["conserved_descriptor_reduced_scaled_matrix"],
        monolithic["conserved_descriptor_reduced_scaled_matrix"],
    )
    assert np.array_equal(
        storage["vertical_descriptor_reduced_scaled_matrix"],
        monolithic["vertical_descriptor_reduced_scaled_matrix"],
    )
    assert np.array_equal(
        rate_derivative["storage_rate_derivative_scaled_matrix"],
        monolithic["storage_rate_derivative_scaled_matrix"],
    )
    assert np.array_equal(
        assembled["evolving_reduced_scaled_jacobian"],
        monolithic["evolving_reduced_scaled_jacobian"],
    )
    assert np.array_equal(
        assembled["evolving_scaled_generator_per_s"],
        monolithic["evolving_scaled_generator_per_s"],
    )
    assert assembled["maximum_scaled_generator_factorization_defect"] < 1.0e-10

    dense_conserved, dense_vertical = (
        _independent_storage_component_matrices(
            context,
            state.primitives.ravel(),
            frozen["primitive_column_scales"],
            frozen["conservation_row_scales"],
            2.0e-6,
        )
    )
    assert np.allclose(
        storage["conserved_descriptor_reduced_scaled_matrix"],
        dense_conserved,
        rtol=3.0e-6,
        atol=3.0e-8,
    )
    assert np.allclose(
        storage["vertical_descriptor_reduced_scaled_matrix"],
        dense_vertical,
        rtol=3.0e-6,
        atol=3.0e-8,
    )


@pytest.mark.parametrize("n_cells", (2, 4))
def test_direct_action_storage_rate_derivative_matches_dense_columns(
    n_cells: int,
) -> None:
    context = make_causal_five_field_regression_context(n_cells)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    frozen = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
    )
    primitive_scales = frozen["primitive_column_scales"]
    row_scales = frozen["conservation_row_scales"]
    base = state.primitives.ravel()
    storage = causal_five_field_reduced_storage_matrices(
        context,
        base,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        finite_difference_step=2.0e-6,
    )
    scaled_residual = (
        causal_five_field_reduced_stationary_residual(base, context)
        / row_scales
    )
    scaled_rate = np.linalg.solve(
        storage["descriptor_reduced_scaled_matrix"],
        -scaled_residual,
    )
    physical_rate = primitive_scales * scaled_rate
    outer_step = 1.0e-3
    difference_step = 1.0e-4
    direct = causal_five_field_reduced_storage_rate_derivatives(
        context,
        base,
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        storage_matrix_difference_step=2.0e-6,
        storage_rate_derivative_step=outer_step,
        storage_difference_step=difference_step,
        backend="direct_action",
    )
    direct_with_unused_inner_step = (
        causal_five_field_reduced_storage_rate_derivatives(
            context,
            base,
            physical_rate,
            primitive_column_scales=primitive_scales,
            conservation_row_scales=row_scales,
            storage_matrix_difference_step=7.0e-5,
            storage_rate_derivative_step=outer_step,
            storage_difference_step=difference_step,
            backend="direct_action",
        )
    )
    direction = np.zeros(5 * n_cells, dtype=float)
    direction[3] = 1.0
    directional = (
        causal_five_field_reduced_storage_rate_directional_derivative(
            context,
            base,
            physical_rate,
            direction,
            primitive_column_scales=primitive_scales,
            conservation_row_scales=row_scales,
            storage_rate_derivative_step=outer_step,
            storage_difference_step=difference_step,
        )
    )

    n_reduced = 5 * n_cells
    dense_conserved = np.empty((n_reduced, n_reduced), dtype=float)
    dense_vertical = np.empty_like(dense_conserved)
    for column in range(n_reduced):
        plus = np.array(base, copy=True)
        minus = np.array(base, copy=True)
        plus[column] += outer_step * primitive_scales[column]
        minus[column] -= outer_step * primitive_scales[column]
        plus_conserved, plus_vertical = (
            _independent_complete_storage_rate_action(
                context,
                plus,
                physical_rate,
                primitive_scales,
                row_scales,
                difference_step,
            )
        )
        minus_conserved, minus_vertical = (
            _independent_complete_storage_rate_action(
                context,
                minus,
                physical_rate,
                primitive_scales,
                row_scales,
                difference_step,
            )
        )
        dense_conserved[:, column] = (
            plus_conserved - minus_conserved
        ) / (2.0 * outer_step)
        dense_vertical[:, column] = (
            plus_vertical - minus_vertical
        ) / (2.0 * outer_step)

    assert direct["storage_rate_derivative_backend"] == "direct_action"
    assert not direct["storage_rate_derivative_uses_inner_storage_matrix"]
    assert direct["storage_matrix_difference_step_applied"] is None
    assert direct["storage_rate_derivative_nested_component_evaluations"] == 0
    assert direct["storage_rate_derivative_nested_mapped_evaluations"] == 0
    assert direct["storage_rate_derivative_direct_action_evaluations"] == 10
    assert direct["storage_rate_derivative_direct_mapped_evaluations"] == 20
    assert direct["storage_rate_derivative_source"] == (
        "independent_outer_colored_complete_storage_rate_action"
    )
    assert directional["storage_action_evaluations"] == 2
    assert np.allclose(
        directional[
            "conserved_storage_rate_directional_derivative_scaled"
        ],
        direct["conserved_storage_rate_derivative_scaled_matrix"]
        @ direction,
        rtol=2.0e-6,
        atol=2.0e-8,
    )
    assert np.allclose(
        directional[
            "vertical_storage_rate_directional_derivative_scaled"
        ],
        direct["vertical_storage_rate_derivative_scaled_matrix"]
        @ direction,
        rtol=2.0e-6,
        atol=2.0e-8,
    )
    assert np.array_equal(
        direct["storage_rate_derivative_scaled_matrix"],
        direct_with_unused_inner_step[
            "storage_rate_derivative_scaled_matrix"
        ],
    )
    assert np.allclose(
        direct["conserved_storage_rate_derivative_scaled_matrix"],
        dense_conserved,
        rtol=2.0e-6,
        atol=2.0e-8,
    )
    assert np.allclose(
        direct["vertical_storage_rate_derivative_scaled_matrix"],
        dense_vertical,
        rtol=2.0e-6,
        atol=2.0e-8,
    )
    assert np.allclose(
        direct["storage_rate_derivative_scaled_matrix"],
        dense_conserved + dense_vertical,
        rtol=2.0e-6,
        atol=2.0e-8,
    )


def test_unified_mapped_storage_matrix_matches_dense_fourth_order() -> None:
    context = make_causal_five_field_regression_context(
        4,
        spatial_reconstruction="quadratic_admissible",
        cell_storage_quadrature="gauss_legendre_4",
    )
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    frozen = causal_five_field_reduced_descriptor_matrices(context, vector)
    primitives = state.primitives.ravel()
    primitive_scales = frozen["primitive_column_scales"]
    row_scales = frozen["conservation_row_scales"]
    step = 2.0e-3
    unified = causal_five_field_unified_mapped_storage_matrix(
        context,
        primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        mapped_storage_difference_step=step,
        mapped_storage_difference_order=4,
        mapped_storage_column_steps=np.full(primitives.size, step),
    )

    n_reduced = primitives.size
    dense = np.empty((n_reduced, n_reduced), dtype=float)
    measures = np.asarray(context.grid.cell_measures)[:, None]
    for column in range(n_reduced):
        values = []
        for multiple in (-2.0, -1.0, 1.0, 2.0):
            perturbed = np.array(primitives, copy=True)
            perturbed[column] += (
                multiple * step * primitive_scales[column]
            )
            mapped = causal_five_field_mapped_conserved_from_primitives(
                context,
                perturbed.reshape(4, 5),
            )
            values.append(
                (
                    measures
                    * np.asarray(mapped)
                    / C
                    / row_scales.reshape(4, 5)
                ).ravel()
            )
        minus_two, minus_one, plus_one, plus_two = values
        dense[:, column] = (
            minus_two
            - 8.0 * minus_one
            + 8.0 * plus_one
            - plus_two
        ) / (12.0 * step)

    np.testing.assert_allclose(
        unified["conserved_descriptor_reduced_scaled_matrix"],
        dense,
        rtol=3.0e-9,
        atol=3.0e-11,
    )
    assert unified["mapped_storage_difference_order"] == 4
    assert unified["mapped_storage_evaluations"] == (
        4 * unified["mapped_storage_component_colors"]
    )
    assert unified["base_reconstruction_admissibility_factors"].shape == (4,)


def test_unified_mapped_storage_mixed_derivative_is_symmetric() -> None:
    context = make_causal_five_field_regression_context(
        3,
        spatial_reconstruction="quadratic_admissible",
        cell_storage_quadrature="gauss_legendre_4",
    )
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    frozen = causal_five_field_reduced_descriptor_matrices(context, vector)
    primitives = state.primitives.ravel()
    primitive_scales = frozen["primitive_column_scales"]
    row_scales = frozen["conservation_row_scales"]
    scaled_rate = np.linspace(-0.2, 0.3, primitives.size)
    scaled_direction = np.linspace(0.25, -0.15, primitives.size)
    common = {
        "context": context,
        "primitive_vector": primitives,
        "primitive_column_scales": primitive_scales,
        "conservation_row_scales": row_scales,
        "mapped_storage_difference_step": 3.0e-3,
        "mapped_storage_difference_order": 4,
        "storage_rate_derivative_step": 2.0e-3,
        "storage_rate_derivative_order": 4,
    }
    rate_first = causal_five_field_unified_mapped_storage_derivatives(
        primitive_rate_per_s=primitive_scales * scaled_rate,
        **common,
    )
    direction_first = causal_five_field_unified_mapped_storage_derivatives(
        primitive_rate_per_s=primitive_scales * scaled_direction,
        **common,
    )
    left = (
        rate_first["conserved_storage_rate_derivative_scaled_matrix"]
        @ scaled_direction
    )
    right = (
        direction_first["conserved_storage_rate_derivative_scaled_matrix"]
        @ scaled_rate
    )
    scale = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        1.0,
    )
    assert np.max(np.abs(left - right)) / scale < 2.0e-5


def test_unified_vector_field_uses_shared_mapped_descriptor() -> None:
    context = make_causal_five_field_regression_context(
        4,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="cell_centered",
        cell_rate_scheme="arithmetic_face",
        cell_source_quadrature="gauss_legendre_4_local_rates",
        cell_storage_quadrature="gauss_legendre_4",
    )
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    frozen = causal_five_field_reduced_descriptor_matrices(context, vector)
    primitives = state.primitives.ravel()
    primitive_scales = frozen["primitive_column_scales"]
    row_scales = frozen["conservation_row_scales"]
    mapped_step = 3.0e-3
    vector_field = causal_five_field_scaled_primitive_vector_field(
        context,
        primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        mapped_storage_backend="unified_audit",
        mapped_storage_difference_step=mapped_step,
        mapped_storage_difference_order=4,
    )
    mapped = causal_five_field_unified_mapped_storage_matrix(
        context,
        primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        mapped_storage_difference_step=mapped_step,
        mapped_storage_difference_order=4,
    )

    assert vector_field["mapped_storage_backend"] == "unified_audit"
    assert vector_field["mass_matrix_source"] == (
        "unified_audit_mapped_plus_legacy_vertical_one_form"
    )
    assert np.array_equal(
        vector_field["conserved_descriptor_reduced_scaled_matrix"],
        mapped["conserved_descriptor_reduced_scaled_matrix"],
    )
    residual = vector_field["scaled_stationary_residual"].ravel()
    rate = vector_field["scaled_primitive_rate_per_s"].ravel()
    assert np.max(
        np.abs(
            vector_field["descriptor_reduced_scaled_matrix"] @ rate
            + residual
        )
    ) < 1.0e-10


def test_branch_frozen_mapped_storage_matches_dense_discrete_map() -> None:
    context = make_causal_five_field_regression_context(
        4,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="cell_centered",
        cell_storage_quadrature="gauss_legendre_4",
    )
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    frozen = causal_five_field_reduced_descriptor_matrices(context, vector)
    primitives = state.primitives.ravel()
    primitive_scales = frozen["primitive_column_scales"]
    row_scales = frozen["conservation_row_scales"]
    branch_frozen = causal_five_field_branch_frozen_mapped_storage_matrix(
        context,
        primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        local_difference_step=1.0e-3,
    )
    dense = causal_five_field_unified_mapped_storage_matrix(
        context,
        primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        mapped_storage_difference_step=2.0e-6,
        mapped_storage_difference_order=4,
        mapped_storage_column_steps=np.full(primitives.size, 2.0e-6),
    )

    np.testing.assert_allclose(
        branch_frozen["mapped_storage_scaled_vector"],
        dense["mapped_storage_scaled_vector"],
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        branch_frozen["conserved_descriptor_reduced_scaled_matrix"],
        dense["conserved_descriptor_reduced_scaled_matrix"],
        rtol=3.0e-7,
        atol=3.0e-8,
    )
    assert np.all(
        branch_frozen["base_reconstruction_admissibility_factors"] == 1.0
    )


def test_branch_frozen_mapped_storage_mixed_action_is_symmetric() -> None:
    context = make_causal_five_field_regression_context(
        3,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_storage_quadrature="gauss_legendre_4",
    )
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    frozen = causal_five_field_reduced_descriptor_matrices(context, vector)
    primitives = state.primitives.ravel()
    primitive_scales = frozen["primitive_column_scales"]
    row_scales = frozen["conservation_row_scales"]
    scaled_rate = np.linspace(-0.2, 0.3, primitives.size)
    scaled_direction = np.linspace(0.25, -0.15, primitives.size)
    common = {
        "context": context,
        "primitive_vector": primitives,
        "primitive_column_scales": primitive_scales,
        "conservation_row_scales": row_scales,
        "local_difference_step": 1.0e-3,
    }
    rate_first = causal_five_field_branch_frozen_mapped_storage_derivatives(
        primitive_rate_per_s=primitive_scales * scaled_rate,
        **common,
    )
    direction_first = (
        causal_five_field_branch_frozen_mapped_storage_derivatives(
            primitive_rate_per_s=primitive_scales * scaled_direction,
            **common,
        )
    )
    left = (
        rate_first["conserved_storage_rate_derivative_scaled_matrix"]
        @ scaled_direction
    )
    right = (
        direction_first["conserved_storage_rate_derivative_scaled_matrix"]
        @ scaled_rate
    )
    scale = max(np.max(np.abs(left)), np.max(np.abs(right)), 1.0)
    assert np.max(np.abs(left - right)) / scale < 3.0e-7


def test_branch_frozen_vector_field_uses_local_mapped_descriptor() -> None:
    context = make_causal_five_field_regression_context(
        4,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_rate_scheme="arithmetic_face",
        cell_source_quadrature="gauss_legendre_4_local_rates",
        cell_storage_quadrature="gauss_legendre_4",
    )
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    frozen = causal_five_field_reduced_descriptor_matrices(context, vector)
    primitives = state.primitives.ravel()
    primitive_scales = frozen["primitive_column_scales"]
    row_scales = frozen["conservation_row_scales"]
    vector_field = causal_five_field_scaled_primitive_vector_field(
        context,
        primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        mapped_storage_backend="branch_frozen_local",
        branch_frozen_local_difference_step=1.0e-3,
    )
    mapped = causal_five_field_branch_frozen_mapped_storage_matrix(
        context,
        primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        local_difference_step=1.0e-3,
    )

    assert vector_field["mapped_storage_backend"] == "branch_frozen_local"
    assert vector_field["mass_matrix_source"] == (
        "branch_frozen_local_mapped_plus_legacy_vertical_one_form"
    )
    assert np.array_equal(
        vector_field["conserved_descriptor_reduced_scaled_matrix"],
        mapped["conserved_descriptor_reduced_scaled_matrix"],
    )


def test_fourth_order_mapped_storage_action_reduces_coarse_step_error() -> None:
    context = make_causal_five_field_regression_context(2)
    state = make_causal_five_field_seed(context)
    base = state.primitives.ravel()
    rate = np.zeros_like(base)
    rate[0::5] = 0.03
    rate[1::5] = -0.02
    rate[3::5] = 0.05

    reference = causal_five_field_reduced_storage_action(
        context,
        base,
        rate,
        storage_difference_step=1.0e-3,
        conserved_difference_order=4,
    )
    second_order = causal_five_field_reduced_storage_action(
        context,
        base,
        rate,
        storage_difference_step=2.0e-2,
        conserved_difference_order=2,
    )
    fourth_order = causal_five_field_reduced_storage_action(
        context,
        base,
        rate,
        storage_difference_step=2.0e-2,
        conserved_difference_order=4,
    )
    target = np.asarray(reference["conserved_storage_per_ct"], dtype=float)
    second_error = np.linalg.norm(
        np.asarray(second_order["conserved_storage_per_ct"], dtype=float)
        - target
    )
    fourth_error = np.linalg.norm(
        np.asarray(fourth_order["conserved_storage_per_ct"], dtype=float)
        - target
    )

    assert reference["conserved_difference_order"] == 4
    assert second_order["conserved_difference_order"] == 2
    assert fourth_order["conserved_difference_order"] == 4
    assert fourth_error < 0.1 * second_error


def test_mapped_storage_action_rejects_unsupported_difference_order() -> None:
    context = make_causal_five_field_regression_context(2)
    state = make_causal_five_field_seed(context)
    with pytest.raises(ValueError, match="must be two, four, or six"):
        causal_five_field_reduced_storage_action(
            context,
            state.primitives.ravel(),
            np.ones(10),
            conserved_difference_order=3,
        )


def test_direct_action_tangent_matches_nonlinear_vector_field_jvp_n4() -> None:
    context = make_causal_five_field_regression_context(
        4,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="cell_centered",
        cell_rate_scheme="arithmetic_face",
        cell_source_quadrature="gauss_legendre_4_local_rates",
        cell_storage_quadrature="gauss_legendre_4",
    )
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    frozen = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
    )
    primitive_scales = frozen["primitive_column_scales"]
    row_scales = frozen["conservation_row_scales"]
    base = state.primitives.ravel()
    storage = causal_five_field_reduced_storage_matrices(
        context,
        base,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        finite_difference_step=2.0e-6,
    )
    scaled_residual = (
        causal_five_field_reduced_stationary_residual(base, context)
        / row_scales
    )
    scaled_rate = np.linalg.solve(
        storage["descriptor_reduced_scaled_matrix"],
        -scaled_residual,
    )
    physical_rate = primitive_scales * scaled_rate
    rate_derivative = causal_five_field_reduced_storage_rate_derivatives(
        context,
        base,
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        storage_matrix_difference_step=2.0e-6,
        storage_rate_derivative_step=1.0e-3,
        backend="direct_action",
    )
    stationary = causal_five_field_reduced_stationary_jacobian(
        context,
        vector,
        finite_difference_step=2.0e-6,
    )
    assembled = causal_five_field_assemble_evolving_tangent(
        storage["descriptor_reduced_scaled_matrix"],
        stationary["stationary_reduced_scaled_jacobian"],
        rate_derivative["storage_rate_derivative_scaled_matrix"],
    )

    direction = np.zeros((4, 5), dtype=float)
    direction[1, (0, 3)] = (0.2, 0.5)
    direction[2, (0, 3)] = (-0.4, 0.3)
    direction = direction.ravel()
    direction /= np.linalg.norm(direction)
    outer_step = 1.0e-3
    plus = causal_five_field_scaled_primitive_vector_field(
        context,
        base + outer_step * primitive_scales * direction,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        finite_difference_step=2.0e-6,
    )["scaled_primitive_rate_per_s"].ravel()
    minus = causal_five_field_scaled_primitive_vector_field(
        context,
        base - outer_step * primitive_scales * direction,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        finite_difference_step=2.0e-6,
    )["scaled_primitive_rate_per_s"].ravel()
    direct_jvp = (plus - minus) / (2.0 * outer_step)
    predicted_jvp = (
        assembled["evolving_scaled_generator_per_s"] @ direction
    )
    comparison_scale = max(
        float(np.max(np.abs(direct_jvp))),
        float(np.max(np.abs(predicted_jvp))),
        1.0,
    )

    assert (
        np.max(np.abs(direct_jvp - predicted_jvp)) / comparison_scale
        < 5.0e-3
    )


def test_independent_tangent_blocks_support_one_step_at_a_time_scans() -> None:
    context = make_causal_five_field_regression_context(2)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    frozen = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
    )
    primitive_scales = frozen["primitive_column_scales"]
    row_scales = frozen["conservation_row_scales"]
    base = state.primitives.ravel()
    storage_reference = causal_five_field_reduced_storage_matrices(
        context,
        base,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        finite_difference_step=2.0e-6,
    )
    scaled_residual = (
        causal_five_field_reduced_stationary_residual(base, context)
        / row_scales
    )
    scaled_rate = np.linalg.solve(
        storage_reference["descriptor_reduced_scaled_matrix"],
        -scaled_residual,
    )
    physical_rate = primitive_scales * scaled_rate
    stationary_reference = causal_five_field_reduced_stationary_jacobian(
        context,
        vector,
        finite_difference_step=2.0e-6,
    )
    rate_reference = causal_five_field_reduced_storage_rate_derivatives(
        context,
        base,
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        storage_matrix_difference_step=2.0e-6,
        storage_rate_derivative_step=1.0e-3,
    )

    stationary_half_step = causal_five_field_reduced_stationary_jacobian(
        context,
        vector,
        finite_difference_step=1.0e-6,
    )
    storage_half_step = causal_five_field_reduced_storage_matrices(
        context,
        base,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        finite_difference_step=1.0e-6,
    )
    outer_half_step = causal_five_field_reduced_storage_rate_derivatives(
        context,
        base,
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        storage_matrix_difference_step=2.0e-6,
        storage_rate_derivative_step=5.0e-4,
    )
    inner_half_step = causal_five_field_reduced_storage_rate_derivatives(
        context,
        base,
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        storage_matrix_difference_step=1.0e-6,
        storage_rate_derivative_step=1.0e-3,
    )

    assert stationary_half_step["finite_difference_step"] == 1.0e-6
    assert storage_half_step["finite_difference_step"] == 1.0e-6
    assert outer_half_step["storage_matrix_difference_step"] == 2.0e-6
    assert outer_half_step["storage_rate_derivative_step"] == 5.0e-4
    assert inner_half_step["storage_matrix_difference_step"] == 1.0e-6
    assert inner_half_step["storage_rate_derivative_step"] == 1.0e-3

    stationary_scan = causal_five_field_assemble_evolving_tangent(
        storage_reference["descriptor_reduced_scaled_matrix"],
        stationary_half_step["stationary_reduced_scaled_jacobian"],
        rate_reference["storage_rate_derivative_scaled_matrix"],
    )
    outer_scan = causal_five_field_assemble_evolving_tangent(
        storage_reference["descriptor_reduced_scaled_matrix"],
        stationary_reference["stationary_reduced_scaled_jacobian"],
        outer_half_step["storage_rate_derivative_scaled_matrix"],
    )
    assert np.array_equal(
        stationary_scan["descriptor_reduced_scaled_matrix"],
        storage_reference["descriptor_reduced_scaled_matrix"],
    )
    assert np.array_equal(
        stationary_scan["storage_rate_derivative_scaled_matrix"],
        rate_reference["storage_rate_derivative_scaled_matrix"],
    )
    assert np.array_equal(
        outer_scan["descriptor_reduced_scaled_matrix"],
        storage_reference["descriptor_reduced_scaled_matrix"],
    )
    assert np.array_equal(
        outer_scan["stationary_reduced_scaled_jacobian"],
        stationary_reference["stationary_reduced_scaled_jacobian"],
    )


def test_evolving_tangent_uses_five_color_local_storage_derivative() -> None:
    context = make_causal_five_field_regression_context(2)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    reduced = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
    )
    with pytest.raises(
        ValueError,
        match="storage_rate_derivative_step must be supplied",
    ):
        causal_five_field_evolving_tangent_matrices(
            context,
            vector,
            reduced_descriptor=reduced,
        )
    audit = causal_five_field_evolving_tangent_matrices(
        context,
        vector,
        reduced_descriptor=reduced,
        storage_rate_derivative_step=1.0e-3,
    )
    vector_field = causal_five_field_scaled_primitive_vector_field(
        context,
        state.primitives.ravel(),
        primitive_column_scales=audit["primitive_column_scales"],
        conservation_row_scales=audit["conservation_row_scales"],
        finite_difference_step=audit["finite_difference_step"],
    )
    base = state.primitives.ravel()
    scaled_rate = audit["scaled_primitive_rate_per_s"].ravel()
    primitive_scales = audit["primitive_column_scales"]
    row_scales = audit["conservation_row_scales"]
    inner_step = audit["finite_difference_step"]
    outer_step = audit["storage_rate_derivative_step"]
    dense_derivative = np.empty((10, 10), dtype=float)
    dense_conserved_derivative = np.empty((10, 10), dtype=float)
    dense_vertical_derivative = np.empty((10, 10), dtype=float)
    for column in range(10):
        plus = np.array(base, copy=True)
        minus = np.array(base, copy=True)
        plus[column] += outer_step * primitive_scales[column]
        minus[column] -= outer_step * primitive_scales[column]
        plus_conserved, plus_vertical = (
            _independent_storage_component_matrices(
                context,
                plus,
                primitive_scales,
                row_scales,
                inner_step,
            )
        )
        minus_conserved, minus_vertical = (
            _independent_storage_component_matrices(
                context,
                minus,
                primitive_scales,
                row_scales,
                inner_step,
            )
        )
        dense_conserved_derivative[:, column] = (
            (plus_conserved - minus_conserved) @ scaled_rate
            / (2.0 * outer_step)
        )
        dense_vertical_derivative[:, column] = (
            (plus_vertical - minus_vertical) @ scaled_rate
            / (2.0 * outer_step)
        )
    dense_derivative[:] = (
        dense_conserved_derivative + dense_vertical_derivative
    )

    colored = audit["storage_rate_derivative_scaled_matrix"]
    off_cell = np.array(dense_derivative, copy=True)
    for cell in range(2):
        local = slice(5 * cell, 5 * (cell + 1))
        off_cell[local, local] = 0.0

    assert audit["frozen_descriptor"] is reduced
    assert np.array_equal(
        vector_field["descriptor_reduced_scaled_matrix"],
        audit["descriptor_reduced_scaled_matrix"],
    )
    assert np.allclose(
        vector_field["scaled_primitive_rate_per_s"],
        audit["scaled_primitive_rate_per_s"],
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert vector_field["storage_component_paired_evaluations"] == 10
    assert audit["storage_component_colors"] == 5
    assert audit["storage_rate_derivative_component_colors"] == 5
    assert audit[
        "storage_rate_derivative_nested_component_evaluations"
    ] == 100
    assert audit[
        "storage_rate_derivative_nested_base_mapped_evaluations"
    ] == 10
    assert audit[
        "storage_rate_derivative_nested_mapped_evaluations"
    ] == 110
    assert audit[
        "vertical_storage_rate_derivative_path_evaluations"
    ] == 20
    assert audit["storage_rate_derivative_source"] == (
        "nested_colored_conserved_matrix_plus_vertical_rate_action"
    )
    assert np.max(np.abs(off_cell)) < 1.0e-8
    assert np.allclose(
        colored,
        dense_derivative,
        rtol=3.0e-3,
        atol=3.0e-6,
    )
    assert np.allclose(
        audit["conserved_storage_rate_derivative_scaled_matrix"],
        dense_conserved_derivative,
        rtol=3.0e-3,
        atol=3.0e-6,
    )
    assert np.allclose(
        audit["vertical_storage_rate_derivative_scaled_matrix"],
        dense_vertical_derivative,
        rtol=3.0e-3,
        atol=3.0e-6,
    )
    assert np.array_equal(
        audit["descriptor_reduced_scaled_matrix"],
        (
            audit["conserved_descriptor_reduced_scaled_matrix"]
            + audit["vertical_descriptor_reduced_scaled_matrix"]
        ),
    )
    assert np.array_equal(
        audit["storage_rate_derivative_scaled_matrix"],
        (
            audit[
                "conserved_storage_rate_derivative_scaled_matrix"
            ]
            + audit["vertical_storage_rate_derivative_scaled_matrix"]
        ),
    )
    assert (
        audit["maximum_scaled_generator_factorization_defect"]
        < 1.0e-10
    )
    assert audit["maximum_relative_storage_action_defect"] < 1.0e-4


def test_evolving_tangent_matches_independent_vector_field_difference() -> None:
    context = make_causal_five_field_regression_context(
        4,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="cell_centered",
        cell_rate_scheme="arithmetic_face",
        cell_source_quadrature="gauss_legendre_4_local_rates",
        cell_storage_quadrature="gauss_legendre_4",
    )
    state = make_causal_five_field_seed(context)
    smooth_charts = state.primitives
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        smooth_charts,
    )
    assert np.array_equal(
        reconstruction.admissibility_factors,
        np.ones(4),
    )
    vector = pack_causal_five_field_state(state)
    audit = causal_five_field_evolving_tangent_matrices(
        context,
        vector,
        storage_rate_derivative_step=1.0e-3,
    )
    base = state.primitives.ravel()
    primitive_scales = audit["primitive_column_scales"]
    row_scales = audit["conservation_row_scales"]
    n_reduced = base.size
    step = audit["storage_rate_derivative_step"]
    direction = np.zeros((4, 5), dtype=float)
    direction[1, (0, 3)] = (0.2, 0.5)
    direction[2, (0, 3)] = (-0.4, 0.3)
    direction = direction.ravel()
    direction /= np.linalg.norm(direction)
    for sign in (-1.0, 1.0):
        perturbed = (
            base + sign * step * primitive_scales * direction
        ).reshape(4, 5)
        assert np.array_equal(
            causal_five_field_reconstruct_face_charts(
                context,
                perturbed,
            ).admissibility_factors,
            np.ones(4),
        )

    def independently_constructed_rate(primitives: np.ndarray) -> np.ndarray:
        conserved, vertical = _independent_storage_component_matrices(
            context,
            primitives,
            primitive_scales,
            row_scales,
            audit["finite_difference_step"],
        )
        mass = conserved + vertical
        residual = (
            causal_five_field_reduced_stationary_residual(
                primitives,
                context,
            )
            / row_scales
        )
        return np.linalg.solve(mass, -residual)

    plus_rate = independently_constructed_rate(
        base + step * primitive_scales * direction
    )
    minus_rate = independently_constructed_rate(
        base - step * primitive_scales * direction
    )
    direct = (plus_rate - minus_rate) / (2.0 * step)
    predicted = (
        audit["evolving_scaled_generator_per_s"] @ direction
    )

    comparison_scale = max(
        float(np.max(np.abs(predicted))),
        float(np.max(np.abs(direct))),
        1.0,
    )
    assert (
        np.max(np.abs(predicted - direct)) / comparison_scale
        < 1.0e-2
    )


def test_direct_storage_matrix_closes_with_production_spatial_stencil() -> None:
    context = make_causal_five_field_regression_context(
        4,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_rate_scheme="arithmetic_face",
        cell_source_quadrature="gauss_legendre_4_local_rates",
        cell_storage_quadrature="gauss_legendre_4",
    )
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )

    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        make_causal_five_field_seed(context).primitives,
    )
    assert np.min(reconstruction.admissibility_factors) < 0.01
    audit = causal_five_field_evolving_tangent_matrices(
        context,
        vector,
        storage_rate_derivative_step=2.0e-6,
    )
    mass = audit["descriptor_reduced_scaled_matrix"]
    conserved_mass = audit[
        "conserved_descriptor_reduced_scaled_matrix"
    ]
    vertical_mass = audit["vertical_descriptor_reduced_scaled_matrix"]
    conserved_rate_derivative = audit[
        "conserved_storage_rate_derivative_scaled_matrix"
    ]
    vertical_rate_derivative = audit[
        "vertical_storage_rate_derivative_scaled_matrix"
    ]
    assert audit["mass_matrix_source"] == (
        "direct_gauss_mapped_vector_storage_one_form"
    )
    assert audit["maximum_relative_storage_action_defect"] < 5.0e-5
    assert audit["maximum_relative_storage_matrix_change"] < 5.0e-4
    assert audit["direct_off_cell_storage_nonzero_count"] > 0
    assert mass.shape == (20, 20)
    assert np.array_equal(mass, conserved_mass + vertical_mass)
    assert np.array_equal(
        audit["storage_rate_derivative_scaled_matrix"],
        conserved_rate_derivative + vertical_rate_derivative,
    )
    assert np.array_equal(vertical_mass[0::5], np.zeros((4, 20)))
    assert np.array_equal(vertical_mass[4::5], np.zeros((4, 20)))
    assert np.array_equal(
        vertical_rate_derivative[0::5],
        np.zeros((4, 20)),
    )
    assert np.array_equal(
        vertical_rate_derivative[4::5],
        np.zeros((4, 20)),
    )
    assert np.all(
        np.max(
            np.abs(vertical_mass.reshape(4, 5, 20)[:, 1:4]),
            axis=(0, 2),
        )
        > 0.0
    )
    assert (
        audit[
            "maximum_scaled_descriptor_component_reconstruction_defect"
        ]
        == 0.0
    )
    assert (
        audit[
            "maximum_scaled_storage_rate_component_reconstruction_defect"
        ]
        == 0.0
    )
    assert (
        audit["maximum_scaled_generator_factorization_defect"]
        < 1.0e-9
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
