from __future__ import annotations

from dataclasses import replace

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_dae_scaling,
    causal_five_field_monolithic_bdf_history_direction,
    causal_five_field_monolithic_discrete_step_matrix,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)


def _problem(n_cells: int = 5):
    context = make_causal_five_field_regression_context(n_cells)
    base = np.asarray(
        make_causal_five_field_seed(context).primitives,
        dtype=float,
    )
    context = replace(
        context,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_storage_quadrature="gauss_legendre_4",
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.array(base[-1], copy=True),
    ).validated()
    state = causal_five_field_state_from_primitives(context, base)
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    scaling = causal_five_field_dae_scaling(state, evaluation)
    dimensions = int(base.size)
    columns = np.asarray(
        scaling.column_scales[dimensions : 2 * dimensions],
        dtype=float,
    ).reshape(base.shape)
    rows = np.asarray(
        scaling.row_scales[:dimensions],
        dtype=float,
    ).reshape(base.shape)
    return context, base, columns, rows


def _directions(base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phase = np.sin(np.linspace(0.0, np.pi, base.shape[0]))[:, None]
    scale = np.maximum(np.max(np.abs(base), axis=0), 1.0e-12)[None, :]
    left = 1.0e-6 * phase * np.asarray(
        [1.0, -0.5, 0.25, -0.125, 0.5],
    )[None, :] * scale
    right = 1.0e-6 * phase * np.asarray(
        [-0.25, 0.5, 1.0, -0.5, 0.125],
    )[None, :] * scale
    return np.asarray([left, right]), np.asarray([0.8 * left, 1.1 * right])


def test_analytic_initial_history_direction_matches_centered_difference() -> None:
    context, base, columns, rows = _problem()
    increment = 2.0e-7 * np.sin(
        np.linspace(0.0, np.pi, base.shape[0]),
    )[:, None] * columns
    new = base + increment
    old_directions, new_directions = _directions(base)
    matrix = causal_five_field_monolithic_discrete_step_matrix(
        context,
        base,
        new,
        1.0e-5,
        1.0e-5,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )
    analytic = causal_five_field_monolithic_bdf_history_direction(
        context,
        base,
        new,
        old_directions,
        new_directions,
        analytic_step_matrix=matrix,
    )
    reference = causal_five_field_monolithic_bdf_history_direction(
        context,
        base,
        new,
        old_directions,
        new_directions,
        directional_step=8.0e-2,
    )
    np.testing.assert_array_equal(
        analytic.previous_primitive_increment,
        reference.previous_primitive_increment,
    )
    for actual, expected in (
        (
            analytic.previous_mapped_storage_increment,
            reference.previous_mapped_storage_increment,
        ),
        (
            analytic.previous_responsive_height_storage_increment,
            reference.previous_responsive_height_storage_increment,
        ),
    ):
        relative = np.linalg.norm(actual - expected) / max(
            np.linalg.norm(actual),
            np.finfo(float).tiny,
        )
        assert relative <= 2.0e-8
