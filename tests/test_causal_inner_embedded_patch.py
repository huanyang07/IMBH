from __future__ import annotations

from dataclasses import replace

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    CausalFiveFieldAdaptiveStepConfig,
    KerrSchildCellSourceRates,
    causal_embedded_patch_flux_audit,
    causal_five_field_colored_central_jacobian,
    causal_five_field_dae_jacobian_sparsity,
    causal_five_field_dae_scaling,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    evaluate_causal_five_field_increment_backward_euler,
    evolve_causal_five_field_fixed_bdf2,
    make_causal_embedded_patch_layout,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    make_kerr_schild_column_grid_from_edges,
    pack_causal_five_field_state,
    restrict_causal_embedded_patch_cell_averages,
)


def _zero_sources(n_cells: int) -> KerrSchildCellSourceRates:
    zeros = np.zeros(int(n_cells), dtype=float)
    return KerrSchildCellSourceRates(
        rest_mass=np.array(zeros, copy=True),
        radial_momentum_over_c=np.array(zeros, copy=True),
        angular_momentum_over_c=np.array(zeros, copy=True),
        killing_energy_over_c2=np.array(zeros, copy=True),
    )


def _embedded_context(
    *,
    parent_cells: int = 6,
    coupling_face: int = 3,
    refinement_ratio: int = 2,
):
    parent = make_causal_five_field_regression_context(
        parent_cells,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_rate_scheme="quadratic_log_radius",
        cell_source_quadrature="gauss_legendre_4_local_rates",
        cell_storage_quadrature="gauss_legendre_4",
    )
    layout = make_causal_embedded_patch_layout(
        parent.grid,
        coupling_face,
        refinement_ratio,
    )
    context = replace(
        parent,
        grid=layout.grid,
        stream_sources=_zero_sources(layout.n_cells),
    ).validated()
    return parent, context, layout


def test_arbitrary_edge_grid_matches_logarithmic_constructor() -> None:
    parent = make_causal_five_field_regression_context(8).grid
    rebuilt = make_kerr_schild_column_grid_from_edges(
        parent.edges,
        parent.gravitational_radius,
    )

    assert np.array_equal(rebuilt.edges, parent.edges)
    assert np.array_equal(rebuilt.centers, parent.centers)
    assert np.array_equal(rebuilt.cell_measures, parent.cell_measures)
    assert np.array_equal(rebuilt.face_measures, parent.face_measures)


def test_embedded_layout_is_nested_and_restricts_conservatively() -> None:
    parent, _context, layout = _embedded_context()
    coupling = layout.parent_coupling_face_index
    ratio = layout.refinement_ratio

    assert layout.n_cells == coupling * ratio + 6 - coupling
    assert layout.coupling_face_index == coupling * ratio
    assert layout.coupling_radius == parent.grid.edges[coupling]
    assert np.array_equal(
        layout.grid.edges[layout.coupling_face_index :],
        parent.grid.edges[coupling:],
    )
    for parent_cell in range(coupling):
        mask = layout.parent_cell_indices == parent_cell
        assert np.isclose(
            np.sum(layout.grid.cell_measures[mask]),
            parent.grid.cell_measures[parent_cell],
            rtol=3.0e-16,
            atol=0.0,
        )

    values = np.column_stack(
        (
            layout.parent_cell_indices + 1.0,
            2.0 * layout.parent_cell_indices - 3.0,
        )
    )
    restricted = restrict_causal_embedded_patch_cell_averages(
        values,
        layout,
    )
    expected = np.column_stack(
        (
            np.arange(6) + 1.0,
            2.0 * np.arange(6) - 3.0,
        )
    )
    assert np.allclose(restricted, expected, rtol=3.0e-16, atol=0.0)
    history = np.stack((values, 3.0 * values), axis=0)
    restricted_history = restrict_causal_embedded_patch_cell_averages(
        history,
        layout,
    )
    assert restricted_history.shape == (2, 6, 2)
    assert np.allclose(
        restricted_history,
        np.stack((expected, 3.0 * expected), axis=0),
        rtol=3.0e-16,
        atol=0.0,
    )


def test_ratio_one_reduces_exactly_to_parent_grid_and_operator() -> None:
    parent = make_causal_five_field_regression_context(
        6,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_rate_scheme="quadratic_log_radius",
        cell_source_quadrature="gauss_legendre_4_local_rates",
        cell_storage_quadrature="gauss_legendre_4",
    )
    layout = make_causal_embedded_patch_layout(parent.grid, 3, 1)
    context = replace(parent, grid=layout.grid).validated()
    seed = make_causal_five_field_seed(parent)
    vector = pack_causal_five_field_state(seed)

    assert np.array_equal(layout.grid.edges, parent.grid.edges)
    assert np.array_equal(layout.grid.cell_measures, parent.grid.cell_measures)
    assert np.array_equal(
        evaluate_causal_five_field_dae(vector, context).residual,
        evaluate_causal_five_field_dae(vector, parent).residual,
    )


def test_coupling_uses_one_flux_with_exact_telescoping() -> None:
    _parent, context, layout = _embedded_context()
    seed = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(seed)
    audit = causal_embedded_patch_flux_audit(context, vector, layout)

    assert audit.passed
    assert audit.maximum_state_flux_defect == 0.0
    assert audit.maximum_telescoping_defect == 0.0
    assert np.array_equal(
        audit.left_residual_contribution,
        -audit.right_residual_contribution,
    )


def test_constant_and_quadratic_charts_cross_coupling_smoothly() -> None:
    _parent, context, layout = _embedded_context()
    constant = np.zeros((layout.n_cells, 5), dtype=float)
    constant[:, 0] = np.log(1.0e3)
    constant[:, 1] = -1.0e-3
    constant[:, 2] = 1.0e-2
    constant[:, 3] = np.log(1.0e6)
    context = replace(
        context,
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.array(
            constant[-1],
            copy=True,
        ),
    ).validated()
    constant_faces = causal_five_field_reconstruct_face_charts(
        context,
        constant,
    )
    face = layout.coupling_face_index
    assert np.array_equal(
        constant_faces.left_face_charts[face],
        constant_faces.right_face_charts[face],
    )

    charts = np.array(constant, copy=True)
    coordinate = np.log(context.grid.centers / layout.coupling_radius)
    charts[:, 0] += 1.0e-4 * (coordinate + 0.2 * coordinate**2)
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        charts,
    )
    assert abs(
        reconstruction.left_face_charts[face, 0]
        - reconstruction.right_face_charts[face, 0]
    ) < 2.0e-13
    assert abs(
        reconstruction.left_face_charts[face, 0] - constant[0, 0]
    ) < 2.0e-13


def test_embedded_colored_jacobian_matches_dense_columns() -> None:
    _parent, context, layout = _embedded_context(
        parent_cells=4,
        coupling_face=2,
        refinement_ratio=2,
    )
    seed = make_causal_five_field_seed(context)
    old_vector = pack_causal_five_field_state(seed)
    stationary = evaluate_causal_five_field_dae(old_vector, context)
    scaling = causal_five_field_dae_scaling(seed, stationary)
    values = np.zeros_like(old_vector)
    step = 2.0e-6

    def residual(scaled_increment: np.ndarray) -> np.ndarray:
        return (
            evaluate_causal_five_field_increment_backward_euler(
                scaling.column_scales * scaled_increment,
                context,
                old_vector=old_vector,
                timestep_seconds=2.0e-8,
            ).residual
            / scaling.row_scales
        )

    dense = np.empty((values.size, values.size), dtype=float)
    for column in range(values.size):
        plus = np.array(values, copy=True)
        minus = np.array(values, copy=True)
        plus[column] += step
        minus[column] -= step
        dense[:, column] = (residual(plus) - residual(minus)) / (2.0 * step)
    pattern = causal_five_field_dae_jacobian_sparsity(
        layout.n_cells,
        spatial_reconstruction=context.spatial_reconstruction,
        boundary_trace_reconstruction=context.boundary_trace_reconstruction,
        cell_rate_scheme=context.cell_rate_scheme,
        cell_source_quadrature=context.cell_source_quadrature,
        cell_storage_quadrature=context.cell_storage_quadrature,
    )
    colored = causal_five_field_colored_central_jacobian(
        residual,
        values,
        pattern,
        finite_difference_step=step,
    ).toarray()
    allowed = pattern.toarray().astype(bool)
    row_scale = np.maximum(np.max(np.abs(dense), axis=1), 1.0e-14)

    assert np.max(
        np.abs(np.where(allowed, 0.0, dense)) / row_scale[:, None]
    ) < 1.0e-11
    assert np.max(np.abs(colored - dense) / row_scale[:, None]) < 1.0e-11


def test_embedded_bdf2_history_replays_bitwise() -> None:
    _parent, context, _layout = _embedded_context(
        parent_cells=4,
        coupling_face=2,
        refinement_ratio=2,
    )
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )
    config = CausalFiveFieldAdaptiveStepConfig(
        minimum_dt=1.0e-10,
        maximum_dt=1.0e-5,
        maximum_scaled_primitive_change=0.2,
        maximum_scaled_total_change=0.25,
        residual_tolerance=1.0e-10,
        algebraic_residual_tolerance=1.0e-10,
        conservation_tolerance=1.0e-9,
        maximum_newton_iterations=12,
    ).validated()
    split: dict[str, object] = {}

    def capture(completed, _total, state, history) -> None:
        if completed == 2:
            split["state"] = np.array(state, copy=True)
            split["history"] = history

    full = evolve_causal_five_field_fixed_bdf2(
        context,
        vector,
        np.zeros_like(vector),
        1.0e-8,
        4.0e-8,
        4,
        config,
        progress=capture,
    )
    history = split["history"]
    replay = evolve_causal_five_field_fixed_bdf2(
        context,
        np.asarray(split["state"], dtype=float),
        history.previous_physical_increment,
        history.previous_timestep_seconds,
        2.0e-8,
        2,
        config,
        startup_with_bdf1=False,
        initial_history=history,
    )

    assert full.passed
    assert replay.passed
    np.testing.assert_array_equal(replay.state_vector, full.state_vector)
    np.testing.assert_array_equal(
        replay.history.previous_physical_increment,
        full.history.previous_physical_increment,
    )
    np.testing.assert_array_equal(
        replay.history.previous_vertical_killing_increment,
        full.history.previous_vertical_killing_increment,
    )
