from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_coordinate_principal_basis,
    causal_five_field_coordinate_principal_components,
    causal_five_field_frozen_principal_generator,
    causal_five_field_lower_stress_relaxation_matrix,
    causal_five_field_manufactured_principal_wave,
    causal_five_field_principal_step_defects,
    causal_quadratic_reconstruction_matrices,
    causal_five_field_shear_fourier_symbols,
    causal_five_field_shear_invariant_subspace,
    causal_five_field_straight_principal_path_jump,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
)


def _inputs():
    context = make_causal_five_field_regression_context(8)
    primitives = make_causal_five_field_seed(context).primitives
    cell = 2
    return (
        context,
        float(context.grid.centers[cell]),
        primitives[cell],
        float(np.diff(context.grid.edges)[cell]),
    )


def test_coordinate_principal_components_use_implemented_source_sign() -> None:
    context, radius, chart, _ = _inputs()
    components = causal_five_field_coordinate_principal_components(
        context,
        radius,
        chart,
    )
    basis = causal_five_field_coordinate_principal_basis(
        context,
        radius,
        chart,
    )

    np.testing.assert_array_equal(
        components.temporal_storage_matrix,
        components.mapped_storage_matrix
        + components.vertical_storage_matrix,
    )
    np.testing.assert_array_equal(
        components.principal_source_matrix,
        components.shear_principal_source_matrix
        + components.vertical_principal_source_matrix,
    )
    np.testing.assert_array_equal(
        components.spatial_principal_matrix,
        components.physical_flux_matrix
        - components.principal_source_matrix,
    )
    np.testing.assert_array_equal(
        basis.temporal_storage_matrix,
        components.temporal_storage_matrix,
    )
    np.testing.assert_array_equal(
        basis.spatial_principal_matrix,
        components.spatial_principal_matrix,
    )
    assert np.count_nonzero(
        components.shear_principal_source_matrix[:4]
    ) == 0
    assert np.count_nonzero(
        components.vertical_principal_source_matrix[4]
    ) == 0


def test_straight_path_jump_linearizes_to_complete_spatial_matrix() -> None:
    context, radius, chart, _ = _inputs()
    components = causal_five_field_coordinate_principal_components(
        context,
        radius,
        chart,
    )
    direction = np.asarray([0.3, -0.2, 0.1, 0.4, -0.2])
    direction *= (
        components.primitive_column_scales / np.linalg.norm(direction)
    )
    epsilon = 1.0e-6
    left = chart - 0.5 * epsilon * direction
    right = chart + 0.5 * epsilon * direction
    forward = causal_five_field_straight_principal_path_jump(
        context,
        radius,
        left,
        right,
    )
    reverse = causal_five_field_straight_principal_path_jump(
        context,
        radius,
        right,
        left,
    )
    linearized = (
        epsilon * components.spatial_principal_matrix @ direction
    )
    scale = max(
        float(np.max(np.abs(forward))),
        float(np.max(np.abs(linearized))),
        np.finfo(float).tiny,
    )

    assert float(np.max(np.abs(forward - linearized)) / scale) <= 2.0e-8
    np.testing.assert_array_equal(reverse, -forward)


def test_shear_subspace_has_positive_physical_energy() -> None:
    context, radius, chart, _ = _inputs()
    shear = causal_five_field_shear_invariant_subspace(
        context,
        radius,
        chart,
    )

    assert shear.maximum_projector_idempotence_defect <= 1.0e-10
    assert shear.maximum_projector_complement_defect <= 2.0e-10
    assert shear.maximum_local_rest_symmetry_defect <= 1.0e-12
    assert shear.maximum_analytic_local_projector_defect <= 1.0e-12
    assert shear.maximum_analytic_local_eigenpair_defect <= 1.0e-12
    assert shear.minimum_local_rest_energy_eigenvalue > 0.0
    assert shear.minimum_coordinate_energy_eigenvalue > 0.0
    assert np.isfinite(shear.coordinate_energy_condition_number)
    np.testing.assert_allclose(
        shear.primitive_left_eigenvectors
        @ shear.primitive_right_eigenvectors,
        np.eye(2),
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.sum(shear.analytic_local_rest_projectors, axis=0),
        np.eye(2),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_fourier_symbols_converge_to_the_complete_continuum_symbol() -> None:
    context, radius, chart, spacing = _inputs()
    components = causal_five_field_coordinate_principal_components(
        context,
        radius,
        chart,
    )
    basis = causal_five_field_coordinate_principal_basis(
        context,
        radius,
        chart,
    )
    lower = causal_five_field_lower_stress_relaxation_matrix(
        context,
        radius,
        chart,
    )
    coarse = causal_five_field_shear_fourier_symbols(
        components,
        basis,
        lower,
        theta=2.0e-2,
        spacing=spacing,
    )
    fine = causal_five_field_shear_fourier_symbols(
        components,
        basis,
        lower,
        theta=1.0e-2,
        spacing=0.5 * spacing,
    )
    coarse_error = np.linalg.norm(
        coarse.monolithic_centered_principal_per_s
        - coarse.continuum_principal_per_s
    )
    fine_error = np.linalg.norm(
        fine.monolithic_centered_principal_per_s
        - fine.continuum_principal_per_s
    )

    assert coarse.wavenumber == fine.wavenumber
    assert fine_error < 0.26 * coarse_error
    assert np.all(
        np.isfinite(fine.current_split_relaxing_per_s)
    )
    assert np.all(
        np.isfinite(fine.monolithic_relaxing_per_s)
    )


def test_principal_derivative_step_sweep_has_a_stable_plateau() -> None:
    context, radius, chart, _ = _inputs()
    components = tuple(
        causal_five_field_coordinate_principal_components(
            context,
            radius,
            chart,
            relative_step=relative_step,
        )
        for relative_step in (5.0e-5, 1.0e-4, 2.0e-4, 4.0e-4)
    )
    defects = causal_five_field_principal_step_defects(components)

    assert max(defects.values()) <= 2.0e-5
    assert defects["spatial_principal_matrix"] <= 2.0e-6


def test_quadratic_face_maps_reproduce_linear_data() -> None:
    centers = np.linspace(-0.9, 0.9, 7)
    edges = np.linspace(-1.05, 1.05, 8)
    left, right = causal_quadratic_reconstruction_matrices(
        centers,
        edges,
    )
    values = 0.7 - 0.3 * centers
    expected = 0.7 - 0.3 * edges

    np.testing.assert_allclose(left[1:-1] @ values, expected[1:-1])
    np.testing.assert_allclose(right[1:-1] @ values, expected[1:-1])
    np.testing.assert_allclose(left.sum(axis=1), np.ones(edges.size))
    np.testing.assert_allclose(right.sum(axis=1), np.ones(edges.size))


def test_manufactured_wave_and_frozen_generators_are_independent() -> None:
    audits = []
    for n_cells in (16, 32):
        context = make_causal_five_field_regression_context(n_cells)
        primitives = make_causal_five_field_seed(context).primitives
        rg = context.grid.gravitational_radius
        audit = causal_five_field_manufactured_principal_wave(
            context,
            primitives,
            np.asarray([0.3, -0.2, 0.1, 0.4, -0.2]),
            support_inner_radius=2.0 * rg,
            support_outer_radius=12.0 * rg,
        )
        audits.append(audit)
    assert (
        audits[1].current_split_relative_l2_error
        < 0.25 * audits[0].current_split_relative_l2_error
    )
    assert (
        audits[1].monolithic_relative_l2_error
        < 0.25 * audits[0].monolithic_relative_l2_error
    )

    context, _radius, primitives, _spacing = _inputs()
    charts = make_causal_five_field_seed(context).primitives
    amplitudes = np.maximum(np.abs(charts), 1.0e-3)
    amplitudes[:, 4] = np.maximum(np.abs(charts[:, 4]), 1.0e-12)
    split = causal_five_field_frozen_principal_generator(
        context,
        charts,
        amplitudes,
        operator="current_split",
    )
    monolithic = causal_five_field_frozen_principal_generator(
        context,
        charts,
        amplitudes,
        operator="monolithic",
    )

    assert split.shape == monolithic.shape == (40, 40)
    assert np.all(np.isfinite(split))
    assert np.all(np.isfinite(monolithic))
    assert not np.array_equal(split, monolithic)
