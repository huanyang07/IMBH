from __future__ import annotations

from dataclasses import replace

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_analytic_local_maps,
    causal_five_field_coordinate_principal_components,
    causal_five_field_frozen_analytic_tangent,
    causal_five_field_radial_analytic_tangent,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (
    _cell_state,
    _local_cell_source_density,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_fluctuation import (
    _explicit_geometry_rates,
)


def _context_and_primitives(n_cells: int = 5):
    context = make_causal_five_field_regression_context(n_cells)
    primitives = make_causal_five_field_seed(context).primitives
    context = replace(
        context,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.array(
            primitives[-1],
            copy=True,
        ),
    ).validated()
    return context, primitives


def test_analytic_local_maps_match_independent_physical_values() -> None:
    context, primitives = _context_and_primitives()
    cell = 2
    radius = float(context.grid.centers[cell])
    chart = primitives[cell]
    analytic = causal_five_field_analytic_local_maps(
        context,
        radius,
        chart,
    )
    state = _cell_state(context, radius, chart)
    np.testing.assert_allclose(
        analytic.mapped_conserved,
        state.conserved,
        rtol=2.0e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        analytic.physical_flux_over_c,
        state.flux_over_c,
        rtol=2.0e-14,
        atol=0.0,
    )

    principal = causal_five_field_coordinate_principal_components(
        context,
        radius,
        chart,
    )
    # The legacy five-point temporal derivative is roundoff-limited on this
    # small inner regression grid: decreasing its relative step below 8e-4
    # increases the discrepancy.  Compare against the best point of that
    # independent step sweep with a scale-invariant matrix norm.
    temporal_reference = causal_five_field_coordinate_principal_components(
        context,
        radius,
        chart,
        relative_step=8.0e-4,
    )
    temporal_relative_defect = np.linalg.norm(
        analytic.temporal_storage_matrix
        - temporal_reference.temporal_storage_matrix
    ) / np.linalg.norm(analytic.temporal_storage_matrix)
    assert temporal_relative_defect <= 2.0e-7
    np.testing.assert_allclose(
        analytic.shear_principal_source_matrix,
        principal.shear_principal_source_matrix,
        rtol=2.0e-9,
        # The independent five-point reference has an absolute roundoff
        # floor in analytically zero columns on this uniform regression seed.
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        analytic.vertical_principal_source_matrix,
        principal.vertical_principal_source_matrix,
        rtol=2.0e-9,
        atol=5.0e-2,
    )

    shear_rate, height_rate = _explicit_geometry_rates(
        context,
        radius,
        chart,
    )
    _total, _depth, components = _local_cell_source_density(
        context,
        state,
        shear_rate=shear_rate,
        height_rate=height_rate,
    )
    for name, values in analytic.lower_source_values.items():
        np.testing.assert_allclose(
            values,
            components[name],
            rtol=5.0e-9,
            atol=5.0e-20,
        )


def test_radial_analytic_tangent_is_additive_and_homogeneous() -> None:
    context, primitives = _context_and_primitives()
    dimensions = int(primitives.size)
    tangent = causal_five_field_radial_analytic_tangent(
        context,
        primitives,
        primitive_column_scales=np.ones(dimensions, dtype=float),
        conservation_row_scales=np.ones(dimensions, dtype=float),
    )
    generator = np.random.default_rng(4123)
    left = generator.normal(size=dimensions)
    right = generator.normal(size=dimensions)
    scale = -0.371
    np.testing.assert_allclose(
        tangent.apply(left + right),
        tangent.apply(left) + tangent.apply(right),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        tangent.apply(scale * left),
        scale * tangent.apply(left),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert tangent.maximum_base_reconstruction_relative_defect <= 1.0e-14
    assert tangent.maximum_projector_closure_defect <= 1.0e-8
    assert tangent.maximum_block_ledger_relative_defect == 0.0
    assert tangent.characteristic_subspaces_frozen is True
    assert tangent.principal_matrix_derivatives_included is True
    assert tangent.characteristic_face_speeds_over_c.shape == (6, 5)
    assert tangent.characteristic_face_radii.shape == (6,)
    assert tangent.minimum_characteristic_spectral_gap > 0.0
    assert np.isfinite(
        tangent.maximum_characteristic_descriptor_condition_number
    )


def test_internal_geometry_step_has_a_declared_stable_tangent() -> None:
    context, primitives = _context_and_primitives()
    dimensions = int(primitives.size)
    tangents = [
        causal_five_field_radial_analytic_tangent(
            context,
            primitives,
            primitive_column_scales=np.ones(dimensions, dtype=float),
            conservation_row_scales=np.ones(dimensions, dtype=float),
            explicit_geometry_log_radius_step=step,
        )
        for step in (1.0e-5, 2.0e-5, 4.0e-5)
    ]
    direction = np.random.default_rng(1219).normal(size=dimensions)
    reference = tangents[1].apply(direction)
    for tangent in tangents:
        relative = np.linalg.norm(
            tangent.apply(direction) - reference
        ) / max(
            np.linalg.norm(reference),
            np.finfo(float).tiny,
        )
        assert relative <= 2.0e-9


def test_frozen_analytic_tangent_uses_one_dae_identity() -> None:
    context, primitives = _context_and_primitives()
    dimensions = int(primitives.size)
    tangent = causal_five_field_radial_analytic_tangent(
        context,
        primitives,
        primitive_column_scales=np.ones(dimensions, dtype=float),
        conservation_row_scales=np.ones(dimensions, dtype=float),
    )
    generator = np.random.default_rng(716)
    production = generator.normal(scale=1.0e-3, size=(dimensions, dimensions))
    descriptor = np.eye(dimensions)
    anchor = generator.normal(scale=1.0e-4, size=(dimensions, dimensions))
    frozen = causal_five_field_frozen_analytic_tangent(
        tangent,
        production,
        descriptor,
        anchor,
    )
    np.testing.assert_allclose(
        descriptor @ frozen.production_scaled_generator_per_s
        + frozen.production_stationary_scaled_jacobian
        + anchor,
        0.0,
        rtol=0.0,
        atol=2.0e-18,
    )
    np.testing.assert_allclose(
        frozen.candidate_scaled_generator_per_s,
        production - frozen.descriptor_solve_scaled_correction,
        rtol=0.0,
        atol=0.0,
    )
    assert frozen.maximum_production_identity_relative_defect <= 1.0e-14
    assert frozen.maximum_descriptor_solve_relative_defect <= 1.0e-14
