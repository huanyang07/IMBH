from __future__ import annotations

from dataclasses import replace

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_analytic_local_maps,
    causal_five_field_monolithic_frozen_tangent,
    causal_five_field_monolithic_storage_rate_action,
    evaluate_causal_five_field_monolithic_backward_euler,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
)


def _context_primitives_and_scales():
    context = make_causal_five_field_regression_context(5)
    primitives = np.asarray(
        make_causal_five_field_seed(context).primitives,
        dtype=float,
    )
    context = replace(
        context,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_storage_quadrature="gauss_legendre_4",
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.array(
            primitives[-1],
            copy=True,
        ),
    ).validated()
    primitive_scales = np.maximum(np.abs(primitives), 1.0).ravel()
    row_scales = []
    for cell, radius in enumerate(context.grid.centers):
        local = causal_five_field_analytic_local_maps(
            context,
            float(radius),
            primitives[cell],
        )
        row_scales.append(
            np.maximum(
                np.abs(
                    context.grid.cell_measures[cell]
                    * local.mapped_conserved
                    / C
                ),
                1.0,
            )
        )
    return (
        context,
        primitives,
        primitive_scales,
        np.asarray(row_scales, dtype=float).ravel(),
    )


def _relative(first: np.ndarray, second: np.ndarray) -> float:
    scale = max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(first - second) / scale)


def test_monolithic_frozen_tangent_is_self_consistent() -> None:
    context, primitives, columns, rows = (
        _context_primitives_and_scales()
    )
    tangent = causal_five_field_monolithic_frozen_tangent(
        context,
        primitives,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )

    assert tangent.uses_center_broken_within_cell_paths
    assert tangent.spatial_tangent.center_broken_within_cell_paths
    assert not tangent.uses_production_generator
    assert not tangent.uses_production_anchor_storage_derivative
    assert tangent.incoming_excision_characteristics == 0
    assert tangent.maximum_node_reconstruction_relative_defect <= 1.0e-12
    assert tangent.maximum_node_partition_of_unity_defect <= 1.0e-12
    assert tangent.maximum_descriptor_component_defect <= 1.0e-12
    assert tangent.maximum_storage_rate_component_defect <= 1.0e-12
    assert tangent.maximum_base_rate_balance_defect <= 1.0e-12
    assert tangent.maximum_generator_factorization_defect <= 1.0e-12
    assert (
        tangent.maximum_centered_storage_action_relative_defect
        <= 1.0e-7
    )

    phase = np.sin(np.linspace(0.0, np.pi, primitives.shape[0]))
    direction = np.zeros_like(primitives)
    direction[:, 0] = 0.3 * phase
    direction[:, 3] = -0.2 * phase
    direction = direction.ravel()
    direction /= np.linalg.norm(direction)
    physical_direction = (columns * direction).reshape(primitives.shape)

    step = 1.0e-5
    plus_action = causal_five_field_monolithic_storage_rate_action(
        context,
        primitives + step * physical_direction,
        tangent.physical_base_rate_per_s.reshape(primitives.shape),
        conservation_row_scales=rows,
    )
    minus_action = causal_five_field_monolithic_storage_rate_action(
        context,
        primitives - step * physical_direction,
        tangent.physical_base_rate_per_s.reshape(primitives.shape),
        conservation_row_scales=rows,
    )
    direct_storage_rate = (plus_action - minus_action) / (2.0 * step)
    analytic_storage_rate = (
        tangent.storage_rate_derivative_scaled_matrix @ direction
    )
    assert _relative(direct_storage_rate, analytic_storage_rate) <= 1.0e-7

    def stationary(offset: float) -> np.ndarray:
        charts = primitives + offset * physical_direction
        evaluation = (
            evaluate_causal_five_field_monolithic_backward_euler(
                charts,
                charts,
                1.0,
                context,
            )
        )
        return evaluation.residual_rows.ravel() / rows

    direct_stationary = (
        -stationary(2.0 * step)
        + 8.0 * stationary(step)
        - 8.0 * stationary(-step)
        + stationary(-2.0 * step)
    ) / (12.0 * step)
    analytic_stationary = tangent.stationary_scaled_jacobian @ direction
    assert _relative(direct_stationary, analytic_stationary) <= 2.0e-6
