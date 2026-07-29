"""Self-consistent frozen tangent of the monolithic descriptor-path DAE.

This module combines three pieces derived from the same declared nonlinear
architecture:

* the reconstructed mapped-plus-responsive-height temporal descriptor;
* the center-broken complete stationary residual and its analytic frozen-
  subspace tangent;
* the derivative of the temporal descriptor acting on the candidate base
  rate obtained from that same residual.

No production generator or production-anchor storage derivative enters the
construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C

from .causal_inner_dae_system import CausalFiveFieldDAEContext
from .causal_inner_monolithic_dae import (
    _node_charts,
    _spatial_nodes,
    causal_five_field_monolithic_storage_increment,
    evaluate_causal_five_field_monolithic_backward_euler,
)
from .causal_inner_radial_linear_tangent import (
    CausalFiveFieldRadialAnalyticTangent,
    causal_five_field_analytic_local_maps,
    causal_five_field_radial_analytic_tangent,
)


_N_FIELDS = 5


@dataclass(frozen=True)
class CausalFiveFieldMonolithicFrozenTangent:
    """One self-consistent continuous tangent at a frozen physical state."""

    base_primitives: np.ndarray
    primitive_column_scales: np.ndarray
    conservation_row_scales: np.ndarray
    node_reconstruction_weights: np.ndarray
    mapped_descriptor_scaled_matrix: np.ndarray
    responsive_height_descriptor_scaled_matrix: np.ndarray
    descriptor_scaled_matrix: np.ndarray
    base_stationary_scaled_residual: np.ndarray
    scaled_base_rate_per_s: np.ndarray
    physical_base_rate_per_s: np.ndarray
    mapped_storage_rate_derivative_scaled_matrix: np.ndarray
    responsive_height_storage_rate_derivative_scaled_matrix: np.ndarray
    storage_rate_derivative_scaled_matrix: np.ndarray
    stationary_scaled_jacobian: np.ndarray
    evolving_scaled_jacobian: np.ndarray
    scaled_generator_per_s: np.ndarray
    spatial_tangent: CausalFiveFieldRadialAnalyticTangent
    maximum_node_reconstruction_relative_defect: float
    maximum_node_partition_of_unity_defect: float
    maximum_descriptor_component_defect: float
    maximum_storage_rate_component_defect: float
    maximum_base_rate_balance_defect: float
    maximum_generator_factorization_defect: float
    maximum_centered_storage_action_relative_defect: float
    centered_storage_action_scaled_step: float
    incoming_excision_characteristics: int
    uses_center_broken_within_cell_paths: bool
    uses_production_generator: bool
    uses_production_anchor_storage_derivative: bool

    def apply(self, scaled_direction: np.ndarray) -> np.ndarray:
        """Apply the continuous generator in fixed scaled coordinates."""

        direction = np.asarray(scaled_direction, dtype=float).ravel()
        if (
            direction.shape != self.scaled_base_rate_per_s.shape
            or np.any(~np.isfinite(direction))
        ):
            raise ValueError("monolithic tangent direction is invalid")
        return np.asarray(self.scaled_generator_per_s @ direction)


def _lagrange_derivative_weights(
    nodes: np.ndarray,
    target: float,
) -> np.ndarray:
    """Differentiate the Lagrange interpolant at one target."""

    coordinates = np.asarray(nodes, dtype=float)
    count = int(coordinates.size)
    result = np.zeros(count, dtype=float)
    for basis in range(count):
        for differentiated in range(count):
            if differentiated == basis:
                continue
            term = 1.0 / (
                coordinates[basis] - coordinates[differentiated]
            )
            for other in range(count):
                if other in (basis, differentiated):
                    continue
                term *= (
                    target - coordinates[other]
                ) / (
                    coordinates[basis] - coordinates[other]
                )
            result[basis] += term
    return result


def _node_reconstruction_weights(
    context: CausalFiveFieldDAEContext,
    base_primitives: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """Return exact affine cell-node reconstruction weights."""

    context = context.validated()
    base = np.asarray(base_primitives, dtype=float)
    n_cells = int(context.grid.centers.size)
    if (
        base.shape != (n_cells, _N_FIELDS)
        or np.any(~np.isfinite(base))
        or context.spatial_reconstruction
        not in {
            "piecewise_constant",
            "plm_unlimited",
            "quadratic_admissible",
        }
    ):
        raise ValueError(
            "monolithic tangent requires a certified affine reconstruction"
        )
    nodes = _spatial_nodes(context)
    sampled, factors = _node_charts(context, base, nodes)
    if not np.array_equal(factors, np.ones_like(factors)):
        raise ValueError(
            "monolithic tangent requires inactive admissibility scaling"
        )

    slope_weights = np.zeros((n_cells, n_cells), dtype=float)
    if context.spatial_reconstruction != "piecewise_constant":
        log_centers = np.log(np.asarray(context.grid.centers, dtype=float))
        for cell in range(1, n_cells - 1):
            left_spacing = log_centers[cell] - log_centers[cell - 1]
            right_spacing = log_centers[cell + 1] - log_centers[cell]
            denominator = left_spacing + right_spacing
            slope_weights[cell, cell - 1] = (
                -right_spacing / left_spacing / denominator
            )
            slope_weights[cell, cell] = (
                right_spacing / left_spacing
                - left_spacing / right_spacing
            ) / denominator
            slope_weights[cell, cell + 1] = (
                left_spacing / right_spacing / denominator
            )
        if (
            context.boundary_trace_reconstruction == "plm_one_sided"
            and n_cells >= 3
        ):
            slope_weights[0, :3] = _lagrange_derivative_weights(
                log_centers[:3],
                float(log_centers[0]),
            )
            slope_weights[-1, -3:] = _lagrange_derivative_weights(
                log_centers[-3:],
                float(log_centers[-1]),
            )

    log_centers = np.log(np.asarray(context.grid.centers, dtype=float))
    weights = []
    cell_indices = []
    radii = []
    measures = []
    for cell, cell_nodes in enumerate(nodes):
        for radius, measure in cell_nodes:
            row = np.zeros(n_cells, dtype=float)
            row[cell] = 1.0
            row += (
                np.log(float(radius)) - log_centers[cell]
            ) * slope_weights[cell]
            weights.append(row)
            cell_indices.append(cell)
            radii.append(float(radius))
            measures.append(float(measure))
    matrix = np.asarray(weights, dtype=float)
    predicted = matrix @ base
    reconstruction_scale = max(
        float(np.max(np.abs(sampled))),
        float(np.max(np.abs(predicted))),
        np.finfo(float).tiny,
    )
    reconstruction_defect = float(
        np.max(np.abs(predicted - sampled)) / reconstruction_scale
    )
    partition_defect = float(
        np.max(np.abs(np.sum(matrix, axis=1) - 1.0))
    )
    return (
        matrix,
        np.asarray(cell_indices, dtype=int),
        np.asarray(radii, dtype=float),
        np.asarray(measures, dtype=float),
        reconstruction_defect,
        partition_defect,
    )


def _descriptor_matrices(
    context: CausalFiveFieldDAEContext,
    base_primitives: np.ndarray,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    node_weights: np.ndarray,
    node_cells: np.ndarray,
    node_radii: np.ndarray,
    node_measures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells = int(context.grid.centers.size)
    dimensions = _N_FIELDS * n_cells
    base = np.asarray(base_primitives, dtype=float)
    columns = np.asarray(primitive_column_scales, dtype=float).reshape(
        n_cells,
        _N_FIELDS,
    )
    rows = np.asarray(conservation_row_scales, dtype=float).reshape(
        n_cells,
        _N_FIELDS,
    )
    mapped = np.zeros((dimensions, dimensions), dtype=float)
    height = np.zeros_like(mapped)
    for weights, cell, radius, measure in zip(
        node_weights,
        node_cells,
        node_radii,
        node_measures,
        strict=True,
    ):
        local = causal_five_field_analytic_local_maps(
            context,
            float(radius),
            np.asarray(weights @ base, dtype=float),
        )
        row_slice = slice(
            _N_FIELDS * int(cell),
            _N_FIELDS * (int(cell) + 1),
        )
        row_factor = float(measure) / C / rows[int(cell)]
        for source_cell in np.flatnonzero(weights):
            coefficient = float(weights[source_cell])
            for component in range(_N_FIELDS):
                column = _N_FIELDS * int(source_cell) + component
                column_factor = (
                    coefficient * columns[source_cell, component]
                )
                mapped[row_slice, column] += (
                    row_factor
                    * local.mapped_conserved_jacobian[:, component]
                    * column_factor
                )
                height[row_slice, column] += (
                    row_factor
                    * local.vertical_storage_matrix[:, component]
                    * column_factor
                )
    return mapped, height


def causal_five_field_monolithic_storage_rate_action(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    physical_rate_per_s: np.ndarray,
    *,
    conservation_row_scales: np.ndarray,
) -> np.ndarray:
    """Apply the reconstructed temporal one-form to one fixed rate."""

    context = context.validated()
    charts = np.asarray(primitive_charts, dtype=float)
    rate = np.asarray(physical_rate_per_s, dtype=float)
    n_cells = int(context.grid.centers.size)
    rows = np.asarray(conservation_row_scales, dtype=float).reshape(
        n_cells,
        _N_FIELDS,
    )
    if (
        charts.shape != (n_cells, _N_FIELDS)
        or rate.shape != charts.shape
        or np.any(~np.isfinite(charts))
        or np.any(~np.isfinite(rate))
    ):
        raise ValueError("monolithic storage-rate action inputs are invalid")
    (
        weights,
        node_cells,
        node_radii,
        node_measures,
        _reconstruction_defect,
        _partition_defect,
    ) = _node_reconstruction_weights(context, charts)
    result = np.zeros_like(charts)
    for node_weight, cell, radius, measure in zip(
        weights,
        node_cells,
        node_radii,
        node_measures,
        strict=True,
    ):
        local = causal_five_field_analytic_local_maps(
            context,
            float(radius),
            np.asarray(node_weight @ charts, dtype=float),
        )
        result[int(cell)] += (
            float(measure)
            / C
            / rows[int(cell)]
            * (
                local.temporal_storage_matrix
                @ np.asarray(node_weight @ rate, dtype=float)
            )
        )
    return result.ravel()


def _storage_rate_derivative_matrices(
    context: CausalFiveFieldDAEContext,
    base_primitives: np.ndarray,
    physical_rate_per_s: np.ndarray,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    node_weights: np.ndarray,
    node_cells: np.ndarray,
    node_radii: np.ndarray,
    node_measures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells = int(context.grid.centers.size)
    dimensions = _N_FIELDS * n_cells
    base = np.asarray(base_primitives, dtype=float)
    rate = np.asarray(physical_rate_per_s, dtype=float)
    columns = np.asarray(primitive_column_scales, dtype=float).reshape(
        n_cells,
        _N_FIELDS,
    )
    rows = np.asarray(conservation_row_scales, dtype=float).reshape(
        n_cells,
        _N_FIELDS,
    )
    mapped = np.zeros((dimensions, dimensions), dtype=float)
    height = np.zeros_like(mapped)
    for weights, cell, radius, measure in zip(
        node_weights,
        node_cells,
        node_radii,
        node_measures,
        strict=True,
    ):
        local = causal_five_field_analytic_local_maps(
            context,
            float(radius),
            np.asarray(weights @ base, dtype=float),
        )
        node_rate = np.asarray(weights @ rate, dtype=float)
        row_slice = slice(
            _N_FIELDS * int(cell),
            _N_FIELDS * (int(cell) + 1),
        )
        row_factor = float(measure) / C / rows[int(cell)]
        for source_cell in np.flatnonzero(weights):
            coefficient = float(weights[source_cell])
            for component in range(_N_FIELDS):
                column = _N_FIELDS * int(source_cell) + component
                column_factor = (
                    coefficient * columns[source_cell, component]
                )
                mapped[row_slice, column] += (
                    row_factor
                    * (
                        local.mapped_conserved_hessian[
                            :,
                            :,
                            component,
                        ]
                        @ node_rate
                    )
                    * column_factor
                )
                height[row_slice, column] += (
                    row_factor
                    * (
                        local.vertical_storage_derivative[
                            :,
                            :,
                            component,
                        ]
                        @ node_rate
                    )
                    * column_factor
                )
    return mapped, height


def causal_five_field_monolithic_frozen_tangent(
    context: CausalFiveFieldDAEContext,
    base_primitives: np.ndarray,
    *,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    path_quadrature_order: int = 6,
    centered_storage_action_scaled_step: float = 1.0e-5,
) -> CausalFiveFieldMonolithicFrozenTangent:
    """Assemble the self-consistent frozen monolithic DAE tangent."""

    context = context.validated()
    base = np.asarray(base_primitives, dtype=float)
    n_cells = int(context.grid.centers.size)
    dimensions = _N_FIELDS * n_cells
    columns = np.asarray(primitive_column_scales, dtype=float).ravel()
    rows = np.asarray(conservation_row_scales, dtype=float).ravel()
    storage_step = float(centered_storage_action_scaled_step)
    if (
        base.shape != (n_cells, _N_FIELDS)
        or columns.shape != (dimensions,)
        or rows.shape != (dimensions,)
        or np.any(~np.isfinite(base))
        or np.any(~np.isfinite(columns))
        or np.any(~np.isfinite(rows))
        or np.any(columns <= 0.0)
        or np.any(rows <= 0.0)
        or not np.isfinite(storage_step)
        or storage_step <= 0.0
    ):
        raise ValueError("monolithic frozen-tangent inputs are invalid")

    (
        node_weights,
        node_cells,
        node_radii,
        node_measures,
        reconstruction_defect,
        partition_defect,
    ) = _node_reconstruction_weights(context, base)
    mapped_descriptor, height_descriptor = _descriptor_matrices(
        context,
        base,
        columns,
        rows,
        node_weights,
        node_cells,
        node_radii,
        node_measures,
    )
    descriptor = mapped_descriptor + height_descriptor

    spatial = causal_five_field_radial_analytic_tangent(
        context,
        base,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        path_quadrature_order=path_quadrature_order,
        center_broken_within_cell_paths=True,
    )
    base_evaluation = evaluate_causal_five_field_monolithic_backward_euler(
        base,
        base,
        1.0,
        context,
        path_quadrature_order=path_quadrature_order,
    )
    scaled_stationary = (
        np.asarray(base_evaluation.residual_rows, dtype=float).ravel()
        / rows
    )
    scaled_rate = np.linalg.solve(descriptor, -scaled_stationary)
    physical_rate = (columns * scaled_rate).reshape(n_cells, _N_FIELDS)
    mapped_storage_rate, height_storage_rate = (
        _storage_rate_derivative_matrices(
            context,
            base,
            physical_rate,
            columns,
            rows,
            node_weights,
            node_cells,
            node_radii,
            node_measures,
        )
    )
    storage_rate = mapped_storage_rate + height_storage_rate
    stationary = np.asarray(
        spatial.candidate_stationary_scaled_jacobian,
        dtype=float,
    )
    evolving = stationary + storage_rate
    generator = -np.linalg.solve(descriptor, evolving)

    rate_scale = max(
        float(np.max(np.abs(scaled_rate))),
        np.finfo(float).tiny,
    )
    half_interval = storage_step / rate_scale
    lower = base - half_interval * physical_rate
    upper = base + half_interval * physical_rate
    centered_increment = causal_five_field_monolithic_storage_increment(
        context,
        lower,
        upper,
    )
    centered_action = (
        np.asarray(
            centered_increment.total_storage_increment,
            dtype=float,
        ).ravel()
        / (2.0 * half_interval * C)
        / rows
    )
    matrix_action = descriptor @ scaled_rate
    action_scale = max(
        float(np.linalg.norm(centered_action)),
        float(np.linalg.norm(matrix_action)),
        np.finfo(float).tiny,
    )
    residual_scale = max(
        float(np.linalg.norm(scaled_stationary)),
        float(np.linalg.norm(descriptor @ scaled_rate)),
        np.finfo(float).tiny,
    )
    generator_scale = max(
        float(np.linalg.norm(evolving)),
        float(np.linalg.norm(descriptor @ generator)),
        np.finfo(float).tiny,
    )
    descriptor_scale = max(
        float(np.linalg.norm(descriptor)),
        np.finfo(float).tiny,
    )
    storage_rate_scale = max(
        float(np.linalg.norm(storage_rate)),
        np.finfo(float).tiny,
    )
    return CausalFiveFieldMonolithicFrozenTangent(
        base_primitives=np.array(base, copy=True),
        primitive_column_scales=np.array(columns, copy=True),
        conservation_row_scales=np.array(rows, copy=True),
        node_reconstruction_weights=np.asarray(
            node_weights,
            dtype=float,
        ),
        mapped_descriptor_scaled_matrix=mapped_descriptor,
        responsive_height_descriptor_scaled_matrix=height_descriptor,
        descriptor_scaled_matrix=descriptor,
        base_stationary_scaled_residual=scaled_stationary,
        scaled_base_rate_per_s=scaled_rate,
        physical_base_rate_per_s=physical_rate.ravel(),
        mapped_storage_rate_derivative_scaled_matrix=(
            mapped_storage_rate
        ),
        responsive_height_storage_rate_derivative_scaled_matrix=(
            height_storage_rate
        ),
        storage_rate_derivative_scaled_matrix=storage_rate,
        stationary_scaled_jacobian=stationary,
        evolving_scaled_jacobian=evolving,
        scaled_generator_per_s=generator,
        spatial_tangent=spatial,
        maximum_node_reconstruction_relative_defect=(
            reconstruction_defect
        ),
        maximum_node_partition_of_unity_defect=partition_defect,
        maximum_descriptor_component_defect=float(
            np.linalg.norm(
                descriptor
                - mapped_descriptor
                - height_descriptor
            )
            / descriptor_scale
        ),
        maximum_storage_rate_component_defect=float(
            np.linalg.norm(
                storage_rate
                - mapped_storage_rate
                - height_storage_rate
            )
            / storage_rate_scale
        ),
        maximum_base_rate_balance_defect=float(
            np.linalg.norm(
                descriptor @ scaled_rate + scaled_stationary
            )
            / residual_scale
        ),
        maximum_generator_factorization_defect=float(
            np.linalg.norm(descriptor @ generator + evolving)
            / generator_scale
        ),
        maximum_centered_storage_action_relative_defect=float(
            np.linalg.norm(centered_action - matrix_action)
            / action_scale
        ),
        centered_storage_action_scaled_step=storage_step,
        incoming_excision_characteristics=(
            base_evaluation.incoming_excision_characteristics
        ),
        uses_center_broken_within_cell_paths=True,
        uses_production_generator=False,
        uses_production_anchor_storage_derivative=False,
    )
