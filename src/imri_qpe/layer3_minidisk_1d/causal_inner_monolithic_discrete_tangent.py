"""Discrete variable-step BDF tangent helpers for the monolithic inner DAE.

Unlike the frozen continuous generator, this module differentiates one
complete accepted BDF residual.  The old primitive state and all three
history actions are active tangent inputs.  It is audit-only and changes no
production integration default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from imri_qpe.constants import C

from .causal_inner_bdf import causal_bdf_coefficients

from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    causal_five_field_colored_central_jacobian,
    causal_five_field_dae_jacobian_color_groups,
)
from .causal_inner_embedded_validation import (
    CausalEmbeddedActiveObservableAudit,
    causal_embedded_active_observable_audit,
)
from .causal_inner_monolithic_bdf import (
    CausalFiveFieldMonolithicBDFHistory,
    evaluate_causal_five_field_monolithic_bdf,
)
from .causal_inner_monolithic_dae import (
    causal_five_field_monolithic_storage_increment,
)
from .causal_inner_monolithic_tangent import _node_reconstruction_weights
from .causal_inner_radial_linear_tangent import (
    CausalFiveFieldRadialAnalyticTangent,
    causal_five_field_analytic_local_maps,
    causal_five_field_radial_analytic_tangent,
)
from .causal_inner_radial_frozen import (
    causal_five_field_radial_reduced_jacobian_pattern,
)


_N_FIELDS = 5


@dataclass(frozen=True)
class CausalFiveFieldMonolithicBDFHistoryDirection:
    """First variation of the complete previous accepted BDF interval."""

    previous_primitive_increment: np.ndarray
    previous_mapped_storage_increment: np.ndarray
    previous_responsive_height_storage_increment: np.ndarray

    def validated(
        self,
        *,
        n_directions: int,
        n_cells: int,
    ) -> CausalFiveFieldMonolithicBDFHistoryDirection:
        shape = (int(n_directions), int(n_cells), _N_FIELDS)
        primitive = np.asarray(self.previous_primitive_increment, dtype=float)
        mapped = np.asarray(self.previous_mapped_storage_increment, dtype=float)
        height = np.asarray(
            self.previous_responsive_height_storage_increment,
            dtype=float,
        )
        if (
            primitive.shape != shape
            or mapped.shape != shape
            or height.shape != shape
            or np.any(~np.isfinite(primitive))
            or np.any(~np.isfinite(mapped))
            or np.any(~np.isfinite(height))
        ):
            raise ValueError("monolithic BDF history direction is invalid")
        return CausalFiveFieldMonolithicBDFHistoryDirection(
            previous_primitive_increment=np.array(primitive, copy=True),
            previous_mapped_storage_increment=np.array(mapped, copy=True),
            previous_responsive_height_storage_increment=np.array(
                height,
                copy=True,
            ),
        )


@dataclass(frozen=True)
class CausalFiveFieldMonolithicDiscreteTangentStep:
    """Block first variation of one complete variable-step BDF residual."""

    new_primitive_directions: np.ndarray
    new_history_directions: CausalFiveFieldMonolithicBDFHistoryDirection
    scaled_step_matrix: np.ndarray
    scaled_right_hand_sides: np.ndarray
    maximum_base_scaled_residual: float
    maximum_step_matrix_jvp_relative_defect: float
    maximum_linear_solve_relative_defect: float
    color_count: int
    matrix_source: str
    finite_difference_step: float
    directional_step: float


@dataclass(frozen=True)
class CausalFiveFieldMonolithicDiscreteStepMatrix:
    """Exact analytic new-endpoint Jacobian of one complete BDF residual."""

    scaled_matrix: np.ndarray
    mapped_storage_scaled_matrix: np.ndarray
    responsive_height_storage_scaled_matrix: np.ndarray
    old_mapped_storage_scaled_matrix: np.ndarray
    old_responsive_height_storage_scaled_matrix: np.ndarray
    stationary_scaled_matrix: np.ndarray
    spatial_tangent: CausalFiveFieldRadialAnalyticTangent
    current_increment_coefficient: float
    previous_increment_coefficient: float
    current_timestep_seconds: float
    maximum_component_closure_defect: float
    maximum_node_reconstruction_relative_defect: float
    maximum_node_partition_of_unity_defect: float
    incoming_excision_characteristics: int


def causal_five_field_monolithic_discrete_step_matrix(
    context: CausalFiveFieldDAEContext,
    old_primitive_charts: np.ndarray,
    new_primitive_charts: np.ndarray,
    timestep_seconds: float,
    previous_timestep_seconds: float | None,
    *,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    temporal_quadrature_order: int = 4,
    path_quadrature_order: int = 6,
    order: int = 2,
) -> CausalFiveFieldMonolithicDiscreteStepMatrix:
    """Assemble the analytic Jacobian with respect to the new BDF endpoint."""

    context = context.validated()
    old = np.asarray(old_primitive_charts, dtype=float)
    new = np.asarray(new_primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    shape = (n_cells, _N_FIELDS)
    columns = np.asarray(primitive_column_scales, dtype=float).reshape(shape)
    rows = np.asarray(conservation_row_scales, dtype=float).reshape(shape)
    quadrature_order = int(temporal_quadrature_order)
    if (
        old.shape != shape
        or new.shape != shape
        or np.any(~np.isfinite(old))
        or np.any(~np.isfinite(new))
        or np.any(columns <= 0.0)
        or np.any(rows <= 0.0)
        or quadrature_order != temporal_quadrature_order
        or not 2 <= quadrature_order <= 12
    ):
        raise ValueError("monolithic discrete step-matrix inputs are invalid")

    (
        node_weights,
        node_cells,
        node_radii,
        node_measures,
        reconstruction_defect,
        partition_defect,
    ) = _node_reconstruction_weights(context, old)
    (
        new_node_weights,
        _new_cells,
        _new_radii,
        _new_measures,
        new_reconstruction_defect,
        new_partition_defect,
    ) = _node_reconstruction_weights(context, new)
    if not np.array_equal(node_weights, new_node_weights):
        raise ValueError("discrete step matrix requires one affine path branch")

    dimensions = old.size
    mapped = np.zeros((dimensions, dimensions), dtype=float)
    height = np.zeros_like(mapped)
    old_mapped = np.zeros_like(mapped)
    old_height = np.zeros_like(mapped)
    primitive_direction = new - old
    temporal_nodes, temporal_weights = np.polynomial.legendre.leggauss(
        quadrature_order
    )
    for temporal_node, temporal_weight in zip(
        temporal_nodes,
        temporal_weights,
        strict=True,
    ):
        fraction = 0.5 * (float(temporal_node) + 1.0)
        weight_t = 0.5 * float(temporal_weight)
        center = old + fraction * primitive_direction
        center_nodes = node_weights @ center
        direction_nodes = node_weights @ primitive_direction
        for node_weight, cell, radius, measure, center_node, direction_node in zip(
            node_weights,
            node_cells,
            node_radii,
            node_measures,
            center_nodes,
            direction_nodes,
            strict=True,
        ):
            local = causal_five_field_analytic_local_maps(
                context,
                float(radius),
                np.asarray(center_node, dtype=float),
            )
            mapped_local = np.array(local.mapped_conserved_jacobian, copy=True)
            height_local = np.array(local.vertical_storage_matrix, copy=True)
            old_mapped_local = -np.array(
                local.mapped_conserved_jacobian,
                copy=True,
            )
            old_height_local = -np.array(
                local.vertical_storage_matrix,
                copy=True,
            )
            for component in range(_N_FIELDS):
                mapped_derivative = (
                    local.mapped_conserved_hessian[:, :, component]
                    @ direction_node
                )
                height_derivative = (
                    local.vertical_storage_derivative[:, :, component]
                    @ direction_node
                )
                mapped_local[:, component] += fraction * mapped_derivative
                height_local[:, component] += fraction * height_derivative
                old_mapped_local[:, component] += (
                    (1.0 - fraction) * mapped_derivative
                )
                old_height_local[:, component] += (
                    (1.0 - fraction) * height_derivative
                )
            row_slice = slice(
                _N_FIELDS * int(cell),
                _N_FIELDS * (int(cell) + 1),
            )
            row_factor = weight_t * float(measure) / C / rows[int(cell)]
            for source_cell in np.flatnonzero(node_weight):
                coefficient = float(node_weight[source_cell])
                for component in range(_N_FIELDS):
                    column = _N_FIELDS * int(source_cell) + component
                    column_factor = (
                        coefficient * columns[source_cell, component]
                    )
                    mapped[row_slice, column] += (
                        row_factor
                        * mapped_local[:, component]
                        * column_factor
                    )
                    height[row_slice, column] += (
                        row_factor
                        * height_local[:, component]
                        * column_factor
                    )
                    old_mapped[row_slice, column] += (
                        row_factor
                        * old_mapped_local[:, component]
                        * column_factor
                    )
                    old_height[row_slice, column] += (
                        row_factor
                        * old_height_local[:, component]
                        * column_factor
                    )

    spatial = causal_five_field_radial_analytic_tangent(
        context,
        new,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        path_quadrature_order=path_quadrature_order,
        center_broken_within_cell_paths=True,
    )
    stationary = np.asarray(
        spatial.candidate_stationary_scaled_jacobian,
        dtype=float,
    )
    selected_order = int(order)
    if selected_order != order or selected_order not in (1, 2):
        raise ValueError("monolithic discrete step-matrix order is invalid")
    coefficients = (
        causal_bdf_coefficients(1, timestep_seconds)
        if selected_order == 1
        else causal_bdf_coefficients(
            2,
            timestep_seconds,
            previous_timestep_seconds,
        )
    )
    storage = mapped + height
    matrix = (
        coefficients.current_increment_coefficient
        / float(timestep_seconds)
        * storage
        + stationary
    )
    scale = max(
        float(np.linalg.norm(matrix)),
        float(np.linalg.norm(storage)),
        float(np.linalg.norm(stationary)),
        np.finfo(float).tiny,
    )
    return CausalFiveFieldMonolithicDiscreteStepMatrix(
        scaled_matrix=matrix,
        mapped_storage_scaled_matrix=mapped,
        responsive_height_storage_scaled_matrix=height,
        old_mapped_storage_scaled_matrix=old_mapped,
        old_responsive_height_storage_scaled_matrix=old_height,
        stationary_scaled_matrix=stationary,
        spatial_tangent=spatial,
        current_increment_coefficient=(
            coefficients.current_increment_coefficient
        ),
        previous_increment_coefficient=(
            coefficients.previous_increment_coefficient
        ),
        current_timestep_seconds=float(timestep_seconds),
        maximum_component_closure_defect=float(
            np.linalg.norm(
                matrix
                - coefficients.current_increment_coefficient
                / float(timestep_seconds)
                * (mapped + height)
                - stationary
            )
            / scale
        ),
        maximum_node_reconstruction_relative_defect=max(
            float(reconstruction_defect),
            float(new_reconstruction_defect),
        ),
        maximum_node_partition_of_unity_defect=max(
            float(partition_defect),
            float(new_partition_defect),
        ),
        incoming_excision_characteristics=(
            spatial.incoming_inner_characteristics
        ),
    )


def causal_five_field_monolithic_discrete_export_directions(
    step_matrix: CausalFiveFieldMonolithicDiscreteStepMatrix,
    primitive_directions: np.ndarray,
    coupling_face_index: int,
) -> tuple[np.ndarray, CausalEmbeddedActiveObservableAudit]:
    """Apply the exact radial Tier-I export map to physical directions."""

    spatial = step_matrix.spatial_tangent
    directions = np.asarray(primitive_directions, dtype=float)
    if directions.ndim == 2:
        directions = directions[None, ...]
    expected = tuple(np.asarray(spatial.base_primitives).shape)
    if (
        directions.ndim != 3
        or tuple(directions.shape[1:]) != expected
        or np.any(~np.isfinite(directions))
    ):
        raise ValueError("discrete export directions are invalid")
    tangent_view = type(
        "_EmbeddedObservableTangentView",
        (),
        {
            "base_primitives": spatial.base_primitives,
            "conservation_row_scales": spatial.conservation_row_scales,
            "spatial_tangent": spatial,
            "stationary_scaled_jacobian": (
                spatial.candidate_stationary_scaled_jacobian
            ),
        },
    )()
    audit = causal_embedded_active_observable_audit(
        tangent_view,
        int(coupling_face_index),
    )
    columns = np.asarray(spatial.primitive_column_scales, dtype=float).ravel()
    scaled = directions.reshape(directions.shape[0], -1) / columns[None, :]
    values = scaled @ np.asarray(audit.observable_map, dtype=float).T
    return np.asarray(values, dtype=float), audit


def causal_five_field_monolithic_bdf_history_from_interval(
    context: CausalFiveFieldDAEContext,
    old_primitive_charts: np.ndarray,
    new_primitive_charts: np.ndarray,
    timestep_seconds: float,
) -> CausalFiveFieldMonolithicBDFHistory:
    """Reconstruct the exact accepted path history from two endpoint states."""

    old = np.asarray(old_primitive_charts, dtype=float)
    new = np.asarray(new_primitive_charts, dtype=float)
    storage = causal_five_field_monolithic_storage_increment(context, old, new)
    return CausalFiveFieldMonolithicBDFHistory(
        previous_primitive_increment=np.asarray(new - old, dtype=float),
        previous_mapped_storage_increment=np.asarray(
            storage.mapped_path_increment,
            dtype=float,
        ),
        previous_responsive_height_storage_increment=np.asarray(
            storage.responsive_height_path_increment,
            dtype=float,
        ),
        previous_timestep_seconds=float(timestep_seconds),
    ).validated(n_cells=old.shape[0])


def _shifted_history(
    base: CausalFiveFieldMonolithicBDFHistory,
    direction: CausalFiveFieldMonolithicBDFHistoryDirection,
    index: int,
    factor: float,
) -> CausalFiveFieldMonolithicBDFHistory:
    return CausalFiveFieldMonolithicBDFHistory(
        previous_primitive_increment=(
            base.previous_primitive_increment
            + factor * direction.previous_primitive_increment[index]
        ),
        previous_mapped_storage_increment=(
            base.previous_mapped_storage_increment
            + factor * direction.previous_mapped_storage_increment[index]
        ),
        previous_responsive_height_storage_increment=(
            base.previous_responsive_height_storage_increment
            + factor
            * direction.previous_responsive_height_storage_increment[index]
        ),
        previous_timestep_seconds=base.previous_timestep_seconds,
        temporal_path_scheme=base.temporal_path_scheme,
    )


def causal_five_field_monolithic_bdf_history_direction(
    context: CausalFiveFieldDAEContext,
    base_old_primitive_charts: np.ndarray,
    base_new_primitive_charts: np.ndarray,
    old_primitive_directions: np.ndarray,
    new_primitive_directions: np.ndarray,
    *,
    directional_step: float = 1.0e-2,
    analytic_step_matrix: (
        CausalFiveFieldMonolithicDiscreteStepMatrix | None
    ) = None,
) -> CausalFiveFieldMonolithicBDFHistoryDirection:
    """Differentiate all stored path actions for one accepted interval.

    When ``analytic_step_matrix`` is supplied, its old/new endpoint storage
    matrices provide the derivative directly.  The centered-difference route
    remains available as an independent audit and for legacy callers.
    """

    old = np.asarray(base_old_primitive_charts, dtype=float)
    new = np.asarray(base_new_primitive_charts, dtype=float)
    old_directions = np.asarray(old_primitive_directions, dtype=float)
    new_directions = np.asarray(new_primitive_directions, dtype=float)
    alpha = float(directional_step)
    if (
        old.shape != new.shape
        or old.ndim != 2
        or old.shape[1] != _N_FIELDS
        or old_directions.ndim != 3
        or old_directions.shape[1:] != old.shape
        or new_directions.shape != old_directions.shape
        or np.any(~np.isfinite(old))
        or np.any(~np.isfinite(new))
        or np.any(~np.isfinite(old_directions))
        or np.any(~np.isfinite(new_directions))
        or not np.isfinite(alpha)
        or alpha <= 0.0
    ):
        raise ValueError("monolithic BDF history-direction inputs are invalid")
    if analytic_step_matrix is not None:
        analytic = analytic_step_matrix
        dimensions = int(old.size)
        expected_matrix_shape = (dimensions, dimensions)
        matrix_fields = (
            analytic.mapped_storage_scaled_matrix,
            analytic.responsive_height_storage_scaled_matrix,
            analytic.old_mapped_storage_scaled_matrix,
            analytic.old_responsive_height_storage_scaled_matrix,
        )
        columns = np.asarray(
            analytic.spatial_tangent.primitive_column_scales,
            dtype=float,
        ).reshape(old.shape)
        rows = np.asarray(
            analytic.spatial_tangent.conservation_row_scales,
            dtype=float,
        ).reshape(old.shape)
        if (
            any(
                np.asarray(matrix).shape != expected_matrix_shape
                for matrix in matrix_fields
            )
            or any(
                np.any(~np.isfinite(matrix))
                for matrix in matrix_fields
            )
            or np.any(columns <= 0.0)
            or np.any(rows <= 0.0)
        ):
            raise ValueError("analytic monolithic history matrix is invalid")
        old_scaled = (old_directions / columns[None, :, :]).reshape(
            old_directions.shape[0],
            -1,
        )
        new_scaled = (new_directions / columns[None, :, :]).reshape(
            new_directions.shape[0],
            -1,
        )
        mapped_scaled = (
            analytic.old_mapped_storage_scaled_matrix @ old_scaled.T
            + analytic.mapped_storage_scaled_matrix @ new_scaled.T
        ).T.reshape(old_directions.shape[0], *old.shape)
        height_scaled = (
            analytic.old_responsive_height_storage_scaled_matrix @ old_scaled.T
            + analytic.responsive_height_storage_scaled_matrix @ new_scaled.T
        ).T.reshape(old_directions.shape[0], *old.shape)
        return CausalFiveFieldMonolithicBDFHistoryDirection(
            previous_primitive_increment=new_directions - old_directions,
            previous_mapped_storage_increment=(
                mapped_scaled * C * rows[None, :, :]
            ),
            previous_responsive_height_storage_increment=(
                height_scaled * C * rows[None, :, :]
            ),
        ).validated(
            n_directions=old_directions.shape[0],
            n_cells=old.shape[0],
        )

    mapped = []
    height = []
    for old_direction, new_direction in zip(
        old_directions,
        new_directions,
        strict=True,
    ):
        plus = causal_five_field_monolithic_storage_increment(
            context,
            old + alpha * old_direction,
            new + alpha * new_direction,
        )
        minus = causal_five_field_monolithic_storage_increment(
            context,
            old - alpha * old_direction,
            new - alpha * new_direction,
        )
        mapped.append(
            (
                plus.mapped_path_increment
                - minus.mapped_path_increment
            )
            / (2.0 * alpha)
        )
        height.append(
            (
                plus.responsive_height_path_increment
                - minus.responsive_height_path_increment
            )
            / (2.0 * alpha)
        )
    return CausalFiveFieldMonolithicBDFHistoryDirection(
        previous_primitive_increment=new_directions - old_directions,
        previous_mapped_storage_increment=np.asarray(mapped, dtype=float),
        previous_responsive_height_storage_increment=np.asarray(
            height,
            dtype=float,
        ),
    ).validated(
        n_directions=old_directions.shape[0],
        n_cells=old.shape[0],
    )


def causal_five_field_monolithic_discrete_tangent_step(
    context: CausalFiveFieldDAEContext,
    base_old_primitive_charts: np.ndarray,
    base_new_primitive_charts: np.ndarray,
    timestep_seconds: float,
    base_history: CausalFiveFieldMonolithicBDFHistory,
    old_primitive_directions: np.ndarray,
    history_directions: CausalFiveFieldMonolithicBDFHistoryDirection,
    *,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    finite_difference_step: float = 2.0e-6,
    directional_step: float = 1.0e-2,
    cell_half_bandwidth: int = 3,
    scaled_step_matrix: np.ndarray | None = None,
    analytic_step_matrix: CausalFiveFieldMonolithicDiscreteStepMatrix | None = None,
    audit_complete_residual: bool = True,
) -> CausalFiveFieldMonolithicDiscreteTangentStep:
    """Advance several perturbations through one exact discrete BDF step."""

    context = context.validated()
    old = np.asarray(base_old_primitive_charts, dtype=float)
    new = np.asarray(base_new_primitive_charts, dtype=float)
    old_directions = np.asarray(old_primitive_directions, dtype=float)
    n_cells = int(context.grid.centers.size)
    n_directions = int(old_directions.shape[0])
    shape = (n_cells, _N_FIELDS)
    columns = np.asarray(primitive_column_scales, dtype=float).reshape(shape)
    rows = np.asarray(conservation_row_scales, dtype=float).reshape(shape)
    history = base_history.validated(n_cells=n_cells)
    directions = history_directions.validated(
        n_directions=n_directions,
        n_cells=n_cells,
    )
    alpha = float(directional_step)
    if (
        old.shape != shape
        or new.shape != shape
        or old_directions.shape != (n_directions, *shape)
        or np.any(~np.isfinite(old))
        or np.any(~np.isfinite(new))
        or np.any(~np.isfinite(old_directions))
        or np.any(columns <= 0.0)
        or np.any(rows <= 0.0)
        or not np.isfinite(alpha)
        or alpha <= 0.0
    ):
        raise ValueError("monolithic discrete-tangent inputs are invalid")

    def scaled_residual(
        old_values: np.ndarray,
        new_values: np.ndarray,
        history_values: CausalFiveFieldMonolithicBDFHistory,
    ) -> np.ndarray:
        evaluation = evaluate_causal_five_field_monolithic_bdf(
            old_values,
            new_values,
            timestep_seconds,
            context,
            order=2,
            history=history_values,
        )
        return np.asarray(evaluation.residual_rows / rows, dtype=float).ravel()

    zero = np.zeros(old.size, dtype=float)

    def new_residual(scaled_new_direction: np.ndarray) -> np.ndarray:
        shifted = new + columns * np.asarray(
            scaled_new_direction,
            dtype=float,
        ).reshape(shape)
        return scaled_residual(old, shifted, history)

    pattern = causal_five_field_radial_reduced_jacobian_pattern(
        n_cells,
        cell_half_bandwidth=cell_half_bandwidth,
    )
    if analytic_step_matrix is not None and scaled_step_matrix is not None:
        raise ValueError("provide one discrete step matrix source")
    if analytic_step_matrix is not None:
        analytic = analytic_step_matrix
        supplied = np.asarray(analytic.scaled_matrix, dtype=float)
        if (
            supplied.shape != (old.size, old.size)
            or np.any(~np.isfinite(supplied))
            or analytic.current_timestep_seconds != float(timestep_seconds)
        ):
            raise ValueError("analytic monolithic step matrix is invalid")
        matrix = csc_matrix(supplied)
        matrix_source = "analytic_complete_discrete_step_matrix"
    elif scaled_step_matrix is None:
        matrix = causal_five_field_colored_central_jacobian(
            new_residual,
            zero,
            pattern,
            finite_difference_step=finite_difference_step,
        ).tocsc()
        matrix_source = "colored_complete_discrete_residual"
    else:
        supplied = np.asarray(scaled_step_matrix, dtype=float)
        if (
            supplied.shape != (old.size, old.size)
            or np.any(~np.isfinite(supplied))
        ):
            raise ValueError("supplied monolithic step matrix is invalid")
        matrix = csc_matrix(supplied)
        matrix_source = "supplied_cached_analytic_step_matrix"

    scaled_old_directions = (
        old_directions / columns[None, :, :]
    ).reshape(n_directions, -1)
    if analytic_step_matrix is None:
        right_hand_sides = []
        for index, old_direction in enumerate(old_directions):
            plus = scaled_residual(
                old + alpha * old_direction,
                new,
                _shifted_history(history, directions, index, alpha),
            )
            minus = scaled_residual(
                old - alpha * old_direction,
                new,
                _shifted_history(history, directions, index, -alpha),
            )
            right_hand_sides.append((plus - minus) / (2.0 * alpha))
        right_hand_side_matrix = np.asarray(right_hand_sides, dtype=float).T
    else:
        old_storage = (
            analytic.old_mapped_storage_scaled_matrix
            + analytic.old_responsive_height_storage_scaled_matrix
        )
        history_storage = (
            directions.previous_mapped_storage_increment
            + directions.previous_responsive_height_storage_increment
        ) / (C * rows[None, :, :])
        right_hand_side_matrix = (
            analytic.current_increment_coefficient
            / analytic.current_timestep_seconds
            * old_storage
            @ scaled_old_directions.T
            + analytic.previous_increment_coefficient
            / analytic.current_timestep_seconds
            * history_storage.reshape(n_directions, -1).T
        )
    factorization = splu(matrix)
    scaled_new_directions = factorization.solve(-right_hand_side_matrix).T
    new_directions = scaled_new_directions.reshape(
        n_directions,
        n_cells,
        _N_FIELDS,
    ) * columns[None, :, :]

    if analytic_step_matrix is None:
        new_history_directions = (
            causal_five_field_monolithic_bdf_history_direction(
                context,
                old,
                new,
                old_directions,
                new_directions,
                directional_step=alpha,
            )
        )
    else:
        mapped_scaled = (
            analytic.old_mapped_storage_scaled_matrix
            @ scaled_old_directions.T
            + analytic.mapped_storage_scaled_matrix
            @ scaled_new_directions.T
        ).T.reshape(n_directions, n_cells, _N_FIELDS)
        height_scaled = (
            analytic.old_responsive_height_storage_scaled_matrix
            @ scaled_old_directions.T
            + analytic.responsive_height_storage_scaled_matrix
            @ scaled_new_directions.T
        ).T.reshape(n_directions, n_cells, _N_FIELDS)
        new_history_directions = (
            CausalFiveFieldMonolithicBDFHistoryDirection(
                previous_primitive_increment=(
                    new_directions - old_directions
                ),
                previous_mapped_storage_increment=(
                    mapped_scaled * C * rows[None, :, :]
                ),
                previous_responsive_height_storage_increment=(
                    height_scaled * C * rows[None, :, :]
                ),
            ).validated(
                n_directions=n_directions,
                n_cells=n_cells,
            )
        )

    sequence = np.arange(old.size, dtype=float)
    audit_direction = np.sin(0.371 * (sequence + 1.0))
    audit_direction /= max(float(np.max(np.abs(audit_direction))), 1.0)
    if audit_complete_residual:
        base_values = scaled_residual(old, new, history)
        direct = (
            new_residual(finite_difference_step * audit_direction)
            - new_residual(-finite_difference_step * audit_direction)
        ) / (2.0 * finite_difference_step)
        predicted = matrix @ audit_direction
        jvp_scale = max(
            float(np.linalg.norm(direct)),
            float(np.linalg.norm(predicted)),
            np.finfo(float).tiny,
        )
        jvp_defect = float(np.linalg.norm(direct - predicted) / jvp_scale)
        base_residual = float(np.max(np.abs(base_values)))
    else:
        jvp_defect = float("nan")
        base_residual = float("nan")
    linear_residual = matrix @ scaled_new_directions.T + right_hand_side_matrix
    linear_scale = max(
        float(np.linalg.norm(right_hand_side_matrix)),
        float(np.linalg.norm(matrix @ scaled_new_directions.T)),
        np.finfo(float).tiny,
    )
    return CausalFiveFieldMonolithicDiscreteTangentStep(
        new_primitive_directions=np.asarray(new_directions, dtype=float),
        new_history_directions=new_history_directions,
        scaled_step_matrix=np.asarray(matrix.toarray(), dtype=float),
        scaled_right_hand_sides=np.asarray(right_hand_side_matrix.T, dtype=float),
        maximum_base_scaled_residual=base_residual,
        maximum_step_matrix_jvp_relative_defect=jvp_defect,
        maximum_linear_solve_relative_defect=float(
            np.linalg.norm(linear_residual) / linear_scale
        ),
        color_count=int(
            len(causal_five_field_dae_jacobian_color_groups(pattern))
        ),
        matrix_source=matrix_source,
        finite_difference_step=float(finite_difference_step),
        directional_step=alpha,
    )
