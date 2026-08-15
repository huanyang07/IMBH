"""Production-neutral monolithic causal-inner descriptor-path DAE.

The rejected WP10c9d5 hybrid replaced a stationary spatial Jacobian inside a
production-anchored frozen generator.  This module instead defines one
primitive-only backward-Euler residual,

``storage(old -> new) / (c dt) + stationary(new) = 0``.

Mapped conserved storage is an exact endpoint increment.  The
responsive-height contribution is a generally non-exact temporal one-form,
so it is integrated as a declared nonconservative product along the same
primitive path.  The complete radial fluctuation ledger supplies every
stationary block and one shared M/J/E face flux.  No production generator,
production base rate, or ``DM[p_dot_production]`` correction enters this
residual.

The signed characteristic interface split remains the production-neutral
audit split certified by the preceding work packages.  Consequently this is
a method/preflight architecture, not a promoted nonlinear solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C

from .causal_inner_characteristic_dissipation import (
    DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
)
from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    _gauss_legendre_cell_nodes_and_measures,
    causal_five_field_reconstruct_face_charts,
)
from .causal_inner_radial_fluctuation import (
    CausalFiveFieldRadialCandidateLedger,
    CausalFiveFieldRadialPathJump,
    causal_five_field_radial_candidate_ledger,
    causal_five_field_radial_extended_path_jump,
)
from .causal_inner_radial_linear_tangent import (
    causal_five_field_analytic_local_maps,
)


_N_FIELDS = 5


@dataclass(frozen=True)
class CausalFiveFieldMonolithicStorageIncrement:
    """Complete cell-integrated temporal increment on one declared path."""

    old_primitive_charts: np.ndarray
    new_primitive_charts: np.ndarray
    mapped_endpoint_increment: np.ndarray
    mapped_path_increment: np.ndarray
    responsive_height_path_increment: np.ndarray
    total_storage_increment: np.ndarray
    temporal_quadrature_order: int
    reconstruction_directional_step: float
    maximum_mapped_path_closure_defect: float
    maximum_affine_reconstruction_path_defect: float
    maximum_path_reconstruction_factor_change: float
    minimum_path_reconstruction_factor: float
    maximum_path_reconstruction_factor: float
    one_flux_reconstruction_for_space_and_storage: bool
    uses_exact_affine_reconstruction_path_derivative: bool
    mapped_storage_is_exact_endpoint_increment: bool
    responsive_height_is_nonconservative_temporal_product: bool


@dataclass(frozen=True)
class CausalFiveFieldMonolithicStorageRate:
    """Direct path evaluation of mapped and height storage rates."""

    mapped_rate_per_s: np.ndarray
    responsive_height_rate_per_s: np.ndarray
    maximum_node_reconstruction_relative_defect: float
    maximum_node_partition_of_unity_defect: float


@dataclass(frozen=True)
class CausalFiveFieldTemporalIntegrabilityAudit:
    """Exterior-derivative and loop audit of the temporal descriptor."""

    radius: float
    primitive_chart: np.ndarray
    primitive_scales: np.ndarray
    derivative_step: float
    vertical_exterior_derivative: np.ndarray
    complete_exterior_derivative: np.ndarray
    relative_vertical_exterior_derivative: float
    relative_complete_exterior_derivative: float
    loop_fields: tuple[int, int]
    loop_amplitude: float
    loop_quadrature_order: int
    first_path_vertical_increment: np.ndarray
    second_path_vertical_increment: np.ndarray
    loop_vertical_increment: np.ndarray
    relative_loop_to_vertical_path: float


@dataclass(frozen=True)
class CausalFiveFieldMonolithicDAEEvaluation:
    """Block-complete primitive-only backward-Euler DAE residual."""

    storage_increment: CausalFiveFieldMonolithicStorageIncrement
    stationary_ledger: CausalFiveFieldRadialCandidateLedger
    mapped_temporal_storage_rows: np.ndarray
    responsive_height_temporal_storage_rows: np.ndarray
    conservative_transport_rows: np.ndarray
    shear_principal_rows: np.ndarray
    height_principal_rows: np.ndarray
    local_stress_relaxation_rows: np.ndarray
    geometry_rows: np.ndarray
    cooling_rows: np.ndarray
    stream_rows: np.ndarray
    lower_height_work_rows: np.ndarray
    residual_rows: np.ndarray
    center_broken_path_adjustment_rows: np.ndarray
    inner_left_half_path: CausalFiveFieldRadialPathJump
    inner_right_half_path: CausalFiveFieldRadialPathJump
    maximum_block_ledger_defect: float
    maximum_center_broken_path_adjustment: float
    incoming_excision_characteristics: int
    uses_production_generator: bool
    uses_production_anchor_storage_derivative: bool


def _validate_charts(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
) -> np.ndarray:
    charts = np.asarray(primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    if (
        charts.shape != (n_cells, _N_FIELDS)
        or np.any(~np.isfinite(charts))
    ):
        raise ValueError("monolithic primitive charts are invalid")
    return charts


def _spatial_nodes(
    context: CausalFiveFieldDAEContext,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Return cell quadrature radii and physical cell-measure weights."""

    if context.cell_storage_quadrature == "midpoint":
        return tuple(
            (
                (
                    float(context.grid.centers[cell]),
                    float(context.grid.cell_measures[cell]),
                ),
            )
            for cell in range(int(context.grid.centers.size))
        )
    return tuple(
        tuple(
            (float(radius), float(weight))
            for radius, weight in zip(
                *_gauss_legendre_cell_nodes_and_measures(context, cell),
                strict=True,
            )
        )
        for cell in range(int(context.grid.centers.size))
    )


def _node_charts(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    nodes: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the single flux/storage reconstruction at cell nodes."""

    charts = _validate_charts(context, primitive_charts)
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        charts,
        purpose="flux",
    )
    log_centers = np.log(np.asarray(context.grid.centers, dtype=float))
    values = []
    for cell, cell_nodes in enumerate(nodes):
        for radius, _weight in cell_nodes:
            values.append(
                charts[cell]
                + reconstruction.limited_slopes[cell]
                * (np.log(radius) - log_centers[cell])
            )
    return (
        np.asarray(values, dtype=float),
        np.asarray(reconstruction.admissibility_factors, dtype=float),
    )


def _integrated_mapped_storage(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    nodes: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cell-integrated mapped storage from the shared reconstruction."""

    node_values, factors = _node_charts(context, primitive_charts, nodes)
    integrated = np.zeros(
        (int(context.grid.centers.size), _N_FIELDS),
        dtype=float,
    )
    flat = 0
    for cell, cell_nodes in enumerate(nodes):
        for radius, weight in cell_nodes:
            local = causal_five_field_analytic_local_maps(
                context,
                radius,
                node_values[flat],
            )
            integrated[cell] += weight * local.mapped_conserved
            flat += 1
    return integrated, factors, node_values


def _relative_norm(difference: np.ndarray, *references: np.ndarray) -> float:
    scale = max(
        *(float(np.linalg.norm(reference)) for reference in references),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(difference) / scale)


def _integrate_local_vertical_one_form(
    context: CausalFiveFieldDAEContext,
    radius: float,
    start: np.ndarray,
    end: np.ndarray,
    *,
    quadrature_order: int,
) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    fractions = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    increment = np.zeros(_N_FIELDS, dtype=float)
    for fraction, weight in zip(fractions, weights, strict=True):
        local = causal_five_field_analytic_local_maps(
            context,
            radius,
            np.asarray(start, dtype=float) + float(fraction) * direction,
        )
        increment += (
            float(weight) * local.vertical_storage_matrix @ direction
        )
    return increment


def causal_five_field_temporal_storage_integrability_audit(
    context: CausalFiveFieldDAEContext,
    radius: float,
    primitive_chart: np.ndarray,
    *,
    primitive_scales: np.ndarray | None = None,
    derivative_step: float = 1.0e-5,
    loop_fields: tuple[int, int] = (2, 3),
    loop_amplitude: float = 1.0e-4,
    loop_quadrature_order: int = 6,
) -> CausalFiveFieldTemporalIntegrabilityAudit:
    """Audit whether the complete temporal descriptor is a state gradient.

    For a state potential ``U(p)`` to exist locally, every row of the
    descriptor must have a vanishing exterior derivative.  This helper
    differentiates the descriptor in scaled primitive coordinates and also
    evaluates the responsive-height one-form around a small rectangular loop.
    A resolved nonzero result means the temporal term must remain a declared
    path-dependent product (or the state must be augmented); it must not be
    represented as an endpoint-only storage map.
    """

    context = context.validated()
    selected_radius = float(radius)
    chart = np.asarray(primitive_chart, dtype=float)
    if (
        not np.isfinite(selected_radius)
        or selected_radius <= 0.0
        or chart.shape != (_N_FIELDS,)
        or np.any(~np.isfinite(chart))
    ):
        raise ValueError("temporal integrability inputs are invalid")
    if primitive_scales is None:
        scales = np.ones(_N_FIELDS, dtype=float)
        scales[4] = max(abs(float(chart[4])), 1.0e-8)
    else:
        scales = np.asarray(primitive_scales, dtype=float)
    step = float(derivative_step)
    amplitude = float(loop_amplitude)
    order = int(loop_quadrature_order)
    first_field, second_field = (int(value) for value in loop_fields)
    if (
        scales.shape != (_N_FIELDS,)
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
        or not np.isfinite(step)
        or not 1.0e-7 <= step <= 1.0e-3
        or not np.isfinite(amplitude)
        or not 1.0e-7 <= amplitude <= 1.0e-2
        or order != loop_quadrature_order
        or not 2 <= order <= 12
        or not 0 <= first_field < second_field < _N_FIELDS
    ):
        raise ValueError("temporal integrability controls are invalid")

    vertical_derivatives = np.zeros(
        (_N_FIELDS, _N_FIELDS, _N_FIELDS),
        dtype=float,
    )
    complete_derivatives = np.zeros_like(vertical_derivatives)
    for field in range(_N_FIELDS):
        perturbation = np.zeros(_N_FIELDS, dtype=float)
        perturbation[field] = step * scales[field]
        plus = causal_five_field_analytic_local_maps(
            context,
            selected_radius,
            chart + perturbation,
        )
        minus = causal_five_field_analytic_local_maps(
            context,
            selected_radius,
            chart - perturbation,
        )
        vertical_derivatives[field] = (
            plus.vertical_storage_matrix - minus.vertical_storage_matrix
        ) / (2.0 * step)
        complete_derivatives[field] = (
            plus.temporal_storage_matrix - minus.temporal_storage_matrix
        ) / (2.0 * step)

    pairs = tuple(
        (left, right)
        for left in range(_N_FIELDS)
        for right in range(left + 1, _N_FIELDS)
    )
    vertical_exterior = np.asarray(
        [
            vertical_derivatives[right, :, left] * scales[left]
            - vertical_derivatives[left, :, right] * scales[right]
            for left, right in pairs
        ],
        dtype=float,
    )
    complete_exterior = np.asarray(
        [
            complete_derivatives[right, :, left] * scales[left]
            - complete_derivatives[left, :, right] * scales[right]
            for left, right in pairs
        ],
        dtype=float,
    )
    vertical_derivative_scale = (
        vertical_derivatives * scales[None, None, :]
    )
    complete_derivative_scale = (
        complete_derivatives * scales[None, None, :]
    )

    first_increment = np.zeros(_N_FIELDS, dtype=float)
    second_increment = np.zeros(_N_FIELDS, dtype=float)
    first_increment[first_field] = amplitude * scales[first_field]
    second_increment[second_field] = amplitude * scales[second_field]
    first_path = (
        _integrate_local_vertical_one_form(
            context,
            selected_radius,
            chart,
            chart + first_increment,
            quadrature_order=order,
        )
        + _integrate_local_vertical_one_form(
            context,
            selected_radius,
            chart + first_increment,
            chart + first_increment + second_increment,
            quadrature_order=order,
        )
    )
    second_path = (
        _integrate_local_vertical_one_form(
            context,
            selected_radius,
            chart,
            chart + second_increment,
            quadrature_order=order,
        )
        + _integrate_local_vertical_one_form(
            context,
            selected_radius,
            chart + second_increment,
            chart + first_increment + second_increment,
            quadrature_order=order,
        )
    )
    loop = first_path - second_path
    return CausalFiveFieldTemporalIntegrabilityAudit(
        radius=selected_radius,
        primitive_chart=np.array(chart, copy=True),
        primitive_scales=np.array(scales, copy=True),
        derivative_step=step,
        vertical_exterior_derivative=vertical_exterior,
        complete_exterior_derivative=complete_exterior,
        relative_vertical_exterior_derivative=_relative_norm(
            vertical_exterior,
            vertical_derivative_scale,
        ),
        relative_complete_exterior_derivative=_relative_norm(
            complete_exterior,
            complete_derivative_scale,
        ),
        loop_fields=(first_field, second_field),
        loop_amplitude=amplitude,
        loop_quadrature_order=order,
        first_path_vertical_increment=np.asarray(first_path, dtype=float),
        second_path_vertical_increment=np.asarray(second_path, dtype=float),
        loop_vertical_increment=np.asarray(loop, dtype=float),
        relative_loop_to_vertical_path=_relative_norm(
            loop,
            first_path,
            second_path,
        ),
    )


def causal_five_field_monolithic_storage_increment(
    context: CausalFiveFieldDAEContext,
    old_primitive_charts: np.ndarray,
    new_primitive_charts: np.ndarray,
    *,
    temporal_quadrature_order: int = 4,
    reconstruction_directional_step: float = 1.0e-2,
) -> CausalFiveFieldMonolithicStorageIncrement:
    """Integrate mapped and responsive-height storage on one common path.

    The global primitive path is straight from ``old`` to ``new``.  At each
    temporal quadrature node the same reconstruction used by the spatial
    residual is evaluated.  Piecewise-constant, unlimited-PLM, and inactive
    quadratic-admissible reconstructions are affine in the primitive charts;
    after verifying that branch at the endpoints and every path node, their
    exact reconstructed-node secant supplies the path derivative.  Other
    branches retain a centered reconstruction derivative.
    ``reconstruction_directional_step`` is a dimensionless fraction of the
    complete old-to-new path used only by that fallback, not an absolute
    primitive perturbation.  The analytic local temporal descriptor supplies
    both a mapped-path cross-check and the responsive-height nonconservative
    one-form.
    """

    context = context.validated()
    old = _validate_charts(context, old_primitive_charts)
    new = _validate_charts(context, new_primitive_charts)
    order = int(temporal_quadrature_order)
    step = float(reconstruction_directional_step)
    if (
        order != temporal_quadrature_order
        or not 2 <= order <= 12
        or not np.isfinite(step)
        or not 1.0e-7 <= step <= 1.0e-2
    ):
        raise ValueError("monolithic storage path controls are invalid")

    nodes = _spatial_nodes(context)
    if np.array_equal(old, new):
        _node_values, factors = _node_charts(context, old, nodes)
        zeros = np.zeros_like(old)
        exact_affine = bool(
            context.spatial_reconstruction
            in {
                "piecewise_constant",
                "plm_unlimited",
                "quadratic_admissible",
            }
            and np.array_equal(factors, np.ones_like(factors))
        )
        return CausalFiveFieldMonolithicStorageIncrement(
            old_primitive_charts=np.array(old, copy=True),
            new_primitive_charts=np.array(new, copy=True),
            mapped_endpoint_increment=np.array(zeros, copy=True),
            mapped_path_increment=np.array(zeros, copy=True),
            responsive_height_path_increment=np.array(zeros, copy=True),
            total_storage_increment=np.array(zeros, copy=True),
            temporal_quadrature_order=order,
            reconstruction_directional_step=step,
            maximum_mapped_path_closure_defect=0.0,
            maximum_affine_reconstruction_path_defect=0.0,
            maximum_path_reconstruction_factor_change=0.0,
            minimum_path_reconstruction_factor=float(np.min(factors)),
            maximum_path_reconstruction_factor=float(np.max(factors)),
            one_flux_reconstruction_for_space_and_storage=True,
            uses_exact_affine_reconstruction_path_derivative=exact_affine,
            mapped_storage_is_exact_endpoint_increment=True,
            responsive_height_is_nonconservative_temporal_product=True,
        )
    old_mapped, old_factors, old_nodes = _integrated_mapped_storage(
        context,
        old,
        nodes,
    )
    new_mapped, new_factors, new_nodes = _integrated_mapped_storage(
        context,
        new,
        nodes,
    )
    endpoint_increment = new_mapped - old_mapped
    mapped_path = np.zeros_like(endpoint_increment)
    height_path = np.zeros_like(endpoint_increment)
    direction = new - old
    temporal_nodes, temporal_weights = np.polynomial.legendre.leggauss(order)
    path_factors = [old_factors, new_factors]
    affine_defects = []
    endpoint_affine_branch = bool(
        context.spatial_reconstruction
        in {
            "piecewise_constant",
            "plm_unlimited",
            "quadratic_admissible",
        }
        and np.array_equal(old_factors, np.ones_like(old_factors))
        and np.array_equal(new_factors, np.ones_like(new_factors))
    )
    exact_node_direction = new_nodes - old_nodes
    uses_exact_affine_derivative = endpoint_affine_branch

    for node, temporal_weight in zip(
        temporal_nodes,
        temporal_weights,
        strict=True,
    ):
        fraction = 0.5 * (float(node) + 1.0)
        weight_t = 0.5 * float(temporal_weight)
        center = old + fraction * direction
        sampled_center_nodes, center_factors = _node_charts(
            context,
            center,
            nodes,
        )
        path_factors.append(center_factors)
        exact_center_nodes = (
            old_nodes + fraction * exact_node_direction
        )
        if endpoint_affine_branch:
            affine_defects.append(
                _relative_norm(
                    sampled_center_nodes - exact_center_nodes,
                    sampled_center_nodes,
                    exact_center_nodes,
                )
            )
        use_exact_at_node = bool(
            endpoint_affine_branch
            and np.array_equal(
                center_factors,
                np.ones_like(center_factors),
            )
        )
        if use_exact_at_node:
            center_nodes = exact_center_nodes
            node_direction = exact_node_direction
        else:
            uses_exact_affine_derivative = False
            plus = center + step * direction
            minus = center - step * direction
            plus_nodes, plus_factors = _node_charts(
                context,
                plus,
                nodes,
            )
            minus_nodes, minus_factors = _node_charts(
                context,
                minus,
                nodes,
            )
            path_factors.extend((plus_factors, minus_factors))
            center_nodes = sampled_center_nodes
            node_direction = (plus_nodes - minus_nodes) / (2.0 * step)

        flat = 0
        for cell, cell_nodes in enumerate(nodes):
            for radius, weight_r in cell_nodes:
                local = causal_five_field_analytic_local_maps(
                    context,
                    radius,
                    center_nodes[flat],
                )
                mapped_path[cell] += (
                    weight_t
                    * weight_r
                    * (
                        local.mapped_conserved_jacobian
                        @ node_direction[flat]
                    )
                )
                height_path[cell] += (
                    weight_t
                    * weight_r
                    * (
                        local.vertical_storage_matrix
                        @ node_direction[flat]
                    )
                )
                flat += 1

    mapped_scale = max(
        float(np.linalg.norm(endpoint_increment)),
        float(np.linalg.norm(mapped_path)),
        np.finfo(float).tiny,
    )
    factors = np.asarray(path_factors, dtype=float)
    return CausalFiveFieldMonolithicStorageIncrement(
        old_primitive_charts=np.array(old, copy=True),
        new_primitive_charts=np.array(new, copy=True),
        mapped_endpoint_increment=np.asarray(
            endpoint_increment,
            dtype=float,
        ),
        mapped_path_increment=np.asarray(mapped_path, dtype=float),
        responsive_height_path_increment=np.asarray(
            height_path,
            dtype=float,
        ),
        total_storage_increment=np.asarray(
            endpoint_increment + height_path,
            dtype=float,
        ),
        temporal_quadrature_order=order,
        reconstruction_directional_step=step,
        maximum_mapped_path_closure_defect=float(
            np.linalg.norm(endpoint_increment - mapped_path)
            / mapped_scale
        ),
        maximum_affine_reconstruction_path_defect=float(
            max(affine_defects, default=0.0)
        ),
        maximum_path_reconstruction_factor_change=float(
            np.max(np.ptp(factors, axis=0))
        ),
        minimum_path_reconstruction_factor=float(np.min(factors)),
        maximum_path_reconstruction_factor=float(np.max(factors)),
        one_flux_reconstruction_for_space_and_storage=True,
        uses_exact_affine_reconstruction_path_derivative=(
            uses_exact_affine_derivative
        ),
        mapped_storage_is_exact_endpoint_increment=True,
        responsive_height_is_nonconservative_temporal_product=True,
    )


def causal_five_field_monolithic_storage_rate(
    context: CausalFiveFieldDAEContext,
    old_primitive_charts: np.ndarray,
    new_primitive_charts: np.ndarray,
    primitive_rate_per_s: np.ndarray,
    *,
    temporal_quadrature_order: int = 4,
) -> CausalFiveFieldMonolithicStorageRate:
    """Evaluate the temporal path action directly in rate coordinates.

    This is algebraically identical to dividing the path increment by the
    timestep, but avoids forming and then dividing a tiny increment during
    the exact small-step KKT audit.
    """

    from .causal_inner_monolithic_tangent import (  # local cycle break
        _node_reconstruction_weights,
    )

    context = context.validated()
    old = _validate_charts(context, old_primitive_charts)
    new = _validate_charts(context, new_primitive_charts)
    rate = np.asarray(primitive_rate_per_s, dtype=float)
    order = int(temporal_quadrature_order)
    if (
        rate.shape != old.shape
        or np.any(~np.isfinite(rate))
        or order != temporal_quadrature_order
        or not 2 <= order <= 12
    ):
        raise ValueError("monolithic storage-rate inputs are invalid")
    mapped_rate = np.zeros_like(old)
    height_rate = np.zeros_like(old)
    direction = new - old
    temporal_nodes, temporal_weights = np.polynomial.legendre.leggauss(order)
    reconstruction_defects = []
    partition_defects = []
    for node, temporal_weight in zip(
        temporal_nodes,
        temporal_weights,
        strict=True,
    ):
        fraction = 0.5 * (float(node) + 1.0)
        weight_t = 0.5 * float(temporal_weight)
        center = old + fraction * direction
        (
            node_weights,
            node_cells,
            node_radii,
            node_measures,
            reconstruction_defect,
            partition_defect,
        ) = _node_reconstruction_weights(context, center)
        reconstruction_defects.append(float(reconstruction_defect))
        partition_defects.append(float(partition_defect))
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
                np.asarray(weights @ center, dtype=float),
            )
            node_rate = np.asarray(weights @ rate, dtype=float)
            mapped_rate[int(cell)] += (
                weight_t
                * float(measure)
                * (local.mapped_conserved_jacobian @ node_rate)
            )
            height_rate[int(cell)] += (
                weight_t
                * float(measure)
                * (local.vertical_storage_matrix @ node_rate)
            )
    return CausalFiveFieldMonolithicStorageRate(
        mapped_rate_per_s=np.asarray(mapped_rate, dtype=float),
        responsive_height_rate_per_s=np.asarray(height_rate, dtype=float),
        maximum_node_reconstruction_relative_defect=max(
            reconstruction_defects,
            default=0.0,
        ),
        maximum_node_partition_of_unity_defect=max(
            partition_defects,
            default=0.0,
        ),
    )


def _center_broken_paths(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    stationary: CausalFiveFieldRadialCandidateLedger,
    *,
    quadrature_order: int,
    relative_step: float,
) -> tuple[
    tuple[CausalFiveFieldRadialPathJump, ...],
    tuple[CausalFiveFieldRadialPathJump, ...],
]:
    left_paths = []
    right_paths = []
    reconstruction = stationary.reconstruction
    for cell, center_chart in enumerate(primitive_charts):
        left_paths.append(
            causal_five_field_radial_extended_path_jump(
                context,
                float(context.grid.edges[cell]),
                float(context.grid.centers[cell]),
                reconstruction.right_face_charts[cell],
                center_chart,
                quadrature_order=quadrature_order,
                relative_step=relative_step,
            )
        )
        right_paths.append(
            causal_five_field_radial_extended_path_jump(
                context,
                float(context.grid.centers[cell]),
                float(context.grid.edges[cell + 1]),
                center_chart,
                reconstruction.left_face_charts[cell + 1],
                quadrature_order=quadrature_order,
                relative_step=relative_step,
            )
        )
    return tuple(left_paths), tuple(right_paths)


def evaluate_causal_five_field_monolithic_backward_euler(
    old_primitive_charts: np.ndarray,
    new_primitive_charts: np.ndarray,
    timestep_seconds: float,
    context: CausalFiveFieldDAEContext,
    *,
    temporal_quadrature_order: int = 4,
    reconstruction_directional_step: float = 1.0e-2,
    path_quadrature_order: int = 6,
    relative_step: float = DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
    stationary_speed_tolerance: float = 1.0e-12,
) -> CausalFiveFieldMonolithicDAEEvaluation:
    """Evaluate one production-neutral monolithic backward-Euler residual."""

    context = context.validated()
    old = _validate_charts(context, old_primitive_charts)
    new = _validate_charts(context, new_primitive_charts)
    timestep = float(timestep_seconds)
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("monolithic timestep must be positive and finite")

    storage = causal_five_field_monolithic_storage_increment(
        context,
        old,
        new,
        temporal_quadrature_order=temporal_quadrature_order,
        reconstruction_directional_step=reconstruction_directional_step,
    )
    stationary = causal_five_field_radial_candidate_ledger(
        context,
        new,
        quadrature_order=path_quadrature_order,
        relative_step=relative_step,
        stationary_speed_tolerance=stationary_speed_tolerance,
    )
    left_paths, right_paths = _center_broken_paths(
        context,
        new,
        stationary,
        quadrature_order=path_quadrature_order,
        relative_step=relative_step,
    )

    shear = np.array(stationary.shear_principal_rows, copy=True)
    height = np.array(stationary.height_principal_rows, copy=True)
    path_adjustment = np.zeros_like(shear)
    for cell, (full, left, right) in enumerate(
        zip(
            stationary.within_cell_paths,
            left_paths,
            right_paths,
            strict=True,
        )
    ):
        shear_adjustment = (
            full.shear_source_path_integral_over_c
            - left.shear_source_path_integral_over_c
            - right.shear_source_path_integral_over_c
        )
        height_adjustment = (
            full.vertical_source_path_integral_over_c
            - left.vertical_source_path_integral_over_c
            - right.vertical_source_path_integral_over_c
        )
        shear[cell] += shear_adjustment
        height[cell] += height_adjustment
        path_adjustment[cell] = shear_adjustment + height_adjustment

    coordinate_timestep = C * timestep
    mapped_temporal = (
        storage.mapped_endpoint_increment / coordinate_timestep
    )
    height_temporal = (
        storage.responsive_height_path_increment / coordinate_timestep
    )
    blocks = (
        mapped_temporal,
        height_temporal,
        stationary.conservative_transport_rows,
        shear,
        height,
        stationary.local_stress_relaxation_rows,
        stationary.geometry_rows,
        stationary.cooling_rows,
        stationary.stream_rows,
        stationary.lower_height_work_rows,
    )
    residual = np.sum(np.asarray(blocks), axis=0)
    reconstructed = (
        mapped_temporal
        + height_temporal
        + stationary.conservative_transport_rows
        + shear
        + height
        + stationary.local_stress_relaxation_rows
        + stationary.geometry_rows
        + stationary.cooling_rows
        + stationary.stream_rows
        + stationary.lower_height_work_rows
    )
    block_scale = max(
        float(np.max(np.abs(residual))),
        max(float(np.max(np.abs(block))) for block in blocks),
        np.finfo(float).tiny,
    )
    stationary_scale = max(
        float(np.max(np.abs(stationary.residual_rows))),
        np.finfo(float).tiny,
    )
    return CausalFiveFieldMonolithicDAEEvaluation(
        storage_increment=storage,
        stationary_ledger=stationary,
        mapped_temporal_storage_rows=np.asarray(
            mapped_temporal,
            dtype=float,
        ),
        responsive_height_temporal_storage_rows=np.asarray(
            height_temporal,
            dtype=float,
        ),
        conservative_transport_rows=np.asarray(
            stationary.conservative_transport_rows,
            dtype=float,
        ),
        shear_principal_rows=np.asarray(shear, dtype=float),
        height_principal_rows=np.asarray(height, dtype=float),
        local_stress_relaxation_rows=np.asarray(
            stationary.local_stress_relaxation_rows,
            dtype=float,
        ),
        geometry_rows=np.asarray(stationary.geometry_rows, dtype=float),
        cooling_rows=np.asarray(stationary.cooling_rows, dtype=float),
        stream_rows=np.asarray(stationary.stream_rows, dtype=float),
        lower_height_work_rows=np.asarray(
            stationary.lower_height_work_rows,
            dtype=float,
        ),
        residual_rows=np.asarray(residual, dtype=float),
        center_broken_path_adjustment_rows=np.asarray(
            path_adjustment,
            dtype=float,
        ),
        inner_left_half_path=left_paths[0],
        inner_right_half_path=right_paths[0],
        maximum_block_ledger_defect=float(
            np.max(np.abs(residual - reconstructed)) / block_scale
        ),
        maximum_center_broken_path_adjustment=float(
            np.max(np.abs(path_adjustment)) / stationary_scale
        ),
        incoming_excision_characteristics=(
            stationary.interfaces.incoming_excision_characteristics
        ),
        uses_production_generator=False,
        uses_production_anchor_storage_derivative=False,
    )
