"""Exact finite-amplitude coordinate-fiber lifting utilities.

The routines in this module are deliberately independent of the causal disk
physics.  They operate in a frozen scaled primitive chart and use a caller
supplied nonlinear coordinate evaluator.  This keeps the WP10c8o lift an
audit-only construction: production residuals, fluxes, and time integrators
are not modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares


CoordinateEvaluator = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CausalWeightedConstraintNormalBasis:
    """Weighted-normal basis for one full-row-rank constraint matrix."""

    basis: np.ndarray
    singular_values: np.ndarray
    numerical_rank: int
    condition_estimate: float
    weighted_orthogonality_defect: float
    row_space_reconstruction_defect: float


@dataclass(frozen=True)
class CausalExactCoordinateLift:
    """One signed, nonlinearly corrected equal-coordinate lift."""

    primitive_vector: np.ndarray
    scaled_increment: np.ndarray
    provisional_scaled_increment: np.ndarray
    coordinate_values: np.ndarray
    normalized_coordinate_residual: np.ndarray
    correction_coordinates: np.ndarray
    weighted_radius: float
    provisional_weighted_radius: float
    maximum_pointwise_amplitude_ratio: float
    provisional_maximum_pointwise_amplitude_ratio: float
    correction_fraction: float
    retained_seed_multiplier: float
    retained_seed_multiplier_defect: float
    weighted_direction_cosine: float
    function_evaluations: int
    jacobian_evaluations: int | None
    optimizer_status: int
    optimizer_message: str
    optimizer_success: bool

    @property
    def maximum_coordinate_defect(self) -> float:
        return float(np.max(np.abs(self.normalized_coordinate_residual)))


@dataclass(frozen=True)
class CausalExactCoordinateLiftPair:
    """Two opposite lifts corrected to the same exact coordinate target."""

    minus: CausalExactCoordinateLift
    plus: CausalExactCoordinateLift
    projected_seed_direction: np.ndarray
    normal_basis: CausalWeightedConstraintNormalBasis
    pairwise_normalized_coordinate_difference: np.ndarray

    @property
    def maximum_pairwise_coordinate_defect(self) -> float:
        return float(
            np.max(
                np.abs(self.pairwise_normalized_coordinate_difference)
            )
        )


def causal_rescale_descriptor_matrix(
    matrix: np.ndarray,
    *,
    source_primitive_scales: np.ndarray,
    source_conservation_scales: np.ndarray,
    target_primitive_scales: np.ndarray,
    target_conservation_scales: np.ndarray,
) -> np.ndarray:
    """Express one scaled descriptor matrix in another frozen chart.

    If ``M_s = R_s^-1 M_physical C_s``, the returned matrix is
    ``R_t^-1 M_physical C_t``.  This is used to compare an independently
    scaled full-DAE/Schur descriptor with the frozen WP10c8i audit chart.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("descriptor matrix must be finite and square")
    size = values.shape[0]
    source_columns = _finite_vector(
        source_primitive_scales,
        name="source primitive scales",
        shape=(size,),
    )
    source_rows = _finite_vector(
        source_conservation_scales,
        name="source conservation scales",
        shape=(size,),
    )
    target_columns = _finite_vector(
        target_primitive_scales,
        name="target primitive scales",
        shape=(size,),
    )
    target_rows = _finite_vector(
        target_conservation_scales,
        name="target conservation scales",
        shape=(size,),
    )
    if (
        np.any(~np.isfinite(values))
        or np.any(source_columns <= 0.0)
        or np.any(source_rows <= 0.0)
        or np.any(target_columns <= 0.0)
        or np.any(target_rows <= 0.0)
    ):
        raise ValueError("descriptor matrix or scales are invalid")
    return (
        (source_rows / target_rows)[:, None]
        * values
        * (target_columns / source_columns)[None, :]
    )


def _finite_vector(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        (shape is not None and array.shape != shape)
        or array.ndim != 1
        or np.any(~np.isfinite(array))
    ):
        raise ValueError(f"{name} has an invalid shape or value")
    return array


def causal_weighted_constraint_normal_basis(
    constraint_matrix: np.ndarray,
    state_weights: np.ndarray,
    *,
    rank_relative_tolerance: float | None = None,
) -> CausalWeightedConstraintNormalBasis:
    """Return a ``W``-orthonormal basis for the constraint normal space.

    If ``C`` is the constraint matrix and ``W`` is diagonal, the returned
    columns ``Q`` satisfy ``Q.T @ W @ Q = I`` and span
    ``range(W^-1 C.T)``.  Consequently adding ``Q a`` to a tangent-null seed
    changes only its weighted-normal component.
    """

    constraints = np.asarray(constraint_matrix, dtype=float)
    weights = np.asarray(state_weights, dtype=float)
    if (
        constraints.ndim != 2
        or constraints.shape[0] == 0
        or constraints.shape[1] == 0
        or weights.shape != (constraints.shape[1],)
        or np.any(~np.isfinite(constraints))
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("weighted constraint inputs are invalid")
    weighted_rows = constraints / np.sqrt(weights)[None, :]
    _left, singular_values, right_t = np.linalg.svd(
        weighted_rows,
        full_matrices=False,
    )
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise ValueError("constraint matrix has no active row")
    tolerance = (
        max(weighted_rows.shape)
        * np.finfo(float).eps
        * singular_values[0]
        if rank_relative_tolerance is None
        else float(rank_relative_tolerance) * singular_values[0]
    )
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("constraint rank tolerance is invalid")
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank != constraints.shape[0]:
        raise ValueError("constraint matrix is not full row rank")
    basis = right_t[:rank].T / np.sqrt(weights)[:, None]
    weighted_gram = basis.T @ (weights[:, None] * basis)
    orthogonality_defect = float(
        np.max(np.abs(weighted_gram - np.eye(rank)))
    )
    projector = basis @ (
        basis.T @ (weights[:, None] * np.eye(weights.size))
    )
    weighted_row_projector = (
        np.sqrt(weights)[:, None]
        * projector
        / np.sqrt(weights)[None, :]
    )
    reference_projector = right_t[:rank].T @ right_t[:rank]
    reconstruction_defect = float(
        np.max(np.abs(weighted_row_projector - reference_projector))
    )
    return CausalWeightedConstraintNormalBasis(
        basis=basis,
        singular_values=singular_values,
        numerical_rank=rank,
        condition_estimate=float(singular_values[0] / singular_values[-1]),
        weighted_orthogonality_defect=orthogonality_defect,
        row_space_reconstruction_defect=reconstruction_defect,
    )


def causal_weighted_constraint_fiber_null_projection(
    direction: np.ndarray,
    state_weights: np.ndarray,
    normal_basis: CausalWeightedConstraintNormalBasis,
) -> np.ndarray:
    """Project one direction onto the weighted constraint-null space."""

    weights = _finite_vector(state_weights, name="state weights")
    vector = _finite_vector(
        direction,
        name="candidate fiber direction",
        shape=weights.shape,
    )
    basis = np.asarray(normal_basis.basis, dtype=float)
    if basis.shape[0] != vector.size:
        raise ValueError("normal basis and direction dimensions differ")
    normal_coordinates = basis.T @ (weights * vector)
    projected = vector - basis @ normal_coordinates
    if np.any(~np.isfinite(projected)):
        raise ValueError("constraint-null projection is not finite")
    return projected


def _weighted_norm(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * np.square(values))))


def _exact_coordinate_lift(
    *,
    base_primitive_vector: np.ndarray,
    primitive_column_scales: np.ndarray,
    state_weights: np.ndarray,
    physical_input_amplitudes: np.ndarray,
    target_coordinate_values: np.ndarray,
    target_coordinate_scales: np.ndarray,
    projected_seed_direction: np.ndarray,
    seed_multiplier: float,
    sign: int,
    normal_basis: CausalWeightedConstraintNormalBasis,
    correction_jacobian: np.ndarray,
    coordinate_evaluator: CoordinateEvaluator,
    maximum_function_evaluations: int,
    optimizer_tolerance: float,
) -> CausalExactCoordinateLift:
    base = _finite_vector(base_primitive_vector, name="base primitive vector")
    scales = _finite_vector(
        primitive_column_scales,
        name="primitive column scales",
        shape=base.shape,
    )
    weights = _finite_vector(
        state_weights,
        name="state weights",
        shape=base.shape,
    )
    amplitudes = _finite_vector(
        physical_input_amplitudes,
        name="physical input amplitudes",
        shape=base.shape,
    )
    target = _finite_vector(
        target_coordinate_values,
        name="target coordinate values",
    )
    coordinate_scales = _finite_vector(
        target_coordinate_scales,
        name="target coordinate scales",
        shape=target.shape,
    )
    seed = _finite_vector(
        projected_seed_direction,
        name="projected seed direction",
        shape=base.shape,
    )
    if (
        np.any(scales <= 0.0)
        or np.any(weights <= 0.0)
        or np.any(amplitudes <= 0.0)
        or np.any(coordinate_scales <= 0.0)
        or sign not in (-1, 1)
        or not np.isfinite(seed_multiplier)
        or seed_multiplier <= 0.0
    ):
        raise ValueError("equal-coordinate lift scales are invalid")
    basis = np.asarray(normal_basis.basis, dtype=float)
    if basis.shape != (base.size, target.size):
        raise ValueError("normal-basis dimension does not match coordinates")
    jacobian = np.asarray(correction_jacobian, dtype=float)
    if (
        jacobian.shape != (target.size, target.size)
        or np.any(~np.isfinite(jacobian))
        or np.linalg.matrix_rank(jacobian) != target.size
    ):
        raise ValueError("normal correction Jacobian is invalid")
    seed_norm_squared = float(np.sum(weights * seed * seed))
    if not np.isfinite(seed_norm_squared) or seed_norm_squared <= 0.0:
        raise ValueError("projected seed direction has zero weighted norm")
    provisional = float(sign) * float(seed_multiplier) * seed

    def trial(correction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scaled_increment = provisional + basis @ correction
        primitives = base + scales * scaled_increment
        coordinates = _finite_vector(
            coordinate_evaluator(primitives),
            name="evaluated nonlinear coordinates",
            shape=target.shape,
        )
        return scaled_increment, coordinates

    def residual(correction: np.ndarray) -> np.ndarray:
        _increment, coordinates = trial(correction)
        return (coordinates - target) / coordinate_scales

    solution = least_squares(
        residual,
        np.zeros(target.size, dtype=float),
        # The anchor constraint matrix is the derivative of the normalized
        # coordinate map in this frozen scaled chart.  Its product with the
        # weighted normal basis is therefore the exact anchor Jacobian of the
        # 34-dimensional corrector.  Keeping that matrix fixed avoids a dense
        # 3-point re-differentiation of the full conservative state map at
        # every trust-region iteration.  The final finite-state coordinate
        # residual, not this local model, remains the binding gate.
        jac=lambda _correction: jacobian,
        method="trf",
        ftol=float(optimizer_tolerance),
        xtol=float(optimizer_tolerance),
        gtol=float(optimizer_tolerance),
        max_nfev=int(maximum_function_evaluations),
        x_scale="jac",
    )
    increment, coordinates = trial(solution.x)
    normalized_residual = (coordinates - target) / coordinate_scales
    physical_increment = scales * increment
    provisional_physical = scales * provisional
    weighted_radius = _weighted_norm(increment, weights)
    provisional_radius = _weighted_norm(provisional, weights)
    correction_radius = _weighted_norm(basis @ solution.x, weights)
    retained_multiplier = float(
        np.sum(weights * seed * increment) / seed_norm_squared
    )
    cosine = float(
        sign
        * np.sum(weights * seed * increment)
        / np.sqrt(seed_norm_squared)
        / max(weighted_radius, np.finfo(float).tiny)
    )
    return CausalExactCoordinateLift(
        primitive_vector=base + physical_increment,
        scaled_increment=increment,
        provisional_scaled_increment=provisional,
        coordinate_values=coordinates,
        normalized_coordinate_residual=normalized_residual,
        correction_coordinates=np.asarray(solution.x, dtype=float),
        weighted_radius=weighted_radius,
        provisional_weighted_radius=provisional_radius,
        maximum_pointwise_amplitude_ratio=float(
            np.max(np.abs(physical_increment) / amplitudes)
        ),
        provisional_maximum_pointwise_amplitude_ratio=float(
            np.max(np.abs(provisional_physical) / amplitudes)
        ),
        correction_fraction=float(
            correction_radius
            / max(provisional_radius, np.finfo(float).tiny)
        ),
        retained_seed_multiplier=retained_multiplier,
        retained_seed_multiplier_defect=float(
            abs(retained_multiplier - sign * seed_multiplier)
        ),
        weighted_direction_cosine=cosine,
        function_evaluations=int(solution.nfev),
        jacobian_evaluations=(
            None if solution.njev is None else int(solution.njev)
        ),
        optimizer_status=int(solution.status),
        optimizer_message=str(solution.message),
        optimizer_success=bool(solution.success),
    )


def causal_exact_equal_coordinate_lift_pair(
    *,
    base_primitive_vector: np.ndarray,
    primitive_column_scales: np.ndarray,
    state_weights: np.ndarray,
    physical_input_amplitudes: np.ndarray,
    target_coordinate_values: np.ndarray,
    target_coordinate_scales: np.ndarray,
    constraint_matrix: np.ndarray,
    seed_direction: np.ndarray,
    seed_multiplier: float,
    coordinate_evaluator: CoordinateEvaluator,
    maximum_function_evaluations: int = 256,
    optimizer_tolerance: float = 1.0e-13,
) -> CausalExactCoordinateLiftPair:
    """Construct an exact opposite pair on one nonlinear coordinate fiber.

    ``constraint_matrix`` must be the anchor Jacobian of the coordinate map
    *after* division by ``target_coordinate_scales``, with respect to the
    frozen scaled primitive chart.  In the moment audit this is the level's
    normalized ``constraint_matrix`` rather than the raw coordinate rows.

    The supplied seed may have a small constraint-normal component; it is
    projected onto the anchor's weighted null space before either lift is
    formed.  The nonlinear corrector then moves only within the weighted
    normal space, so the signed seed projection is preserved exactly up to
    solver roundoff.
    """

    constraints = np.asarray(constraint_matrix, dtype=float)
    target = _finite_vector(
        target_coordinate_values,
        name="target coordinate values",
    )
    if constraints.ndim != 2 or constraints.shape[0] != target.size:
        raise ValueError("constraint and coordinate dimensions differ")
    normal = causal_weighted_constraint_normal_basis(
        constraints,
        state_weights,
    )
    projected = causal_weighted_constraint_fiber_null_projection(
        seed_direction,
        state_weights,
        normal,
    )
    correction_jacobian = constraints @ normal.basis
    minus = _exact_coordinate_lift(
        base_primitive_vector=base_primitive_vector,
        primitive_column_scales=primitive_column_scales,
        state_weights=state_weights,
        physical_input_amplitudes=physical_input_amplitudes,
        target_coordinate_values=target,
        target_coordinate_scales=target_coordinate_scales,
        projected_seed_direction=projected,
        seed_multiplier=seed_multiplier,
        sign=-1,
        normal_basis=normal,
        correction_jacobian=correction_jacobian,
        coordinate_evaluator=coordinate_evaluator,
        maximum_function_evaluations=maximum_function_evaluations,
        optimizer_tolerance=optimizer_tolerance,
    )
    plus = _exact_coordinate_lift(
        base_primitive_vector=base_primitive_vector,
        primitive_column_scales=primitive_column_scales,
        state_weights=state_weights,
        physical_input_amplitudes=physical_input_amplitudes,
        target_coordinate_values=target,
        target_coordinate_scales=target_coordinate_scales,
        projected_seed_direction=projected,
        seed_multiplier=seed_multiplier,
        sign=1,
        normal_basis=normal,
        correction_jacobian=correction_jacobian,
        coordinate_evaluator=coordinate_evaluator,
        maximum_function_evaluations=maximum_function_evaluations,
        optimizer_tolerance=optimizer_tolerance,
    )
    pair_difference = (
        plus.coordinate_values - minus.coordinate_values
    ) / np.asarray(target_coordinate_scales, dtype=float)
    return CausalExactCoordinateLiftPair(
        minus=minus,
        plus=plus,
        projected_seed_direction=projected,
        normal_basis=normal,
        pairwise_normalized_coordinate_difference=pair_difference,
    )


def causal_gate_normalized_pair_half_spread(
    minus_values: np.ndarray,
    plus_values: np.ndarray,
    gates: np.ndarray,
) -> np.ndarray:
    """Return ``abs(plus-minus)/(2*gate)`` element by element."""

    minus = _finite_vector(minus_values, name="minus output values")
    plus = _finite_vector(
        plus_values,
        name="plus output values",
        shape=minus.shape,
    )
    gate_values = _finite_vector(
        gates,
        name="output gates",
        shape=minus.shape,
    )
    if np.any(gate_values <= 0.0):
        raise ValueError("output gates must be positive")
    return 0.5 * np.abs(plus - minus) / gate_values
