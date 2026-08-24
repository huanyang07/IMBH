"""Metric-preconditioned local charts for a conservative coordinate map.

The original coordinate is retained as the physical observable.  The metric
chart applies an invertible, block-diagonal row transformation only to the
numerical residual and its Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


Array = np.ndarray


def _finite(value, *, ndim: int, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result.copy()


def _relative(left: Array, right: Array) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(a - b)
        / max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), np.finfo(float).tiny)
    )


def _broyden_update(matrix: Array, step: Array, residual_change: Array) -> Array:
    direction = np.asarray(step, dtype=float)
    denominator = float(direction @ direction)
    if denominator <= np.finfo(float).tiny:
        return np.asarray(matrix, dtype=float)
    return np.asarray(matrix, dtype=float) + np.outer(
        np.asarray(residual_change) - np.asarray(matrix) @ direction,
        direction,
    ) / denominator


def block_whitening_transform(
    jacobian: Array, block_sizes: tuple[int, ...]
) -> tuple[Array, dict]:
    """Return independent inverse-square-root row-Gram block whitening."""
    matrix = _finite(jacobian, ndim=2, name="jacobian")
    sizes = tuple(int(value) for value in block_sizes)
    if any(value <= 0 for value in sizes) or sum(sizes) != matrix.shape[0]:
        raise ValueError("block sizes do not partition the Jacobian rows")
    transform = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=float)
    closures = []
    block_conditions = []
    offset = 0
    for size in sizes:
        stop = offset + size
        block = matrix[offset:stop]
        left, singular, _right = np.linalg.svd(block, full_matrices=False)
        if len(singular) != size or singular[-1] <= np.finfo(float).tiny:
            raise ValueError("coordinate block lost row rank")
        local = (left * (1.0 / singular)[None, :]) @ left.T
        whitened = local @ block
        transform[offset:stop, offset:stop] = local
        closures.append(
            float(
                np.linalg.norm(
                    whitened @ whitened.T - np.eye(size), ord=np.inf
                )
            )
        )
        block_conditions.append(float(singular[0] / singular[-1]))
        offset = stop
    metric = transform @ matrix
    singular = np.linalg.svd(metric, compute_uv=False)
    return transform, {
        "block_sizes": sizes,
        "block_condition_numbers": tuple(block_conditions),
        "block_whitening_closure_defects": tuple(closures),
        "maximum_block_whitening_closure_defect": float(max(closures)),
        "metric_jacobian_condition_number": float(singular[0] / singular[-1]),
        "metric_jacobian_minimum_singular_value": float(singular[-1]),
        "metric_jacobian_maximum_singular_value": float(singular[0]),
    }


@dataclass(frozen=True)
class ConservativeMetricChart:
    """An invertible numerical metric over an unchanged physical coordinate."""

    anchor_coordinate: Array
    transform: Array
    block_sizes: tuple[int, ...]
    tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        anchor = _finite(self.anchor_coordinate, ndim=1, name="anchor_coordinate")
        transform = _finite(self.transform, ndim=2, name="transform")
        if transform.shape != (len(anchor), len(anchor)):
            raise ValueError("metric transform has the wrong shape")
        sizes = tuple(int(value) for value in self.block_sizes)
        if any(value <= 0 for value in sizes) or sum(sizes) != len(anchor):
            raise ValueError("metric blocks do not partition the coordinate")
        tolerance = float(self.tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("metric chart tolerance must be positive")
        inverse = np.linalg.inv(transform)
        inverse_closure = float(
            max(
                np.linalg.norm(transform @ inverse - np.eye(len(anchor)), ord=np.inf),
                np.linalg.norm(inverse @ transform - np.eye(len(anchor)), ord=np.inf),
            )
        )
        symmetry = float(np.linalg.norm(transform - transform.T, ord=np.inf))
        off_block = transform.copy()
        offset = 0
        for size in sizes:
            stop = offset + size
            off_block[offset:stop, offset:stop] = 0.0
            offset = stop
        off_block_defect = float(np.linalg.norm(off_block, ord=np.inf))
        if (
            inverse_closure > tolerance
            or symmetry > tolerance
            or off_block_defect > tolerance
            or np.min(np.linalg.eigvalsh(transform)) <= 0.0
        ):
            raise ValueError("metric transform is not admissible")
        for value in (anchor, transform, inverse):
            value.setflags(write=False)
        object.__setattr__(self, "anchor_coordinate", anchor)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "block_sizes", sizes)
        object.__setattr__(self, "_inverse", inverse)
        object.__setattr__(self, "inverse_closure_defect", inverse_closure)
        object.__setattr__(self, "symmetry_defect", symmetry)
        object.__setattr__(self, "off_block_defect", off_block_defect)

    @property
    def dimension(self) -> int:
        return int(len(self.anchor_coordinate))

    @property
    def inverse_transform(self) -> Array:
        return self._inverse

    def encode(self, original_coordinate: Array) -> Array:
        value = _finite(
            original_coordinate, ndim=1, name="original_coordinate"
        )
        if value.shape != self.anchor_coordinate.shape:
            raise ValueError("original coordinate has the wrong shape")
        return self.transform @ (value - self.anchor_coordinate)

    def decode(self, metric_coordinate: Array) -> Array:
        value = _finite(metric_coordinate, ndim=1, name="metric_coordinate")
        if value.shape != self.anchor_coordinate.shape:
            raise ValueError("metric coordinate has the wrong shape")
        return self.anchor_coordinate + self.inverse_transform @ value

    def push_rate(self, original_rate: Array) -> Array:
        value = _finite(original_rate, ndim=1, name="original_rate")
        if value.shape != self.anchor_coordinate.shape:
            raise ValueError("original rate has the wrong shape")
        return self.transform @ value

    def pull_rate(self, metric_rate: Array) -> Array:
        value = _finite(metric_rate, ndim=1, name="metric_rate")
        if value.shape != self.anchor_coordinate.shape:
            raise ValueError("metric rate has the wrong shape")
        return self.inverse_transform @ value

    def transform_jacobian(self, original_jacobian: Array) -> Array:
        value = _finite(
            original_jacobian, ndim=2, name="original_jacobian"
        )
        if value.shape[0] != self.dimension:
            raise ValueError("original Jacobian has the wrong row count")
        return self.transform @ value


@dataclass(frozen=True)
class MetricRetractionPolicy:
    maximum_iterations: int
    refresh_iteration_reserve: int
    maximum_exact_refreshes: int
    line_factors: tuple[float, ...]
    original_coordinate_tolerance: float
    metric_coordinate_tolerance: float
    gauge_tolerance: float
    maximum_anchor_departure: float
    maximum_metric_augmented_condition: float


def metric_augmented_jacobian(
    original_coordinate_jacobian: Array,
    gauge_basis: Array,
    chart: ConservativeMetricChart,
) -> tuple[Array, float]:
    coordinate = chart.transform_jacobian(original_coordinate_jacobian)
    gauge = _finite(gauge_basis, ndim=2, name="gauge_basis")
    if gauge.shape[0] != original_coordinate_jacobian.shape[1]:
        raise ValueError("gauge basis has the wrong physical dimension")
    augmented = np.vstack((coordinate, gauge.T))
    return augmented, float(np.linalg.cond(augmented))


def _metric_residual(raw: Array, chart: ConservativeMetricChart) -> Array:
    value = np.asarray(raw, dtype=float)
    return np.concatenate(
        (chart.transform @ value[: chart.dimension], value[chart.dimension :])
    )


def metric_transport_retract(
    *,
    exact_chart,
    model,
    initial_state: Array,
    target_original_coordinate: Array,
    gauge_basis: Array,
    anchor_delta: Array,
    anchor_metric_augmented: Array,
    chart: ConservativeMetricChart,
    policy: MetricRetractionPolicy,
) -> tuple[Array, Array, dict]:
    """Retract with a metric residual while auditing the original residual."""
    state = np.asarray(initial_state, dtype=float).copy()
    matrix = np.asarray(anchor_metric_augmented, dtype=float).copy()
    residual_history = []
    line_factors = []
    refreshes = 0
    corrections = 0
    condition_numbers = [float(np.linalg.cond(matrix))]
    began = time.perf_counter()

    def residuals(candidate: Array) -> tuple[Array, Array, Array]:
        raw, factors = exact_chart._residual(
            model,
            candidate,
            target_original_coordinate,
            gauge_basis,
            anchor_delta,
        )
        return np.asarray(raw), _metric_residual(raw, chart), factors

    def norms(raw: Array, metric: Array) -> tuple[float, float, float, float]:
        original_inf = float(np.max(np.abs(raw[: chart.dimension])))
        metric_inf = float(np.max(np.abs(metric[: chart.dimension])))
        gauge_inf = float(np.max(np.abs(raw[chart.dimension :])))
        return original_inf, metric_inf, gauge_inf, max(
            original_inf, metric_inf, gauge_inf
        )

    def exact_metric_matrix(candidate: Array) -> tuple[Array, float, bool]:
        original, metrics = exact_chart._augmented_jacobian(
            model, candidate, gauge_basis
        )
        transformed = np.asarray(original, dtype=float).copy()
        transformed[: chart.dimension] = (
            chart.transform @ transformed[: chart.dimension]
        )
        condition = float(np.linalg.cond(transformed))
        rank_ok = bool(
            metrics["augmented_rank"] == exact_chart.PHYSICAL_DIMENSION
            and np.linalg.matrix_rank(transformed) == exact_chart.PHYSICAL_DIMENSION
        )
        return transformed, condition, rank_ok

    iteration = 0
    while iteration <= policy.maximum_iterations:
        raw, metric, factors = residuals(state)
        original_inf, metric_inf, gauge_inf, combined = norms(raw, metric)
        residual_history.append({
            "original_coordinate_infinity": original_inf,
            "metric_coordinate_infinity": metric_inf,
            "gauge_infinity": gauge_inf,
            "combined_infinity": combined,
        })
        if (
            original_inf <= policy.original_coordinate_tolerance
            and metric_inf <= policy.metric_coordinate_tolerance
            and gauge_inf <= policy.gauge_tolerance
        ):
            physical = exact_chart._physical_audit(model, state, factors)
            return state, matrix, {
                "passed": bool(physical["passed"]),
                "original_coordinate_residual_infinity": original_inf,
                "metric_coordinate_residual_infinity": metric_inf,
                "gauge_residual_infinity": gauge_inf,
                "transport_corrections": corrections,
                "target_exact_refreshes": refreshes,
                "accepted_line_factors": line_factors,
                "residual_history": residual_history,
                "maximum_metric_augmented_condition_number": max(condition_numbers),
                "maximum_scaled_anchor_departure": float(
                    np.max(np.abs(exact_chart._delta(model, state) - anchor_delta))
                ),
                "wall_seconds": float(time.perf_counter() - began),
                **physical,
            }
        if iteration == policy.maximum_iterations:
            break
        if (
            iteration
            >= policy.maximum_iterations - policy.refresh_iteration_reserve
            and refreshes < policy.maximum_exact_refreshes
        ):
            matrix, condition, rank_ok = exact_metric_matrix(state)
            refreshes += 1
            condition_numbers.append(condition)
            if (
                not rank_ok
                or condition > policy.maximum_metric_augmented_condition
            ):
                break
        correction = np.linalg.solve(matrix, metric)
        old_delta = exact_chart._delta(model, state)
        accepted = False
        for factor in policy.line_factors:
            proposed_delta = old_delta - factor * correction
            if (
                float(np.max(np.abs(proposed_delta - anchor_delta)))
                > policy.maximum_anchor_departure
            ):
                continue
            proposed = exact_chart._state_from_delta(model, proposed_delta)
            trial_raw, trial_metric, _trial_factors = residuals(proposed)
            _original, _metric, _gauge, trial_combined = norms(
                trial_raw, trial_metric
            )
            if trial_combined < combined:
                step = proposed_delta - old_delta
                matrix = _broyden_update(matrix, step, trial_metric - metric)
                state = proposed
                line_factors.append(float(factor))
                corrections += 1
                accepted = True
                break
        if not accepted:
            if refreshes >= policy.maximum_exact_refreshes:
                break
            matrix, condition, rank_ok = exact_metric_matrix(state)
            refreshes += 1
            condition_numbers.append(condition)
            if (
                not rank_ok
                or condition > policy.maximum_metric_augmented_condition
            ):
                break
            continue
        iteration += 1

    raw, metric, factors = residuals(state)
    original_inf, metric_inf, gauge_inf, _combined = norms(raw, metric)
    physical = exact_chart._physical_audit(model, state, factors)
    return state, matrix, {
        "passed": False,
        "original_coordinate_residual_infinity": original_inf,
        "metric_coordinate_residual_infinity": metric_inf,
        "gauge_residual_infinity": gauge_inf,
        "transport_corrections": corrections,
        "target_exact_refreshes": refreshes,
        "accepted_line_factors": line_factors,
        "residual_history": residual_history,
        "maximum_metric_augmented_condition_number": max(condition_numbers),
        "maximum_scaled_anchor_departure": float(
            np.max(np.abs(exact_chart._delta(model, state) - anchor_delta))
        ),
        "wall_seconds": float(time.perf_counter() - began),
        **physical,
    }
