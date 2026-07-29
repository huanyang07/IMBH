"""Control-volume localization helpers for frozen radial histories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix

from .causal_inner_dae_system import (
    causal_five_field_dae_jacobian_color_groups,
)


_CENTRAL_DERIVATIVE_STENCILS = {
    2: {
        -1: -0.5,
        1: 0.5,
    },
    4: {
        -2: 1.0 / 12.0,
        -1: -2.0 / 3.0,
        1: 2.0 / 3.0,
        2: -1.0 / 12.0,
    },
    6: {
        -3: -1.0 / 60.0,
        -2: 3.0 / 20.0,
        -1: -3.0 / 4.0,
        1: 3.0 / 4.0,
        2: -3.0 / 20.0,
        3: 1.0 / 60.0,
    },
}


@dataclass(frozen=True)
class CausalRadialHistoryConvergence:
    """Three-grid convergence of one multicomponent time history."""

    component_scales: np.ndarray
    component_normalization_scales: np.ndarray
    component_activity_thresholds: np.ndarray
    significant_components: np.ndarray
    component_coarse_medium_differences: np.ndarray
    component_medium_fine_differences: np.ndarray
    component_observed_orders: np.ndarray
    component_history_cosines: np.ndarray
    component_error_cosines: np.ndarray
    # Backward-compatible name used by the immutable WP10c9d5b evidence.
    component_fine_signed_cosines: np.ndarray
    component_passed: np.ndarray
    coarse_medium_difference: float
    medium_fine_difference: float
    observed_order: float | None
    history_cosine: float
    error_cosine: float
    # Backward-compatible name used by the immutable WP10c9d5b evidence.
    fine_signed_cosine: float
    passed: bool


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size < 1 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    """Return a signed cosine with an explicit zero-vector convention."""

    left = np.asarray(first, dtype=float).ravel()
    right = np.asarray(second, dtype=float).ravel()
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    tiny = np.finfo(float).tiny
    if left_norm <= tiny and right_norm <= tiny:
        return 1.0
    if left_norm <= tiny or right_norm <= tiny:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _validated_derivative_orders(orders) -> tuple[int, ...]:
    values = tuple(int(order) for order in orders)
    if (
        not values
        or len(set(values)) != len(values)
        or any(order not in _CENTRAL_DERIVATIVE_STENCILS for order in values)
    ):
        raise ValueError("central derivative orders must be unique 2, 4, or 6")
    return values


def causal_radial_high_order_directional_derivatives(
    function,
    base: np.ndarray,
    direction: np.ndarray,
    *,
    finite_difference_step: float,
    derivative_orders=(4, 6),
) -> dict[int, np.ndarray]:
    """Evaluate several centered high-order JVPs from one shared stencil.

    The returned actions use the same residual samples for every requested
    order. This is intended for frozen audit derivatives whose residual
    evaluation contains cancellation-amplified roundoff; it does not change
    the nonlinear residual itself.
    """

    point = _finite_vector(base, name="base")
    vector = _finite_vector(direction, name="direction")
    step = float(finite_difference_step)
    orders = _validated_derivative_orders(derivative_orders)
    if (
        vector.shape != point.shape
        or not np.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError("high-order directional derivative inputs are invalid")
    multipliers = sorted(
        {
            multiplier
            for order in orders
            for multiplier in _CENTRAL_DERIVATIVE_STENCILS[order]
        }
    )
    samples = {
        multiplier: _finite_vector(
            function(point + multiplier * step * vector),
            name=f"function(base + {multiplier} * step * direction)",
        )
        for multiplier in multipliers
    }
    if len({values.shape for values in samples.values()}) != 1:
        raise ValueError("high-order directional residual shape changed")
    reference = next(iter(samples.values()))
    return {
        order: np.asarray(
            sum(
                (
                    weight * samples[multiplier]
                    for multiplier, weight
                    in _CENTRAL_DERIVATIVE_STENCILS[order].items()
                ),
                start=np.zeros_like(reference),
            )
            / step,
            dtype=float,
        )
        for order in orders
    }


def causal_radial_colored_block_jacobian_family(
    function,
    base: np.ndarray,
    pattern: np.ndarray | csc_matrix | csr_matrix,
    *,
    finite_difference_step: float,
    derivative_orders=(4, 6),
) -> dict[int, dict[str, csr_matrix]]:
    """Assemble block Jacobians at several orders from one colored sweep."""

    point = _finite_vector(base, name="base")
    declared = (
        pattern.tocsc()
        if hasattr(pattern, "tocsc")
        else csc_matrix(np.asarray(pattern, dtype=bool))
    )
    step = float(finite_difference_step)
    orders = _validated_derivative_orders(derivative_orders)
    if (
        declared.shape != (point.size, point.size)
        or not np.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError("colored high-order Jacobian inputs are invalid")
    reference = {
        str(name): _finite_vector(values, name=f"block {name}")
        for name, values in function(point).items()
    }
    if (
        not reference
        or any(values.shape != point.shape for values in reference.values())
    ):
        raise ValueError("colored high-order block dimensions are invalid")
    matrices = {
        order: {
            name: lil_matrix(declared.shape, dtype=float)
            for name in reference
        }
        for order in orders
    }
    multipliers = sorted(
        {
            multiplier
            for order in orders
            for multiplier in _CENTRAL_DERIVATIVE_STENCILS[order]
        }
    )
    for group in causal_five_field_dae_jacobian_color_groups(declared):
        samples = {}
        for multiplier in multipliers:
            candidate = np.array(point, copy=True)
            candidate[group] += multiplier * step
            values = function(candidate)
            if set(values) != set(reference):
                raise ValueError("colored high-order residual keys changed")
            samples[multiplier] = {
                name: _finite_vector(
                    values[name],
                    name=f"{multiplier} step block {name}",
                )
                for name in reference
            }
            if any(
                values.shape != point.shape
                for values in samples[multiplier].values()
            ):
                raise ValueError(
                    "colored high-order residual shape changed"
                )
        differences = {
            order: {
                name: (
                    sum(
                        (
                            weight * samples[multiplier][name]
                            for multiplier, weight
                            in _CENTRAL_DERIVATIVE_STENCILS[
                                order
                            ].items()
                        ),
                        start=np.zeros_like(reference[name]),
                    )
                    / step
                )
                for name in reference
            }
            for order in orders
        }
        for column in group:
            start = declared.indptr[column]
            stop = declared.indptr[column + 1]
            rows = declared.indices[start:stop]
            for order in orders:
                for name in reference:
                    matrices[order][name][rows, column] = (
                        differences[order][name][rows, None]
                    )
    return {
        order: {
            name: matrix.tocsr()
            for name, matrix in order_matrices.items()
        }
        for order, order_matrices in matrices.items()
    }


def causal_radial_colored_block_jacobians(
    function,
    base: np.ndarray,
    pattern: np.ndarray | csc_matrix | csr_matrix,
    *,
    finite_difference_step: float,
) -> dict[str, csr_matrix]:
    """Assemble several same-shape block Jacobians from one colored sweep."""

    point = _finite_vector(base, name="base")
    declared = (
        pattern.tocsc()
        if hasattr(pattern, "tocsc")
        else csc_matrix(np.asarray(pattern, dtype=bool))
    )
    step = float(finite_difference_step)
    if (
        declared.shape != (point.size, point.size)
        or not np.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError("colored block-Jacobian inputs are invalid")
    reference = {
        str(name): _finite_vector(values, name=f"block {name}")
        for name, values in function(point).items()
    }
    if (
        not reference
        or any(values.shape != point.shape for values in reference.values())
    ):
        raise ValueError("colored block residual dimensions are invalid")
    matrices = {
        name: lil_matrix(declared.shape, dtype=float)
        for name in reference
    }
    for group in causal_five_field_dae_jacobian_color_groups(declared):
        plus_point = np.array(point, copy=True)
        minus_point = np.array(point, copy=True)
        plus_point[group] += step
        minus_point[group] -= step
        plus = function(plus_point)
        minus = function(minus_point)
        if set(plus) != set(reference) or set(minus) != set(reference):
            raise ValueError("colored block residual keys changed")
        differences = {}
        for name in reference:
            plus_values = _finite_vector(
                plus[name],
                name=f"plus block {name}",
            )
            minus_values = _finite_vector(
                minus[name],
                name=f"minus block {name}",
            )
            if (
                plus_values.shape != point.shape
                or minus_values.shape != point.shape
            ):
                raise ValueError("colored block residual shape changed")
            differences[name] = (
                plus_values - minus_values
            ) / (2.0 * step)
        for column in group:
            start = declared.indptr[column]
            stop = declared.indptr[column + 1]
            rows = declared.indices[start:stop]
            for name, difference in differences.items():
                matrices[name][rows, column] = difference[rows, None]
    return {
        name: matrix.tocsr()
        for name, matrix in matrices.items()
    }


def causal_radial_prefix_face_fluxes(
    inner_face_history: np.ndarray,
    conservative_transport_rows: np.ndarray,
) -> np.ndarray:
    """Recover every face flux from the inner flux and cell differences."""

    inner = np.asarray(inner_face_history, dtype=float)
    transport = np.asarray(conservative_transport_rows, dtype=float)
    if (
        inner.ndim != 2
        or transport.ndim != 3
        or transport.shape[0] != inner.shape[0]
        or transport.shape[2] != inner.shape[1]
        or np.any(~np.isfinite(inner))
        or np.any(~np.isfinite(transport))
    ):
        raise ValueError("prefix face-flux inputs are invalid")
    faces = np.empty(
        (inner.shape[0], transport.shape[1] + 1, inner.shape[1]),
        dtype=float,
    )
    faces[:, 0] = inner
    faces[:, 1:] = (
        inner[:, None, :] + np.cumsum(transport, axis=1)
    )
    return faces


def causal_radial_history_convergence(
    coarse: np.ndarray,
    medium: np.ndarray,
    fine: np.ndarray,
    *,
    minimum_order: float,
    maximum_fine_normalized_difference: float,
    minimum_fine_signed_cosine: float,
    minimum_relative_activity: float = 1.0e-8,
    component_reference_scales: np.ndarray | None = None,
    minimum_error_cosine: float | None = None,
) -> CausalRadialHistoryConvergence:
    """Return three-grid history metrics with explicit physical scaling.

    If ``component_reference_scales`` is omitted, each nonzero component is
    response-normalized, preserving the conservative WP10c9d5b convention.
    If fixed positive reference scales are supplied, they control both
    normalization and the relative-activity threshold.  The history cosine
    compares the medium and fine trajectories; the error cosine compares the
    coarse-medium and medium-fine refinement-error trajectories.
    """

    histories = [
        np.asarray(values, dtype=float)
        for values in (coarse, medium, fine)
    ]
    if (
        any(values.ndim != 2 for values in histories)
        or histories[0].shape != histories[1].shape
        or histories[0].shape != histories[2].shape
        or any(np.any(~np.isfinite(values)) for values in histories)
    ):
        raise ValueError("history convergence inputs are invalid")
    scales = np.max(np.abs(np.stack(histories, axis=0)), axis=(0, 1))
    relative_activity = float(minimum_relative_activity)
    if not np.isfinite(relative_activity) or relative_activity < 0.0:
        raise ValueError("minimum relative activity is invalid")
    if component_reference_scales is None:
        normalization_scales = np.maximum(scales, np.finfo(float).tiny)
        activity_thresholds = np.full_like(scales, np.finfo(float).tiny)
    else:
        normalization_scales = np.asarray(
            component_reference_scales,
            dtype=float,
        ).ravel()
        if (
            normalization_scales.shape != scales.shape
            or np.any(~np.isfinite(normalization_scales))
            or np.any(normalization_scales <= 0.0)
        ):
            raise ValueError("component reference scales are invalid")
        activity_thresholds = relative_activity * normalization_scales
    significant = scales > activity_thresholds
    if not np.any(significant):
        raise ValueError("history convergence has no significant component")
    safe_scales = normalization_scales[significant]
    normalized = [
        values[:, significant] / safe_scales[None, :]
        for values in histories
    ]
    component_coarse_medium = np.sqrt(
        np.mean((normalized[1] - normalized[0]) ** 2, axis=0)
    )
    component_medium_fine = np.sqrt(
        np.mean((normalized[2] - normalized[1]) ** 2, axis=0)
    )
    component_orders = np.full(scales.size, np.nan, dtype=float)
    component_history_cosines = np.full(scales.size, np.nan, dtype=float)
    component_error_cosines = np.full(scales.size, np.nan, dtype=float)
    component_passes = np.zeros(scales.size, dtype=bool)
    component_coarse_medium_full = np.full(
        scales.size,
        np.nan,
        dtype=float,
    )
    component_medium_fine_full = np.full(
        scales.size,
        np.nan,
        dtype=float,
    )
    for local, component in enumerate(np.flatnonzero(significant)):
        first = float(component_coarse_medium[local])
        second = float(component_medium_fine[local])
        component_coarse_medium_full[component] = first
        component_medium_fine_full[component] = second
        if second <= np.finfo(float).tiny:
            component_order = float("inf") if first > second else np.nan
        elif first <= np.finfo(float).tiny:
            component_order = np.nan
        else:
            component_order = float(np.log2(first / second))
        component_orders[component] = component_order
        medium_component = normalized[1][:, local]
        fine_component = normalized[2][:, local]
        coarse_medium_error = (
            normalized[1][:, local] - normalized[0][:, local]
        )
        medium_fine_error = (
            normalized[2][:, local] - normalized[1][:, local]
        )
        component_history_cosine = _cosine(
            medium_component,
            fine_component,
        )
        component_error_cosine = _cosine(
            coarse_medium_error,
            medium_fine_error,
        )
        component_history_cosines[component] = component_history_cosine
        component_error_cosines[component] = component_error_cosine
        component_passes[component] = bool(
            (
                second <= np.finfo(float).tiny
                or (
                    np.isfinite(component_order)
                    and component_order >= float(minimum_order)
                )
            )
            and second <= float(maximum_fine_normalized_difference)
            and component_history_cosine
            >= float(minimum_fine_signed_cosine)
            and (
                minimum_error_cosine is None
                or component_error_cosine >= float(minimum_error_cosine)
            )
        )
    coarse_medium = float(
        np.sqrt(np.mean((normalized[1] - normalized[0]) ** 2))
    )
    medium_fine = float(
        np.sqrt(np.mean((normalized[2] - normalized[1]) ** 2))
    )
    if medium_fine <= np.finfo(float).tiny:
        order = float("inf") if coarse_medium > medium_fine else None
    elif coarse_medium <= np.finfo(float).tiny:
        order = None
    else:
        order = float(np.log2(coarse_medium / medium_fine))
    history_cosine = _cosine(
        normalized[1].ravel(),
        normalized[2].ravel(),
    )
    error_cosine = _cosine(
        (normalized[1] - normalized[0]).ravel(),
        (normalized[2] - normalized[1]).ravel(),
    )
    order_passed = bool(
        medium_fine <= np.finfo(float).tiny
        or (order is not None and order >= float(minimum_order))
    )
    passed = bool(
        order_passed
        and medium_fine <= float(maximum_fine_normalized_difference)
        and history_cosine >= float(minimum_fine_signed_cosine)
        and (
            minimum_error_cosine is None
            or error_cosine >= float(minimum_error_cosine)
        )
        and np.all(component_passes[significant])
    )
    return CausalRadialHistoryConvergence(
        component_scales=np.asarray(scales, dtype=float),
        component_normalization_scales=np.asarray(
            normalization_scales,
            dtype=float,
        ),
        component_activity_thresholds=np.asarray(
            activity_thresholds,
            dtype=float,
        ),
        significant_components=np.asarray(significant, dtype=bool),
        component_coarse_medium_differences=np.asarray(
            component_coarse_medium_full,
            dtype=float,
        ),
        component_medium_fine_differences=np.asarray(
            component_medium_fine_full,
            dtype=float,
        ),
        component_observed_orders=component_orders,
        component_history_cosines=component_history_cosines,
        component_error_cosines=component_error_cosines,
        component_fine_signed_cosines=component_history_cosines,
        component_passed=component_passes,
        coarse_medium_difference=coarse_medium,
        medium_fine_difference=medium_fine,
        observed_order=order,
        history_cosine=history_cosine,
        error_cosine=error_cosine,
        fine_signed_cosine=history_cosine,
        passed=passed,
    )


def causal_radial_first_consecutive_recovery(
    passed: np.ndarray,
    *,
    required_consecutive: int = 2,
) -> int | None:
    """Return the first passing index in a consecutive recovery run."""

    flags = np.asarray(passed, dtype=bool)
    required = int(required_consecutive)
    if flags.ndim != 1 or flags.size < 1 or required < 1:
        raise ValueError("recovery flags are invalid")
    for start in range(max(0, flags.size - required + 1)):
        if np.all(flags[start : start + required]):
            return int(start)
    return None
