"""Control-volume localization helpers for frozen radial histories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix

from .causal_inner_dae_system import (
    causal_five_field_dae_jacobian_color_groups,
)


@dataclass(frozen=True)
class CausalRadialHistoryConvergence:
    """Three-grid convergence of one multicomponent time history."""

    component_scales: np.ndarray
    significant_components: np.ndarray
    component_coarse_medium_differences: np.ndarray
    component_medium_fine_differences: np.ndarray
    component_observed_orders: np.ndarray
    component_fine_signed_cosines: np.ndarray
    component_passed: np.ndarray
    coarse_medium_difference: float
    medium_fine_difference: float
    observed_order: float | None
    fine_signed_cosine: float
    passed: bool


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size < 1 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


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
) -> CausalRadialHistoryConvergence:
    """Return response-normalized three-grid history metrics."""

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
    activity_floor = max(
        float(minimum_relative_activity) * np.finfo(float).tiny,
        np.finfo(float).tiny,
    )
    significant = scales > activity_floor
    if not np.any(significant):
        raise ValueError("history convergence has no significant component")
    safe_scales = np.maximum(scales[significant], np.finfo(float).tiny)
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
    component_cosines = np.full(scales.size, np.nan, dtype=float)
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
        component_denominator = float(
            np.linalg.norm(medium_component)
            * np.linalg.norm(fine_component)
        )
        component_cosine = (
            1.0
            if component_denominator <= np.finfo(float).tiny
            else float(
                np.dot(medium_component, fine_component)
                / component_denominator
            )
        )
        component_cosines[component] = component_cosine
        component_passes[component] = bool(
            (
                second <= np.finfo(float).tiny
                or (
                    np.isfinite(component_order)
                    and component_order >= float(minimum_order)
                )
            )
            and second <= float(maximum_fine_normalized_difference)
            and component_cosine >= float(minimum_fine_signed_cosine)
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
    medium_vector = normalized[1].ravel()
    fine_vector = normalized[2].ravel()
    denominator = float(
        np.linalg.norm(medium_vector) * np.linalg.norm(fine_vector)
    )
    cosine = (
        1.0
        if denominator <= np.finfo(float).tiny
        else float(np.dot(medium_vector, fine_vector) / denominator)
    )
    order_passed = bool(
        medium_fine <= np.finfo(float).tiny
        or (order is not None and order >= float(minimum_order))
    )
    passed = bool(
        order_passed
        and medium_fine <= float(maximum_fine_normalized_difference)
        and cosine >= float(minimum_fine_signed_cosine)
        and np.all(component_passes[significant])
    )
    return CausalRadialHistoryConvergence(
        component_scales=np.asarray(scales, dtype=float),
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
        component_fine_signed_cosines=component_cosines,
        component_passed=component_passes,
        coarse_medium_difference=coarse_medium,
        medium_fine_difference=medium_fine,
        observed_order=order,
        fine_signed_cosine=cosine,
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
