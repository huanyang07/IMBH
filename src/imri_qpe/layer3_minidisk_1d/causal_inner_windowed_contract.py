"""Windowed, variable-coefficient convergence tools for the inner DAE.

This module is audit-only.  It supplies small, independently testable
operations used by WP10c9d6c6a2:

* smooth finite-interval windows;
* overlap-continuous physical characteristic fields;
* exact proper-measure restriction between nested grids; and
* a three-level Richardson comparison in one fixed physical norm.

It does not construct or alter a production residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_N_FIELDS = 5


def _positive_vector(
    values: np.ndarray,
    *,
    size: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=float).ravel()
    if (
        result.shape != (size,)
        or np.any(~np.isfinite(result))
        or np.any(result <= 0.0)
    ):
        raise ValueError(f"{name} must be a positive finite vector")
    return result


def causal_sine_power_window(
    log_radii: np.ndarray,
    *,
    lower_log_radius: float,
    upper_log_radius: float,
    power: int,
) -> np.ndarray:
    """Return ``sin(pi xi)**power`` on one declared finite interval."""

    values = np.asarray(log_radii, dtype=float)
    lower = float(lower_log_radius)
    upper = float(upper_log_radius)
    exponent = int(power)
    if (
        np.any(~np.isfinite(values))
        or not np.isfinite(lower)
        or not np.isfinite(upper)
        or upper <= lower
        or exponent < 2
    ):
        raise ValueError("sine-power window inputs are invalid")
    coordinate = (values - lower) / (upper - lower)
    inside = (coordinate >= 0.0) & (coordinate <= 1.0)
    result = np.zeros_like(values)
    result[inside] = np.sin(np.pi * coordinate[inside]) ** exponent
    return result


@dataclass(frozen=True)
class CausalAlignedCharacteristicField:
    """One overlap-continuous five-family physical basis."""

    physical_right_eigenvectors: np.ndarray
    minimum_adjacent_overlap: float
    maximum_dimensionless_norm_defect: float


def causal_align_characteristic_field(
    physical_right_eigenvectors: np.ndarray,
    field_scales: np.ndarray,
) -> CausalAlignedCharacteristicField:
    """Normalize and sign-align an ordered characteristic basis by overlap.

    The family ordering is assumed to have already been selected by the
    physical characteristic construction.  This routine deliberately changes
    signs only; it cannot hide a family permutation.
    """

    physical = np.array(
        physical_right_eigenvectors,
        dtype=float,
        copy=True,
    )
    if (
        physical.ndim != 3
        or physical.shape[1:] != (_N_FIELDS, _N_FIELDS)
        or physical.shape[0] < 2
        or np.any(~np.isfinite(physical))
    ):
        raise ValueError("characteristic field has the wrong shape")
    scales = _positive_vector(
        field_scales,
        size=_N_FIELDS,
        name="field scales",
    )
    dimensionless = physical / scales[None, :, None]
    norms = np.linalg.norm(dimensionless, axis=1)
    if np.any(norms <= np.finfo(float).tiny):
        raise ValueError("characteristic field contains a singular vector")
    physical /= norms[:, None, :]
    dimensionless /= norms[:, None, :]

    minimum_overlap = 1.0
    for node in range(1, physical.shape[0]):
        for family in range(_N_FIELDS):
            overlap = float(
                np.dot(
                    dimensionless[node - 1, :, family],
                    dimensionless[node, :, family],
                )
            )
            minimum_overlap = min(minimum_overlap, abs(overlap))
            if overlap < 0.0:
                physical[node, :, family] *= -1.0
                dimensionless[node, :, family] *= -1.0
    norm_defect = float(
        np.max(np.abs(np.linalg.norm(dimensionless, axis=1) - 1.0))
    )
    return CausalAlignedCharacteristicField(
        physical_right_eigenvectors=physical,
        minimum_adjacent_overlap=minimum_overlap,
        maximum_dimensionless_norm_defect=norm_defect,
    )


def causal_restrict_proper_cell_averages(
    fine_values: np.ndarray,
    fine_cell_measures: np.ndarray,
    *,
    refinement_factor: int,
) -> np.ndarray:
    """Restrict nested proper-measure cell averages exactly."""

    values = np.asarray(fine_values)
    measures = np.asarray(fine_cell_measures, dtype=float).ravel()
    factor = int(refinement_factor)
    if (
        values.ndim < 2
        or values.shape[-1] != _N_FIELDS
        or values.shape[-2] != measures.size
        or factor < 1
        or measures.size % factor
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(measures))
        or np.any(measures <= 0.0)
    ):
        raise ValueError("nested restriction inputs are invalid")
    coarse_cells = measures.size // factor
    reshaped = values.reshape(
        values.shape[:-2] + (coarse_cells, factor, _N_FIELDS)
    )
    grouped_measures = measures.reshape(coarse_cells, factor)
    numerator = np.sum(
        reshaped
        * grouped_measures.reshape(
            (1,) * (values.ndim - 2)
            + (coarse_cells, factor, 1)
        ),
        axis=-2,
    )
    denominator = np.sum(grouped_measures, axis=1)
    return numerator / denominator.reshape(
        (1,) * (values.ndim - 2) + (coarse_cells, 1)
    )


def causal_trapezoid_weights(times: np.ndarray) -> np.ndarray:
    """Return positive trapezoid weights normalized to unit sum."""

    values = np.asarray(times, dtype=float).ravel()
    if (
        values.size < 2
        or np.any(~np.isfinite(values))
        or np.any(np.diff(values) <= 0.0)
    ):
        raise ValueError("times must be finite and strictly increasing")
    increments = np.diff(values)
    weights = np.empty_like(values)
    weights[0] = 0.5 * increments[0]
    weights[-1] = 0.5 * increments[-1]
    if values.size > 2:
        weights[1:-1] = 0.5 * (
            increments[:-1] + increments[1:]
        )
    return weights / float(np.sum(weights))


def causal_field_history_inner_product(
    left: np.ndarray,
    right: np.ndarray,
    *,
    cell_measures: np.ndarray,
    field_scales: np.ndarray,
    time_weights: np.ndarray,
) -> float:
    """Return one proper-measure, fixed-field-scale history product."""

    first = np.asarray(left)
    second = np.asarray(right)
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("history arrays must have shape (time, cell, 5)")
    times, cells, fields = first.shape
    if fields != _N_FIELDS:
        raise ValueError("history arrays must contain five fields")
    measures = _positive_vector(
        cell_measures,
        size=cells,
        name="cell measures",
    )
    scales = _positive_vector(
        field_scales,
        size=_N_FIELDS,
        name="field scales",
    )
    weights = _positive_vector(
        time_weights,
        size=times,
        name="time weights",
    )
    weights = weights / float(np.sum(weights))
    normalized_left = first / scales[None, None, :]
    normalized_right = second / scales[None, None, :]
    spatial = measures / float(np.sum(measures))
    value = np.einsum(
        "tci,tci,t,c->",
        np.conjugate(normalized_left),
        normalized_right,
        weights,
        spatial,
    )
    return float(np.real(value))


def causal_field_history_norm(
    values: np.ndarray,
    *,
    cell_measures: np.ndarray,
    field_scales: np.ndarray,
    time_weights: np.ndarray,
) -> float:
    """Return the norm induced by :func:`causal_field_history_inner_product`."""

    product = causal_field_history_inner_product(
        values,
        values,
        cell_measures=cell_measures,
        field_scales=field_scales,
        time_weights=time_weights,
    )
    return float(np.sqrt(max(product, 0.0)))


@dataclass(frozen=True)
class CausalWindowedRichardsonResult:
    """One three-grid variable-coefficient windowed comparison."""

    observed_order: float
    minimum_significant_component_order: float
    refinement_error_cosine: float
    coarse_medium_history_norm: float
    medium_fine_history_norm: float
    maximum_coarse_reference_relative_error: float
    history_coarse_reference_relative_error: float
    reference_choice_to_fine_difference_ratio: float
    observed_reference: np.ndarray
    fixed_second_order_reference: np.ndarray


def causal_windowed_richardson_reference(
    coarse: np.ndarray,
    medium_on_coarse: np.ndarray,
    fine_on_coarse: np.ndarray,
    *,
    times: np.ndarray,
    coarse_cell_measures: np.ndarray,
    field_scales: np.ndarray,
    relative_activity: float = 1.0e-10,
) -> CausalWindowedRichardsonResult:
    """Build observed-order and fixed-second-order references."""

    first = np.asarray(coarse, dtype=float)
    second = np.asarray(medium_on_coarse, dtype=float)
    third = np.asarray(fine_on_coarse, dtype=float)
    if (
        first.shape != second.shape
        or first.shape != third.shape
        or first.ndim != 3
        or first.shape[-1] != _N_FIELDS
        or np.any(~np.isfinite(first))
        or np.any(~np.isfinite(second))
        or np.any(~np.isfinite(third))
        or not 0.0 < float(relative_activity) < 1.0
    ):
        raise ValueError("Richardson histories are invalid")
    weights = causal_trapezoid_weights(times)
    difference_coarse = first - second
    difference_fine = second - third
    norm_coarse = causal_field_history_norm(
        difference_coarse,
        cell_measures=coarse_cell_measures,
        field_scales=field_scales,
        time_weights=weights,
    )
    norm_fine = causal_field_history_norm(
        difference_fine,
        cell_measures=coarse_cell_measures,
        field_scales=field_scales,
        time_weights=weights,
    )
    tiny = np.finfo(float).tiny
    order = float(
        np.log2(max(norm_coarse, tiny) / max(norm_fine, tiny))
    )
    denominator = max(
        float(
            np.sqrt(
                causal_field_history_inner_product(
                    difference_coarse,
                    difference_coarse,
                    cell_measures=coarse_cell_measures,
                    field_scales=field_scales,
                    time_weights=weights,
                )
                * causal_field_history_inner_product(
                    difference_fine,
                    difference_fine,
                    cell_measures=coarse_cell_measures,
                    field_scales=field_scales,
                    time_weights=weights,
                )
            )
        ),
        tiny,
    )
    cosine = float(
        causal_field_history_inner_product(
            difference_coarse,
            difference_fine,
            cell_measures=coarse_cell_measures,
            field_scales=field_scales,
            time_weights=weights,
        )
        / denominator
    )

    component_orders = []
    global_activity = max(norm_coarse, norm_fine, tiny)
    for field in range(_N_FIELDS):
        field_scale = np.ones(_N_FIELDS, dtype=float)
        field_scale[field] = field_scales[field]
        masked_coarse = np.zeros_like(difference_coarse)
        masked_fine = np.zeros_like(difference_fine)
        masked_coarse[..., field] = difference_coarse[..., field]
        masked_fine[..., field] = difference_fine[..., field]
        field_coarse = causal_field_history_norm(
            masked_coarse,
            cell_measures=coarse_cell_measures,
            field_scales=field_scale,
            time_weights=weights,
        )
        field_fine = causal_field_history_norm(
            masked_fine,
            cell_measures=coarse_cell_measures,
            field_scales=field_scale,
            time_weights=weights,
        )
        if max(field_coarse, field_fine) >= (
            float(relative_activity) * global_activity
        ):
            component_orders.append(
                float(
                    np.log2(
                        max(field_coarse, tiny)
                        / max(field_fine, tiny)
                    )
                )
            )
    minimum_component_order = (
        min(component_orders) if component_orders else float("inf")
    )

    observed_factor = 1.0 / max(2.0**order - 1.0, tiny)
    observed_reference = third + observed_factor * (third - second)
    fixed_reference = third + (third - second) / 3.0
    initial_norm = causal_field_history_norm(
        np.broadcast_to(
            third[0:1],
            third.shape,
        ),
        cell_measures=coarse_cell_measures,
        field_scales=field_scales,
        time_weights=weights,
    )
    initial_norm = max(initial_norm, tiny)
    coarse_reference = first - observed_reference
    history_error = causal_field_history_norm(
        coarse_reference,
        cell_measures=coarse_cell_measures,
        field_scales=field_scales,
        time_weights=weights,
    ) / initial_norm
    time_errors = []
    for index in range(first.shape[0]):
        one_time = np.ones(2, dtype=float)
        duplicated = np.stack(
            (coarse_reference[index], coarse_reference[index]),
            axis=0,
        )
        time_errors.append(
            causal_field_history_norm(
                duplicated,
                cell_measures=coarse_cell_measures,
                field_scales=field_scales,
                time_weights=one_time,
            )
            / initial_norm
        )
    reference_choice = causal_field_history_norm(
        observed_reference - fixed_reference,
        cell_measures=coarse_cell_measures,
        field_scales=field_scales,
        time_weights=weights,
    ) / max(norm_fine, tiny)
    return CausalWindowedRichardsonResult(
        observed_order=order,
        minimum_significant_component_order=float(
            minimum_component_order
        ),
        refinement_error_cosine=cosine,
        coarse_medium_history_norm=norm_coarse,
        medium_fine_history_norm=norm_fine,
        maximum_coarse_reference_relative_error=float(
            max(time_errors)
        ),
        history_coarse_reference_relative_error=float(history_error),
        reference_choice_to_fine_difference_ratio=float(
            reference_choice
        ),
        observed_reference=observed_reference,
        fixed_second_order_reference=fixed_reference,
    )
