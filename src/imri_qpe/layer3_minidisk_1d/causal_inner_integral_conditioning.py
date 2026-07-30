"""Prospective conditioning gates for signed physical integrals.

The historical component-order gate remains the primary route.  This module
defines a stricter alternate route for a scalar integral whose direct
refinement order is ill-conditioned because separately convergent physical
bands cancel.  It changes no evolution or residual operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalIntegralConditioningDecision:
    """Result of one prospectively declared scalar-component gate."""

    passed: bool
    route: str
    active_band_count: int
    maximum_cancellation_ratio: float
    absolute_band_error_envelope: float


def causal_cancellation_ratio(
    band_errors: np.ndarray,
    *,
    time_weights: np.ndarray,
) -> float:
    """Return norm of the signed sum divided by summed band norms."""

    errors = np.asarray(band_errors, dtype=float)
    weights = np.asarray(time_weights, dtype=float)
    if errors.ndim != 2 or weights.shape != (errors.shape[0],):
        raise ValueError("band-error history shape is invalid")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("time weights must be nonnegative and nonzero")
    if np.any(~np.isfinite(errors)) or np.any(~np.isfinite(weights)):
        raise ValueError("band-error history must be finite")
    signed = np.sum(errors, axis=1)
    numerator = float(np.sqrt(np.sum(weights * signed**2)))
    band_norms = np.sqrt(np.sum(weights[:, None] * errors**2, axis=0))
    denominator = float(np.sum(band_norms))
    return numerator / max(denominator, np.finfo(float).tiny)


def causal_absolute_band_error_envelope(
    band_errors: np.ndarray,
    *,
    physical_scale: float,
) -> float:
    """Return the sum of per-band maximum normalized errors."""

    errors = np.asarray(band_errors, dtype=float)
    scale = float(physical_scale)
    if errors.ndim != 2 or np.any(~np.isfinite(errors)):
        raise ValueError("band-error history must be a finite matrix")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("physical scale must be positive and finite")
    return float(np.sum(np.max(np.abs(errors), axis=0)) / scale)


def causal_integral_conditioning_decision(
    *,
    global_rms_order: float,
    global_maximum_order: float,
    global_fine_maximum: float,
    cell_rms_orders: np.ndarray,
    active_cells: np.ndarray,
    band_rms_orders: np.ndarray,
    band_maximum_orders: np.ndarray,
    band_error_cosines: np.ndarray,
    active_bands: np.ndarray,
    absolute_band_error_envelope: float,
    coarse_medium_cancellation_ratio: float,
    medium_fine_cancellation_ratio: float,
    direct_sum_defect: float,
    gram_closure_defect: float,
    continuum_uncertainty_to_fine: float,
    minimum_order: float,
    minimum_error_cosine: float,
    maximum_fine_difference: float,
    maximum_cancellation_ratio: float,
    maximum_ledger_defect: float,
    maximum_continuum_ratio: float,
) -> CausalIntegralConditioningDecision:
    """Apply the frozen direct or cancellation-conditioned component gate."""

    cell_orders = np.asarray(cell_rms_orders, dtype=float)
    active_cell_mask = np.asarray(active_cells, dtype=bool)
    band_rms = np.asarray(band_rms_orders, dtype=float)
    band_max = np.asarray(band_maximum_orders, dtype=float)
    band_cosines = np.asarray(band_error_cosines, dtype=float)
    active = np.asarray(active_bands, dtype=bool)
    if (
        cell_orders.ndim != 1
        or active_cell_mask.shape != cell_orders.shape
        or not np.any(active_cell_mask)
        or band_rms.ndim != 1
        or band_max.shape != band_rms.shape
        or band_cosines.shape != band_rms.shape
        or active.shape != band_rms.shape
        or not np.any(active)
    ):
        raise ValueError("component-conditioning arrays are invalid")
    scalars = np.asarray(
        [
            global_rms_order,
            global_maximum_order,
            global_fine_maximum,
            absolute_band_error_envelope,
            coarse_medium_cancellation_ratio,
            medium_fine_cancellation_ratio,
            direct_sum_defect,
            gram_closure_defect,
            continuum_uncertainty_to_fine,
            minimum_order,
            minimum_error_cosine,
            maximum_fine_difference,
            maximum_cancellation_ratio,
            maximum_ledger_defect,
            maximum_continuum_ratio,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(scalars)):
        raise ValueError("component-conditioning scalars must be finite")
    common = bool(
        global_fine_maximum <= maximum_fine_difference
        and direct_sum_defect <= maximum_ledger_defect
        and gram_closure_defect <= maximum_ledger_defect
        and continuum_uncertainty_to_fine <= maximum_continuum_ratio
    )
    direct = bool(
        common
        and global_rms_order >= minimum_order
        and global_maximum_order >= minimum_order
    )
    maximum_ratio = max(
        float(coarse_medium_cancellation_ratio),
        float(medium_fine_cancellation_ratio),
    )
    if direct:
        return CausalIntegralConditioningDecision(
            passed=True,
            route="direct_component_order",
            active_band_count=int(np.count_nonzero(active)),
            maximum_cancellation_ratio=maximum_ratio,
            absolute_band_error_envelope=float(
                absolute_band_error_envelope
            ),
        )
    alternate = bool(
        common
        and np.min(cell_orders[active_cell_mask]) >= minimum_order
        and np.min(band_rms[active]) >= minimum_order
        and np.min(band_max[active]) >= minimum_order
        and np.min(band_cosines[active]) >= minimum_error_cosine
        and absolute_band_error_envelope <= maximum_fine_difference
        and maximum_ratio <= maximum_cancellation_ratio
    )
    return CausalIntegralConditioningDecision(
        passed=alternate,
        route=(
            "cancellation_conditioned_band_envelope"
            if alternate
            else "failed"
        ),
        active_band_count=int(np.count_nonzero(active)),
        maximum_cancellation_ratio=maximum_ratio,
        absolute_band_error_envelope=float(absolute_band_error_envelope),
    )
