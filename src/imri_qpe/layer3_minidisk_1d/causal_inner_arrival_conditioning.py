"""Conditioning diagnostics for amplified causal-inner arrival histories.

The helpers in this module are audit-only.  They do not change an evolution
operator, a packet, or a historical validation decision.  They separate the
absolute initial-energy scale used by WP10c9d6c7c2b4 from response-relative,
shape, peak-time, Richardson, nuisance-envelope, and terminal-tail measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalQuadraticPeak:
    """Continuous-time peak estimated from three neighboring samples."""

    sample_index: int
    sample_time_seconds: float
    sample_value: float
    interpolated_time_seconds: float
    interpolated_value: float
    interpolation_used: bool


@dataclass(frozen=True)
class CausalHistoryConditioning:
    """Three-level conditioning measures for one scalar history."""

    absolute_fine_maximum_difference: float
    response_scale: float
    response_relative_fine_maximum_difference: float
    weighted_rms_order: float
    refinement_error_cosine: float
    amplitudes: np.ndarray
    amplitude_order: float
    amplitude_relative_fine_difference: float
    shape_fine_maximum_difference: float
    shape_weighted_rms_order: float
    shape_refinement_error_cosine: float
    fixed_second_order_reference_difference: float
    observed_order_reference_difference: float
    extrapolation_model_spread: float


@dataclass(frozen=True)
class CausalHistoryUncertaintyEnvelope:
    """Conservative deterministic bounds for two refinement-error vectors."""

    coarse_medium_components_l2: dict[str, float]
    medium_fine_components_l2: dict[str, float]
    coarse_medium_components_linf: dict[str, float]
    medium_fine_components_linf: dict[str, float]
    coarse_medium_conservative_l2: float
    medium_fine_conservative_l2: float
    coarse_medium_conservative_linf: float
    medium_fine_conservative_linf: float
    coarse_medium_observable: bool
    medium_fine_observable: bool


@dataclass(frozen=True)
class CausalHorizonCompleteness:
    """Terminal-tail measures for one nonnegative arrival history."""

    peak_value: float
    terminal_to_peak: float
    final_window_range_to_peak: float
    terminal_slope_horizon_to_peak: float
    peak_time_fraction: float
    complete: bool


def _validate_history_triplet(
    coarse: np.ndarray,
    medium: np.ndarray,
    fine: np.ndarray,
    times_seconds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(item, dtype=float) for item in (coarse, medium, fine))
    times = np.asarray(times_seconds, dtype=float)
    if (
        times.ndim != 1
        or times.size < 3
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or any(item.shape != times.shape for item in arrays)
        or any(np.any(~np.isfinite(item)) for item in arrays)
    ):
        raise ValueError("arrival-history triplet is invalid")
    return arrays[0], arrays[1], arrays[2], times


def causal_trapezoid_weights(times_seconds: np.ndarray) -> np.ndarray:
    """Return normalized trapezoid weights on a strictly increasing grid."""

    times = np.asarray(times_seconds, dtype=float)
    if (
        times.ndim != 1
        or times.size < 2
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("time grid is invalid")
    weights = np.empty_like(times)
    weights[0] = 0.5 * (times[1] - times[0])
    weights[-1] = 0.5 * (times[-1] - times[-2])
    weights[1:-1] = 0.5 * (times[2:] - times[:-2])
    return weights / np.sum(weights)


def _weighted_norm(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * np.asarray(values, dtype=float) ** 2)))


def _order(first: float, second: float) -> float:
    tiny = np.finfo(float).tiny
    return float(np.log2(max(abs(first), tiny) / max(abs(second), tiny)))


def _cosine(first: np.ndarray, second: np.ndarray, weights: np.ndarray) -> float:
    left = np.sqrt(weights) * np.asarray(first, dtype=float)
    right = np.sqrt(weights) * np.asarray(second, dtype=float)
    denominator = max(
        float(np.linalg.norm(left) * np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.dot(left, right) / denominator)


def causal_quadratic_peak(
    times_seconds: np.ndarray,
    values: np.ndarray,
) -> CausalQuadraticPeak:
    """Estimate a peak without moving any measurement window."""

    times = np.asarray(times_seconds, dtype=float)
    history = np.asarray(values, dtype=float)
    if (
        times.ndim != 1
        or history.shape != times.shape
        or times.size < 3
        or np.any(~np.isfinite(times))
        or np.any(~np.isfinite(history))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("peak inputs are invalid")
    index = int(np.argmax(history))
    sample_time = float(times[index])
    sample_value = float(history[index])
    if index == 0 or index == times.size - 1:
        return CausalQuadraticPeak(
            sample_index=index,
            sample_time_seconds=sample_time,
            sample_value=sample_value,
            interpolated_time_seconds=sample_time,
            interpolated_value=sample_value,
            interpolation_used=False,
        )
    local_times = times[index - 1 : index + 2]
    local_values = history[index - 1 : index + 2]
    shifted = local_times - sample_time
    quadratic = np.polyfit(shifted, local_values, 2)
    curvature, slope, intercept = (float(item) for item in quadratic)
    if not np.isfinite(curvature) or curvature >= 0.0:
        return CausalQuadraticPeak(
            sample_index=index,
            sample_time_seconds=sample_time,
            sample_value=sample_value,
            interpolated_time_seconds=sample_time,
            interpolated_value=sample_value,
            interpolation_used=False,
        )
    offset = -slope / (2.0 * curvature)
    if offset < shifted[0] or offset > shifted[-1]:
        return CausalQuadraticPeak(
            sample_index=index,
            sample_time_seconds=sample_time,
            sample_value=sample_value,
            interpolated_time_seconds=sample_time,
            interpolated_value=sample_value,
            interpolation_used=False,
        )
    value = curvature * offset**2 + slope * offset + intercept
    return CausalQuadraticPeak(
        sample_index=index,
        sample_time_seconds=sample_time,
        sample_value=sample_value,
        interpolated_time_seconds=float(sample_time + offset),
        interpolated_value=float(value),
        interpolation_used=True,
    )


def causal_arrival_history_conditioning(
    coarse: np.ndarray,
    medium: np.ndarray,
    fine: np.ndarray,
    *,
    times_seconds: np.ndarray,
) -> CausalHistoryConditioning:
    """Separate absolute, response-relative, amplitude, and shape errors."""

    coarse, medium, fine, times = _validate_history_triplet(
        coarse,
        medium,
        fine,
        times_seconds,
    )
    weights = causal_trapezoid_weights(times)
    first = medium - coarse
    second = fine - medium
    first_norm = _weighted_norm(first, weights)
    second_norm = _weighted_norm(second, weights)
    response_scale = max(
        float(np.max(np.abs(fine))),
        float(np.max(np.abs(medium))),
        float(np.max(np.abs(coarse))),
        np.finfo(float).tiny,
    )
    amplitudes = np.asarray(
        [
            causal_quadratic_peak(times, item).interpolated_value
            for item in (coarse, medium, fine)
        ],
        dtype=float,
    )
    amplitude_scale = max(float(np.max(np.abs(amplitudes))), np.finfo(float).tiny)
    shapes = [
        item / max(abs(amplitude), np.finfo(float).tiny)
        for item, amplitude in zip(
            (coarse, medium, fine),
            amplitudes,
            strict=True,
        )
    ]
    shape_first = shapes[1] - shapes[0]
    shape_second = shapes[2] - shapes[1]
    observed_order = _order(first_norm, second_norm)
    safe_observed_order = (
        observed_order if np.isfinite(observed_order) and observed_order > 0.0 else 2.0
    )
    fixed_reference = fine + (fine - medium) / (2.0**2 - 1.0)
    observed_reference = fine + (fine - medium) / (
        2.0**safe_observed_order - 1.0
    )
    return CausalHistoryConditioning(
        absolute_fine_maximum_difference=float(np.max(np.abs(second))),
        response_scale=response_scale,
        response_relative_fine_maximum_difference=float(
            np.max(np.abs(second)) / response_scale
        ),
        weighted_rms_order=observed_order,
        refinement_error_cosine=_cosine(first, second, weights),
        amplitudes=amplitudes,
        amplitude_order=_order(
            amplitudes[1] - amplitudes[0],
            amplitudes[2] - amplitudes[1],
        ),
        amplitude_relative_fine_difference=float(
            abs(amplitudes[2] - amplitudes[1]) / amplitude_scale
        ),
        shape_fine_maximum_difference=float(
            np.max(np.abs(shape_second))
        ),
        shape_weighted_rms_order=_order(
            _weighted_norm(shape_first, weights),
            _weighted_norm(shape_second, weights),
        ),
        shape_refinement_error_cosine=_cosine(
            shape_first,
            shape_second,
            weights,
        ),
        fixed_second_order_reference_difference=float(
            _weighted_norm(fixed_reference - fine, weights) / response_scale
        ),
        observed_order_reference_difference=float(
            _weighted_norm(observed_reference - fine, weights) / response_scale
        ),
        extrapolation_model_spread=float(
            _weighted_norm(fixed_reference - observed_reference, weights)
            / response_scale
        ),
    )


def causal_history_uncertainty_envelope(
    coarse: np.ndarray,
    medium: np.ndarray,
    fine: np.ndarray,
    *,
    times_seconds: np.ndarray,
    variations: dict[str, np.ndarray],
    observability_factor: float,
) -> CausalHistoryUncertaintyEnvelope:
    """Bound refinement-error changes by a conservative nuisance envelope.

    Each variation array has shape ``(n_variants, 3, n_times)``.  The bound
    for one category is the largest change in the corresponding refinement
    error across its variants.  Category bounds are summed; RSS is not used.
    """

    coarse, medium, fine, times = _validate_history_triplet(
        coarse,
        medium,
        fine,
        times_seconds,
    )
    factor = float(observability_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("observability factor must be positive")
    weights = causal_trapezoid_weights(times)
    nominal_cm = medium - coarse
    nominal_mf = fine - medium
    cm_l2: dict[str, float] = {}
    mf_l2: dict[str, float] = {}
    cm_linf: dict[str, float] = {}
    mf_linf: dict[str, float] = {}
    for name, raw in variations.items():
        array = np.asarray(raw, dtype=float)
        if (
            array.ndim != 3
            or array.shape[1:] != (3, times.size)
            or array.shape[0] < 1
            or np.any(~np.isfinite(array))
        ):
            raise ValueError(f"history variation {name!r} is invalid")
        delta_cm = (array[:, 1] - array[:, 0]) - nominal_cm[None]
        delta_mf = (array[:, 2] - array[:, 1]) - nominal_mf[None]
        cm_l2[name] = float(
            np.max(
                np.sqrt(np.sum(weights[None] * delta_cm**2, axis=1))
            )
        )
        mf_l2[name] = float(
            np.max(
                np.sqrt(np.sum(weights[None] * delta_mf**2, axis=1))
            )
        )
        cm_linf[name] = float(np.max(np.abs(delta_cm)))
        mf_linf[name] = float(np.max(np.abs(delta_mf)))
    total_cm_l2 = float(sum(cm_l2.values()))
    total_mf_l2 = float(sum(mf_l2.values()))
    total_cm_linf = float(sum(cm_linf.values()))
    total_mf_linf = float(sum(mf_linf.values()))
    return CausalHistoryUncertaintyEnvelope(
        coarse_medium_components_l2=cm_l2,
        medium_fine_components_l2=mf_l2,
        coarse_medium_components_linf=cm_linf,
        medium_fine_components_linf=mf_linf,
        coarse_medium_conservative_l2=total_cm_l2,
        medium_fine_conservative_l2=total_mf_l2,
        coarse_medium_conservative_linf=total_cm_linf,
        medium_fine_conservative_linf=total_mf_linf,
        coarse_medium_observable=bool(
            _weighted_norm(nominal_cm, weights) >= factor * total_cm_l2
        ),
        medium_fine_observable=bool(
            _weighted_norm(nominal_mf, weights) >= factor * total_mf_l2
        ),
    )


def causal_horizon_completeness(
    times_seconds: np.ndarray,
    values: np.ndarray,
    *,
    final_window_fraction: float,
    maximum_terminal_to_peak: float,
    maximum_final_window_range_to_peak: float,
    maximum_terminal_slope_horizon_to_peak: float,
) -> CausalHorizonCompleteness:
    """Test whether a nonnegative arrival history has cleared its band."""

    times = np.asarray(times_seconds, dtype=float)
    history = np.asarray(values, dtype=float)
    fraction = float(final_window_fraction)
    if (
        times.ndim != 1
        or history.shape != times.shape
        or times.size < 5
        or np.any(~np.isfinite(times))
        or np.any(~np.isfinite(history))
        or np.any(np.diff(times) <= 0.0)
        or np.min(history) < -1.0e-12 * max(float(np.max(np.abs(history))), 1.0)
        or not 0.0 < fraction < 1.0
    ):
        raise ValueError("horizon-completeness inputs are invalid")
    peak = causal_quadratic_peak(times, history)
    scale = max(abs(peak.interpolated_value), np.finfo(float).tiny)
    start = max(0, int(np.floor((1.0 - fraction) * times.size)))
    terminal_to_peak = float(abs(history[-1]) / scale)
    final_range = float(np.ptp(history[start:]) / scale)
    derivatives = np.gradient(history, times)
    slope = float(
        abs(np.mean(derivatives[-min(5, derivatives.size) :]))
        * (times[-1] - times[0])
        / scale
    )
    peak_fraction = float(
        (peak.interpolated_time_seconds - times[0])
        / (times[-1] - times[0])
    )
    complete = bool(
        terminal_to_peak <= maximum_terminal_to_peak
        and final_range <= maximum_final_window_range_to_peak
        and slope <= maximum_terminal_slope_horizon_to_peak
    )
    return CausalHorizonCompleteness(
        peak_value=peak.interpolated_value,
        terminal_to_peak=terminal_to_peak,
        final_window_range_to_peak=final_range,
        terminal_slope_horizon_to_peak=slope,
        peak_time_fraction=peak_fraction,
        complete=complete,
    )
