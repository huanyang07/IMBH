"""Strict status semantics for the conservative metric-chart retraction.

The original atlas implementation is retained byte-for-byte because it is
hash-locked by completed scientific packages.  Its failed-return mapping is
ambiguous: the physical audit's ``passed`` value overwrites the nonlinear
closure value.  This versioned adapter preserves the numerical algorithm and
adds independent physical, closure, and chart-condition statuses.
"""

from __future__ import annotations

from typing import Any

from .conservative_metric_chart_atlas import (
    ConservativeMetricChart,
    MetricRetractionPolicy,
    block_whitening_transform,
    metric_augmented_jacobian,
    metric_transport_retract as _legacy_metric_transport_retract,
)


def strict_retraction_status(
    legacy_metrics: dict[str, Any],
    policy: MetricRetractionPolicy,
) -> dict[str, Any]:
    """Return explicit, fail-closed statuses for a legacy retraction result."""
    result = dict(legacy_metrics)
    physical_passed = bool(result["passed"])
    nonlinear_closure_passed = bool(
        result["original_coordinate_residual_infinity"]
        <= policy.original_coordinate_tolerance
        and result["metric_coordinate_residual_infinity"]
        <= policy.metric_coordinate_tolerance
        and result["gauge_residual_infinity"] <= policy.gauge_tolerance
    )
    chart_condition_passed = bool(
        result["maximum_metric_augmented_condition_number"]
        <= policy.maximum_metric_augmented_condition
    )
    result.update(
        {
            "legacy_passed": physical_passed,
            "physical_passed": physical_passed,
            "nonlinear_closure_passed": nonlinear_closure_passed,
            "chart_condition_passed": chart_condition_passed,
            "passed": bool(
                physical_passed
                and nonlinear_closure_passed
                and chart_condition_passed
            ),
            "status_semantics": "strict_v2",
        }
    )
    return result


def metric_transport_retract_strict(
    *,
    exact_chart,
    model,
    initial_state,
    target_original_coordinate,
    gauge_basis,
    anchor_delta,
    anchor_metric_augmented,
    chart: ConservativeMetricChart,
    policy: MetricRetractionPolicy,
):
    """Run the frozen retraction algorithm and correct only its status map."""
    state, matrix, legacy_metrics = _legacy_metric_transport_retract(
        exact_chart=exact_chart,
        model=model,
        initial_state=initial_state,
        target_original_coordinate=target_original_coordinate,
        gauge_basis=gauge_basis,
        anchor_delta=anchor_delta,
        anchor_metric_augmented=anchor_metric_augmented,
        chart=chart,
        policy=policy,
    )
    return state, matrix, strict_retraction_status(legacy_metrics, policy)


__all__ = (
    "ConservativeMetricChart",
    "MetricRetractionPolicy",
    "block_whitening_transform",
    "metric_augmented_jacobian",
    "metric_transport_retract_strict",
    "strict_retraction_status",
)
