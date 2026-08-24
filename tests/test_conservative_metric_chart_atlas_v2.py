from __future__ import annotations

from imri_qpe.layer3_minidisk_1d.conservative_metric_chart_atlas import (
    MetricRetractionPolicy,
)
from imri_qpe.layer3_minidisk_1d.conservative_metric_chart_atlas_v2 import (
    strict_retraction_status,
)


def _policy() -> MetricRetractionPolicy:
    return MetricRetractionPolicy(
        maximum_iterations=8,
        refresh_iteration_reserve=2,
        maximum_exact_refreshes=1,
        line_factors=(1.0, 0.5),
        original_coordinate_tolerance=1.0e-10,
        metric_coordinate_tolerance=1.0e-9,
        gauge_tolerance=1.0e-10,
        maximum_anchor_departure=0.1,
        maximum_metric_augmented_condition=10.0,
    )


def test_physical_pass_cannot_overwrite_failed_nonlinear_closure() -> None:
    result = strict_retraction_status(
        {
            "passed": True,
            "original_coordinate_residual_infinity": 7.9e-2,
            "metric_coordinate_residual_infinity": 1.2e-1,
            "gauge_residual_infinity": 1.5e-2,
            "maximum_metric_augmented_condition_number": 54.3,
        },
        _policy(),
    )
    assert result["legacy_passed"] is True
    assert result["physical_passed"] is True
    assert result["nonlinear_closure_passed"] is False
    assert result["chart_condition_passed"] is False
    assert result["passed"] is False


def test_strict_status_requires_all_three_independent_gates() -> None:
    result = strict_retraction_status(
        {
            "passed": True,
            "original_coordinate_residual_infinity": 1.0e-12,
            "metric_coordinate_residual_infinity": 2.0e-12,
            "gauge_residual_infinity": 3.0e-13,
            "maximum_metric_augmented_condition_number": 2.0,
        },
        _policy(),
    )
    assert result["physical_passed"] is True
    assert result["nonlinear_closure_passed"] is True
    assert result["chart_condition_passed"] is True
    assert result["passed"] is True
