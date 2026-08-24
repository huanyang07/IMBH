from __future__ import annotations

import pytest

from imri_qpe.layer3_minidisk_1d.adaptive_metric_chart_continuation import (
    AdaptiveMetricChartPolicy,
    blind_midpoint_required,
    strict_chart_failure_is_retryable,
    transition_after_attempt,
)


def _policy() -> AdaptiveMetricChartPolicy:
    return AdaptiveMetricChartPolicy(2.5e-4, 2.0e-3, 2.0, 4, 4)


def test_growth_requires_four_accepts_ending_at_blind_segment() -> None:
    policy = _policy()
    assert blind_midpoint_required(96, policy)
    transition = transition_after_attempt(
        policy=policy,
        span_seconds=1.0e-3,
        tentative_segment_number=96,
        accepted=True,
        physical_failure=False,
        accepted_since_growth=3,
    )
    assert transition["next_span_seconds"] == 2.0e-3
    assert transition["accepted_since_growth"] == 0


def test_local_chart_failure_halves_without_becoming_physical_failure() -> None:
    metrics = {
        "physical_passed": True,
        "nonlinear_closure_passed": False,
        "chart_condition_passed": False,
    }
    assert strict_chart_failure_is_retryable(metrics)
    transition = transition_after_attempt(
        policy=_policy(),
        span_seconds=2.0e-3,
        tentative_segment_number=97,
        accepted=False,
        physical_failure=False,
        accepted_since_growth=0,
    )
    assert transition["next_span_seconds"] == 1.0e-3
    assert transition["stop_reason"] is None


def test_physical_failure_stops_and_minimum_numerical_failure_stops() -> None:
    physical = transition_after_attempt(
        policy=_policy(),
        span_seconds=1.0e-3,
        tentative_segment_number=97,
        accepted=False,
        physical_failure=True,
        accepted_since_growth=2,
    )
    assert physical["stop_reason"] == "physical_failure"
    numerical = transition_after_attempt(
        policy=_policy(),
        span_seconds=2.5e-4,
        tentative_segment_number=97,
        accepted=False,
        physical_failure=False,
        accepted_since_growth=2,
    )
    assert numerical["stop_reason"] == "minimum_span_numerical_failure"


def test_policy_rejects_growth_above_two() -> None:
    with pytest.raises(ValueError):
        AdaptiveMetricChartPolicy(2.5e-4, 2.0e-3, 2.1, 4, 4)
