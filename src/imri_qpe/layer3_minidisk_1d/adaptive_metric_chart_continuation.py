"""Fail-closed adaptive policy for strict conservative metric-chart patches."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AdaptiveMetricChartPolicy:
    minimum_span_seconds: float
    maximum_span_seconds: float
    growth_factor: float
    accepted_segments_before_growth: int
    blind_midpoint_frequency: int

    def __post_init__(self) -> None:
        values = (
            self.minimum_span_seconds,
            self.maximum_span_seconds,
            self.growth_factor,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("adaptive spans and growth must be finite and positive")
        if self.minimum_span_seconds > self.maximum_span_seconds:
            raise ValueError("minimum span exceeds maximum span")
        if self.growth_factor > 2.0:
            raise ValueError("adaptive metric-chart growth may not exceed two")
        if self.accepted_segments_before_growth <= 0:
            raise ValueError("accepted-segment growth count must be positive")
        if self.blind_midpoint_frequency <= 0:
            raise ValueError("blind midpoint frequency must be positive")


def blind_midpoint_required(
    tentative_segment_number: int,
    policy: AdaptiveMetricChartPolicy,
) -> bool:
    return int(tentative_segment_number) % policy.blind_midpoint_frequency == 0


def strict_chart_failure_is_retryable(strict_metrics: dict) -> bool:
    """A physically admissible local chart failure is a span rejection."""
    return bool(
        strict_metrics["physical_passed"]
        and (
            not strict_metrics["nonlinear_closure_passed"]
            or not strict_metrics["chart_condition_passed"]
        )
    )


def transition_after_attempt(
    *,
    policy: AdaptiveMetricChartPolicy,
    span_seconds: float,
    tentative_segment_number: int,
    accepted: bool,
    physical_failure: bool,
    accepted_since_growth: int,
) -> dict:
    """Return the next span without ever mutating accepted history."""
    span = float(span_seconds)
    if physical_failure:
        return {
            "next_span_seconds": span,
            "accepted_since_growth": 0,
            "stop_reason": "physical_failure",
        }
    if accepted:
        count = int(accepted_since_growth) + 1
        next_span = span
        if (
            blind_midpoint_required(tentative_segment_number, policy)
            and count >= policy.accepted_segments_before_growth
            and span < policy.maximum_span_seconds
        ):
            next_span = min(
                policy.growth_factor * span,
                policy.maximum_span_seconds,
            )
            count = 0
        return {
            "next_span_seconds": next_span,
            "accepted_since_growth": count,
            "stop_reason": None,
        }
    if span > policy.minimum_span_seconds:
        return {
            "next_span_seconds": max(
                0.5 * span,
                policy.minimum_span_seconds,
            ),
            "accepted_since_growth": 0,
            "stop_reason": None,
        }
    return {
        "next_span_seconds": span,
        "accepted_since_growth": 0,
        "stop_reason": "minimum_span_numerical_failure",
    }


__all__ = (
    "AdaptiveMetricChartPolicy",
    "blind_midpoint_required",
    "strict_chart_failure_is_retryable",
    "transition_after_attempt",
)
