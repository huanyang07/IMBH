from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_curvature_adaptive_arclength_continuation_execution_wp10c9d6c7c3b5c4f25fi as target  # noqa: E402


def test_manifest_authorizes_exact_execution() -> None:
    lock = target._validate_manifest(require_clean=False)
    assert lock["summary"]["passed"]
    assert lock["summary"]["authorized_next"] == target.manifest.AUTHORIZED_NEXT
    assert lock["contract"]["truth_system"]["autonomous"]


def test_variable_step_ab2_is_exact_for_quadratic_path() -> None:
    previous_time = 0.2
    current_time = 0.5
    span = 0.7
    path = lambda value: np.asarray([value**2, 3.0 * value**2])
    rate = lambda value: np.asarray([2.0 * value, 6.0 * value])
    prediction = target._variable_step_ab2(
        path(current_time),
        rate(current_time),
        rate(previous_time),
        span,
        current_time - previous_time,
    )
    np.testing.assert_allclose(prediction, path(current_time + span), atol=2.0e-15)


def test_cubic_hermite_is_exact_for_cubic_path_and_rate() -> None:
    path = lambda value: np.asarray([value, value**2, value**3])
    rate = lambda value: np.asarray([1.0, 2.0 * value, 3.0 * value**2])
    left_time = 0.4
    span = 0.9
    for fraction in (0.0, 0.2, 0.5, 0.8, 1.0):
        coordinate, derivative = target._hermite(
            path(left_time),
            rate(left_time),
            path(left_time + span),
            rate(left_time + span),
            span,
            fraction,
        )
        sample_time = left_time + fraction * span
        np.testing.assert_allclose(coordinate, path(sample_time), atol=2.0e-15)
        np.testing.assert_allclose(derivative, rate(sample_time), atol=2.0e-14)


def test_endpoint_integral_defect_is_zero_for_constant_rate() -> None:
    left = np.asarray([1.0, -2.0])
    rate = np.asarray([3.0, 4.0])
    span = 0.25
    right = left + span * rate
    assert target._endpoint_integral_defect(left, rate, right, rate, span) < 1.0e-15


def test_section_root_localizes_cubic_crossing() -> None:
    left = np.asarray([-1.0])
    right = np.asarray([1.0])
    left_rate = np.asarray([2.0])
    right_rate = np.asarray([2.0])
    fraction = target._section_root_fraction(
        left, left_rate, right, right_rate, 1.0, np.asarray([0.0]), np.asarray([1.0])
    )
    assert fraction is not None
    assert abs(fraction - 0.5) < 1.0e-14


def test_parent_seed_has_complete_state_and_exact_previous_rate() -> None:
    data = target._parent_data()
    assert data["current_coordinate"].shape == (470,)
    assert data["current_state"].shape == (112, 5)
    assert data["previous_rate"].shape == (470,)
    assert data["previous_span_seconds"] == 2.5e-4


def test_canonical_package_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        target.CANONICAL_DIRECTORY / "continuation_execution_metrics.json"
    )
    assert summary["classification"] == metrics["classification"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert metrics["fixed_Q_physical_rate_calls"] == 0
    assert metrics["fixed_Q_reaction_calls"] == 0
