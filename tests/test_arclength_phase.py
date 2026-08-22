from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from imri_qpe.layer3_minidisk_1d.arclength_phase import (
    AdaptiveArclengthPolicy,
    FastRegimePolicy,
    arclength_picard_window,
    classify_fast_regime,
    normalized_phase_rate,
)


def test_normalized_phase_rate_has_unit_unweighted_norm() -> None:
    direction, speed = normalized_phase_rate(np.asarray((3.0, 4.0)))
    assert speed == 5.0
    np.testing.assert_allclose(direction, np.asarray((0.6, 0.8)))


def test_arclength_picard_is_exact_for_constant_direction_and_speed() -> None:
    rate = np.asarray((3.0, 4.0))

    def evaluator(_coordinate: np.ndarray, _time_seconds: float) -> np.ndarray:
        return rate

    result = arclength_picard_window(
        start_coordinate=np.asarray((1.0, -2.0)),
        start_time_seconds=7.0,
        arclength_span=0.25,
        basis=np.asarray(((0.6,), (0.8,))),
        evaluator=evaluator,
        node_count=5,
    )
    np.testing.assert_allclose(result["endpoint"], np.asarray((1.15, -1.8)))
    assert result["endpoint_time_seconds"] == pytest.approx(7.05)
    assert np.max(result["full_defects"]) < 2.0e-12
    assert np.max(result["time_mapping_defects"]) < 2.0e-12
    assert np.min(result["direction_cosines"]) > 1.0 - 2.0e-12


def test_arclength_removes_variable_speed_from_a_straight_path() -> None:
    def evaluator(coordinate: np.ndarray, _time_seconds: float) -> np.ndarray:
        return np.asarray((2.0 + coordinate[0], 0.0))

    result = arclength_picard_window(
        start_coordinate=np.zeros(2),
        start_time_seconds=0.0,
        arclength_span=0.1,
        basis=np.asarray(((1.0,), (0.0,))),
        evaluator=evaluator,
        node_count=5,
    )
    np.testing.assert_allclose(result["coordinates"][:, 1], 0.0)
    np.testing.assert_allclose(result["endpoint"], np.asarray((0.1, 0.0)))
    assert np.max(result["full_defects"]) < 2.0e-12


def test_adaptive_policy_fails_closed_at_retry_and_span_limits() -> None:
    policy = AdaptiveArclengthPolicy()
    assert policy.retry_span(0.025, 0) == 0.0125
    assert policy.next_span(0.025, growth_margin_passed=True) == pytest.approx(0.0375)
    assert policy.next_span(0.05, growth_margin_passed=True) == 0.05
    with pytest.raises(RuntimeError, match="retry budget"):
        policy.retry_span(0.025, 2)


def test_regime_candidates_require_persistence_and_geometry() -> None:
    policy = FastRegimePolicy()
    common = {
        "terminal_speed_ratio": 1.0,
        "closest_return_distance_over_local_span": 1.0,
        "closest_return_direction_cosine": 0.0,
        "policy": policy,
    }
    assert classify_fast_regime(
        legacy_exit_run=1, equilibrium_run=0, recurrence_run=0, **common
    ) == "continuing_fast_branch"
    assert classify_fast_regime(
        legacy_exit_run=2, equilibrium_run=0, recurrence_run=0, **common
    ) == "legacy_transverse_exit_candidate"
    assert classify_fast_regime(
        legacy_exit_run=0,
        equilibrium_run=2,
        recurrence_run=0,
        terminal_speed_ratio=1.0e-4,
        closest_return_distance_over_local_span=1.0,
        closest_return_direction_cosine=0.0,
        policy=policy,
    ) == "fast_equilibrium_candidate"
    assert classify_fast_regime(
        legacy_exit_run=0,
        equilibrium_run=0,
        recurrence_run=2,
        terminal_speed_ratio=1.0,
        closest_return_distance_over_local_span=0.05,
        closest_return_direction_cosine=0.999,
        policy=policy,
    ) == "recurrent_orbit_candidate"
