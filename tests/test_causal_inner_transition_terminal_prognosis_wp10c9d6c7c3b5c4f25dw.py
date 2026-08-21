from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_transition_terminal_prognosis_wp10c9d6c7c3b5c4f25dw as prognosis


def test_flat_or_increasing_tail_never_authorizes_more_microsteps() -> None:
    fractions = np.linspace(0.9, 1.0, 17)
    metrics = prognosis._prognosis(fractions, np.full(12, 1200.0))
    assert not metrics["every_tail_fraction_decreases"]
    assert metrics["forecast_roots_to_hidden_exit"] is None
    assert not metrics["bounded_extension_warranted"]


def test_nearby_decreasing_crossing_within_cost_can_authorize_extension() -> None:
    fractions = np.concatenate((np.ones(11), np.linspace(0.55, 0.30, 6)))
    metrics = prognosis._prognosis(fractions, np.full(12, 60.0))
    assert metrics["every_tail_fraction_decreases"]
    assert metrics["forecast_roots_to_hidden_exit"] <= 24
    assert metrics["forecast_wall_hours_to_hidden_exit"] <= 10.0
    assert metrics["bounded_extension_warranted"]


def test_decreasing_but_remote_crossing_is_not_enough() -> None:
    fractions = np.concatenate((np.ones(11), np.linspace(1.0, 0.99, 6)))
    metrics = prognosis._prognosis(fractions, np.full(12, 1200.0))
    assert metrics["every_tail_fraction_decreases"]
    assert not metrics["crossing_within_root_budget"]
    assert not metrics["bounded_extension_warranted"]
