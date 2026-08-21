from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_cold_branch_anchor_preflight_wp10c9d6c7c3b5c4f25dy as preflight


def test_oblique_rate_decomposition_is_exact_for_coordinate_split() -> None:
    geometry = {
        "R": np.hstack((np.eye(2), np.zeros((2, 3)))),
        "L": np.vstack((np.eye(2), np.zeros((3, 2)))),
        "Z": np.vstack((np.zeros((2, 3)), np.eye(3))),
        "Q": np.hstack((np.zeros((3, 2)), np.eye(3))),
    }
    rate = np.arange(1.0, 6.0)
    result = preflight._decompose_rate(rate, geometry)
    assert result["decomposition_relative_defect"] < 1.0e-15
    assert np.array_equal(result["macro_rate"], rate[:2])
    assert np.array_equal(result["hidden_rate"], rate[2:])


def test_first_complete_pass_stops_selection() -> None:
    metrics = [
        {"complete_pass": False},
        {"complete_pass": True},
        {"complete_pass": True},
    ]
    assert preflight._select_first_pass(metrics) == 1
    assert preflight._select_first_pass(metrics[:1]) is None


def test_candidate_state_ladder_contains_only_four_revealed_times() -> None:
    states = preflight._candidate_states()
    assert tuple(states) == preflight.manifest.CANDIDATE_TIMES_SECONDS
    assert all(state.shape == (112, 5) for state in states.values())


def test_correct_exact_chart_helper_exists() -> None:
    assert hasattr(preflight.exact_chart, "_model_and_inputs")
    assert not hasattr(preflight.exact_chart, "_model_inputs")
