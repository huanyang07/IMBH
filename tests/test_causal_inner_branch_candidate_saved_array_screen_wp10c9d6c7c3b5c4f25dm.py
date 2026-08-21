from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_branch_candidate_saved_array_screen_wp10c9d6c7c3b5c4f25dm as screen


def test_saved_screen_contract_is_fail_closed() -> None:
    architecture = screen._read(screen.PARENT_ARCHITECTURE)
    dependency = architecture["branch_first_dependency"]
    assert architecture["prospective_branch_candidate_screen"][
        "must_select_distinct_cold_and_hot_candidates"
    ]
    assert dependency["branch_candidate_must_not_be_the_exact_20ms_transition_anchor"]
    assert screen.MAXIMUM_HIDDEN_SECANT_FRACTION == 0.25
    assert screen.MINIMUM_TRANSITION_MACRO_DISTANCE == 0.05


def test_saved_nonsealed_candidate_screen() -> None:
    metrics, arrays, audit = screen._screen(require_clean=False)
    assert metrics["classification"] == screen.COLD_ONLY_CLASSIFICATION
    assert metrics["candidate_count"] == 8
    assert metrics["cold_candidate_count"] >= 1
    assert metrics["hot_candidate_count"] == 0
    assert metrics["selected_cold_candidate"] == "full_model_12ms"
    assert metrics["selected_hot_candidate"] is None
    assert not metrics["distinct_cold_hot_pair_supported"]
    assert not metrics["sealed_16ms_opened"]
    assert not metrics["exact_20ms_transition_anchor_selected"]
    assert np.all(arrays["candidate_authentic_history_available"])
    assert np.all(arrays["candidate_physical_guard_pass"])
    assert np.max(arrays["candidate_saved_secant_hidden_fraction"][:4]) < 0.25
    assert np.min(arrays["candidate_saved_secant_hidden_fraction"][4:]) > 0.99
    assert np.max(arrays["candidate_macro_distance_from_transition_anchor"][4:]) < 1.0e-3
    assert audit["checks"]["cold_candidate_supported"]
    assert not audit["checks"]["hot_candidate_supported"]
    assert audit["checks"]["no_truth"]
    assert audit["checks"]["sealed_budget"]
