from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_cold_branch_anchor_preflight_manifest_wp10c9d6c7c3b5c4f25dx as manifest


def test_candidates_are_fail_fast_and_exclude_sealed_states() -> None:
    policy = manifest._contract()["candidate_policy"]
    assert tuple(policy["times_seconds_in_fail_fast_order"]) == (0.012, 0.008, 0.005, 0.002)
    assert policy["evaluate_one_candidate_at_a_time"]
    assert policy["stop_at_first_complete_pass"]
    assert policy["sealed_16ms_state_forbidden"]
    assert policy["20ms_transition_state_forbidden"]


def test_preflight_has_no_root_or_propagation_budget() -> None:
    budgets = manifest._contract()["budgets"]
    assert budgets["maximum_exact_fixed_Q_rate_evaluations"] == 4
    assert budgets["complete_generator_assemblies"] == 0
    assert budgets["hidden_branch_roots"] == 0
    assert budgets["propagated_states"] == 0
    assert budgets["new_transition_microsteps"] == 0


def test_root_only_follows_a_complete_pass() -> None:
    policy = manifest._contract()["decision_policy"]
    assert policy["first_complete_pass"] == "authorize_definitions_only_single_cold_hidden_root"
    assert policy["root_not_executed_in_preflight"]
    assert policy["hot_branch_and_complete_impulse_blocked"]
