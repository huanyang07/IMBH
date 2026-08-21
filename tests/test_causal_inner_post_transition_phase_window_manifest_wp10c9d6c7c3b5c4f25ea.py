from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_post_transition_phase_window_manifest_wp10c9d6c7c3b5c4f25ea as manifest


def test_post_transition_window_is_bounded_and_root_free() -> None:
    contract = manifest._contract()
    phase = contract["phase_discretization"]
    budget = contract["truth_budget"]
    assert phase["basis_rank"] == 4
    assert phase["node_count"] == 8
    assert phase["full_duration_seconds"] == 2.0e-7
    assert phase["no_sequential_BDF_microsteps"]
    assert budget["maximum_exact_continuous_fixed_Q_rate_calls"] == 43
    assert budget["new_nonlinear_fixed_Q_roots"] == 0
    assert budget["new_BDF_microsteps"] == 0


def test_post_transition_window_does_not_claim_cycle_or_hot_exit() -> None:
    decision = manifest._contract()["decision"]
    assert decision["pass_does_not_claim_hot_exit"]
    assert not decision["predictive_cycle_execution_authorized"]
    assert not decision["reduced_slow_evolution_authorized"]
    assert decision["either_outcome_authorizes"] == "definitions_only_cycle_map_architecture_decision"
