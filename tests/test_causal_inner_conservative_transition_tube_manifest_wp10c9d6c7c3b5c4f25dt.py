from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_conservative_transition_tube_manifest_wp10c9d6c7c3b5c4f25dt as manifest


def test_surrogate_is_conservative_by_construction() -> None:
    surrogate = manifest._contract()["surrogate"]
    assert surrogate["state"] == "(q_entry_82,s_scalar)"
    assert "L(q_entry+ell_q(s))" in surrogate["coordinate_lift"]
    assert surrogate["online_y470_residual_calls"] == 0
    assert surrogate["online_truth_calls"] == 0


def test_rank_fit_is_train_only() -> None:
    policy = manifest._contract()["training_policy"]
    assert policy["basis_fit_uses_training_states_only"]
    assert policy["held_out_states_never_select_rank_or_coefficients"]
    assert set(policy["training_state_indices"]).isdisjoint(
        policy["held_out_state_indices"]
    )


def test_partial_terminal_ledger_is_not_relabelled_as_an_impulse() -> None:
    contract = manifest._contract()
    assert contract["surrogate"]["partial_reset_is_not_a_hot_exit_impulse"]
    assert not contract["decision_policy"]["complete_impulse_fit_authorized"]
    assert not contract["decision_policy"]["reduced_cycle_authorized"]
