from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import run_causal_inner_transition_phase_collocation_manifest_wp10c9d6c7c3b5c4f25e8 as manifest


def test_transition_contract_uses_exact_rate_witnesses() -> None:
    contract = manifest._contract()
    witness = contract["vector_field_witness"]
    assert witness["maximum_calls"] == 8
    assert witness["nonlinear_roots"] == 0
    assert witness["propagated_states"] == 0
    assert witness["secants_are_not_truth"]


def test_transition_pass_scope_is_bounded() -> None:
    decision = manifest._contract()["decision"]
    assert decision["post_transition_execution_not_yet_authorized"]
    assert decision["hot_exit_not_claimed"]
    assert not decision["predictive_cycle_authorized"]
