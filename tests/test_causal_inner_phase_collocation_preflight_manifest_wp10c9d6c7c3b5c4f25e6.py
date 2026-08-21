from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_phase_collocation_preflight_manifest_wp10c9d6c7c3b5c4f25e6 as manifest


def test_saved_evidence_audit_is_fail_closed() -> None:
    audit = manifest._audit_saved_evidence()
    assert audit["cold_exact_full_model_rate_witnesses_available"]
    assert not audit["transition_saved_continuous_rate_witnesses_available"]
    assert not audit["post_transition_accepted_state_available"]
    assert not audit["complete_cycle_event_sequence_available"]


def test_contract_forbids_overclaiming_secants() -> None:
    contract = manifest._contract(manifest._audit_saved_evidence())
    boundary = contract["evidence_boundary"]
    assert boundary["transition_full_vector_field_claim_from_secants_forbidden"]
    assert boundary["post_transition_extrapolation_as_truth_forbidden"]
    assert not boundary["predictive_cycle_authorized"]
