from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / (
    "run_causal_inner_face36_fixed_q_primary_evidence_aggregation_manifest_"
    "wp10c9d6c7c3b5c4f24e14q.py"
)
SPEC = importlib.util.spec_from_file_location("e14q_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_contract_is_aggregation_only_and_preserves_parent():
    contract = MODULE.CONTRACT
    assert contract["definitions_only"] is True
    assert contract["aggregation_only"] is True
    assert contract["physical_root_execution_authorized"] is False
    assert contract["historical_parent_classification_preserved"] == (
        "bounded_continuation_failed"
    )


def test_contract_binds_primary_scientific_cost_and_policy_evidence():
    evidence = MODULE.CONTRACT["binding_primary_evidence"]
    assert evidence["accepted_main_BDF2_roots"] == 4
    assert evidence["accepted_main_horizon_seconds"] == 4.0e-7
    assert evidence["suffix_replay_bitwise"] is True
    assert evidence["matched_endpoint_half_step_audit_passed"] is True
    assert evidence["maximum_warm_to_cold_wall_time_ratio"] == 0.75
    assert evidence["certified_control_comparison_residual"] == 1.0e-12
    assert evidence["maximum_reaction_action_relative_difference"] == 1.0e-8


def test_contract_authorizes_only_heldout_manifest_after_pass():
    assert MODULE.CONTRACT["decision"]["pass_authorizes_only"] == (
        "definitions_only_heldout_continuation_manifest"
    )
    assert all(MODULE.CONTRACT["hard_stops"].values())


def test_parent_lock_closes_current_canonical_packages():
    lock = MODULE._parent_lock()
    assert lock["historical_retry_summary"]["classification"] == (
        "bounded_continuation_failed"
    )
    assert lock["policy_certificate_summary"]["passed"] is True


def test_frozen_manifest_when_available():
    path = MODULE.ARTIFACT_DIRECTORY / "summary.json"
    if not path.exists():
        pytest.skip("prospective manifest is not frozen yet")
    summary = MODULE._read(path)
    assert summary["passed"] is True
    assert summary["primary_evidence_aggregation_authorized"] is True
    assert summary["heldout_continuation_manifest_authorized"] is False
    MODULE._checksums(MODULE.ARTIFACT_DIRECTORY)
