import json

import pytest

import run_causal_inner_legacy_cycle_evidence_compatibility_audit_wp10c9d6c7c3b5c4f25fizzp1 as runner


def test_audit_partitions_legacy_payloads_without_direct_reuse():
    metrics = runner._audit()
    assert metrics["passed"]
    assert metrics["direct_binding_reuse_count"] == 0
    assert metrics["facts"]["trajectory_primitive_shape"] == (65, 112, 5)
    assert metrics["facts"]["target_field_count"] == 11
    assert not metrics["facts"]["old_cycle_observed"]
    assert not metrics["facts"]["old_hot_exit_observed"]
    assert metrics["facts"]["time_coverage_fraction"] < 3.0e-8
    assert not metrics["complete_cycle_execution_authorized"]


def test_five_field_boundary_is_seed_not_eleven_field_certificate():
    metrics = runner._audit()
    inventory = {row["label"]: row for row in metrics["compatibility_inventory"]}
    boundary = inventory["legacy_fixed_exterior_characteristics"]
    assert boundary["classification"] == "candidate_seed_after_deterministic_lift_and_new_audit"
    assert metrics["facts"]["old_boundary_field_count"] == 5
    assert metrics["facts"]["old_boundary_inner_incoming_characteristics"] == 0


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="audit not canonicalized")
def test_canonical_audit_closes():
    hashes = runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert hashes and summary["passed"]
    assert summary["direct_binding_reuse_count"] == 0
    assert summary["complete_cycle_steps"] == 0
    assert summary["authorized_next"] == runner.AUTHORIZED_NEXT
