import json

import pytest

import run_causal_inner_legacy_cycle_evidence_compatibility_manifest_wp10c9d6c7c3b5c4f25fizzp as runner


def test_contract_preserves_field_and_cycle_mismatch():
    contract = runner._contract()
    checks = contract["binding_checks"]
    assert checks["legacy_profile_shape"] == [112, 5]
    assert checks["target_field_count"] == 11
    assert not checks["old_cycle_observed"]
    assert not checks["old_hot_exit_observed"]
    assert checks["direct_binding_reuse_count"] == 0
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


def test_parent_and_all_legacy_inputs_are_hash_locked():
    parent_hashes, legacy_hashes = runner._validate_parent()
    assert "summary.json" in parent_hashes
    assert set(legacy_hashes) == set(runner.LEGACY_INPUTS)


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_manifest_closes():
    hashes = runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert hashes and summary["passed"] and summary["definitions_only"]
    assert summary["authorized_next"] == runner.AUTHORIZED_NEXT
    assert summary["complete_cycle_steps"] == 0
