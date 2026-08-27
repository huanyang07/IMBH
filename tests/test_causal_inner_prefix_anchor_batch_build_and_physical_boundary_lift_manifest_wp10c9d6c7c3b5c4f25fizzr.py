import json

import pytest

import run_causal_inner_prefix_anchor_batch_build_and_physical_boundary_lift_manifest_wp10c9d6c7c3b5c4f25fizzr as runner


def test_manifest_separates_port_payloads_from_missing_slow_forcing():
    contract=runner._contract();batch=contract["prefix_port_batch"]
    assert batch["candidate_anchor_count"]==913
    assert batch["native_radial_cells"]==112
    assert not batch["slow_forcing_b_included"]
    assert batch["new_truth_calls"]==0


def test_boundary_contract_is_characteristic_and_fail_closed():
    boundary=runner._contract()["boundary_lift"]
    assert boundary["inner"]["expected_incoming_count"]==0
    assert boundary["outer"]["expected_incoming_count"]==11
    assert not boundary["outer"]["cycle_wide_loading_complete"]
    assert not runner._contract()["claim_boundary"]["complete_cycle_execution_authorized"]


def test_parent_prefix_cover_is_hash_locked():
    hashes,_=runner._validate_parent();assert "summary.json" in hashes


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(),reason="manifest not frozen")
def test_canonical_manifest_closes():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary=json.loads((runner.CANONICAL_DIRECTORY/"summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert summary["candidate_anchor_count"]==913
    assert not summary["complete_cycle_execution_authorized"]
    assert summary["complete_cycle_steps"]==0
