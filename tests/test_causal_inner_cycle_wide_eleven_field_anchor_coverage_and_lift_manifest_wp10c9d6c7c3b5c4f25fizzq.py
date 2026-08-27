import json

import pytest

import run_causal_inner_cycle_wide_eleven_field_anchor_coverage_and_lift_manifest_wp10c9d6c7c3b5c4f25fizzq as runner


def test_manifest_corrects_physical_grid_without_erasing_94_cell_proof():
    contract=runner._contract();grid=contract["grid_correction"]
    assert grid["physical_context_cells"]==112
    assert grid["eleven_field_global_dimension"]==1232
    assert grid["previous_periodic_proof_cells"]==94
    assert grid["previous_certificate_scope_preserved"]=="94-cell periodic scalability proof"
    assert grid["production_size_claim_superseded"]
    assert not grid["radial_remap_required_for_native_lift"]
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


def test_native_lift_and_full_cycle_coverage_are_distinct():
    contract=runner._contract()
    assert contract["native_lift"]["native_cellwise_lift"]
    assert contract["coverage"]["prefix_coverage_cannot_establish_cycle_coverage"]
    assert contract["physical_grid_global_AP_dry_run"]["global_state_dimension"]==1232


def test_parent_audit_is_hash_locked():
    hashes,_=runner._validate_parent();assert "summary.json" in hashes


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(),reason="manifest not frozen")
def test_canonical_manifest_closes():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary=json.loads((runner.CANONICAL_DIRECTORY/"summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert summary["physical_grid_cells"]==112
    assert summary["complete_cycle_steps"]==0
