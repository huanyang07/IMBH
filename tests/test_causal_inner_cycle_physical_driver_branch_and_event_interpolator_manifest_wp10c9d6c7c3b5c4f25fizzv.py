import json

import pytest

import run_causal_inner_cycle_physical_driver_branch_and_event_interpolator_manifest_wp10c9d6c7c3b5c4f25fizzv as runner


def test_parent_boundary_action_is_hash_locked():
    assert "summary.json" in runner._validate_parent()


def test_contract_adds_missing_guard_geometry_and_convex_structure():
    contract = runner._contract()
    additions = contract["schema_v2_extension"]["event_additions"]
    assert "reduced_guard_normals5" in additions
    assert "event_simplices6" in additions
    assert contract["driver_interpolation"]["outside_hull"].startswith("fail closed")
    assert contract["branch_interpolation"]["boundary_reaudit"].endswith("0/11")
    assert contract["binding_structure_gates"]["complete_cycle_steps"] == 0


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_interpolator_manifest_is_definitions_only():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert summary["schema_v2_guard_geometry_frozen"]
    assert not summary["cycle_interpolator_certified"]
    assert not summary["physical_payloads_acquired"]
    assert not summary["complete_cycle_execution_authorized"]
    assert summary["complete_cycle_steps"] == 0
