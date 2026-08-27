import json

import pytest

import run_causal_inner_cycle_interpolator_guard_sheet_dimension_correction_manifest_wp10c9d6c7c3b5c4f25fizzv0 as runner


def test_correction_preserves_parent_and_fixes_codimension():
    _, parent = runner._validate_parent()
    correction = runner._contract()
    assert "event_simplices6" in parent["schema_v2_extension"]["event_additions"]
    assert correction["binding_correction"]["event_guard_ambient_dimension"] == 5
    assert correction["binding_correction"]["event_guard_intrinsic_dimension"] == 4
    assert correction["binding_correction"]["event_simplices6_forbidden"]
    assert correction["claim_boundary"]["complete_cycle_steps"] == 0


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="correction not frozen")
def test_canonical_correction_is_prospective_and_nonexecuting():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert summary["supersedes_prior_interpolator_manifest"]
    assert summary["event_guard_intrinsic_dimension"] == 4
    assert summary["event_simplex_vertex_count"] == 5
    assert not summary["cycle_interpolator_certified"]
    assert not summary["complete_cycle_execution_authorized"]
