import json

import numpy as np
import pytest

import run_causal_inner_cycle_physical_driver_branch_and_event_interpolator_structure_certificate_wp10c9d6c7c3b5c4f25fizzv1 as runner


def test_corrected_parent_is_hash_locked():
    _, original, correction = runner._validate_parent()
    assert original["binding_structure_gates"]["complete_cycle_steps"] == 0
    assert correction["binding_correction"]["event_simplices6_forbidden"]


def test_fixture_has_full_driver_branch_and_event_structure():
    driver, branch, events, additions, conservation, normal = runner._fixture()
    assert driver["slow_forcing1232_per_second"].shape == (5, 6, 2, 1232)
    assert branch["radial_matrices112x11x11"].shape == (16, 112, 11, 11)
    assert events["reduced_guard_normals5"].shape == (16, 5)
    assert additions["branch_simplices"].shape == (2, 6)
    assert additions["event_simplices"].shape == (2, 5)
    assert np.linalg.norm(conservation @ normal - np.eye(4)) <= 2e-12


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="certificate not executed")
def test_canonical_interpolator_structure_certificate_closes():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    metrics = json.loads((runner.CANONICAL_DIRECTORY / "interpolator_metrics.json").read_text())
    assert summary["passed"] and summary["cycle_interpolator_structure_certified"]
    assert summary["event_guard_sheet_dimension_corrected"]
    assert metrics["minimum_branch_source_nullity"] >= 4
    assert metrics["branch_boundary_incoming_counts"] == [[0, 11], [0, 11]]
    assert metrics["all_anchor_reproductions_bitwise"]
    assert metrics["outside_hull_rejections"] == 2
    assert metrics["checkpoint_roundtrip_bitwise"]
    assert not summary["physical_payloads_acquired"]
    assert not summary["complete_cycle_execution_authorized"]
    assert summary["complete_cycle_steps"] == 0
