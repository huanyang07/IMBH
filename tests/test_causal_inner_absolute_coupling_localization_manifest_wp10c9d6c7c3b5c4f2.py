from __future__ import annotations

import hashlib
import json

import run_causal_inner_absolute_coupling_localization_manifest_wp10c9d6c7c3b5c4f2 as c4f2


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_is_existing_state_only_and_preserves_stop():
    manifest = c4f2._manifest()
    assert manifest["definitions_only"]
    assert not manifest["new_trajectory"]
    assert not manifest["observable_memory_propagation_authorized"]
    assert not manifest["absolute_closure_fit_authorized"]
    assert manifest["response_certificate_preserved"]


def test_decomposition_is_exact_but_not_a_physical_lift():
    manifest = c4f2._manifest()
    decomposition = manifest["decomposition"]
    assert decomposition["exact_sum_closure_required"]
    assert decomposition["shared_parent_reference_is_diagnostic_not_a_physical_lift"]
    assert c4f2.TRANSITION_FACE not in c4f2.INTERIOR_CONTROL_FACES
    assert len(c4f2.INTERIOR_CONTROL_FACES) >= 2


def test_decision_thresholds_are_prospective():
    gates = c4f2._manifest()["prospective_gates"]
    assert gates["minimum_spatial_order"] == 0.75
    assert gates["minimum_error_direction_cosine"] == 0.90
    assert gates["maximum_decomposition_closure_defect"] == 1.0e-12
    assert gates["maximum_shared_parent_operator_fraction_of_middle_fine_difference_for_state_classification"] == 0.10
    assert gates["minimum_shared_parent_operator_fraction_for_operator_classification"] == 0.50


def test_canonical_summary_and_hashes():
    summary = _read(c4f2.SUMMARY_PATH)
    assert summary["passed"] and summary["definitions_only"]
    assert summary["parent_negative_result_preserved"]
    assert not summary["new_trajectory_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    for line in (c4f2.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((c4f2.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest() == expected
