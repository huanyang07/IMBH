from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_selected_time_absolute_coupling_localization_wp10c9d6c7c3b5c4f3 as c4f3


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_transition_failure_is_spatially_localized():
    summary = _read(c4f3.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["transition_only_failure"]
    assert summary["passing_interior_control_faces"] == 3
    assert not summary["face_metrics"]["48"]["passed"]
    for face in ("44", "46", "47"):
        assert summary["face_metrics"][face]["passed"]
        assert summary["face_metrics"][face]["order"] > 1.9


def test_diagnostic_parent_lift_is_rejected_for_causal_assignment():
    summary = _read(c4f3.SUMMARY_PATH)
    assert summary["classification"] == "fine_complement_dominates_absolute_coupling_difference"
    assert summary["transition_fine_complement_fraction_of_middle_fine_difference"] > 1.0
    assert summary["transition_shared_parent_operator_fraction_of_middle_fine_difference"] > 1.0
    assert summary["decomposition_closure_defect"] == 0.0
    assert summary["authorized_next"] == "definitions_only_fine_complement_exact_JVP_manifest"


def test_no_reduction_authorization_or_physical_failure():
    summary = _read(c4f3.SUMMARY_PATH)
    assert not summary["physical_failure_detected"]
    assert summary["response_certificate_preserved"]
    assert not summary["absolute_closure_fit_authorized"]
    assert not summary["observable_memory_propagation_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_decisive_arrays_and_hashes_close():
    with np.load(c4f3.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        assert arrays["actual_fluxes"].shape == (3, 4, 6, 3)
        assert arrays["reference_fluxes"].shape == (3, 4, 6, 3)
    for line in (c4f3.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((c4f3.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest() == expected
