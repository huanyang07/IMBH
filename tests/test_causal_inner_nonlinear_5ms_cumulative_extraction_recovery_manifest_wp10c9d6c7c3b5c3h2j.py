from __future__ import annotations

import json

import run_causal_inner_nonlinear_5ms_cumulative_extraction_recovery_manifest_wp10c9d6c7c3b5c3h2j as manifest


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_scan_is_prospective_and_uses_declared_faces() -> None:
    payload = _read(manifest.MANIFEST_PATH)
    assert tuple(payload["candidate_coarse_face_indices"]) == manifest.CANDIDATE_COARSE_FACE_INDICES
    assert payload["minimum_consecutive_passing_surfaces"] == 2
    assert payload["no_shortened_window_may_certify"]
    assert tuple(payload["binding_window_seconds"]) == (0.002, 0.005)


def test_manifest_authorizes_only_operator_neutral_audit() -> None:
    summary = _read(manifest.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["cumulative_recovery_audit_authorized"]
    assert not summary["new_propagation_authorized"]
    assert not summary["fourth_duration_rung_manifest_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
