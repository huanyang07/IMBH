from __future__ import annotations

import hashlib
import json

import run_causal_inner_finite_memory_selection_manifest_wp10c9d6c7c3b5c4f25h as f25h


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_parent_is_stable_hash_locked_and_not_physical_failure():
    summary, metrics, hashes = f25h._validate_parent()
    assert summary["passed"]
    assert metrics["remaining_unresolved_spectral_abscissa_per_second"] < 0.0
    assert not summary["physical_failure_detected"]
    assert "promotion.npz" in hashes


def test_memory_candidates_fit_online_dimension_budget():
    contract = f25h._contract()
    stage = contract["balanced_truncation"]
    assert stage["candidate_memory_orders"] == (0, 2, 4, 6)
    assert stage["base_resolved_dimension"] == 106
    assert stage["base_resolved_dimension"] + max(stage["candidate_memory_orders"]) <= 114
    assert stage["selection_rule"] == "smallest_candidate_passing_every_binding_gate"


def test_dynamic_memory_error_and_dissipation_are_binding():
    gates = f25h._contract()["candidate_pass_requires"]
    assert gates["maximum_normalized_dynamic_transfer_relative_error_max"] == 0.25
    assert gates["RMS_normalized_dynamic_transfer_relative_error_max"] == 0.10
    assert gates["DC_normalized_dynamic_transfer_relative_error_max"] == 0.10
    assert gates["reduced_spectral_abscissa_per_second_max"] < 0.0
    assert gates["lyapunov_certificate_minimum_eigenvalue_min"] > 0.0


def test_budget_forbids_truth_and_new_generator():
    budget = f25h._contract()["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_generator_assemblies"] == 0
    assert budget["allowed_truth_anchors"] == 0


def test_no_pass_uses_predeclared_larger_pde_fallback():
    decision = f25h._contract()["decision"]
    assert "larger_conservative_coarse_PDE" in decision["no_candidate_passes"]


def test_canonical_manifest_when_available():
    summary_path = f25h.ARTIFACT_DIRECTORY / "summary.json"
    if not summary_path.exists():
        return
    summary = _read(summary_path)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["maximum_candidate_online_dimension"] == 112
    assert not summary["memory_fit_executed"]
    for line in (f25h.ARTIFACT_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25h.ARTIFACT_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
