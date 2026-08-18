from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_rank_adaptive_common_memory_audit_wp10c9d6c7c3b5c4f25q as f25q


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_is_locked_and_forbids_new_truth(monkeypatch):
    for name, value in f25q.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25q._validate_manifest(require_clean=False)
    budget = frozen["contract"]["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0


def test_rank_pass_uses_stability_and_alignment():
    gates = {
        "common_basis_orthogonality_defect_max": 1.0e-10,
        "augmented_restriction_lifting_identity_defect_max": 1.0e-10,
        "augmented_restriction_stable_annihilation_defect_max": 1.0e-10,
        "remaining_unresolved_spectral_abscissa_per_second_max": -1.0e-8,
        "minimum_cross_anchor_basis_principal_cosine": 0.75,
    }
    metrics = {
        "common_modal_basis_orthogonality_defect": 1.0e-14,
        "common_augmented_restriction_lifting_identity_defect": 1.0e-14,
        "common_augmented_restriction_stable_annihilation_defect": 1.0e-14,
        "remaining_common_unresolved_spectral_abscissa_per_second": -2.0,
    }
    assert f25q._rank_pass(metrics, metrics, 0.99, gates)
    unstable = dict(
        metrics,
        remaining_common_unresolved_spectral_abscissa_per_second=1.0,
    )
    assert not f25q._rank_pass(metrics, unstable, 0.99, gates)
    assert not f25q._rank_pass(metrics, metrics, 0.5, gates)


def test_classification_preserves_scientific_boundaries():
    selected = {"common_rank": 34, "memory_order": 120}
    assert f25q._classification(True, 1, selected) == (
        f25q.PASS_CLASSIFICATION,
        "definitions_only_R32_rank_adaptive_memory_online_prototype_manifest",
        True,
    )
    assert f25q._classification(True, 1, None) == (
        f25q.CAP_FAIL_CLASSIFICATION,
        "definitions_only_local_fiber_parametric_memory_architecture_manifest",
        False,
    )
    assert f25q._classification(True, 0, None)[0] == f25q.CHART_FAIL_CLASSIFICATION
    assert f25q._classification(False, 1, selected)[0] == f25q.NUMERICAL_FAIL_CLASSIFICATION


def test_candidate_score_is_one_at_all_limits():
    gates = {
        "maximum_normalized_dynamic_transfer_relative_error_max": 0.25,
        "RMS_normalized_dynamic_transfer_relative_error_max": 0.10,
        "DC_normalized_dynamic_transfer_relative_error_max": 0.10,
        "maximum_normalized_total_transfer_relative_error_max": 0.25,
        "RMS_normalized_total_transfer_relative_error_max": 0.10,
        "DC_normalized_total_transfer_relative_error_max": 0.10,
    }
    metrics = {}
    for prefix in ("training", "heldout"):
        for key, value in gates.items():
            metrics[f"{prefix}_{key.removesuffix('_max')}"] = value
    assert np.isclose(f25q._candidate_score(metrics, gates), 1.0)


def test_canonical_result_when_available():
    summary_path = f25q.CANONICAL_DIRECTORY / "summary.json"
    if not summary_path.exists():
        return
    summary = _read(summary_path)
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    assert summary["new_full_560_direction_generator_assemblies"] == 0
    assert not summary["physical_failure_detected"]
    for line in (f25q.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25q.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
