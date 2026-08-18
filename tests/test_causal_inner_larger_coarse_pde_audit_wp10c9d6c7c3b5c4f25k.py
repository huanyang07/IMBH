from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_larger_coarse_pde_audit_wp10c9d6c7c3b5c4f25k as f25k


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_is_hash_locked_and_authorizes_no_truth(monkeypatch):
    for name, value in f25k.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25k._validate_manifest(require_clean=False)
    budget = frozen["contract"]["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0
    assert budget["allowed_memory_coefficients_fit"] == 0


def test_R32_groups_split_every_R16_group_and_preserve_face_36_boundary():
    parent = f25k.projection_tools.parent._coarse_groups(112)
    groups = f25k._R32_groups(112)
    assert len(groups) == 32
    for index, group in enumerate(parent):
        assert groups[2 * index][0] == group[0]
        assert groups[2 * index + 1][1] == group[1]
        assert groups[2 * index][1] == groups[2 * index + 1][0]
    boundaries = {start for start, _ in groups} | {groups[-1][1]}
    assert 72 in boundaries


def test_R32_storage_uses_mapped_MJE_and_complete_thermal_stress():
    mapped = np.eye(10)
    height = 2.0 * np.eye(10)
    rows = np.ones((2, 5))
    original_groups = f25k._R32_groups
    original_dimension = f25k.manifest.STORAGE_DIMENSION
    original_cells = f25k.manifest.COARSE_CELLS
    try:
        f25k._R32_groups = lambda count: ((0, 1), (1, 2))
        f25k.manifest.STORAGE_DIMENSION = 10
        f25k.manifest.COARSE_CELLS = 2
        _, physical, scales = f25k._R32_storage_restriction(mapped, height, rows)
    finally:
        f25k._R32_groups = original_groups
        f25k.manifest.STORAGE_DIMENSION = original_dimension
        f25k.manifest.COARSE_CELLS = original_cells
    for field in range(5):
        expected = 1.0 if field in f25k.CONSERVATIVE_FIELDS else 3.0
        assert physical[field, field] == f25k.C * expected
        assert scales[field] == f25k.C * expected


def test_no_memory_error_is_zero_for_a_direct_transfer():
    direct = np.asarray(((2.0,),))
    transfer = np.repeat(direct[None, :, :], 3, axis=0).astype(complex)
    errors, metrics = f25k._no_memory_errors(
        transfer, direct, np.asarray(((1.0,),)), np.asarray(((1.0,),))
    )
    assert np.array_equal(errors, np.zeros(3))
    assert metrics["maximum_normalized_total_transfer_relative_error"] == 0.0


def test_classification_is_fail_closed():
    assert f25k._classification(
        {"projection_algebra_passed": True, "remaining_unresolved_strictly_stable": True, "dimension_budget_passed": False}
    ) == (f25k.DIMENSION_FAIL_CLASSIFICATION, None)
    assert f25k._classification(
        {
            "projection_algebra_passed": True,
            "remaining_unresolved_strictly_stable": True,
            "dimension_budget_passed": True,
            "no_memory_closure_passed": False,
        }
    ) == (
        f25k.CLOSURE_FAIL_CLASSIFICATION,
        "definitions_only_reduced_architecture_reassessment_manifest",
    )


def test_canonical_result_when_available():
    summary_path = f25k.CANONICAL_DIRECTORY / "summary.json"
    if not summary_path.exists():
        return
    summary = _read(summary_path)
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    assert summary["new_full_560_direction_generator_assemblies"] == 0
    assert summary["memory_coefficients_fit"] == 0
    assert not summary["physical_failure_detected"]
    for line in (f25k.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25k.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
