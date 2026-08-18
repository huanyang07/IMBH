from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import run_causal_inner_invariant_projection_spectrum_audit_wp10c9d6c7c3b5c4f25e as f25e


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_is_hash_locked_and_authorizes_no_truth(monkeypatch):
    for name, value in f25e.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25e._validate_manifest(require_clean=False)
    budget = frozen["contract"]["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_new_full_560_direction_descriptor_assemblies"] == 0
    assert budget["saved_complete_generator_must_be_reused"]


def test_storage_candidate_uses_mapped_MJE_and_complete_thermal_stress():
    mapped = np.zeros((10, 10))
    height = np.zeros((10, 10))
    for row in range(10):
        mapped[row, row] = 1.0
        height[row, row] = 2.0
    rows = np.ones((2, 5))
    original_groups = f25e.parent._coarse_groups
    original_cells = f25e.manifest.PRIMARY_CELLS
    original_storage = f25e.manifest.STORAGE_DIMENSION
    try:
        f25e.parent._coarse_groups = lambda count: [(0, 1), (1, 2)]
        f25e.manifest.PRIMARY_CELLS = 2
        f25e.manifest.STORAGE_DIMENSION = 10
        normalized, physical, scales = f25e._candidate_storage_restriction(
            mapped, height, rows
        )
    finally:
        f25e.parent._coarse_groups = original_groups
        f25e.manifest.PRIMARY_CELLS = original_cells
        f25e.manifest.STORAGE_DIMENSION = original_storage
    assert normalized.shape == (10, 10)
    for field in range(5):
        expected = 1.0 if field in f25e.CONSERVATIVE_FIELDS else 3.0
        assert physical[field, field] == pytest.approx(f25e.C * expected)
        assert scales[field] == pytest.approx(f25e.C * expected)


def test_complete_qr_projection_gives_one_right_inverse_and_complement():
    storage = np.asarray(((1.0, 0.0, 0.0, 0.0), (0.0, 2.0, 0.0, 0.0)))
    dual = np.asarray(((0.0, 0.0, 3.0, 0.0),))
    resolved, lifting, complement, metrics = f25e._complete_qr_projection(
        storage, dual
    )
    assert np.allclose(resolved @ lifting, np.eye(3))
    assert np.allclose(resolved @ complement, 0.0)
    assert np.allclose(complement.T @ complement, np.eye(1))
    assert metrics["resolved_rank"] == 3
    assert metrics["unresolved_dimension"] == 1


def test_transfer_audit_detects_stable_and_unstable_poles():
    generator = np.diag((-2.0, 0.5, -4.0))
    lifting = np.asarray(((1.0,), (0.0,), (0.0,)))
    complement = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    output = np.asarray(((1.0, 2.0, -1.0),))
    transfer, poles, metrics = f25e._transfer_from_schur(
        generator, lifting, complement, output, np.asarray((0.1, 1.0, 10.0))
    )
    assert transfer.shape == (4, 1, 1)
    assert poles.shape == (2,)
    assert metrics["unstable_unresolved_pole_count"] == 1
    assert metrics["maximum_frequency_solve_relative_residual"] < 1.0e-12
    assert metrics["maximum_transfer_conjugate_symmetry_relative_defect"] < 1.0e-12


def test_classification_promotes_unstable_or_cycle_scale_modes():
    stage_1 = {"passed": True}
    classification, next_artifact = f25e._classification(
        stage_1,
        {
            "passed": True,
            "unstable_unresolved_pole_count": 1,
            "cycle_scale_stable_unresolved_pole_count": 0,
        },
    )
    assert classification == f25e.UNSTABLE_CLASSIFICATION
    assert next_artifact == "definitions_only_resolved_mode_promotion_manifest"


def test_canonical_result_when_available():
    summary_path = f25e.CANONICAL_DIRECTORY / "summary.json"
    if not summary_path.exists():
        pytest.skip("projection/spectrum audit not canonicalized yet")
    summary = _read(summary_path)
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    assert summary["new_full_560_direction_descriptor_assemblies"] == 0
    assert summary["memory_coefficients_fit"] == 0
    assert not summary["full_anchor_campaign_authorized"]
    assert not summary["online_reduced_solver_implementation_authorized"]
    for line in (f25e.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25e.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
