from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_common_resolved_subspace_cross_anchor_preflight_wp10c9d6c7c3b5c4f25o as f25o


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_is_locked_and_allows_one_generator(monkeypatch):
    for name, value in f25o.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25o._validate_manifest(require_clean=False)
    budget = frozen["contract"]["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 1
    assert frozen["contract"]["common_basis_memory"]["order"] == 96


def test_common_reference_recovers_exact_union_rank():
    primary = np.eye(6)[:, :2]
    heldout = np.eye(6)[:, 1:4]
    basis, singular, rank = f25o._common_reference_basis(primary, heldout, 1.0e-10)
    assert rank == 4
    assert basis.shape == (6, 4)
    assert singular.shape == (5,)
    assert singular[3] > 1.0e-10 * singular[0]
    assert singular[4] <= 1.0e-10 * singular[0]
    combined = np.column_stack((primary, heldout))
    assert np.linalg.norm(combined - basis @ (basis.T @ combined)) <= 1.0e-14


def test_anchor_common_basis_preserves_local_promoted_space_and_stability():
    restriction = np.asarray(((1.0, 0.0, 0.0, 0.0),))
    lifting = restriction.T
    complement = np.eye(4)[:, 1:]
    promoted = np.eye(4)[:, 1:2]
    reference = np.eye(4)[:, 1:3]
    generator = np.diag((-3.0, 0.5, -2.0, -4.0))
    output = np.eye(4)
    arrays, metrics = f25o._anchor_common_basis(
        reference, complement, lifting, restriction, promoted, generator, output
    )
    assert arrays["aligned_common_basis"].shape == (4, 2)
    assert metrics["local_promoted_subspace_projection_relative_defect"] <= 1.0e-14
    assert metrics["remaining_common_unresolved_dimension"] == 1
    assert metrics["remaining_common_unresolved_spectral_abscissa_per_second"] == -4.0


def test_classification_distinguishes_global_chart_atlas_and_failure():
    assembly = {"passed": True}
    global_result = {
        "numerical_passed": True,
        "memory_passed": True,
        "global_chart_alignment_passed": True,
    }
    assert f25o._classification(assembly, global_result) == (
        f25o.GLOBAL_PASS_CLASSIFICATION,
        "definitions_only_R32_R96_online_prototype_manifest",
        True,
    )
    atlas_result = dict(global_result, global_chart_alignment_passed=False)
    assert f25o._classification(assembly, atlas_result) == (
        f25o.ATLAS_PASS_CLASSIFICATION,
        "definitions_only_two_chart_conservative_atlas_manifest",
        True,
    )
    failed = dict(global_result, memory_passed=False)
    assert f25o._classification(assembly, failed)[0] == f25o.MEMORY_FAIL_CLASSIFICATION
    assert not f25o._classification(assembly, failed)[2]


def test_canonical_result_when_available():
    summary_path = f25o.CANONICAL_DIRECTORY / "summary.json"
    if not summary_path.exists():
        return
    summary = _read(summary_path)
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    assert summary["new_full_560_direction_generator_assemblies"] == 1
    assert summary["new_truth_anchors"] == 1
    assert not summary["physical_failure_detected"]
    for line in (f25o.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25o.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
