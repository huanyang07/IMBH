from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_resolved_mode_promotion_audit_wp10c9d6c7c3b5c4f25g as f25g


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_is_hash_locked_and_authorizes_no_truth(monkeypatch):
    for name, value in f25g.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25g._validate_manifest(require_clean=False)
    budget = frozen["contract"]["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_new_full_560_direction_descriptor_assemblies"] == 0
    assert budget["allowed_memory_coefficients_fit"] == 0


def test_ordered_real_schur_promotion_makes_nonstable_modes_explicit():
    generator = np.diag((-3.0, -2.0, 0.5, 1.0))
    restriction = np.asarray(((1.0, 0.0, 0.0, 0.0),))
    lifting = restriction.T
    complement = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    arrays, metrics = f25g._ordered_real_schur_promotion(
        generator, restriction, lifting, complement, stability_margin=1.0e-8
    )
    assert metrics["parent_nonstable_dimension"] == 2
    assert metrics["augmented_resolved_dimension"] == 3
    assert metrics["remaining_unresolved_dimension"] == 1
    assert metrics["remaining_unresolved_spectral_abscissa_per_second"] < 0.0
    assert np.allclose(
        arrays["augmented_resolved_restriction"] @ arrays["augmented_resolved_lifting"],
        np.eye(3),
    )
    assert np.allclose(
        arrays["augmented_resolved_restriction"] @ arrays["remaining_stable_truth_basis"],
        0.0,
    )


def test_stable_transfer_is_solved_at_dc_and_conjugate_frequencies():
    generator = np.diag((-1.0, -4.0, 0.5))
    lifting = np.asarray(((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)))
    stable_basis = np.asarray(((0.0,), (1.0,), (0.0,)))
    output = np.asarray(((1.0, 2.0, -1.0),))
    frequencies = np.asarray((0.0, 0.1, 1.0, 10.0))
    transfer, arrays, metrics = f25g._stable_transfer(
        generator, lifting, stable_basis, output, frequencies
    )
    assert transfer.shape == (4, 1, 2)
    assert arrays["stable_observation"].shape == (1, 1)
    assert metrics["frequency_count_including_DC"] == 4
    assert metrics["maximum_frequency_solve_relative_residual"] < 1.0e-12
    assert metrics["maximum_transfer_conjugate_symmetry_relative_defect"] < 1.0e-12


def test_classification_is_fail_closed():
    assert f25g._classification({"promotion_budget_passed": False}) == (
        f25g.BUDGET_FAIL_CLASSIFICATION,
        None,
    )
    assert f25g._classification(
        {"promotion_budget_passed": True, "remaining_unresolved_strictly_stable": False}
    ) == (f25g.STABILITY_FAIL_CLASSIFICATION, None)
    assert f25g._classification(
        {
            "promotion_budget_passed": True,
            "remaining_unresolved_strictly_stable": True,
            "passed": True,
        }
    ) == (
        f25g.PASS_CLASSIFICATION,
        "definitions_only_mode_selection_and_finite_memory_manifest",
    )


def test_canonical_result_when_available():
    summary_path = f25g.CANONICAL_DIRECTORY / "summary.json"
    if not summary_path.exists():
        return
    summary = _read(summary_path)
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    assert summary["new_full_560_direction_descriptor_assemblies"] == 0
    assert summary["memory_coefficients_fit"] == 0
    assert not summary["physical_instability_claim_made"]
    assert not summary["online_reduced_solver_implementation_authorized"]
    for line in (f25g.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25g.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
