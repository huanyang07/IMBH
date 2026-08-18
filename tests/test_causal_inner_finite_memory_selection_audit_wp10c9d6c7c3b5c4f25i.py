from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_finite_memory_selection_audit_wp10c9d6c7c3b5c4f25i as f25i


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_is_hash_locked_and_authorizes_no_truth(monkeypatch):
    for name, value in f25i.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25i._validate_manifest(require_clean=False)
    budget = frozen["contract"]["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_new_full_generator_assemblies"] == 0
    assert budget["allowed_truth_anchors"] == 0


def test_normalization_is_fixed_and_finite():
    forcing = np.asarray(((3.0, 0.0), (4.0, 0.0)))
    observation = np.asarray(((0.0, 2.0),))
    direct = np.asarray(((0.0, 0.0),))
    bn, cn, dn, input_scales, output_scales = f25i._normalize_system(
        forcing, observation, direct
    )
    assert input_scales[0] == 5.0
    assert input_scales[1] == 1.0
    assert output_scales[0] == 2.0
    assert np.linalg.norm(bn[:, 0]) == 1.0
    assert np.linalg.norm(cn[0]) == 1.0
    assert np.array_equal(dn, direct)


def test_balancing_and_truncation_preserve_a_stable_low_rank_system():
    operator = np.diag((-1.0, -2.0, -5.0))
    forcing = np.asarray(((1.0,), (0.3,), (0.0,)))
    observation = np.asarray(((1.0, 0.2, 0.0),))
    balanced, metrics = f25i._balanced_realization(operator, forcing, observation)
    assert metrics["controllability_gramian_relative_residual"] < 1.0e-12
    assert metrics["observability_gramian_relative_residual"] < 1.0e-12
    reduced_operator, reduced_forcing, reduced_observation, truncation = f25i._truncate_balanced(
        operator, forcing, observation, balanced, 2
    )
    assert reduced_operator.shape == (2, 2)
    assert reduced_forcing.shape == (2, 1)
    assert reduced_observation.shape == (1, 2)
    assert np.max(np.real(np.linalg.eigvals(reduced_operator))) < 0.0
    assert truncation["biorthogonality_defect"] < 1.0e-10


def test_candidate_zero_memory_retains_direct_but_fails_dynamic_gate():
    operator = np.asarray(((-1.0,),))
    forcing = np.asarray(((1.0,),))
    observation = np.asarray(((1.0,),))
    direct = np.asarray(((0.2,),))
    frequencies = np.asarray((0.0, 1.0))
    reference = f25i._frequency_response(operator, forcing, observation, direct, frequencies)
    balanced, _ = f25i._balanced_realization(operator, forcing, observation)
    gates = {
        "reduced_spectral_abscissa_per_second_max": -1.0e-8,
        "lyapunov_dissipation_residual_max": 1.0e-9,
        "lyapunov_certificate_minimum_eigenvalue_min": 1.0e-14,
        "maximum_normalized_dynamic_transfer_relative_error_max": 0.25,
        "RMS_normalized_dynamic_transfer_relative_error_max": 0.10,
        "DC_normalized_dynamic_transfer_relative_error_max": 0.10,
        "maximum_normalized_total_transfer_relative_error_max": 0.10,
    }
    _, metrics = f25i._candidate_metrics(
        0, operator, forcing, observation, direct, frequencies, reference, balanced, gates
    )
    assert metrics["maximum_normalized_dynamic_transfer_relative_error"] == 1.0
    assert not metrics["passed"]


def test_classification_selects_cross_anchor_or_fallback():
    assert f25i._classification(
        {"full_order_numerical_passed": True, "selected_order": 4}
    ) == (f25i.PASS_CLASSIFICATION, "definitions_only_cross_anchor_closure_database_manifest")
    assert f25i._classification(
        {"full_order_numerical_passed": True, "selected_order": None}
    ) == (
        f25i.COMPACT_FAIL_CLASSIFICATION,
        "definitions_only_larger_conservative_coarse_PDE_manifest",
    )


def test_canonical_result_when_available():
    summary_path = f25i.CANONICAL_DIRECTORY / "summary.json"
    if not summary_path.exists():
        return
    summary = _read(summary_path)
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    assert summary["new_full_generator_assemblies"] == 0
    assert summary["truth_anchors_queried"] == 0
    assert not summary["production_memory_coefficients_authorized"]
    for line in (f25i.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25i.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
