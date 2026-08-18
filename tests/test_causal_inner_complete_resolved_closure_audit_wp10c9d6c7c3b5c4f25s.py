from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_complete_resolved_closure_audit_wp10c9d6c7c3b5c4f25s as f25s


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_is_locked_and_truth_work_is_forbidden(monkeypatch):
    for name, value in f25s.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25s._validate_manifest(require_clean=False)
    budget = frozen["contract"]["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0


def test_complete_blocks_reconstruct_generator_and_output():
    generator = np.diag((-1.0, -2.0, 3.0, -4.0))
    restriction = np.asarray(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)))
    lifting = restriction.T
    output = np.asarray(((1.0, 2.0, 3.0, 4.0),))
    blocks, metrics = f25s._complete_blocks(generator, restriction, lifting, output)
    assert metrics["coordinate_reconstruction_relative_defect"] <= 1.0e-14
    assert metrics["stable_spectral_abscissa_per_second"] == -2.0
    assert blocks["resolved_direct"].shape == (2, 2)
    assert blocks["resolved_observation"].shape == (2, 2)
    assert blocks["face_observation"].shape == (1, 2)


def test_block_pass_applies_training_and_heldout_limits():
    gates = {
        "maximum_normalized_dynamic_transfer_relative_error_max": 0.25,
        "RMS_normalized_dynamic_transfer_relative_error_max": 0.10,
    }
    metrics = {
        f"{prefix}_{name.removesuffix('_max')}": maximum
        for prefix in ("training", "heldout")
        for name, maximum in gates.items()
    }
    assert f25s._block_pass(metrics, gates)
    metrics["heldout_RMS_normalized_dynamic_transfer_relative_error"] = 0.11
    assert not f25s._block_pass(metrics, gates)


def test_spectral_metrics_exact_full_order_closure_matches():
    exact = np.diag((2.0, -3.0, -4.0))
    reduced_operator = np.asarray(((-4.0,),))
    reduced_forcing = np.zeros((1, 2))
    reduced_observation = np.zeros((2, 1))
    reduced_direct = np.diag((2.0, -3.0))
    metrics = f25s._spectral_metrics(
        exact,
        reduced_operator,
        reduced_forcing,
        reduced_observation,
        reduced_direct,
        np.ones(2),
        np.ones(2),
        -1.0e-8,
    )
    assert metrics["exact_nonstable_eigenvalue_count"] == 1
    assert metrics["reduced_nonstable_eigenvalue_count"] == 1
    assert metrics["bidirectional_nearest_nonstable_eigenvalue_relative_defect"] == 0.0


def test_classification_requires_numerical_and_complete_closure_pass():
    selected = {"memory_order": 112}
    assert f25s._classification(True, selected) == (
        f25s.PASS_CLASSIFICATION,
        "definitions_only_bounded_R196_memory_online_integrator_manifest",
        True,
    )
    assert f25s._classification(True, None)[0] == f25s.CAP_FAIL_CLASSIFICATION
    assert f25s._classification(False, selected)[0] == f25s.NUMERICAL_FAIL_CLASSIFICATION


def test_canonical_result_when_available():
    summary_path = f25s.CANONICAL_DIRECTORY / "summary.json"
    if not summary_path.exists():
        return
    summary = _read(summary_path)
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    assert summary["new_full_560_direction_generator_assemblies"] == 0
    assert not summary["physical_failure_detected"]
    for line in (f25s.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25s.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
