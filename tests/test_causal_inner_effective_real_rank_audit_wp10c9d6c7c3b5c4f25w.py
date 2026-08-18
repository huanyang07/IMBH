from __future__ import annotations

import numpy as np

import run_causal_inner_effective_real_rank_audit_wp10c9d6c7c3b5c4f25w as f25w


def test_manifest_is_locked_and_truth_work_is_forbidden(monkeypatch):
    for name, value in f25w.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25w._validate_manifest(require_clean=False)
    budget = frozen["contract"]["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0


def test_effective_rank_ignores_small_schur_leakage_but_requires_gap():
    definition = {
        "expected_rank": 3,
        "retained_singular_value_relative_floor": 1.0e-6,
        "first_discarded_to_last_retained_ratio_max": 5.0e-10,
    }
    passed = {
        "ordered_schur_count": 3,
        "effective_rank": 3,
        "last_retained_to_leading_ratio": 0.2,
        "first_discarded_to_last_retained_ratio": 1.0e-12,
    }
    assert f25w._effective_gate_passed(passed, definition)
    failed = {**passed, "first_discarded_to_last_retained_ratio": 1.0e-5}
    assert not f25w._effective_gate_passed(failed, definition)


def test_effective_rank_of_real_diagonal_cluster_is_exact():
    generator = np.diag((2.0, 1.0, -1.0, -2.0))
    metrics, singular = f25w._effective_rank_metrics(
        generator,
        transpose=False,
        threshold=-1.0e-8,
        expected_rank=2,
        relative_cutoff=5.0e-10,
    )
    assert metrics["ordered_schur_count"] == 2
    assert metrics["effective_rank"] == 2
    assert singular[1] > 0.9
    assert singular[2] == 0.0
