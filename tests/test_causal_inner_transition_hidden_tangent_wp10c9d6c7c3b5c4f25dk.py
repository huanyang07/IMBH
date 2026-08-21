from __future__ import annotations

import numpy as np

import run_causal_inner_transition_hidden_tangent_wp10c9d6c7c3b5c4f25dk as f25dk


def test_manifest_authorizes_saved_generator_execution_only() -> None:
    frozen = f25dk._validate_parent(require_clean=False)
    assert frozen["parent_hashes"]
    assert "tangent_diagnostic_manifest_frozen" in frozen["parent_classification"]


def test_candidate_gate_uses_invariance_and_physical_capture() -> None:
    passing = {
        "hidden_tangent_invariance_relative_defect": 0.05,
        "hidden_physical_tangent_energy_capture": 0.95,
        "minimum_seed_rate_action_energy_capture": 0.999,
        "current_primary_rate_action_energy_capture": 0.99,
        "current_primary_gauge_fixed_physical_action_energy_capture": 0.99,
        "basis_orthonormality_infinity_defect": 1.0e-15,
        "action_macro_annihilation_infinity_defect": 1.0e-15,
        "rank": 8,
    }
    assert f25dk._candidate_passes(passing)
    failing = dict(passing)
    failing["hidden_tangent_invariance_relative_defect"] = 0.11
    assert not f25dk._candidate_passes(failing)


def test_residual_enrichment_is_orthogonal() -> None:
    basis = np.eye(6)[:, :2]
    response = np.column_stack((np.array([1, 0, 1, 0, 0, 0]), np.array([0, 1, 0, 1, 0, 0])))
    new = f25dk._new_residual_directions(basis, response, 2)
    assert new.shape == (6, 2)
    assert np.linalg.norm(basis.T @ new, ord=np.inf) <= 1.0e-14
    assert np.linalg.norm(new.T @ new - np.eye(2), ord=np.inf) <= 1.0e-14


def test_canonical_tangent_if_present() -> None:
    if not f25dk.CANONICAL_DIRECTORY.exists():
        return
    f25dk._checksums(f25dk.CANONICAL_DIRECTORY)
    summary = f25dk._read(f25dk.CANONICAL_DIRECTORY / "summary.json")
    payload = f25dk._read(
        f25dk.CANONICAL_DIRECTORY / "transition_hidden_tangent_metrics.json"
    )
    assert summary["new_exact_fixed_Q_rate_calls"] == 0
    assert summary["new_complete_generator_assemblies"] == 0
    assert summary["new_nonlinear_roots"] == 0
    assert summary["new_chart_retractions"] == 0
    assert summary["propagated_states"] == 0
    assert not summary["sealed_16ms_opened"]
    assert summary["full470_offline_transition_reference_preserved"]
    assert not summary["transition_trajectory_authorized"]
    assert not summary["online_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert payload["metrics"]["coordinate_Jacobian_evaluations"] <= 400
