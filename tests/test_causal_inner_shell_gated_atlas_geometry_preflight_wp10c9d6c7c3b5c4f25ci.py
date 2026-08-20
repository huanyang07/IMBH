from __future__ import annotations

import json

import numpy as np

import run_causal_inner_shell_gated_atlas_geometry_preflight_wp10c9d6c7c3b5c4f25ci as f25ci


def test_manifest_authorizes_geometry_preflight():
    frozen = f25ci._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25ci.WORK_PACKAGE
    assert frozen["directions"].shape == (4, 28)
    assert np.array_equal(frozen["bounds"], np.asarray((0.0125, 0.015)))


def test_retraction_adapter_uses_frozen_line_search():
    frozen = f25ci._validate_manifest(require_clean=False)
    local = f25ci._local_contract(frozen["contract"], 0.0125)
    assert local["binding_preflight_gates"]["maximum_final_scaled_component"] == 0.0125
    assert local["exact_geometric_retraction"]["line_factors"] == list(f25ci.LINE_FACTORS)
    assert len(f25ci.LINE_FACTORS) == 12


def test_direction_design_matches_parent_and_is_normalized():
    frozen = f25ci._validate_manifest(require_clean=False)
    assert np.allclose(np.linalg.norm(frozen["directions"], axis=1), 1.0, rtol=0.0, atol=1.0e-14)
    assert len(set(frozen["labels"])) == 4


def test_classification_is_fail_fast_and_cost_closed():
    contract = f25ci.manifest._contract()
    metrics = {
        "passing_rung_count": 1,
        "new_nonbase_continuous_rate_evaluations": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_physical_states": 0,
    }
    decision = f25ci._classify(metrics, contract)
    assert decision["classification"] == f25ci.PARTIAL_CLASSIFICATION
    assert decision["largest_passing_component_bound"] == 0.0125
    assert all(decision["budget_checks"].values())


def test_canonical_geometry_if_present():
    if not f25ci.CANONICAL_DIRECTORY.exists():
        return
    f25ci._checksums(f25ci.CANONICAL_DIRECTORY)
    summary = json.loads((f25ci.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((f25ci.CANONICAL_DIRECTORY / "geometry_metrics.json").read_text(encoding="utf-8"))
    assert summary["passed"]
    assert summary["classification"] in {f25ci.FULL_CLASSIFICATION, f25ci.PARTIAL_CLASSIFICATION, f25ci.FAIL_CLASSIFICATION}
    assert all(metrics["decision"]["budget_checks"].values())
    assert not summary["geometry_candidate_became_atlas_center"]
    assert not summary["trajectory_authorized"]
    assert not summary["predictive_cycle_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
