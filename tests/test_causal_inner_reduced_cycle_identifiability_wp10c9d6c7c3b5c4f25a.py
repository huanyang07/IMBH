from __future__ import annotations

import hashlib
import json

import pytest

import run_causal_inner_reduced_cycle_identifiability_wp10c9d6c7c3b5c4f25a as f25a


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_parent_manifest_is_locked_and_authorizes_only_this_screen():
    summary, lock = f25a._validate_parent()
    assert summary["passed"]
    assert summary["evidence_only_identifiability_authorized"]
    assert not summary["online_reduced_solver_implementation_authorized"]
    assert "architecture_contract.json" in lock["package_hashes"]


def test_evidence_rejects_direct_cycle_cost():
    truth = f25a._evidence_ledger()["truth_solver"]
    assert truth["direct_truth_wall_hours_per_microsecond"] > 3.5
    assert truth["direct_truth_wall_days_per_millisecond"] > 149.0
    assert truth["direct_truth_wall_years_per_fiducial_cycle"] > 2.0e8
    assert truth["minimum_required_speedup_for_three_days"] > 2.0e10
    assert truth["dominant_cost"] == "monolithic_residual"
    assert truth["dense_bordered_solve_is_not_a_material_bottleneck"]


def test_existing_data_reject_instantaneous_markov_and_direct_two_mode_laws():
    evidence = f25a._evidence_ledger()
    assert not evidence["instantaneous_markov_counterexample"][
        "instantaneous_markov_closure_supported"
    ]
    two = evidence["two_mode_output_closure"]
    assert two["maximum_significant_direction_error"] == pytest.approx(1.0437719690983103)
    assert two["maximum_significant_direction_error"] > two["gate"]
    assert not two["supported"]


def test_leading_two_are_retained_but_six_explicit_modes_are_rejected():
    evidence = f25a._evidence_ledger()
    six = evidence["six_mode_coordinate"]
    guard = evidence["leading_two_plus_guard"]
    assert six["minimum_full_cross_grid_projector_cosine"] < 0.9
    assert six["minimum_leading_two_projector_cosine"] > 0.95
    assert not six["explicit_six_mode_dynamic_coordinate_supported"]
    assert guard["leading_two_state_coordinate_supported"]
    assert guard["guard_required"]
    assert guard["online_guard_truth_calls_per_step_allowed"] == 0


def test_exactly_one_predeclared_architecture_is_selected():
    decision = f25a._decision()
    selected = [
        name for name, result in decision["candidates"].items() if result.get("selected")
    ]
    assert selected == ["cellwise_Q5_FV_plus_a2_finite_memory_hybrid"]
    assert decision["architecture_supported_for_offline_identification"]
    assert not decision["coefficients_identifiable_from_existing_committed_data"]
    assert not decision["online_solver_implementation_authorized"]


def test_next_manifest_is_pathwise_and_prospective():
    frozen = f25a._decision()["next_manifest_must_freeze"]
    assert frozen["pathwise_anchor_count_range"] == (10, 30)
    assert frozen["global_tensor_product_Q_grid_forbidden"]
    assert frozen["middle_layout_is_primary"]
    assert frozen["fine_layout_is_sparse_validation_only"]
    assert frozen["memory_orders"] == (0, 2, 4, 6)
    assert frozen["training_validation_split_frozen_before_truth_queries"]


def test_canonical_result_when_available():
    summary_path = f25a.ARTIFACT_DIRECTORY / "summary.json"
    if not summary_path.exists():
        pytest.skip("identifiability screen not canonicalized yet")
    summary = _read(summary_path)
    assert summary["passed"]
    assert summary["analysis_only"]
    assert summary["offline_closure_database_manifest_authorized"]
    assert not summary["online_reduced_solver_implementation_authorized"]
    assert not summary["predictive_cycle_authorized"]
    for line in (f25a.ARTIFACT_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((f25a.ARTIFACT_DIRECTORY / name).read_bytes()).hexdigest() == expected
