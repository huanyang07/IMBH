from __future__ import annotations

import hashlib
import json

import pytest

import run_causal_inner_pathwise_offline_closure_database_manifest_wp10c9d6c7c3b5c4f25b as f25b


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_parent_is_hash_locked_and_authorizes_only_the_database_manifest():
    summary, lock = f25b._validate_parent()
    assert summary["passed"]
    assert summary["offline_closure_database_manifest_authorized"]
    assert not summary["online_reduced_solver_implementation_authorized"]
    assert lock["decision"]["selected_architecture"] == (
        "cellwise_Q5_FV_plus_a2_finite_memory_hybrid"
    )


def test_anchor_split_is_frozen_before_truth_queries():
    schedule = f25b._anchor_schedule()
    assert schedule["initial_anchor_count"] == 18
    assert len(schedule["initial_training_ids"]) == 12
    assert len(schedule["sealed_heldout_ids"]) == 6
    assert set(schedule["fine_validation_ids"]) <= set(schedule["sealed_heldout_ids"])
    assert schedule["maximum_total_middle_anchors"] == 30
    assert schedule["adaptive_rule"]["heldout_data_may_not_select_new_anchors"]
    assert schedule["adaptive_rule"]["maximum_new_anchors_per_branch"] == 3


def test_every_branch_has_training_heldout_and_fine_coverage():
    schedule = f25b._anchor_schedule()
    slots = schedule["initial_slots"]
    for branch in f25b.BRANCH_ORDER:
        branch_slots = [slot for slot in slots if slot["branch"] == branch]
        assert any(slot["role"] == "training" for slot in branch_slots)
        assert any(slot["role"] == "heldout" for slot in branch_slots)
        assert any(slot["fine_layout_query"] for slot in branch_slots)


def test_database_is_pathwise_conservative_and_finite_memory():
    contract = f25b._database_contract()
    assert contract["identification_scope"]["pathwise_not_global_state_space"]
    assert contract["identification_scope"]["global_tensor_product_Q_grid_forbidden"]
    assert contract["resolved_coordinates"]["candidate_memory_orders"] == (0, 2, 4, 6)
    assert contract["resolved_coordinates"][
        "restriction_must_preserve_cell_integrated_M_J_E_exactly"
    ]
    assert contract["descriptor_reduction"][
        "two_stable_modes_are_removed_from_the_kernel_and_retained_explicitly"
    ]
    assert contract["memory_selection"]["post_heldout_order_change_forbidden"]


def test_frequency_grid_spans_cycle_to_certified_fast_scale():
    grid = f25b._frequency_grid()
    assert grid["count"] == 32
    assert len(grid["values_per_second"]) == 32
    assert grid["values_per_second"] == sorted(grid["values_per_second"])
    assert grid["angular_frequency_min_per_second"] == pytest.approx(
        2.0 * 3.141592653589793 / (6.7 * 86400.0)
    )
    assert grid["angular_frequency_max_per_second"] == pytest.approx(
        3.141592653589793 / 1.0e-7
    )
    assert grid["includes_exact_DC_evaluation_separately"]


def test_branch_existence_and_one_zone_labels_are_not_assumed():
    construction = f25b._anchor_schedule()["physical_anchor_construction"]
    assert not construction["branch_existence_is_assumed"]
    assert not construction["one_zone_thresholds_define_truth_anchors"]
    contract = f25b._database_contract()
    assert contract["claim_boundary"]["exploratory_one_zone_switches_are_not_truth_labels"]


def test_single_anchor_pilot_is_nonpropagating_and_cost_bounded():
    pilot = f25b._pilot_contract()
    assert pilot["seed_branch_label"] == "unclassified"
    assert pilot["allowed_new_nonlinear_roots"] == 0
    assert pilot["allowed_exact_continuous_descriptor_assemblies"] == 1
    assert pilot["allowed_frequency_points"] == 33
    assert pilot["allowed_short_truth_burst_steps"] == 0
    assert pilot["allowed_fine_layout_queries"] == 0
    assert pilot["maximum_wall_hours"] == 6.0
    assert pilot["full_campaign_truth_budget"]["maximum_total_execution_wall_hours"] == 72.0
    for relative, expected in pilot["seed_hashes"].items():
        assert hashlib.sha256((f25b.ROOT / relative).read_bytes()).hexdigest() == expected


def test_validation_gates_are_prospective_and_predictive_cycle_remains_blocked():
    contract = f25b._database_contract()
    gates = contract["binding_validation_gates"]
    assert gates["truth_root_complete_residual_max"] == 1.0e-10
    assert gates["significant_transfer_direction_error_max"] == 0.25
    assert gates["leading_two_projector_cosine_min"] == 0.95
    assert gates["M_J_E_telescope_relative_defect_max"] == 5.0e-12
    assert "predictive_QPE_cycle" in contract["claim_boundary"][
        "database_pass_may_not_authorize"
    ]


def test_canonical_result_when_available():
    summary_path = f25b.ARTIFACT_DIRECTORY / "summary.json"
    if not summary_path.exists():
        pytest.skip("closure-database manifest not canonicalized yet")
    summary = _read(summary_path)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["single_anchor_descriptor_pilot_authorized"]
    assert not summary["full_anchor_campaign_authorized"]
    assert not summary["online_reduced_solver_implementation_authorized"]
    assert not summary["predictive_cycle_authorized"]
    for line in (f25b.ARTIFACT_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25b.ARTIFACT_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
