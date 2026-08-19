from __future__ import annotations

import run_causal_inner_first_conditional_branch_seed_manifest_wp10c9d6c7c3b5c4f25ap as f25ap


def test_parent_architecture_certificate_is_locked():
    validated = f25ap._validate_parent()
    assert validated["summary"]["passed"]
    assert validated["summary"]["authorized_next"] == (
        "definitions_only_first_conditional_fast_branch_seed_manifest"
    )


def test_only_exact_integrable_finite_amplitude_coordinates_are_used():
    contract = f25ap._contract()
    coordinates = contract["finite_amplitude_coordinate_map"]
    assert coordinates["dimension"] == 162
    assert coordinates["mapped_storage_coordinates"] == 160
    assert coordinates["explicit_stable_coordinates"] == 2
    assert not coordinates[
        "responsive_height_one_form_is_a_finite_amplitude_coordinate"
    ]


def test_preflight_is_zero_truth_call_and_cannot_launch_a_root():
    contract = f25ap._contract()
    predictor = contract["direct_predictor"]
    assert predictor["nonbase_physical_truth_calls"] == 0
    assert not predictor["direct_root_may_run_in_this_work_package"]
    assert not contract["claim_boundary"]["physical_branch_root_attempted"]


def test_unsafe_direct_predictor_routes_only_to_prospective_homotopy():
    contract = f25ap._contract()
    assert "homotopy_manifest_authorized" in contract["decision"]["direct_unsafe"]
    homotopy = contract["homotopy_if_needed"]
    assert homotopy["tau_zero_anchor_is_exact"]
    assert homotopy["tau_one_is_conditional_branch_stationarity"]
    assert not homotopy["tiny_forward_BDF_steps_are_used"]
    assert not contract["claim_boundary"]["reduced_slow_evolution_authorized"]
