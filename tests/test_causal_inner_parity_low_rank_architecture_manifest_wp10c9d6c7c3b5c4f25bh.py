from __future__ import annotations

import run_causal_inner_parity_low_rank_architecture_manifest_wp10c9d6c7c3b5c4f25bh as f25bh


def test_rejected_axial_chart_and_rate_evidence_are_locked():
    frozen = f25bh._validate_parents()
    assert not frozen["rejected"]["passed"]
    assert frozen["summary_0p01"]["passed"]
    assert frozen["summary_0p005"]["passed"]


def test_diagnosis_is_transparent_and_uses_no_new_truth_calls():
    contract = f25bh._contract()
    assert contract["post_result_architecture_diagnosis"]
    assert not contract["independent_validation_claimed"]
    parity = contract["parity_decomposition"]
    assert parity["new_truth_rate_evaluations"] == 0
    assert parity["new_retractions"] == 0
    assert parity["propagated_states"] == 0


def test_candidate_architecture_preserves_470_partition():
    contract = f25bh._contract()
    partition = contract["fixed_470_state_partition"]
    assert 162 + 280 + 28 == 470
    assert partition["online_truth_calls_per_macrostep"] == 0
    candidate = contract["candidate_departure_architecture"]
    assert candidate["quadratic_output_rank"] == 3
    assert candidate["cubic_output_rank"] == 4
    assert candidate["compressed_full_polynomial_coefficient_upper_bound"] == 588


def test_pass_can_only_authorize_prospective_mixed_direction_database():
    contract = f25bh._contract()
    assert contract["decision"]["diagnosis_consistent"]["authorizes_only"] == (
        "definitions_only_active8_mixed_direction_parity_database_manifest"
    )
    assert not contract["claim_boundary"]["mixed_direction_coefficients_identified"]
    assert not contract["claim_boundary"]["predictive_cycle_authorized"]
