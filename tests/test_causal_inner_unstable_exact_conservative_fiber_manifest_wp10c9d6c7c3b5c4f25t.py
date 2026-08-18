from __future__ import annotations

import run_causal_inner_unstable_exact_conservative_fiber_manifest_wp10c9d6c7c3b5c4f25t as f25t


def test_parent_is_binding_rejection_and_inputs_are_hash_locked():
    summary, hashes = f25t._validate_parent()
    assert not summary["passed"]
    assert not summary["physical_failure_detected"]
    assert summary["authorized_next"].startswith("definitions_only_structured")
    assert "summary.json" in hashes


def test_contract_keeps_nonstable_fiber_exact_and_forbids_truth_work():
    contract = f25t._contract()
    budget = contract["execution_budget"]
    partition = contract["spectral_partition"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0
    assert partition["expected_nonstable_dimension_at_each_anchor"] == 28
    assert partition["reduce_nonstable_subspace"] is False
    assert partition["ordinary_balanced_truncation_of_closed_feedback"] is False


def test_contract_preserves_conservative_coordinates_and_R320_cap():
    contract = f25t._contract()
    gates = contract["binding_gates"]
    architecture = contract["prospective_online_state_if_passed"]
    assert gates["R32_stable_coordinate_rank_equal"] == 162
    assert gates["remaining_stable_memory_budget_min"] == 112
    assert architecture["maximum_dimension"] == 320
    assert architecture["face_flux_single_valued_before_conservative_divergence"]
    assert contract["claim_boundary"]["reduced_slow_evolution_authorized"] is False
