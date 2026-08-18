from __future__ import annotations

import run_causal_inner_effective_real_rank_manifest_wp10c9d6c7c3b5c4f25v as f25v


def test_parent_rejection_is_preserved():
    summary, hashes = f25v._validate_parent()
    assert not summary["passed"]
    assert not summary["physical_failure_detected"]
    assert "metrics.json" in hashes


def test_rank_definition_is_explicit_and_not_machine_epsilon():
    definition = f25v._contract()["effective_real_rank"]
    assert definition["expected_rank"] == 28
    assert definition["relative_cutoff"] == 5.0e-10
    assert definition["first_discarded_to_last_retained_ratio_max"] == 5.0e-10
    assert definition["machine_epsilon_rank_is_nonbinding_diagnostic"]


def test_all_substantive_gates_and_claim_boundaries_remain_binding():
    contract = f25v._contract()
    gates = contract["unchanged_binding_gates"]
    assert gates["spectral_projector_commutator_relative_defect_max"] == 5.0e-9
    assert gates["stable_complement_spectral_abscissa_per_second_max"] == -1.0e-8
    assert gates["R32_stable_coordinate_rank_equal"] == 162
    assert gates["remaining_stable_memory_budget_min"] == 112
    assert not contract["claim_boundary"]["online_integrator_implementation_authorized"]
