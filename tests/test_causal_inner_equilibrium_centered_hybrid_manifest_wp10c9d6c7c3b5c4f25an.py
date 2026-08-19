from __future__ import annotations

import run_causal_inner_equilibrium_centered_hybrid_manifest_wp10c9d6c7c3b5c4f25an as f25an


def test_parent_geometry_and_stable_kernel_are_locked():
    parents = f25an._validate_parents()
    assert parents["geometry_summary"]["passed"]
    assert parents["stable_summary"]["passed"]


def test_state_partition_and_square_conditional_branch_problem():
    contract = f25an._contract()
    partition = contract["state_partition"]
    assert partition["resolved_physical_observables"] == 162
    assert partition["hidden_full_order_fiber"] == 398
    assert partition["stable_memory_upper_bound"] == 280
    assert partition["event_eliminated_departure_bundle"] == 28
    assert partition["truncated_stable_remainder"] == 90
    assert 162 + 398 == 560
    branch = contract["conditional_fast_branch"]
    assert branch["square_equation_count"] == 560
    assert branch["moving_16ms_and_20ms_checkpoints_are_assumed_to_be_branch_roots"] is False


def test_exact_conservation_and_unstable_event_elimination_are_structural():
    contract = f25an._contract()
    online = contract["online_continuous_dynamics"]
    assert online["conservative_components"] == ["mass", "angular_momentum", "energy"]
    assert online["online_full_order_truth_calls_per_macrostep"] == 0
    assert online["linearly_macro_propagated_unstable_coordinates"] == 0
    assert contract["fast_transition_collocation"]["global_Q3_preserved_without_external_impulse"]


def test_claim_boundary_authorizes_only_first_branch_seed_manifest():
    contract = f25an._contract()
    assert contract["decision"]["pass_authorizes_only"] == "definitions_only_first_conditional_fast_branch_seed_manifest"
    assert not contract["claim_boundary"]["physical_conditional_branch_found"]
    assert not contract["claim_boundary"]["predictive_cycle_authorized"]
    assert not contract["claim_boundary"]["reduced_slow_evolution_authorized"]
