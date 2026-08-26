import run_causal_inner_entropy_complete_adaptive_selective_refresh_cycle_readiness_manifest_wp10c9d6c7c3b5c4f25fizr as manifest


def test_parent_selective_patch_is_certified():
    validated = manifest._validate_parent(require_clean=False)
    assert validated["summary"]["passed"]
    assert validated["metrics"]["accepted_absolute_horizon_seconds"] == 0.012


def test_contract_is_bounded_and_fail_closed():
    contract = manifest._contract()
    execution = contract["bounded_execution"]
    assert execution["new_macrosteps"] == 50
    assert execution["maximum_new_truth_operator_calls"] == 50
    assert execution["new_global_roots"] == 0
    assert contract["numerical_gates"]["fail_on_first_rejected_step"]
    assert contract["claim_boundary"]["bounded_transient_execution_authorized"]
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


def test_architecture_requires_a_separate_slaving_certificate():
    contract = manifest._contract()
    assert contract["mathematical_architecture"]["online_cycle_truth_calls"] == 0
    assert contract["slaving_observation"]["fresh_tangent_certificate_still_required"]
    assert not contract["claim_boundary"]["instantaneous_fast_graph_authorized"]
    assert not contract["claim_boundary"]["48_coordinate_cycle_solver_authorized"]


def test_naive_patchwise_cycle_is_rejected():
    diagnostics = manifest._diagnostics(manifest._validate_parent(require_clean=False))
    assert diagnostics["naive_fixed_patch_steps_per_cycle"] > 100_000_000
    assert diagnostics["naive_selective_truth_calls_per_cycle"] > 1_000_000_000
    assert diagnostics["naive_patchwise_cycle_route_rejected_as_infeasible"]
