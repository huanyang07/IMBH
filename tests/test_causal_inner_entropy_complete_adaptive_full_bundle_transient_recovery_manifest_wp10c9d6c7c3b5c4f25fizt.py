import run_causal_inner_entropy_complete_adaptive_full_bundle_transient_recovery_manifest_wp10c9d6c7c3b5c4f25fizt as manifest


def test_fixed_step_rejection_is_preserved():
    validated = manifest._validate_parent(require_clean=False)
    assert not validated["summary"]["passed"]
    assert validated["summary"]["accepted_steps"] == 36
    assert validated["metrics"]["new_truth_operator_calls"] == 36


def test_adaptive_retry_is_prospective_and_bounded():
    contract = manifest._contract()
    adaptive = contract["adaptive_AB2"]
    bounded = contract["bounded_execution"]
    assert adaptive["initial_timestep_seconds"] == 0.002
    assert adaptive["minimum_timestep_seconds"] == 0.000125
    assert adaptive["shrink_factor"] == 0.5
    assert bounded["maximum_new_truth_operator_calls"] == 128
    assert bounded["target_absolute_elapsed_seconds"] == 0.212


def test_physical_failures_are_not_retried_or_hidden():
    contract = manifest._contract()
    assert contract["adaptive_AB2"]["physical_failure_is_not_retryable"]
    assert contract["binding_gates"]["minimum_timestep_failure_is_binding"]
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]
