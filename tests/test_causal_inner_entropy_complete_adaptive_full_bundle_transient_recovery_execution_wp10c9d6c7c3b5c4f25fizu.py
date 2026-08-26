import numpy as np

import run_causal_inner_entropy_complete_adaptive_full_bundle_transient_recovery_execution_wp10c9d6c7c3b5c4f25fizu as execution


def test_variable_step_ab2_reduces_to_equal_step_formula():
    state = np.arange(80.0).reshape(16, 5)
    current = np.full((16, 5), 3.0)
    previous = np.full((16, 5), 1.0)
    candidate = execution._variable_ab2_candidate(
        state, current, previous, 0.25, 0.25
    )
    np.testing.assert_array_equal(candidate, state + 1.0)


def test_variable_step_ab2_integral_matches_rate_formula():
    current = np.arange(115.0)
    previous = 0.5 * current
    integral = execution._variable_ab2_integral(current, previous, 0.1, 0.2)
    ratio = 0.5
    expected = 0.1 * ((1.0 + ratio / 2.0) * current - ratio / 2.0 * previous)
    np.testing.assert_array_equal(integral, expected)


def test_checkpoint_roundtrip_is_bitwise():
    arrays = {
        "state": np.arange(80.0).reshape(16, 5),
        "time": np.asarray([0.15600000000000003]),
    }
    assert execution._bitwise_roundtrip(arrays)


def test_truth_hyperbolicity_exception_is_classified_fail_closed():
    exception = ValueError("generalized eigenvalues is not real within the declared tolerance")
    assert execution._is_hyperbolicity_failure(exception)
    assert not execution._is_hyperbolicity_failure(ValueError("unrelated failure"))


def test_parent_authorizes_the_adaptive_execution():
    validated = execution._validate_parent(require_clean=False)
    assert validated["summary"]["adaptive_recovery_execution_authorized"]
    assert validated["contract"]["bounded_execution"][
        "maximum_new_truth_operator_calls"
    ] == 128


def test_classifications_do_not_conflate_budget_and_physics():
    assert execution.BUDGET_CLASSIFICATION != execution.FAIL_CLASSIFICATION
    assert execution.OPEN_CLASSIFICATION != execution.SLAVING_CLASSIFICATION


def test_slaving_authorization_names_a_terminal_tangent_certificate():
    assert "terminal_fast_graph_tangent_certificate" in execution.SLAVING_AUTHORIZED_NEXT
    assert "transient_geometry" in execution.OPEN_AUTHORIZED_NEXT
