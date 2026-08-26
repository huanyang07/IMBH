import numpy as np

import run_causal_inner_entropy_complete_bounded_full_bundle_transient_acquisition_execution_wp10c9d6c7c3b5c4f25fizs as execution


def test_parent_authorizes_only_the_bounded_execution():
    validated = execution._validate_parent(require_clean=False)
    assert validated["summary"]["bounded_transient_execution_authorized"]
    assert not validated["summary"]["complete_cycle_execution_authorized"]
    assert validated["contract"]["bounded_execution"]["new_macrosteps"] == 50


def test_equal_step_ab2_formula():
    state = np.arange(80.0).reshape(16, 5)
    current = np.full((16, 5), 3.0)
    previous = np.full((16, 5), 1.0)
    candidate = execution._ab2_candidate(state, current, previous, 0.25)
    np.testing.assert_array_equal(candidate, state + 1.0)


def test_embedded_defect_is_zero_for_constant_rate():
    state = np.ones((16, 5))
    rate = np.full((16, 5), 2.0)
    candidate = state + 0.1 * rate
    defect = execution._embedded_defect(
        state, candidate, rate, rate, np.ones((16, 5)), 0.1
    )
    assert defect < 1.0e-15


def test_slaving_observation_uses_both_absolute_and_relative_gates():
    contract = execution.parent._contract()
    scales = np.ones((16, 5))
    rate = np.zeros((16, 5))
    rate[:, :3] = 2.0
    rate[:, 3:] = 0.05
    record = execution._slaving_record(rate, scales, contract)
    assert record["instantaneous_slaving_observation_passed"]
    rate[:, 3:] = 0.2
    record = execution._slaving_record(rate, scales, contract)
    assert not record["instantaneous_slaving_observation_passed"]


def test_failed_candidate_cannot_authorize_a_cycle():
    assert execution.FAIL_CLASSIFICATION != execution.OPEN_CLASSIFICATION
    assert execution.FAIL_CLASSIFICATION != execution.SLAVING_CLASSIFICATION
