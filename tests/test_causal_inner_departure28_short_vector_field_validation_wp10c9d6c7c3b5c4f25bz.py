from __future__ import annotations

import numpy as np

import run_causal_inner_departure28_short_vector_field_validation_wp10c9d6c7c3b5c4f25bz as f25bz


def test_frozen_manifest_is_exactly_the_committed_parent():
    frozen = f25bz._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25bz.WORK_PACKAGE
    assert frozen["contract"]["reference_sequence"][
        "forecast_must_be_serialized_and_hashed_before_truth_root"
    ]


def test_rk4_is_exact_for_constant_rate():
    initial = np.arange(6, dtype=float)
    rate = np.linspace(-2.0, 3.0, 6)
    result = f25bz._rk4(lambda _state: rate, initial, 0.25, 2)
    assert np.array_equal(result, initial + 0.25 * rate)


def test_endpoint_errors_respect_the_162_280_28_partition():
    truth = np.ones(470)
    start = np.zeros(470)
    predicted = truth.copy()
    predicted[:162] += 0.1
    predicted[162:442] += 0.2
    predicted[442:] += 0.3
    errors = f25bz._endpoint_errors(predicted, truth, start)
    assert np.isclose(errors["q162"], 0.1)
    assert np.isclose(errors["z280"], 0.2)
    assert np.isclose(errors["a28"], 0.3)
    assert 0.1 < errors["full"] < 0.3


def test_zero_departure_extension_is_explicit_and_finite():
    # Exercise the zero-extension without constructing the expensive model.
    model = object.__new__(f25bz.ReducedVectorField)
    result = f25bz.ReducedVectorField.nonlinear_departure(model, np.zeros(28))
    assert np.array_equal(result, np.zeros(28))


def test_truth_policy_cannot_use_the_reduced_forecast_as_predictor():
    contract = f25bz.manifest._contract()
    assert contract["reference_sequence"]["truth_predictor"] == (
        "accepted_history_predictor_not_reduced_forecast"
    )
    gates = contract["binding_prospective_forecast_gates"]
    assert gates["truth_root_maximum_scaled_residual"] == 1.0e-10
    assert gates["truth_root_maximum_exact_Jacobian_assemblies"] == 1
    assert contract["decision"]["physical_microburst_authorized"] is False
    assert contract["decision"]["reduced_slow_evolution_authorized"] is False
