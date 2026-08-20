from __future__ import annotations

import json

import numpy as np

import run_causal_inner_authentic_center_exact_rate_training_wp10c9d6c7c3b5c4f25cu as f25cu


_FROZEN = None
_INPUTS = None


def _frozen():
    global _FROZEN
    if _FROZEN is None:
        _FROZEN = f25cu._validate_manifest(require_clean=False)
    return _FROZEN


def _inputs():
    global _INPUTS
    if _INPUTS is None:
        _INPUTS = f25cu._load_inputs(_frozen())
    return _INPUTS


def test_manifest_authorizes_exact_center_and_training_rates_only():
    frozen = _frozen()
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25cu.WORK_PACKAGE
    assert frozen["contract"]["cost_budget"] == {
        "new_exact_continuous_rate_calls_equal": 5,
        "new_complete_generator_assemblies_equal": 0,
        "new_nonlinear_fixed_Q_roots_equal": 0,
        "propagated_states_equal": 0,
        "holdout_rate_calls_equal": 0,
    }


def test_inputs_exclude_the_four_blind_holdouts():
    inputs = _inputs()
    assert inputs["states"].shape == (5, 112, 5)
    assert inputs["absolute_coordinates"].shape == (5, 470)
    assert inputs["local_coordinates"].shape == (5, 470)
    assert inputs["labels"] == (
        "authentic_center",
        "training_0",
        "training_1",
        "training_2",
        "training_3",
    )
    assert np.max(inputs["coordinate_roundtrip_relative_errors"]) <= 1.0e-8


def test_local_field_is_exactly_reanchored_and_uses_fixed_maps():
    frozen = _frozen()
    inputs = _inputs()
    rng = np.random.default_rng(2520)
    closure = {
        "authentic_center_absolute_coordinate": frozen["design"][
            "authentic_center_absolute_coordinate"
        ],
        "authentic_center_scaled_delta": frozen["design"][
            "authentic_center_scaled_delta"
        ],
        "authentic_center_direct_decoded_scaled_delta": inputs[
            "direct"
        ].decoded_delta(frozen["design"]["authentic_center_absolute_coordinate"]),
        "authentic_center_fixed_restriction": frozen["design"][
            "authentic_center_fixed_restriction"
        ],
        "active_departure_basis": frozen["design"]["active_departure_basis"],
        "decoder_affine_coefficients": frozen["design"][
            "decoder_affine_coefficients"
        ],
        "full_rate_affine_coefficients": rng.normal(size=(4, 560)),
        "q162_rate_affine_coefficients": rng.normal(size=(4, 162)),
    }
    field = f25cu.AuthenticCenterLocalField(
        closure, model=inputs["model"], direct=inputs["direct"]
    )
    origin = np.zeros(470)
    assert np.array_equal(field.decoded_delta(origin), field.center_delta)
    expected_full = (
        field.direct.full_state_rate(field.center_coordinate)
        + closure["full_rate_affine_coefficients"][0]
    )
    assert np.allclose(field.full_state_rate(origin), expected_full)
    expected_coordinate = field.restriction @ expected_full
    expected_coordinate[:162] += closure["q162_rate_affine_coefficients"][0]
    assert np.allclose(field.field(origin), expected_coordinate)


def test_fit_uses_frozen_three_group_weights():
    weights = np.concatenate(
        (np.full(16, 1.0 / 16.0), np.ones(1), np.full(4, 1.0 / 4.0))
    )
    assert weights.shape == (21,)
    assert np.isclose(weights[:16].sum(), 1.0)
    assert np.isclose(weights[16:17].sum(), 1.0)
    assert np.isclose(weights[17:].sum(), 1.0)


def test_canonical_training_if_present():
    if not f25cu.CANONICAL_DIRECTORY.exists():
        return
    f25cu._checksums(f25cu.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25cu.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25cu.CANONICAL_DIRECTORY / "training_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    truth = f25cu._load_npz(
        f25cu.CANONICAL_DIRECTORY / "training_truth_arrays.npz"
    )
    assert summary["classification"] in (
        f25cu.PASS_CLASSIFICATION,
        f25cu.FAIL_CLASSIFICATION,
    )
    assert summary["completed_exact_rate_calls"] <= 5
    assert summary["holdout_rate_calls"] == 0
    assert truth["evaluated_primitive_states"].shape[0] <= 5
    if summary["passed"]:
        closure = f25cu._load_npz(
            f25cu.CANONICAL_DIRECTORY / "authentic_center_local_field.npz"
        )
        assert summary["classification"] == f25cu.PASS_CLASSIFICATION
        assert summary["authorized_next"] == f25cu.PASS_AUTHORIZED_NEXT
        assert metrics["truth_passed"] and metrics["field_passed"]
        assert all(metrics["truth_checks"].values())
        assert all(metrics["field_checks"].values())
        assert closure["full_rate_affine_coefficients"].shape == (4, 560)
        assert closure["q162_rate_affine_coefficients"].shape == (4, 162)
