from __future__ import annotations

import json

import numpy as np

import run_causal_inner_partitioned_authentic_center_field_blind_validation_wp10c9d6c7c3b5c4f25cw as f25cw


_FROZEN = None
_INPUTS = None


def _frozen():
    global _FROZEN
    if _FROZEN is None:
        _FROZEN = f25cw._validate_manifest(require_clean=False)
    return _FROZEN


def _inputs():
    global _INPUTS
    if _INPUTS is None:
        _INPUTS = f25cw._load_inputs(_frozen())
    return _INPUTS


def test_partitioned_manifest_authorizes_exactly_four_blind_calls():
    frozen = _frozen()
    gates = frozen["contract"]["blind_holdout_execution"]
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25cw.WORK_PACKAGE
    assert gates["count"] == 4
    assert gates["completed_exact_rate_calls_equal"] == 4
    assert gates["coefficients_may_not_change"]


def test_inputs_are_only_the_four_frozen_holdouts():
    inputs = _inputs()
    assert inputs["states"].shape == (4, 112, 5)
    assert inputs["local_coordinates"].shape == (4, 470)
    assert inputs["absolute_coordinates"].shape == (4, 470)
    assert np.array_equal(inputs["partition_weights"], np.ones(4))
    assert np.max(inputs["coordinate_roundtrip_relative_errors"]) <= 1.0e-8


def test_progress_dimensions_and_budget_are_exact():
    assert f25cw._progress_array_shapes() == {
        "total_rates_per_second": (560,),
        "free_rates_per_second": (560,),
        "physical_reaction_actions_per_second": (560,),
        "multiplier_coordinates_per_second": (3,),
        "exact_coordinate_rates_per_second": (470,),
        "predicted_full_rates_per_second": (560,),
        "predicted_coordinate_rates_per_second": (470,),
        "exact_q162_Jacobians": (162, 560),
        "predicted_q162_Jacobians": (162, 560),
        "decoded_scaled_deltas": (560,),
        "decoded_absolute_coordinates": (470,),
    }
    assert f25cw.HOLDOUT_COUNT == 4


def test_online_prediction_forbids_coordinate_jacobian():
    inputs = _inputs()
    predicted, wall = f25cw._online_prediction_without_coordinate_jacobian(
        inputs, 0
    )
    assert predicted["full_rate"].shape == (560,)
    assert predicted["coordinate_rate"].shape == (470,)
    assert predicted["q162_Jacobian"].shape == (162, 560)
    assert predicted["decoded_state"].shape == (112, 5)
    assert wall >= 0.0


def test_canonical_blind_validation_if_present():
    if not f25cw.CANONICAL_DIRECTORY.exists():
        return
    f25cw._checksums(f25cw.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25cw.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25cw.CANONICAL_DIRECTORY / "validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    arrays = f25cw._load_npz(
        f25cw.CANONICAL_DIRECTORY / "validation_arrays.npz"
    )
    assert summary["classification"] in (
        f25cw.PASS_CLASSIFICATION,
        f25cw.FAIL_CLASSIFICATION,
    )
    assert summary["completed_exact_rate_calls"] <= 4
    assert not summary["coefficients_refit_after_holdout_truth"]
    assert arrays["holdout_primitive_states"].shape[0] <= 4
    if summary["passed"]:
        assert summary["classification"] == f25cw.PASS_CLASSIFICATION
        assert summary["authorized_next"] == f25cw.PASS_AUTHORIZED_NEXT
        assert metrics["passed"] and all(metrics["checks"].values())
