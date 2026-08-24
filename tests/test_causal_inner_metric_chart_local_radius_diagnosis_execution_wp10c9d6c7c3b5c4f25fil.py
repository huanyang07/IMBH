from __future__ import annotations

import numpy as np

import run_causal_inner_metric_chart_local_radius_diagnosis_execution_wp10c9d6c7c3b5c4f25fil as target


def test_manifest_authorizes_only_local_radius_execution() -> None:
    lock = target._validate_manifest(require_clean=False)
    assert lock["summary"]["classification"] == target.manifest.CLASSIFICATION
    assert lock["contract"]["scope"]["exact_free_field_calls"] == 0
    assert lock["contract"]["scope"]["new_trajectory"] is False


def test_targets_use_same_accepted_history_and_requested_spans() -> None:
    seed = target._seed()
    targets = [target._target(seed, span) for span in target.manifest.SPAN_LADDER_SECONDS]
    assert all(value.shape == (470,) for value in targets)
    assert not np.array_equal(targets[0], targets[1])
    assert not np.array_equal(targets[1], targets[2])
    np.testing.assert_array_equal(
        targets[0],
        target.manifest.parent.execution._variable_step_ab2(
            seed["current_coordinate470"],
            seed["current_coordinate_rate470_per_s"],
            seed["previous_coordinate_rate470_per_s"],
            2.0e-3,
            seed["previous_span_seconds"],
        ),
    )


def test_positive_classification_selects_largest_smaller_pass() -> None:
    def record(span, strict_passed, closure, condition, physical, passed):
        return (
            {
                "span_seconds": span,
                "physical_passed": physical,
                "passed": passed,
                "strict_retraction": {
                    "passed": strict_passed,
                    "nonlinear_closure_passed": closure,
                    "chart_condition_passed": condition,
                },
            },
            {},
        )

    records = [
        record(0.002, False, False, False, True, False),
        record(0.001, True, True, True, True, True),
        record(0.0005, True, True, True, True, True),
    ]
    classification, passed, selected = target._classify(
        records, seed_roundtrip_bitwise=True, wall_seconds=10.0
    )
    assert classification == target.PASS_CLASSIFICATION
    assert passed
    assert selected == 0.001


def test_physical_failure_cannot_be_reclassified_as_local_radius() -> None:
    base = {
        "passed": False,
        "nonlinear_closure_passed": False,
        "chart_condition_passed": False,
    }
    records = [
        ({"span_seconds": 0.002, "physical_passed": False, "passed": False, "strict_retraction": base}, {}),
        ({"span_seconds": 0.001, "physical_passed": True, "passed": True, "strict_retraction": {"passed": True, "nonlinear_closure_passed": True, "chart_condition_passed": True}}, {}),
        ({"span_seconds": 0.0005, "physical_passed": True, "passed": True, "strict_retraction": {"passed": True, "nonlinear_closure_passed": True, "chart_condition_passed": True}}, {}),
    ]
    classification, passed, selected = target._classify(
        records, seed_roundtrip_bitwise=True, wall_seconds=10.0
    )
    assert classification == target.PHYSICAL_FAILURE_CLASSIFICATION
    assert not passed
    assert selected is None
