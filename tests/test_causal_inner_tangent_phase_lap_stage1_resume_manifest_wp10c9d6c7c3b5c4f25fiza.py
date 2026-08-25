from __future__ import annotations

import numpy as np

import run_causal_inner_tangent_phase_lap_stage1_resume_manifest_wp10c9d6c7c3b5c4f25fiza as target


def _evaluation():
    lock = target._validate_parent(require_clean=False)
    return lock, target._evaluate(lock)


def test_parent_is_the_prospective_boundary_recovery() -> None:
    lock = target._validate_parent(require_clean=False)
    values = lock["metrics"]["gate_values"]
    assert lock["summary"]["classification"] == target.parent.PASS_CLASSIFICATION
    assert lock["summary"]["passed"]
    assert lock["summary"]["authorized_next"] == target.WORK_PACKAGE
    assert values["accepted_segments"] == 1
    assert values["selected_metric_block_sizes"] == [442, 28]
    assert values["maximum_metric_coordinate_jacobian_condition"] < 10.0
    assert values["phase_geometry_passed"]


def test_resume_seed_contains_exactly_43_accepted_endpoints() -> None:
    seed = target._resume_seed()
    assert seed["accepted_endpoint_coordinates470"].shape == (43, 470)
    assert seed["accepted_endpoint_primitive_states"].shape == (43, 112, 5)
    assert seed["accepted_endpoint_coordinate_rates470_per_s"].shape == (43, 470)
    assert seed["accepted_phase_increments"].shape == (43,)
    assert int(seed["accepted_segments_total"]) == 239
    assert int(seed["accepted_segments_new"]) == 0
    assert int(seed["attempts"]) == 0
    assert seed["selected_metric_block_sizes"].tolist() == [442, 28]


def test_resume_seed_terminal_state_is_boundary_recovery_endpoint() -> None:
    seed = target._resume_seed()
    recovery = target._load_npz(
        target.parent.CANONICAL_DIRECTORY / "boundary_recovery_arrays.npz"
    )
    np.testing.assert_array_equal(
        seed["current_coordinate470"], recovery["current_coordinate470"]
    )
    np.testing.assert_array_equal(
        seed["current_primitive_state"], recovery["current_primitive_state"]
    )
    np.testing.assert_array_equal(
        seed["metric_transform470x470"], recovery["metric_transform470x470"]
    )


def test_only_five_endpoints_and_one_blind_are_planned() -> None:
    _lock, (metrics, _seed, definitions) = _evaluation()
    observations = metrics["observations"]
    scope = definitions["contract"]["scope"]
    assert metrics["passed"]
    assert observations["remaining_accepted"] == 5
    assert observations["resume_segment_numbers"] == [240, 241, 242, 243, 244]
    assert observations["blind_midpoint_segment_numbers"] == [240]
    assert observations["maximum_exact_field_and_retraction_units"] == 6
    assert observations["projected_resume_wall_hours"] < 1.5
    assert scope["maximum_attempted_endpoints"] == 5
    assert scope["maximum_exact_free_field_calls"] == 6


def test_original_stage_rejection_and_metric_gate_are_preserved() -> None:
    _lock, (_metrics, _seed, definitions) = _evaluation()
    contract = definitions["contract"]
    assert contract["preserved_classifications"]["original_stage1_boundary"] == (
        target.parent.manifest.parent.PHYSICAL_FAILURE_CLASSIFICATION
    )
    assert contract["computational_chart"]["block_sizes"] == [442, 28]
    assert contract["computational_chart"][
        "maximum_metric_and_augmented_condition"
    ] == 10.0
    assert "retroactively pass the rejected f25fix execution" in contract["forbidden"]


def test_stage1_completion_does_not_authorize_cycle_or_reduced_model() -> None:
    _lock, (metrics, _seed, definitions) = _evaluation()
    architecture = definitions["architecture"]
    assert metrics["authorized_next"] == target.AUTHORIZED_NEXT
    assert not metrics["complete_cycle_execution_authorized"]
    assert not metrics["reduced_slow_evolution_authorized"]
    assert architecture["stage1_completion_is_not_a_phase_lap_or_cycle"]
    assert "tabulated averaged drift" in architecture[
        "online_reduced_architecture_unchanged"
    ]
