from __future__ import annotations

import math

import numpy as np

import run_causal_inner_tangent_phase_lap_recurrence_manifest_wp10c9d6c7c3b5c4f25fiw as target


def _evaluation():
    lock = target._validate_parent(require_clean=False)
    return lock, target._evaluate(lock)


def test_parent_is_the_prospectively_validated_holdout() -> None:
    lock = target._validate_parent(require_clean=False)
    values = lock["metrics"]["gate_values"]
    assert lock["summary"]["classification"] == target.parent.PASS_CLASSIFICATION
    assert lock["summary"]["authorized_next"] == target.WORK_PACKAGE
    assert values["accepted_segments"] == 16
    assert values["all_phase_holdouts_passed"]
    assert not values["phase_lap_observed"]
    assert not values["cycle_observed"]


def test_measured_cost_and_phase_support_three_bounded_stages() -> None:
    _lock, (metrics, _seed, _definitions) = _evaluation()
    observed = metrics["observations"]
    assert metrics["classification"] == target.CLASSIFICATION
    assert metrics["passed"]
    assert metrics["definitions_only"]
    assert observed["segments_to_lap_at_observed_minimum"] == 141
    assert observed["planned_accepted_segments"] == 144
    assert observed["planned_exact_field_calls"] == 162
    assert observed["projected_stage1_wall_hours"] < target.STAGE_WALL_HOURS
    assert 15.0 < observed["projected_full_acquisition_wall_hours"] < 18.0


def test_seed_registers_phase_and_section_at_the_terminal_state() -> None:
    _lock, (_metrics, seed, _definitions) = _evaluation()
    transform = seed["phase_observer_metric_transform470x470"]
    tangent = seed["phase_lap_reference_unit_tangent470"]
    covector = seed["registered_section_covector470"]
    reference = seed["phase_lap_reference_coordinate470"]
    assert seed["phase_training_raw_rates470_per_s"].shape == (12, 470)
    assert transform.shape == (470, 470)
    np.testing.assert_allclose(np.linalg.norm(tangent), 1.0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(covector, transform.T @ tangent, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(reference, seed["current_coordinate470"])
    assert float(covector @ (reference - reference)) == 0.0
    assert float(seed["unwrapped_phase_advance_radians"]) == 0.0
    assert float(seed["accumulated_metric_path_length"]) == 0.0


def test_phase_lap_is_separate_from_coarse_recurrence_and_cycle() -> None:
    _lock, (metrics, _seed, definitions) = _evaluation()
    contract = definitions["contract"]
    gates = contract["coarse_recurrence_candidate_requires"]
    assert contract["phase_registration"]["phase_lap_radians"] == 2.0 * math.pi
    assert contract["staged_scope"]["only_stage1_is_authorized_now"]
    assert len(contract["staged_scope"]["stages"]) == 3
    assert gates[
        "maximum_metric_state_return_distance_over_accumulated_path_length"
    ] == 0.10
    assert gates["minimum_metric_tangent_cosine"] == 0.99
    assert gates["registered_section_bracket"] == "g_previous < 0 <= g_current"
    assert not metrics["complete_cycle_execution_authorized"]
    assert not metrics["reduced_slow_evolution_authorized"]


def test_online_architecture_excludes_truth_and_micro_time_stepping() -> None:
    _lock, (_metrics, _seed, definitions) = _evaluation()
    architecture = definitions["architecture"]
    reduced = architecture[
        "slow_architecture_only_after_periodic_family_is_certified"
    ]
    assert architecture["phase_lap_is_not_a_cycle"]
    assert "multiple shooting" in architecture[
        "cycle_refinement_if_coarse_candidate_passes"
    ]["then"]
    assert reduced["online_state"] == "slow Q, mode label, and event state only"
    for forbidden in ("truth integration", "nonlinear roots", "micro-BDF"):
        assert forbidden in reduced["online_forbidden"]


def test_only_stage1_execution_is_authorized() -> None:
    _lock, (metrics, _seed, definitions) = _evaluation()
    stage = definitions["contract"]["staged_scope"]["stage1"]
    assert metrics["authorized_next"] == target.AUTHORIZED_NEXT
    assert stage["accepted_segments"] == 48
    assert stage["maximum_exact_free_field_calls"] == 54
    assert stage["maximum_retractions"] == 54
    assert stage["maximum_wall_hours"] == 7.0
