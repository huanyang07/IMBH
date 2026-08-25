from __future__ import annotations

import numpy as np

import run_causal_inner_tangent_phase_lap_stage2_hyperbolicity_boundary_manifest_wp10c9d6c7c3b5c4f25fizda as target


def _evaluation():
    validated = target._validate_execution_contract(require_clean=False)
    return validated, target._evaluate(validated)


def test_interrupted_execution_has_exactly_23_accepted_endpoints() -> None:
    validated = target._validate_execution_contract(require_clean=False)
    assert len(validated["records"]) == 23
    assert all(item["accepted"] for item in validated["records"])
    assert int(validated["terminal"]["accepted_segments_new"]) == 23
    assert int(validated["terminal"]["accepted_segments_total"]) == 267
    assert float(validated["terminal"]["elapsed_seconds"]) == (
        0.18575000000000014
    )


def test_failed_candidate_retraction_passed_but_was_not_propagated() -> None:
    validated = target._validate_execution_contract(require_clean=False)
    failed = target._attempt_directory(target.FAILED_ATTEMPT_INDEX)
    assert validated["prediction"]["tentative_segment_number"] == 268
    assert validated["retraction"]["passed"]
    assert validated["retraction"]["physical_passed"]
    assert validated["retraction"]["maximum_metric_augmented_condition_number"] < 10
    assert not (failed / "endpoint_field.json").exists()
    assert not (failed / "attempt.json").exists()
    assert not (failed / "accepted_checkpoint.npz").exists()


def test_boundary_seed_carries_only_the_accepted_71_endpoint_chain() -> None:
    validated, (_metrics, seed, _definitions) = _evaluation()
    assert seed["combined_accepted_endpoint_coordinates470"].shape == (71, 470)
    assert seed["combined_accepted_endpoint_primitive_states"].shape == (
        71,
        112,
        5,
    )
    assert seed["combined_accepted_endpoint_coordinate_rates470_per_s"].shape == (
        71,
        470,
    )
    assert seed["combined_accepted_phase_increments"].shape == (71,)
    np.testing.assert_array_equal(
        seed["current_primitive_state"],
        validated["terminal"]["current_primitive_state"],
    )
    assert not np.array_equal(
        seed["current_primitive_state"], seed["failed_retracted_primitive_state"]
    )


def test_diagnostic_is_independent_nonpropagating_and_bounded() -> None:
    _validated, (metrics, _seed, definitions) = _evaluation()
    scope = definitions["contract"]["diagnostic_scope"]
    assert metrics["passed"]
    assert scope["independent_five_point_relative_steps"] == [
        1.0e-3,
        2.0e-4,
        2.0e-5,
    ]
    assert scope["maximum_new_free_field_calls"] == 0
    assert scope["maximum_new_retractions"] == 0
    assert scope["maximum_wall_hours"] == 0.25
    assert scope["nonpropagating"]


def test_complex_pair_cannot_be_coerced_into_a_flux_split() -> None:
    _validated, (_metrics, _seed, definitions) = _evaluation()
    forbidden = definitions["contract"]["forbidden"]
    assert "replace the complex pair by its real parts" in forbidden
    assert "relax the 1e-10 real-eigensystem tolerance" in forbidden
    assert "use a complex invariant subspace as a hyperbolic flux split" in forbidden


def test_only_a_prospective_two_half_step_bracket_can_follow() -> None:
    _validated, (metrics, _seed, definitions) = _evaluation()
    architecture = definitions["architecture"]
    assert metrics["authorized_next"] == target.AUTHORIZED_NEXT
    assert "two authentic 0.125 ms steps" in architecture["next_if_genuine"]
    assert not metrics["complete_cycle_execution_authorized"]
    assert not metrics["reduced_slow_evolution_authorized"]
