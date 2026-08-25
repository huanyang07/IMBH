from __future__ import annotations

import numpy as np

import run_causal_inner_tangent_phase_hyperbolicity_two_half_step_bracket_manifest_wp10c9d6c7c3b5c4f25fizdc as target


def test_parent_is_the_nonpropagating_genuine_boundary_diagnosis() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.PASS_CLASSIFICATION
    assert validated["summary"]["nonpropagating"]
    assert not validated["summary"]["failed_candidate_propagated"]


def test_seed_contains_only_accepted_history_for_propagation() -> None:
    seed = target._half_step_seed()
    assert seed["accepted_endpoint_coordinates470"].shape == (71, 470)
    assert seed["accepted_endpoint_primitive_states"].shape == (71, 112, 5)
    np.testing.assert_array_equal(
        seed["current_primitive_state"],
        seed["accepted_endpoint_primitive_states"][-1],
    )
    assert not np.array_equal(
        seed["current_primitive_state"],
        seed["failed_retracted_primitive_state"],
    )


def test_two_steps_exactly_subdivide_the_rejected_full_step() -> None:
    contract = target._contract()
    scope = contract["scope"]
    assert 2.0 * scope["half_step_seconds"] == target.FULL_STEP_SECONDS
    assert scope["maximum_accepted_half_steps"] == 2
    assert scope["maximum_attempted_half_steps"] == 2
    assert scope["tentative_segment_numbers"] == [268, 269]
    assert scope["blind_midpoint_segment_numbers"] == []
    assert scope["maximum_exact_free_field_calls"] == 2
    assert scope["maximum_retractions"] == 2


def test_hyperbolicity_is_checked_before_each_exact_field() -> None:
    gate = target._contract()["binding_hyperbolicity_gate"]
    assert gate["all_113_face_generalized_pencils_checked"]
    assert gate["checked_after_retraction_before_exact_field"]
    assert gate["maximum_imaginary_coordinate_speed"] == 1.0e-10
    assert gate["no_complex_flux_split"]
    assert gate["no_real_part_coercion"]


def test_only_two_accepted_half_steps_can_authorize_continuation() -> None:
    branches = target._contract()["classification_branches"]
    assert "halved-step stage2 continuation manifest" in branches[
        "both_authentic_half_steps_pass"
    ]
    assert "propagate nothing" in branches["first_half_step_is_complex"]
    assert "accept only the first half step" in branches[
        "first_passes_second_is_complex"
    ]


def test_complete_cycle_and_reduced_evolution_are_forbidden() -> None:
    forbidden = target._contract()["forbidden"]
    assert "propagate the saved failed full-step candidate" in forbidden
    assert "replace a complex eigenvalue by its real part" in forbidden
    assert "authorize a phase lap, complete cycle, or reduced slow evolution" in forbidden
