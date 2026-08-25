from __future__ import annotations

import numpy as np

import run_causal_inner_tangent_phase_lap_recurrence_stage2_manifest_wp10c9d6c7c3b5c4f25fizc as target


def _evaluation():
    lock = target._validate_parent(require_clean=False)
    return lock, target._evaluate(lock)


def test_parent_is_the_completed_stage1_chain() -> None:
    lock = target._validate_parent(require_clean=False)
    values = lock["metrics"]["gate_values"]
    assert lock["summary"]["classification"] == target.parent.PASS_CLASSIFICATION
    assert lock["summary"]["passed"]
    assert lock["summary"]["authorized_next"] == target.WORK_PACKAGE
    assert values["combined_accepted_phase_endpoints"] == 48
    assert values["cumulative_phase_advance_radians"] > 2.0
    assert not values["phase_lap_observed"]
    assert not values["coarse_recurrence_candidate_observed"]
    assert values["all_endpoint_metric_blocks_are_442_plus_28"]


def test_stage2_seed_is_the_exact_48_endpoint_terminal_history() -> None:
    seed = target._stage2_seed()
    arrays = target._load_npz(
        target.parent.CANONICAL_DIRECTORY / "stage1_resume_arrays.npz"
    )
    assert seed["accepted_endpoint_coordinates470"].shape == (48, 470)
    assert seed["accepted_endpoint_primitive_states"].shape == (48, 112, 5)
    assert seed["accepted_endpoint_coordinate_rates470_per_s"].shape == (48, 470)
    assert seed["accepted_phase_increments"].shape == (48,)
    assert seed["accepted_cumulative_phase_advance_radians"].shape == (48,)
    assert int(seed["accepted_segments_total"]) == 244
    assert int(seed["accepted_segments_new"]) == 0
    assert int(seed["attempts"]) == 0
    assert int(seed["acquisition_stage"]) == 2
    assert seed["selected_metric_block_sizes"].tolist() == [442, 28]
    np.testing.assert_array_equal(
        seed["current_coordinate470"], arrays["current_coordinate470"]
    )
    np.testing.assert_array_equal(
        seed["current_primitive_state"], arrays["current_primitive_state"]
    )


def test_stage2_scope_is_exactly_48_endpoints_and_six_blinds() -> None:
    _lock, (metrics, _seed, definitions) = _evaluation()
    observations = metrics["observations"]
    scope = definitions["contract"]["scope"]
    assert metrics["passed"]
    assert observations["stage2_segment_numbers"] == list(range(245, 293))
    assert observations["blind_midpoint_segment_numbers"] == [
        248,
        256,
        264,
        272,
        280,
        288,
    ]
    assert observations["maximum_exact_field_and_retraction_units"] == 54
    assert observations["projected_stage2_wall_hours"] < 7.0
    assert scope["new_accepted_phase_endpoints"] == 48
    assert scope["maximum_attempted_endpoints"] == 52
    assert scope["maximum_exact_free_field_calls"] == 54


def test_stage2_cannot_reach_phase_lap_under_binding_increment_gate() -> None:
    _lock, (metrics, _seed, definitions) = _evaluation()
    observations = metrics["observations"]
    proof = definitions["contract"]["phase_lap_scope_proof"]
    architecture = definitions["architecture"]
    assert not observations[
        "phase_lap_reachable_within_stage2_under_binding_increment_gate"
    ]
    assert proof["maximum_terminal_phase_under_binding_increment_gate"] < (
        proof["two_pi_radians"]
    )
    assert not proof["phase_lap_reachable_within_stage2"]
    assert architecture[
        "stage2_phase_lap_is_impossible_under_its_binding_increment_ceiling"
    ]


def test_coupled_chart_and_all_original_gates_are_preserved() -> None:
    _lock, (_metrics, _seed, definitions) = _evaluation()
    contract = definitions["contract"]
    chart = contract["computational_chart"]
    gates = contract["binding_stage2_gates"]
    assert chart["block_sizes"] == [442, 28]
    assert chart["maximum_metric_and_augmented_condition"] == 10.0
    assert chart["primitive_state_and_original_coordinate_unchanged"]
    assert chart["all_physics_and_ledgers_in_original_coordinates"]
    assert gates["all_original_retraction_and_physical_gates_unchanged"]
    assert gates["all_checkpoint_roundtrips_bitwise"]
    assert gates["suffix_history_replay_bitwise"]
    assert gates["accepted_history_only_propagation"]


def test_stage2_authorizes_no_cycle_or_reduced_execution() -> None:
    _lock, (metrics, _seed, definitions) = _evaluation()
    architecture = definitions["architecture"]
    forbidden = definitions["contract"]["forbidden"]
    assert metrics["authorized_next"] == target.AUTHORIZED_NEXT
    assert not metrics["complete_cycle_execution_authorized"]
    assert not metrics["reduced_slow_evolution_authorized"]
    assert architecture["stage2_is_not_a_complete_cycle_execution"]
    assert "authorize complete-cycle execution or reduced slow evolution" in forbidden
    assert "event-limited stage3" in architecture["next_if_passed"]
