from __future__ import annotations

import numpy as np

import run_causal_inner_coupled_physical_memory_metric_chart_recovery_manifest_wp10c9d6c7c3b5c4f25fiy as target


def _evaluation():
    lock = target._validate_parent(require_clean=False)
    return lock, target._evaluate(lock)


def test_parent_is_the_unpropagated_stage1_metric_boundary() -> None:
    lock = target._validate_parent(require_clean=False)
    values = lock["metrics"]["gate_values"]
    assert lock["summary"]["classification"] == target.parent.PHYSICAL_FAILURE_CLASSIFICATION
    assert not lock["summary"]["passed"]
    assert lock["summary"]["authorized_next"] is None
    assert values["accepted_segments"] == 42
    assert values["attempted_segments"] == 43
    assert values["maximum_metric_coordinate_jacobian_condition"] > 10.0
    assert values["minimum_reconstruction_factor"] == 1.0
    assert values["minimum_scattering_optical_depth"] > 1.0
    assert values["maximum_height_ratio"] < 0.5


def test_nested_partition_selects_coupled_physical_memory_chart() -> None:
    _lock, (metrics, _seed, _definitions) = _evaluation()
    original, coupled, full = metrics["chart_candidates"]
    assert metrics["passed"]
    assert metrics["classification"] == target.CLASSIFICATION
    assert metrics["selected_candidate_index"] == 1
    assert metrics["selected_block_sizes"] == [442, 28]
    assert not original["selection_passed"]
    assert original["metric_jacobian_condition_number"] > 5.0
    assert coupled["selection_passed"]
    assert coupled["metric_jacobian_condition_number"] < 3.0
    assert coupled["metric_augmented_condition_number"] < 3.0
    assert coupled["transition_from_parent_condition_number"] < 10.0
    assert full["selection_passed"]
    assert full["metric_jacobian_condition_number"] < 1.0 + 1e-9


def test_recovery_seed_reanchors_without_advancing_history() -> None:
    _lock, (metrics, seed, _definitions) = _evaluation()
    parent_arrays = target._parent_arrays()
    assert metrics["new_truth_evaluations"] == 0
    assert metrics["new_retractions"] == 0
    assert metrics["new_accepted_segments"] == 0
    np.testing.assert_array_equal(
        seed["current_coordinate470"], parent_arrays["current_coordinate470"]
    )
    np.testing.assert_array_equal(
        seed["current_primitive_state"], parent_arrays["current_primitive_state"]
    )
    np.testing.assert_array_equal(
        seed["accepted_endpoint_coordinates470"],
        parent_arrays["accepted_endpoint_coordinates470"],
    )
    assert int(seed["accepted_segments_new"]) == 42
    assert int(seed["attempts"]) == 43
    assert seed["selected_metric_block_sizes"].tolist() == [442, 28]
    assert seed["next_candidate_target470"].shape == (470,)


def test_original_jacobian_is_reconstructed_from_committed_metric_rows() -> None:
    _lock, (metrics, seed, _definitions) = _evaluation()
    parent_arrays = target._parent_arrays()
    jacobian = seed["reconstructed_anchor_jacobian470x560"]
    reconstructed_metric = parent_arrays["metric_transform470x470"] @ jacobian
    np.testing.assert_allclose(
        reconstructed_metric,
        parent_arrays["metric_augmented560x560"][:470],
        rtol=0.0,
        atol=1e-10,
    )
    assert (
        metrics["anchor_jacobian_reconstruction_relative_defect"]
        <= target.MAXIMUM_JACOBIAN_RECONSTRUCTION_DEFECT
    )


def test_recovery_preserves_phase_observer_and_original_physics() -> None:
    _lock, (metrics, seed, definitions) = _evaluation()
    original_seed = target._load_npz(
        target.parent.manifest.CANONICAL_DIRECTORY / "continuation_seed.npz"
    )
    contract = definitions["contract"]
    architecture = definitions["architecture"]
    np.testing.assert_array_equal(
        seed["phase_observer_metric_transform470x470"],
        original_seed["phase_observer_metric_transform470x470"],
    )
    assert contract["selected_partition"] == [442, 28]
    assert contract["selected_partition_semantics"]["original_coordinate_unchanged"]
    assert contract["selected_partition_semantics"][
        "physics_and_ledgers_remain_in_original_coordinates"
    ]
    assert "W R(q)=0 iff R(q)=0" in architecture["why_root_and_physics_are_invariant"]
    assert not metrics["complete_cycle_execution_authorized"]
    assert not metrics["reduced_slow_evolution_authorized"]


def test_only_one_boundary_endpoint_is_authorized() -> None:
    _lock, (metrics, _seed, definitions) = _evaluation()
    scope = definitions["contract"]["authorized_scope"]
    assert metrics["authorized_next"] == target.AUTHORIZED_NEXT
    assert scope["maximum_new_accepted_segments"] == 1
    assert scope["maximum_retractions"] == 1
    assert scope["maximum_exact_free_field_calls"] == 1
    assert scope["segment_seconds"] == 2.5e-4
    assert scope["fixed_Q_calls"] == 0
    assert scope["nonlinear_roots"] == 0


def test_condition_threshold_is_preserved_not_relaxed() -> None:
    _lock, (_metrics, _seed, definitions) = _evaluation()
    contract = definitions["contract"]
    gates = contract["binding_recovery_gates"]
    assert gates["maximum_metric_jacobian_condition"] == 10.0
    assert gates["maximum_metric_augmented_condition"] == 10.0
    assert "raise or remove the metric condition threshold" in contract["forbidden"]
