from __future__ import annotations

from pathlib import Path

import numpy as np

import run_causal_inner_coupled_physical_memory_metric_chart_boundary_recovery_execution_wp10c9d6c7c3b5c4f25fiz as target


def test_manifest_authorizes_exactly_one_boundary_attempt() -> None:
    lock = target._validate_manifest(require_clean=False)
    scope = lock["contract"]["authorized_scope"]
    assert lock["summary"]["authorized_next"] == target.WORK_PACKAGE
    assert scope["maximum_new_accepted_segments"] == 1
    assert scope["maximum_retractions"] == 1
    assert scope["maximum_exact_free_field_calls"] == 1
    assert lock["contract"]["selected_partition"] == [442, 28]


def test_initial_progress_uses_recovery_chart_without_advancing() -> None:
    seed = target._seed()
    progress = target._initial_progress()
    assert progress["accepted_segments_new"] == 0
    assert progress["attempts"] == 0
    assert progress["elapsed_seconds"] == target.INITIAL_ELAPSED_SECONDS
    assert progress["next_span"] == target.manifest.RECOVERY_SEGMENT_SECONDS
    np.testing.assert_array_equal(
        progress["metric_transform"], seed["metric_transform470x470"]
    )
    assert seed["selected_metric_block_sizes"].tolist() == [442, 28]


def test_next_candidate_is_recomputed_bitwise_from_accepted_history() -> None:
    seed = target._seed()
    progress = target._initial_progress()
    candidate = target.engine.execution._variable_step_ab2(
        progress["current_coordinate"],
        progress["current_rate"],
        progress["previous_rate"],
        progress["next_span"],
        progress["previous_span"],
    )
    np.testing.assert_array_equal(candidate, seed["next_candidate_target470"])


def test_phase_history_and_accumulation_are_carried_from_accepted_records() -> None:
    seed = target._seed()
    history = target._phase_history()
    prior = target._prior_accumulation()
    np.testing.assert_array_equal(
        history,
        seed["accepted_endpoint_coordinate_rates470_per_s"][-12:],
    )
    assert prior["cumulative_phase_advance_radians"] == float(
        seed["accepted_cumulative_phase_advance_radians"][-1]
    )
    assert prior["cumulative_metric_path_length"] == float(
        seed["accepted_cumulative_metric_path_lengths"][-1]
    )
    assert prior["registered_section_value"] == float(
        seed["accepted_registered_section_values"][-1]
    )


def test_execution_context_selects_two_block_metric_and_restores_globals() -> None:
    original_blocks = target.suffix._block_sizes
    original_manifest = target.engine.manifest
    with target._execution_context():
        assert target.suffix._block_sizes() == (442, 28)
        assert target.engine.manifest is target.sys.modules[target.__name__]
        assert target.parent._seed is target._seed
    assert target.suffix._block_sizes is original_blocks
    assert target.engine.manifest is original_manifest


def test_one_record_replay_fails_closed_without_checkpoint(tmp_path: Path) -> None:
    original = target.SCRATCH_DIRECTORY
    try:
        target.SCRATCH_DIRECTORY = tmp_path
        metrics = {"accepted": True, "attempt_index": 0}
        arrays = {"candidate_target470": target._seed()["next_candidate_target470"]}
        replay, attempt = target._one_record_replay(
            [(metrics, arrays)], target._initial_progress()
        )
    finally:
        target.SCRATCH_DIRECTORY = original
    assert not replay
    assert attempt is None


def test_mocked_pass_requires_chart_phase_recurrence_and_replay(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(target, "SCRATCH_DIRECTORY", tmp_path)
    directory = tmp_path / "attempt_0000"
    directory.mkdir(parents=True, exist_ok=True)
    target._helper()._write_json(
        directory / "attempt.json",
        {
            "attempt_index": 0,
            "accepted": True,
            "physical_failure": False,
            "phase_geometry": {"passed": True, "phase_increment": 0.05},
            "recurrence_geometry": {
                "cumulative_phase_advance_radians": 2.0,
                "cumulative_metric_path_length": 1.2,
                "phase_lap_observed": False,
                "coarse_recurrence_candidate": False,
                "endpoint_registered_section_value": 1.0,
                "endpoint_return_distance_over_path_length": 0.9,
            },
            "endpoint_field": {
                "physical_passed": True,
                "metric_chart": {
                    "block_sizes": [442, 28],
                    "metric_jacobian_condition_number": 3.0,
                    "metric_augmented_condition_number": 3.0,
                },
            },
        },
    )
    target._save_npz(
        directory / "attempt.npz",
        {"candidate_target470": target._seed()["next_candidate_target470"]},
    )
    base = {
        "passed": True,
        "classification": target.PASS_CLASSIFICATION,
        "authorized_next": target.AUTHORIZED_NEXT,
        "gate_values": {
            "accepted_segments": 1,
            "attempted_segments": 1,
            "terminal_elapsed_seconds": 0.17875,
            "exact_free_field_calls": 1,
            "retractions": 1,
            "all_accepted_checkpoint_roundtrips_bitwise": True,
            "suffix_history_replay_bitwise": True,
            "maximum_raw_coordinate_jacobian_condition": 9000.0,
            "maximum_metric_coordinate_jacobian_condition": 3.0,
            "maximum_accepted_endpoint_integral_defect": 0.0006,
        },
    }
    classified, _arrays = target._classify(base, {})
    assert classified["passed"]
    assert classified["classification"] == target.PASS_CLASSIFICATION
    assert classified["authorized_next"] == target.AUTHORIZED_NEXT
