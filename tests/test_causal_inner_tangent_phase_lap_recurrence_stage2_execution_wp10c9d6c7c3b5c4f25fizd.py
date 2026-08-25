from __future__ import annotations

from pathlib import Path

import numpy as np

import run_causal_inner_tangent_phase_lap_recurrence_stage2_execution_wp10c9d6c7c3b5c4f25fizd as target


def test_manifest_authorizes_only_the_bounded_stage2_execution() -> None:
    lock = target._validate_manifest(require_clean=False)
    scope = lock["contract"]["scope"]
    proof = lock["contract"]["phase_lap_scope_proof"]
    assert lock["summary"]["authorized_next"] == target.WORK_PACKAGE
    assert scope["new_accepted_phase_endpoints"] == 48
    assert scope["maximum_attempted_endpoints"] == 52
    assert scope["maximum_exact_free_field_calls"] == 54
    assert scope["blind_midpoint_segment_numbers"] == [
        248,
        256,
        264,
        272,
        280,
        288,
    ]
    assert not proof["phase_lap_reachable_within_stage2"]


def test_initial_progress_is_the_exact_stage1_terminal_checkpoint() -> None:
    seed = target._seed()
    progress = target._initial_progress()
    assert progress["accepted_segments_total"] == 244
    assert progress["accepted_segments_new"] == 0
    assert progress["attempts"] == 0
    assert progress["elapsed_seconds"] == target.INITIAL_ELAPSED_SECONDS
    np.testing.assert_array_equal(
        progress["current_coordinate"], seed["current_coordinate470"]
    )
    np.testing.assert_array_equal(
        progress["current_state"], seed["current_primitive_state"]
    )
    assert seed["accepted_endpoint_coordinates470"].shape == (48, 470)
    assert seed["selected_metric_block_sizes"].tolist() == [442, 28]


def test_phase_history_and_accumulation_use_only_accepted_seed(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(target, "SCRATCH_DIRECTORY", tmp_path)
    seed = target._seed()
    history = target._phase_history()
    prior = target._prior_accumulation()
    np.testing.assert_array_equal(
        history,
        seed["accepted_endpoint_coordinate_rates470_per_s"][-12:],
    )
    assert prior["cumulative_phase_advance_radians"] == float(
        seed["unwrapped_phase_advance_radians"]
    )
    assert prior["cumulative_metric_path_length"] == float(
        seed["accumulated_metric_path_length"]
    )


def test_stage2_blind_midpoint_schedule_is_preserved() -> None:
    progress = target._initial_progress()
    assert progress["accepted_segments_total"] + 1 == 245
    with target._execution_context():
        policy = target.engine._policy()
        assert not target.engine.blind_midpoint_required(245, policy)
        assert target.engine.blind_midpoint_required(248, policy)
        assert target.engine.blind_midpoint_required(256, policy)


def test_execution_context_selects_coupled_chart_and_restores_globals() -> None:
    original_blocks = target.suffix._block_sizes
    original_manifest = target.engine.manifest
    with target._execution_context():
        assert target.suffix._block_sizes() == (442, 28)
        assert target.engine.manifest is target.sys.modules[target.__name__]
        assert target.phase._seed is target._seed
    assert target.suffix._block_sizes is original_blocks
    assert target.engine.manifest is original_manifest


def test_accepted_attempt_inventory_ignores_provisional_record(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(target, "SCRATCH_DIRECTORY", tmp_path)
    directory = tmp_path / "attempt_0000"
    directory.mkdir()
    target._helper()._write_json(
        directory / "attempt.json",
        {"accepted": True, "phase_geometry": None, "recurrence_geometry": None},
    )
    target._save_npz(
        directory / "attempt.npz",
        {"accepted_coordinate_rate470_per_s": np.zeros(470)},
    )
    assert target._accepted_attempts() == []


def test_stage2_completion_still_cannot_authorize_cycle() -> None:
    lock = target._validate_manifest(require_clean=False)
    forbidden = lock["contract"]["forbidden"]
    assert "treat stage2 as a phase lap or cycle certificate" in forbidden
    assert "authorize complete-cycle execution or reduced slow evolution" in forbidden
