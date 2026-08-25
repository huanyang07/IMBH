from __future__ import annotations

from pathlib import Path

import numpy as np

import run_causal_inner_tangent_phase_lap_stage1_resume_execution_wp10c9d6c7c3b5c4f25fizb as target


def test_manifest_authorizes_only_the_five_endpoint_resume() -> None:
    lock = target._validate_manifest(require_clean=False)
    scope = lock["contract"]["scope"]
    assert lock["summary"]["authorized_next"] == target.WORK_PACKAGE
    assert scope["new_accepted_endpoints"] == 5
    assert scope["maximum_attempted_endpoints"] == 5
    assert scope["maximum_exact_free_field_calls"] == 6
    assert scope["blind_midpoint_segment_numbers"] == [240]


def test_initial_progress_is_the_accepted_recovery_checkpoint() -> None:
    seed = target._seed()
    progress = target._initial_progress()
    assert progress["accepted_segments_total"] == 239
    assert progress["accepted_segments_new"] == 0
    assert progress["attempts"] == 0
    assert progress["elapsed_seconds"] == target.INITIAL_ELAPSED_SECONDS
    np.testing.assert_array_equal(
        progress["current_coordinate"], seed["current_coordinate470"]
    )
    assert seed["selected_metric_block_sizes"].tolist() == [442, 28]


def test_phase_history_uses_only_accepted_canonical_endpoints(
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


def test_first_resume_endpoint_is_the_scheduled_blind_segment() -> None:
    progress = target._initial_progress()
    assert progress["accepted_segments_total"] + 1 == 240
    with target._execution_context():
        policy = target.engine._policy()
        assert target.engine.blind_midpoint_required(240, policy)
        assert not target.engine.blind_midpoint_required(241, policy)


def test_execution_context_selects_coupled_chart_and_restores_globals() -> None:
    original_blocks = target.suffix._block_sizes
    original_manifest = target.engine.manifest
    with target._execution_context():
        assert target.suffix._block_sizes() == (442, 28)
        assert target.engine.manifest is target.sys.modules[target.__name__]
        assert target.phase._seed is target._seed
    assert target.suffix._block_sizes is original_blocks
    assert target.engine.manifest is original_manifest


def test_accepted_attempt_inventory_ignores_provisional_engine_record(
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


def test_phase_lap_and_cycle_remain_unauthorized_by_stage_completion() -> None:
    lock = target._validate_manifest(require_clean=False)
    contract = lock["contract"]
    assert "stage1 complete" in contract["classification_branches"][
        "five_endpoints_and_all_gates_pass"
    ]
    assert "authorize a complete cycle or reduced slow evolution" in contract[
        "forbidden"
    ]
