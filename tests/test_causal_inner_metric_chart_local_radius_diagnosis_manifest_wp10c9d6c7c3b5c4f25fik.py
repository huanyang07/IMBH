from __future__ import annotations

import numpy as np

import run_causal_inner_metric_chart_local_radius_diagnosis_manifest_wp10c9d6c7c3b5c4f25fik as target


def test_parent_is_fail_closed_wide_resume_result() -> None:
    lock = target._validate_parent(require_clean=False)
    assert lock["classification"] == target.parent.NUMERICAL_FAILURE_CLASSIFICATION
    assert lock["terminal_elapsed_seconds"] == target.INITIAL_ELAPSED_SECONDS
    assert lock["accepted_segments"] == target.INITIAL_WIDE_ACCEPTED_SEGMENTS


def test_manifest_is_nonpropagating_retraction_only_diagnosis() -> None:
    contract = target._contract()
    assert contract["scope"]["span_ladder_seconds"] == [0.002, 0.001, 0.0005]
    assert contract["scope"]["maximum_retractions"] == 3
    assert contract["scope"]["exact_free_field_calls"] == 0
    assert contract["scope"]["new_trajectory"] is False
    assert contract["scope"]["accepted_history_mutation"] is False
    assert contract["gates"]["strict_status_requires_physical_closure_and_condition"]


def test_seed_is_exact_accepted_terminal_history() -> None:
    seed = target._seed()
    assert float(seed["elapsed_seconds"]) == target.INITIAL_ELAPSED_SECONDS
    assert int(seed["accepted_segments_total"]) == target.INITIAL_ACCEPTED_SEGMENTS
    assert float(seed["previous_span_seconds"]) == target.PREVIOUS_SEGMENT_SECONDS
    assert seed["current_coordinate470"].shape == (470,)
    assert seed["current_primitive_state"].shape == (112, 5)
    assert seed["current_metric_augmented560x560"].shape == (560, 560)
    assert np.all(np.isfinite(seed["current_coordinate_rate470_per_s"]))


def test_positive_outcome_authorizes_only_recovery_manifest() -> None:
    contract = target._contract()
    assert "recovery manifest" in contract["decision"]["positive_authorizes"]
    assert any("complete-cycle" in value for value in contract["forbidden"])
