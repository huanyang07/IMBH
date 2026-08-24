from __future__ import annotations

import numpy as np

import run_causal_inner_adaptive_metric_chart_cycle_readiness_execution_wp10c9d6c7c3b5c4f25fir as target


def test_manifest_authorizes_only_bounded_readiness_execution() -> None:
    lock = target._validate_manifest(require_clean=False)
    assert lock["summary"]["classification"] == target.manifest.CLASSIFICATION
    assert lock["contract"]["scope"]["maximum_accepted_segments"] == 32
    assert not lock["summary"]["complete_cycle_execution_authorized"]


def test_initial_progress_maps_terminal_seed_without_loss() -> None:
    seed = target._seed()
    progress = target._initial_progress()
    np.testing.assert_array_equal(
        progress["current_coordinate"], seed["current_coordinate470"]
    )
    np.testing.assert_array_equal(
        progress["metric_augmented"], seed["metric_augmented560x560"]
    )
    assert progress["accepted_segments_total"] == 100
    assert progress["accepted_since_growth"] == 2


def test_engine_context_is_isolated() -> None:
    original_manifest = target.engine.manifest
    original_scratch = target.engine.SCRATCH_DIRECTORY
    with target._engine_context():
        assert target.engine.manifest is target.manifest
        assert target.engine.SCRATCH_DIRECTORY == target.SCRATCH_DIRECTORY
        assert target.engine._initial_progress is target._initial_progress
    assert target.engine.manifest is original_manifest
    assert target.engine.SCRATCH_DIRECTORY == original_scratch


def test_geometry_classification_detects_turn_without_cycle() -> None:
    lock = target._validate_manifest(require_clean=False)
    seed = target._seed()
    normal = seed["section_normal470"]
    start = seed["start_coordinate470"]
    coordinate = seed["current_coordinate470"] + 1.0e-3 * normal
    rate = normal.copy()
    rate *= -1.0 / np.linalg.norm(rate)
    metrics = {
        "passed": True,
        "classification": target.OPEN_CLASSIFICATION,
        "authorized_next": target.OPEN_AUTHORIZED_NEXT,
        "gate_values": {
            "terminal_elapsed_seconds": target.manifest.INITIAL_ELAPSED_SECONDS
            + 5.0e-4
        },
    }
    arrays = {
        "accepted_endpoint_coordinates470": coordinate[None, :],
        "accepted_endpoint_coordinate_rates470_per_s": rate[None, :],
        "accepted_segment_seconds": np.asarray([5.0e-4]),
        "section_normal470": normal,
        "start_coordinate470": start,
    }
    classified, output = target._classify_geometry(metrics, arrays, lock)
    assert classified["classification"] == target.TURN_CLASSIFICATION
    assert classified["gate_values"]["section_turning_point_observed"]
    assert not classified["gate_values"]["cycle_observed"]
    np.testing.assert_array_equal(output["section_turn_indices"], [0])
