from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_hot_mode_arclength_event_diagnosis_wp10c9d6c7c3b5c4f25f1 as target


def test_five_accepted_windows_are_contiguous_and_locked() -> None:
    locked = target._validate_inputs(require_clean=False)
    assert len(locked["window_hashes"]) == 5
    assert [item["window_index"] for item in locked["window_summaries"]] == [1, 2, 3, 4, 5]


def test_arclength_phase_is_positive_and_well_conditioned() -> None:
    for directory in target._window_directories():
        record, arrays = target._window_diagnostic(directory)
        assert record["coordinate_arclength"] > 0.0
        assert record["phase_speed_ratio"] >= target.MINIMUM_WITHIN_WINDOW_PHASE_SPEED_RATIO
        assert record["chord_efficiency"] >= target.MINIMUM_CHORD_EFFICIENCY
        assert np.all(np.diff(arrays["phase_nodes"]) > 0.0)
        assert arrays["phase_nodes"][0] == 0.0
        assert arrays["phase_nodes"][-1] == 1.0


def test_four_node_holdout_supports_a_five_node_preflight() -> None:
    locked = target._validate_inputs(require_clean=False)
    metrics, _arrays = target._diagnose(locked)
    assert metrics["passed"]
    assert metrics["gates"]["four_node_direction_holdout"]
    assert metrics["gates"]["four_node_inverse_speed_holdout"]
    assert metrics["selected_architecture"]["next_truth_node_count_recommendation"] == 5
    assert metrics["selected_architecture"]["fixed_time_window_06_superseded_before_execution"]


def test_canonical_package_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(target.CANONICAL_DIRECTORY / "arclength_event_metrics.json")
    assert summary["passed"]
    assert summary["weighted_coordinate_arclength_selected"]
    assert not summary["fixed_time_window_06_authorized"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
    assert metrics["classification"] == target.CLASSIFICATION
