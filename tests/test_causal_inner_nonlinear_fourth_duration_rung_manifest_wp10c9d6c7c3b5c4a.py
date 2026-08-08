from __future__ import annotations

import json

import numpy as np

import run_causal_inner_nonlinear_fourth_duration_rung_manifest_wp10c9d6c7c3b5c4a as c4a


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_targets_are_views_of_one_integer_master_table() -> None:
    manifest = _read(c4a.MANIFEST_PATH)
    targets = manifest["canonical_targets"]
    master = np.asarray(targets["master_microseconds"], dtype=np.int64)
    seconds = master.astype(np.float64) * 1.0e-6
    for name in ("main", "replay", "strict", "pilot"):
        indices = np.asarray(targets[f"{name}_indices"], dtype=int)
        actual = np.asarray(targets[f"{name}_seconds"], dtype=np.float64)
        assert np.array_equal(actual.view(np.uint64), seconds[indices].view(np.uint64))


def test_extraction_partition_semantics_are_frozen() -> None:
    manifest = _read(c4a.MANIFEST_PATH)
    contract = manifest["extraction_partition_contract"]
    assert contract["physical_radius_rg"] == c4a.SELECTED_EXTRACTION_RADIUS_RG
    assert tuple(contract["coarse_middle_fine_face_indices"]) == (2, 4, 8)
    assert contract["extraction_flux_is_not_pointwise_horizon_flux"]
    assert contract["excision_to_extraction_buffer_remains_inside_microdomain"]
    assert contract["raw_inner_face_rejection_preserved"]


def test_manifest_authorizes_only_cost_pilot() -> None:
    summary = _read(c4a.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["ten_ms_cost_pilot_authorized"]
    assert not summary["ten_ms_screen_propagation_authorized"]
    assert not summary["twenty_ms_completion_manifest_authorized"]
    assert not summary["twenty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_runtime_policy_is_soft_below_48_hours() -> None:
    manifest = _read(c4a.MANIFEST_PATH)
    pilot = manifest["pilot"]
    assert pilot["projected_wall_hours_at_or_below_24"] == "continue_automatically"
    assert pilot["projected_wall_hours_24_to_48"] == "continue_after_optimization_review"
    assert pilot["projected_wall_hours_above_48"] == "stop_and_optimize_before_full_screen"
    assert pilot["runtime_projection_is_not_a_physical_gate"]
