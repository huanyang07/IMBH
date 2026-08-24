from __future__ import annotations

import numpy as np

import run_causal_inner_metric_chart_boundary_crossing_manifest_wp10c9d6c7c3b5c4f25fie as target


def test_parent_overlap_authorizes_only_boundary_manifest() -> None:
    lock = target._validate_parent(require_clean=False)
    assert lock["overlap_classification"] == target.parent.PASS_CLASSIFICATION
    assert target.parent.AUTHORIZED_NEXT == (
        "WP10c9d6c7c3b5c4f25fie_metric_chart_boundary_crossing_manifest"
    )


def test_seed_reproduces_rejected_boundary_target_without_propagating_it() -> None:
    seed = target._seed()
    assert target._relative(
        seed["candidate_target470"], seed["saved_boundary_target470"]
    ) <= target.MAXIMUM_SAVED_TARGET_RELATIVE_DEFECT
    assert float(seed["segment_seconds"]) == target.SEGMENT_SECONDS
    assert float(seed["previous_span_seconds"]) == target.SEGMENT_SECONDS


def test_contract_preserves_original_physics_and_limits_scope() -> None:
    contract = target._contract()
    assert contract["truth_system"]["physical_coordinate"] == "original q=C(u)"
    assert contract["chart_policy"][
        "all_original_physical_gates_except_raw_condition_remain_binding"
    ]
    assert contract["scope"]["new_exact_free_field_calls"] == 2
    assert contract["scope"]["new_accepted_segments_maximum"] == 1
    assert contract["history"]["blind_midpoint_required"]
    assert any(
        value.startswith("relax rank") for value in contract["forbidden"]
    )
