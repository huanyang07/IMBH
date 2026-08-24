from __future__ import annotations

import run_causal_inner_metric_chart_short_suffix_manifest_wp10c9d6c7c3b5c4f25fig as target


def test_boundary_crossing_authorizes_suffix_only() -> None:
    lock = target._validate_parent(require_clean=False)
    assert lock["classification"] == target.parent.PASS_CLASSIFICATION
    assert target.parent.AUTHORIZED_NEXT == (
        "WP10c9d6c7c3b5c4f25fig_metric_chart_short_suffix_manifest"
    )


def test_suffix_scope_is_four_fixed_segments_with_one_blind_midpoint() -> None:
    contract = target._contract()
    assert contract["suffix"]["new_segments"] == 4
    assert contract["suffix"]["tentative_segment_numbers"] == (73, 74, 75, 76)
    assert contract["suffix"]["blind_midpoint_segment"] == 76
    assert contract["scope"]["new_exact_free_field_calls"] == 5
    assert contract["scope"]["new_physical_time_seconds"] == 1.0e-3


def test_seed_is_an_arbitrary_bdf_free_continuation_checkpoint() -> None:
    seed = target._seed()
    assert float(seed["elapsed_seconds"]) == target.INITIAL_ELAPSED_SECONDS
    assert int(seed["accepted_segments_total"]) == 72
    assert seed["current_metric_augmented560x560"].shape == (560, 560)
    assert seed["current_gauge_basis560x90"].shape == (560, 90)
