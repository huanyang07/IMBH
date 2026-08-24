from __future__ import annotations

import numpy as np

import run_causal_inner_adaptive_metric_chart_radius_recovery_manifest_wp10c9d6c7c3b5c4f25fim as target


def test_parent_identifies_one_millisecond_local_radius() -> None:
    lock = target._validate_parent(require_clean=False)
    assert lock["classification"] == target.parent.PASS_CLASSIFICATION
    assert target.SEGMENT_SECONDS == 1.0e-3


def test_recovery_is_one_blind_audited_segment() -> None:
    contract = target._contract()
    assert contract["history"]["tentative_segment_number"] == 92
    assert contract["history"]["blind_midpoint_required"] is True
    assert contract["scope"]["new_exact_free_field_calls"] == 2
    assert contract["scope"]["new_accepted_segments_maximum"] == 1
    assert contract["history"]["failed_2_ms_candidate_never_propagated"]


def test_seed_matches_diagnosed_target_and_accepted_history() -> None:
    seed = target._seed()
    assert float(seed["elapsed_seconds"]) == target.PARENT_ELAPSED_SECONDS
    assert int(seed["accepted_segments_total"]) == 91
    assert float(seed["segment_seconds"]) == target.SEGMENT_SECONDS
    assert seed["candidate_target470"].shape == (470,)
    np.testing.assert_allclose(
        seed["candidate_target470"], seed["diagnosed_target470"], rtol=0.0, atol=0.0
    )


def test_pass_authorizes_only_adaptive_continuation_manifest() -> None:
    contract = target._contract()
    assert "adaptive continuation manifest" in contract["decision"]["pass_authorizes"]
    assert any("complete-cycle" in value for value in contract["forbidden"])
