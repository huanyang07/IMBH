from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_nonlinear_coarse_middle_20ms_checkpoint_analysis_wp10c9d6c7c3b5c4e4 as c4e4


def _summary() -> dict:
    return json.loads(c4e4.SUMMARY_PATH.read_text(encoding="utf-8"))


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_analysis_is_evidence_only_and_stops_before_fine() -> None:
    summary = _summary()
    assert summary["passed"]
    assert summary["analysis_completed"]
    assert not summary["twenty_ms_spatial_checkpoint_certified"]
    assert not summary["fine_twenty_ms_propagation_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert not summary["physical_failure_detected"]


def test_checkpoint_decision_obeys_frozen_gates() -> None:
    summary = _summary()
    analysis = summary["analysis"]
    contract = analysis["analysis_contract"]["screening_gates"]
    expected = True
    for name in (
        "state",
        "instantaneous_extraction",
        "cumulative_extraction",
        "window_mean_extraction",
    ):
        item = analysis[name]
        expected &= (
            item["maximum_normalized_difference"]
            <= contract["maximum_normalized_state_or_extraction_difference"]
            and item["cross_grid_response_history_cosine"]
            >= contract["minimum_cross_grid_response_history_cosine"]
            and item["temporal_uncertainty_fraction_of_spatial_difference"]
            <= contract[
                "maximum_temporal_uncertainty_fraction_of_spatial_difference"
            ]
        )
    assert analysis["checkpoint_screen_passed"] is bool(expected)
    assert summary["coarse_middle_twenty_ms_checkpoint_screen_passed"] is bool(
        expected
    )
    assert summary["fine_completion_manifest_authorized"] is bool(expected)


def test_common_times_and_physical_ledgers_are_exact() -> None:
    summary = _summary()
    analysis = summary["analysis"]
    with np.load(c4e4.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        common = arrays["common_times_seconds"]
        quadrature = arrays["quadrature_times_seconds"]
        assert common[0] == 0.005
        assert common[-1] == 0.020
        assert quadrature[0] == 0.005
        assert quadrature[-1] == 0.020
        assert arrays["coarse_state_response"].shape[1:] == (64, 5)
        assert arrays["middle_state_response"].shape == arrays[
            "coarse_state_response"
        ].shape
        assert arrays["coarse_extraction_response"].shape[1] == 13
    assert analysis["maximum_early_extraction_identity_defect"] <= 1.0e-12
    assert analysis["maximum_shared_conservative_face_defect"] <= 1.0e-12
    assert analysis["maximum_local_block_ledger_defect"] <= 1.0e-11
    assert analysis["maximum_source_double_count_defect"] <= 1.0e-12
    assert analysis["maximum_incoming_excision_characteristics"] == 0


def test_canonical_hashes_and_implementation_identity_match() -> None:
    recorded = {}
    for line in (c4e4.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    for name, digest in recorded.items():
        assert _sha256(c4e4.CANONICAL_DIRECTORY / name) == digest
    provenance = json.loads(c4e4.PROVENANCE_PATH.read_text(encoding="utf-8"))
    for relative, digest in provenance["implementation_source_hashes"].items():
        assert _sha256(c4e4.ROOT / relative) == digest
