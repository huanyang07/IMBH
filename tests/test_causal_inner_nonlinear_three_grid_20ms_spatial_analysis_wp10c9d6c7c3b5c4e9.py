from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_nonlinear_three_grid_20ms_spatial_analysis_wp10c9d6c7c3b5c4e9 as c4e9


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_analysis_contract_preserves_spatial_and_uncertainty_gates():
    contract = c4e9._contract()
    assert contract["definitions_frozen_before_analysis"]
    assert not contract["propagation_executed"]
    assert contract["spatial_gates"]["minimum_rms_order"] == 0.75
    assert contract["spatial_gates"]["maximum_fine_normalized_difference"] == 0.05
    assert contract["uncertainty_gates"][
        "maximum_temporal_fraction_of_middle_fine_difference"
    ] == 0.10
    assert contract["uncertainty_gates"][
        "maximum_surrogate_fraction_of_middle_fine_difference"
    ] == 0.10
    assert contract["uncertainty_construction"][
        "surrogate_gate_is_binding_even_when_spatial_difference_is_unobservable"
    ]


def test_canonical_result_certifies_state_but_requires_fine_anchor_for_extraction():
    summary = _read(c4e9.SUMMARY_PATH)
    analysis = summary["analysis"]
    assert summary["passed"]
    assert summary["analysis_completed"]
    assert summary["state_twenty_ms_spatial_contract_certified"]
    assert not summary["fine_twenty_ms_spatial_certificate_issued"]
    assert summary["full_fine_generic_anchor_required"]
    assert summary["fine_generic_anchor_manifest_authorized"]
    assert not summary["full_fine_generic_anchor_authorized"]
    assert not summary["physical_failure_detected"]
    assert analysis["aggregate_extraction_order_difference_direction_gates_passed"]
    assert not analysis["extraction_component_and_surrogate_contract_certified"]


def test_state_and_aggregate_extraction_metrics_are_second_order_and_aligned():
    analysis = _read(c4e9.SUMMARY_PATH)["analysis"]
    state = analysis["state"]
    assert state["observed_rms_order"] > 1.9
    assert state["minimum_significant_component_order"] > 1.9
    assert state["refinement_error_cosine"] > 0.99
    assert state["surrogate_gate_passed"]
    for name in (
        "instantaneous_extraction",
        "cumulative_extraction",
        "window_mean_extraction",
    ):
        metric = analysis[name]
        assert metric["observed_rms_order"] > 1.9
        assert metric["observed_maximum_order"] > 1.9
        assert metric["refinement_error_cosine"] > 0.99
        assert metric["maximum_fine_normalized_difference"] < 2.0e-7
        assert not metric["surrogate_gate_passed"]


def test_decisive_arrays_have_one_common_parent_and_all_three_grids():
    with np.load(c4e9.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        assert arrays["coarse_state_response"].shape == (14, 64, 5)
        assert arrays["middle_state_response"].shape == (14, 64, 5)
        assert arrays["fine_tangent_state_response"].shape == (14, 64, 5)
        assert arrays["coarse_extraction_response"].shape == (14, 13)
        assert arrays["fine_tangent_extraction_response"].shape == (14, 13)
        assert arrays["fine_tangent_cumulative_extraction_response"].shape == (
            17,
            13,
        )
        assert arrays["fine_tangent_window_mean_extraction_response"].shape == (
            3,
            13,
        )


def test_canonical_hashes_close():
    for line in (c4e9.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(c4e9.CANONICAL_DIRECTORY / name) == expected
