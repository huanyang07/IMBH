from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_nonlinear_final_three_grid_20ms_spatial_reanalysis_wp10c9d6c7c3b5c4e12 as c4e12


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_inherits_gates_and_uses_actual_fine_anchor():
    contract = c4e12._contract()
    assert contract["definitions_inherited_unchanged_from_c4e9"]
    assert not contract["propagation_executed"]
    assert contract["scope"]["fine_response"] == "continuous_nonlinear_generic_anchor"
    assert contract["spatial_gates"] == c4e12.c4e9.SPATIAL_GATES
    assert contract["uncertainty_gates"]["fine_nonlinear_surrogate_uncertainty"] == 0.0


def test_final_certificate_and_all_channels_pass():
    summary = _read(c4e12.SUMMARY_PATH)
    analysis = summary["analysis"]
    assert summary["passed"]
    assert summary["fine_twenty_ms_spatial_certificate_issued"]
    assert summary["state_twenty_ms_spatial_contract_certified"]
    assert summary["extraction_twenty_ms_spatial_contract_certified"]
    assert summary["reduced_architecture_manifest_authorized"]
    assert analysis["fine_response_is_actual_nonlinear_anchor"]
    assert analysis["fine_surrogate_uncertainty"] == 0.0
    for name in (
        "state",
        "instantaneous_extraction",
        "cumulative_extraction",
        "window_mean_extraction",
    ):
        assert analysis[name]["channel_certifying"]
        assert analysis[name]["raw_spatial_contract_passed"]
        assert analysis[name]["temporal_gate_passed"]


def test_orders_differences_and_directions_are_strong():
    analysis = _read(c4e12.SUMMARY_PATH)["analysis"]
    for name in (
        "state",
        "instantaneous_extraction",
        "cumulative_extraction",
        "window_mean_extraction",
    ):
        metric = analysis[name]
        assert metric["observed_rms_order"] >= 1.90
        assert metric["observed_maximum_order"] >= 1.90
        assert metric["minimum_significant_component_order"] >= 1.75
        assert metric["maximum_fine_normalized_difference"] < 2.0e-6
        assert metric["history_cosine"] > 0.99999
        assert metric["refinement_error_cosine"] > 0.995
        assert metric["temporal_uncertainty_fraction_of_middle_fine_difference"] < 0.10


def test_decisive_arrays_contain_actual_fine_responses():
    with np.load(c4e12.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        assert arrays["fine_nonlinear_state_response"].shape == (14, 64, 5)
        assert arrays["fine_nonlinear_extraction_response"].shape == (14, 13)
        assert arrays["fine_nonlinear_cumulative_extraction_response"].shape == (17, 13)
        assert arrays["fine_nonlinear_window_mean_extraction_response"].shape == (3, 13)
        assert np.array_equal(arrays["surrogate_uncertainties"], np.zeros(4))


def test_later_scientific_stops_remain_binding():
    summary = _read(c4e12.SUMMARY_PATH)
    assert not summary["fifty_ms_manifest_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["physical_failure_detected"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_canonical_hashes_close():
    for line in (c4e12.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(c4e12.CANONICAL_DIRECTORY / name) == expected
