from __future__ import annotations

import json

import numpy as np

import run_causal_inner_nonlinear_optimized_middle_20ms_completion_wp10c9d6c7c3b5c4e3 as c4e3


def test_optimized_completion_passes_and_stops_before_fine() -> None:
    summary = json.loads(c4e3.SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["passed"]
    assert summary["base"]["passed"]
    assert summary["tangent"]["passed"]
    assert summary["anchor"]["passed"]
    assert summary["extraction_tangent"]["passed"]
    assert summary["coarse_middle_twenty_ms_checkpoint_analysis_authorized"]
    assert not summary["fine_twenty_ms_propagation_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_temporal_and_replay_contracts_pass() -> None:
    summary = json.loads(c4e3.SUMMARY_PATH.read_text(encoding="utf-8"))
    base = summary["base"]
    assert base["selected_maximum_timestep_seconds"] in (4.0e-4, 8.0e-4, 1.2e-3)
    assert base["maximum_local_error_bound"] <= 2.5e-4
    assert base["sum_local_error_bounds"] <= 5.0e-3
    for replay in summary["serialized_replays"].values():
        assert replay["checkpoint_roundtrip_bitwise"]
        assert replay["last_step_replay_bitwise"]
        assert replay["maximum_scaled_residual"] <= 1.0e-10


def test_extraction_tangent_and_target_identities_pass() -> None:
    summary = json.loads(c4e3.SUMMARY_PATH.read_text(encoding="utf-8"))
    extraction = summary["extraction_tangent"]
    assert extraction["instantaneous"][
        "discrepancy_fraction_of_observable_response"
    ] <= 0.01
    assert extraction["cumulative"][
        "discrepancy_fraction_of_observable_response"
    ] <= 0.01
    assert extraction["window_mean"][
        "discrepancy_fraction_of_observable_response"
    ] <= 0.01
    assert extraction["maximum_step_sensitivity_fraction_of_response"] <= 1.0e-4
    with np.load(c4e3.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        ids = tuple(np.rint(arrays["base__output_times"] * 1.0e6).astype(int))
        assert ids == c4e3.OUTPUT_TARGET_MICROSECONDS
        assert arrays["extraction__tangent_directions"].shape[1] == len(c4e3.PROFILES)
