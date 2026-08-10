from __future__ import annotations

import json

import numpy as np

import run_causal_inner_nonlinear_twenty_ms_checkpoint_assessment_wp10c9d6c7c3b5c4d1 as c4d1


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_checkpoint_assessment_passes_exact_boundary_and_response_gates() -> None:
    assessment = _read(c4d1.ASSESSMENT_PATH)
    assert assessment["passed"]
    assert assessment["boundary"]["passed"]
    assert assessment["boundary"]["base"]["output_states_bitwise"]
    assert assessment["boundary"]["perturbed"]["output_states_bitwise"]
    assert all(assessment["gates"].values())


def test_checkpoint_response_metrics_match_canonical_arrays() -> None:
    summary = _read(c4d1.SUMMARY_PATH)
    with np.load(c4d1.DECISIVE_ARRAYS) as arrays:
        assert arrays["output_times"][0] == 0.010
        assert arrays["output_times"][-1] == 0.020
        assert np.isclose(
            summary["maximum_scaled_state_response"],
            np.max(arrays["scaled_state_max"]),
        )
        assert np.isclose(
            summary["maximum_scaled_extraction_partition_response"],
            np.max(arrays["scaled_extraction_partition_max"]),
        )
        assert np.isclose(
            summary["state_rms_ratio_twenty_over_ten"],
            arrays["scaled_state_rms"][-1] / arrays["scaled_state_rms"][0],
        )


def test_assessment_does_not_overclaim_reduction() -> None:
    assessment = _read(c4d1.ASSESSMENT_PATH)
    summary = _read(c4d1.SUMMARY_PATH)
    interpretation = assessment["interpretation"]
    assert not interpretation["attraction_or_memory_loss_demonstrated"]
    assert not interpretation["multiple_equal_Q_lifts_tested"]
    assert not summary["physical_failure_detected"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert summary["twenty_ms_spatial_checkpoint_manifest_authorized"]
    assert summary["authorized_next"].endswith("twenty_ms_spatial_checkpoint_manifest")
