from __future__ import annotations

import json

import numpy as np

import run_causal_inner_nonlinear_middle_20ms_temporal_reference_shadow_wp10c9d6c7c3b5c4e6 as c4e6


def test_stage_order_is_fail_fast_and_contains_only_short_shadows():
    assert c4e6.STAGE_ORDER == (
        "coarse_base_main",
        "coarse_perturbed_main",
        "coarse_base_strict",
        "coarse_perturbed_strict",
        "middle_base_strict",
        "middle_anchor_strict",
    )
    assert c4e6.c4e5.STOP_MICROSECONDS - c4e6.c4e5.START_MICROSECONDS == 400


def test_window_metrics_measure_response_specific_state_and_extraction_error():
    main_times = np.asarray((0.0160, 0.0164))
    strict_times = np.asarray((0.0160, 0.0161, 0.0162, 0.0163, 0.0164))
    main_state = np.asarray((((0.0,),), ((2.0,),)))
    strict_state = np.asarray(
        (((0.0,),), ((0.5,),), ((1.0,),), ((1.5,),), ((2.1,),))
    )
    main_extraction = np.asarray(((0.0,), (2.0,)))
    strict_extraction = np.asarray(((0.0,), (0.5,), (1.0,), (1.5,), (2.1,)))
    metrics = c4e6._window_metrics(
        main_times,
        main_state,
        main_extraction,
        strict_times,
        strict_state,
        strict_extraction,
        np.ones(1),
        np.ones(1),
    )
    assert np.isclose(metrics["state"], 0.1)
    assert np.isclose(metrics["instantaneous_extraction"], 0.1)
    assert metrics["cumulative_extraction"] > 0.0
    assert metrics["window_mean_extraction"] > 0.0


def test_canonical_temporal_reference_decision_is_self_consistent():
    summary = json.loads(c4e6.SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["passed"]
    assert summary["analysis_completed"]
    assert (
        summary["fine_twenty_ms_manifest_authorized"]
        == summary["temporal_reference_hardened"]
    )
    assert not summary["fine_twenty_ms_propagation_authorized"]
    assert not summary["physical_failure_detected"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    for item in summary["analysis"]["observables"].values():
        assert item["passed"] == (item["temporal_to_spatial_fraction"] <= 0.10)


def test_canonical_payload_hashes_verify():
    expected = {}
    for line in (c4e6.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", maxsplit=1)
        expected[name] = digest
    assert expected
    assert all(
        c4e6._sha256(c4e6.CANONICAL_DIRECTORY / name) == digest
        for name, digest in expected.items()
    )
