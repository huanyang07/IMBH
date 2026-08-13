from __future__ import annotations

import json

import run_causal_inner_nonlinear_middle_20ms_temporal_reference_manifest_wp10c9d6c7c3b5c4e5 as c4e5


def test_temporal_reference_contract_is_short_and_response_specific():
    contract = c4e5._manifest()
    assert contract["definitions_frozen_before_propagation"]
    assert contract["scope"]["start_microseconds"] == 16_000
    assert contract["scope"]["stop_microseconds"] == 16_400
    assert contract["scope"]["strict_target_microseconds"] == (
        16_000,
        16_100,
        16_200,
        16_300,
        16_400,
    )
    assert contract["temporal_uncertainty"]["safety_factor"] == 2.0
    assert (
        contract["temporal_uncertainty"][
            "maximum_fraction_of_spatial_difference"
        ]
        == 0.10
    )
    assert contract["method_gates"]["main_local_error_maximum"] == 2.5e-4
    assert contract["method_gates"]["strict_local_error_maximum"] == 3.125e-5


def test_temporal_reference_contract_keeps_later_work_blocked():
    contract = c4e5._manifest()
    assert not contract["decision"]["fine_propagation_directly_authorized"]
    assert contract["scope"]["raw_inner_face_is_not_a_slow_export"]
    assert "do_not_rerun_full_5_to_20ms_coarse_or_middle_histories" in contract[
        "hard_stops"
    ]


def test_canonical_manifest_summary_is_prospective():
    summary = json.loads(c4e5.SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["temporal_reference_shadow_authorized"]
    assert not summary["fine_twenty_ms_manifest_authorized"]
    assert not summary["fine_twenty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
