from __future__ import annotations

import json

import run_causal_inner_nonlinear_twenty_ms_checkpoint_assessment_manifest_wp10c9d6c7c3b5c4d as c4d


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_assessment_is_evidence_only_and_prospective() -> None:
    manifest = _read(c4d.MANIFEST_PATH)
    assert manifest["definitions_only"]
    assert not manifest["propagation_executed"]
    assert manifest["assessment_metrics"]["ten_to_twenty_boundary_state_bitwise"]
    assert manifest["assessment_metrics"][
        "endpoint_response_ratio_twenty_over_ten"
    ]


def test_interpretation_cannot_claim_attraction_or_reduction() -> None:
    manifest = _read(c4d.MANIFEST_PATH)
    contract = manifest["interpretation_contract"]
    assert contract["amplitude_trend_is_not_an_attractor_classification"]
    assert contract["one_perturbation_is_not_multiple_equal_Q_fast_lifts"]
    assert contract["no_fixed_Q_or_slow_reduction_claim_can_be_authorized"]


def test_manifest_authorizes_only_checkpoint_assessment() -> None:
    summary = _read(c4d.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["twenty_ms_checkpoint_assessment_authorized"]
    assert not summary["twenty_ms_spatial_checkpoint_manifest_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
