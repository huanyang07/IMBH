from __future__ import annotations

import json

import run_causal_inner_nonlinear_twenty_ms_spatial_checkpoint_manifest_wp10c9d6c7c3b5c4e as c4e


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_spatial_manifest_is_middle_first_and_definitions_only() -> None:
    manifest = _read(c4e.MANIFEST_PATH)
    assert manifest["definitions_only"]
    assert not manifest["propagation_executed"]
    stages = manifest["execution_stages"]
    assert stages[0]["name"] == "middle_5_to_6ms_cost_pilot"
    assert stages[0]["authorized"]
    assert not stages[1]["authorized"]
    assert not stages[3]["authorized"]


def test_binding_export_is_certified_extraction_partition() -> None:
    manifest = _read(c4e.MANIFEST_PATH)
    scope = manifest["scientific_scope"]
    assert scope["binding_extraction_face_indices"] == [2, 4, 8]
    assert scope["binding_extraction_radius_rg"] == c4e.EXTRACTION_RADIUS_RG
    assert scope["raw_inner_face_is_not_a_binding_slow_export"]


def test_cost_policy_does_not_weaken_scientific_gates() -> None:
    manifest = _read(c4e.MANIFEST_PATH)
    assert manifest["resource_policy"]["twenty_four_hours_is_soft_not_binding"]
    assert manifest["method_gates"]["maximum_scaled_nonlinear_residual"] == 1.0e-10
    assert manifest["spatial_certificate_gates"]["minimum_state_rms_order"] == 0.75
    assert manifest["spatial_certificate_gates"][
        "maximum_temporal_uncertainty_fraction_of_spatial_difference"
    ] == 0.10


def test_manifest_authorizes_only_middle_pilot() -> None:
    summary = _read(c4e.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["middle_six_ms_cost_pilot_authorized"]
    assert not summary["middle_twenty_ms_completion_authorized"]
    assert not summary["fine_twenty_ms_propagation_authorized"]
    assert not summary["twenty_ms_spatial_checkpoint_certified"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
