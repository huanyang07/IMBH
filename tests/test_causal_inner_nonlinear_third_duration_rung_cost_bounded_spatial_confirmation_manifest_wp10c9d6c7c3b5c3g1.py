from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_third_duration_rung_cost_bounded_spatial_confirmation_manifest_wp10c9d6c7c3b5c3g1 as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_manifest_preserves_c3g_and_freezes_only_calibration() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    summary = _read(runner.SUMMARY_PATH)
    assert manifest["definitions_only"] is True
    assert manifest["propagation_executed"] is False
    assert manifest["c3h_bruteforce_execution"]["completed_scientific_evidence"] is False
    assert manifest["c3h_bruteforce_execution"]["prospective_route_superseded"] is True
    assert summary["parent_classification_preserved"] == (
        "third_duration_rung_spatial_confirmation_manifest_frozen_"
        "middle_fine_generic_propagation_authorized"
    )
    assert summary["discrete_BDF_tangent_calibration_authorized"] is True
    assert summary["middle_cost_bounded_propagation_authorized"] is False
    assert summary["fine_cost_bounded_propagation_authorized"] is False


def test_complete_discrete_history_and_surrogate_gates_are_binding() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    calibration = manifest["phase_h1_discrete_tangent_calibration"]
    required = manifest["scientific_assessment"]["required_refinements"]
    gates = calibration["gates"]
    assert calibration["new_long_trajectory_required"] is False
    assert tuple(calibration["tangent_history_state"]) == (
        "primitive_increment",
        "mapped_storage_path_increment",
        "responsive_height_storage_path_increment",
    )
    assert "linearize_mapped_and_responsive_height_history_actions" in required
    assert gates["maximum_scaled_state_response_discrepancy"] == 5.0e-3
    assert gates["maximum_scaled_Tier_I_response_discrepancy"] == 5.0e-3
    assert gates["surrogate_fraction_of_fine_difference_acceptance_budget"] == 0.10
    assert manifest["scientific_assessment"][
        "frozen_continuous_generator_is_not_an_acceptable_substitute"
    ] is True


def test_fine_estimator_is_triage_and_cost_is_capped() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    fine = manifest["phase_h3_fine_confirmation"]
    final = manifest["binding_final_certificate"]
    cost = manifest["cost_contract"]
    assert fine["fine_decision_gates"][
        "defect_estimator_is_triage_not_final_certificate"
    ] is True
    assert fine["minimum_binding_execution"] == (
        "one_fine_nonlinear_base_plus_block_discrete_BDF_tangent"
    )
    assert final["no_spatial_certificate_from_defect_estimator_alone"] is True
    assert final["surrogate_error_is_added_to_not_subtracted_from_error_budget"] is True
    assert cost["maximum_total_projected_new_nonlinear_wall_hours_before_execution"] == 24.0
    assert cost["maximum_single_unattended_stage_wall_hours"] == 12.0


def test_downstream_work_remains_blocked() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["third_duration_rung_spatial_convergence_certified"] is False
    assert summary["fourth_duration_rung_manifest_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((runner.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest() == digest
