import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_q_plus_a_reaction_coordinate_preflight_wp10c9d6c7c3b5c4f15"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f15_scoped_reaction_pass_and_two_mode_rejection():
    summary = _read("summary.json")
    assert summary["audit_completed"]
    assert not summary["passed"]
    assert summary["reaction_map_preflight_passed"]
    assert summary["endpoint_coordinate_preflight_passed"]
    assert summary["two_mode_aggregate_output_gate_passed"]
    assert not summary["two_mode_significant_direction_gate_passed"]
    assert not summary["dynamic_state_amplitude_history_gate_evaluated"]
    assert not summary["retained_coordinate_preflight_passed"]
    assert summary["minimum_passing_output_oriented_dimension"] == 6
    assert summary["classification"] == (
        "face36_two_mode_coordinate_preflight_rejected_"
        "six_mode_manifest_authorized"
    )


def test_c4f15_reaction_and_endpoint_coordinate_gates_close():
    summary = _read("summary.json")
    config = _read("config.json")
    gates = config["prospective_gates"]
    assert (
        summary["maximum_DQ_M_inverse_BQ_identity_defect"]
        <= gates["maximum_DQ_M_inverse_BQ_identity_defect"]
    )
    assert (
        summary["maximum_KKT_linear_solve_relative_defect"]
        <= gates["maximum_KKT_linear_solve_relative_defect"]
    )
    assert (
        summary["maximum_reaction_ledger_relative_defect"]
        <= gates["maximum_reaction_ledger_relative_defect"]
    )
    assert (
        summary["maximum_KKT_Schur_condition_number"]
        <= gates["maximum_KKT_Schur_condition_number"]
    )
    assert (
        summary["maximum_state_lift_Q3_defect"]
        <= gates["maximum_state_lift_Q3_defect"]
    )
    assert (
        summary["maximum_dual_biorthogonality_defect"]
        <= gates["maximum_dual_biorthogonality_defect"]
    )
    for label in ("middle", "fine"):
        assert len(summary["reactions"][label]) == 4
        assert max(
            item["reaction_support_relative_defect"]
            for item in summary["reactions"][label]
        ) == 0.0
        coordinate = summary["coordinates_at_5ms"][label]
        assert coordinate["dual_normalized_slow_lift_annihilation_defect"] <= 1.0e-10


def test_c4f15_output_reconstruction_enforces_both_gates():
    summary = _read("summary.json")
    config = _read("config.json")
    gates = config["prospective_gates"]
    two = summary["two_mode_output_reconstruction"]
    six = summary["minimum_passing_dimension_output_reconstruction"]
    assert (
        two["maximum_output_weighted_RMS_error"]
        <= gates["maximum_two_mode_face36_output_weighted_RMS_error"]
    )
    assert (
        two["maximum_significant_direction_error"]
        > gates["maximum_two_mode_face36_significant_direction_error"]
    )
    assert (
        six["maximum_output_weighted_RMS_error"]
        <= gates["maximum_two_mode_face36_output_weighted_RMS_error"]
    )
    assert (
        six["maximum_significant_direction_error"]
        <= gates["maximum_two_mode_face36_significant_direction_error"]
    )


def test_c4f15_arrays_and_hash_manifest_are_self_consistent():
    with np.load(ARTIFACT / "decisive_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["two_mode_consensus_direction_coefficients"].shape == (29, 2)
        assert arrays["minimum_passing_consensus_direction_coefficients"].shape == (29, 6)
        assert arrays["middle__two_mode_state_lifts_scaled"].shape[1] == 2
        assert arrays["fine__two_mode_descriptor_dual"].shape[0] == 2
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest


def test_c4f15_preserves_all_hard_stops():
    summary = _read("summary.json")
    assert not summary["new_trajectory_executed"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["nonlinear_retained_mode_pilot_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert not summary["physical_failure_detected"]
    assert summary["raw_face48_export_rejection_preserved"]
    assert summary["guard_complement_retained"]
