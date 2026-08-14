import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_six_mode_dynamic_coordinate_replay_wp10c9d6c7c3b5c4f17"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f17_middle_completed_and_fine_was_blocked():
    summary = _read("summary.json")
    assert not summary["passed"]
    assert summary["middle_completed"]
    assert not summary["fine_executed"]
    assert summary["middle"]["steps"] == 39
    assert not summary["middle"][
        "passed_method_and_single_layout_coordinate_gates"
    ]
    assert summary["classification"] == (
        "face36_six_mode_middle_dynamic_coordinate_preflight_rejected_"
        "fine_blocked_numerical_audit_recovery_manifest_authorized"
    )


def test_c4f17_failure_is_exactly_the_two_frozen_numerical_audits():
    summary = _read("summary.json")
    config = _read("config.json")
    gates = config["prospective_gates"]
    middle = summary["middle"]
    assert summary["failed_gates"] == [
        "dual_normalized_slow_annihilation",
        "face36_output_map",
    ]
    assert middle["maximum_JVP_defect"] <= gates[
        "maximum_step_matrix_JVP_relative_defect"
    ]
    assert middle["maximum_linear_solve_defect"] <= gates[
        "maximum_block_linear_solve_relative_defect"
    ]
    assert middle["maximum_Q3_leakage"] <= gates["maximum_Q3_leakage"]
    assert middle["maximum_incoming_characteristics"] == 0
    assert middle["dual_normalized_slow_annihilation_defect"] > gates[
        "maximum_normalized_slow_lift_annihilation_defect"
    ]
    assert middle["maximum_face36_output_map_defect"] > gates[
        "maximum_face36_output_map_relative_defect"
    ]


def test_c4f17_commits_complete_middle_histories():
    with np.load(ARTIFACT / "decisive_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["times"].shape == (40,)
        assert arrays["middle_state_directions"].shape[:2] == (40, 6)
        assert arrays["middle_face36_outputs"].shape == (40, 6, 3)
        assert arrays["middle_amplitude_transitions"].shape == (40, 6, 6)
        assert arrays["middle_Q3_leakage"].shape == (40, 6)
        assert np.count_nonzero(np.isfinite(arrays["middle_JVP_defects"])) == 4
        assert np.count_nonzero(
            np.isfinite(arrays["middle_face36_output_map_defects"])
        ) == 5


def test_c4f18_recovery_is_analysis_only_and_does_not_relax_gates():
    manifest = _read("recovery_manifest.json")
    assert manifest["definitions_only"]
    assert manifest["uses_saved_c4f17_middle_state_direction_history"]
    assert not manifest["reruns_middle_propagation"]
    assert not manifest["runs_fine_propagation"]
    assert manifest["stable_dual_audit"]["tolerance_relaxation_forbidden"]
    assert manifest["face36_directional_JVP_plateau"][
        "tolerance_relaxation_forbidden"
    ]
    assert manifest["stable_dual_audit"][
        "maximum_normalized_slow_lift_annihilation_defect"
    ] == 1.0e-10
    assert manifest["face36_directional_JVP_plateau"][
        "maximum_relative_defect"
    ] == 1.0e-8


def test_c4f17_hash_manifest_and_hard_stops_are_preserved():
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
    summary = _read("summary.json")
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["nonlinear_retained_mode_pilot_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert not summary["physical_failure_detected"]
    assert summary["guard_complement_retained"]
    assert summary["raw_face48_export_rejection_preserved"]
