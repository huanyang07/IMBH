from __future__ import annotations

import hashlib
import json

import run_causal_inner_complete_resolved_closure_manifest_wp10c9d6c7c3b5c4f25r as f25r


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_parent_selects_R34_M112_and_authorizes_prototype_manifest():
    summary, metrics, hashes = f25r._validate_parent()
    assert summary["passed"]
    assert summary["selected_common_rank"] == 34
    assert summary["selected_memory_order"] == 112
    assert metrics["selected"]["joint_passed"]
    assert "decisive_basis.npz" in hashes


def test_complete_closure_dimensions_obey_R320():
    contract = f25r._contract()
    closure = contract["complete_closure"]
    assert closure["resolved_dimension"] == 196
    assert closure["stable_dimension"] == 364
    assert closure["candidate_orders"] == [112, 120, 124]
    assert closure["online_dimensions"] == [308, 316, 320]
    assert all(value <= 320 for value in closure["online_dimensions"])


def test_contract_binds_resolved_feedback_and_face_flux():
    contract = f25r._contract()
    gates = contract["complete_closure"][
        "pass_requires_at_both_anchors_on_training_and_heldout"
    ]
    assert set(gates) == {"resolved_self_energy", "conservative_face_flux"}
    blocks = contract["complete_closure"]["exact_blocks"]
    assert blocks["resolved_memory_observation"] == "R_A_S"
    assert blocks["face_memory_observation"] == "O_face_S"


def test_contract_forbids_truth_and_online_execution():
    contract = f25r._contract()
    budget = contract["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0
    assert not contract["claim_boundary"]["online_integrator_implementation_authorized"]
    assert not contract["claim_boundary"]["physical_failure_can_be_declared"]


def test_canonical_manifest_when_available():
    summary_path = f25r.ARTIFACT_DIRECTORY / "summary.json"
    if not summary_path.exists():
        return
    summary = _read(summary_path)
    assert summary["passed"]
    assert summary["authorized_next"] == "WP10c9d6c7c3b5c4f25s"
    for line in (f25r.ARTIFACT_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25r.ARTIFACT_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
