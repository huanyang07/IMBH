import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_six_mode_coordinate_manifest_wp10c9d6c7c3b5c4f16"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f16_freezes_scoped_six_mode_dynamic_preflight():
    summary = _read("summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert not summary["trajectory_executed"]
    assert summary["mode_dimension"] == 6
    assert summary["six_mode_output_closure_passed"]
    assert not summary["six_mode_dynamic_coordinate_certified"]
    assert summary["weak_enrichment_individual_mode_identity_rejected"]
    assert summary["dynamic_coordinate_preflight_authorized"]


def test_c4f16_preserves_leading_block_and_exposes_weak_block():
    summary = _read("summary.json")
    metrics = summary["basis_metrics"]
    assert metrics["minimum_six_mode_form_capture"] >= 0.99
    assert metrics["sigma6_over_sigma7_gap"] >= 5.0
    assert metrics["minimum_middle_fine_leading_block_cosine"] >= 0.95
    assert metrics["minimum_middle_fine_six_mode_cosine"] < 0.90
    manifest = _read("dynamic_coordinate_manifest.json")
    architecture = manifest["coordinate_architecture"]
    assert architecture["individual_mode_matching_for_dimensions_3_to_6_forbidden"]
    assert architecture["Procrustes_or_projector_comparison_required"]
    assert architecture["weak_enrichment_block"]["dimensions"] == [2, 6]


def test_c4f16_cost_and_fail_fast_order_are_frozen():
    manifest = _read("dynamic_coordinate_manifest.json")
    preflight = manifest["authorized_dynamic_coordinate_preflight"]
    assert preflight["directions"] == 6
    assert preflight["run_middle_first"]
    assert preflight["run_fine_only_after_middle_method_and_coordinate_gates_pass"]
    assert preflight["save_state_direction_history_at_all_committed_outputs"]
    cost = manifest["cost_contract"]
    assert cost["one_factorization_six_RHS_per_step"]
    assert cost["no_repeated_29_direction_propagation"]
    assert cost["expected_total_wall_hours"] == [3.5, 5.0]


def test_c4f16_basis_and_hashes_are_self_consistent():
    with np.load(ARTIFACT / "six_mode_basis.npz", allow_pickle=False) as arrays:
        basis = arrays["six_mode_consensus_direction_coefficients"]
        assert basis.shape == (29, 6)
        np.testing.assert_allclose(basis.T @ basis, np.eye(6), atol=1.0e-12)
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest


def test_c4f16_keeps_reduction_and_nonlinear_work_blocked():
    summary = _read("summary.json")
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["nonlinear_retained_mode_pilot_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert not summary["physical_failure_detected"]
    assert summary["guard_complement_retained"]
    assert summary["raw_face48_export_rejection_preserved"]
