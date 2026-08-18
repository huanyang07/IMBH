from __future__ import annotations

import hashlib
import json

import run_causal_inner_larger_coarse_pde_manifest_wp10c9d6c7c3b5c4f25j as f25j


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_parent_compact_memory_rejection_is_hash_locked_and_not_physical():
    summary, metrics, hashes = f25j._validate_parent()
    assert not summary["passed"]
    assert summary["full_order_numerical_passed"]
    assert summary["selected_order"] is None
    assert not summary["physical_failure_detected"]
    assert "metrics.json" in hashes


def test_R32_is_the_predeclared_larger_conservative_validation_grid():
    grid = f25j._contract()["R32_conservative_grid"]
    assert grid["truth_cells"] == 112
    assert grid["coarse_cells"] == 32
    assert grid["cellwise_storage_dimension"] == 160
    assert grid["base_resolved_dimension"] == 162
    assert grid["original_R16_boundaries_are_every_second_R32_boundary"]
    assert grid["face_36_exterior_partition_remains_exact_group_boundary"]
    assert grid["interior_M_J_E_fluxes_must_telescope_exactly"]


def test_dimension_budget_and_promotion_are_fail_closed():
    promotion = f25j._contract()["ordered_schur_promotion"]
    assert promotion["promote_every_nonstable_compressed_coordinate"]
    assert promotion["maximum_promoted_dimension"] == 30
    assert promotion["maximum_online_continuous_dimension"] == 192
    assert f25j.BASE_RESOLVED_DIMENSION + promotion["maximum_promoted_dimension"] == 192


def test_no_memory_total_transfer_gates_are_frozen():
    gates = f25j._contract()["no_memory_closure_screen"]["pass_requires"]
    assert gates["maximum_normalized_total_transfer_relative_error_max"] == 0.25
    assert gates["RMS_normalized_total_transfer_relative_error_max"] == 0.10
    assert gates["DC_normalized_total_transfer_relative_error_max"] == 0.10


def test_execution_budget_forbids_truth_generator_and_memory_fit():
    budget = f25j._contract()["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0
    assert budget["allowed_memory_coefficients_fit"] == 0


def test_canonical_manifest_when_available():
    summary_path = f25j.ARTIFACT_DIRECTORY / "summary.json"
    if not summary_path.exists():
        return
    summary = _read(summary_path)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["base_resolved_dimension"] == 162
    assert not summary["memory_fit_executed"]
    for line in (f25j.ARTIFACT_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25j.ARTIFACT_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
