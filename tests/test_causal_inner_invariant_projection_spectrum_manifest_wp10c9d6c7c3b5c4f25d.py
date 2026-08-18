from __future__ import annotations

import hashlib
import json

import pytest

import run_causal_inner_invariant_projection_spectrum_manifest_wp10c9d6c7c3b5c4f25d as f25d


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_parent_negative_certificate_is_hash_locked_and_not_physical():
    summary, hashes = f25d._validate_parent()
    assert not summary["passed"]
    assert not summary["physical_failure_detected"]
    assert "descriptor_A.npz" in hashes
    assert "projection.npz" in hashes


def test_stage_1_is_minimal_invariant_compatible_R82():
    stage = f25d._contract()["stage_1_projection"]
    assert stage["resolved_dimension"] == 82
    assert stage["mapped_only_field_indices"] == (0, 2, 3)
    assert stage["mapped_plus_responsive_height_field_indices"] == (1, 4)
    assert stage["fixed_Q_rows_are_formed_from_mapped_storage_only"]
    assert stage["automatic_84_coordinate_rescue_forbidden"]
    assert stage["pass_requires"]["constraint_rowspace_relative_defect_max"] == 5.0e-10


def test_stage_2_is_fail_closed_and_fits_no_memory():
    stage = f25d._contract()["stage_2_spectrum_transfer"]
    assert stage["requires_stage_1_pass"]
    assert stage["frequency_grid"]["count_excluding_DC"] == 32
    assert stage["frequency_grid"]["exact_DC_is_separate"]
    assert stage["unstable_modes_may_not_be_fit_as_stable_memory"]
    assert stage["no_memory_coefficients_are_fit_in_this_package"]


def test_execution_budget_reuses_saved_generator_and_forbids_truth():
    budget = f25d._contract()["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_descriptor_assemblies"] == 0
    assert budget["saved_complete_generator_must_be_reused"]
    assert budget["maximum_wall_hours"] == 2.0


def test_frequency_grid_spans_declared_cycle_and_fast_audit_scales():
    grid = f25d._frequency_grid()
    assert grid["values_per_second"] == sorted(grid["values_per_second"])
    assert grid["angular_frequency_min_per_second"] == pytest.approx(
        2.0 * 3.141592653589793 / (6.7 * 86400.0)
    )
    assert grid["angular_frequency_max_per_second"] == pytest.approx(
        3.141592653589793 / 1.0e-7
    )
    assert grid[
        "high_frequency_samples_are_diagnostic_not_explicit_online_step_constraints"
    ]


def test_claim_boundary_keeps_campaign_and_cycle_blocked():
    claims = f25d._contract()["claim_boundary"]
    assert not claims["full_anchor_campaign_authorized"]
    assert not claims["memory_fit_authorized_in_this_package"]
    assert not claims["online_solver_authorized"]
    assert not claims["predictive_cycle_authorized"]
    assert not claims["reduced_slow_evolution_authorized"]


def test_canonical_manifest_when_available():
    summary_path = f25d.ARTIFACT_DIRECTORY / "summary.json"
    if not summary_path.exists():
        pytest.skip("projection/spectrum manifest not frozen yet")
    summary = _read(summary_path)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["stage_1_projection_authorized"]
    assert summary["stage_2_spectrum_transfer_authorized_only_after_stage_1_pass"]
    assert not summary["full_anchor_campaign_authorized"]
    for line in (f25d.ARTIFACT_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25d.ARTIFACT_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
