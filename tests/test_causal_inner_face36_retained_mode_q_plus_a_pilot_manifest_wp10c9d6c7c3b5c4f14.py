from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_face36_retained_mode_q_plus_a_pilot_manifest_wp10c9d6c7c3b5c4f14 as c4f14


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_c4f14_freezes_two_mode_consensus_basis_without_overinterpretation():
    manifest = _read(c4f14.MANIFEST_PATH)
    summary = _read(c4f14.SUMMARY_PATH)
    assert summary["observable_memory_dimension"] == 2
    assert manifest["retained_amplitudes"]["names"] == ["a1", "a2"]
    assert "oscillatory_phase_pair" in manifest["retained_amplitudes"]["not_assumed"]
    metrics = summary["consensus_basis_metrics"]
    assert metrics["minimum_form_energy_capture"] >= 0.99
    assert metrics["minimum_layout_subspace_cosine"] >= 0.95
    assert metrics["spectral_gap_sigma2_over_sigma3"] >= 5.0
    with np.load(c4f14.MODE_BASIS_PATH, allow_pickle=False) as arrays:
        basis = arrays["consensus_direction_coefficients"]
        assert basis.shape == (29, 2)
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1.0e-12)


def test_c4f14_freezes_physical_Q_and_macro_only_reaction_support():
    manifest = _read(c4f14.MANIFEST_PATH)
    assert manifest["slow_state"]["Q"] == [
        "exact_mapped_M_parent_cells_36_to_64",
        "exact_mapped_J_parent_cells_36_to_64",
        "exact_mapped_E_parent_cells_36_to_64",
    ]
    reaction = manifest["physical_constraint_reaction"]
    assert reaction["reaction_support"] == "macro_only_parent_cells_48_to_64"
    assert reaction["B_Q_equals_DQ_transpose_forbidden"]
    assert reaction["Euclidean_projection_forbidden"]
    assert reaction["reaction_must_not_modify_micro_core_or_duplicate_guard"]
    assert reaction["reaction_M_J_E_and_work_must_be_ledgered"]
    assert set(reaction["raw_physical_reaction_channels"]) == {
        "mass_loading",
        "external_torque",
        "external_heating",
    }
    assert "inverse_of_DQ_M_inverse_B_raw" in reaction["normalized_map"]


def test_c4f14_authorizes_only_analysis_preflight():
    manifest = _read(c4f14.MANIFEST_PATH)
    summary = _read(c4f14.SUMMARY_PATH)
    assert manifest["definitions_only"]
    assert not manifest["trajectory_executed"]
    assert manifest["reaction_map_preflight_authorized"]
    assert not manifest["fixed_Q_micro_solver_authorized"]
    assert not manifest["nonlinear_retained_mode_pilot_authorized"]
    assert not manifest["reduced_slow_evolution_authorized"]
    assert not manifest["fifty_ms_propagation_authorized"]
    assert not summary["physical_reaction_map_derived"]
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c4f15_analysis_only_Q_plus_a_reaction_map_"
        "and_coordinate_preflight"
    )


def test_c4f14_preserves_overlap_and_cost_stops():
    manifest = _read(c4f14.MANIFEST_PATH)
    partition = manifest["physical_partition"]
    assert partition["shared_exchange_parent_face"] == 36
    assert partition["macro_owns_guard_physical_inventory_exactly_once"]
    assert partition["micro_guard_fine_complement_retained"]
    assert manifest["slow_state"]["raw_face48_flux_forbidden"]
    assert manifest["conditional_nonlinear_pilot_after_preflight"][
        "maximum_full_nonlinear_anchor_lifts"
    ] == 2
    assert manifest["cost_contract"]["no_50ms_or_full_fine_trajectory"]


def test_c4f14_canonical_hashes_close():
    for line in (c4f14.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha(c4f14.CANONICAL_DIRECTORY / name) == expected
