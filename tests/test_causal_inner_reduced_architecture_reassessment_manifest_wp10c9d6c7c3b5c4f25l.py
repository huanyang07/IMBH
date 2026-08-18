from __future__ import annotations

import hashlib
import json

import run_causal_inner_reduced_architecture_reassessment_manifest_wp10c9d6c7c3b5c4f25l as f25l


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_parent_R32_rejection_is_locked_and_not_physical():
    summary, metrics, hashes = f25l._validate_parent()
    assert not summary["passed"]
    assert summary["projection_algebra_passed"]
    assert summary["remaining_unresolved_strictly_stable"]
    assert not summary["physical_failure_detected"]
    assert metrics["augmented_resolved_dimension"] == 180
    assert "R32_transfer.npz" in hashes


def test_candidate_families_and_dimension_budget_are_prospective():
    contract = f25l._contract()
    candidates = contract["candidate_families"]
    assert tuple(candidates["global_balanced_controls"]["orders"]) == (24, 48, 96)
    coherent = candidates["coherent_spatial_channel_models"]
    assert tuple(coherent["temporal_orders"]) == (24, 48, 96)
    assert tuple(coherent["spatial_channel_ranks"]) == (1, 2, 3)
    assert coherent["direct_map_is_never_projected"]
    assert contract["online_budget"]["maximum_online_continuous_dimension"] == 320
    assert f25l.BASE_ONLINE_DIMENSION + f25l.MAXIMUM_MEMORY_DIMENSION <= 320


def test_training_and_heldout_transfer_gates_are_identical():
    validation = f25l._contract()["normalization_and_validation"]
    gates = validation["candidate_pass_requires_training_and_heldout"]
    assert validation["training_frequencies"] == "all_33_parent_frequencies"
    assert "midpoint" in validation["heldout_frequencies"]
    assert gates["maximum_normalized_dynamic_transfer_relative_error_max"] == 0.25
    assert gates["RMS_normalized_dynamic_transfer_relative_error_max"] == 0.10
    assert gates["DC_normalized_dynamic_transfer_relative_error_max"] == 0.10
    assert gates["maximum_normalized_total_transfer_relative_error_max"] == 0.25


def test_execution_budget_forbids_truth_and_generator_work():
    budget = f25l._contract()["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0
    assert budget["allowed_new_truth_anchors"] == 0
    assert budget["allowed_saved_generator_memory_fits"] == 12


def test_canonical_manifest_when_available():
    summary_path = f25l.ARTIFACT_DIRECTORY / "summary.json"
    if not summary_path.exists():
        return
    summary = _read(summary_path)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["saved_R32_memory_selection_authorized"]
    assert not summary["physical_failure_detected"]
    for line in (f25l.ARTIFACT_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25l.ARTIFACT_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
