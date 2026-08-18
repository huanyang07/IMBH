from __future__ import annotations

import hashlib
import json

import run_causal_inner_common_resolved_subspace_cross_anchor_manifest_wp10c9d6c7c3b5c4f25n as f25n


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_parent_R32_order_96_selection_is_locked_and_scoped():
    summary, metrics, hashes = f25n._validate_parent()
    assert summary["passed"]
    assert summary["selected_memory_order"] == 96
    assert summary["selected_online_continuous_dimension"] == 276
    assert not summary["physical_failure_detected"]
    assert metrics["full_order_numerical_passed"]
    assert "candidate_models.npz" in hashes


def test_execution_budget_allows_one_generator_and_no_trajectory():
    budget = f25n._contract()["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 1
    assert budget["allowed_new_truth_anchors"] == 1
    assert budget["allowed_common_basis_memory_fits"] == 2
    assert budget["maximum_wall_hours"] == 2.5


def test_heldout_anchor_is_exact_committed_16ms_middle_state():
    anchor = f25n._contract()["anchors"]["heldout_16ms"]
    assert anchor["time_seconds"] == 0.016
    assert anchor["layout"] == "same_committed_112_cell_middle_layout"
    assert anchor["state_must_be_copied_bitwise_without_projection"]
    assert f25n.MIDDLE_ARRAYS.exists()


def test_common_union_replaces_raw_Schur_interpolation():
    common = f25n._contract()["common_resolved_subspace"]
    assert common["union_numerical_rank_relative_cutoff"] == 1.0e-10
    assert common["maximum_common_promoted_dimension"] == 62
    assert "Procrustes" in common["anchor_local_realization"]
    assert common["coordinate_policy"]["raw_anchor_local_Schur_vectors_may_not_be_interpolated"]
    assert (
        f25n.PHYSICAL_R32_DIMENSION
        + f25n.MAXIMUM_COMMON_PROMOTED_DIMENSION
        + f25n.MEMORY_ORDER
        == f25n.MAXIMUM_ONLINE_CONTINUOUS_DIMENSION
    )


def test_memory_gates_and_atlas_branch_are_frozen():
    contract = f25n._contract()
    memory = contract["common_basis_memory"]
    gates = memory["pass_requires_at_each_anchor_on_training_and_heldout"]
    assert memory["order"] == 96
    assert gates["maximum_normalized_dynamic_transfer_relative_error_max"] == 0.25
    assert gates["RMS_normalized_dynamic_transfer_relative_error_max"] == 0.10
    assert gates["DC_normalized_dynamic_transfer_relative_error_max"] == 0.10
    assert "atlas" in contract["decisions"]["local_gates_pass_global_chart_alignment_fails"]
    assert contract["common_resolved_subspace"]["coordinate_policy"][
        "below_threshold_selects_local_atlas_not_physical_failure"
    ]


def test_canonical_manifest_when_available():
    summary_path = f25n.ARTIFACT_DIRECTORY / "summary.json"
    if not summary_path.exists():
        return
    summary = _read(summary_path)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["heldout_16ms_generator_preflight_authorized"]
    assert not summary["new_generator_assembly_executed"]
    assert not summary["physical_failure_detected"]
    for line in (f25n.ARTIFACT_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25n.ARTIFACT_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
