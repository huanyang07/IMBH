from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_face36_augmented_memory_screen_wp10c9d6c7c3b5c4f13 as c4f13


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_preserves_analysis_only_face36_architecture():
    manifest = c4f13._validate_authorization()
    contract = c4f13._contract(manifest)
    assert not contract["new_nonlinear_trajectory"]
    assert not contract["fixed_Q_constraint"]
    assert not contract["per_step_projection"]
    assert contract["binding_output"] == "shared_face36_M_J_E_flux"
    assert c4f13.TOTAL_DIRECTIONS == 29
    audit = contract["face36_finite_difference_audit"]
    assert audit["relative_step"] == 5.0e-4
    assert audit["step_source"] == "existing_certified_c4f4_c4f5_face_flux_JVP_plateau"
    assert audit["gate_unchanged"]


def test_null_projection_is_exact_for_full_row_rank_constraint():
    directions = np.arange(35, dtype=float).reshape(5, 7)
    constraint = np.zeros((3, 7), dtype=float)
    constraint[:, :3] = np.eye(3)
    projected, defect = c4f13.c4f1._project_null(directions, constraint)
    assert defect <= 1.0e-14
    assert np.max(np.abs(projected @ constraint.T)) <= 1.0e-14


def test_committed_history_uses_valid_production_path_label():
    trajectory = c4f13.c4f1._middle_trajectory()
    history = c4f13._history(trajectory, 0)
    assert history.temporal_path_scheme == "straight_primitive_path"


def test_canonical_result_and_arrays_close_when_screen_is_complete():
    summary = _read(c4f13.SUMMARY_PATH)
    assert summary["analysis_only_memory_screen_completed"]
    assert not summary["new_nonlinear_trajectory_executed"]
    assert summary["initial_null_projection_only"]
    assert not summary["per_step_projection_executed"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    with np.load(c4f13.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        assert arrays["middle__face36_outputs"].shape == (40, 29, 3)
        assert arrays["fine__face36_outputs"].shape == (40, 29, 3)
        assert arrays["middle__guard_mapped"].shape == (40, 29, 12, 3)
        assert arrays["fine__guard_height_history"].shape == (40, 29, 12, 2)


def test_middle_first_and_fine_method_gates_are_reported():
    summary = _read(c4f13.SUMMARY_PATH)
    for label in ("middle", "fine"):
        result = summary[label]
        assert result["face36_output_map_audit_count"] == 4
        assert result["directions"] == 29
        assert result["steps"] == 39
        assert result["maximum_incoming_characteristics"] == 0


def test_checkpoint_finalize_keeps_numerical_and_metadata_identity_separate():
    provenance = _read(c4f13.PROVENANCE_PATH)
    assert provenance["finalization_from_completed_checkpoints"]
    assert (
        provenance["source_hashes"][c4f13.THIS_RUNNER]
        == c4f13.COMPLETED_NUMERICAL_RUNNER_SHA256
    )
    assert provenance["finalization_source_hashes"][c4f13.THIS_RUNNER] == _sha(
        c4f13.ROOT / c4f13.THIS_RUNNER
    )
    assert provenance["scientific_status"] == "CERTIFIED"
    catalog = _read(c4f13.CANONICAL_SUMMARY)
    assert (
        provenance["source_parent_commit"]
        == catalog["latest_source_parent_commit"]
    )


def test_canonical_hashes_close():
    for line in (c4f13.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha(c4f13.CANONICAL_DIRECTORY / name) == expected
