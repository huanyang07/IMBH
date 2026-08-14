import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_six_mode_fine_replay_manifest_wp10c9d6c7c3b5c4f19"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f19_is_definitions_only_and_authorizes_only_fine_replay():
    summary = _read("summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert not summary["trajectory_executed"]
    assert summary["saved_middle_history_reused"]
    assert summary["middle_replay_forbidden"]
    assert summary["fine_dynamic_coordinate_replay_authorized"]
    assert summary["authorized_next"].endswith(
        "analysis_only_fine_six_mode_dynamic_coordinate_replay"
    )


def test_c4f19_freezes_stable_dual_and_selected_face36_plateau():
    manifest = _read("fine_replay_manifest.json")
    dual = manifest["stable_dual_contract"]
    assert dual["methods"] == ["reduced_QR", "thin_SVD"]
    assert dual["normal_equations_forbidden"]
    assert dual["maximum_biorthogonality_defect"] == 1.0e-10
    assert dual["maximum_normalized_slow_lift_annihilation_defect"] == 1.0e-10
    assert dual["maximum_relative_QR_SVD_dual_difference"] == 1.0e-8
    audit = manifest["face36_directional_JVP_contract"]
    assert audit["selected_relative_steps"] == [5.0e-5, 1.0e-4]
    assert audit["time_ids_microseconds"] == [5000, 5400, 10000, 16000, 20000]
    assert audit["directions"] == 6
    assert audit["maximum_relative_defect_at_each_selected_step"] == 1.0e-8
    assert audit["same_pair_required_for_all_times_and_directions"]


def test_c4f19_freezes_cross_grid_coordinate_and_output_gates():
    cross = _read("fine_replay_manifest.json")["cross_resolution_contract"]
    assert cross["restrict_middle_and_fine_state_directions_to_common_parent"]
    assert cross["leading_block_dimensions"] == [0, 2]
    assert cross["weak_enrichment_block_dimensions"] == [2, 6]
    assert cross["align_weak_block_by_orthogonal_Procrustes"]
    assert cross["individual_weak_mode_matching_forbidden"]
    assert cross["minimum_leading_block_projector_cosine"] == 0.95
    assert cross["minimum_full_subspace_projector_cosine"] == 0.90
    assert cross["maximum_six_mode_output_weighted_RMS_error"] == 0.10
    assert cross["maximum_six_mode_significant_direction_error"] == 0.25
    assert cross["guard_complement_retained_without_smallness_assumption"]


def test_c4f19_freezes_cost_and_fail_fast_controls():
    manifest = _read("fine_replay_manifest.json")
    replay = manifest["authorized_fine_replay"]
    assert replay["layout"] == "fine"
    assert replay["uses_saved_c4f17_middle_state_direction_history"]
    assert not replay["reruns_middle_propagation"]
    assert replay["one_factorization_six_RHS_per_step"]
    assert replay["run_initial_dual_and_face36_audits_before_full_propagation"]
    cost = manifest["cost_contract"]
    assert cost["expected_fine_replay_and_selected_audits_wall_hours"] == [3.5, 5.5]
    assert cost["selected_two_step_JVP_audit_replaces_six_step_sweep"]
    assert cost["full_middle_replay_forbidden"]
    assert cost["stop_after_initial_audits_if_they_fail"]


def test_c4f19_keeps_physics_and_reduction_work_blocked():
    summary = _read("summary.json")
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["nonlinear_retained_mode_pilot_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert not summary["physical_failure_detected"]
    assert summary["guard_complement_retained"]
    assert summary["raw_face48_export_rejection_preserved"]


def test_c4f19_hashes_are_self_consistent():
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    assert set(entries) == {
        "config.json",
        "fine_replay_manifest.json",
        "summary.json",
        "provenance.json",
    }
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
