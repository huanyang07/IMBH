import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_constrained_history_manifest_"
    "wp10c9d6c7c3b5c4f24d"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f24d_authorizes_only_the_execution_preflight() -> None:
    summary = _read("summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert not summary["trajectory_executed"]
    assert not summary["physical_operator_changed"]
    assert summary["synthetic_history_limit_rejection_preserved"]
    assert summary["constrained_history_execution_preflight_authorized"]
    assert not summary["one_Q_execution_manifest_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["one_Q_propagation_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24d_requires_an_accepted_bdf1_history_for_bdf2() -> None:
    manifest = _read("execution_manifest.json")
    chain = manifest["execution_shaped_chain"]
    assert chain["timestep_ladder_seconds"] == [1.0e-7, 5.0e-8, 2.5e-8]
    assert chain["synthetic_backward_tangent_projection_forbidden"]
    assert "accepted_BDF1" in chain["history"]
    assert "accepted_BDF1_history" in chain["continuation"]
    assert manifest["residual_contract"]["binding_solver_residual"] == (
        "increment_primary_complete_BDF"
    )
    assert manifest["residual_contract"]["direct_rate_form"] == (
        "independent_parity_audit_only"
    )


def test_c4f24d_freezes_two_states_replay_and_unchanged_gates() -> None:
    manifest = _read("execution_manifest.json")
    assert [item["time_seconds"] for item in manifest["committed_states"]] == [
        0.020,
        0.016,
    ]
    gates = manifest["binding_gates"]
    assert gates["maximum_scaled_residual"] == 1.0e-10
    assert gates["maximum_Q3_relative_defect"] == 1.0e-12
    assert gates["minimum_state_rate_convergence_order"] == 0.9
    assert gates["minimum_reaction_action_convergence_order"] == 0.9
    assert manifest["durability"]["bitwise_required"]
    assert manifest["durability"]["replay_BDF2_from_serialized_BDF1_history"]


def test_c4f24d_preserves_cost_and_scientific_stops() -> None:
    manifest = _read("execution_manifest.json")
    assert manifest["cost_contract"]["one_exact_matrix_refresh_per_root"]
    assert manifest["cost_contract"]["fail_fast_before_refined_rungs"]
    assert manifest["cost_contract"]["no_long_trajectory"]
    assert not manifest["one_Q_execution_manifest_authorized"]
    assert not manifest["fixed_Q_micro_solver_authorized"]
    assert not manifest["one_Q_propagation_authorized"]
    assert not manifest["reduced_slow_evolution_authorized"]
    assert manifest["authorized_next"].endswith(
        "exact_constrained_BDF1_startup_BDF2_history_preflight"
    )


def test_c4f24d_provenance_hashes_the_source_bundle() -> None:
    provenance = _read("provenance.json")
    artifact_commit = subprocess.run(
        (
            "git",
            "log",
            "-1",
            "--format=%H",
            "--",
            str((ARTIFACT / "provenance.json").relative_to(ROOT)),
        ),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    for relative, digest in provenance["source_hashes"].items():
        if artifact_commit:
            contents = subprocess.run(
                ("git", "show", f"{artifact_commit}:{relative}"),
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
        else:
            contents = (ROOT / relative).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == digest
