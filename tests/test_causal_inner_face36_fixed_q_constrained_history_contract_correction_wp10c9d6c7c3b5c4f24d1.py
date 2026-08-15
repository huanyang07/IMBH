import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_constrained_history_contract_correction_"
    "wp10c9d6c7c3b5c4f24d1"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f24d1_blocks_execution_until_implementation_hardening() -> None:
    summary = _read("summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert not summary["trajectory_executed"]
    assert summary["uncorrected_f24e_execution_blocked"]
    assert summary["implementation_preflight_authorized"]
    assert not summary["physical_history_execution_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24d1_makes_increment_primary_binding() -> None:
    contract = _read("corrected_contract.json")
    temporal = contract["binding_temporal_contract"]
    assert temporal["nonlinear_root"] == "increment_primary_complete_BDF_only"
    assert not temporal["binding_solver_passes_direct_rate"]
    assert temporal["direct_rate_evaluation"] == "post_root_parity_audit_only"
    assert temporal["maximum_direct_rate_increment_parity_defect"] == 1.0e-9


def test_c4f24d1_freezes_fail_closed_method_gates() -> None:
    contract = _read("corrected_contract.json")
    acceptance = contract["acceptance_contract"]
    assert acceptance["single_fail_closed_acceptance_record"]
    assert acceptance["BDF2_requires_accepted_BDF1_history"]
    assert acceptance["minimum_path_reconstruction_factor"] == 1.0 - 1.0e-12
    assert acceptance["maximum_scaled_residual"] == 1.0e-10
    assert acceptance["maximum_Q3_relative_defect"] == 1.0e-12
    reaction = contract["reaction_contract"]
    assert reaction["required_numerical_rank"] == 3
    assert reaction["maximum_raw_Schur_condition_number"] == 1.0e8
    assert reaction["state_normalized_channels"] == (
        "audit_only_not_nonlinear_kernel"
    )


def test_c4f24d1_names_the_actual_solver_and_restart_contract() -> None:
    contract = _read("corrected_contract.json")
    solver = contract["solver_contract"]
    assert not solver["matrix_free_claim"]
    assert solver["subsequent_updates"] == (
        "dense_rank_one_Broyden_secant_updates"
    )
    assert solver["maximum_complete_Jacobian_assemblies_per_binding_root"] == 1
    restart = contract["restart_contract"]
    assert restart["Q3_target_and_constraint_scales"]
    assert restart["recompute_state_local_reaction_after_reload"]
    assert restart["bitwise_BDF2_replay"]


def test_c4f24d1_separates_constraint_diagnostics() -> None:
    diagnostics = _read("corrected_contract.json")["constraint_diagnostics"]
    assert diagnostics["reaction_channel_ledger"] != diagnostics[
        "constraint_action_ledger"
    ]
    assert diagnostics["endpoint_Q3_constraint"] == "binding"
    assert "not_subject" in diagnostics["finite_BDF_DQ3_times_state_rate"]


def test_c4f24d1_provenance_hashes_the_source_bundle() -> None:
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
