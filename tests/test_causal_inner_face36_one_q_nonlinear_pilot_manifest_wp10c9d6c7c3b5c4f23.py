import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_one_q_nonlinear_pilot_manifest_"
    "wp10c9d6c7c3b5c4f23"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f23_is_definitions_only_and_authorizes_only_step_preflight():
    summary = _read("summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert not summary["trajectory_executed"]
    assert not summary["physical_operator_changed"]
    assert summary["state_dependent_fixed_Q_step_preflight_authorized"]
    assert not summary["state_dependent_constrained_tangent_certified"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["one_Q_nonlinear_pilot_propagation_authorized"]


def test_c4f23_requires_complete_state_dependent_KKT_linearization():
    contract = _read("pilot_manifest.json")["state_dependent_constrained_system"]
    assert contract["finite_step_constraint"] == "Q3(p_new)-Q_target=0"
    assert contract["frozen_P_times_G_is_reference_only"]
    assert contract["Euclidean_projection_forbidden"]
    assert contract["manual_primitive_freezing_forbidden"]
    assert contract["constraint_reaction_M_J_E_ledgers_required"]
    assert set(contract["linearization_must_include"]) >= {
        "D_M",
        "D_R",
        "D_DQ3",
        "D_B_Q",
        "multiplier_coupling",
    }


def test_c4f23_freezes_fail_fast_step_and_JVP_gates():
    manifest = _read("pilot_manifest.json")
    preflight = manifest["authorized_state_dependent_step_preflight"]
    gates = manifest["preflight_gates"]
    assert not preflight["new_physical_trajectory"]
    assert not preflight["new_tangent_trajectory"]
    assert preflight["verify_zero_multiplier_reduces_to_unconstrained_residual"]
    assert preflight["verify_small_timestep_limit_against_c4f22_continuous_KKT"]
    assert gates["maximum_augmented_step_scaled_residual"] == 1.0e-10
    assert gates["maximum_dense_colored_Jacobian_relative_defect"] == 1.0e-9
    assert gates["maximum_directional_JVP_relative_defect"] == 1.0e-8
    assert gates["incoming_excision_characteristics"] == 0


def test_c4f23_limits_eventual_nonlinear_work_and_preserves_guard():
    manifest = _read("pilot_manifest.json")
    pilot = manifest["conditional_one_Q_pilot"]
    cost = manifest["cost_contract"]
    assert pilot["not_authorized_by_this_manifest"]
    assert pilot["screen_24_equal_Q_lifts_by_one_block_tangent"]
    assert pilot["maximum_full_nonlinear_anchor_lifts"] == 2
    assert pilot["fail_fast_windows_ms"] == [0.2, 1.0, 2.0, 5.0, 10.0, 20.0]
    assert not pilot["fine_full_trajectory_automatic"]
    assert cost["one_factorization_24_RHS"]
    assert cost["sparse_nonlinear_anchors_only"]
    assert cost["no_automatic_fine_or_50ms_run"]
    assert not manifest["inherited_architecture"]["guard_mixing_or_decay_assumed"]


def test_c4f23_keeps_long_and_reduced_evolution_blocked():
    summary = _read("summary.json")
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert not summary["guard_mixing_or_decay_claimed"]
    assert summary["raw_face48_export_rejection_preserved"]


def test_c4f23_hashes_are_self_consistent():
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
