import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_bounded_continuation_cost_manifest_"
    "wp10c9d6c7c3b5c4f24e14a"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_bounded_continuation_cost_manifest_"
    "wp10c9d6c7c3b5c4f24e14a"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_c4f24e14a_is_definitions_only() -> None:
    contract = RUNNER.CONTRACT
    assert contract["definitions_only"]
    assert not contract["trajectory_may_execute"]
    assert contract["authorized_objective"].startswith("implementation_only")
    assert not contract["authorization_boundaries"][
        "bounded_continuation_execution_authorized"
    ]
    assert not contract["authorization_boundaries"][
        "fixed_Q_micro_solver_authorized"
    ]
    assert not contract["authorization_boundaries"][
        "reduced_slow_evolution_authorized"
    ]


def test_c4f24e14a_freezes_complete_history_resolution() -> None:
    _, _, inventory = RUNNER._validate_parent_and_seed()
    assert inventory["canonical_root_arrays_complete"]
    assert not inventory["complete_storage_history_present"]
    seed = RUNNER.CONTRACT["seed"]
    assert seed["synthetic_or_projected_history_forbidden"]
    assert "straight_primitive_path" in seed["complete_history_policy"]
    assert "authentic_coarse_startup" in seed["complete_history_policy"]


def test_c4f24e14a_freezes_reaction_coordinate_rebase() -> None:
    reaction = RUNNER.CONTRACT["reaction_coordinate_contract"]
    assert reaction["serialized_solver_matrix_multiplier_basis"] == (
        "raw_reaction_channels"
    )
    assert reaction["step_residual_multiplier_basis"] == "frozen_normalized"
    assert "inverse_T_new" in reaction["predictor_rebase"]
    assert "inverse_T_old" in reaction["matrix_multiplier_column_rebase"]
    assert reaction["physical_reaction_action_invariance_required"]
    assert not reaction["multiplier_coordinate_equality_binding"]


def test_c4f24e14a_freezes_bounded_primary_pilot() -> None:
    pilot = RUNNER.CONTRACT["prospective_primary_pilot"]
    assert not pilot["execution_authorized_by_this_manifest"]
    assert pilot["new_BDF2_roots"] == 4
    assert pilot["cold_roots"] == 1
    assert pilot["warm_roots"] == 3
    assert pilot["new_physical_horizon_seconds"] == 4.0e-7
    assert not pilot["warm_roots_force_initial_exact_matrix"]
    assert pilot["maximum_warm_exact_refreshes_per_root"] == 1
    assert pilot["same_history_cold_shadow_at_warm_root_index"] == 2
    assert pilot["bitwise_two_step_suffix_replay_required"]
    assert pilot["conditional_matched_endpoint_half_step_audit"]
    assert pilot["pilot_interpretation"] == "infrastructure_and_cost_only"


def test_c4f24e14a_preserves_scientific_gates_and_separates_cost() -> None:
    gates = RUNNER.CONTRACT["inherited_step_gates"]
    decisions = RUNNER.CONTRACT["prospective_decision_gates"]
    classifications = RUNNER.CONTRACT["result_classifications"]
    assert gates["maximum_scaled_residual"] == 1.0e-10
    assert gates["maximum_Q3_relative_defect"] == 1.0e-12
    assert gates["minimum_path_reconstruction_factor"] == 1.0 - 1.0e-12
    assert gates["incoming_excision_characteristics"] == 0
    assert decisions["minimum_warm_roots_without_exact_refresh"] == 2
    assert decisions["maximum_same_history_warm_to_cold_wall_time_ratio"] == (
        0.75
    )
    assert classifications["scientific_pass_cost_fail"] == (
        "bounded_continuation_valid_cost_failed"
    )


def test_c4f24e14a_freeze_writes_complete_temporary_package(
    monkeypatch,
    tmp_path,
) -> None:
    artifact = tmp_path / "manifest"
    monkeypatch.setattr(RUNNER, "ARTIFACT_DIRECTORY", artifact)
    monkeypatch.setattr(RUNNER, "_tracked_tree_is_clean", lambda: True)
    monkeypatch.setattr(RUNNER, "_catalog", lambda directory, summary: None)
    monkeypatch.setattr(RUNNER, "_git", lambda *arguments: "frozen-git-id")
    summary = RUNNER._freeze()
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["implementation_preflight_authorized"]
    assert not summary["bounded_continuation_execution_authorized"]
    authorization = _read(artifact, "parent_authorization.json")
    assert not authorization["input_execution_contract"][
        "one_Q_execution_manifest_authorized"
    ]
    assert authorization["final_decision"][
        "one_Q_execution_manifest_authorized"
    ]
    inventory = _read(artifact, "seed_inventory.json")
    assert inventory["canonical_root_arrays_complete"]
    assert not inventory["complete_storage_history_present"]
    entries = {}
    for line in (artifact / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    assert set(entries) == {
        "execution_manifest.json",
        "parent_authorization.json",
        "provenance.json",
        "seed_inventory.json",
        "summary.json",
    }
    for name, digest in entries.items():
        assert hashlib.sha256((artifact / name).read_bytes()).hexdigest() == digest


def test_c4f24e14a_committed_manifest_closes_when_present() -> None:
    if not MANIFEST.exists():
        pytest.skip("canonical e14a manifest is recorded in the next commit")
    summary = _read(MANIFEST, "summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["implementation_preflight_authorized"]
    assert not summary["bounded_continuation_execution_authorized"]
    entries = {}
    for line in (MANIFEST / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((MANIFEST / name).read_bytes()).hexdigest() == digest


def test_c4f24e14a_cli_is_freeze_only(monkeypatch) -> None:
    monkeypatch.setattr(
        RUNNER,
        "_freeze",
        lambda: {"passed": True, "classification": "mock_freeze"},
    )
    monkeypatch.setattr(sys, "argv", ["runner", "--freeze"])
    RUNNER.main()


def test_c4f24e14a_cli_requires_freeze(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit) as raised:
        RUNNER.main()
    assert str(raised.value) == "select --freeze"
