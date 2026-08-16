import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_bounded_continuation_manifest_"
    "wp10c9d6c7c3b5c4f24e14c"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_primary_bounded_continuation_manifest_"
    "wp10c9d6c7c3b5c4f24e14c"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_c4f24e14c_parent_and_seed_are_authorized() -> None:
    parent = RUNNER._parent_authorization()
    seed = RUNNER._seed_lock()
    assert parent["e14b_summary"][
        "primary_pilot_execution_manifest_authorized"
    ]
    assert not parent["e14b_summary"][
        "bounded_continuation_execution_authorized"
    ]
    assert seed["primitive_history_bitwise"]
    assert seed["current_order"] == 2
    assert seed["next_order"] == 2
    assert not seed["has_nonlinear_solver_state"]


def test_c4f24e14c_freezes_cold_then_three_warm_roots() -> None:
    roots = RUNNER.CONTRACT["main_root_sequence"]
    assert [root["label"] for root in roots] == [
        "cold_1",
        "warm_1",
        "warm_2",
        "warm_3",
    ]
    assert roots[0]["initial_exact_complete_matrix_required"]
    assert roots[0]["maximum_exact_assemblies"] == 2
    assert all(
        not root["initial_exact_complete_matrix_required"]
        and root["maximum_exact_assemblies"] == 1
        for root in roots[1:]
    )
    assert all(root["may_define_main_history"] for root in roots)


def test_c4f24e14c_predictor_and_reaction_coordinates_are_restartable() -> None:
    predictor = RUNNER.CONTRACT["predictor_contract"]
    solver = RUNNER.CONTRACT["solver_contract"]
    assert "checkpoint_previous_primitive_increment" in predictor[
        "state_rate_predictor"
    ]
    assert predictor[
        "state_rate_predictor_must_be_reconstructed_identically_after_restart"
    ]
    assert predictor["physical_reaction_action_invariance_is_binding"]
    assert not predictor["multiplier_coordinate_equality_is_binding"]
    assert solver["serialized_solver_matrix_basis"] == "raw_reaction_channels"
    assert solver["warm_matrix_multiplier_columns_rebased_to_current_transform"]


def test_c4f24e14c_freezes_replay_shadow_and_half_step_controls() -> None:
    replay = RUNNER.CONTRACT["restart_replay_contract"]
    shadow = RUNNER.CONTRACT["same_history_cold_shadow"]
    half = RUNNER.CONTRACT["matched_endpoint_half_step_audit"]
    assert replay["restart_checkpoint"] == "after_warm_1_before_warm_2"
    assert replay["replayed_suffix"] == ["warm_2", "warm_3"]
    assert "raw_coordinate_Broyden_matrix_and_anchor" in replay[
        "bitwise_fields"
    ]
    assert shadow["main_root"] == "warm_2"
    assert not shadow["may_define_main_history"]
    assert shadow["maximum_warm_to_cold_wall_time_ratio"] == 0.75
    assert half["full_step_root"] == "warm_3"
    assert half["half_timestep_seconds"] == 5.0e-8
    assert half["number_of_cold_half_steps"] == 2
    assert not half["half_steps_may_define_main_history"]


def test_c4f24e14c_preserves_scientific_and_authorization_boundaries() -> None:
    gates = RUNNER.CONTRACT["inherited_step_gates"]
    stops = RUNNER.CONTRACT["hard_stops"]
    classes = RUNNER.CONTRACT["classification_contract"]
    assert gates["maximum_scaled_residual"] == 1.0e-10
    assert gates["maximum_Q3_relative_defect"] == 1.0e-12
    assert gates["minimum_path_reconstruction_factor"] == 1.0 - 1.0e-12
    assert gates["incoming_excision_characteristics"] == 0
    assert classes["scientific_pass_cost_fail"] == (
        "bounded_continuation_valid_cost_failed"
    )
    assert classes["scientific_failure_may_not_be_reclassified_as_cost_failure"]
    assert stops["no_heldout_continuation"]
    assert stops["no_fixed_Q_micro_solver"]
    assert stops["no_reduced_slow_evolution"]


def test_c4f24e14c_freeze_writes_prospective_package(
    monkeypatch,
    tmp_path,
) -> None:
    artifact = tmp_path / "manifest"
    monkeypatch.setattr(RUNNER, "ARTIFACT_DIRECTORY", artifact)
    monkeypatch.setattr(RUNNER, "_tracked_tree_is_clean", lambda: True)
    monkeypatch.setattr(RUNNER, "_catalog", lambda summary: None)
    monkeypatch.setattr(RUNNER, "_git", lambda *arguments: "frozen-git-id")
    summary = RUNNER._freeze()
    assert summary["passed"]
    assert summary["definitions_only"]
    assert not summary["trajectory_executed"]
    assert summary["primary_bounded_continuation_execution_authorized"]
    assert not summary["heldout_continuation_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    contract = _read(artifact, "execution_manifest.json")
    assert not contract["trajectory_may_execute_during_freeze"]
    entries = {}
    for line in (artifact / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    assert set(entries) == {
        "execution_manifest.json",
        "parent_authorization.json",
        "provenance.json",
        "seed_lock.json",
        "summary.json",
    }
    for name, digest in entries.items():
        assert hashlib.sha256((artifact / name).read_bytes()).hexdigest() == (
            digest
        )


def test_c4f24e14c_committed_manifest_closes_when_present() -> None:
    if not MANIFEST.exists():
        pytest.skip("canonical e14c manifest is recorded in the next commit")
    summary = _read(MANIFEST, "summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert not summary["trajectory_executed"]
    assert summary["primary_bounded_continuation_execution_authorized"]
    entries = {}
    for line in (MANIFEST / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((MANIFEST / name).read_bytes()).hexdigest() == (
            digest
        )


def test_c4f24e14c_cli_is_freeze_only(monkeypatch) -> None:
    monkeypatch.setattr(
        RUNNER,
        "_freeze",
        lambda: {"passed": True, "classification": "mock_freeze"},
    )
    monkeypatch.setattr(sys, "argv", ["runner", "--freeze"])
    RUNNER.main()


def test_c4f24e14c_cli_requires_freeze(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit) as raised:
        RUNNER.main()
    assert str(raised.value) == "select --freeze"
