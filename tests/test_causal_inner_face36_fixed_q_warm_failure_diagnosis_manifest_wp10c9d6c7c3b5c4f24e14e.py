import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_warm_failure_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f24e14e"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_warm_failure_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f24e14e"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_e14e_locks_the_binding_parent_failure() -> None:
    failure = RUNNER._failure_lock()
    summary = failure["summary"]
    assert summary["classification"] == "bounded_continuation_failed"
    assert summary["accepted_main_BDF2_roots"] == 1
    assert summary["attempted_main_BDF2_roots"] == 2
    assert "checkpoint_cold_1.npz" in failure["array_inventories"]
    assert "result_warm_1.npz" in failure["array_inventories"]


def test_e14e_freezes_failure_aware_accounting_repairs() -> None:
    repair = RUNNER.CONTRACT["repair_preflight"]
    assert repair["required_solver_counters"] == [
        "total_broyden_updates",
        "broyden_updates_since_last_exact",
    ]
    assert repair["reset_updates_since_last_exact_on_every_exact_assembly"]
    assert repair["legacy_counter_semantics_must_be_marked_untrusted"]
    assert repair["accepted_horizon_uses_only_accepted_roots"]
    assert repair["accepted_ledgers_exclude_rejected_candidates"]
    assert repair["profiling_requires_call_counts"]
    assert repair["profiling_requires_exclusive_wall_times"]


def test_e14e_endpoint_diagnostic_is_nonpropagating() -> None:
    diagnostic = RUNNER.CONTRACT["endpoint_diagnostic_preflight"]
    assert diagnostic["committed_residual"] == 5.708109263036221e-9
    assert diagnostic["residual_reproduction"] == "bitwise"
    assert diagnostic["maximum_exact_complete_jacobian_assemblies"] == 1
    assert diagnostic["maximum_exact_newton_corrections"] == 1
    assert not diagnostic["continuation_state_may_be_constructed"]
    assert not diagnostic["rejected_endpoint_may_enter_history"]
    assert diagnostic["full_matrix_frobenius_defect_is_diagnostic_only"]


def test_e14e_iteration_reserve_is_the_primary_conditional_trigger() -> None:
    policy = RUNNER.CONTRACT["conditional_next_policy"]
    assert "maximum_iterations_minus_two" in policy[
        "primary_refresh_trigger"
    ]
    assert "four_relative_backtracks" in policy["secondary_refresh_trigger"]
    assert policy["maximum_exact_assemblies_per_warm_root"] == 1
    assert policy["unchanged_maximum_newton_iterations"] == 8
    assert policy["unchanged_maximum_scaled_residual"] == 1.0e-10
    assert policy["zero_refresh_is_not_a_cost_gate"]


def test_e14e_preserves_all_hard_stops() -> None:
    stops = RUNNER.CONTRACT["hard_stops"]
    assert stops["no_physical_root_in_this_manifest"]
    assert stops["no_gate_relaxation"]
    assert stops["no_iteration_budget_increase"]
    assert stops["no_full_four_root_retry"]
    assert stops["no_heldout_continuation"]
    assert stops["no_fixed_Q_micro_solver"]
    assert stops["no_reduced_slow_evolution"]


def test_e14e_freeze_writes_a_prospective_package(monkeypatch, tmp_path) -> None:
    artifact = tmp_path / "manifest"
    monkeypatch.setattr(RUNNER, "ARTIFACT_DIRECTORY", artifact)
    monkeypatch.setattr(RUNNER, "_tracked_tree_is_clean", lambda: True)
    monkeypatch.setattr(RUNNER, "_catalog", lambda summary: None)
    monkeypatch.setattr(RUNNER, "_git", lambda *arguments: "frozen-git-id")
    summary = RUNNER._freeze()
    assert summary["passed"]
    assert summary["definitions_only"]
    assert not summary["trajectory_executed"]
    assert summary["accounting_repair_preflight_authorized"]
    assert not summary["endpoint_diagnostic_execution_authorized"]
    assert not summary["warm_policy_execution_authorized"]
    entries = {}
    for line in (artifact / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    assert set(entries) == {
        "diagnosis_contract.json",
        "parent_failure_lock.json",
        "provenance.json",
        "summary.json",
    }
    for name, digest in entries.items():
        assert hashlib.sha256((artifact / name).read_bytes()).hexdigest() == digest


def test_e14e_committed_manifest_closes_when_present() -> None:
    if not MANIFEST.exists():
        pytest.skip("canonical e14e manifest is recorded in the next commit")
    summary = _read(MANIFEST, "summary.json")
    assert summary["passed"]
    assert summary["accounting_repair_preflight_authorized"]
    entries = {}
    for line in (MANIFEST / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((MANIFEST / name).read_bytes()).hexdigest() == digest


def test_e14e_cli_is_freeze_only(monkeypatch) -> None:
    monkeypatch.setattr(
        RUNNER,
        "_freeze",
        lambda: {"passed": True, "classification": "mock_freeze"},
    )
    monkeypatch.setattr(sys, "argv", ["runner", "--freeze"])
    RUNNER.main()


def test_e14e_cli_requires_freeze(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit) as raised:
        RUNNER.main()
    assert str(raised.value) == "select --freeze"
