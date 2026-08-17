import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_cold_shadow_endpoint_diagnostic_manifest_"
    "wp10c9d6c7c3b5c4f24e14m"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_cold_shadow_endpoint_diagnostic_"
    "manifest_wp10c9d6c7c3b5c4f24e14m"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_e14m_parent_failure_is_locked_to_cold_shadow_action() -> None:
    parent = RUNNER._parent_lock()
    shadow = parent["decisive_metrics"]["same_history_cold_shadow"]
    assert parent["summary"]["classification"] == "bounded_continuation_failed"
    assert parent["decisive_metrics"]["replay_passed"]
    assert parent["decisive_metrics"]["matched_half_step_passed"]
    assert shadow["root"]["accepted"]
    assert shadow["cost_passed"]
    assert not shadow["scientific_passed"]
    assert shadow["scaled_state_absolute_defect"] <= 1.0e-8
    assert shadow["reaction_action_relative_defect"] > 1.0e-8


def test_e14m_authorizes_exactly_one_nonpropagating_correction() -> None:
    diagnostic = RUNNER.CONTRACT["authorized_diagnostic"]
    assert diagnostic["maximum_exact_complete_jacobian_assemblies"] == 1
    assert diagnostic["maximum_exact_newton_corrections"] == 1
    assert diagnostic["continuation_state_may_be_constructed"] is False
    assert diagnostic["candidate_may_enter_history"] is False
    assert diagnostic["unchanged_maximum_scaled_residual"] == 1.0e-10
    assert diagnostic["unchanged_maximum_reaction_action_relative_difference"] == 1.0e-8


def test_e14m_preserves_all_hard_stops() -> None:
    assert all(RUNNER.CONTRACT["hard_stops"].values())


def test_e14m_committed_manifest_closes_when_present() -> None:
    if not CANONICAL.exists():
        pytest.skip("canonical e14m manifest is recorded after freeze")
    summary = _read(CANONICAL / "summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["cold_shadow_endpoint_diagnostic_execution_authorized"]
    entries = {}
    for line in (CANONICAL / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest() == digest


def test_e14m_cli_is_freeze_only(monkeypatch) -> None:
    monkeypatch.setattr(RUNNER, "_freeze", lambda: {"passed": True})
    monkeypatch.setattr(sys, "argv", ["runner", "--freeze"])
    RUNNER.main()


def test_e14m_cli_requires_freeze(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit) as raised:
        RUNNER.main()
    assert str(raised.value) == "select --freeze"
