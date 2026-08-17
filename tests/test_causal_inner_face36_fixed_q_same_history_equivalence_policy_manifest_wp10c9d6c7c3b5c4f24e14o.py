import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_same_history_equivalence_policy_manifest_"
    "wp10c9d6c7c3b5c4f24e14o"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_same_history_equivalence_policy_"
    "manifest_wp10c9d6c7c3b5c4f24e14o"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_e14o_positive_diagnosis_is_locked() -> None:
    parent = RUNNER._parent_lock()
    assert parent["diagnosis_summary"]["passed"]
    assert (
        parent["diagnosis_summary"]["classification"]
        == "cold_shadow_residual_limited_action_equivalence_diagnosed"
    )
    assert parent["diagnosis_metrics"]["positive_diagnosis"]
    assert parent["historical_retry_summary"]["classification"] == "bounded_continuation_failed"


def test_e14o_keeps_production_gate_and_tightens_only_control_comparison() -> None:
    assert RUNNER.CONTRACT["production_step_acceptance"] == {
        "maximum_scaled_residual": 1.0e-10,
        "unchanged": True,
    }
    policy = RUNNER.CONTRACT["equivalence_control_policy"]
    assert policy["maximum_scaled_residual_before_state_action_comparison"] == 1.0e-12
    assert policy["maximum_scaled_state_difference"] == 1.0e-8
    assert policy["maximum_reaction_action_relative_difference"] == 1.0e-8
    assert not policy["polished_control_may_define_history"]
    assert policy["maximum_endpoint_polish_exact_assemblies"] == 1


def test_e14o_preserves_all_hard_stops() -> None:
    assert all(RUNNER.CONTRACT["hard_stops"].values())


def test_e14o_committed_manifest_closes_when_present() -> None:
    if not CANONICAL.exists():
        pytest.skip("canonical e14o manifest is recorded after freeze")
    summary = _read(CANONICAL / "summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["same_history_equivalence_policy_certificate_authorized"]
    entries = {}
    for line in (CANONICAL / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest() == digest


def test_e14o_cli_requires_freeze(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit) as raised:
        RUNNER.main()
    assert str(raised.value) == "select --freeze"
