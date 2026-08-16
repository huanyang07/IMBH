import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_warm_policy_manifest_"
    "wp10c9d6c7c3b5c4f24e14i"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_warm_policy_manifest_"
    "wp10c9d6c7c3b5c4f24e14i"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_e14i_positive_diagnosis_is_locked() -> None:
    parent = RUNNER._parent_lock()
    assert parent["summary"]["warm_policy_manifest_authorized"]
    assert parent["diagnostic_metrics"]["positive_diagnosis"]


def test_e14i_warm_policy_reserves_two_corrections() -> None:
    warm = RUNNER.CONTRACT["warm_root"]
    assert warm["maximum_newton_iterations"] == 8
    assert warm["primary_trigger_iteration"] == 6
    assert warm["maximum_exact_assemblies"] == 1
    assert warm["forced_initial_exact_assembly"] is False
    assert warm["maximum_scaled_residual"] == 1.0e-10


def test_e14i_cost_uses_time_not_zero_refresh() -> None:
    cost = RUNNER.CONTRACT["cost_gate"]
    assert cost["warm_to_same_history_cold_wall_ratio_maximum"] == 0.75
    assert cost["zero_exact_refresh_required"] is False


def test_e14i_committed_manifest_closes_when_present() -> None:
    if not CANONICAL.exists():
        pytest.skip("canonical e14i manifest is recorded after freeze")
    summary = _read(CANONICAL / "summary.json")
    assert summary["passed"]
    assert summary["one_root_warm_policy_execution_authorized"]
    entries = {}
    for line in (CANONICAL / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest() == digest


def test_e14i_cli_requires_freeze(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit) as raised:
        RUNNER.main()
    assert str(raised.value) == "select --freeze"
