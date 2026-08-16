import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_exact_endpoint_diagnostic_manifest_"
    "wp10c9d6c7c3b5c4f24e14g"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_exact_endpoint_diagnostic_manifest_"
    "wp10c9d6c7c3b5c4f24e14g"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_e14g_parent_replay_is_locked() -> None:
    parent = RUNNER._parent_lock()
    assert parent["summary"]["endpoint_diagnostic_manifest_authorized"]
    assert parent["endpoint_replay"]["bitwise_residual_reproduction"]


def test_e14g_authorizes_exactly_one_nonpropagating_correction() -> None:
    diagnostic = RUNNER.CONTRACT["authorized_diagnostic"]
    assert diagnostic["maximum_exact_complete_jacobian_assemblies"] == 1
    assert diagnostic["maximum_exact_newton_corrections"] == 1
    assert diagnostic["continuation_state_may_be_constructed"] is False
    assert diagnostic["rejected_or_corrected_candidate_may_enter_history"] is False
    assert diagnostic["unchanged_maximum_scaled_residual"] == 1.0e-10


def test_e14g_preserves_all_hard_stops() -> None:
    assert all(RUNNER.CONTRACT["hard_stops"].values())


def test_e14g_committed_manifest_closes_when_present() -> None:
    if not CANONICAL.exists():
        pytest.skip("canonical e14g manifest is recorded after freeze")
    summary = _read(CANONICAL / "summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["exact_endpoint_diagnostic_execution_authorized"]
    entries = {}
    for line in (CANONICAL / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest() == digest


def test_e14g_cli_is_freeze_only(monkeypatch) -> None:
    monkeypatch.setattr(RUNNER, "_freeze", lambda: {"passed": True})
    monkeypatch.setattr(sys, "argv", ["runner", "--freeze"])
    RUNNER.main()


def test_e14g_cli_requires_freeze(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit) as raised:
        RUNNER.main()
    assert str(raised.value) == "select --freeze"
