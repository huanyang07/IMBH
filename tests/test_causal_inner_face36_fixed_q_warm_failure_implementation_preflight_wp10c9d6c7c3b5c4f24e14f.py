import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_warm_failure_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e14f"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_warm_failure_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e14f"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_e14f_parent_authorizes_only_repair_preflight() -> None:
    parent = RUNNER._validate_parent_contract()
    assert parent["summary"]["accounting_repair_preflight_authorized"]
    assert not parent["summary"]["endpoint_diagnostic_execution_authorized"]
    assert parent["failure"]["summary"]["classification"] == (
        "bounded_continuation_failed"
    )


def test_e14f_implementation_contract_is_complete() -> None:
    checks = RUNNER._implementation_checks()
    assert checks
    assert all(checks.values()), checks


def test_e14f_reconstructs_true_cold_matrix_age() -> None:
    metrics = _read(RUNNER.E14D_DIRECTORY / "metrics.json")
    reconstructed = RUNNER._reconstruct_legacy_counter(metrics)
    assert reconstructed["legacy_serialized_value"] == 6
    assert reconstructed["total_broyden_updates"] == 6
    assert reconstructed["broyden_updates_since_last_exact"] == 1
    assert reconstructed["exact_assemblies"] == 2


def test_e14f_committed_package_closes_when_present() -> None:
    if not CANONICAL.exists():
        pytest.skip("canonical e14f preflight is recorded after execution")
    summary = _read(CANONICAL / "summary.json")
    replay = _read(CANONICAL / "endpoint_replay.json")
    assert summary["passed"]
    assert summary["accounting_repair_certified"]
    assert summary["endpoint_diagnostic_manifest_authorized"]
    assert not summary["endpoint_diagnostic_execution_authorized"]
    assert replay["bitwise_residual_reproduction"]
    entries = {}
    for line in (CANONICAL / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest() == digest


def test_e14f_cli_runs_only_explicit_mode(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        RUNNER,
        "_run",
        lambda: {"passed": True, "classification": "mock_preflight"},
    )
    monkeypatch.setattr(sys, "argv", ["runner", "--run"])
    RUNNER.main()
    assert '"passed": true' in capsys.readouterr().out


def test_e14f_cli_requires_run(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit) as raised:
        RUNNER.main()
    assert str(raised.value) == "select --run"
