import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_same_history_equivalence_policy_"
    "wp10c9d6c7c3b5c4f24e14p"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_same_history_equivalence_policy_"
    "wp10c9d6c7c3b5c4f24e14p"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_e14p_production_and_equivalence_tolerances_are_distinct() -> None:
    assert RUNNER.PRODUCTION_RESIDUAL_TOLERANCE == 1.0e-10
    assert RUNNER.EQUIVALENCE_RESIDUAL_TOLERANCE == 1.0e-12
    assert RUNNER.STATE_EQUIVALENCE_TOLERANCE == 1.0e-8
    assert RUNNER.ACTION_EQUIVALENCE_TOLERANCE == 1.0e-8


def test_e14p_committed_result_closes_when_present() -> None:
    if not CANONICAL.exists():
        pytest.skip("canonical e14p result is recorded after execution")
    summary = _read(CANONICAL / "summary.json")
    metrics = _read(CANONICAL / "metrics.json")
    assert summary["certificate_executed"]
    assert not summary["trajectory_executed"]
    assert not summary["production_step_acceptance_changed"]
    assert metrics["accepted_control_before_polish"]
    assert metrics["polish_required"]
    assert not metrics["nonpropagation"]["continuation_state_constructed"]
    entries = {}
    for line in (CANONICAL / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest() == digest


def test_e14p_cli_requires_run(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit) as raised:
        RUNNER.main()
    assert str(raised.value) == "select --run"
