import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_warm_policy_certificate_"
    "wp10c9d6c7c3b5c4f24e14j"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_warm_policy_certificate_"
    "wp10c9d6c7c3b5c4f24e14j"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_e14j_committed_result_closes_when_present() -> None:
    if not CANONICAL.exists():
        pytest.skip("canonical e14j result is recorded after execution")
    summary = _read(CANONICAL / "summary.json")
    metrics = _read(CANONICAL / "metrics.json")
    assert summary["one_warm_root_executed"]
    assert not summary["full_primary_retry_execution_authorized"]
    if summary["scientific_passed"]:
        assert metrics["checkpoint_roundtrip"]["bitwise_roundtrip"]
        assert metrics["endpoint_agreement"]["passed"]
        assert metrics["warm_root"]["exact_Jacobian_reasons"] == [
            "iteration_reserve"
        ]
    entries = {}
    for line in (CANONICAL / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest() == digest


def test_e14j_cli_requires_run(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit) as raised:
        RUNNER.main()
    assert str(raised.value) == "select --run"
