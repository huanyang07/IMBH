import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_exact_endpoint_diagnostic_"
    "wp10c9d6c7c3b5c4f24e14h"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_exact_endpoint_diagnostic_"
    "wp10c9d6c7c3b5c4f24e14h"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_e14h_correction_metrics_are_identity_for_equal_matrices() -> None:
    matrix = np.diag([1.0, 2.0, 3.0])
    correction = np.asarray([0.5, -0.25, 0.125])
    metrics = RUNNER._correction_metrics(
        matrix, matrix, correction, correction
    )
    assert metrics["correction_cosine"] == 1.0
    assert metrics["correction_angle_radians"] == 0.0
    assert metrics["exact_to_carried_correction_norm_ratio"] == 1.0
    assert metrics["exact_jacobian_action_defect_on_carried_correction"] == 0.0
    assert metrics["carried_matrix_action_defect_on_exact_correction"] == 0.0


def test_e14h_committed_result_closes_when_present() -> None:
    if not CANONICAL.exists():
        pytest.skip("canonical e14h result is recorded after execution")
    summary = _read(CANONICAL / "summary.json")
    metrics = _read(CANONICAL / "metrics.json")
    assert summary["diagnostic_executed"]
    assert not summary["trajectory_executed"]
    assert summary["exact_jacobian_assemblies"] == 1
    assert summary["exact_newton_corrections"] == 1
    assert not metrics["nonpropagation"]["continuation_state_constructed"]
    entries = {}
    for line in (CANONICAL / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest() == digest


def test_e14h_cli_requires_run(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit) as raised:
        RUNNER.main()
    assert str(raised.value) == "select --run"
