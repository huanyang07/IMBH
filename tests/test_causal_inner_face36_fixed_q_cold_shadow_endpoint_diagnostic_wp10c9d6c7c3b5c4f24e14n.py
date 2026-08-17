import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_cold_shadow_endpoint_diagnostic_"
    "wp10c9d6c7c3b5c4f24e14n"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_cold_shadow_endpoint_diagnostic_"
    "wp10c9d6c7c3b5c4f24e14n"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_e14n_reaction_action_uses_physical_state_space_action() -> None:
    raw_lift = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
    transform = np.asarray([[2.0, 0.0], [0.5, 1.0]])
    multipliers = np.asarray([0.25, -0.5])
    evaluation = type(
        "Evaluation",
        (),
        {"reaction": type("Reaction", (), {"raw_reaction_lift": raw_lift})()},
    )()
    checkpoint = type(
        "Checkpoint", (), {"next_reaction_channel_transform": transform}
    )()
    expected = raw_lift @ transform @ multipliers
    assert np.array_equal(
        RUNNER._reaction_action(evaluation, checkpoint, multipliers), expected
    )


def test_e14n_committed_result_closes_when_present() -> None:
    if not CANONICAL.exists():
        pytest.skip("canonical e14n result is recorded after execution")
    summary = _read(CANONICAL / "summary.json")
    metrics = _read(CANONICAL / "metrics.json")
    assert summary["diagnostic_executed"]
    assert not summary["trajectory_executed"]
    assert summary["exact_jacobian_assemblies"] == 1
    assert summary["exact_newton_corrections"] == 1
    assert all(metrics["saved_endpoint_reproduction"].values())
    assert not metrics["nonpropagation"]["continuation_state_constructed"]
    entries = {}
    for line in (CANONICAL / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest() == digest


def test_e14n_cli_requires_run(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit) as raised:
        RUNNER.main()
    assert str(raised.value) == "select --run"
