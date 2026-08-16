import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_e14l_root_policy_is_cold_then_iteration_reserve_warm() -> None:
    cold = RUNNER._root_policy("cold_1")
    assert cold["initial_exact_jacobian_required"]
    assert cold["exact_jacobian_refresh_policy"] == "on_line_search_failure"
    assert cold["maximum_exact_jacobian_refreshes"] == 2
    for label in ("warm_1", "warm_2", "warm_3"):
        warm = RUNNER._root_policy(label)
        assert not warm["initial_exact_jacobian_required"]
        assert warm["use_carried_solver_state"]
        assert warm["maximum_exact_jacobian_refreshes"] == 1
        assert warm["exact_jacobian_refresh_policy"] == (
            "on_line_search_failure_or_iteration_reserve"
        )


def test_e14l_nonpropagating_controls_remain_cold() -> None:
    for label in ("cold_shadow", "half_1", "half_2"):
        policy = RUNNER._root_policy(label)
        assert policy["cold"]
        assert not policy["use_carried_solver_state"]


def test_e14l_does_not_mutate_historical_runner_globals() -> None:
    before = RUNNER.e14d.WORK_PACKAGE
    with RUNNER._legacy_runtime():
        assert RUNNER.e14d.WORK_PACKAGE == RUNNER.WORK_PACKAGE
    assert RUNNER.e14d.WORK_PACKAGE == before


def test_e14l_committed_result_closes_when_present() -> None:
    if not CANONICAL.exists():
        pytest.skip("canonical e14l result is recorded after execution")
    summary = _read(CANONICAL / "summary.json")
    assert summary["trajectory_executed"]
    entries = {}
    for line in (CANONICAL / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest() == digest


def test_e14l_cli_requires_mode(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit):
        RUNNER.main()
