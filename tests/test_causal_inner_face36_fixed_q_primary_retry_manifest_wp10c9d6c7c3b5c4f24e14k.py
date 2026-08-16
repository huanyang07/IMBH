import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_retry_manifest_"
    "wp10c9d6c7c3b5c4f24e14k"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_primary_retry_manifest_"
    "wp10c9d6c7c3b5c4f24e14k"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_e14k_locks_positive_warm_policy_parent() -> None:
    parent = RUNNER._parent_lock()
    assert parent["summary"]["full_primary_retry_manifest_authorized"]
    assert parent["summary"]["parent_classification_preserved"] == (
        "bounded_continuation_failed"
    )


def test_e14k_replaces_only_warm_solver_policy() -> None:
    solver = RUNNER.CONTRACT["solver_contract"]
    roots = RUNNER.CONTRACT["main_root_sequence"]
    assert [root["label"] for root in roots] == [
        "cold_1",
        "warm_1",
        "warm_2",
        "warm_3",
    ]
    assert solver["warm_refresh_policy"] == (
        "on_line_search_failure_or_iteration_reserve"
    )
    assert solver["warm_iteration_reserve_trigger"] == 6
    assert solver["warm_failed_relative_backtrack_trigger"] == 4
    assert solver["maximum_newton_iterations"] == 8
    assert not solver["zero_refresh_warm_root_required"]


def test_e14k_cost_binds_same_history_wall_time() -> None:
    gates = RUNNER.CONTRACT["trajectory_gates"]
    shadow = RUNNER.CONTRACT["same_history_cold_shadow"]
    assert gates["same_history_warm_to_cold_wall_ratio_maximum"] == 0.75
    assert gates["warm_refresh_count_is_diagnostic_not_binding"]
    assert shadow["maximum_warm_to_cold_wall_time_ratio"] == 0.75


def test_e14k_committed_manifest_closes_when_present() -> None:
    if not CANONICAL.exists():
        pytest.skip("canonical e14k manifest is recorded after freeze")
    summary = _read(CANONICAL / "summary.json")
    assert summary["passed"]
    assert summary["primary_bounded_continuation_execution_authorized"]
    entries = {}
    for line in (CANONICAL / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest() == digest


def test_e14k_cli_requires_freeze(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit, match="select --freeze"):
        RUNNER.main()
