import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_continuation_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e14b"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_continuation_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e14b"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_c4f24e14b_parent_authorizes_implementation_only() -> None:
    parent = RUNNER._parent_contract()
    assert parent["passed"]
    assert parent["implementation_preflight_authorized"]
    assert not parent["bounded_continuation_execution_authorized"]
    assert not parent["fixed_Q_micro_solver_authorized"]
    assert not parent["reduced_slow_evolution_authorized"]


def test_c4f24e14b_implementation_contract_closes() -> None:
    contract = RUNNER._implementation_contract()
    assert contract["all_checks_passed"]
    assert all(contract["checks"].values())
    assert not contract["physical_operator_changed"]
    assert not contract["production_defaults_changed"]
    assert contract["canonical_solver_state_multiplier_basis"] == (
        "raw_reaction_channels"
    )
    assert contract["binding_step_multiplier_basis"] == "frozen_normalized"
    assert contract["warm_exact_refresh_policy"] == (
        "line_search_failure_only"
    )


def test_c4f24e14b_runner_cannot_advance_or_solve_a_root() -> None:
    source = (ROOT / RUNNER.THIS_RUNNER).read_text(encoding="utf-8")
    forbidden = (
        "solve_causal_five_field_fixed_q_bdf(",
        "solve_causal_five_field_fixed_q_backward_euler(",
        "advance_causal_five_field_monolithic_bdf(",
    )
    assert all(token not in source for token in forbidden)
    assert '"trajectory_executed": False' in source
    assert '"nonlinear_root_solved": False' in source


def test_c4f24e14b_focused_suite_contains_method_and_runner_tests() -> None:
    assert "tests/test_causal_inner_fixed_q.py" in RUNNER.FOCUSED_TESTS
    assert "tests/test_causal_inner_monolithic_bdf.py" in RUNNER.FOCUSED_TESTS
    assert RUNNER.THIS_TEST in RUNNER.FOCUSED_TESTS


def test_c4f24e14b_committed_preflight_closes_when_present() -> None:
    if not ARTIFACT.exists():
        pytest.skip("canonical e14b preflight is recorded in the next commit")
    summary = _read(ARTIFACT, "summary.json")
    metrics = _read(ARTIFACT, "seed_reconstruction_metrics.json")
    assert summary["passed"]
    assert not summary["trajectory_executed"]
    assert summary["primary_pilot_execution_manifest_authorized"]
    assert not summary["bounded_continuation_execution_authorized"]
    assert metrics["passed"]
    assert not metrics["trajectory_executed"]
    assert not metrics["nonlinear_root_solved"]
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == (
            digest
        )
