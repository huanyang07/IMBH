from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_hybrid_phase_memory_architecture_manifest_wp10c9d6c7c3b5c4f25e0 as manifest


def test_dynamic_phase_replaces_algebraic_slaving() -> None:
    hypotheses = manifest._contract()["hypotheses"]
    assert hypotheses["memoryless_critical_graph"].endswith("=0")
    assert "s" in hypotheses["dynamic_phase_tube"]
    assert "r_perp" in hypotheses["reduced_dynamics"]["normal_defect"]


def test_online_contract_is_truth_free_and_cycle_feasible() -> None:
    online = manifest._contract()["online_contract"]
    assert online["state"] == "(q_in_R82,s_scalar,mode_discrete)"
    assert online["online_truth_calls"] == 0
    assert online["online_470_roots"] == 0
    assert online["minimum_average_macrostep_seconds"] >= 5.0


def test_missing_cycle_truth_remains_binding() -> None:
    scope = manifest._contract()["evidence_scope"]
    assert not scope["complete_cycle_truth_available"]
    assert not scope["hot_exit_truth_available"]
    assert not scope["reduced_cycle_authorized"]
