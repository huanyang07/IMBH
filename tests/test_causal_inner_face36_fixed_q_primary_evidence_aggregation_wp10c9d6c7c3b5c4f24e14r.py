from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / (
    "run_causal_inner_face36_fixed_q_primary_evidence_aggregation_"
    "wp10c9d6c7c3b5c4f24e14r.py"
)
SPEC = importlib.util.spec_from_file_location("e14r_aggregation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _parents():
    return (
        MODULE._read(MODULE.RETRY_DIRECTORY / "summary.json"),
        MODULE._read(MODULE.RETRY_DIRECTORY / "metrics.json"),
        MODULE._read(MODULE.DIAGNOSIS_DIRECTORY / "summary.json"),
        MODULE._read(MODULE.DIAGNOSIS_DIRECTORY / "metrics.json"),
        MODULE._read(MODULE.POLICY_DIRECTORY / "summary.json"),
        MODULE._read(MODULE.POLICY_DIRECTORY / "metrics.json"),
    )


def test_committed_parent_evidence_passes_aggregation_logic():
    parents = _parents()
    contract = {
        "binding_primary_evidence": {
            "accepted_main_BDF2_roots": 4,
            "accepted_main_horizon_seconds": 4.0e-7,
            "maximum_warm_to_cold_wall_time_ratio": 0.75,
            "maximum_warm_to_cold_residual_evaluation_ratio": 0.75,
            "certified_control_comparison_residual": 1.0e-12,
            "maximum_scaled_state_difference": 1.0e-8,
            "maximum_reaction_action_relative_difference": 1.0e-8,
        }
    }
    result = MODULE._evaluate_evidence(*parents, contract)
    assert result["passed"] is True
    assert all(result["gates"].values())


def test_action_equivalence_gate_fails_closed():
    parents = list(_parents())
    policy_metrics = copy.deepcopy(parents[5])
    policy_metrics["polished_to_warm_comparison"][
        "reaction_action_relative_defect"
    ] = 1.01e-8
    parents[5] = policy_metrics
    contract = {
        "binding_primary_evidence": {
            "accepted_main_BDF2_roots": 4,
            "accepted_main_horizon_seconds": 4.0e-7,
            "maximum_warm_to_cold_wall_time_ratio": 0.75,
            "maximum_warm_to_cold_residual_evaluation_ratio": 0.75,
            "certified_control_comparison_residual": 1.0e-12,
            "maximum_scaled_state_difference": 1.0e-8,
            "maximum_reaction_action_relative_difference": 1.0e-8,
        }
    }
    result = MODULE._evaluate_evidence(*parents, contract)
    assert result["passed"] is False
    assert result["gates"]["certified_comparison_policy"] is False


def test_canonical_aggregation_when_available():
    path = MODULE.CANONICAL_DIRECTORY / "summary.json"
    if not path.exists():
        pytest.skip("prospective aggregation has not executed yet")
    summary = MODULE._read(path)
    assert summary["classification"] == (
        "primary_bounded_continuation_evidence_certified"
    )
    assert summary["passed"] is True
    assert summary["heldout_continuation_manifest_authorized"] is True
    assert summary["heldout_continuation_execution_authorized"] is False
    MODULE._checksums(MODULE.CANONICAL_DIRECTORY)
