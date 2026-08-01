from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts/"
    "run_causal_inner_monolithic_bdf_restarted_step3_"
    "wp10c9d6c7c3b1b2b1.py"
)
SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_bdf_restarted_step3_"
    "wp10c9d6c7c3b1b2b1/summary.json"
)
CHECKSUMS = SUMMARY.parent / "SHA256SUMS.txt"


def _runner():
    spec = importlib.util.spec_from_file_location("c3b1b2b1_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_restarted_step3_contract_preserves_full_frozen_matrix():
    module = _runner()
    assert module.LAYOUTS == tuple(module.c3b1b2a.LAYOUTS)
    assert len(module.LAYOUTS) == 3
    assert len(module._case_sequence()) == 16
    assert len(set(module._case_sequence())) == 16
    assert module.MAXIMUM_SCALED_RESIDUAL == 1.0e-10


def test_canonical_restarted_step3_evidence_if_present():
    if not SUMMARY.exists():
        pytest.skip("canonical c3b1b2b1 evidence has not been generated")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c7c3b1b2b1"
    assert summary["expected_case_count"] == 48
    assert summary["completed_case_count"] <= 48
    if summary["passed"]:
        assert summary["completed_case_count"] == 48
        assert summary["serialized_checkpoint_step3_depth_certified"]
        assert summary["direct_split_replay_comparison_authorized"]
        assert (
            summary["classification"]
            == "full_profile_variant_restarted_bdf2_step3_depth_certified_"
            "direct_split_replay_comparison_authorized"
        )
    else:
        assert not summary["direct_split_replay_comparison_authorized"]
    assert not summary["fourth_step_depth_authorized"]
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_canonical_restarted_step3_checksums_if_present():
    if not CHECKSUMS.exists():
        pytest.skip("canonical c3b1b2b1 checksums have not been generated")
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha256(CHECKSUMS.parent / filename) == digest
