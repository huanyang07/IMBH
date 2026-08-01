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
    "run_causal_inner_monolithic_bdf_step3_replay_"
    "wp10c9d6c7c3b1b2b2.py"
)
SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_bdf_step3_replay_"
    "wp10c9d6c7c3b1b2b2/summary.json"
)
CHECKSUMS = SUMMARY.parent / "SHA256SUMS.txt"


def _runner():
    spec = importlib.util.spec_from_file_location("c3b1b2b2_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_step3_replay_contract_preserves_full_frozen_matrix():
    module = _runner()
    assert module.LAYOUTS == tuple(module.c3b1b2a.LAYOUTS)
    assert len(module.LAYOUTS) == 3
    assert len(module._case_sequence()) == 16
    assert len(set(module._case_sequence())) == 16
    assert module.MAXIMUM_SCALED_RESIDUAL == 1.0e-10


def test_step3_replay_contract_compares_complete_temporal_state():
    module = _runner()
    assert module.REPLAY_FIELDS == (
        "step3_old_state",
        "step3_previous_primitive_increment",
        "step3_previous_mapped_storage_increment",
        "step3_previous_height_storage_increment",
        "step3_primitive_increment",
        "step3_final_state",
        "step3_mapped_storage_increment",
        "step3_height_storage_increment",
    )


def test_canonical_step3_replay_evidence_if_present():
    if not SUMMARY.exists():
        pytest.skip("canonical c3b1b2b2 evidence has not been generated")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c7c3b1b2b2"
    assert summary["expected_case_count"] == 48
    assert summary["completed_case_count"] <= 48
    if summary["passed"]:
        assert summary["completed_case_count"] == 48
        assert summary["all_cases_bitwise_equal"]
        assert summary["maximum_bitwise_replay_absolute_difference"] == 0.0
        assert summary["direct_restarted_step3_bitwise_replay_certified"]
        assert summary["fourth_step_depth_authorized"]
        assert (
            summary["classification"]
            == "full_profile_variant_bdf2_step3_direct_restarted_bitwise_"
            "replay_certified_fourth_step_depth_authorized"
        )
    else:
        assert not summary["fourth_step_depth_authorized"]
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_canonical_step3_replay_checksums_if_present():
    if not CHECKSUMS.exists():
        pytest.skip("canonical c3b1b2b2 checksums have not been generated")
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha256(CHECKSUMS.parent / filename) == digest
