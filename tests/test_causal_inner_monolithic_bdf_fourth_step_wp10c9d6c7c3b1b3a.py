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
    "run_causal_inner_monolithic_bdf_fourth_step_"
    "wp10c9d6c7c3b1b3a.py"
)
SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_bdf_fourth_step_"
    "wp10c9d6c7c3b1b3a/summary.json"
)
CHECKSUMS = SUMMARY.parent / "SHA256SUMS.txt"


def _runner():
    spec = importlib.util.spec_from_file_location("c3b1b3a_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fourth_step_contract_preserves_full_frozen_matrix():
    module = _runner()
    assert module.LAYOUTS == tuple(module.c3b1b2a.LAYOUTS)
    assert len(module.LAYOUTS) == 3
    assert len(module._case_sequence()) == 16
    assert len(set(module._case_sequence())) == 16
    assert module.MAXIMUM_SCALED_RESIDUAL == 1.0e-10


def test_fourth_step_parent_is_full_bitwise_replay_certificate():
    module = _runner()
    parent = json.loads(
        (module.PARENT_DIRECTORY / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert parent["passed"]
    assert parent["all_cases_bitwise_equal"]
    assert parent["fourth_step_depth_authorized"]


def test_canonical_fourth_step_evidence_if_present():
    if not SUMMARY.exists():
        pytest.skip("canonical c3b1b3a evidence has not been generated")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c7c3b1b3a"
    assert summary["expected_case_count"] == 48
    assert summary["completed_case_count"] <= 48
    if summary["passed"]:
        assert summary["completed_case_count"] == 48
        assert summary["full_four_step_method_preflight_certified"]
        assert summary["all_final_checkpoint_roundtrips_bitwise"]
        assert summary["nonlinear_spatial_export_manifest_authorized"]
        assert (
            summary["classification"]
            == "full_profile_variant_four_step_monolithic_bdf_method_"
            "preflight_certified_nonlinear_spatial_export_manifest_"
            "authorized"
        )
    else:
        assert not summary["nonlinear_spatial_export_manifest_authorized"]
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_canonical_fourth_step_checksums_if_present():
    if not CHECKSUMS.exists():
        pytest.skip("canonical c3b1b3a checksums have not been generated")
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha256(CHECKSUMS.parent / filename) == digest
