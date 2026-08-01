from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts/"
    "run_causal_inner_monolithic_bdf_step2_screen_"
    "wp10c9d6c7c3b1b2a.py"
)
SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_bdf_step2_screen_"
    "wp10c9d6c7c3b1b2a/summary.json"
)
CHECKSUMS = SUMMARY.parent / "SHA256SUMS.txt"


def _runner():
    spec = importlib.util.spec_from_file_location("c3b1b2a_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_step2_contract_preserves_full_frozen_matrix():
    module = _runner()
    assert module.LAYOUTS == tuple(module.c3b1b1.LAYOUTS)
    assert len(module.LAYOUTS) == 3
    assert len(module._case_sequence()) == 16
    assert len(set(module._case_sequence())) == 16
    assert module.TIMESTEP_SECONDS > 0.0
    assert module.MAXIMUM_SCALED_RESIDUAL == 1.0e-10


def test_reconstructed_history_uses_exact_parent_increment(monkeypatch):
    module = _runner()
    previous_old = np.zeros((2, 5))
    previous_increment = np.full((2, 5), 0.25)
    previous_new = previous_old + previous_increment
    parent_arrays = {
        "coarse__case__old_state": previous_old,
        "coarse__case__primitive_increment": previous_increment,
        "coarse__case__final_state": previous_new,
    }
    sentinel_storage = object()
    sentinel_history = object()
    monkeypatch.setattr(
        module,
        "causal_five_field_monolithic_storage_increment",
        lambda context, old, new: (
            sentinel_storage
            if np.array_equal(old, previous_old)
            and np.array_equal(new, previous_new)
            else None
        ),
    )
    monkeypatch.setattr(
        module,
        "causal_five_field_monolithic_bdf_history",
        lambda increment, storage, timestep: (
            sentinel_history
            if np.array_equal(increment, previous_increment)
            and storage is sentinel_storage
            and timestep == module.TIMESTEP_SECONDS
            else None
        ),
    )
    result = module._reconstruct_history(
        {"context": object()},
        parent_arrays,
        "coarse__case",
    )
    assert result is sentinel_history


def test_canonical_step2_evidence_if_present():
    if not SUMMARY.exists():
        pytest.skip("canonical c3b1b2a evidence has not been generated")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c7c3b1b2a"
    assert summary["expected_case_count"] == 48
    assert summary["completed_case_count"] <= 48
    if summary["passed"]:
        assert summary["completed_case_count"] == 48
        assert summary["restart_roundtrip_certified"]
        assert summary["all_restart_roundtrips_bitwise"]
        assert summary["split_replay_depth_authorized"]
        assert (
            summary["classification"]
            == "full_profile_variant_bdf2_step2_screen_certified_"
            "restart_replay_depth_authorized"
        )
    else:
        assert not summary["split_replay_depth_authorized"]
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_canonical_step2_checksums_if_present():
    if not CHECKSUMS.exists():
        pytest.skip("canonical c3b1b2a checksums have not been generated")
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha256(CHECKSUMS.parent / filename) == digest
