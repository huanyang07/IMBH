from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_arclength_segment_manifest_wp10c9d6c7c3b5c4f25f4 as target


def test_transport_parent_passes_and_authorizes_manifest() -> None:
    lock = target._validate_parent(require_clean=False)
    assert set(lock) == {"transport_hashes", "window_05_hashes"}


def test_contract_freezes_first_truth_segment() -> None:
    contract = target._contract(target._validate_parent(require_clean=False))
    assert contract["phase_system"]["node_count"] == 5
    assert contract["phase_system"]["arclength_span"] == 2.5e-2
    assert contract["seed"]["fixed_time_window_06_is_not_executed"]
    assert contract["cost_boundary"]["new_nonlinear_fixed_Q_roots_equal"] == 0


def test_canonical_manifest_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    assert summary["passed"]
    assert summary["arclength_segment_execution_authorized"]
    assert not summary["fixed_time_window_06_authorized"]
