from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_coordinate_chart_conditioning_diagnosis_manifest_wp10c9d6c7c3b5c4f25fia as target  # noqa: E402


def test_parent_negative_certificate_is_locked() -> None:
    lock = target._validate_parent(require_clean=False)
    assert lock["summary"]["classification"] == target.parent.PHYSICAL_CLASSIFICATION
    assert lock["metrics"]["gate_values"]["terminal_elapsed_seconds"] == (
        0.11125000000000008
    )


def test_saved_witnesses_reproduce_terminal_crossing() -> None:
    metrics, arrays = target._lock_witnesses()
    np.testing.assert_array_equal(arrays["attempt_indices"], target.WITNESS_ATTEMPTS)
    np.testing.assert_array_equal(arrays["accepted"], target.EXPECTED_ACCEPTED)
    assert arrays["primitive_states"].shape == (6, 112, 5)
    assert arrays["scaled_free_rates560_per_s"].shape == (6, 560)
    assert metrics["saved_condition_numbers"][-2] < target.RAW_CONDITION_GATE
    assert metrics["saved_condition_numbers"][-1] > target.RAW_CONDITION_GATE
    assert metrics["all_nonconditioning_physical_gates_passed"]


def test_contract_is_nonpropagating_and_fail_closed() -> None:
    contract = target._contract()
    assert contract["scope"]["new_exact_coordinate_jacobians"] == 6
    assert contract["scope"]["new_exact_free_field_calls"] == 0
    assert contract["scope"]["new_retractions"] == 0
    assert contract["scope"]["new_trajectory_segments"] == 0
    assert contract["coordinate_blocks"]["total_rows"] == 470
    assert "relax the historical raw condition gate" in contract["forbidden"]


def test_canonical_manifest_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        target.CANONICAL_DIRECTORY / "diagnosis_contract.json"
    )
    assert summary["classification"] == target.CLASSIFICATION
    assert summary["definitions_only"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
    assert contract["authorized_execution"] == target.AUTHORIZED_NEXT
