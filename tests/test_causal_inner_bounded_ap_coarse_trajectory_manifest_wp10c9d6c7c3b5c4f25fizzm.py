import json
from pathlib import Path

import pytest

import run_causal_inner_bounded_ap_coarse_trajectory_manifest_wp10c9d6c7c3b5c4f25fizzm as runner


def test_contract_freezes_bounded_ap_trajectory_without_cycle_authorization():
    contract = runner._contract()
    assert contract["trajectory"]["stiffness_ratios"] == [1.0, 100.0, 1000.0]
    assert contract["trajectory"]["step_counts"] == [8, 16, 32]
    assert contract["offline_physical_atlas"]["online_truth_calls"] == 0
    assert contract["gates"]["required_source_nullity"] == 4
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


def test_parent_certificate_is_hash_locked():
    hashes = runner._validate_parent()
    assert "summary.json" in hashes


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_manifest_closes():
    hashes = runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    contract = json.loads((runner.CANONICAL_DIRECTORY / "trajectory_contract.json").read_text())
    assert hashes
    assert summary["passed"] and summary["definitions_only"]
    assert summary["authorized_next"] == runner.AUTHORIZED_NEXT
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]
    assert Path(runner.REPORT_PATH).exists()
