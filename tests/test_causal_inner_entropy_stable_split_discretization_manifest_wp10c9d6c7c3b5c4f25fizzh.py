import hashlib
import json

import pytest

import run_causal_inner_entropy_stable_split_discretization_manifest_wp10c9d6c7c3b5c4f25fizzh as target


def test_contract_freezes_entropy_stable_composition_only():
    target._validate_parent()
    contract = target._contract()
    assert contract["spatial_discretization"]["interface_entropy_inequality"]
    assert contract["time_discretization"]["formal_order"] == 2
    assert contract["nonlinear_policy"]["rejection"] == "rejected states never enter history"
    assert contract["kernel"]["trajectory_steps"] == 0
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_manifest_is_definitions_only_and_checksum_complete():
    summary = json.loads((target.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert not summary["split_discretization_certified"]
    assert not summary["trajectory_authorized"]
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((target.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
