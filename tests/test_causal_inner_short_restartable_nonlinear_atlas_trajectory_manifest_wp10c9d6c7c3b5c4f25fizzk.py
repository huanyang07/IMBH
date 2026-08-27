import hashlib
import json

import pytest

import run_causal_inner_short_restartable_nonlinear_atlas_trajectory_manifest_wp10c9d6c7c3b5c4f25fizzk as target


def test_contract_is_short_restartable_and_does_not_overclaim_ports():
    target._validate_parent()
    contract = target._contract()
    assert contract["cases"]["dimensionless_horizon"] == 0.32
    assert [item["steps"] for item in contract["cases"]["matched_ladder"]] == [
        16,
        32,
        64,
    ]
    assert contract["restart"]["suffix_replay"].startswith("remaining 16")
    assert not contract["claim_boundary"]["full_eleven_field_trajectory_certified"]
    assert not contract["claim_boundary"]["physical_time_horizon_claimed"]
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


@pytest.mark.skipif(
    not target.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen"
)
def test_canonical_manifest_is_definitions_only():
    summary = json.loads(
        (target.CANONICAL_DIRECTORY / "summary.json").read_text()
    )
    assert summary["passed"] and summary["definitions_only"]
    assert not summary["equilibrium_core_trajectory_certified"]
    assert not summary["full_eleven_field_trajectory_certified"]
    assert not summary["complete_cycle_execution_authorized"]
    for line in (
        target.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    ).read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256(
            (target.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == expected
