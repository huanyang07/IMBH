import hashlib
import json

import pytest

import run_causal_inner_short_restartable_equilibrium_core_trajectory_wp10c9d6c7c3b5c4f25fizzk1 as target


def test_manifest_authorizes_only_two_case_core_trajectory():
    _, contract = target._validate_parent()
    assert contract["authorized_next"] == target.WORK_PACKAGE
    assert contract["cases"]["dimensionless_horizon"] == 0.32
    assert target.CASES == (("primary", 0, 0), ("held_out", 30, 1))
    assert not contract["claim_boundary"]["full_eleven_field_trajectory_certified"]
    assert not contract["claim_boundary"]["physical_time_horizon_claimed"]


@pytest.mark.skipif(
    not target.CANONICAL_DIRECTORY.exists(), reason="trajectory not run"
)
def test_canonical_short_trajectory_certificate():
    summary = json.loads(
        (target.CANONICAL_DIRECTORY / "summary.json").read_text()
    )
    metrics = json.loads(
        (target.CANONICAL_DIRECTORY / "trajectory_metrics.json").read_text()
    )
    assert summary["passed"] == metrics["passed"]
    assert not summary["full_eleven_field_trajectory_certified"]
    assert not summary["physical_time_horizon_claimed"]
    assert not summary["complete_cycle_execution_authorized"]
    if metrics["passed"]:
        assert metrics["passing_case_count"] == 2
        assert metrics["minimum_matched_endpoint_order"] >= 1.8
        assert metrics["maximum_cumulative_conservation_relative_defect"] <= 2e-10
        assert metrics["maximum_cumulative_entropy_relative_defect"] <= 2e-10
        assert metrics["maximum_trust_radius_fraction"] <= 1.0
        assert metrics["all_checkpoint_roundtrips_bitwise"]
        assert metrics["all_suffix_replays_bitwise"]
    for line in (
        target.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    ).read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256(
            (target.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == expected
