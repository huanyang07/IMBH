import hashlib
import json

import pytest

import run_causal_inner_entropy_stable_split_discretization_kernel_wp10c9d6c7c3b5c4f25fizzh1 as target


def test_manifest_authorizes_only_frozen_split_kernel():
    _, contract = target._validate_parent()
    assert contract["authorized_next"] == target.WORK_PACKAGE
    assert contract["kernel"]["physical_anchors"] == 47
    assert contract["kernel"]["second_order_gate"] == 1.8
    assert contract["kernel"]["trajectory_steps"] == 0


@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(), reason="kernel not run")
def test_canonical_split_certificate_is_fail_closed():
    summary = json.loads((target.CANONICAL_DIRECTORY / "summary.json").read_text())
    metrics = json.loads((target.CANONICAL_DIRECTORY / "kernel_metrics.json").read_text())
    assert summary["passed"] and summary["entropy_stable_split_discretization_certified"]
    assert not summary["trajectory_authorized"]
    assert metrics["passing_witness_count"] == 47
    assert metrics["minimum_matched_horizon_observed_order"] >= 1.8
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((target.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
