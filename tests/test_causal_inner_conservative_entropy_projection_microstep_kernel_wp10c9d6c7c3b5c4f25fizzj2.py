import hashlib
import json

import numpy as np
import pytest

import run_causal_inner_conservative_entropy_projection_microstep_kernel_wp10c9d6c7c3b5c4f25fizzj2 as target


def test_manifest_authorizes_only_bounded_projection_kernel():
    _, contract = target._validate_parent()
    assert contract["authorized_next"] == target.WORK_PACKAGE
    assert contract["kernel"]["selected_witness_indices"] == [0, 10, 20, 30, 40, 46]
    assert contract["kernel"]["patches_per_witness"] == 2
    assert contract["kernel"]["step_halving_order_gate"] == 1.8
    assert not contract["kernel"]["trajectory_authorized"]


def test_frozen_patch_patterns_are_zero_mean_and_within_trust_region():
    patterns = np.asarray(target.PATCH_PATTERNS)
    assert patterns.shape == (2, 3, 4)
    assert np.max(np.abs(np.sum(patterns, axis=1))) <= 1.0e-18
    normalized = patterns / np.asarray((0.01, 0.01, 0.002, 0.002))
    assert np.max(np.abs(normalized)) <= 1.0


@pytest.mark.skipif(
    not target.CANONICAL_DIRECTORY.exists(), reason="kernel not run"
)
def test_canonical_projection_microstep_certificate():
    summary = json.loads(
        (target.CANONICAL_DIRECTORY / "summary.json").read_text()
    )
    metrics = json.loads(
        (target.CANONICAL_DIRECTORY / "kernel_metrics.json").read_text()
    )
    assert summary["passed"] == metrics["passed"]
    assert summary["prior_rejections_preserved"]
    assert not summary["trajectory_authorized"]
    if metrics["passed"]:
        assert metrics["passing_patch_count"] == 12
        assert metrics["minimum_matched_step_halving_order"] >= 1.8
        assert metrics["maximum_recovery_residual"] <= 1.0e-11
        assert metrics["maximum_conservation_relative_defect"] <= 2.0e-12
        assert metrics["maximum_entropy_relative_defect"] <= 2.0e-11
        assert metrics["maximum_projection_correction_relative_norm"] <= 0.05
    for line in (
        target.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    ).read_text().splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256(
            (target.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest()
        assert actual == expected
