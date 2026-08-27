import hashlib
import json

import pytest

import run_causal_inner_fully_split_physical_port_atlas_kernel_wp10c9d6c7c3b5c4f25fizzg1 as target


def test_manifest_authorizes_only_port_atlas_kernel():
    _, contract = target._validate_parent()
    assert contract["authorized_next"] == target.WORK_PACKAGE
    assert contract["architecture"]["full_tensor_no_projection"]
    assert contract["kernel"]["trajectory_steps"] == 0


@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(), reason="kernel not run")
def test_canonical_kernel_is_fail_closed_and_checksum_complete():
    summary = json.loads((target.CANONICAL_DIRECTORY / "summary.json").read_text())
    metrics = json.loads((target.CANONICAL_DIRECTORY / "kernel_metrics.json").read_text())
    assert summary["passed"] and summary["prior_rejections_preserved"]
    assert summary["fully_split_physical_port_atlas_kernel_certified"]
    assert not summary["trajectory_authorized"]
    assert metrics["physical_witness_count"] == 47
    assert metrics["passing_witness_count"] == 47
    assert metrics["maximum_full_shear_frame_constraint_defect"] <= 2e-13
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((target.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
