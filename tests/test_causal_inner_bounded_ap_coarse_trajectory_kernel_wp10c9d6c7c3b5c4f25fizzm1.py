import json

import numpy as np
import pytest

import run_causal_inner_bounded_ap_coarse_trajectory_kernel_wp10c9d6c7c3b5c4f25fizzm1 as runner


def test_wave_number_is_smooth_positive_and_nonconstant():
    values = [runner._wave_number(time, 2.0) for time in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert min(values) > 0.0
    assert np.ptp(values) > 0.09


def test_parent_manifest_is_hash_locked():
    hashes, contract = runner._validate_parent()
    assert "summary.json" in hashes
    assert contract["offline_physical_atlas"]["online_truth_calls"] == 0
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="kernel not executed")
def test_canonical_kernel_closes_and_remains_precycle():
    hashes = runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    metrics = json.loads((runner.CANONICAL_DIRECTORY / "trajectory_metrics.json").read_text())
    assert hashes
    assert summary["passed"] and summary["bounded_AP_coarse_trajectory_certified"]
    assert metrics["online_truth_calls"] == 0
    assert metrics["all_checkpoints_bitwise"] and metrics["all_suffix_replays_bitwise"]
    assert not summary["complete_cycle_execution_authorized"]
