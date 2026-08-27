import json

import pytest

import run_causal_inner_reduced_hybrid_cycle_kernel_certificate_wp10c9d6c7c3b5c4f25fizzx1 as runner


def test_parent_cycle_kernel_manifest_is_hash_locked():
    assert "summary.json" in runner._validate_parent()[0]


def test_certificate_fixture_is_explicitly_nonphysical():
    kernel, _ = runner._fixture()
    assert kernel.metadata["synthetic_fixture"] is True
    assert kernel.require_physical is False


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="certificate not run")
def test_canonical_cycle_kernel_certificate_passes_without_authorizing_cycle():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    metrics = json.loads((runner.CANONICAL_DIRECTORY / "kernel_metrics.json").read_text())
    assert summary["passed"] and summary["reduced_hybrid_cycle_kernel_certified"]
    assert summary["synthetic_fixture_only"]
    assert not summary["physical_model_complete"]
    assert not summary["complete_cycle_execution_authorized"]
    assert metrics["complete_cycle_steps"] == 0
