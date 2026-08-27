import hashlib
import json

import pytest

import run_causal_inner_physical_entropy_congruence_and_ap_kernel_wp10c9d6c7c3b5c4f25fizzl1 as target


def test_manifest_authorizes_only_congruence_and_ap_kernel():
    _, contract = target._validate_parent()
    assert contract["authorized_next"] == target.WORK_PACKAGE
    assert contract["physical_congruence"]["witness_indices"] == [0, 10, 20, 30, 40, 46]
    assert contract["cycle_cost_contract"]["online_truth_residual_calls"] == 0
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(), reason="kernel not run")
def test_canonical_congruence_and_ap_certificate():
    summary = json.loads((target.CANONICAL_DIRECTORY / "summary.json").read_text()); metrics = json.loads((target.CANONICAL_DIRECTORY / "kernel_metrics.json").read_text())
    assert summary["passed"] == metrics["passed"]
    assert summary["old_algebraic_kernel_preserved_as_rest_fixture"]
    assert not summary["complete_cycle_execution_authorized"]
    if metrics["passed"]:
        assert metrics["passing_witness_count"] == 6
        assert metrics["maximum_whitened_symmetry_relative_defect"] <= 2e-6
        assert metrics["maximum_Valencia_spectrum_absolute_defect"] <= 2e-6
        assert metrics["maximum_core_reconstruction_relative_defect"] <= 2e-6
        assert metrics["maximum_AP_semigroup_expansivity"] <= 1e-10
        assert metrics["maximum_AP_composition_defect"] <= 2e-11
        assert metrics["maximum_AP_stiff_limit_defect"] <= 2e-8
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1); assert hashlib.sha256((target.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest() == expected
