import hashlib
import json

import pytest

import run_causal_inner_dynamic_height_physical_entropy_convexity_diagnostic_wp10c9d6c7c3b5c4f25fizzd1 as target


def test_frozen_manifest_authorizes_exact_height_diagnostic():
    _, contract = target._validate_parent()
    assert contract["authorized_next"] == target.WORK_PACKAGE
    assert contract["candidate_common_potential"]["height_affinity"] == (
        "eta_H=d eta/d Z_H"
    )


@pytest.mark.skipif(
    not target.CANONICAL_DIRECTORY.exists(), reason="diagnostic not run"
)
def test_canonical_rejection_is_failure_aware_and_checksum_closed():
    summary = json.loads(
        (target.CANONICAL_DIRECTORY / "summary.json").read_text()
    )
    metrics = json.loads(
        (target.CANONICAL_DIRECTORY / "diagnostic_metrics.json").read_text()
    )
    assert summary["audit_completed"]
    assert not summary["passed"]
    assert summary["fixed_height_physical_potential_preserved"]
    assert summary["split_architecture_manifest_authorized"]
    assert not summary["complete_cycle_execution_authorized"]
    assert metrics["convex_witness_count"] < metrics["physical_witness_count"]
    assert metrics["global_minimum_equilibrated_entropy_Hessian_eigenvalue"] < 0.0
    for line in (
        target.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    ).read_text().splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256(
            (target.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest()
        assert actual == expected
