import hashlib
import json

import pytest

import run_causal_inner_split_godunov_port_hamiltonian_proof_kernel_wp10c9d6c7c3b5c4f25fizze1 as target


def test_manifest_authorizes_only_split_proof_kernel():
    _, contract = target._validate_parent()
    assert contract["authorized_next"] == target.WORK_PACKAGE
    assert contract["proof_kernel"]["same_47_physical_witnesses"]
    assert contract["proof_kernel"]["trajectory_steps"] == 0


@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(), reason="proof not run")
def test_canonical_split_proof_stops_before_physical_shear():
    summary = json.loads((target.CANONICAL_DIRECTORY / "summary.json").read_text())
    metrics = json.loads((target.CANONICAL_DIRECTORY / "proof_metrics.json").read_text())
    assert summary["passed"] and summary["split_kernel_certified"]
    assert summary["one_piece_height_rejection_preserved"]
    assert not summary["full_shear_physical_potential_certified"]
    assert not summary["complete_cycle_execution_authorized"]
    assert metrics["passing_witness_count"] == 47
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((target.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest() == expected
