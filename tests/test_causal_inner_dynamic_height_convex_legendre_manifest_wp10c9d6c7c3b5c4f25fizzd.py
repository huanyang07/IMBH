import hashlib
import json

import pytest

import run_causal_inner_dynamic_height_convex_legendre_manifest_wp10c9d6c7c3b5c4f25fizzd as target


def test_contract_freezes_exact_physical_entropy_convexity_question():
    target._validate_parent()
    contract = target._contract()
    assert contract["diagnostic"]["same_frozen_physical_witnesses"] == 47
    assert contract["diagnostic"]["centered_Hessian_step_factors"] == [
        0.002,
        0.001,
        0.0005,
    ]
    assert contract["candidate_common_potential"]["density"] == (
        "rho=surface_mass**2/(2*Z_H)"
    )
    assert contract["decision"]["fail"]["authorized_next"] == target.FAILURE_NEXT
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


@pytest.mark.skipif(
    not target.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen"
)
def test_canonical_manifest_is_definitions_only_and_checksum_closed():
    summary = json.loads(
        (target.CANONICAL_DIRECTORY / "summary.json").read_text()
    )
    assert summary["passed"] and summary["definitions_only"]
    assert summary["fixed_height_potential_certified"]
    assert not summary["dynamic_height_potential_certified"]
    assert not summary["complete_cycle_execution_authorized"]
    for line in (
        target.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    ).read_text().splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256(
            (target.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest()
        assert actual == expected
