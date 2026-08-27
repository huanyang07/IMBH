import hashlib
import json

import pytest

import run_causal_inner_local_nonlinear_atlas_trust_region_manifest_wp10c9d6c7c3b5c4f25fizzi as target


def test_contract_freezes_physical_path_and_STF_connection():
    target._validate_parent(); contract = target._contract()
    assert contract["nonlinear_equilibrium_interface"]["tadmor_identity"].startswith("Delta v")
    assert contract["moving_five_STF_connection"]["connection"].startswith("orthogonal polar")
    assert contract["trust_region"]["path_nodes_must_remain_physical"]
    assert contract["kernel"]["trajectory_steps"] == 0
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_manifest_is_definitions_only():
    summary = json.loads((target.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert not summary["nonlinear_atlas_trust_region_certified"]
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((target.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest() == expected
