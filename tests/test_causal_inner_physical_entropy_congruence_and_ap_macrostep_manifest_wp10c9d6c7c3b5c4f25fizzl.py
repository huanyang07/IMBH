import hashlib
import json

import pytest

import run_causal_inner_physical_entropy_congruence_and_ap_macrostep_manifest_wp10c9d6c7c3b5c4f25fizzl as target


def test_contract_repairs_coordinate_map_and_freezes_cycle_cost():
    target._validate_parent(); contract = target._contract()
    assert contract["diagnosis"]["old_algebraic_kernel_preserved"]
    assert "Kerr-Schild" in contract["diagnosis"]["old_coordinate_map_status"]
    assert contract["physical_congruence"]["primitive_step"] == 3e-4
    assert contract["AP_macrostep"]["mode_policy"].startswith("retain neutral")
    assert contract["cycle_cost_contract"]["maximum_online_macrosteps"] == 100000
    assert contract["cycle_cost_contract"]["online_truth_residual_calls"] == 0
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_manifest_is_definitions_only():
    summary = json.loads((target.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert not summary["physical_entropy_congruence_certified"]
    assert not summary["AP_macrostep_certified"]
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((target.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest() == expected
