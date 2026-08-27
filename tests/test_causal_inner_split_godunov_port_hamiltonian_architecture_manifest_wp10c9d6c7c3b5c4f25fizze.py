import hashlib
import json

import pytest

import run_causal_inner_split_godunov_port_hamiltonian_architecture_manifest_wp10c9d6c7c3b5c4f25fizze as target


def test_contract_preserves_field_count_and_energy_port():
    target._validate_parent()
    contract = target._contract()
    assert contract["field_decomposition"]["total_fields"] == 11
    assert contract["field_decomposition"]["Godunov_transport_fields"] == 9
    assert contract["field_decomposition"]["vertical_port_fields"] == 2
    assert contract["vertical_port_generator"]["reversible_operator"] == "J_H=-J_H^T"
    assert contract["composition"]["method"].startswith("symmetric")
    assert not contract["claim_boundary"]["cycle_execution_authorized"]


@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_manifest_is_checksum_closed():
    summary = json.loads((target.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert summary["one_piece_height_rejection_preserved"]
    assert not summary["split_kernel_certified"]
    assert not summary["complete_cycle_execution_authorized"]
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((target.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest() == expected
