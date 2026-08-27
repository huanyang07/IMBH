import json

import numpy as np
import pytest

import run_causal_inner_prefix_port_payload_and_boundary_structure_certificate_wp10c9d6c7c3b5c4f25fizzr1 as runner


def test_parent_manifest_is_hash_locked_and_does_not_supply_missing_forcing():
    hashes, contract = runner._validate_parent()
    assert "summary.json" in hashes
    assert contract["prefix_port_batch"]["candidate_anchor_count"] == 913
    assert not contract["prefix_port_batch"]["slow_forcing_b_included"]


def test_boundary_sign_contract_is_outward_normal_and_fail_closed():
    _, contract = runner._validate_parent()
    boundary = contract["boundary_lift"]
    assert boundary["incoming_definition"] == "negative eigenvalues of outward-normal A_n"
    assert boundary["inner"]["expected_incoming_count"] == 0
    assert boundary["outer"]["expected_incoming_count"] == 11
    assert not boundary["outer"]["cycle_wide_loading_complete"]


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="certificate not executed")
def test_canonical_prefix_ports_and_boundaries_close_without_cycle_claim():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    metrics = json.loads(
        (runner.CANONICAL_DIRECTORY / "port_and_boundary_metrics.json").read_text()
    )
    assert summary["passed"] and summary["prefix_port_payloads_built"]
    assert summary["eleven_field_boundary_structure_certified"]
    assert metrics["candidate_anchor_count"] == 913
    assert metrics["inner_incoming_counts"] == [0]
    assert metrics["outer_incoming_counts"] == [11]
    assert metrics["all_boundary_audits_passed"]
    assert not summary["outer_cycle_loading_complete"]
    assert not summary["complete_cycle_execution_authorized"]
    assert summary["complete_cycle_steps"] == 0
    with np.load(
        runner.CANONICAL_DIRECTORY / "prefix_port_payloads.npz", allow_pickle=False
    ) as payload:
        assert payload["corrected_radial_matrices11x11"].shape == (913, 11, 11)
        assert payload["source_matrices11x11"].shape == (913, 11, 11)
        assert payload["boundary_incoming_projectors"].shape == (19, 11, 11)
