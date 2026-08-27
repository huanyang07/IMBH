import json

import numpy as np
import pytest

import run_causal_inner_nonperiodic_native_global_ap_boundary_action_structure_certificate_wp10c9d6c7c3b5c4f25fizzu1 as runner


def test_parent_manifest_is_hash_locked_and_nonexecuting():
    _, contract = runner._validate_parent()
    assert contract["certificate"]["complete_cycle_steps"] == 0
    assert contract["authorized_next"] == runner.WORK_PACKAGE


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="certificate not executed")
def test_canonical_boundary_action_certificate_closes_fail_closed():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    metrics = json.loads(
        (runner.CANONICAL_DIRECTORY / "boundary_action_metrics.json").read_text()
    )
    audit = metrics["operator_audit"]
    assert summary["passed"] and summary["nonperiodic_global_AP_boundary_action_certified"]
    assert summary["pure_inner_excision_certified"]
    assert summary["eleven_characteristic_outer_affine_loading_certified"]
    assert audit["energy_identity_relative_defect"] <= 5e-12
    assert audit["inner_incoming_count"] == 0
    assert audit["outer_incoming_count"] == 11
    assert metrics["checkpoint_roundtrip_bitwise"] and metrics["suffix_replay_bitwise"]
    assert not summary["physical_model_complete"]
    assert not summary["complete_cycle_execution_authorized"]
    assert summary["complete_cycle_steps"] == 0
    with np.load(
        runner.CANONICAL_DIRECTORY / "boundary_action_arrays.npz", allow_pickle=False
    ) as payload:
        assert payload["outer_control_dense"].shape == (1232, 11)
        assert np.array_equal(
            payload["uninterrupted_suffix_state"], payload["replayed_suffix_state"]
        )
