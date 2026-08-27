import json

import numpy as np
import pytest

import run_causal_inner_cycle_physical_input_bundle_schema_and_validator_certificate_wp10c9d6c7c3b5c4f25fizzt1 as runner


def test_parent_manifest_is_hash_locked_and_external_inputs_remain_missing():
    hashes, contract = runner._validate_parent()
    assert "physical_input_acquisition_contract.json" in hashes
    assert all(contract["current_missing_payloads"].values())


def test_synthetic_fixture_covers_all_canonical_bundle_parts():
    metadata, driver, branch, events, heldout, conservation = runner._synthetic_bundle()
    assert metadata["synthetic_fixture"] and not metadata["physical_model_complete"]
    assert driver["slow_forcing1232_per_second"].shape == (9, 3, 3, 1232)
    assert branch["radial_matrices112x11x11"].shape == (10, 112, 11, 11)
    assert events["pre_states1232"].shape == (8, 1232)
    assert heldout["withheld_phase_windows"].shape == (2, 2)
    assert conservation.shape == (4, 1232)


def test_certificate_passes_structure_but_not_physical_use():
    metrics, _metadata, _fixture = runner._certificate()
    assert metrics["passed"] and metrics["structurally_passed"]
    assert not metrics["physically_usable"]
    assert metrics["synthetic_fixture_rejected_when_physical_required"]
    assert metrics["bundle_roundtrip_bitwise"]
    assert not metrics["complete_cycle_execution_authorized"]
    assert metrics["complete_cycle_steps"] == 0


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="certificate not executed")
def test_canonical_validator_certificate_closes():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["input_schema_and_validator_certified"]
    assert summary["synthetic_fixture_only"]
    assert not summary["physical_model_complete"]
    assert not summary["physical_payloads_acquired"]
    assert not summary["complete_cycle_execution_authorized"]
    with np.load(
        runner.CANONICAL_DIRECTORY / "synthetic_fixture_arrays.npz", allow_pickle=False
    ) as payload:
        assert payload["conservation_map4x1232"].shape == (4, 1232)
