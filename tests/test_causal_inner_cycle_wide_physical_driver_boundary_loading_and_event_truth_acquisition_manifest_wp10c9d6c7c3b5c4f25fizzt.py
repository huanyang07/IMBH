import json

import pytest

import run_causal_inner_cycle_wide_physical_driver_boundary_loading_and_event_truth_acquisition_manifest_wp10c9d6c7c3b5c4f25fizzt as runner


def test_parent_reset_structure_is_locked_but_not_physical_calibration():
    hashes, summary, metrics = runner._validate_parent()
    assert "reset_and_guard_arrays.npz" in hashes
    assert summary["reset_and_guard_structure_certified"]
    assert not summary["events_and_resets_physically_calibrated"]
    assert metrics["complete_cycle_steps"] == 0


def test_fiducial_period_is_not_promoted_to_physical_input():
    contract = runner._contract()
    boundary = contract["scientific_boundary"]
    assert boundary["fiducial_cycle_seconds"] == 578880.0
    assert "runtime target only" in boundary["fiducial_period_role"]
    assert not contract["required_external_physical_model"][
        "currently_present_in_repository_as_cycle_wide_binding_data"
    ]


def test_acquisition_separates_driver_branch_event_and_holdout_truth():
    contract = runner._contract()
    bundle = contract["canonical_input_bundle"]
    assert set(("driver_npz", "branch_npz", "events_npz", "heldout_truth_npz")) <= set(bundle)
    assert contract["driver_and_boundary_gates"]["positive_phase_rate"]
    assert contract["event_truth_acquisition"]["online_event_truth_calls"] == 0
    assert "before the first coefficient fit" in contract["prospective_holdouts"]["leakage"]


def test_manifest_stops_before_physical_acquisition_and_cycle_execution():
    contract = runner._contract()
    missing = contract["current_missing_payloads"]
    assert all(missing.values())
    boundary = contract["cost_and_execution_boundary"]
    assert not boundary["complete_cycle_runner_may_exist_in_this_package"]
    assert boundary["complete_cycle_steps"] == 0
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_manifest_closes():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert not summary["physical_model_complete"]
    assert not summary["physical_payloads_acquired"]
    assert not summary["complete_cycle_execution_authorized"]
    assert summary["complete_cycle_steps"] == 0
