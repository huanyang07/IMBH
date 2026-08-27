import json

import numpy as np
import pytest

import run_causal_inner_conservative_entropy_reset_and_guard_localization_structure_certificate_wp10c9d6c7c3b5c4f25fizzs1 as runner


def test_parent_and_prefix_payloads_are_hash_locked():
    parent_hashes, prefix_hashes, contract = runner._validate_parent()
    assert "summary.json" in parent_hashes
    assert "prefix_port_payloads.npz" in prefix_hashes
    assert contract["claim_boundary"]["complete_cycle_steps"] == 0
    assert not contract["required_physical_inputs"]["impact_guard_and_reset_truth"]


def test_native_ledger_map_has_four_rows_and_1232_columns():
    positions, cells, scales, roots = runner._profile_zero_payload()
    measures = np.linspace(1.0, 2.0, 112)
    physical, scaled, row_scales, weights, normalized = runner._assemble_native_ledger_map(
        measures, scales, roots
    )
    assert len(positions) == 112
    np.testing.assert_array_equal(cells, np.arange(112))
    assert physical.shape == scaled.shape == (4, 1232)
    assert row_scales.shape == (4,)
    assert weights.shape == (1232,)
    assert normalized.shape == (112,)
    np.testing.assert_allclose(np.linalg.norm(scaled, axis=1), 1.0, atol=2e-15)


def test_guard_structure_is_full_half_and_checkpoint_consistent():
    metrics, arrays = runner._guard_structure()
    assert metrics["orientation"] == "negative_to_positive"
    assert metrics["event_time_absolute_defect"] <= 2e-12
    assert metrics["event_state_infinity_defect"] <= 2e-12
    assert metrics["full_to_half_event_time_absolute_defect"] <= 2e-12
    assert metrics["full_to_half_event_state_infinity_defect"] <= 2e-12
    assert metrics["checkpoint_replay_bitwise"]
    assert arrays["guard_event_state"].shape == (1232,)


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="certificate not executed")
def test_canonical_certificate_closes_without_physical_event_claim():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    metrics = json.loads(
        (runner.CANONICAL_DIRECTORY / "reset_and_guard_metrics.json").read_text()
    )
    assert summary["passed"] and summary["reset_and_guard_structure_certified"]
    assert metrics["all_reset_audits_passed"]
    assert metrics["reset_checkpoint_replay_bitwise"]
    assert not summary["events_and_resets_physically_calibrated"]
    assert not summary["complete_cycle_execution_authorized"]
    assert summary["complete_cycle_steps"] == 0
    with np.load(
        runner.CANONICAL_DIRECTORY / "reset_and_guard_arrays.npz", allow_pickle=False
    ) as payload:
        assert payload["scaled_conservation_map4x1232"].shape == (4, 1232)
        assert payload["minimum_norm_normal1232x4"].shape == (1232, 4)
