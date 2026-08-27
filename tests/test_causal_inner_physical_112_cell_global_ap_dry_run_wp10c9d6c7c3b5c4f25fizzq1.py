import json

import pytest

import run_causal_inner_physical_112_cell_global_ap_dry_run_wp10c9d6c7c3b5c4f25fizzq1 as runner


def test_corrected_global_contract_is_112_by_11():
    _,contract=runner._validate_parent();spec=contract["physical_grid_global_AP_dry_run"]
    assert spec["radial_cells"]==112
    assert spec["fields_per_cell"]==11
    assert spec["global_state_dimension"]==1232
    assert spec["gates"]["online_truth_calls"]==0


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(),reason="certificate not executed")
def test_canonical_native_grid_certificate_closes():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary=json.loads((runner.CANONICAL_DIRECTORY/"summary.json").read_text())
    metrics=json.loads((runner.CANONICAL_DIRECTORY/"global_dry_run_metrics.json").read_text())
    assert summary["passed"] and summary["physical_112_cell_global_AP_certified"]
    assert summary["physical_context_cells"]==112
    assert summary["global_state_dimension"]==1232
    assert metrics["all_checkpoints_bitwise"] and metrics["all_suffix_replays_bitwise"]
    assert metrics["maximum_projected_100k_step_wall_days"]<=3.0
    assert summary["complete_cycle_steps"]==0
