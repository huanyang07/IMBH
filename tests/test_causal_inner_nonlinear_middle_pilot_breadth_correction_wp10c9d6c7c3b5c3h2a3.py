from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_middle_pilot_breadth_correction_wp10c9d6c7c3b5c3h2a3 as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_historical_pilot_scope_is_corrected_explicitly() -> None:
    summary = _read(runner.SUMMARY_PATH)
    audit = summary["audit"]
    assert summary["pilot_generic_science_and_cost_result_retained"] is True
    assert summary["pilot_five_profile_wording_superseded"] is True
    assert audit["historical_pilot_tangent_direction_count"] == 1
    assert audit["corrected_profile_count"] == 5


def test_five_profile_restart_directions_are_committed() -> None:
    with np.load(runner.DECISIVE_ARRAYS, allow_pickle=False) as payload:
        assert payload["state_directions"].shape == (7, 5, 112, 5)
        assert payload["Tier_I_export_directions"].shape == (7, 5, 13)
        assert payload["primitive_history_directions"].shape == (7, 5, 112, 5)
        assert payload["mapped_history_directions"].shape == (7, 5, 112, 5)
        assert payload["height_history_directions"].shape == (7, 5, 112, 5)
        assert tuple(payload["profile_names"].tolist()) == runner.PROFILES


def test_generic_column_closes_and_all_tangent_gates_pass() -> None:
    audit = _read(runner.SUMMARY_PATH)["audit"]
    assert audit["generic_state_relative_closure_defect"] <= 1.0e-12
    assert audit["generic_Tier_I_relative_closure_defect"] <= 1.0e-12
    assert audit["maximum_step_matrix_jvp_relative_defect"] <= 1.0e-6
    assert audit["maximum_linear_solve_relative_defect"] <= 1.0e-10
    assert audit["maximum_matrix_component_closure_defect"] <= 1.0e-12
    assert audit["maximum_incoming_excision_characteristics"] == 0
    assert audit["maximum_export_active_prefix_ledger_defect"] <= 1.0e-12
    assert audit["maximum_export_transport_telescoping_defect"] <= 1.0e-12


def test_cost_conclusion_and_authorizations_are_narrow() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["middle_1ms_propagation_authorized"] is True
    assert summary["middle_2ms_propagation_authorized"] is False
    assert summary["fine_cost_bounded_propagation_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert summary["audit"]["routine_five_profile_block_step_median_wall_seconds"] < 0.1


def test_canonical_hashes_close() -> None:
    lines = (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines()
    assert len(lines) == 4
    for line in lines:
        digest, name = line.split("  ", 1)
        payload = (runner.CANONICAL_DIRECTORY / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
