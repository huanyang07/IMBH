from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_middle_tangent_hardening_audit_wp10c9d6c7c3b5c3h2a1 as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_all_profile_middle_short_step_passes_relative_surrogate_gates() -> None:
    summary = _read(runner.SUMMARY_PATH)
    audit = summary["audit"]
    gates = summary["gates"]
    assert summary["passed"] is True
    assert tuple(audit["profiles"]) == runner.PROFILES
    for channel in ("state", "instantaneous_Tier_I"):
        report = audit[channel]
        assert report["discrepancy_fraction_of_observable_response"] <= gates[
            "maximum_discrepancy_fraction_of_observable_response"
        ]
    assert audit["state"]["maximum_scaled_discrepancy"] <= gates[
        "maximum_absolute_scaled_state_discrepancy"
    ]
    assert audit["instantaneous_Tier_I"]["maximum_scaled_discrepancy"] <= gates[
        "maximum_absolute_scaled_Tier_I_discrepancy"
    ]


def test_analytic_history_and_variable_step_ratio_audits_pass() -> None:
    summary = _read(runner.SUMMARY_PATH)
    audit = summary["audit"]
    gates = summary["gates"]
    assert audit["mapped_history_relative_defect"] <= gates[
        "maximum_analytic_to_centered_history_relative_defect"
    ]
    assert audit["responsive_height_history_relative_defect"] <= gates[
        "maximum_analytic_to_centered_history_relative_defect"
    ]
    assert tuple(float(key) for key in audit["ratio_audits"]) == (0.5, 1.0, 2.0)
    for report in audit["ratio_audits"].values():
        assert report["maximum_step_matrix_jvp_relative_defect"] <= gates[
            "maximum_internal_discrete_residual_jvp_relative_defect"
        ]
        assert report["maximum_linear_solve_relative_defect"] <= gates[
            "maximum_linear_solve_relative_defect"
        ]
        assert report["incoming_excision_characteristics"] == 0


def test_only_middle_cost_pilot_is_authorized() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["new_physical_trajectory_executed"] is False
    assert summary["middle_0p2ms_cost_pilot_authorized"] is True
    assert summary["middle_1ms_propagation_authorized"] is False
    assert summary["fine_cost_bounded_propagation_authorized"] is False
    assert summary["third_duration_rung_spatial_convergence_certified"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        payload = (runner.CANONICAL_DIRECTORY / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
