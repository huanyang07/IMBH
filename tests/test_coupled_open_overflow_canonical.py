from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "results/canonical/coupled_open_overflow_eigenvalue"


def test_open_overflow_eigenvalue_reaches_full_rank_open_control() -> None:
    summary = json.loads((CASE / "summary.json").read_text())
    assert summary["reached_open_boundary"]
    open_stage = summary["stages"][-1]
    assert open_stage["accepted"]
    assert abs(open_stage["mdot_inner_over_stream"] - 0.16903) < 5.0e-4
    assert abs(open_stage["overflow_fraction"] - 0.83097) < 5.0e-4
    assert open_stage["max_tidal_band_H_over_R"] < 0.05
    assert abs(open_stage["outer_torque_relative"]) < 1.0e-8
    for key in ("chi_0", "chi_1", "mesh_144_96"):
        audit = summary["rank_audits"][key]
        assert (
            audit["ranks_by_relative_threshold"]["1e-10"]
            == audit["jacobian_shape"][1]
        )
        assert audit["preboundary_nullity"] == 2
        assert audit["interface_response_rank"] == 2
        assert audit["sonic_rank"] == 2


def test_open_overflow_mesh_failure_selects_time_evolution() -> None:
    summary = json.loads((CASE / "summary.json").read_text())
    assert not summary["mesh_gate"]
    assert summary["next_stage"] == "coupled_mass_energy_time_evolution"
    accepted = summary["mesh_rows"][-2]
    rejected = summary["mesh_rows"][-1]
    assert (accepted["n_inner"], accepted["n_outer"]) == (144, 96)
    assert accepted["accepted"]
    assert (rejected["n_inner"], rejected["n_outer"]) == (168, 112)
    assert not rejected["accepted"]
    assert rejected["block_maximum_residuals"]["outer_stress"] > 0.1
    assert rejected["block_maximum_residuals"]["outer_energy"] > 0.1
