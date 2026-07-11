from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "results/canonical/coupled_mesh_interface_certification"


def test_coupled_mesh_sequence_passes_convergence_and_rank_gates() -> None:
    result = json.loads((CASE / "mesh_summary.json").read_text())

    assert result["mesh_certification_gate"]
    assert result["finest_pair_relative_luminosity_shift"] < 0.01
    assert result["finest_pair_relative_max_H_over_R_shift"] < 0.02
    assert [(row["n_inner"], row["n_outer"]) for row in result["rows"]] == [
        (96, 64),
        (144, 96),
        (192, 128),
    ]
    for row in result["rows"]:
        assert row["accepted"]
        assert row["rank_gate"]
        assert row["primitive_gate"]
        assert row["maximum_residual"] < 1.0e-7


def test_coupled_interface_sequence_is_invariant_on_fixed_physical_band() -> None:
    result = json.loads((CASE / "interface_summary.json").read_text())

    assert result["interface_position_gate"]
    assert result["relative_composite_luminosity_spread"] < 0.01
    assert result["relative_max_common_band_H_over_R_spread"] < 0.02
    assert result["relative_max_outer_H_over_R_spread"] > 0.02
    actual = [row["actual_interface_rg"] for row in result["rows"]]
    assert actual == sorted(actual)
    for row in result["rows"]:
        assert row["accepted"]
        assert row["rank_gate"]
        assert row["primitive_gate"]
        assert row["rank_audit"]["ranks_by_relative_threshold"]["1e-10"] == 772
        assert row["rank_audit"]["interface_response_rank"] == 2
        assert row["rank_audit"]["sonic_rank"] == 2
        assert max(abs(value) for value in row["continuity_residual"]) < 1.0e-10
