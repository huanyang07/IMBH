from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "results/canonical/coupled_inner_outer_rank_prototype/summary.json"
)


def test_coupled_inner_outer_rank_prototype_passes_declared_gates() -> None:
    result = json.loads(SUMMARY.read_text())

    assert result["actual_interface_rg"] == 40.04153642035986
    assert result["n_inner"] == 96
    assert result["n_outer"] == 64
    assert result["unknown_count"] == result["residual_count"] == 388
    assert result["reached_full_coupling"]

    final = result["stages"][-1]
    assert final["mu"] == 1.0
    assert final["accepted"]
    assert final["maximum_residual"] < 1.0e-7
    assert max(abs(value) for value in final["continuity_residual"]) < 1.0e-10
    assert max(abs(value) for value in final["primitive_audits"].values()) < 0.01

    for audit in result["rank_audits"].values():
        assert audit["ranks_by_relative_threshold"]["1e-10"] == 388
        assert audit["preboundary_nullity"] == 2
        assert audit["interface_response_rank"] == 2
        assert audit["sonic_rank"] == 2
