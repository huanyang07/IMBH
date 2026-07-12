from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical/time_dae_flux_primary_prototype"


def _load(name: str):
    return json.loads((CANONICAL / name).read_text())


def test_flux_primary_canonical_rank_and_step_gates() -> None:
    rank = _load("flux_primary_rank.json")
    steps = _load("backward_euler_steps.json")
    endpoint = _load("endpoint_audit.json")

    assert not endpoint["Ninner168_Nouter112"]["accepted"]
    for row in rank["meshes"]:
        assert row["algebraic_equilibrated_rank"] == row["algebraic_dimension"]
        assert row["descriptor_equilibrated_rank"] == row["total_dimension"]
        assert row["maximum_radial_mach"] < 0.1
    accepted_by_mesh = {
        row["outer_cells"]: [step for step in row["steps"] if step["accepted"]]
        for row in steps["meshes"]
    }
    assert set(accepted_by_mesh) == {16, 32}
    for accepted in accepted_by_mesh.values():
        assert accepted
        best = min(accepted, key=lambda item: item["maximum_residual"])
        assert best["maximum_residual"] < 1.0e-8
        assert best["mass_defect"] < 1.0e-9
        assert best["angular_defect"] < 1.0e-9
        assert best["energy_defect"] < 1.0e-8


def test_flux_primary_canonical_provenance_is_scoped() -> None:
    provenance = _load("provenance.json")

    assert provenance["numerical_status"] == "SUPPORTED BUT NOT FULLY CERTIFIED"
    assert provenance["physical_status"] == "DIAGNOSTIC ONLY"
    assert "outer-only" in provenance["claim_scope"]
