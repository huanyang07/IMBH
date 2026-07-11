from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "results/canonical/pressure_supported_interface_pilot/summary.json"
)


def test_pressure_supported_pilot_is_coarse_grid_only() -> None:
    result = json.loads(SUMMARY.read_text())
    assert result["classification"] == "COARSE_GRID_ONLY_NOT_MESH_SUPPORTED"
    coarse = [row for row in result["rows"] if row["N_reservoir"] == 64]
    fine = [row for row in result["rows"] if row["N_reservoir"] == 128]
    assert all(row["converged"] for row in coarse)
    assert all(row["pressure_support_fraction_reached"] == 1.0 for row in coarse)
    assert not any(row["converged"] for row in fine)
    assert all(
        row["primitive_mismatch"]["maximum_absolute"] > 0.3
        for row in coarse
    )
