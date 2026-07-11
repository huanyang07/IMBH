from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/canonical/two_domain_interface_sweep/summary.json"


def test_two_domain_composite_passes_flux_but_not_primitive_gate() -> None:
    result = json.loads(SUMMARY.read_text())
    assert result["all_converged"]
    for resolution in ("128", "256"):
        audit = result["by_resolution"][resolution]
        assert audit["flux_gate"]
        assert audit["interface_position_gate"]
        assert not audit["primitive_continuity_gate"]
        assert audit["maximum_primitive_mismatch"] > 0.3
        assert (
            audit["relative_spreads"]["composite_Lrad_over_LEdd"]
            < 3.0e-3
        )
