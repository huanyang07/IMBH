from __future__ import annotations

import json
from pathlib import Path

import numpy as np


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


def test_coupled_state_same_mesh_interpolation_is_identity() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_coupled_inner_outer_rank_prototype import build_coupled_case

    from imri_qpe.layer3_minidisk_1d import (
        interpolate_coupled_state_components,
        pack_coupled_state,
    )

    _canonical, _index, _radius, context, state = build_coupled_case(12, 8)
    components = interpolate_coupled_state_components(
        state,
        context,
        context.inner_params,
        context.outer_grid,
    )
    reconstructed = pack_coupled_state(*components, context)

    np.testing.assert_allclose(reconstructed, state, rtol=0.0, atol=2.0e-14)
