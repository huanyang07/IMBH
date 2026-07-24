from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_complete_rate_healing_wp10c8s as wp10c8s


def test_case_matrix_covers_independent_physical_families() -> None:
    indices = tuple(row["mode_index"] for row in wp10c8s.CASE_SPECS)
    assert indices == (0, 1, 2, 3, 4, 7)
    assert len({row["family"] for row in wp10c8s.CASE_SPECS}) == len(indices)
    assert sum(row["parent_case_id"] is not None for row in wp10c8s.CASE_SPECS) == 2


def test_healing_contract_uses_refined_nested_timesteps() -> None:
    rapid = wp10c8s._healing_contract("mode_0_inner_stress_existing")
    persistent = wp10c8s._healing_contract(
        "mode_7_source_shell_stress"
    )
    assert rapid["coarse_subdivisions"] == 10
    assert rapid["fine_subdivisions"] == 20
    assert persistent["coarse_subdivisions"] == 50
    assert persistent["fine_subdivisions"] == 100
    assert (
        rapid["coarse_timestep_seconds"]
        == 2.0 * rapid["fine_timestep_seconds"]
    )


def test_dominant_support_identifies_localized_shell() -> None:
    values = np.zeros((8, 5), dtype=float)
    values[2:4, 3] = (3.0, 1.0)
    result = wp10c8s._dominant_support(
        values,
        radius_rg=np.arange(8, dtype=float) + 1.0,
        shell_edge_indices=np.asarray((0, 4, 8)),
    )
    assert result["controlling_cell"] == 2
    assert result["controlling_field"] == 3
    assert result["controlling_shell"] == 0
    assert result["controlling_shell_l1_fraction"] == 1.0
    assert result["localized_in_one_shell"]


def test_cross_mesh_tangent_gate_accepts_aligned_response() -> None:
    arrays = {
        "n64_t_0p025_top_slow_rate_responses": np.asarray(
            ((1.0, 2.0, -1.0),)
        ),
        "n128_t_0p025_top_slow_rate_responses": np.asarray(
            ((1.05, 2.10, -1.05),)
        ),
    }
    result = wp10c8s._cross_mesh_tangent_row(arrays, 0)
    assert result["passed"]
    assert result["absolute_response_cosine"] > 0.999999


def test_healing_decision_accepts_resolved_decay() -> None:
    names = np.asarray(("slow_rate_a", "slow_rate_b"), dtype="U")
    times = np.asarray((0.0, 0.5, 1.0))
    coarse_values = np.asarray(
        ((2.0, 0.4), (0.4, 0.08), (0.06, 0.02))
    )
    fine_values = np.asarray(
        ((2.0, 0.4), (0.38, 0.075), (0.05, 0.015))
    )
    zeros = np.zeros((3, 1), dtype=float)
    coarse = {
        "times": times,
        "full_names": names,
        "full_spreads": coarse_values,
        "coordinate_spreads": zeros,
    }
    fine = {
        "times": times,
        "full_names": names,
        "full_spreads": fine_values,
        "coordinate_spreads": zeros,
    }
    result, _arrays = wp10c8s._healing_decision(
        coarse=coarse,
        fine=fine,
    )
    assert result["temporal_uncertainty_passed"]
    assert result["factor_two_decay_passed"]
    assert result["final_healing_gate_passed"]
    assert result["natural_healing_passed"]


def test_healing_decision_rejects_persistent_mode() -> None:
    names = np.asarray(("slow_rate_a",), dtype="U")
    times = np.asarray((0.0, 1.0))
    zeros = np.zeros((2, 1), dtype=float)
    coarse = {
        "times": times,
        "full_names": names,
        "full_spreads": np.asarray(((2.0,), (1.9,))),
        "coordinate_spreads": zeros,
    }
    fine = {
        "times": times,
        "full_names": names,
        "full_spreads": np.asarray(((2.0,), (1.89,))),
        "coordinate_spreads": zeros,
    }
    result, _arrays = wp10c8s._healing_decision(
        coarse=coarse,
        fine=fine,
    )
    assert result["temporal_uncertainty_passed"]
    assert not result["factor_two_decay_passed"]
    assert not result["natural_healing_passed"]


def test_healing_decision_resolves_persistence_despite_large_uncertainty() -> None:
    names = np.asarray(("slow_rate_a",), dtype="U")
    times = np.asarray((0.0, 0.5, 1.0))
    zeros = np.zeros((3, 1), dtype=float)
    coarse = {
        "times": times,
        "full_names": names,
        "full_spreads": np.asarray(((100.0,), (40.0,), (30.0,))),
        "coordinate_spreads": zeros,
    }
    fine = {
        "times": times,
        "full_names": names,
        "full_spreads": np.asarray(((100.0,), (35.0,), (20.0,))),
        "coordinate_spreads": zeros,
    }
    result, arrays = wp10c8s._healing_decision(
        coarse=coarse,
        fine=fine,
    )
    assert not result["temporal_uncertainty_passed"]
    assert result["final_maximum_lower_spread"] == 10.0
    assert result["persistence_separated_from_healing_gate"]
    assert arrays["persistent_output_mask"][0]
    assert not result["natural_healing_passed"]
