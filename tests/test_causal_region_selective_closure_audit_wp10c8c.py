from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_region_selective_closure_audit_wp10c8c as wp10c8c


def test_weighted_percentile_respects_weight_ordering() -> None:
    value = wp10c8c._weighted_percentile(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([0.8, 0.1, 0.1]),
        0.75,
    )

    assert value == 1.0


def test_component_indices_select_requested_cells_and_fields() -> None:
    values = wp10c8c._component_indices(
        4,
        (1, 4),
        np.asarray([False, True, False, True]),
    )

    np.testing.assert_array_equal(values, [6, 9, 16, 19])


def test_schur_closure_recovers_known_manifold() -> None:
    dynamic = np.asarray(
        [
            [-4.0, 2.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, -0.5],
        ]
    )

    closure = wp10c8c._schur_closure(dynamic, np.asarray([0]))

    np.testing.assert_allclose(closure["manifold"], [[0.5, 0.0]])
    np.testing.assert_allclose(
        closure["effective_operator"],
        [[-0.5, 0.0], [0.0, -0.5]],
    )
    assert closure["solve_relative_defect"] < 1.0e-14


def test_lift_places_fast_and_retained_components() -> None:
    dynamic = np.asarray(
        [
            [-4.0, 2.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, -0.5],
        ]
    )
    closure = wp10c8c._schur_closure(dynamic, np.asarray([0]))

    lifted = wp10c8c._lift(np.asarray([2.0, 4.0]), closure)

    np.testing.assert_allclose(lifted, [1.0, 2.0, 4.0])


def test_physical_component_gate_rejects_large_radial_defect() -> None:
    metrics = wp10c8c._physical_component_metrics(
        (1,),
        np.asarray([True, True]),
        np.asarray([1.0, 1.0]),
        {
            "radial_momentum_stationary_balance": np.asarray([0.0, 1.0]),
        },
    )

    assert not metrics["passed"]


def test_direction_audit_accepts_exact_decoupled_closure() -> None:
    dynamic = np.diag([-4.0, -1.0, -0.5])
    closure = wp10c8c._schur_closure(dynamic, np.asarray([0]))
    direction = np.asarray([0.0, 1.0, 0.0])
    operators = {
        "profile": np.eye(3),
        "baseline": {},
    }

    audit = wp10c8c._direction_audit(
        dynamic,
        closure,
        operators,
        direction,
    )

    assert audit["passed"]
    assert audit["manifold_invariance_relative_defect"] == 0.0
