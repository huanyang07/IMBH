from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_conservative_transition_tube_wp10c9d6c7c3b5c4f25du as tube


def test_training_table_interpolation_reproduces_a_linear_table() -> None:
    times = np.arange(18, dtype=float)
    train = np.asarray(tube.manifest.geometry_manifest.TRAIN_STATE_INDICES)
    table = np.stack((times[train], 2.0 * times[train]), axis=1)
    predicted, brackets = tube._interpolate_training_table(
        times, table, tube.manifest.geometry_manifest.TRAIN_STATE_INDICES
    )
    assert np.array_equal(predicted, np.stack((times, 2.0 * times), axis=1))
    assert brackets.shape == (18, 2)


def test_exact_linear_conservative_tube_passes() -> None:
    times = np.arange(18, dtype=float)
    macro_restriction = np.hstack((np.eye(2), np.zeros((2, 3))))
    macro_lift = np.vstack((np.eye(2), np.zeros((3, 2))))
    hidden_basis = np.vstack((np.zeros((2, 3)), np.eye(3)))
    macro = np.zeros((18, 2))
    macro[:, 0] = 1.0e-4 * times
    hidden = np.zeros((18, 3))
    hidden[:, 0] = times
    coordinates = (macro_lift @ macro.T + hidden_basis @ hidden.T).T
    metrics, arrays = tube._fit_tube(
        times,
        coordinates,
        macro,
        hidden,
        macro_restriction,
        macro_lift,
        hidden_basis,
    )
    assert metrics["passed"]
    assert metrics["selected_hidden_embedding_rank"] == 1
    assert metrics["failed_gates"] == []
    assert np.max(np.abs(arrays["predicted_coordinates470"] - coordinates)) < 1.0e-12


def test_minimum_rank_uses_energy_threshold() -> None:
    rank, singular_values, basis = tube._minimum_rank(np.diag((4.0, 2.0, 1.0)), 0.8)
    assert rank == 2
    assert singular_values.shape == (3,)
    assert basis.shape == (3, 2)
