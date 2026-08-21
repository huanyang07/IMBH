from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_transition_tube_geometry_wp10c9d6c7c3b5c4f25ds as audit


def test_piecewise_linear_holdout_reconstruction_is_exact_for_a_line() -> None:
    times = np.arange(18, dtype=float)
    coordinates = np.stack((times, 2.0 * times), axis=1)
    predicted, chords, brackets = audit._holdout_interpolation(
        times,
        coordinates,
        audit.manifest.TRAIN_STATE_INDICES,
        audit.manifest.HOLDOUT_STATE_INDICES,
    )
    assert np.array_equal(
        predicted,
        coordinates[np.asarray(audit.manifest.HOLDOUT_STATE_INDICES)],
    )
    assert np.all(chords > 0.0)
    assert brackets.shape == (len(audit.manifest.HOLDOUT_STATE_INDICES), 2)


def test_exact_oblique_split_and_scalar_line_pass_synthetic_geometry() -> None:
    times = np.linspace(0.0, 17.0, 18)
    macro_restriction = np.hstack((np.eye(2), np.zeros((2, 3))))
    hidden_dual = np.hstack((np.zeros((3, 2)), np.eye(3)))
    hidden_basis = np.vstack((np.zeros((2, 3)), np.eye(3)))
    coordinates = np.zeros((18, 5))
    coordinates[:, 2] = np.linspace(0.0, 1.0, 18)
    hidden_fractions = np.ones(17)
    metrics, arrays = audit._analyze_geometry(
        times,
        coordinates,
        hidden_fractions,
        macro_restriction,
        hidden_basis,
        hidden_dual,
    )
    assert metrics["passed"]
    assert metrics["transition_dynamic_dimension"] == 1
    assert metrics["selected_hidden_embedding_rank"] == 1
    assert metrics["failed_gates"] == []
    assert np.max(arrays["holdout_errors"]) < 1.0e-14


def test_rank_selector_returns_minimum_energy_rank() -> None:
    matrix = np.diag((4.0, 2.0, 1.0))
    rank, singular_values, basis = audit._rank_for_energy(matrix, 0.8)
    assert rank == 2
    assert singular_values.shape == (3,)
    assert basis.shape == (3, 2)
