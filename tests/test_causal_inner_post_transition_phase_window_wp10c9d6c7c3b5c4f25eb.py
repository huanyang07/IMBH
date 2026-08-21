from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_post_transition_phase_window_wp10c9d6c7c3b5c4f25eb as window


def test_canonical_basis_has_requested_rank_and_small_training_defect() -> None:
    phase = np.linspace(0.0, 1.0, 8)
    rates = np.stack(
        (
            1.0 + phase,
            2.0 - phase,
            phase**2,
            np.sin(phase),
            np.zeros_like(phase),
        ),
        axis=1,
    )
    basis, singular = window._canonical_basis(rates, 4)
    assert basis.shape == (5, 4)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=2.0e-14)
    assert np.all(np.diff(singular) <= 0.0)
    projected = (rates @ basis) @ basis.T
    np.testing.assert_allclose(projected, rates, atol=2.0e-14)


def test_picard_window_is_exact_for_constant_field() -> None:
    constant = np.asarray((2.0, -3.0, 0.5))

    def evaluator(_coordinate: np.ndarray, _time: float) -> np.ndarray:
        return constant

    result = window._picard_window(
        start_coordinate=np.asarray((1.0, 2.0, 3.0)),
        start_time_seconds=0.4,
        duration_seconds=0.2,
        basis=np.eye(3),
        evaluator=evaluator,
        node_count=8,
    )
    np.testing.assert_allclose(result["endpoint"], np.asarray((1.4, 1.4, 3.1)), atol=3.0e-14)
    assert float(np.max(result["projected_defects"])) < 5.0e-11
    assert float(np.max(result["full_defects"])) < 5.0e-11
    assert float(np.min(result["direction_cosines"])) > 1.0 - 2.0e-14
