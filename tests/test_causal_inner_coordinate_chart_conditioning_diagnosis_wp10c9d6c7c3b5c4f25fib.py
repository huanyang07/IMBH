from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_coordinate_chart_conditioning_diagnosis_wp10c9d6c7c3b5c4f25fib as target  # noqa: E402


def test_manifest_authorizes_only_saved_state_diagnosis() -> None:
    lock = target._validate_manifest(require_clean=False)
    assert lock["summary"]["authorized_next"] == target.manifest.AUTHORIZED_NEXT
    assert lock["contract"]["scope"]["new_exact_free_field_calls"] == 0
    assert lock["contract"]["scope"]["new_trajectory_segments"] == 0


def test_row_equilibration_normalizes_every_row() -> None:
    matrix = np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    equilibrated, norms = target._row_equilibrate(matrix)
    np.testing.assert_allclose(norms, [5.0, 2.0])
    np.testing.assert_allclose(np.linalg.norm(equilibrated, axis=1), 1.0)


def test_block_whitening_has_identity_row_gram() -> None:
    matrix = np.asarray([[2.0, 1.0, 0.0], [0.0, 3.0, 4.0]])
    whitened, transform, closure, right = target._whiten_block(matrix)
    np.testing.assert_allclose(transform @ matrix, whitened)
    np.testing.assert_allclose(whitened @ whitened.T, np.eye(2), atol=2.0e-15)
    assert closure < 2.0e-15
    assert right.shape == (2, 3)


def test_principal_angles_detect_orthogonal_rows() -> None:
    left = np.asarray([[1.0, 0.0, 0.0]])
    right = np.asarray([[0.0, 1.0, 0.0]])
    metrics = target._principal_angle_metrics(left, right)
    assert metrics["maximum_principal_cosine"] == 0.0
    assert metrics["minimum_principal_angle_radians"] == 0.5 * np.pi


def test_canonical_diagnosis_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        target.CANONICAL_DIRECTORY / "conditioning_metrics.json"
    )
    assert summary["classification"] == metrics["classification"]
    assert summary["historical_raw_condition_rejection_preserved"]
    assert not summary["new_trajectory"]
    assert metrics["gate_values"]["new_exact_free_field_calls"] == 0
