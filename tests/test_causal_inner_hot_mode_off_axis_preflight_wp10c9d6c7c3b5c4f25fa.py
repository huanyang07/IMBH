from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_hot_mode_off_axis_preflight_wp10c9d6c7c3b5c4f25fa as target


def test_targets_leave_the_artificial_sampling_axis() -> None:
    arrays = target._helper()._load_npz(
        target.hot.CANONICAL_DIRECTORY / "hot_free_field_arrays.npz"
    )
    targets = target._target_coordinates(arrays)
    assert tuple(targets) == ("physical_half", "physical_full", "diagonal_full")
    coordinates = np.asarray(arrays["coordinates5x470"])
    rates = np.asarray(arrays["coordinate_free_rates5x470_per_s"])
    center = target.manifest.HOT_CENTER_INDEX
    artificial = coordinates[-1] - coordinates[0]
    physical = rates[center]
    cosine = abs(float(artificial @ physical)) / (
        np.linalg.norm(artificial) * np.linalg.norm(physical)
    )
    assert cosine < 0.1
    assert not np.array_equal(targets["physical_full"], coordinates[center])


def test_extended_basis_selection_is_fail_closed() -> None:
    rng = np.random.default_rng(812)
    basis, _ = np.linalg.qr(rng.normal(size=(16, 4)))
    coefficients = rng.normal(size=(8, 4))
    hidden = coefficients @ basis.T
    selected, rank, attempts = target._select_basis(
        hidden,
        np.asarray((0, 2, 4, 6)),
        np.asarray((1, 3, 5, 7)),
    )
    assert rank in target.manifest.HIDDEN_RATE_RANKS
    assert selected.shape == (16, rank)
    assert str(rank) in attempts


def test_canonical_package_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        target.CANONICAL_DIRECTORY / "hot_mode_off_axis_metrics.json"
    )
    assert summary["passed"] == metrics["passed"]
    expected = target.PASS_CLASSIFICATION if metrics["passed"] else target.FAIL_CLASSIFICATION
    assert summary["classification"] == expected
    assert metrics["gate_values"]["new_fixed_Q_reaction_calls"] == 0
    assert summary["authorized_next"] == (
        target.AUTHORIZED_NEXT if metrics["passed"] else None
    )
