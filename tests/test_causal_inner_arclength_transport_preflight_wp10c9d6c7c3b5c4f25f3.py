from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_arclength_transport_preflight_wp10c9d6c7c3b5c4f25f3 as target


def test_broyden_update_satisfies_latest_secant() -> None:
    matrix = np.eye(3)
    step = np.asarray((1.0, -2.0, 0.5))
    change = np.asarray((2.0, 3.0, -1.0))
    updated = target._broyden_update(matrix, step, change)
    np.testing.assert_allclose(updated @ step, change)


def test_manifest_when_present_is_frozen() -> None:
    if not target.manifest.CANONICAL_DIRECTORY.exists():
        return
    locked = target._validate_manifest(require_clean=False)
    assert locked["contract"]["work_package"] == target.manifest.WORK_PACKAGE


def test_canonical_preflight_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(target.CANONICAL_DIRECTORY / "transport_metrics.json")
    assert summary["passed"]
    assert metrics["passed"]
    assert metrics["new_exact_fixed_Q_rate_calls"] == 0
    assert metrics["gates"]["canonical_state_replay"]
