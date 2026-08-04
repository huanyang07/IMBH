from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_third_duration_rung_manifest_wp10c9d6c7c3b5c3a as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_targets_have_one_integer_source() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    targets = manifest["canonical_targets"]
    assert targets["single_source_slices_required"] is True
    assert targets["independent_linspace_construction_forbidden"] is True
    assert np.array_equal(
        np.asarray(targets["main_microseconds"], dtype=float) * 1.0e-6,
        np.asarray(targets["main_seconds"], dtype=float),
    )


def test_only_screen_is_authorized() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["propagation_executed"] is False
    assert summary["third_duration_rung_screen_authorized"] is True
    assert summary["third_duration_rung_completion_manifest_authorized"] is False
    assert summary["third_duration_rung_completion_propagation_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c3b_third_duration_rung_screen"
    )


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(
            (runner.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == digest
