from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_third_duration_rung_completion_manifest_wp10c9d6c7c3b5c3c as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_all_targets_are_slices_of_one_integer_source() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    targets = manifest["canonical_targets"]
    master = np.asarray(targets["master_microseconds"], dtype=int)
    for name in ("main", "replay", "strict"):
        indices = np.asarray(targets[f"{name}_indices"], dtype=int)
        microseconds = np.asarray(targets[f"{name}_microseconds"], dtype=int)
        seconds = np.asarray(targets[f"{name}_seconds"], dtype=float)
        assert np.array_equal(master[indices], microseconds)
        assert np.array_equal(microseconds.astype(float) * 1.0e-6, seconds)
    assert targets["independent_target_construction_forbidden"] is True


def test_only_coarse_completion_is_authorized() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["propagation_executed"] is False
    assert summary["coarse_third_duration_rung_completion_authorized"] is True
    assert (
        summary["coarse_third_duration_rung_completion_propagation_authorized"]
        is True
    )
    assert summary["third_duration_rung_breadth_manifest_authorized"] is False
    assert summary["third_duration_rung_spatial_confirmation_authorized"] is False
    assert summary["fourth_duration_rung_manifest_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c3d_coarse_third_duration_rung_completion"
    )


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(
            (runner.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == digest
