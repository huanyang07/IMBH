from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_canonical_time_replay_manifest_wp10c9d6c7c3b5c2e as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_target_grid_difference_is_frozen() -> None:
    summary = _read(runner.SUMMARY_PATH)
    diagnostic = summary["target_grid_diagnostic"]
    assert diagnostic["differing_indices"] == [1]
    assert diagnostic["maximum_spacing_units"] == 1.0
    assert diagnostic["canonical_target_hex"][1] != diagnostic["legacy_target_hex"][1]


def test_only_canonical_time_audit_is_authorized() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["propagation_executed"] is False
    assert summary["canonical_time_replay_audit_authorized"] is True
    assert summary["third_duration_rung_manifest_authorized"] is False
    assert summary["third_duration_rung_propagation_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c2e1_canonical_time_replay_audit"
    )


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(
            (runner.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == digest
