from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_canonical_time_replay_audit_wp10c9d6c7c3b5c2e1 as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_same_target_replays_are_bitwise() -> None:
    summary = _read(runner.SUMMARY_PATH)
    for trajectory in ("base", "perturbed"):
        same = summary["trajectory_reports"][trajectory][
            "same_target_direct_serialized"
        ]
        assert all(same.values())


def test_target_grid_mechanism_and_response_pass() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["method_report"]["passed"] is True
    assert summary["response_comparison"]["passed"] is True
    for trajectory in ("base", "perturbed"):
        report = summary["trajectory_reports"][trajectory]
        assert report["passed"] is True
        assert report["legacy_canonical_comparison"]["first_difference_index"] == 1


def test_only_third_rung_manifest_is_authorized() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["classification"] == (
        "canonical_target_replay_bitwise_certified_third_rung_manifest_authorized"
    )
    assert summary["third_duration_rung_manifest_authorized"] is True
    assert summary["third_duration_rung_propagation_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c3a_third_duration_rung_manifest"
    )


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(
            (runner.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == digest
