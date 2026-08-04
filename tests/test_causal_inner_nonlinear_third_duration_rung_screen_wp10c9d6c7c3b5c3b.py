from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_third_duration_rung_screen_wp10c9d6c7c3b5c3b as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_trajectories_and_replay_pass() -> None:
    summary = _read(runner.SUMMARY_PATH)
    for trajectory in ("base", "perturbed"):
        report = summary["trajectory_reports"][trajectory]
        assert report["passed"] is True
        assert report["main_report"]["passed"] is True
        assert report["replay_report"]["passed"] is True
        assert report["strict_report"]["passed"] is True
        assert all(report["replay_bitwise"].values())


def test_screen_authorizes_only_completion_manifest() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["strict_response_comparison"]["passed"] is True
    assert summary["third_duration_rung_completion_manifest_authorized"] is True
    assert summary["third_duration_rung_completion_propagation_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c3c_third_duration_rung_completion_manifest"
    )


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(
            (runner.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == digest
