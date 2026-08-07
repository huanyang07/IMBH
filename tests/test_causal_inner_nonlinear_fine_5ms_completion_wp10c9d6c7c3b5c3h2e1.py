from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_fine_5ms_completion_wp10c9d6c7c3b5c3h2e1 as runner  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fine_completion_passes_and_stops_are_preserved() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["full_fine_generic_anchor_executed"]
    assert summary["final_spatial_certificate_analysis_authorized"]
    assert not summary["middle_fine_5ms_spatial_certificate_issued"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_exact_layout_schedule_and_audits() -> None:
    config = _read(runner.CONFIG_PATH)
    assert config["layout"] == runner.FINE_LAYOUT
    assert config["coupling_face"] == 192
    assert config["target_microseconds"] == list(runner.TARGET_MICROSECONDS)
    assert config["audit_target_microseconds"] == list(
        runner.AUDIT_TARGET_MICROSECONDS
    )


def test_method_anchor_and_replay_gates_pass() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["base"]["passed"]
    assert summary["tangent"]["passed"]
    assert summary["anchor"]["passed"]
    assert summary["serialized_replays"]["base"]["last_step_replay_bitwise"]
    assert summary["serialized_replays"]["anchor"]["last_step_replay_bitwise"]


def test_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(runner.CANONICAL_DIRECTORY / name) == expected
