from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_middle_2ms_continuation_wp10c9d6c7c3b5c3h2c1 as runner  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_summary_passes_and_authorizes_only_5ms_manifest() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["middle_5ms_completion_manifest_authorized"]
    assert not summary["middle_5ms_propagation_authorized"]
    assert not summary["fine_cost_bounded_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_base_tangent_anchor_and_replays_pass() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["base"]["passed"]
    assert summary["base"]["rejected_attempts"] == 0
    assert summary["tangent"]["passed"]
    assert summary["anchor"]["passed"]
    assert summary["serialized_replays"]["base"]["last_step_replay_bitwise"]
    assert summary["serialized_replays"]["anchor"]["last_step_replay_bitwise"]


def test_canonical_target_bits_and_five_profile_shapes() -> None:
    with np.load(runner.DECISIVE_ARRAYS, allow_pickle=False) as payload:
        with np.load(runner.h2b1.DECISIVE_ARRAYS, allow_pickle=False) as parent:
            inherited = parent["base__output_times"][-1]
        expected = np.asarray(
            [inherited, *[float(value) * 1.0e-6 for value in runner.TARGET_MICROSECONDS[1:]]],
            dtype=np.float64,
        )
        assert np.array_equal(payload["base__output_times"], expected)
        assert payload["tangent__state_directions"].shape[1] == 5
        assert payload["tangent__export_directions"].shape[1:] == (5, 13)


def test_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(runner.CANONICAL_DIRECTORY / name) == expected
