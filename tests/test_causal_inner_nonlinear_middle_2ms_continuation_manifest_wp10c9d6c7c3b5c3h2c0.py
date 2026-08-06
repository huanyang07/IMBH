from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_middle_2ms_continuation_manifest_wp10c9d6c7c3b5c3h2c0 as runner  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_authorizes_only_2ms_propagation() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["middle_2ms_propagation_authorized"]
    assert not summary["middle_5ms_propagation_authorized"]
    assert not summary["fine_cost_bounded_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_cost_optimized_targets_are_frozen() -> None:
    manifest = _read(runner.MANIFEST_PATH)["continuation"]
    assert manifest["canonical_target_microseconds"] == [1000, 1200, 1600, 2000]
    assert manifest["strict_sample_target_microseconds"] == [1200, 1600, 2000]
    assert manifest["maximum_timestep_seconds"] == 4.0e-4
    assert manifest["target_1ms_inherited_bitwise_from_h2b1"]


def test_scientific_gates_and_stops_remain_binding() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    assert manifest["base_contract"]["maximum_scaled_residual"] == 1.0e-10
    assert manifest["base_contract"]["incoming_excision_characteristics"] == 0
    assert manifest["tangent_contract"]["all_five_profiles_in_one_block"]
    assert manifest["anchor_contract"]["replay_exact_base_schedule"]
    assert not manifest["downstream_stops"]["middle_5ms_propagation_authorized"]


def test_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(runner.CANONICAL_DIRECTORY / name) == expected
