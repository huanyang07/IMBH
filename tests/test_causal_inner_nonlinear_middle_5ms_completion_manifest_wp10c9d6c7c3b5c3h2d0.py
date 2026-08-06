from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_middle_5ms_completion_manifest_wp10c9d6c7c3b5c3h2d0 as runner  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_authorizes_only_middle_5ms_propagation() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["middle_5ms_propagation_authorized"]
    assert not summary["middle_5ms_spatial_confirmation_certified"]
    assert not summary["fine_cost_bounded_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_cost_optimized_targets_and_samples_are_frozen() -> None:
    manifest = _read(runner.MANIFEST_PATH)["continuation"]
    assert manifest["canonical_target_microseconds"] == [
        2000, 2400, 2800, 3200, 3600, 4000, 4400, 4800, 5000
    ]
    assert manifest["strict_sample_target_microseconds"] == [2400, 3600, 5000]
    assert manifest["tangent_audit_target_microseconds"] == [2400, 3600, 5000]
    assert manifest["maximum_timestep_seconds"] == 4.0e-4
    assert manifest["target_2ms_inherited_bitwise_from_h2c1"]


def test_scientific_gates_and_stops_remain_binding() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    assert manifest["base_contract"]["maximum_scaled_residual"] == 1.0e-10
    assert manifest["base_contract"]["full_step_doubling_on_every_accepted_comparison"]
    assert manifest["tangent_contract"]["all_five_profiles_in_one_block"]
    assert manifest["anchor_contract"][
        "maximum_tangent_discrepancy_fraction_of_observable_Tier_I"
    ] == 0.01
    assert not manifest["downstream_stops"]["fine_cost_bounded_propagation_authorized"]


def test_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(runner.CANONICAL_DIRECTORY / name) == expected
