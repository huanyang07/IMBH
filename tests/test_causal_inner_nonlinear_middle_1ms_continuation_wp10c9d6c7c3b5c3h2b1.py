from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_middle_1ms_continuation_wp10c9d6c7c3b5c3h2b1 as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_middle_1ms_base_and_method_gates_pass() -> None:
    summary = _read(runner.SUMMARY_PATH)
    base = summary["base"]
    assert summary["passed"] is True
    assert base["passed"] is True
    assert base["rejected_attempts"] == 0
    assert base["maximum_scaled_residual"] <= 1.0e-10
    assert base["maximum_discrete_ledger_defect"] <= 1.0e-12
    assert base["maximum_mapped_endpoint_path_closure_defect"] <= 1.0e-9
    assert base["minimum_path_reconstruction_factor"] >= 1.0
    assert base["maximum_incoming_excision_characteristics"] == 0


def test_corrected_five_profile_tangent_passes() -> None:
    summary = _read(runner.SUMMARY_PATH)
    tangent = summary["tangent"]
    assert tangent["passed"] is True
    assert tangent["maximum_step_matrix_jvp_relative_defect"] <= 1.0e-6
    assert tangent["maximum_linear_solve_relative_defect"] <= 1.0e-10
    assert tangent["maximum_matrix_component_closure_defect"] <= 1.0e-12
    assert tangent["maximum_export_active_prefix_ledger_defect"] <= 1.0e-12
    with np.load(runner.DECISIVE_ARRAYS, allow_pickle=False) as payload:
        assert payload["tangent__state_directions"].shape[1] == 5
        assert payload["tangent__export_directions"].shape[1] == 5


def test_generic_anchor_and_cumulative_response_pass() -> None:
    anchor = _read(runner.SUMMARY_PATH)["anchor"]
    assert anchor["passed"] is True
    for key in ("state", "instantaneous_Tier_I", "cumulative_Tier_I"):
        metrics = anchor[key]
        assert metrics["discrepancy_fraction_of_observable_response"] <= 0.01
        assert metrics["history_cosine"] >= 0.99
    assert anchor["maximum_sampled_state_error_estimate"] <= 2.5e-4
    assert anchor["maximum_sampled_export_error_estimate"] <= 2.5e-4


def test_serialized_replays_are_bitwise() -> None:
    replays = _read(runner.SUMMARY_PATH)["serialized_replays"]
    for item in replays.values():
        assert item["checkpoint_roundtrip_bitwise"] is True
        assert item["last_step_replay_bitwise"] is True
        assert item["maximum_scaled_residual"] <= 1.0e-10


def test_authorizations_remain_narrow() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["middle_2ms_continuation_manifest_authorized"] is True
    assert summary["middle_2ms_propagation_authorized"] is False
    assert summary["middle_5ms_spatial_confirmation_certified"] is False
    assert summary["fine_cost_bounded_propagation_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False


def test_canonical_targets_and_hashes_close() -> None:
    with np.load(runner.DECISIVE_ARRAYS, allow_pickle=False) as payload:
        times = payload["base__output_times"]
        expected = np.asarray(runner.TARGET_MICROSECONDS, dtype=float) * 1.0e-6
        assert np.array_equal(times, expected)
        assert payload["base__accepted_times"][-1] == 1.0e-3
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        payload = (runner.CANONICAL_DIRECTORY / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
