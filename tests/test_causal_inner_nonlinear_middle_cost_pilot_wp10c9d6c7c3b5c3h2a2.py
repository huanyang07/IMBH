from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_middle_cost_pilot_wp10c9d6c7c3b5c3h2a2 as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_controller_exposes_optional_accepted_step_recording() -> None:
    signature = inspect.signature(runner.c2._controller_segment)
    assert signature.parameters["record_accepted_steps"].default is False
    assert signature.parameters["log_prefix"].default == "c2"


def test_pilot_base_tangent_anchor_and_replay_pass() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["base"]["passed"] is True
    assert summary["base"]["projection_sample_sufficient"] is True
    assert summary["tangent"]["passed"] is True
    assert summary["anchor"]["passed"] is True
    for replay in summary["serialized_replays"].values():
        assert replay["checkpoint_roundtrip_bitwise"] is True
        assert replay["last_step_replay_bitwise"] is True


def test_anchor_surrogate_passes_absolute_and_response_relative_gates() -> None:
    summary = _read(runner.SUMMARY_PATH)
    gates = _read(runner.CONFIG_PATH)["surrogate_gates"]
    for channel, absolute_gate in (
        ("state", "maximum_absolute_scaled_state_discrepancy"),
        ("instantaneous_Tier_I", "maximum_absolute_scaled_Tier_I_discrepancy"),
    ):
        report = summary["anchor"][channel]
        assert report["maximum_scaled_discrepancy"] <= gates[absolute_gate]
        assert report["discrepancy_fraction_of_observable_response"] <= gates[
            "maximum_discrepancy_fraction_of_observable_response"
        ]


def test_projection_is_measured_but_not_a_scientific_gate() -> None:
    projection = _read(runner.SUMMARY_PATH)["cost_projection"]
    assert projection["pilot_accepted_steps"] >= runner.MINIMUM_PROJECTION_STEPS
    assert projection["simulated_remaining_steps"] > 0
    assert projection["projected_total_wall_hours"] > 0.0
    assert projection["cost_projection_is_not_a_scientific_gate"] is True
    assert projection["resource_tier"] in {
        "automatic_continuation",
        "optimization_review",
        "explicit_cost_benefit_decision",
    }


def test_only_continuation_manifest_is_authorized() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["middle_1ms_continuation_manifest_authorized"] is True
    assert summary["middle_1ms_propagation_authorized"] is False
    assert summary["middle_5ms_spatial_confirmation_certified"] is False
    assert summary["fine_cost_bounded_propagation_authorized"] is False
    assert summary["third_duration_rung_spatial_convergence_certified"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        payload = (runner.CANONICAL_DIRECTORY / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
