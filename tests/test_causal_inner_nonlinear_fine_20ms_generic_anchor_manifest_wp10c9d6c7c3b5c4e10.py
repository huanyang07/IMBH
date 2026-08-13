from __future__ import annotations

import hashlib
import json

import run_causal_inner_nonlinear_fine_20ms_generic_anchor_manifest_wp10c9d6c7c3b5c4e10 as c4e10


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_authorizes_only_one_fine_generic_anchor():
    contract = c4e10._manifest()
    assert contract["definitions_frozen_before_propagation"]
    assert contract["scope"]["layout"] == c4e10.FINE_LAYOUT
    assert contract["scope"]["profile"] == c4e10.GENERIC_PROFILE
    assert contract["anchor_contract"]["continuous_nonlinear_generic_anchor_required"]
    assert not contract["reuse_contract"]["rerun_fine_base"]
    assert not contract["reuse_contract"]["reassemble_five_profile_block_tangent"]
    assert not contract["reuse_contract"]["run_other_profiles_nonlinearly"]


def test_schedule_and_temporal_audits_are_prospective():
    contract = c4e10._manifest()
    assert contract["scope"]["target_microseconds"] == c4e10.TARGET_MICROSECONDS
    assert contract["scope"]["temporal_audit_target_microseconds"] == (8_000, 14_000, 20_000)
    assert contract["anchor_contract"]["maximum_timestep_seconds"] == 4.0e-4
    assert contract["anchor_contract"]["maximum_scaled_nonlinear_residual"] == 1.0e-10
    assert contract["anchor_contract"]["same_target_last_step_replay_bitwise"]


def test_output_reuses_unchanged_spatial_contract():
    contract = c4e10._manifest()
    output = contract["output_contract"]
    assert output["actual_state_response_at_every_accepted_time"]
    assert output["actual_extraction_response_at_every_accepted_time"]
    assert output["rerun_c4e9_with_actual_fine_response"]
    assert output["state_spatial_gate_may_not_be_weakened"]
    assert output["extraction_component_and_aggregate_gates_may_not_be_weakened"]


def test_canonical_summary_preserves_stops_and_cost_scope():
    summary = _read(c4e10.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["fine_generic_anchor_propagation_authorized"]
    assert not summary["fine_base_rerun_authorized"]
    assert not summary["other_nonlinear_profiles_authorized"]
    assert not summary["fine_twenty_ms_spatial_certificate_issued"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert 6.0 < summary["projected_raw_wall_hours"] < 10.0
    assert summary["projected_safe_wall_hours"] < 12.0


def test_canonical_hashes_close():
    for line in (c4e10.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(c4e10.CANONICAL_DIRECTORY / name) == expected
