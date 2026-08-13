from __future__ import annotations

import hashlib
import json

import run_causal_inner_nonlinear_cost_bounded_fine_20ms_manifest_wp10c9d6c7c3b5c4e7 as c4e7


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fine_contract_uses_exact_layout_faces_and_staged_targets():
    contract = c4e7._manifest()
    scope = contract["scientific_scope"]
    assert contract["definitions_frozen_before_propagation"]
    assert scope["layout"] == c4e7.FINE_LAYOUT
    assert scope["start_microseconds"] == 5_000
    assert scope["pilot_stop_microseconds"] == 6_000
    assert scope["stop_microseconds"] == 20_000
    assert scope["coupling_face"] == 192
    assert scope["extraction_face"] == 8
    assert scope["target_microseconds"] == c4e7.TARGET_MICROSECONDS
    assert scope["audit_target_microseconds"] == c4e7.AUDIT_TARGET_MICROSECONDS


def test_cost_controls_and_no_automatic_anchor_are_frozen():
    contract = c4e7._manifest()
    assert contract["controller"]["maximum_timestep_seconds"] == 4.0e-4
    assert contract["controller"][
        "larger_timestep_preflight_forbidden_after_middle_0p8ms_failure"
    ]
    assert contract["cost_control"]["pilot_reprojection_required_at_6ms"]
    assert contract["cost_control"][
        "pilot_requires_at_least_two_routine_steps_and_one_audit"
    ]
    assert contract["cost_control"]["projected_raw_wall_hours"] > 15.0
    assert contract["cost_control"]["projected_safe_wall_hours"] < 40.0
    assert contract["minimum_work"][
        "full_fine_anchor_may_not_launch_inside_this_campaign"
    ]
    assert contract["nonlinear_remainder"]["automatic_full_anchor_forbidden"]
    assert contract["nonlinear_remainder"][
        "one_step_nonlinear_generic_anchor_shadow_at_audit_targets"
    ]
    assert contract["nonlinear_remainder"][
        "continuous_or_full_generic_anchor_forbidden"
    ]
    assert contract["cost_control"]["projected_sampled_anchor_count"] == len(
        c4e7.AUDIT_TARGET_MICROSECONDS
    )


def test_spatial_and_uncertainty_gates_are_prospective():
    contract = c4e7._manifest()
    gates = contract["spatial_certificate"]
    assert gates["minimum_RMS_order"] == 0.75
    assert gates["minimum_refinement_error_cosine"] == 0.90
    assert gates["maximum_fine_normalized_difference"] == 0.05
    assert gates["maximum_temporal_fraction_of_middle_fine_difference"] == 0.10
    assert gates["maximum_surrogate_fraction_of_middle_fine_difference"] == 0.10
    assert gates["unobservable_difference_yields_upper_bound_not_order"]


def test_canonical_summary_authorizes_only_base_and_tangent_campaign():
    summary = _read(c4e7.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["fine_base_block_tangent_propagation_authorized"]
    assert not summary["full_fine_generic_anchor_required"]
    assert not summary["full_fine_generic_anchor_authorized"]
    assert not summary["fine_twenty_ms_spatial_certificate_issued"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_canonical_hashes_close():
    for line in (c4e7.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(c4e7.CANONICAL_DIRECTORY / name) == expected
