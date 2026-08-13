from __future__ import annotations

import numpy as np

import run_causal_inner_nonlinear_cost_bounded_fine_20ms_base_tangent_wp10c9d6c7c3b5c4e8 as c4e8


def test_fine_execution_scope_and_faces_are_exact():
    assert c4e8.FINE_LAYOUT == c4e8.c4e7.FINE_LAYOUT
    assert c4e8.COUPLING_FACE == 192
    assert c4e8.EXTRACTION_FACE == 8
    assert c4e8.TARGET_MICROSECONDS[0] == 5_000
    assert c4e8.TARGET_MICROSECONDS[-1] == 20_000
    assert c4e8.MAXIMUM_TIMESTEP_SECONDS == 4.0e-4


def test_initial_fine_base_and_tangent_continue_canonical_5ms_state():
    c4e8._patch_modules()
    configuration = c4e8.h2b1._configuration()
    base, report = c4e8._initial_base(configuration)
    tangent, tangent_report = c4e8._initial_tangent()
    parent = c4e8._parent_arrays()
    assert report["passed_so_far"]
    assert base["accepted_states"].shape == (1, 208, 5)
    assert np.array_equal(
        base["accepted_states"][0], parent["base__accepted_states"][-1]
    )
    assert tangent["state_directions"].shape == (1, 5, 208, 5)
    assert np.array_equal(
        tangent["state_directions"][0], parent["tangent__state_directions"][-1]
    )
    assert tangent_report["maximum_incoming_excision_characteristics"] == 0


def test_source_identity_binds_manifest_runner_and_solver_dependencies():
    identity = c4e8._source_identity()
    assert c4e8.THIS_RUNNER in identity
    assert c4e8.THIS_TEST in identity
    assert c4e8.c4e7.THIS_RUNNER in identity
    assert c4e8.c4e3.THIS_RUNNER in identity
    assert c4e8.h2b1.CONTROLLER_RELATIVE in identity
    assert c4e8.h2b1.MODULE_RELATIVE in identity


def test_pilot_report_requires_three_steps_two_routine_and_one_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(c4e8, "PILOT_SUMMARY_PATH", tmp_path / "pilot.json")
    base = {
        "accepted_times": np.asarray([0.0050, 0.0054, 0.0058, 0.0060]),
        "accepted_timesteps": np.asarray([0.0004, 0.0004, 0.0002]),
        "accepted_step_wall_seconds": np.asarray([30.0, 10.0, 12.0]),
        "audit_flags": np.asarray([True, False, False]),
        "step_maximum_scaled_residuals": np.asarray([1e-11, 1e-11, 1e-11]),
        "step_maximum_discrete_ledger_defects": np.zeros(3),
        "step_maximum_mapped_closure_defects": np.asarray([1e-12] * 3),
        "step_minimum_reconstruction_factors": np.ones(3),
        "step_incoming_excision_characteristics": np.zeros(3, dtype=int),
        "step_extraction_identity_defects": np.asarray([1e-13] * 3),
        "local_error_bounds": np.asarray([1e-8, 1e-8, 1e-8]),
        "retries": np.zeros(3, dtype=int),
    }
    report = c4e8._pilot_report(base, {"reports": {"base": {}}})
    assert report["passed"]
    assert report["full_continuation_authorized"]
    assert report["routine_steps"] == 2
    assert report["audit_steps"] == 1
