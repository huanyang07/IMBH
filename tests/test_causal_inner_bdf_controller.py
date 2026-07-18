from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    CausalFiveFieldAdaptiveBDF2Config,
    CausalFiveFieldAdaptiveBDF2Restart,
    CausalFiveFieldAdaptiveStepConfig,
    advance_causal_five_field_adaptive_bdf2,
    advance_causal_five_field_increment_bdf,
    causal_five_field_adaptive_bdf2_restarts_equal,
    load_causal_five_field_adaptive_bdf2_restart,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
    save_causal_five_field_adaptive_bdf2_restart,
)


def _step_config() -> CausalFiveFieldAdaptiveStepConfig:
    return CausalFiveFieldAdaptiveStepConfig(
        minimum_dt=1.0e-10,
        maximum_dt=1.0e-5,
        maximum_scaled_primitive_change=0.2,
        maximum_scaled_total_change=0.25,
        residual_tolerance=1.0e-10,
        algebraic_residual_tolerance=1.0e-10,
        conservation_tolerance=1.0e-9,
        maximum_newton_iterations=12,
    ).validated()


def _controller_config(context):
    return CausalFiveFieldAdaptiveBDF2Config(
        step_config=_step_config(),
        cooling_inner_cutoff=(
            6.0 * context.grid.gravitational_radius
        ),
        minimum_dt=1.0e-10,
        maximum_dt=1.0e-5,
        local_error_gate_fraction=1.0,
        audit_interval=1,
    ).validated()


def _startup():
    context = make_causal_five_field_regression_context(4)
    initial = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )
    timestep = 1.0e-8
    startup = advance_causal_five_field_increment_bdf(
        context,
        initial,
        timestep,
        np.zeros_like(initial),
        _step_config(),
        order=1,
    )
    assert startup.accepted
    assert startup.history is not None
    return context, initial, startup, timestep


def test_adaptive_bdf2_config_rejects_unsafe_growth() -> None:
    context = make_causal_five_field_regression_context(4)
    with pytest.raises(ValueError, match="factors"):
        CausalFiveFieldAdaptiveBDF2Config(
            step_config=_step_config(),
            cooling_inner_cutoff=(
                6.0 * context.grid.gravitational_radius
            ),
            minimum_dt=1.0e-10,
            maximum_dt=1.0e-5,
            maximum_factor=2.1,
        ).validated()


def test_adaptive_bdf2_accepts_predictor_and_independent_audit() -> None:
    context, initial, startup, timestep = _startup()
    result = advance_causal_five_field_adaptive_bdf2(
        context,
        startup.state_vector,
        startup.history,
        np.zeros_like(initial),
        timestep,
        timestep,
        _controller_config(context),
        next_order=2,
        accepted_bdf2_steps=0,
    )

    assert result.accepted
    assert result.order == 2
    assert result.accepted_bdf2_steps == 1
    assert result.dt_used > 0.0
    assert result.dt_next > 0.0
    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert attempt.accepted
    assert attempt.local_gate_audit is not None
    assert attempt.local_gate_audit["passed"]
    assert attempt.independent_audit is not None
    assert attempt.independent_audit.passed
    assert attempt.implicit_solves == 3
    np.testing.assert_array_equal(
        result.older_physical_increment,
        startup.history.previous_physical_increment,
    )


def test_adaptive_bdf2_restart_round_trips_complete_state(tmp_path) -> None:
    context, initial, startup, timestep = _startup()
    result = advance_causal_five_field_adaptive_bdf2(
        context,
        startup.state_vector,
        startup.history,
        np.zeros_like(initial),
        timestep,
        timestep,
        _controller_config(context),
        next_order=2,
        accepted_bdf2_steps=0,
    )
    assert result.accepted
    ledger = result.physical_interval_ledger
    restart = CausalFiveFieldAdaptiveBDF2Restart(
        state_vector=result.state_vector,
        history=result.history,
        older_physical_increment=result.older_physical_increment,
        older_timestep_seconds=result.older_timestep_seconds,
        cumulative_actual_conserved_storage=(
            ledger.actual_conserved_storage
        ),
        cumulative_actual_vertical_storage=(
            ledger.actual_vertical_storage
        ),
        cumulative_boundary_transport=(
            ledger.trapezoidal_boundary_transport
        ),
        cumulative_endogenous_source=(
            ledger.trapezoidal_endogenous_source
        ),
        cumulative_stream_source=(
            ledger.exact_prescribed_stream_source
        ),
        cumulative_closure_defect=ledger.closure_defect,
        elapsed_time=2.0 * timestep,
        dt_next=result.dt_next,
        next_order=2,
        accepted_steps=2,
        accepted_bdf2_steps=1,
        rejected_attempts=0,
        audit_count=1,
        provenance={"work_package": "WP10c7c", "case": "roundtrip"},
    )
    path = tmp_path / "adaptive_bdf2_restart.npz"
    save_causal_five_field_adaptive_bdf2_restart(
        path,
        context,
        restart,
    )
    restored = load_causal_five_field_adaptive_bdf2_restart(
        path,
        context,
    )

    assert causal_five_field_adaptive_bdf2_restarts_equal(
        restart,
        restored,
    )
