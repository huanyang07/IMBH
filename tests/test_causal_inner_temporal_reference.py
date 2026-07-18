from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    CausalFiveFieldAdaptiveStepConfig,
    evolve_causal_five_field_fixed_reference,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
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


def test_causal_fixed_reference_validates_subdivisions() -> None:
    context = make_causal_five_field_regression_context(4)
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )

    with pytest.raises(ValueError, match="subdivisions must be positive"):
        evolve_causal_five_field_fixed_reference(
            context,
            vector,
            np.zeros_like(vector),
            1.0e-8,
            1.0e-8,
            0,
            _step_config(),
            physical_ledger_tolerance=1.0e-9,
        )


def test_causal_fixed_reference_reaches_exact_step_count() -> None:
    context = make_causal_five_field_regression_context(4)
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )
    progress: list[tuple[int, int]] = []
    result = evolve_causal_five_field_fixed_reference(
        context,
        vector,
        np.zeros_like(vector),
        1.0e-8,
        1.0e-8,
        1,
        _step_config(),
        physical_ledger_tolerance=1.0e-9,
        progress=lambda completed, total: progress.append(
            (completed, total)
        ),
    )

    assert result.passed
    assert result.completed_steps == 1
    assert result.subdivisions == 1
    assert result.timestep_seconds == 1.0e-8
    assert result.state_gates["passed"]
    assert result.maximum_scaled_residual <= 1.0e-10
    assert result.maximum_physical_ledger_relative_defect <= 1.0e-9
    assert result.function_evaluations > 0
    assert result.jacobian_evaluations > 0
    assert progress == [(1, 1)]
    np.testing.assert_allclose(
        result.previous_physical_increment,
        result.state_vector - vector,
        rtol=3.0e-5,
        atol=1.0e-12,
    )
