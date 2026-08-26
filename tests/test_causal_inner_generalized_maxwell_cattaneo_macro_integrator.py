from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_atlas import (
    OUTPUT_SIZE,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_integrator import (
    ExactAffineMacroSystem,
    ExactAffineMacroTransition,
    macro_rate_output_matrix,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_thermodynamic_macro_atlas import (
    ThermodynamicAffineMacroAtlas,
)


def _atlas() -> ThermodynamicAffineMacroAtlas:
    anchor = np.ones((16, 5))
    anchor[:, 3] = -0.2
    anchor[:, 4] = 1.0e-4
    base = np.zeros(OUTPUT_SIZE)
    base[0] = -1.0e-12
    base[3] = -0.9e-12
    jacobian = np.zeros((OUTPUT_SIZE, 80))
    jacobian[0, 0] = -1.0e-15
    jacobian[3, 0] = 1.0e-15
    return ThermodynamicAffineMacroAtlas(
        anchor_macro_state=anchor,
        macro_coordinate_scales=np.abs(anchor),
        base_normalized_output=base,
        normalized_output_jacobian=jacobian,
        output_component_scales=np.ones(OUTPUT_SIZE),
        trust_coordinate_infinity=1.0,
        macro_coordinate_pullback=np.tile(np.eye(5), (16, 1, 1)),
    )


def test_rate_output_matrix_has_exact_flux_telescoping() -> None:
    matrix = macro_rate_output_matrix()
    outputs = np.arange(OUTPUT_SIZE, dtype=float)
    rates = (matrix @ outputs).reshape(16, 5)
    expected = np.zeros(3)
    flux = outputs[:51].reshape(17, 3)
    sources = np.zeros((16, 3))
    sources[:, 1:] = outputs[51:83].reshape(16, 2)
    from imri_qpe.constants import C

    expected[:] = C * (sources.sum(axis=0) - (flux[-1] - flux[0]))
    np.testing.assert_allclose(rates[:, :3].sum(axis=0), expected, rtol=2e-15)


def test_exact_affine_transition_has_semigroup_and_integrated_ledger() -> None:
    system = ExactAffineMacroSystem.from_atlas(_atlas())
    full = ExactAffineMacroTransition.build(
        system, 2.0e-4, trust_coordinate_infinity=1.0
    )
    half = ExactAffineMacroTransition.build(
        system, 1.0e-4, trust_coordinate_infinity=1.0
    )
    initial = np.array(system.atlas.anchor_macro_state, copy=True)
    one = full.step(initial)
    first = half.step(initial)
    second = half.step(first.macro_state)
    np.testing.assert_allclose(one.macro_state, second.macro_state, rtol=2e-13)
    assert one.state_ledger_relative_defect <= 5.0e-13
    assert first.state_ledger_relative_defect <= 5.0e-13
    assert second.state_ledger_relative_defect <= 5.0e-13


def test_transition_rejects_endpoint_outside_trust() -> None:
    system = ExactAffineMacroSystem.from_atlas(_atlas())
    transition = ExactAffineMacroTransition.build(
        system, 1.0, trust_coordinate_infinity=1.0e-6
    )
    with np.testing.assert_raises_regex(ValueError, "trust box"):
        transition.step(system.atlas.anchor_macro_state)
