from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_atlas import (
    ConservativeAffineMacroAtlas,
    ConservativeMacroOutputs,
    OUTPUT_SIZE,
    conservative_ledger_relative_defect,
    macro_output_component_scales,
    pack_macro_outputs,
    restrict_entropy_complete_macro,
    unpack_macro_outputs,
)


def _outputs() -> ConservativeMacroOutputs:
    flux = -np.arange(51, dtype=float).reshape(17, 3) - 1.0
    source = np.zeros((16, 3))
    source[:, 1:] = np.arange(32, dtype=float).reshape(16, 2) + 2.0
    auxiliary = np.arange(32, dtype=float).reshape(16, 2) / 100.0
    packed = np.concatenate((flux.ravel(), source[:, 1:].ravel(), auxiliary.ravel()))
    return unpack_macro_outputs(packed)


def test_exact_seven_to_one_restriction() -> None:
    targets = np.arange(112 * 3, dtype=float).reshape(112, 3) + 1.0
    charts = np.zeros((112, 7))
    charts[:, 1] = np.linspace(-0.4, -0.2, 112)
    charts[:, 4] = np.linspace(1.0e-5, 2.0e-4, 112)
    restricted = restrict_entropy_complete_macro(targets, charts)
    np.testing.assert_array_equal(
        restricted[:, :3], targets.reshape(16, 7, 3).sum(axis=1)
    )
    mass = targets[:, 0].reshape(16, 7)
    weights = mass / mass.sum(axis=1)[:, None]
    np.testing.assert_allclose(
        restricted[:, 3], (weights * charts[:, 1].reshape(16, 7)).sum(axis=1)
    )


def test_macro_output_pack_roundtrip_and_ledger() -> None:
    outputs = _outputs()
    packed = pack_macro_outputs(outputs)
    assert packed.shape == (OUTPUT_SIZE,)
    recovered = unpack_macro_outputs(packed)
    np.testing.assert_array_equal(recovered.MJE_face_fluxes_over_c, outputs.MJE_face_fluxes_over_c)
    np.testing.assert_array_equal(recovered.MJE_cell_sources_per_ct, outputs.MJE_cell_sources_per_ct)
    assert conservative_ledger_relative_defect(recovered) <= 2.0e-16


def test_affine_atlas_reproduces_base_and_enforces_trust_box() -> None:
    outputs = _outputs()
    scales = macro_output_component_scales(outputs)
    base = pack_macro_outputs(outputs) / scales
    state = np.ones((16, 5))
    state[:, 3] = -0.3
    state[:, 4] = 1.0e-4
    atlas = ConservativeAffineMacroAtlas(
        anchor_macro_state=state,
        macro_coordinate_scales=np.abs(state),
        base_normalized_output=base,
        normalized_output_jacobian=np.zeros((OUTPUT_SIZE, 80)),
        output_component_scales=scales,
        trust_coordinate_infinity=0.05,
    )
    np.testing.assert_allclose(
        pack_macro_outputs(atlas.evaluate(state)),
        pack_macro_outputs(outputs),
        rtol=2.0e-16,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="trust box"):
        atlas.evaluate(1.1 * state)


def test_mass_source_must_be_zero() -> None:
    outputs = _outputs()
    bad_source = np.array(outputs.MJE_cell_sources_per_ct, copy=True)
    bad_source[0, 0] = 1.0
    with pytest.raises(ValueError, match="mass source"):
        pack_macro_outputs(
            ConservativeMacroOutputs(
                MJE_face_fluxes_over_c=outputs.MJE_face_fluxes_over_c,
                MJE_cell_sources_per_ct=bad_source,
                auxiliary_rates_per_second=outputs.auxiliary_rates_per_second,
                macro_rates_per_second=outputs.macro_rates_per_second,
            )
        )


def test_affine_atlas_applies_block_pullback() -> None:
    outputs = _outputs()
    output_scales = macro_output_component_scales(outputs)
    base = pack_macro_outputs(outputs) / output_scales
    anchor = np.ones((16, 5))
    anchor[:, 3] = -0.3
    anchor[:, 4] = 1.0e-4
    jacobian = np.zeros((OUTPUT_SIZE, 80))
    jacobian[0, 0] = 2.0
    pullback = np.tile(np.eye(5), (16, 1, 1))
    pullback[0, 0, 0] = 3.0
    atlas = ConservativeAffineMacroAtlas(
        anchor_macro_state=anchor,
        macro_coordinate_scales=np.abs(anchor),
        base_normalized_output=base,
        normalized_output_jacobian=jacobian,
        output_component_scales=output_scales,
        trust_coordinate_infinity=0.05,
        macro_coordinate_pullback=pullback,
    )
    candidate = np.array(anchor, copy=True)
    candidate[0, 0] *= 1.01
    predicted = pack_macro_outputs(atlas.evaluate(candidate)) / output_scales
    assert abs(predicted[0] - base[0] - 0.06) <= 2.0e-15
