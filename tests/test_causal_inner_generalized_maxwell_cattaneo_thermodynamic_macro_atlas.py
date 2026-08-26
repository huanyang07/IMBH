from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_atlas import (
    ConservativeMacroOutputs,
    OUTPUT_SIZE,
    macro_output_component_scales,
    pack_macro_outputs,
    unpack_macro_outputs,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_thermodynamic_macro_atlas import (
    ThermodynamicAffineMacroAtlas,
)


def _outputs() -> ConservativeMacroOutputs:
    packed = np.zeros(OUTPUT_SIZE)
    packed[:51] = -np.arange(51, dtype=float) - 1.0
    packed[51:] = np.arange(64, dtype=float) + 2.0
    return unpack_macro_outputs(packed)


def test_thermodynamic_atlas_applies_block_pullback() -> None:
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
    atlas = ThermodynamicAffineMacroAtlas(
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
    candidate[0, 0] *= 1.1
    with pytest.raises(ValueError, match="trust box"):
        atlas.evaluate(candidate)
