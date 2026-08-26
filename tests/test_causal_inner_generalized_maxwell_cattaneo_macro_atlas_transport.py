from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_atlas import (
    ConservativeMacroOutputs,
    OUTPUT_SIZE,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_atlas_transport import (
    transport_thermodynamic_affine_macro_atlas,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_thermodynamic_macro_atlas import (
    ThermodynamicAffineMacroAtlas,
)


def _outputs(scale: float) -> ConservativeMacroOutputs:
    flux = scale * np.arange(51, dtype=float).reshape(17, 3) + 1.0
    source = np.zeros((16, 3))
    source[:, 1:] = scale * np.arange(32, dtype=float).reshape(16, 2) + 2.0
    auxiliary = scale * np.arange(32, dtype=float).reshape(16, 2) + 3.0
    rates = np.column_stack((np.ones((16, 3)), auxiliary))
    return ConservativeMacroOutputs(flux, source, auxiliary, rates)


def test_transport_preserves_the_physical_output_tangent() -> None:
    generator = np.random.default_rng(14)
    old_scales = np.exp(generator.normal(size=(16, 5)))
    old_pullbacks = np.asarray(
        [np.eye(5) + 0.01 * generator.normal(size=(5, 5)) for _ in range(16)]
    )
    source = ThermodynamicAffineMacroAtlas(
        anchor_macro_state=np.column_stack(
            (
                np.ones((16, 3)),
                np.zeros((16, 1)),
                1.0e-4 * np.ones((16, 1)),
            )
        ),
        macro_coordinate_scales=old_scales,
        base_normalized_output=np.ones(OUTPUT_SIZE),
        normalized_output_jacobian=generator.normal(size=(OUTPUT_SIZE, 80)),
        output_component_scales=np.exp(generator.normal(size=OUTPUT_SIZE)),
        trust_coordinate_infinity=0.15,
        macro_coordinate_pullback=old_pullbacks,
    )
    new_scales = np.exp(generator.normal(size=(16, 5)))
    new_tangents = np.asarray(
        [np.eye(5) + 0.01 * generator.normal(size=(5, 5)) for _ in range(16)]
    )
    new_pullbacks = np.asarray([np.linalg.inv(item) for item in new_tangents])
    transported = transport_thermodynamic_affine_macro_atlas(
        source,
        new_anchor_macro_state=source.anchor_macro_state,
        new_macro_coordinate_scales=new_scales,
        new_macro_chart_tangents=new_tangents,
        new_macro_coordinate_pullbacks=new_pullbacks,
        new_base_outputs=_outputs(0.1),
        trust_coordinate_infinity=0.15,
    )
    assert transported.new_pullback_inverse_closure_infinity < 1.0e-12
    assert transported.physical_output_tangent_relative_infinity_defect < 1.0e-12
    assert transported.old_to_new_chart_transport.shape == (80, 80)


def test_transport_rejects_nonpositive_state_scales() -> None:
    source = ThermodynamicAffineMacroAtlas(
        anchor_macro_state=np.column_stack(
            (np.ones((16, 3)), np.zeros((16, 1)), 1.0e-4 * np.ones((16, 1)))
        ),
        macro_coordinate_scales=np.ones((16, 5)),
        base_normalized_output=np.ones(OUTPUT_SIZE),
        normalized_output_jacobian=np.ones((OUTPUT_SIZE, 80)),
        output_component_scales=np.ones(OUTPUT_SIZE),
        trust_coordinate_infinity=0.15,
        macro_coordinate_pullback=np.repeat(np.eye(5)[None, :, :], 16, axis=0),
    )
    invalid = np.ones((16, 5))
    invalid[0, 0] = 0.0
    try:
        transport_thermodynamic_affine_macro_atlas(
            source,
            new_anchor_macro_state=source.anchor_macro_state,
            new_macro_coordinate_scales=invalid,
            new_macro_chart_tangents=np.repeat(np.eye(5)[None, :, :], 16, axis=0),
            new_macro_coordinate_pullbacks=np.repeat(
                np.eye(5)[None, :, :], 16, axis=0
            ),
            new_base_outputs=_outputs(0.1),
            trust_coordinate_infinity=0.15,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive state scales were accepted")
