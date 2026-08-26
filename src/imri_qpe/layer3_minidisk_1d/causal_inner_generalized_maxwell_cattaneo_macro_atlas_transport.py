"""Exact coordinate transport for a thermodynamic affine macro patch.

The transported derivative represents the same physical output tangent as
the source patch.  Only its input chart, state/output normalization, and base
output are changed.  This module is separate from the already certified atlas
implementation so its provenance remains immutable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .causal_inner_generalized_maxwell_cattaneo_macro_atlas import (
    MACRO_CELLS,
    MACRO_FIELDS,
    ConservativeMacroOutputs,
    macro_output_component_scales,
    pack_macro_outputs,
)
from .causal_inner_generalized_maxwell_cattaneo_thermodynamic_macro_atlas import (
    ThermodynamicAffineMacroAtlas,
)


@dataclass(frozen=True)
class TransportedThermodynamicMacroPatch:
    """A reanchored atlas plus coordinate-invariance diagnostics."""

    atlas: ThermodynamicAffineMacroAtlas
    old_to_new_chart_transport: np.ndarray
    maximum_transport_block_condition_number: float
    new_pullback_inverse_closure_infinity: float
    physical_output_tangent_relative_infinity_defect: float


def _block_diagonal(blocks: np.ndarray) -> np.ndarray:
    values = np.asarray(blocks, dtype=float)
    if values.shape != (MACRO_CELLS, MACRO_FIELDS, MACRO_FIELDS):
        raise ValueError("transport blocks have the wrong shape")
    result = np.zeros(
        (MACRO_CELLS * MACRO_FIELDS, MACRO_CELLS * MACRO_FIELDS), dtype=float
    )
    for cell, block in enumerate(values):
        start = cell * MACRO_FIELDS
        result[start : start + MACRO_FIELDS, start : start + MACRO_FIELDS] = block
    return result


def _physical_output_tangent(atlas: ThermodynamicAffineMacroAtlas) -> np.ndarray:
    """Differentiate packed physical outputs with respect to physical X."""

    scales = np.asarray(atlas.macro_coordinate_scales, dtype=float)
    pullbacks = np.asarray(atlas.macro_coordinate_pullback, dtype=float)
    output_scales = np.asarray(atlas.output_component_scales, dtype=float)
    normalized_jacobian = np.asarray(atlas.normalized_output_jacobian, dtype=float)
    state_pullback_blocks = np.asarray(
        [pullbacks[cell] @ np.diag(1.0 / scales[cell]) for cell in range(MACRO_CELLS)]
    )
    return (
        output_scales[:, None]
        * normalized_jacobian
        @ _block_diagonal(state_pullback_blocks)
    )


def transport_thermodynamic_affine_macro_atlas(
    source: ThermodynamicAffineMacroAtlas,
    *,
    new_anchor_macro_state,
    new_macro_coordinate_scales,
    new_macro_chart_tangents,
    new_macro_coordinate_pullbacks,
    new_base_outputs: ConservativeMacroOutputs,
    trust_coordinate_infinity: float,
) -> TransportedThermodynamicMacroPatch:
    """Reexpress a source output tangent in a new thermodynamic chart.

    If ``q_i=(X-X_i)/S_Xi`` and ``z_i=P_i q_i``, the new chart tangent
    satisfies ``dX=diag(S_X3) T_3 dz_3``.  Hence

    ``dz_2/dz_3 = P_2 diag(S_X3/S_X2) T_3``.

    The output normalization is changed independently, leaving the physical
    derivative ``dY/dX`` invariant up to floating-point roundoff.
    """

    anchor = np.asarray(new_anchor_macro_state, dtype=float)
    new_scales = np.asarray(new_macro_coordinate_scales, dtype=float)
    new_tangents = np.asarray(new_macro_chart_tangents, dtype=float)
    new_pullbacks = np.asarray(new_macro_coordinate_pullbacks, dtype=float)
    old_scales = np.asarray(source.macro_coordinate_scales, dtype=float)
    old_pullbacks = np.asarray(source.macro_coordinate_pullback, dtype=float)
    trust = float(trust_coordinate_infinity)
    expected_state = (MACRO_CELLS, MACRO_FIELDS)
    expected_maps = (MACRO_CELLS, MACRO_FIELDS, MACRO_FIELDS)
    if (
        anchor.shape != expected_state
        or new_scales.shape != expected_state
        or old_scales.shape != expected_state
        or new_tangents.shape != expected_maps
        or new_pullbacks.shape != expected_maps
        or old_pullbacks.shape != expected_maps
        or any(
            np.any(~np.isfinite(item))
            for item in (
                anchor,
                new_scales,
                old_scales,
                new_tangents,
                new_pullbacks,
                old_pullbacks,
            )
        )
        or np.any(new_scales <= 0.0)
        or np.any(old_scales <= 0.0)
        or not np.isfinite(trust)
        or trust <= 0.0
    ):
        raise ValueError("transported thermodynamic atlas inputs are invalid")
    transport_blocks = np.asarray(
        [
            old_pullbacks[cell]
            @ np.diag(new_scales[cell] / old_scales[cell])
            @ new_tangents[cell]
            for cell in range(MACRO_CELLS)
        ]
    )
    transport = _block_diagonal(transport_blocks)
    new_output_scales = macro_output_component_scales(new_base_outputs)
    new_base = pack_macro_outputs(new_base_outputs) / new_output_scales
    physical_chart_tangent = (
        np.asarray(source.output_component_scales)[:, None]
        * np.asarray(source.normalized_output_jacobian)
        @ transport
    )
    new_normalized_jacobian = physical_chart_tangent / new_output_scales[:, None]
    atlas = ThermodynamicAffineMacroAtlas(
        anchor_macro_state=np.array(anchor, copy=True),
        macro_coordinate_scales=np.array(new_scales, copy=True),
        base_normalized_output=np.array(new_base, copy=True),
        normalized_output_jacobian=np.array(new_normalized_jacobian, copy=True),
        output_component_scales=np.array(new_output_scales, copy=True),
        trust_coordinate_infinity=trust,
        macro_coordinate_pullback=np.array(new_pullbacks, copy=True),
    )
    closure = float(
        np.max(
            np.abs(
                np.einsum("kij,kjl->kil", new_pullbacks, new_tangents)
                - np.eye(MACRO_FIELDS)[None, :, :]
            )
        )
    )
    old_physical = _physical_output_tangent(source)
    new_physical = _physical_output_tangent(atlas)
    physical_defect = float(
        np.linalg.norm(new_physical - old_physical, ord=np.inf)
        / max(np.linalg.norm(old_physical, ord=np.inf), np.finfo(float).tiny)
    )
    return TransportedThermodynamicMacroPatch(
        atlas=atlas,
        old_to_new_chart_transport=transport,
        maximum_transport_block_condition_number=float(
            np.max(np.linalg.cond(transport_blocks))
        ),
        new_pullback_inverse_closure_infinity=closure,
        physical_output_tangent_relative_infinity_defect=physical_defect,
    )


__all__ = (
    "TransportedThermodynamicMacroPatch",
    "transport_thermodynamic_affine_macro_atlas",
)
