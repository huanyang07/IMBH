"""Admissible thermodynamic coordinates for the conservative macro atlas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .causal_inner_generalized_maxwell_cattaneo_macro_atlas import (
    FINE_CELLS,
    FINE_PER_MACRO,
    MACRO_CELLS,
    MACRO_FIELDS,
    ConservativeMacroOutputs,
    restrict_entropy_complete_macro,
    unpack_macro_outputs,
)
from .causal_inner_generalized_maxwell_cattaneo_semidiscrete import (
    generalized_maxwell_cattaneo_hydrostatic_embedding,
)
from .causal_inner_generalized_maxwell_cattaneo_slow_manifold import (
    generalized_maxwell_cattaneo_slow_targets,
)


THERMODYNAMIC_CHART_SCALES = np.asarray((1.0, 0.1, 1.0, 0.1, 1.0e-4))


def _macro_state(values) -> np.ndarray:
    state = np.asarray(values, dtype=float)
    if state.shape != (MACRO_CELLS, MACRO_FIELDS) or np.any(~np.isfinite(state)):
        raise ValueError("macro state must be finite and have shape (16, 5)")
    if np.any(state[:, :3] <= 0.0):
        raise ValueError("macro M/J/E coordinates must remain positive")
    if np.any(np.abs(state[:, 3]) >= 1.0):
        raise ValueError("macro radial velocity must remain subluminal")
    return state


@dataclass(frozen=True)
class ThermodynamicAffineMacroAtlas:
    """One conservative affine patch addressed through local chart pullbacks."""

    anchor_macro_state: np.ndarray
    macro_coordinate_scales: np.ndarray
    base_normalized_output: np.ndarray
    normalized_output_jacobian: np.ndarray
    output_component_scales: np.ndarray
    trust_coordinate_infinity: float
    macro_coordinate_pullback: np.ndarray

    def inferred_chart_coordinate(self, macro_state) -> np.ndarray:
        state = _macro_state(macro_state)
        scales = np.asarray(self.macro_coordinate_scales, dtype=float)
        pullback = np.asarray(self.macro_coordinate_pullback, dtype=float)
        if scales.shape != (MACRO_CELLS, MACRO_FIELDS):
            raise ValueError("macro coordinate scales have the wrong shape")
        if pullback.shape != (MACRO_CELLS, MACRO_FIELDS, MACRO_FIELDS):
            raise ValueError("macro coordinate pullback has the wrong shape")
        raw = (state - self.anchor_macro_state) / scales
        return np.einsum("kij,kj->ki", pullback, raw)

    def evaluate(self, macro_state) -> ConservativeMacroOutputs:
        coordinate = self.inferred_chart_coordinate(macro_state)
        if float(np.max(np.abs(coordinate))) > float(
            self.trust_coordinate_infinity
        ):
            raise ValueError("macro state leaves the affine atlas trust box")
        normalized = (
            self.base_normalized_output
            + self.normalized_output_jacobian @ coordinate.ravel()
        )
        return unpack_macro_outputs(normalized * self.output_component_scales)


def thermodynamic_chart_lift(
    context, anchor_primitive_charts, chart_coordinates
) -> np.ndarray:
    """Generate a hydrostatic truth state from 16 admissible local charts.

    Coordinate order is ``(lnSigma, beta_phi, lnT, beta_r, chi)``.  Each
    shift is shared by the seven fine cells belonging to one macro block.
    """

    anchor = np.asarray(anchor_primitive_charts, dtype=float)
    coordinates = np.asarray(chart_coordinates, dtype=float)
    if (
        anchor.shape != (FINE_CELLS, 7)
        or coordinates.shape != (MACRO_CELLS, MACRO_FIELDS)
        or int(context.grid.centers.size) != FINE_CELLS
        or np.any(~np.isfinite(anchor))
        or np.any(~np.isfinite(coordinates))
    ):
        raise ValueError("thermodynamic chart lift inputs are invalid")
    repeated = np.repeat(coordinates, FINE_PER_MACRO, axis=0)
    chart5 = np.array(anchor[:, :5], copy=True)
    chart5[:, 0] += THERMODYNAMIC_CHART_SCALES[0] * repeated[:, 0]
    chart5[:, 2] += THERMODYNAMIC_CHART_SCALES[1] * repeated[:, 1]
    chart5[:, 3] += THERMODYNAMIC_CHART_SCALES[2] * repeated[:, 2]
    chart5[:, 1] += THERMODYNAMIC_CHART_SCALES[3] * repeated[:, 3]
    chart5[:, 4] += THERMODYNAMIC_CHART_SCALES[4] * repeated[:, 4]
    if np.any(chart5[:, 1] ** 2 + chart5[:, 2] ** 2 >= 1.0):
        raise ValueError("thermodynamic chart lift leaves the subluminal cone")
    return np.asarray(
        [
            generalized_maxwell_cattaneo_hydrostatic_embedding(
                values,
                proper_vertical_frequency=float(
                    context.vertical_frequency.frequency(float(radius))
                ),
            )
            for radius, values in zip(context.grid.centers, chart5, strict=True)
        ]
    )


def thermodynamic_macro_chart_pullback(
    context,
    anchor_primitive_charts,
    *,
    derivative_step: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return anchor macro state, scales, chart tangents, and their inverses."""

    anchor = np.asarray(anchor_primitive_charts, dtype=float)
    step = float(derivative_step)
    if anchor.shape != (FINE_CELLS, 7) or not np.isfinite(step) or step <= 0.0:
        raise ValueError("thermodynamic macro pullback inputs are invalid")
    anchor_targets = generalized_maxwell_cattaneo_slow_targets(context, anchor)
    anchor_macro = restrict_entropy_complete_macro(anchor_targets, anchor)
    field_maximum = np.max(np.abs(anchor_macro), axis=0)
    scales = np.maximum(np.abs(anchor_macro), 1.0e-12 * field_maximum[None, :])
    tangents = np.empty((MACRO_CELLS, MACRO_FIELDS, MACRO_FIELDS), dtype=float)
    for cell in range(MACRO_CELLS):
        for coordinate in range(MACRO_FIELDS):
            direction = np.zeros((MACRO_CELLS, MACRO_FIELDS), dtype=float)
            direction[cell, coordinate] = step
            plus_charts = thermodynamic_chart_lift(context, anchor, direction)
            minus_charts = thermodynamic_chart_lift(context, anchor, -direction)
            plus_targets = generalized_maxwell_cattaneo_slow_targets(
                context, plus_charts
            )
            minus_targets = generalized_maxwell_cattaneo_slow_targets(
                context, minus_charts
            )
            plus_macro = restrict_entropy_complete_macro(plus_targets, plus_charts)
            minus_macro = restrict_entropy_complete_macro(minus_targets, minus_charts)
            tangents[cell, :, coordinate] = (
                plus_macro[cell] - minus_macro[cell]
            ) / (2.0 * step * scales[cell])
    pullbacks = np.asarray([np.linalg.inv(matrix) for matrix in tangents])
    return anchor_macro, scales, tangents, pullbacks


__all__ = (
    "THERMODYNAMIC_CHART_SCALES",
    "ThermodynamicAffineMacroAtlas",
    "thermodynamic_chart_lift",
    "thermodynamic_macro_chart_pullback",
)
