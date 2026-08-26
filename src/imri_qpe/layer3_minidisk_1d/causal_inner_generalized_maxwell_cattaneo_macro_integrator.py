"""Exact affine propagation and offline chart inversion for the macro atlas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from imri_qpe.constants import C

from .causal_inner_generalized_maxwell_cattaneo_macro_atlas import (
    MACRO_CELLS,
    MACRO_FIELDS,
    OUTPUT_SIZE,
    pack_macro_outputs,
    restrict_entropy_complete_macro,
)
from .causal_inner_generalized_maxwell_cattaneo_slow_manifold import (
    generalized_maxwell_cattaneo_slow_targets,
)
from .causal_inner_generalized_maxwell_cattaneo_thermodynamic_macro_atlas import (
    ThermodynamicAffineMacroAtlas,
    thermodynamic_chart_lift,
)


STATE_SIZE = MACRO_CELLS * MACRO_FIELDS
AUGMENTED_STATE_SIZE = STATE_SIZE + 1


def macro_rate_output_matrix() -> np.ndarray:
    """Return the exact linear map from packed outputs to the 80 state rates."""

    matrix = np.zeros((STATE_SIZE, OUTPUT_SIZE), dtype=float)
    for cell in range(MACRO_CELLS):
        for field in range(3):
            matrix[5 * cell + field, 3 * cell + field] += C
            matrix[5 * cell + field, 3 * (cell + 1) + field] -= C
        for field in range(1, 3):
            matrix[5 * cell + field, 51 + 2 * cell + field - 1] += C
        for field in range(2):
            matrix[5 * cell + 3 + field, 83 + 2 * cell + field] = 1.0
    return matrix


def _block_diagonal_pullback(pullbacks: np.ndarray) -> np.ndarray:
    blocks = np.asarray(pullbacks, dtype=float)
    if blocks.shape != (MACRO_CELLS, MACRO_FIELDS, MACRO_FIELDS):
        raise ValueError("macro pullback blocks have the wrong shape")
    matrix = np.zeros((STATE_SIZE, STATE_SIZE), dtype=float)
    for cell in range(MACRO_CELLS):
        start = MACRO_FIELDS * cell
        matrix[start : start + MACRO_FIELDS, start : start + MACRO_FIELDS] = (
            blocks[cell]
        )
    return matrix


@dataclass(frozen=True)
class ExactAffineMacroSystem:
    """The normalized affine system induced by one certified atlas patch."""

    atlas: ThermodynamicAffineMacroAtlas
    rate_output_matrix: np.ndarray
    block_pullback: np.ndarray
    normalized_output_state_matrix: np.ndarray
    normalized_rate_matrix: np.ndarray
    normalized_rate_offset: np.ndarray
    augmented_generator: np.ndarray

    @classmethod
    def from_atlas(cls, atlas: ThermodynamicAffineMacroAtlas):
        rate_output = macro_rate_output_matrix()
        pullback = _block_diagonal_pullback(atlas.macro_coordinate_pullback)
        jacobian = np.asarray(atlas.normalized_output_jacobian, dtype=float)
        state_scales = np.asarray(atlas.macro_coordinate_scales, dtype=float).ravel()
        output_scales = np.asarray(atlas.output_component_scales, dtype=float)
        base = np.asarray(atlas.base_normalized_output, dtype=float)
        if (
            jacobian.shape != (OUTPUT_SIZE, STATE_SIZE)
            or state_scales.shape != (STATE_SIZE,)
            or np.any(state_scales <= 0.0)
            or output_scales.shape != (OUTPUT_SIZE,)
            or base.shape != (OUTPUT_SIZE,)
        ):
            raise ValueError("affine macro atlas arrays are inconsistent")
        normalized_output_state = jacobian @ pullback
        physical_output_state = output_scales[:, None] * normalized_output_state
        physical_base_output = output_scales * base
        normalized_rate = (
            rate_output @ physical_output_state
        ) / state_scales[:, None]
        normalized_offset = (rate_output @ physical_base_output) / state_scales
        generator = np.zeros(
            (AUGMENTED_STATE_SIZE, AUGMENTED_STATE_SIZE), dtype=float
        )
        generator[:STATE_SIZE, :STATE_SIZE] = normalized_rate
        generator[:STATE_SIZE, STATE_SIZE] = normalized_offset
        return cls(
            atlas=atlas,
            rate_output_matrix=rate_output,
            block_pullback=pullback,
            normalized_output_state_matrix=normalized_output_state,
            normalized_rate_matrix=normalized_rate,
            normalized_rate_offset=normalized_offset,
            augmented_generator=generator,
        )

    def normalized_state(self, macro_state) -> np.ndarray:
        state = np.asarray(macro_state, dtype=float)
        anchor = np.asarray(self.atlas.anchor_macro_state, dtype=float)
        scales = np.asarray(self.atlas.macro_coordinate_scales, dtype=float)
        if state.shape != (MACRO_CELLS, MACRO_FIELDS):
            raise ValueError("macro state has the wrong shape")
        return ((state - anchor) / scales).ravel()

    def macro_state(self, normalized_state) -> np.ndarray:
        values = np.asarray(normalized_state, dtype=float)
        if values.shape != (STATE_SIZE,) or np.any(~np.isfinite(values)):
            raise ValueError("normalized macro state is invalid")
        return np.asarray(self.atlas.anchor_macro_state) + np.asarray(
            self.atlas.macro_coordinate_scales
        ) * values.reshape(MACRO_CELLS, MACRO_FIELDS)

    def augmented_state(self, macro_state) -> np.ndarray:
        return np.concatenate((self.normalized_state(macro_state), [1.0]))

    def inferred_chart_coordinate(self, macro_state) -> np.ndarray:
        return (
            self.block_pullback @ self.normalized_state(macro_state)
        ).reshape(MACRO_CELLS, MACRO_FIELDS)

    def normalized_packed_output(self, macro_state) -> np.ndarray:
        q = self.normalized_state(macro_state)
        return np.asarray(self.atlas.base_normalized_output) + (
            self.normalized_output_state_matrix @ q
        )

    def packed_output(self, macro_state) -> np.ndarray:
        return np.asarray(self.atlas.output_component_scales) * (
            self.normalized_packed_output(macro_state)
        )

    def macro_rate(self, macro_state) -> np.ndarray:
        return (
            self.rate_output_matrix @ self.packed_output(macro_state)
        ).reshape(MACRO_CELLS, MACRO_FIELDS)


@dataclass(frozen=True)
class ExactAffineMacroStep:
    previous_macro_state: np.ndarray
    macro_state: np.ndarray
    integrated_packed_output: np.ndarray
    state_ledger_relative_defect: float
    maximum_endpoint_chart_coordinate: float


@dataclass(frozen=True)
class ExactAffineMacroTransition:
    """A precomputed state and exact output-integral map for one timestep."""

    system: ExactAffineMacroSystem
    timestep_seconds: float
    state_transition: np.ndarray
    normalized_output_integral: np.ndarray
    trust_coordinate_infinity: float

    @classmethod
    def build(
        cls,
        system: ExactAffineMacroSystem,
        timestep_seconds: float,
        *,
        trust_coordinate_infinity: float,
    ):
        timestep = float(timestep_seconds)
        trust = float(trust_coordinate_infinity)
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("macro timestep must be positive and finite")
        if not np.isfinite(trust) or trust <= 0.0:
            raise ValueError("macro trust coordinate must be positive and finite")
        total = AUGMENTED_STATE_SIZE + OUTPUT_SIZE
        augmented = np.zeros((total, total), dtype=float)
        augmented[:AUGMENTED_STATE_SIZE, :AUGMENTED_STATE_SIZE] = (
            system.augmented_generator
        )
        augmented[
            AUGMENTED_STATE_SIZE:, :STATE_SIZE
        ] = system.normalized_output_state_matrix
        augmented[AUGMENTED_STATE_SIZE:, STATE_SIZE] = np.asarray(
            system.atlas.base_normalized_output
        )
        transition = expm(timestep * augmented)
        return cls(
            system=system,
            timestep_seconds=timestep,
            state_transition=np.asarray(
                transition[:AUGMENTED_STATE_SIZE, :AUGMENTED_STATE_SIZE]
            ),
            normalized_output_integral=np.asarray(
                transition[AUGMENTED_STATE_SIZE:, :AUGMENTED_STATE_SIZE]
            ),
            trust_coordinate_infinity=trust,
        )

    def apply_augmented(self, augmented_state) -> np.ndarray:
        values = np.asarray(augmented_state, dtype=float)
        if values.shape != (AUGMENTED_STATE_SIZE,):
            raise ValueError("augmented macro state has the wrong shape")
        return self.state_transition @ values

    def step(self, macro_state) -> ExactAffineMacroStep:
        previous = np.asarray(macro_state, dtype=float)
        augmented = self.system.augmented_state(previous)
        if (
            float(
                np.max(
                    np.abs(self.system.inferred_chart_coordinate(previous))
                )
            )
            > self.trust_coordinate_infinity
        ):
            raise ValueError("macro step starts outside the atlas trust box")
        advanced = self.apply_augmented(augmented)
        if abs(float(advanced[-1]) - 1.0) > 1.0e-13:
            raise RuntimeError("affine homogeneous coordinate did not close")
        next_state = self.system.macro_state(advanced[:STATE_SIZE])
        maximum_chart = float(
            np.max(np.abs(self.system.inferred_chart_coordinate(next_state)))
        )
        if maximum_chart > self.trust_coordinate_infinity:
            raise ValueError("macro step endpoint leaves the atlas trust box")
        if np.any(next_state[:, :3] <= 0.0):
            raise ValueError("macro step produces nonpositive M/J/E")
        if np.any(np.abs(next_state[:, 3]) >= 1.0):
            raise ValueError("macro step produces superluminal radial velocity")
        integrated_output = np.asarray(self.system.atlas.output_component_scales) * (
            self.normalized_output_integral @ augmented
        )
        actual_change = np.asarray(
            self.system.atlas.macro_coordinate_scales, dtype=float
        ).ravel() * (advanced[:STATE_SIZE] - augmented[:STATE_SIZE])
        ledger_change = self.system.rate_output_matrix @ integrated_output
        scale = max(
            float(np.max(np.abs(actual_change))),
            float(np.max(np.abs(ledger_change))),
            np.finfo(float).tiny,
        )
        ledger_defect = float(np.max(np.abs(actual_change - ledger_change)) / scale)
        return ExactAffineMacroStep(
            previous_macro_state=np.array(previous, copy=True),
            macro_state=np.asarray(next_state),
            integrated_packed_output=integrated_output,
            state_ledger_relative_defect=ledger_defect,
            maximum_endpoint_chart_coordinate=maximum_chart,
        )


@dataclass(frozen=True)
class ThermodynamicMacroReconstruction:
    primitive_charts: np.ndarray
    macro_state: np.ndarray
    chart_coordinates: np.ndarray
    maximum_macro_state_roundtrip_relative_defect: float
    newton_corrections: int
    maximum_local_jacobian_condition_number: float


def _macro_from_chart_coordinate(context, anchor_charts, coordinate) -> tuple[np.ndarray, np.ndarray]:
    charts = thermodynamic_chart_lift(context, anchor_charts, coordinate)
    targets = generalized_maxwell_cattaneo_slow_targets(context, charts)
    return restrict_entropy_complete_macro(targets, charts), charts


def reconstruct_thermodynamic_macro_state(
    context,
    anchor_primitive_charts,
    target_macro_state,
    *,
    anchor_macro_state,
    macro_coordinate_scales,
    macro_coordinate_pullbacks,
    derivative_step: float = 1.0e-5,
    maximum_newton_corrections: int = 8,
    relative_tolerance: float = 1.0e-10,
    maximum_chart_coordinate_infinity: float = 0.12,
) -> ThermodynamicMacroReconstruction:
    """Invert one patch through independent blockwise thermodynamic Newton solves."""

    target = np.asarray(target_macro_state, dtype=float)
    anchor_macro = np.asarray(anchor_macro_state, dtype=float)
    scales = np.asarray(macro_coordinate_scales, dtype=float)
    pullbacks = np.asarray(macro_coordinate_pullbacks, dtype=float)
    step = float(derivative_step)
    maximum = int(maximum_newton_corrections)
    tolerance = float(relative_tolerance)
    trust = float(maximum_chart_coordinate_infinity)
    if (
        target.shape != (MACRO_CELLS, MACRO_FIELDS)
        or anchor_macro.shape != target.shape
        or scales.shape != target.shape
        or np.any(scales <= 0.0)
        or pullbacks.shape != (MACRO_CELLS, MACRO_FIELDS, MACRO_FIELDS)
        or step <= 0.0
        or maximum < 0
        or tolerance <= 0.0
        or trust <= 0.0
    ):
        raise ValueError("thermodynamic macro reconstruction inputs are invalid")
    raw = (target - anchor_macro) / scales
    coordinate = np.einsum("kij,kj->ki", pullbacks, raw)
    greatest_condition = 0.0
    for correction in range(maximum + 1):
        current, charts = _macro_from_chart_coordinate(
            context, anchor_primitive_charts, coordinate
        )
        residual = (current - target) / scales
        defect = float(np.max(np.abs(residual)))
        if defect <= tolerance:
            return ThermodynamicMacroReconstruction(
                primitive_charts=charts,
                macro_state=current,
                chart_coordinates=np.asarray(coordinate),
                maximum_macro_state_roundtrip_relative_defect=defect,
                newton_corrections=correction,
                maximum_local_jacobian_condition_number=greatest_condition,
            )
        if correction == maximum:
            break
        tangents = np.empty(
            (MACRO_CELLS, MACRO_FIELDS, MACRO_FIELDS), dtype=float
        )
        for field in range(MACRO_FIELDS):
            plus_coordinate = np.array(coordinate, copy=True)
            minus_coordinate = np.array(coordinate, copy=True)
            plus_coordinate[:, field] += step
            minus_coordinate[:, field] -= step
            plus, _ = _macro_from_chart_coordinate(
                context, anchor_primitive_charts, plus_coordinate
            )
            minus, _ = _macro_from_chart_coordinate(
                context, anchor_primitive_charts, minus_coordinate
            )
            tangents[:, :, field] = (plus - minus) / (2.0 * step * scales)
        conditions = np.linalg.cond(tangents)
        greatest_condition = max(greatest_condition, float(np.max(conditions)))
        corrections = np.asarray(
            [
                np.linalg.solve(tangents[cell], -residual[cell])
                for cell in range(MACRO_CELLS)
            ]
        )
        accepted = False
        for backtrack in range(8):
            candidate = coordinate + (0.5**backtrack) * corrections
            if float(np.max(np.abs(candidate))) > trust:
                continue
            candidate_macro, _ = _macro_from_chart_coordinate(
                context, anchor_primitive_charts, candidate
            )
            candidate_defect = float(
                np.max(np.abs((candidate_macro - target) / scales))
            )
            if candidate_defect < defect:
                coordinate = candidate
                accepted = True
                break
        if not accepted:
            raise RuntimeError("thermodynamic macro reconstruction line search failed")
    raise RuntimeError("thermodynamic macro reconstruction did not converge")


__all__ = (
    "AUGMENTED_STATE_SIZE",
    "STATE_SIZE",
    "ExactAffineMacroStep",
    "ExactAffineMacroSystem",
    "ExactAffineMacroTransition",
    "ThermodynamicMacroReconstruction",
    "macro_rate_output_matrix",
    "reconstruct_thermodynamic_macro_state",
)
