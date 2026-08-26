"""Conservative fine/macro maps for the entropy-complete partial equilibrium.

The online state has sixteen radial blocks with exact integrated mass,
angular momentum, and total energy, plus mass-weighted radial velocity and
causal shear stress.  Atlas outputs retain single-valued face fluxes, so the
three conservative ledgers close algebraically at every online evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C

from .causal_inner_generalized_maxwell_cattaneo_quasisteady import (
    HydrostaticInvariantReconstruction,
    reconstruct_hydrostatic_fixed_invariants,
)


FINE_CELLS = 112
MACRO_CELLS = 16
FINE_PER_MACRO = 7
MACRO_FIELDS = 5
OUTPUT_SIZE = 115
SLOW_ROWS = np.asarray((0, 2, 3), dtype=int)


@dataclass(frozen=True)
class ConservativeMacroOutputs:
    """Physical coarse outputs and their induced 16-by-5 state rate."""

    MJE_face_fluxes_over_c: np.ndarray
    MJE_cell_sources_per_ct: np.ndarray
    auxiliary_rates_per_second: np.ndarray
    macro_rates_per_second: np.ndarray


@dataclass(frozen=True)
class ConservativeAffineMacroAtlas:
    """One radius-one affine atlas patch in normalized coordinates."""

    anchor_macro_state: np.ndarray
    macro_coordinate_scales: np.ndarray
    base_normalized_output: np.ndarray
    normalized_output_jacobian: np.ndarray
    output_component_scales: np.ndarray
    trust_coordinate_infinity: float

    def evaluate(self, macro_state) -> ConservativeMacroOutputs:
        state = _macro_state(macro_state)
        coordinate = (state - self.anchor_macro_state) / self.macro_coordinate_scales
        if float(np.max(np.abs(coordinate))) > float(self.trust_coordinate_infinity):
            raise ValueError("macro state leaves the affine atlas trust box")
        normalized = self.base_normalized_output + self.normalized_output_jacobian @ coordinate.ravel()
        return unpack_macro_outputs(normalized * self.output_component_scales)


def _macro_state(values) -> np.ndarray:
    state = np.asarray(values, dtype=float)
    if state.shape != (MACRO_CELLS, MACRO_FIELDS) or np.any(~np.isfinite(state)):
        raise ValueError("macro state must be finite and have shape (16, 5)")
    if np.any(state[:, :3] <= 0.0):
        raise ValueError("macro M/J/E coordinates must remain positive")
    if np.any(np.abs(state[:, 3]) >= 1.0):
        raise ValueError("macro radial velocity must remain subluminal")
    return state


def restrict_entropy_complete_macro(slow_targets_MJE, primitive_charts) -> np.ndarray:
    """Restrict 112 hydrostatic truth cells to the exact 16-cell macro ledger."""

    targets = np.asarray(slow_targets_MJE, dtype=float)
    charts = np.asarray(primitive_charts, dtype=float)
    if (
        targets.shape != (FINE_CELLS, 3)
        or charts.shape != (FINE_CELLS, 7)
        or np.any(~np.isfinite(targets))
        or np.any(~np.isfinite(charts))
        or np.any(targets <= 0.0)
    ):
        raise ValueError("fine entropy-complete macro restriction inputs are invalid")
    grouped_targets = targets.reshape(MACRO_CELLS, FINE_PER_MACRO, 3)
    macro_targets = np.sum(grouped_targets, axis=1)
    mass = grouped_targets[:, :, 0]
    weights = mass / np.sum(mass, axis=1)[:, None]
    grouped_charts = charts.reshape(MACRO_CELLS, FINE_PER_MACRO, 7)
    radial = np.sum(weights * grouped_charts[:, :, 1], axis=1)
    stress = np.sum(weights * grouped_charts[:, :, 4], axis=1)
    return np.column_stack((macro_targets, radial, stress))


def prolong_entropy_complete_macro(
    context,
    anchor_slow_targets_MJE,
    anchor_primitive_charts,
    macro_state,
    *,
    constraint_tolerance: float = 1.0e-10,
) -> HydrostaticInvariantReconstruction:
    """Lift a macro state with fixed anchor subcell shapes and exact ledgers."""

    anchor_targets = np.asarray(anchor_slow_targets_MJE, dtype=float)
    anchor_charts = np.asarray(anchor_primitive_charts, dtype=float)
    state = _macro_state(macro_state)
    if (
        anchor_targets.shape != (FINE_CELLS, 3)
        or anchor_charts.shape != (FINE_CELLS, 7)
        or int(context.grid.centers.size) != FINE_CELLS
        or np.any(anchor_targets <= 0.0)
    ):
        raise ValueError("entropy-complete macro prolongation anchor is invalid")
    grouped_targets = anchor_targets.reshape(MACRO_CELLS, FINE_PER_MACRO, 3)
    anchor_macro_targets = np.sum(grouped_targets, axis=1)
    fractions = grouped_targets / anchor_macro_targets[:, None, :]
    lifted_targets = (fractions * state[:, None, :3]).reshape(FINE_CELLS, 3)
    new_mass = lifted_targets[:, 0].reshape(MACRO_CELLS, FINE_PER_MACRO)
    new_weights = new_mass / np.sum(new_mass, axis=1)[:, None]
    grouped_charts = anchor_charts.reshape(MACRO_CELLS, FINE_PER_MACRO, 7)
    anchor_radial_mean = np.sum(new_weights * grouped_charts[:, :, 1], axis=1)
    anchor_stress_mean = np.sum(new_weights * grouped_charts[:, :, 4], axis=1)
    radial = (
        grouped_charts[:, :, 1]
        + (state[:, 3] - anchor_radial_mean)[:, None]
    ).reshape(FINE_CELLS)
    stress = (
        grouped_charts[:, :, 4]
        + (state[:, 4] - anchor_stress_mean)[:, None]
    ).reshape(FINE_CELLS)
    return reconstruct_hydrostatic_fixed_invariants(
        context,
        lifted_targets,
        radial,
        stress,
        template_charts=anchor_charts,
        constraint_tolerance=float(constraint_tolerance),
    )


def restricted_truth_outputs(
    *,
    slow_targets_MJE,
    primitive_charts,
    weighted_shared_MJE_fluxes_over_c,
    weighted_MJE_sources_per_ct,
    radial_stress_rates_per_second,
) -> ConservativeMacroOutputs:
    """Restrict exact fine flux/source/rate data without losing M/J/E closure."""

    targets = np.asarray(slow_targets_MJE, dtype=float)
    charts = np.asarray(primitive_charts, dtype=float)
    face_fluxes = np.asarray(weighted_shared_MJE_fluxes_over_c, dtype=float)
    sources = np.asarray(weighted_MJE_sources_per_ct, dtype=float)
    auxiliary = np.asarray(radial_stress_rates_per_second, dtype=float)
    if (
        targets.shape != (FINE_CELLS, 3)
        or charts.shape != (FINE_CELLS, 7)
        or face_fluxes.shape != (FINE_CELLS + 1, 3)
        or sources.shape != (FINE_CELLS, 3)
        or auxiliary.shape != (FINE_CELLS, 2)
        or any(np.any(~np.isfinite(item)) for item in (targets, charts, face_fluxes, sources, auxiliary))
    ):
        raise ValueError("fine truth-output restriction inputs are invalid")
    fine_MJE_rates = C * (sources - (face_fluxes[1:] - face_fluxes[:-1]))
    macro_MJE_rates = np.sum(
        fine_MJE_rates.reshape(MACRO_CELLS, FINE_PER_MACRO, 3), axis=1
    )
    grouped_targets = targets.reshape(MACRO_CELLS, FINE_PER_MACRO, 3)
    grouped_charts = charts.reshape(MACRO_CELLS, FINE_PER_MACRO, 7)
    mass = grouped_targets[:, :, 0]
    total_mass = np.sum(mass, axis=1)
    weights = mass / total_mass[:, None]
    aux_values = grouped_charts[:, :, (1, 4)]
    aux_mean = np.sum(weights[:, :, None] * aux_values, axis=1)
    grouped_aux_rates = auxiliary.reshape(MACRO_CELLS, FINE_PER_MACRO, 2)
    grouped_mass_rates = fine_MJE_rates[:, 0].reshape(MACRO_CELLS, FINE_PER_MACRO)
    macro_aux_rates = (
        np.sum(mass[:, :, None] * grouped_aux_rates, axis=1)
        + np.sum(
            (aux_values - aux_mean[:, None, :])
            * grouped_mass_rates[:, :, None],
            axis=1,
        )
    ) / total_mass[:, None]
    macro_sources = np.sum(
        sources.reshape(MACRO_CELLS, FINE_PER_MACRO, 3), axis=1
    )
    macro_rates = np.column_stack((macro_MJE_rates, macro_aux_rates))
    return ConservativeMacroOutputs(
        MJE_face_fluxes_over_c=np.array(face_fluxes[::FINE_PER_MACRO], copy=True),
        MJE_cell_sources_per_ct=np.asarray(macro_sources),
        auxiliary_rates_per_second=np.asarray(macro_aux_rates),
        macro_rates_per_second=np.asarray(macro_rates),
    )


def truth_outputs_from_radial_operator(operator) -> ConservativeMacroOutputs:
    """Restrict one entropy-complete seven-field operator evaluation."""

    charts = np.asarray(operator.primitive_charts)
    exact_targets = np.asarray(operator.exact_integrated_states)[:, SLOW_ROWS]
    return restricted_truth_outputs(
        slow_targets_MJE=exact_targets,
        primitive_charts=charts,
        weighted_shared_MJE_fluxes_over_c=np.asarray(
            operator.weighted_shared_exact_fluxes_over_c
        )[:, SLOW_ROWS],
        weighted_MJE_sources_per_ct=np.asarray(
            operator.weighted_equation_sources_per_ct
        )[:, SLOW_ROWS],
        radial_stress_rates_per_second=C
        * np.asarray(operator.primitive_rates_per_ct)[:, (1, 4)],
    )


def pack_macro_outputs(outputs: ConservativeMacroOutputs) -> np.ndarray:
    """Pack fluxes, nonzero sources, and auxiliary rates into 115 entries."""

    flux = np.asarray(outputs.MJE_face_fluxes_over_c, dtype=float)
    source = np.asarray(outputs.MJE_cell_sources_per_ct, dtype=float)
    auxiliary = np.asarray(outputs.auxiliary_rates_per_second, dtype=float)
    if flux.shape != (17, 3) or source.shape != (16, 3) or auxiliary.shape != (16, 2):
        raise ValueError("macro output shapes are invalid")
    if not np.array_equal(source[:, 0], np.zeros(16)):
        raise ValueError("macro mass source must be identically zero")
    return np.concatenate((flux.ravel(), source[:, 1:].ravel(), auxiliary.ravel()))


def unpack_macro_outputs(values) -> ConservativeMacroOutputs:
    """Unpack 115 entries and induce the conservative macro state rate."""

    packed = np.asarray(values, dtype=float)
    if packed.shape != (OUTPUT_SIZE,) or np.any(~np.isfinite(packed)):
        raise ValueError("packed macro output is invalid")
    flux = packed[:51].reshape(17, 3)
    source = np.zeros((16, 3), dtype=float)
    source[:, 1:] = packed[51:83].reshape(16, 2)
    auxiliary = packed[83:].reshape(16, 2)
    MJE_rates = C * (source - (flux[1:] - flux[:-1]))
    rates = np.column_stack((MJE_rates, auxiliary))
    return ConservativeMacroOutputs(
        MJE_face_fluxes_over_c=flux,
        MJE_cell_sources_per_ct=source,
        auxiliary_rates_per_second=auxiliary,
        macro_rates_per_second=rates,
    )


def macro_output_component_scales(outputs: ConservativeMacroOutputs) -> np.ndarray:
    """Return fixed componentwise scales for normalized atlas fitting."""

    flux = np.asarray(outputs.MJE_face_fluxes_over_c)
    source = np.asarray(outputs.MJE_cell_sources_per_ct)[:, 1:]
    auxiliary = np.asarray(outputs.auxiliary_rates_per_second)
    tiny = np.finfo(float).tiny
    flux_scale = np.maximum(np.max(np.abs(flux), axis=0), tiny)
    source_scale = np.maximum(np.max(np.abs(source), axis=0), tiny)
    auxiliary_scale = np.maximum(np.max(np.abs(auxiliary), axis=0), tiny)
    return np.concatenate(
        (
            np.tile(flux_scale, 17),
            np.tile(source_scale, 16),
            np.tile(auxiliary_scale, 16),
        )
    )


def conservative_ledger_relative_defect(outputs: ConservativeMacroOutputs) -> float:
    """Audit telescoping M/J/E closure of an online atlas output."""

    rates = np.asarray(outputs.macro_rates_per_second)[:, :3]
    flux = np.asarray(outputs.MJE_face_fluxes_over_c)
    source = np.asarray(outputs.MJE_cell_sources_per_ct)
    expected = C * (np.sum(source, axis=0) - (flux[-1] - flux[0]))
    actual = np.sum(rates, axis=0)
    scale = max(float(np.max(np.abs(actual))), float(np.max(np.abs(expected))), np.finfo(float).tiny)
    return float(np.max(np.abs(actual - expected)) / scale)


__all__ = (
    "ConservativeAffineMacroAtlas",
    "ConservativeMacroOutputs",
    "FINE_CELLS",
    "FINE_PER_MACRO",
    "MACRO_CELLS",
    "MACRO_FIELDS",
    "OUTPUT_SIZE",
    "conservative_ledger_relative_defect",
    "macro_output_component_scales",
    "pack_macro_outputs",
    "prolong_entropy_complete_macro",
    "restrict_entropy_complete_macro",
    "restricted_truth_outputs",
    "truth_outputs_from_radial_operator",
    "unpack_macro_outputs",
)
