"""Exact audit ledgers for causal characteristic-family transfer.

This module is audit-only.  It does not alter the production DAE, numerical
flux, boundary treatment, or time integrator.  It supplies three algebraic
tools used by WP10c9c0c:

* an exact cellwise decomposition into the five declared characteristic
  families;
* a pairwise Gram ledger for cross-mesh errors and family cross-work;
* a radius-resolved quadratic-energy work ledger for an exactly decomposed
  frozen generator.

Every returned ledger closes by direct summation.  The routines deliberately
avoid assigning a physical interpretation to a large pairwise term; that
interpretation remains the responsibility of the evidence campaign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .causal_inner_characteristic_phase import (
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
    CausalFiveFieldCharacteristicBasis,
    causal_five_field_characteristic_basis,
)
from .causal_inner_dae_system import CausalFiveFieldDAEContext


_N_FIELDS = 5
_N_FAMILIES = len(CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES)


@dataclass(frozen=True)
class CausalCharacteristicFamilyProjectors:
    """Cellwise oblique projectors for the five principal families."""

    family_labels: tuple[str, ...]
    primitive_projectors: np.ndarray
    maximum_identity_closure_defect: float
    maximum_idempotence_defect: float
    maximum_cross_projector_defect: float
    maximum_basis_condition_number: float
    maximum_eigenpair_defect: float


@dataclass(frozen=True)
class CausalPairwiseGramLedger:
    """Pairwise decomposition of one weighted squared norm."""

    family_labels: tuple[str, ...]
    pairwise_gram: np.ndarray
    component_squared_norms: np.ndarray
    total_squared_norm: np.ndarray
    reconstructed_squared_norm: np.ndarray
    component_relative_amplitudes: np.ndarray
    maximum_closure_defect: float


@dataclass(frozen=True)
class CausalLocalEnergyWorkLedger:
    """Cellwise quadratic energy and exact generator-block work."""

    times_seconds: np.ndarray
    energy_by_cell: np.ndarray
    rate_by_cell_per_s: np.ndarray
    rate_by_block_and_cell_per_s: dict[str, np.ndarray]
    cumulative_work_by_block_and_cell: dict[str, np.ndarray]
    maximum_instantaneous_block_closure_defect: float
    maximum_integrated_energy_closure_defect: float


@dataclass(frozen=True)
class CausalBlockFamilyTransferLedger:
    """Exact block/source/receiver ledger in one declared metric."""

    block_names: tuple[str, ...]
    family_labels: tuple[str, ...]
    global_cross_work_per_s: np.ndarray
    reconstructed_rate_history: np.ndarray
    maximum_rate_action_closure_defect: float
    maximum_cross_work_closure_defect: float


def _relative_maximum_defect(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    scale = max(
        float(np.max(np.abs(first_values))),
        float(np.max(np.abs(second_values))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(first_values - second_values)) / scale)


def _cumulative_trapezoid(
    times: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    time_values = np.asarray(times, dtype=float)
    rate_values = np.asarray(values, dtype=float)
    if (
        time_values.ndim != 1
        or rate_values.shape[0] != time_values.size
        or np.any(~np.isfinite(time_values))
        or np.any(~np.isfinite(rate_values))
        or np.any(np.diff(time_values) <= 0.0)
    ):
        raise ValueError("cumulative-work inputs are invalid")
    result = np.zeros_like(rate_values)
    time_shape = (time_values.size - 1,) + (1,) * (
        rate_values.ndim - 1
    )
    result[1:] = np.cumsum(
        0.5
        * (rate_values[:-1] + rate_values[1:])
        * np.diff(time_values).reshape(time_shape),
        axis=0,
    )
    return result


def causal_five_field_characteristic_family_projectors(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    primitive_amplitudes: np.ndarray,
) -> tuple[
    CausalCharacteristicFamilyProjectors,
    tuple[CausalFiveFieldCharacteristicBasis, ...],
]:
    """Return the exact five-family projector partition in every cell."""

    context = context.validated()
    charts = np.asarray(primitive_charts, dtype=float)
    amplitudes = np.asarray(primitive_amplitudes, dtype=float)
    n_cells = int(context.grid.centers.size)
    if (
        charts.shape != (n_cells, _N_FIELDS)
        or amplitudes.shape != charts.shape
        or np.any(~np.isfinite(charts))
        or np.any(~np.isfinite(amplitudes))
        or np.any(amplitudes <= 0.0)
    ):
        raise ValueError("characteristic-family projector inputs are invalid")

    projectors = np.empty(
        (_N_FAMILIES, n_cells, _N_FIELDS, _N_FIELDS),
        dtype=float,
    )
    bases = []
    maximum_identity = 0.0
    maximum_idempotence = 0.0
    maximum_cross = 0.0
    maximum_condition = 0.0
    maximum_eigenpair = 0.0
    identity = np.eye(_N_FIELDS)
    for cell, radius in enumerate(context.grid.centers):
        basis = causal_five_field_characteristic_basis(
            context,
            float(radius),
            charts[cell],
            amplitudes[cell],
        )
        right = np.asarray(
            basis.dimensionless_right_eigenvectors,
            dtype=float,
        )
        left = np.linalg.inv(right)
        for family in range(_N_FAMILIES):
            projectors[family, cell] = np.outer(
                right[:, family],
                left[family],
            )
        closure = np.sum(projectors[:, cell], axis=0)
        maximum_identity = max(
            maximum_identity,
            float(np.max(np.abs(closure - identity))),
        )
        for first in range(_N_FAMILIES):
            maximum_idempotence = max(
                maximum_idempotence,
                float(
                    np.max(
                        np.abs(
                            projectors[first, cell]
                            @ projectors[first, cell]
                            - projectors[first, cell]
                        )
                    )
                ),
            )
            for second in range(_N_FAMILIES):
                if first == second:
                    continue
                maximum_cross = max(
                    maximum_cross,
                    float(
                        np.max(
                            np.abs(
                                projectors[first, cell]
                                @ projectors[second, cell]
                            )
                        )
                    ),
                )
        maximum_condition = max(
            maximum_condition,
            float(basis.condition_number),
        )
        maximum_eigenpair = max(
            maximum_eigenpair,
            float(basis.maximum_eigenpair_defect),
        )
        bases.append(basis)

    return (
        CausalCharacteristicFamilyProjectors(
            family_labels=CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
            primitive_projectors=projectors,
            maximum_identity_closure_defect=maximum_identity,
            maximum_idempotence_defect=maximum_idempotence,
            maximum_cross_projector_defect=maximum_cross,
            maximum_basis_condition_number=maximum_condition,
            maximum_eigenpair_defect=maximum_eigenpair,
        ),
        tuple(bases),
    )


def causal_five_field_characteristic_family_decomposition(
    dimensionless_primitive_values: np.ndarray,
    projectors: CausalCharacteristicFamilyProjectors,
) -> np.ndarray:
    """Decompose a field or history; family is the leading output axis."""

    values = np.asarray(dimensionless_primitive_values, dtype=float)
    matrices = np.asarray(projectors.primitive_projectors, dtype=float)
    n_cells = matrices.shape[1]
    if (
        matrices.shape
        != (_N_FAMILIES, n_cells, _N_FIELDS, _N_FIELDS)
        or values.shape[-2:] != (n_cells, _N_FIELDS)
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("characteristic-family decomposition is invalid")
    flat = values.reshape(-1, n_cells, _N_FIELDS)
    components = np.einsum(
        "fcij,tcj->ftci",
        matrices,
        flat,
        optimize=True,
    )
    return components.reshape(
        (_N_FAMILIES,) + values.shape[:-2] + (n_cells, _N_FIELDS)
    )


def causal_pairwise_weighted_gram_ledger(
    components: np.ndarray,
    cell_weights: np.ndarray,
) -> CausalPairwiseGramLedger:
    """Return the exact pairwise Gram ledger of a component history.

    ``components`` has shape ``(family, time, cell, field)``.  The scalar
    product is Euclidean in the supplied dimensionless primitive chart and
    uses the caller's positive cell weights.
    """

    values = np.asarray(components, dtype=float)
    weights = np.asarray(cell_weights, dtype=float)
    if (
        values.ndim != 4
        or values.shape[0] != _N_FAMILIES
        or values.shape[3] != _N_FIELDS
        or weights.shape != (values.shape[2],)
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("pairwise Gram-ledger inputs are invalid")
    normalized = weights / np.sum(weights)
    gram = np.einsum(
        "ftci,gtci,c->tfg",
        values,
        values,
        normalized,
        optimize=True,
    )
    component_norms = np.diagonal(gram, axis1=1, axis2=2)
    total_values = np.sum(values, axis=0)
    total_norm = np.einsum(
        "tci,tci,c->t",
        total_values,
        total_values,
        normalized,
        optimize=True,
    )
    reconstructed = np.sum(gram, axis=(1, 2))
    relative = np.sqrt(
        np.maximum(component_norms, 0.0)
        / np.maximum(total_norm[:, None], np.finfo(float).tiny)
    )
    return CausalPairwiseGramLedger(
        family_labels=CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
        pairwise_gram=gram,
        component_squared_norms=component_norms,
        total_squared_norm=total_norm,
        reconstructed_squared_norm=reconstructed,
        component_relative_amplitudes=relative,
        maximum_closure_defect=_relative_maximum_defect(
            reconstructed,
            total_norm,
        ),
    )


def causal_pairwise_family_cross_work(
    family_state_histories: np.ndarray,
    generator_per_s: np.ndarray,
    cell_weights: np.ndarray,
) -> np.ndarray:
    """Return directed family cross-work ``<x_f, G x_g>``."""

    values = np.asarray(family_state_histories, dtype=float)
    generator = np.asarray(generator_per_s, dtype=float)
    weights = np.asarray(cell_weights, dtype=float)
    if (
        values.ndim != 4
        or values.shape[0] != _N_FAMILIES
        or values.shape[3] != _N_FIELDS
        or generator.shape
        != (
            values.shape[2] * _N_FIELDS,
            values.shape[2] * _N_FIELDS,
        )
        or weights.shape != (values.shape[2],)
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(generator))
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("family cross-work inputs are invalid")
    normalized = weights / np.sum(weights)
    actions = np.asarray(
        [
            state.reshape(state.shape[0], -1) @ generator.T
            for state in values
        ],
        dtype=float,
    ).reshape(values.shape)
    return np.einsum(
        "ftci,gtci,c->tfg",
        values,
        actions,
        normalized,
        optimize=True,
    )


def causal_block_family_receiver_action(
    source_family_history: np.ndarray,
    generator_block_per_s: np.ndarray,
    projectors: CausalCharacteristicFamilyProjectors,
) -> np.ndarray:
    """Apply one block to one source family and split receiver families.

    The output shape is ``(receiver, time, cell, field)``.  Summing the
    receiver axis reproduces the unprojected block action.
    """

    source = np.asarray(source_family_history, dtype=float)
    generator = np.asarray(generator_block_per_s, dtype=float)
    matrices = np.asarray(projectors.primitive_projectors, dtype=float)
    if (
        source.ndim != 3
        or source.shape[2] != _N_FIELDS
        or matrices.shape
        != (
            _N_FAMILIES,
            source.shape[1],
            _N_FIELDS,
            _N_FIELDS,
        )
        or generator.shape
        != (
            source.shape[1] * _N_FIELDS,
            source.shape[1] * _N_FIELDS,
        )
        or np.any(~np.isfinite(source))
        or np.any(~np.isfinite(generator))
    ):
        raise ValueError("block-family receiver-action inputs are invalid")
    action = (
        source.reshape(source.shape[0], -1) @ generator.T
    ).reshape(source.shape)
    return np.einsum(
        "rcij,tcj->rtci",
        matrices,
        action,
        optimize=True,
    )


def causal_block_family_transfer_ledger(
    family_state_histories: np.ndarray,
    generator_blocks_per_s: dict[str, np.ndarray],
    projectors: CausalCharacteristicFamilyProjectors,
    cell_weights: np.ndarray,
) -> CausalBlockFamilyTransferLedger:
    """Return the exact four-index common-mode work ledger.

    The work convention is

    ``W[k, t, receiver, source] = <x_receiver, G_k x_source>``.

    Receiver-projected rate actions are accumulated separately so that both
    the work ledger and the full vector rate close independently.
    """

    values = np.asarray(family_state_histories, dtype=float)
    weights = np.asarray(cell_weights, dtype=float)
    if (
        values.ndim != 4
        or values.shape[0] != _N_FAMILIES
        or values.shape[3] != _N_FIELDS
        or weights.shape != (values.shape[2],)
        or not generator_blocks_per_s
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("block-family transfer-ledger inputs are invalid")
    normalized = weights / np.sum(weights)
    block_names = tuple(generator_blocks_per_s)
    n_times = values.shape[1]
    global_work = np.empty(
        (len(block_names), n_times, _N_FAMILIES, _N_FAMILIES),
        dtype=float,
    )
    reconstructed_rate = np.zeros_like(values[0])
    for block_index, name in enumerate(block_names):
        block = np.asarray(generator_blocks_per_s[name], dtype=float)
        for source in range(_N_FAMILIES):
            receiver_actions = causal_block_family_receiver_action(
                values[source],
                block,
                projectors,
            )
            reconstructed_rate += np.sum(receiver_actions, axis=0)
            unprojected_action = np.sum(receiver_actions, axis=0)
            global_work[block_index, :, :, source] = np.einsum(
                "rtci,tci,c->tr",
                values,
                unprojected_action,
                normalized,
                optimize=True,
            )

    full_generator = np.sum(
        np.asarray(
            [
                np.asarray(generator_blocks_per_s[name], dtype=float)
                for name in block_names
            ]
        ),
        axis=0,
    )
    total_state = np.sum(values, axis=0)
    full_rate = (
        total_state.reshape(n_times, -1) @ full_generator.T
    ).reshape(total_state.shape)
    total_work = np.einsum(
        "tci,tci,c->t",
        total_state,
        full_rate,
        normalized,
        optimize=True,
    )
    reconstructed_work = np.sum(global_work, axis=(0, 2, 3))
    return CausalBlockFamilyTransferLedger(
        block_names=block_names,
        family_labels=CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
        global_cross_work_per_s=global_work,
        reconstructed_rate_history=reconstructed_rate,
        maximum_rate_action_closure_defect=_relative_maximum_defect(
            reconstructed_rate,
            full_rate,
        ),
        maximum_cross_work_closure_defect=_relative_maximum_defect(
            reconstructed_work,
            total_work,
        ),
    )


def causal_local_quadratic_energy_work_ledger(
    state_history: np.ndarray,
    times_seconds: np.ndarray,
    cell_energy_grams: np.ndarray,
    generator_blocks_per_s: dict[str, np.ndarray],
) -> CausalLocalEnergyWorkLedger:
    """Return an exact cell-by-cell work map for generator blocks."""

    states = np.asarray(state_history, dtype=float)
    times = np.asarray(times_seconds, dtype=float)
    grams = np.asarray(cell_energy_grams, dtype=float)
    if (
        states.ndim != 3
        or states.shape[2] != _N_FIELDS
        or times.shape != (states.shape[0],)
        or grams.shape
        != (states.shape[1], _N_FIELDS, _N_FIELDS)
        or not generator_blocks_per_s
        or np.any(~np.isfinite(states))
        or np.any(~np.isfinite(times))
        or np.any(~np.isfinite(grams))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("local energy-work inputs are invalid")
    n_cells = states.shape[1]
    n_reduced = n_cells * _N_FIELDS
    flat = states.reshape(states.shape[0], n_reduced)
    actions = {}
    for name, matrix in generator_blocks_per_s.items():
        operator = np.asarray(matrix, dtype=float)
        if (
            operator.shape != (n_reduced, n_reduced)
            or np.any(~np.isfinite(operator))
        ):
            raise ValueError(f"generator block {name!r} is invalid")
        actions[name] = (flat @ operator.T).reshape(states.shape)
    total_action = np.sum(np.asarray(list(actions.values())), axis=0)
    energy = 0.5 * np.einsum(
        "tci,cij,tcj->tc",
        states,
        grams,
        states,
        optimize=True,
    )
    rate_by_block = {
        name: np.einsum(
            "tci,cij,tcj->tc",
            states,
            grams,
            action,
            optimize=True,
        )
        for name, action in actions.items()
    }
    total_rate = np.einsum(
        "tci,cij,tcj->tc",
        states,
        grams,
        total_action,
        optimize=True,
    )
    reconstructed_rate = np.sum(
        np.asarray(list(rate_by_block.values())),
        axis=0,
    )
    cumulative = {
        name: _cumulative_trapezoid(times, rate)
        for name, rate in rate_by_block.items()
    }
    cumulative_total = np.sum(
        np.asarray(list(cumulative.values())),
        axis=0,
    )
    energy_change = energy - energy[0]
    return CausalLocalEnergyWorkLedger(
        times_seconds=times,
        energy_by_cell=energy,
        rate_by_cell_per_s=total_rate,
        rate_by_block_and_cell_per_s=rate_by_block,
        cumulative_work_by_block_and_cell=cumulative,
        maximum_instantaneous_block_closure_defect=(
            _relative_maximum_defect(reconstructed_rate, total_rate)
        ),
        maximum_integrated_energy_closure_defect=(
            _relative_maximum_defect(cumulative_total, energy_change)
        ),
    )
