"""Positive-energy family-transfer ledgers for the frozen causal-inner DAE.

This module is audit-only.  It does not change the production residual,
boundary treatment, coupling operator, or time integrator.  It provides:

* a polynomial spectral-projector construction that is independent of
  eigenvector normalization and inversion; and
* an exact block/source/receiver power ledger for the implemented
  descriptor-reduced semidiscrete generator.

For a fixed positive cell metric ``H`` and energy-orthogonal projectors
``P_f``, the family energy and directed block work are

``E_f = 1/2 <P_f u, H P_f u>``

and

``W[b, f, g] = <P_f u, H G_b P_g u>``.

Summing over blocks, receivers, and sources reproduces
``<u, H G u>`` to roundoff.  The resulting transfer is the exact transfer of
the implemented frozen semidiscrete DAE; it is not by itself a claim that a
particular family conversion is physical rather than truncation error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csr_matrix

from .causal_inner_characteristic_phase import (
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
)


_N_FIELDS = 5
_N_FAMILIES = len(CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES)


@dataclass(frozen=True)
class CausalPolynomialProjectorAudit:
    """Polynomial spectral projectors of one separated real pencil."""

    family_labels: tuple[str, ...]
    characteristic_speeds: np.ndarray
    primitive_projectors: np.ndarray
    primitive_energy_metric: np.ndarray
    minimum_spectral_gap: float
    maximum_identity_defect: float
    maximum_idempotence_defect: float
    maximum_cross_projector_defect: float
    maximum_eigenpair_defect: float
    maximum_energy_orthogonality_defect: float
    maximum_symmetrizer_defect: float
    maximum_imaginary_part: float


@dataclass(frozen=True)
class CausalPhysicalFamilyTransferLedger:
    """Exact positive-energy transfer ledger for one physical history."""

    family_labels: tuple[str, ...]
    block_names: tuple[str, ...]
    times_seconds: np.ndarray
    family_energy: np.ndarray
    total_energy: np.ndarray
    family_power_per_s: np.ndarray
    total_power_per_s: np.ndarray
    block_source_receiver_power_per_s: np.ndarray
    integrated_block_source_receiver_work: np.ndarray
    integrated_block_source_receiver_cell_work: np.ndarray
    maximum_family_partition_defect: float
    maximum_power_closure_defect: float
    maximum_block_matrix_closure_defect: float
    maximum_integrated_energy_defect: float


def _relative_maximum_defect(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(left - right)) / scale)


def causal_polynomial_spectral_projectors(
    temporal_storage_matrix: np.ndarray,
    spatial_principal_matrix: np.ndarray,
    field_scales: np.ndarray,
    *,
    minimum_relative_gap: float = 1.0e-10,
) -> CausalPolynomialProjectorAudit:
    """Build spectral projectors as polynomials of ``A^{-1} B``.

    For separated eigenvalues, the Lagrange polynomial

    ``P_f = product_{g != f} (K - lambda_g I)/(lambda_f-lambda_g)``

    defines the invariant projector without selecting, normalizing, or
    inverting an eigenvector matrix.  The only eigensolver use here is to
    obtain the unordered eigenvalues.
    """

    temporal = np.asarray(temporal_storage_matrix, dtype=float)
    spatial = np.asarray(spatial_principal_matrix, dtype=float)
    scales = np.asarray(field_scales, dtype=float).ravel()
    if (
        temporal.shape != (_N_FIELDS, _N_FIELDS)
        or spatial.shape != temporal.shape
        or scales.shape != (_N_FIELDS,)
        or np.any(~np.isfinite(temporal))
        or np.any(~np.isfinite(spatial))
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
        or not np.isfinite(minimum_relative_gap)
        or minimum_relative_gap <= 0.0
    ):
        raise ValueError("polynomial-projector inputs are invalid")

    evolution = np.linalg.solve(temporal, spatial)
    scale_matrix = np.diag(scales)
    inverse_scale = np.diag(1.0 / scales)
    dimensionless = inverse_scale @ evolution @ scale_matrix
    values = np.linalg.eigvals(dimensionless)
    maximum_imaginary = float(np.max(np.abs(np.imag(values))))
    if maximum_imaginary > 1.0e-10:
        raise RuntimeError("polynomial-projector spectrum is not real")
    speeds = np.sort(np.real(values))
    gaps = np.abs(speeds[:, None] - speeds[None, :])
    gaps[np.diag_indices_from(gaps)] = np.inf
    minimum_gap = float(np.min(gaps))
    spectral_scale = max(float(np.max(np.abs(speeds))), 1.0)
    if minimum_gap <= minimum_relative_gap * spectral_scale:
        raise RuntimeError("polynomial-projector spectrum is unresolved")

    identity = np.eye(_N_FIELDS)
    dimensionless_projectors = []
    for family, selected in enumerate(speeds):
        projector = np.eye(_N_FIELDS)
        for other, value in enumerate(speeds):
            if other == family:
                continue
            projector = (
                projector @ (dimensionless - value * identity)
                / (selected - value)
            )
        dimensionless_projectors.append(projector)
    dimensionless_projectors = np.asarray(
        dimensionless_projectors,
        dtype=float,
    )
    projectors = np.einsum(
        "ij,fjk,kl->fil",
        scale_matrix,
        dimensionless_projectors,
        inverse_scale,
        optimize=True,
    )
    dimensionless_energy = np.sum(
        np.einsum(
            "fji,fjk->fik",
            dimensionless_projectors,
            dimensionless_projectors,
            optimize=True,
        ),
        axis=0,
    )
    energy = inverse_scale @ dimensionless_energy @ inverse_scale

    identity_defect = float(
        np.max(
            np.abs(
                np.sum(dimensionless_projectors, axis=0) - identity
            )
        )
    )
    idempotence = 0.0
    cross = 0.0
    eigenpair = 0.0
    orthogonality = 0.0
    energy_scale = max(float(np.max(np.abs(energy))), np.finfo(float).tiny)
    for first, speed in enumerate(speeds):
        projector = dimensionless_projectors[first]
        idempotence = max(
            idempotence,
            float(np.max(np.abs(projector @ projector - projector))),
        )
        residual = dimensionless @ projector - speed * projector
        residual_scale = max(
            float(np.max(np.abs(dimensionless @ projector))),
            float(np.max(np.abs(speed * projector))),
            np.finfo(float).tiny,
        )
        eigenpair = max(
            eigenpair,
            float(np.max(np.abs(residual)) / residual_scale),
        )
        for second in range(_N_FAMILIES):
            if first == second:
                continue
            cross = max(
                cross,
                float(
                    np.max(
                        np.abs(
                            projector @ dimensionless_projectors[second]
                        )
                    )
                ),
            )
            orthogonality = max(
                orthogonality,
                float(
                    np.max(
                        np.abs(
                            projectors[first].T
                            @ energy
                            @ projectors[second]
                        )
                    )
                    / energy_scale
                ),
            )
    flux = energy @ evolution
    flux_scale = max(float(np.max(np.abs(flux))), np.finfo(float).tiny)
    return CausalPolynomialProjectorAudit(
        family_labels=CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
        characteristic_speeds=speeds,
        primitive_projectors=projectors,
        primitive_energy_metric=energy,
        minimum_spectral_gap=minimum_gap,
        maximum_identity_defect=identity_defect,
        maximum_idempotence_defect=float(idempotence),
        maximum_cross_projector_defect=float(cross),
        maximum_eigenpair_defect=float(eigenpair),
        maximum_energy_orthogonality_defect=float(orthogonality),
        maximum_symmetrizer_defect=float(
            np.max(np.abs(flux - flux.T)) / flux_scale
        ),
        maximum_imaginary_part=maximum_imaginary,
    )


def causal_physical_family_transfer_ledger(
    physical_state_history: np.ndarray,
    times_seconds: np.ndarray,
    *,
    log_edges: np.ndarray,
    primitive_energy_metrics: np.ndarray,
    primitive_projectors: np.ndarray,
    scaled_generator_per_s: np.ndarray,
    scaled_generator_blocks_per_s: dict[str, np.ndarray] | None = None,
    descriptor_scaled_matrix: np.ndarray | None = None,
    scaled_residual_blocks: dict[str, np.ndarray] | None = None,
    primitive_column_scales: np.ndarray,
    lower_face: int,
    upper_face: int,
) -> CausalPhysicalFamilyTransferLedger:
    """Resolve exact semidiscrete power by block, source, and receiver.

    ``primitive_projectors`` has layout ``(cell, family, field, field)``.
    Generator blocks act in the scaled primitive coordinates used by the
    frozen tangent, while the input history and returned energy are physical.
    """

    states = np.asarray(physical_state_history, dtype=float)
    times = np.asarray(times_seconds, dtype=float)
    edges = np.asarray(log_edges, dtype=float)
    energy = np.asarray(primitive_energy_metrics, dtype=float)
    projectors = np.asarray(primitive_projectors, dtype=float)
    generator = np.asarray(scaled_generator_per_s, dtype=float)
    columns = np.asarray(primitive_column_scales, dtype=float).ravel()
    if states.ndim != 3 or states.shape[2] != _N_FIELDS:
        raise ValueError("family-transfer history has wrong shape")
    n_times, n_cells, _fields = states.shape
    dimensions = n_cells * _N_FIELDS
    lower = int(lower_face)
    upper = int(upper_face)
    generator_blocks_supplied = scaled_generator_blocks_per_s is not None
    residual_blocks_supplied = scaled_residual_blocks is not None
    if (
        times.shape != (n_times,)
        or n_times < 2
        or edges.shape != (n_cells + 1,)
        or energy.shape != (n_cells, _N_FIELDS, _N_FIELDS)
        or projectors.shape
        != (n_cells, _N_FAMILIES, _N_FIELDS, _N_FIELDS)
        or columns.shape != (dimensions,)
        or generator.shape != (dimensions, dimensions)
        or generator_blocks_supplied == residual_blocks_supplied
        or np.any(~np.isfinite(states))
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
        or np.any(~np.isfinite(energy))
        or np.any(~np.isfinite(projectors))
        or np.any(~np.isfinite(columns))
        or np.any(~np.isfinite(generator))
        or np.any(columns <= 0.0)
        or not 0 <= lower < upper <= n_cells
    ):
        raise ValueError("family-transfer inputs are invalid")
    if residual_blocks_supplied:
        descriptor = np.asarray(descriptor_scaled_matrix, dtype=float)
        if (
            descriptor.shape != (dimensions, dimensions)
            or np.any(~np.isfinite(descriptor))
        ):
            raise ValueError("family-transfer descriptor is invalid")
        supplied_blocks = scaled_residual_blocks
        descriptor_factor = lu_factor(descriptor)
    else:
        descriptor = None
        supplied_blocks = scaled_generator_blocks_per_s
        descriptor_factor = None
    assert supplied_blocks is not None
    block_names = tuple(supplied_blocks)
    matrices = {
        name: np.asarray(matrix, dtype=float)
        for name, matrix in supplied_blocks.items()
    }
    if any(
        matrix.shape != (dimensions, dimensions)
        or np.any(~np.isfinite(matrix))
        for matrix in matrices.values()
    ):
        raise ValueError("family-transfer generator block is invalid")

    weights = np.zeros(n_cells, dtype=float)
    weights[lower:upper] = np.diff(edges)[lower:upper]
    family_states = np.einsum(
        "ckij,tcj->tkci",
        projectors,
        states,
        optimize=True,
    )
    family_energy = 0.5 * np.einsum(
        "tkci,cij,tkcj,c->tk",
        family_states,
        energy,
        family_states,
        weights,
        optimize=True,
    )
    total_energy = 0.5 * np.einsum(
        "tci,cij,tcj,c->t",
        states,
        energy,
        states,
        weights,
        optimize=True,
    )
    partition_defect = _relative_maximum_defect(
        np.sum(family_energy, axis=1),
        total_energy,
    )

    source_scaled = (
        family_states.transpose(1, 0, 2, 3).reshape(
            _N_FAMILIES * n_times,
            dimensions,
        )
        / columns[None, :]
    )
    block_work = np.empty(
        (
            len(block_names),
            n_times,
            _N_FAMILIES,
            _N_FAMILIES,
        ),
        dtype=float,
    )
    integrated_cell_work = np.empty(
        (
            len(block_names),
            _N_FAMILIES,
            _N_FAMILIES,
            n_cells,
        ),
        dtype=float,
    )
    columns_by_cell = columns.reshape(n_cells, _N_FIELDS)
    reconstructed_source_action_scaled = np.zeros_like(source_scaled)
    for block_index, name in enumerate(block_names):
        raw_action = csr_matrix(matrices[name]) @ source_scaled.T
        if residual_blocks_supplied:
            assert descriptor_factor is not None
            action_values = -lu_solve(descriptor_factor, raw_action)
        else:
            action_values = raw_action
        reconstructed_source_action_scaled += action_values.T
        action_scaled = action_values.T.reshape(
            _N_FAMILIES,
            n_times,
            n_cells,
            _N_FIELDS,
        )
        action_physical = (
            action_scaled * columns_by_cell[None, None, :, :]
        )
        cell_power = np.einsum(
            "trci,cij,stcj,c->trsc",
            family_states,
            energy,
            action_physical,
            weights,
            optimize=True,
        )
        block_work[block_index] = np.sum(cell_power, axis=-1)
        integrated_cell_work[block_index] = np.trapezoid(
            cell_power,
            times,
            axis=0,
        )

    family_power = np.sum(block_work, axis=(0, 3))
    total_power = np.sum(family_power, axis=1)
    scaled_states = states.reshape(n_times, dimensions) / columns[None, :]
    direct_action = (
        csr_matrix(generator) @ scaled_states.T
    ).T.reshape(n_times, n_cells, _N_FIELDS)
    direct_action *= columns_by_cell[None, :, :]
    direct_power = np.einsum(
        "tci,cij,tcj,c->t",
        states,
        energy,
        direct_action,
        weights,
        optimize=True,
    )
    power_defect = _relative_maximum_defect(total_power, direct_power)
    direct_source_action_scaled = (
        csr_matrix(generator) @ source_scaled.T
    ).T
    block_matrix_defect = _relative_maximum_defect(
        direct_source_action_scaled,
        reconstructed_source_action_scaled,
    )
    integrated_work = np.trapezoid(block_work, times, axis=1)
    energy_change = family_energy[-1] - family_energy[0]
    integrated_family_power = np.trapezoid(
        family_power,
        times,
        axis=0,
    )
    integration_scale = max(
        float(np.max(np.abs(energy_change))),
        float(np.max(np.abs(integrated_family_power))),
        float(
            np.max(
                np.sum(
                    np.abs(integrated_work),
                    axis=(0, 2),
                )
            )
        ),
        np.finfo(float).tiny,
    )
    integrated_defect = float(
        np.max(np.abs(energy_change - integrated_family_power))
        / integration_scale
    )
    return CausalPhysicalFamilyTransferLedger(
        family_labels=CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
        block_names=block_names,
        times_seconds=np.array(times, copy=True),
        family_energy=family_energy,
        total_energy=total_energy,
        family_power_per_s=family_power,
        total_power_per_s=total_power,
        block_source_receiver_power_per_s=block_work,
        integrated_block_source_receiver_work=integrated_work,
        integrated_block_source_receiver_cell_work=integrated_cell_work,
        maximum_family_partition_defect=partition_defect,
        maximum_power_closure_defect=power_defect,
        maximum_block_matrix_closure_defect=block_matrix_defect,
        maximum_integrated_energy_defect=integrated_defect,
    )
