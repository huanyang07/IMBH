"""Exact semidiscrete energy accounting for the frozen causal-inner DAE.

This module is audit-only.  It does not alter the production or candidate
residual.  Given the frozen descriptor, generator, physical energy metric,
stationary-residual blocks, and the actual shared-face linear maps, it
constructs the exact quadratic control-volume energy identity

``d(1/2 z.T W z)/dt = z.T W G z``.

The descriptor dual ``D^{-T} W z`` is used to expose every residual block and
every shared conservative face.  This is the numerical energy transfer of the
implemented semidiscrete DAE; it need not equal the continuum symmetrizer flux
used by the preceding scattering audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class CausalSemidiscreteEnergyHistory:
    """Exact control-volume energy and power histories."""

    stored_energy: np.ndarray
    stored_energy_rate: np.ndarray
    direct_generator_power: np.ndarray
    block_powers: dict[str, np.ndarray]
    conservative_face_powers: np.ndarray
    maximum_generator_power_defect: float
    maximum_block_power_defect: float
    maximum_face_power_defect: float


def causal_scaled_control_energy_metric(
    primitive_energy_metrics: np.ndarray,
    log_edges: np.ndarray,
    primitive_column_scales: np.ndarray,
    lower_face: int,
    upper_face: int,
) -> np.ndarray:
    """Return the scaled-coordinate quadratic metric of one cell band."""

    energy = np.asarray(primitive_energy_metrics, dtype=float)
    edges = np.asarray(log_edges, dtype=float)
    columns = np.asarray(primitive_column_scales, dtype=float).ravel()
    cells = int(energy.shape[0])
    fields = int(energy.shape[1]) if energy.ndim == 3 else 0
    lower = int(lower_face)
    upper = int(upper_face)
    if (
        fields <= 0
        or energy.shape != (cells, fields, fields)
        or edges.shape != (cells + 1,)
        or columns.shape != (cells * fields,)
        or np.any(~np.isfinite(energy))
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
        or np.any(~np.isfinite(columns))
        or np.any(columns <= 0.0)
        or not 0 <= lower < upper <= cells
    ):
        raise ValueError("scaled control-energy inputs are invalid")
    metric = np.zeros((cells * fields, cells * fields), dtype=float)
    widths = np.diff(edges)
    scales = columns.reshape(cells, fields)
    for cell in range(lower, upper):
        selected = slice(fields * cell, fields * (cell + 1))
        metric[selected, selected] = (
            energy[cell]
            * scales[cell, :, None]
            * scales[cell, None, :]
            * widths[cell]
        )
    symmetry_scale = max(float(np.max(np.abs(metric))), np.finfo(float).tiny)
    minimum_block_eigenvalue = min(
        float(
            np.min(
                np.linalg.eigvalsh(
                    metric[
                        fields * cell : fields * (cell + 1),
                        fields * cell : fields * (cell + 1),
                    ]
                )
            )
        )
        for cell in range(lower, upper)
    )
    if (
        np.max(np.abs(metric - metric.T)) / symmetry_scale > 1.0e-11
        or minimum_block_eigenvalue < -1.0e-11 * symmetry_scale
    ):
        raise ValueError("scaled control-energy metric is not positive")
    return metric


def causal_semidiscrete_generator_components(
    descriptor_scaled_matrix: np.ndarray,
    *,
    stationary_scaled_blocks: dict[str, np.ndarray],
    mapped_storage_rate_scaled_matrix: np.ndarray,
    responsive_height_storage_rate_scaled_matrix: np.ndarray,
) -> tuple[dict[str, np.ndarray], float]:
    """Return descriptor-reduced generator blocks and their closure defect."""

    descriptor = np.asarray(descriptor_scaled_matrix, dtype=float)
    mapped = np.asarray(mapped_storage_rate_scaled_matrix, dtype=float)
    height = np.asarray(
        responsive_height_storage_rate_scaled_matrix,
        dtype=float,
    )
    if (
        descriptor.ndim != 2
        or descriptor.shape[0] != descriptor.shape[1]
        or mapped.shape != descriptor.shape
        or height.shape != descriptor.shape
        or not stationary_scaled_blocks
        or np.any(~np.isfinite(descriptor))
        or np.any(~np.isfinite(mapped))
        or np.any(~np.isfinite(height))
    ):
        raise ValueError("semidiscrete generator-component inputs are invalid")
    matrices = {
        name: np.asarray(matrix, dtype=float)
        for name, matrix in stationary_scaled_blocks.items()
    }
    if any(
        matrix.shape != descriptor.shape or np.any(~np.isfinite(matrix))
        for matrix in matrices.values()
    ):
        raise ValueError("stationary generator block is invalid")
    factor = lu_factor(descriptor)
    components = {
        name: -lu_solve(factor, matrix)
        for name, matrix in matrices.items()
    }
    components["mapped_storage_rate_derivative"] = -lu_solve(
        factor,
        mapped,
    )
    components["responsive_height_storage_rate_derivative"] = -lu_solve(
        factor,
        height,
    )
    full_residual = (
        sum(
            matrices.values(),
            start=np.zeros_like(descriptor),
        )
        + mapped
        + height
    )
    generator = -lu_solve(factor, full_residual)
    reconstructed = sum(
        components.values(),
        start=np.zeros_like(generator),
    )
    scale = max(
        float(np.linalg.norm(generator)),
        float(np.linalg.norm(reconstructed)),
        np.finfo(float).tiny,
    )
    defect = float(np.linalg.norm(generator - reconstructed) / scale)
    return components, defect


def _relative_history_defect(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(left - right)) / scale)


def causal_semidiscrete_control_energy_history(
    scaled_history: np.ndarray,
    *,
    scaled_energy_metric: np.ndarray,
    descriptor_scaled_matrix: np.ndarray,
    scaled_generator_per_s: np.ndarray,
    stationary_scaled_blocks: dict[str, np.ndarray],
    mapped_storage_rate_scaled_matrix: np.ndarray,
    responsive_height_storage_rate_scaled_matrix: np.ndarray,
    conservation_row_scales: np.ndarray,
    shared_face_flux_scaled_jacobians: np.ndarray,
) -> CausalSemidiscreteEnergyHistory:
    """Evaluate the exact block- and face-resolved energy identity.

    ``scaled_history`` has shape ``(time, case, degree_of_freedom)``.
    Shared-face maps act on the same scaled state and return the physical
    integrated face residual.  Face powers are signed contributions to the
    control-volume energy rate and sum exactly to the conservative-transport
    block power.
    """

    states = np.asarray(scaled_history, dtype=float)
    metric = np.asarray(scaled_energy_metric, dtype=float)
    descriptor = np.asarray(descriptor_scaled_matrix, dtype=float)
    generator = np.asarray(scaled_generator_per_s, dtype=float)
    rows = np.asarray(conservation_row_scales, dtype=float).ravel()
    faces = np.asarray(shared_face_flux_scaled_jacobians, dtype=float)
    if states.ndim != 3:
        raise ValueError("scaled energy history must be three-dimensional")
    times, cases, dimensions = states.shape
    if (
        times < 2
        or cases < 1
        or metric.shape != (dimensions, dimensions)
        or descriptor.shape != metric.shape
        or generator.shape != metric.shape
        or rows.shape != (dimensions,)
        or np.any(~np.isfinite(states))
        or np.any(~np.isfinite(metric))
        or np.any(~np.isfinite(descriptor))
        or np.any(~np.isfinite(generator))
        or np.any(~np.isfinite(rows))
        or np.any(rows <= 0.0)
        or faces.ndim != 3
        or faces.shape[2] != dimensions
        or np.any(~np.isfinite(faces))
    ):
        raise ValueError("semidiscrete control-energy inputs are invalid")
    fields = int(faces.shape[1])
    cells = int(faces.shape[0] - 1)
    if fields <= 0 or dimensions != cells * fields:
        raise ValueError("shared-face layout is inconsistent")

    _components, component_matrix_defect = (
        causal_semidiscrete_generator_components(
            descriptor,
            stationary_scaled_blocks=stationary_scaled_blocks,
            mapped_storage_rate_scaled_matrix=(
                mapped_storage_rate_scaled_matrix
            ),
            responsive_height_storage_rate_scaled_matrix=(
                responsive_height_storage_rate_scaled_matrix
            ),
        )
    )
    flattened = states.reshape(times * cases, dimensions)
    weighted = flattened @ metric
    descriptor_factor = lu_factor(descriptor.T)
    dual = lu_solve(descriptor_factor, weighted.T).T
    rate = flattened @ generator.T
    stored = 0.5 * np.einsum(
        "bi,bi->b",
        flattened,
        weighted,
        optimize=True,
    )
    direct_power = np.einsum(
        "bi,bi->b",
        weighted,
        rate,
        optimize=True,
    )

    block_powers: dict[str, np.ndarray] = {}
    for name, matrix in {
        **stationary_scaled_blocks,
        "mapped_storage_rate_derivative": (
            mapped_storage_rate_scaled_matrix
        ),
        "responsive_height_storage_rate_derivative": (
            responsive_height_storage_rate_scaled_matrix
        ),
    }.items():
        residual_action = csr_matrix(np.asarray(matrix, dtype=float)) @ (
            flattened.T
        )
        block_powers[name] = -np.einsum(
            "bi,ib->b",
            dual,
            residual_action,
            optimize=True,
        ).reshape(times, cases)

    face_action = (
        csr_matrix(faces.reshape((cells + 1) * fields, dimensions))
        @ flattened.T
    ).T.reshape(times * cases, cells + 1, fields)
    physical_dual = (dual / rows[None, :]).reshape(
        times * cases,
        cells,
        fields,
    )
    face_power = np.empty(
        (times * cases, cells + 1),
        dtype=float,
    )
    face_power[:, 0] = np.einsum(
        "bi,bi->b",
        physical_dual[:, 0],
        face_action[:, 0],
        optimize=True,
    )
    face_power[:, 1:-1] = np.einsum(
        "bfi,bfi->bf",
        physical_dual[:, 1:] - physical_dual[:, :-1],
        face_action[:, 1:-1],
        optimize=True,
    )
    face_power[:, -1] = -np.einsum(
        "bi,bi->b",
        physical_dual[:, -1],
        face_action[:, -1],
        optimize=True,
    )

    block_total = sum(
        block_powers.values(),
        start=np.zeros((times, cases), dtype=float),
    )
    conservative_name = "candidate_conservative_transport"
    if conservative_name not in block_powers:
        raise ValueError("conservative transport block is missing")
    face_total = np.sum(
        face_power.reshape(times, cases, cells + 1),
        axis=-1,
    )
    reshaped_direct = direct_power.reshape(times, cases)
    return CausalSemidiscreteEnergyHistory(
        stored_energy=stored.reshape(times, cases),
        stored_energy_rate=direct_power.reshape(times, cases),
        direct_generator_power=reshaped_direct,
        block_powers=block_powers,
        conservative_face_powers=face_power.reshape(
            times,
            cases,
            cells + 1,
        ),
        maximum_generator_power_defect=component_matrix_defect,
        maximum_block_power_defect=_relative_history_defect(
            reshaped_direct,
            block_total,
        ),
        maximum_face_power_defect=_relative_history_defect(
            block_powers[conservative_name],
            face_total,
        ),
    )
