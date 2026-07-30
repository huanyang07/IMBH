"""Audit helpers for causal-inner embedded-grid discrimination.

The embedded layouts are ordinary nonoverlapping finite-volume grids.  The
helpers in this module therefore do not introduce a coupling algorithm.  They
only expose:

* the thirteen physical exports of the cells inside a declared coupling face;
* exact conservative-transport telescoping at that face; and
* fixed-physical-scale characteristic energies on a common parent grid.

They are deliberately independent of the prospective packet definitions and
their acceptance thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_N_FIELDS = 5
_CONSERVATIVE_FIELDS = np.asarray((0, 2, 4), dtype=int)


@dataclass(frozen=True)
class CausalEmbeddedActiveObservableAudit:
    """Linear maps and ledger closure for one embedded active domain."""

    observable_map: np.ndarray
    lower_height_cell_map: np.ndarray
    conservative_transport_telescoping_defect: float
    active_prefix_ledger_defect: float
    active_cell_count: int
    coupling_face_index: int


def causal_embedded_active_observable_audit(
    tangent,
    coupling_face_index: int,
) -> CausalEmbeddedActiveObservableAudit:
    """Return the active-domain export map of one monolithic tangent.

    The returned thirteen rows are, in order, inner M/J/E flux, coupling-face
    M/J/E flux, active-domain net M/J/E drive, cooling J/E, and lower
    responsive-height work J/E.
    """

    cells = int(np.asarray(tangent.base_primitives).shape[0])
    face = int(coupling_face_index)
    if not 1 <= face < cells:
        raise ValueError("embedded coupling face is invalid")
    rows = np.asarray(tangent.conservation_row_scales, dtype=float).ravel()
    dimensions = cells * _N_FIELDS
    if rows.shape != (dimensions,) or np.any(~np.isfinite(rows)):
        raise ValueError("embedded tangent row scales are invalid")

    spatial = tangent.spatial_tangent
    face_maps = np.asarray(
        spatial.shared_face_flux_scaled_jacobians,
        dtype=float,
    )
    if face_maps.shape != (cells + 1, _N_FIELDS, dimensions):
        raise ValueError("embedded shared-face map has the wrong shape")
    stationary = (
        np.asarray(tangent.stationary_scaled_jacobian, dtype=float)
        * rows[:, None]
    ).reshape(cells, _N_FIELDS, dimensions)
    cooling = (
        np.asarray(
            spatial.block_scaled_jacobians["candidate_cooling"],
            dtype=float,
        )
        * rows[:, None]
    ).reshape(cells, _N_FIELDS, dimensions)
    lower_height = (
        np.asarray(
            spatial.block_scaled_jacobians[
                "candidate_lower_height_work"
            ],
            dtype=float,
        )
        * rows[:, None]
    ).reshape(cells, _N_FIELDS, dimensions)

    observable = np.concatenate(
        (
            face_maps[0, _CONSERVATIVE_FIELDS],
            face_maps[face, _CONSERVATIVE_FIELDS],
            -np.sum(
                stationary[:face, _CONSERVATIVE_FIELDS],
                axis=0,
            ),
            -np.sum(
                cooling[:face, _CONSERVATIVE_FIELDS[1:]],
                axis=0,
            ),
            -np.sum(
                lower_height[:face, _CONSERVATIVE_FIELDS[1:]],
                axis=0,
            ),
        ),
        axis=0,
    )
    lower_height_cell_map = -lower_height

    transport = (
        np.asarray(
            spatial.block_scaled_jacobians[
                "candidate_conservative_transport"
            ],
            dtype=float,
        )
        * rows[:, None]
    ).reshape(cells, _N_FIELDS, dimensions)
    transport_sum = -np.sum(
        transport[:face, _CONSERVATIVE_FIELDS],
        axis=0,
    )
    shared_face_difference = (
        face_maps[0, _CONSERVATIVE_FIELDS]
        - face_maps[face, _CONSERVATIVE_FIELDS]
    )
    scale = max(
        float(np.linalg.norm(transport_sum)),
        float(np.linalg.norm(shared_face_difference)),
        np.finfo(float).tiny,
    )
    telescoping = float(
        np.linalg.norm(transport_sum - shared_face_difference) / scale
    )
    source_remainder = stationary - transport
    prefix_net = shared_face_difference - np.sum(
        source_remainder[:face, _CONSERVATIVE_FIELDS],
        axis=0,
    )
    direct_net = observable[6:9]
    ledger_scale = max(
        float(np.linalg.norm(prefix_net)),
        float(np.linalg.norm(direct_net)),
        np.finfo(float).tiny,
    )
    prefix_defect = float(
        np.linalg.norm(prefix_net - direct_net) / ledger_scale
    )
    return CausalEmbeddedActiveObservableAudit(
        observable_map=observable,
        lower_height_cell_map=lower_height_cell_map,
        conservative_transport_telescoping_defect=telescoping,
        active_prefix_ledger_defect=prefix_defect,
        active_cell_count=face,
        coupling_face_index=face,
    )


def causal_embedded_active_direct_observables(
    evaluation,
    coupling_face_index: int,
) -> np.ndarray:
    """Evaluate the same thirteen active-domain observables nonlinearly."""

    face = int(coupling_face_index)
    fluxes = np.asarray(
        evaluation.stationary_ledger.interfaces
        .candidate_shared_face_fluxes_over_c,
        dtype=float,
    )
    residual = np.asarray(evaluation.residual_rows, dtype=float)
    cooling = np.asarray(evaluation.cooling_rows, dtype=float)
    lower_height = np.asarray(
        evaluation.lower_height_work_rows,
        dtype=float,
    )
    if (
        residual.ndim != 2
        or residual.shape[1] != _N_FIELDS
        or not 1 <= face < residual.shape[0]
        or fluxes.shape != (residual.shape[0] + 1, _N_FIELDS)
        or cooling.shape != residual.shape
        or lower_height.shape != residual.shape
    ):
        raise ValueError("embedded nonlinear evaluation has the wrong shape")
    return np.concatenate(
        (
            fluxes[0, _CONSERVATIVE_FIELDS],
            fluxes[face, _CONSERVATIVE_FIELDS],
            -np.sum(residual[:face, _CONSERVATIVE_FIELDS], axis=0),
            -np.sum(
                cooling[:face, _CONSERVATIVE_FIELDS[1:]],
                axis=0,
            ),
            -np.sum(
                lower_height[:face, _CONSERVATIVE_FIELDS[1:]],
                axis=0,
            ),
        )
    )


def causal_dimensionless_characteristic_inverse(
    physical_right_eigenvectors: np.ndarray,
    field_scales: np.ndarray,
) -> np.ndarray:
    """Return cellwise inverses after fixed-physical-scale normalization."""

    right = np.asarray(physical_right_eigenvectors, dtype=float)
    scales = np.asarray(field_scales, dtype=float).ravel()
    if (
        right.ndim != 3
        or right.shape[1:] != (_N_FIELDS, _N_FIELDS)
        or scales.shape != (_N_FIELDS,)
        or np.any(~np.isfinite(right))
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError("characteristic-energy basis is invalid")
    dimensionless = right / scales[None, :, None]
    norms = np.linalg.norm(dimensionless, axis=1)
    if np.any(norms <= np.finfo(float).tiny):
        raise ValueError("characteristic-energy basis is singular")
    normalized = dimensionless / norms[:, None, :]
    return np.asarray(
        [np.linalg.inv(matrix) for matrix in normalized],
        dtype=float,
    )


def causal_characteristic_energy_history(
    physical_history: np.ndarray,
    dimensionless_characteristic_inverse: np.ndarray,
    field_scales: np.ndarray,
    cell_measures: np.ndarray,
) -> np.ndarray:
    """Return per-cell, per-family energy for one physical state history."""

    values = np.asarray(physical_history, dtype=float)
    inverse = np.asarray(
        dimensionless_characteristic_inverse,
        dtype=float,
    )
    scales = np.asarray(field_scales, dtype=float).ravel()
    measures = np.asarray(cell_measures, dtype=float).ravel()
    if (
        values.ndim != 3
        or values.shape[-1] != _N_FIELDS
        or inverse.shape
        != (values.shape[1], _N_FIELDS, _N_FIELDS)
        or scales.shape != (_N_FIELDS,)
        or measures.shape != (values.shape[1],)
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(inverse))
        or np.any(~np.isfinite(measures))
        or np.any(measures <= 0.0)
    ):
        raise ValueError("characteristic-energy history is invalid")
    dimensionless = values / scales[None, None, :]
    coefficients = np.einsum(
        "cij,tcj->tci",
        inverse,
        dimensionless,
    )
    return np.abs(coefficients) ** 2 * measures[None, :, None]
