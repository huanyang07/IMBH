"""Cell-integral helpers for lower-height-work localization audits."""

from __future__ import annotations

import numpy as np


def causal_restrict_cell_integrals(
    fine_values: np.ndarray,
    *,
    refinement_factor: int,
) -> np.ndarray:
    """Restrict nested cell integrals by exact summation."""

    values = np.asarray(fine_values, dtype=float)
    factor = int(refinement_factor)
    if (
        values.ndim < 1
        or factor < 1
        or values.shape[-1] % factor
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("cell-integral restriction inputs are invalid")
    coarse_cells = values.shape[-1] // factor
    return np.sum(
        values.reshape(values.shape[:-1] + (coarse_cells, factor)),
        axis=-1,
    )


def causal_prefix_suffix_histories(
    cell_integrals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return inclusive prefix and suffix sums over the final cell axis."""

    values = np.asarray(cell_integrals, dtype=float)
    if values.ndim < 1 or np.any(~np.isfinite(values)):
        raise ValueError("cell-integral histories are invalid")
    prefix = np.cumsum(values, axis=-1)
    suffix = np.flip(
        np.cumsum(np.flip(values, axis=-1), axis=-1),
        axis=-1,
    )
    return prefix, suffix


def causal_signed_band_gram_matrix(
    band_histories: np.ndarray,
    *,
    physical_scale: float,
    time_weights: np.ndarray,
) -> np.ndarray:
    """Return the full signed time-weighted Gram matrix of radial bands."""

    values = np.asarray(band_histories, dtype=float)
    weights = np.asarray(time_weights, dtype=float).ravel()
    scale = float(physical_scale)
    if (
        values.ndim != 2
        or values.shape[0] != weights.size
        or values.shape[1] < 1
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.isfinite(scale)
        or scale <= 0.0
        or float(np.sum(weights)) <= 0.0
    ):
        raise ValueError("band-Gram inputs are invalid")
    normalized_weights = weights / float(np.sum(weights))
    normalized = values / scale
    return np.einsum(
        "tb,tc,t->bc",
        normalized,
        normalized,
        normalized_weights,
    )


def causal_partition_cell_integrals(
    cell_integrals: np.ndarray,
    edge_indices: np.ndarray,
) -> np.ndarray:
    """Sum cell integrals over a complete declared edge partition."""

    values = np.asarray(cell_integrals, dtype=float)
    edges = np.asarray(edge_indices, dtype=int).ravel()
    if (
        values.ndim < 1
        or edges.size < 2
        or edges[0] != 0
        or edges[-1] != values.shape[-1]
        or np.any(np.diff(edges) <= 0)
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("cell-integral partition is invalid")
    return np.stack(
        [
            np.sum(values[..., left:right], axis=-1)
            for left, right in zip(edges[:-1], edges[1:], strict=True)
        ],
        axis=-1,
    )
