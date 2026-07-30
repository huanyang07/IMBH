"""Audit helpers for a frozen causal-inner packet manifest."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


_N_FIELDS = 5


def causal_canonical_json_sha256(payload: object) -> str:
    """Hash one JSON-compatible payload with stable encoding."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def causal_array_sha256(values: np.ndarray) -> str:
    """Hash dtype, shape, and bytes of one contiguous numerical array."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CausalCharacteristicPurity:
    """Energy fractions and reconstruction closure in one local basis."""

    family_energy_fractions: np.ndarray
    maximum_reconstruction_defect: float
    minimum_active_cell_selected_fraction: float


def causal_characteristic_purity(
    physical_cell_averages: np.ndarray,
    physical_right_eigenvectors: np.ndarray,
    field_scales: np.ndarray,
    cell_measures: np.ndarray,
    *,
    selected_family: int,
    relative_activity: float = 1.0e-6,
) -> CausalCharacteristicPurity:
    """Project one finite-volume profile into its local physical families."""

    values = np.asarray(physical_cell_averages, dtype=float)
    bases = np.asarray(physical_right_eigenvectors, dtype=float)
    scales = np.asarray(field_scales, dtype=float).ravel()
    measures = np.asarray(cell_measures, dtype=float).ravel()
    family = int(selected_family)
    if (
        values.ndim != 2
        or values.shape[1] != _N_FIELDS
        or bases.shape != (values.shape[0], _N_FIELDS, _N_FIELDS)
        or scales.shape != (_N_FIELDS,)
        or measures.shape != (values.shape[0],)
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(bases))
        or np.any(~np.isfinite(scales))
        or np.any(~np.isfinite(measures))
        or np.any(scales <= 0.0)
        or np.any(measures <= 0.0)
        or not 0 <= family < _N_FIELDS
        or not 0.0 < float(relative_activity) < 1.0
    ):
        raise ValueError("characteristic-purity inputs are invalid")
    dimensionless_values = values / scales[None, :]
    dimensionless_bases = bases / scales[None, :, None]
    coefficients = np.empty_like(dimensionless_values)
    reconstruction_defect = 0.0
    for cell in range(values.shape[0]):
        coefficients[cell] = np.linalg.solve(
            dimensionless_bases[cell],
            dimensionless_values[cell],
        )
        reconstructed = (
            dimensionless_bases[cell] @ coefficients[cell]
        )
        scale = max(
            float(np.linalg.norm(dimensionless_values[cell])),
            np.finfo(float).tiny,
        )
        reconstruction_defect = max(
            reconstruction_defect,
            float(
                np.linalg.norm(
                    reconstructed - dimensionless_values[cell]
                )
                / scale
            ),
        )
    energy = np.einsum(
        "ci,c->i",
        np.abs(coefficients) ** 2,
        measures,
    )
    fractions = energy / max(float(np.sum(energy)), np.finfo(float).tiny)
    cell_energy = np.sum(np.abs(coefficients) ** 2, axis=1)
    active = cell_energy >= (
        float(relative_activity) * float(np.max(cell_energy))
    )
    selected = (
        np.abs(coefficients[:, family]) ** 2
        / np.maximum(cell_energy, np.finfo(float).tiny)
    )
    return CausalCharacteristicPurity(
        family_energy_fractions=fractions,
        maximum_reconstruction_defect=reconstruction_defect,
        minimum_active_cell_selected_fraction=float(
            np.min(selected[active])
        ),
    )


def causal_scaled_variant_defect(
    base: np.ndarray,
    variant: np.ndarray,
    *,
    expected_factor: float,
) -> float:
    """Return a relative defect from one declared sign/amplitude scaling."""

    reference = float(expected_factor) * np.asarray(base, dtype=float)
    candidate = np.asarray(variant, dtype=float)
    if candidate.shape != reference.shape:
        raise ValueError("scaled variant has the wrong shape")
    scale = max(
        float(np.linalg.norm(reference)),
        float(np.linalg.norm(candidate)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(candidate - reference) / scale)
