"""Characteristic-packet diagnostics for the causal five-field system.

This module is audit-only.  It builds smooth packets in the five local-rest
principal families without changing the production finite-volume flux or its
boundary maps.  The returned packet is expressed in a caller-supplied
dimensionless primitive chart so the same continuum construction can be used
on nested or hybrid grids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .causal_inner_dae import audit_causal_five_field_principal
from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    _cell_state,
)


CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES = (
    "inward_acoustic",
    "inward_shear",
    "material",
    "outward_shear",
    "outward_acoustic",
)


@dataclass(frozen=True)
class CausalFiveFieldCharacteristicBasis:
    """One local five-family basis in physical and dimensionless charts."""

    family_labels: tuple[str, ...]
    local_rest_speeds_over_c: np.ndarray
    coordinate_speeds_over_c: np.ndarray
    physical_right_eigenvectors: np.ndarray
    dimensionless_right_eigenvectors: np.ndarray
    maximum_eigenpair_defect: float
    condition_number: float


@dataclass(frozen=True)
class CausalCharacteristicPacketMoments:
    """Activity, position, width, and family leakage of one packet history."""

    l2_amplitude: np.ndarray
    log_radius_centroid: np.ndarray
    log_radius_width: np.ndarray
    selected_family_fraction: np.ndarray
    opposite_family_fraction: np.ndarray


def _family_index(family: str) -> int:
    try:
        return CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES.index(str(family))
    except ValueError as exc:
        raise ValueError("unsupported causal characteristic family") from exc


def causal_five_field_characteristic_basis(
    context: CausalFiveFieldDAEContext,
    radius: float,
    primitive_chart: np.ndarray,
    primitive_amplitudes: np.ndarray,
) -> CausalFiveFieldCharacteristicBasis:
    """Return the ordered local-rest principal basis at one radius.

    The eigenvectors solve the exact responsive-height local-rest principal
    matrix already certified by :func:`audit_causal_five_field_principal`.
    Coordinate speeds are the corresponding Valencia acoustic/contact/shear
    speeds on the moving Kerr--Schild background.  Eigenvectors are scaled to
    unit Euclidean norm in the supplied dimensionless primitive chart.
    """

    context = context.validated()
    chart = np.asarray(primitive_chart, dtype=float)
    amplitudes = np.asarray(primitive_amplitudes, dtype=float)
    if (
        chart.shape != (5,)
        or amplitudes.shape != (5,)
        or np.any(~np.isfinite(chart))
        or np.any(~np.isfinite(amplitudes))
        or np.any(amplitudes <= 0.0)
    ):
        raise ValueError("characteristic-basis inputs are invalid")
    state = _cell_state(context, float(radius), chart)
    audit = audit_causal_five_field_principal(
        state.geometry,
        context.vertical_frequency.eos(float(radius)),
        state.closure,
        surface_density=state.primitive.surface_density,
        radial_velocity_over_c=state.primitive.radial_velocity_over_c,
        azimuthal_velocity_over_c=state.primitive.azimuthal_velocity_over_c,
        temperature=state.thermodynamics.temperature,
    )
    mass = np.asarray(audit.local_rest_mass_matrix, dtype=float)
    flux = np.asarray(audit.local_rest_flux_matrix, dtype=float)
    physical = np.asarray(
        audit.local_rest_right_eigenvectors,
        dtype=float,
    )
    dimensionless = physical / amplitudes[:, None]
    norms = np.linalg.norm(dimensionless, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("characteristic eigenvector is singular")
    physical = physical / norms[None, :]
    dimensionless = dimensionless / norms[None, :]
    for column in range(dimensionless.shape[1]):
        pivot = int(np.argmax(np.abs(dimensionless[:, column])))
        if dimensionless[pivot, column] < 0.0:
            physical[:, column] *= -1.0
            dimensionless[:, column] *= -1.0
    local_speeds = np.asarray(
        audit.numerical_local_rest_speeds_over_c,
        dtype=float,
    )
    residual = flux @ physical - mass @ (
        physical * local_speeds[None, :]
    )
    scale = max(
        float(np.max(np.abs(flux @ physical))),
        float(np.max(np.abs(mass @ physical))),
        np.finfo(float).tiny,
    )
    return CausalFiveFieldCharacteristicBasis(
        family_labels=CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
        local_rest_speeds_over_c=local_speeds,
        coordinate_speeds_over_c=np.asarray(
            audit.coordinate_speeds_over_c,
            dtype=float,
        ),
        physical_right_eigenvectors=physical,
        dimensionless_right_eigenvectors=dimensionless,
        maximum_eigenpair_defect=float(np.max(np.abs(residual)) / scale),
        condition_number=float(np.linalg.cond(dimensionless)),
    )


def causal_compact_log_radius_envelope(
    radius: np.ndarray,
    *,
    support_inner_radius: float,
    support_outer_radius: float,
) -> np.ndarray:
    """Return a smooth compact bump with unit peak in ``ln(radius)``."""

    values = np.asarray(radius, dtype=float)
    inner = float(support_inner_radius)
    outer = float(support_outer_radius)
    if (
        values.ndim != 1
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
        or not np.isfinite(inner)
        or not np.isfinite(outer)
        or inner <= 0.0
        or outer <= inner
    ):
        raise ValueError("compact-envelope inputs are invalid")
    midpoint = 0.5 * (np.log(inner) + np.log(outer))
    half_width = 0.5 * (np.log(outer) - np.log(inner))
    coordinate = (np.log(values) - midpoint) / half_width
    envelope = np.zeros_like(values)
    inside = np.abs(coordinate) < 1.0
    envelope[inside] = np.exp(
        1.0 - 1.0 / (1.0 - coordinate[inside] ** 2)
    )
    return envelope


def causal_five_field_characteristic_packet(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    primitive_amplitudes: np.ndarray,
    *,
    family: str,
    support_inner_radius: float,
    support_outer_radius: float,
) -> tuple[np.ndarray, tuple[CausalFiveFieldCharacteristicBasis, ...]]:
    """Return one continuum-matched, dimensionless family packet."""

    context = context.validated()
    charts = np.asarray(primitive_charts, dtype=float)
    amplitudes = np.asarray(primitive_amplitudes, dtype=float)
    n_cells = int(context.grid.centers.size)
    if (
        charts.shape != (n_cells, 5)
        or amplitudes.shape != (n_cells, 5)
        or np.any(~np.isfinite(charts))
        or np.any(~np.isfinite(amplitudes))
        or np.any(amplitudes <= 0.0)
    ):
        raise ValueError("characteristic-packet charts are invalid")
    selected = _family_index(family)
    bases = []
    directions = np.empty_like(charts)
    previous = None
    for cell, radius in enumerate(context.grid.centers):
        basis = causal_five_field_characteristic_basis(
            context,
            float(radius),
            charts[cell],
            amplitudes[cell],
        )
        direction = np.array(
            basis.dimensionless_right_eigenvectors[:, selected],
            copy=True,
        )
        if previous is not None and np.dot(previous, direction) < 0.0:
            direction *= -1.0
        directions[cell] = direction
        previous = direction
        bases.append(basis)
    envelope = causal_compact_log_radius_envelope(
        np.asarray(context.grid.centers, dtype=float),
        support_inner_radius=support_inner_radius,
        support_outer_radius=support_outer_radius,
    )
    return directions * envelope[:, None], tuple(bases)


def causal_five_field_characteristic_coefficients(
    dimensionless_primitive_values: np.ndarray,
    bases: tuple[CausalFiveFieldCharacteristicBasis, ...],
) -> np.ndarray:
    """Project a primitive field or history into the local family basis."""

    values = np.asarray(dimensionless_primitive_values, dtype=float)
    n_cells = len(bases)
    if values.shape[-2:] != (n_cells, 5) or np.any(~np.isfinite(values)):
        raise ValueError("characteristic coefficient inputs are invalid")
    flat = values.reshape(-1, n_cells, 5)
    coefficients = np.empty_like(flat)
    for cell, basis in enumerate(bases):
        coefficients[:, cell] = np.linalg.solve(
            basis.dimensionless_right_eigenvectors,
            flat[:, cell].T,
        ).T
    return coefficients.reshape(values.shape)


def causal_characteristic_packet_moments(
    dimensionless_primitive_history: np.ndarray,
    bases: tuple[CausalFiveFieldCharacteristicBasis, ...],
    radius: np.ndarray,
    cell_measures: np.ndarray,
    *,
    family: str,
) -> CausalCharacteristicPacketMoments:
    """Return conservative activity moments for one packet history."""

    values = np.asarray(dimensionless_primitive_history, dtype=float)
    radii = np.asarray(radius, dtype=float)
    measures = np.asarray(cell_measures, dtype=float)
    n_cells = len(bases)
    if (
        values.ndim != 3
        or values.shape[1:] != (n_cells, 5)
        or radii.shape != (n_cells,)
        or measures.shape != (n_cells,)
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(radii))
        or np.any(~np.isfinite(measures))
        or np.any(radii <= 0.0)
        or np.any(measures <= 0.0)
    ):
        raise ValueError("packet-moment inputs are invalid")
    selected = _family_index(family)
    opposite = None if selected == 2 else 4 - selected
    coefficients = causal_five_field_characteristic_coefficients(
        values,
        bases,
    )
    energy = np.sum(coefficients**2, axis=2)
    weighted = measures[None, :] * energy
    total = np.maximum(
        np.sum(weighted, axis=1),
        np.finfo(float).tiny,
    )
    log_radius = np.log(radii)
    centroid = np.sum(weighted * log_radius[None, :], axis=1) / total
    width = np.sqrt(
        np.sum(
            weighted * (log_radius[None, :] - centroid[:, None]) ** 2,
            axis=1,
        )
        / total
    )
    selected_energy = np.sum(
        measures[None, :] * coefficients[:, :, selected] ** 2,
        axis=1,
    )
    opposite_energy = (
        np.zeros_like(selected_energy)
        if opposite is None
        else np.sum(
            measures[None, :] * coefficients[:, :, opposite] ** 2,
            axis=1,
        )
    )
    return CausalCharacteristicPacketMoments(
        l2_amplitude=np.sqrt(total / np.sum(measures)),
        log_radius_centroid=centroid,
        log_radius_width=width,
        selected_family_fraction=selected_energy / total,
        opposite_family_fraction=opposite_energy / total,
    )
