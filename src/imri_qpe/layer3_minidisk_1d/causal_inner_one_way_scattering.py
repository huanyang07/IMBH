"""Audit helpers for one-way causal-inner scattering histories.

The helpers in this module do not modify the production residual.  They
measure normalization-invariant characteristic energy on a uniform grid,
integrate prospectively frozen time windows, and close a complete
control-volume energy ledger.  All quantities use the physical primitive
chart; time histories are arranged as ``(time, case, cell, field)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C


_N_FIELDS = 5


@dataclass(frozen=True)
class CausalOneWayEnergyHistory:
    """Face fluxes, stored energy, and physical work histories."""

    incident_total_flux: np.ndarray
    transmitted_total_flux: np.ndarray
    incident_family_fluxes: np.ndarray
    transmitted_family_fluxes: np.ndarray
    stored_energy: np.ndarray
    lower_work_by_block: dict[str, np.ndarray]
    background_gradient_work: np.ndarray


@dataclass(frozen=True)
class CausalOneWayIntegratedLedger:
    """Integrated one-way transmission and complete energy balance."""

    incident_energy: np.ndarray
    transmitted_energy: np.ndarray
    incident_family_energy: np.ndarray
    transmitted_family_energy: np.ndarray
    transmission: np.ndarray
    family_transmission: np.ndarray
    stored_energy_change: np.ndarray
    lower_work_by_block: dict[str, np.ndarray]
    background_gradient_work: np.ndarray
    discrete_remainder_work: np.ndarray
    ledger_residual: np.ndarray
    maximum_relative_ledger_defect: float


def _validate_history(
    physical_history: np.ndarray,
    energy_metrics: np.ndarray,
    flux_metrics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = np.asarray(physical_history, dtype=float)
    energy = np.asarray(energy_metrics, dtype=float)
    flux = np.asarray(flux_metrics, dtype=float)
    if (
        history.ndim != 4
        or history.shape[-1] != _N_FIELDS
        or energy.shape
        != (history.shape[2], _N_FIELDS, _N_FIELDS)
        or flux.shape != energy.shape
        or np.any(~np.isfinite(history))
        or np.any(~np.isfinite(energy))
        or np.any(~np.isfinite(flux))
    ):
        raise ValueError("one-way energy-history inputs are invalid")
    return history, energy, flux


def causal_linear_face_trace(
    physical_history: np.ndarray,
    face_index: int,
) -> np.ndarray:
    """Return the common centered quadratic trace at one log-grid face.

    On the uniform log grids used by the scattering contract this is the
    average of the two unlimited PLM traces.  It therefore follows the
    declared production reconstruction while remaining side-neutral for the
    physical diagnostic.  The two-cell fallback is used only next to a
    boundary.
    """

    history = np.asarray(physical_history, dtype=float)
    face = int(face_index)
    if (
        history.ndim != 4
        or history.shape[-1] != _N_FIELDS
        or not 1 <= face < history.shape[2]
        or np.any(~np.isfinite(history))
    ):
        raise ValueError("face-trace inputs are invalid")
    if 2 <= face <= history.shape[2] - 2:
        left_trace = history[:, :, face - 1] + 0.25 * (
            history[:, :, face] - history[:, :, face - 2]
        )
        right_trace = history[:, :, face] - 0.25 * (
            history[:, :, face + 1] - history[:, :, face - 1]
        )
        return 0.5 * (left_trace + right_trace)
    return 0.5 * (history[:, :, face - 1] + history[:, :, face])


def causal_inward_energy_flux(
    trace: np.ndarray,
    flux_metric: np.ndarray,
    projectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return total and family-resolved inward energy flux per second."""

    state = np.asarray(trace, dtype=float)
    metric = np.asarray(flux_metric, dtype=float)
    projector = np.asarray(projectors, dtype=float)
    if (
        state.ndim != 3
        or state.shape[-1] != _N_FIELDS
        or metric.shape != (_N_FIELDS, _N_FIELDS)
        or projector.shape != (_N_FIELDS, _N_FIELDS, _N_FIELDS)
        or np.any(~np.isfinite(state))
        or np.any(~np.isfinite(metric))
        or np.any(~np.isfinite(projector))
    ):
        raise ValueError("inward-energy-flux inputs are invalid")
    total = -0.5 * C * np.einsum(
        "tci,ij,tcj->tc",
        state,
        metric,
        state,
        optimize=True,
    )
    projected = np.einsum(
        "fij,tcj->tcfi",
        projector,
        state,
        optimize=True,
    )
    families = -0.5 * C * np.einsum(
        "tcfi,ij,tcfj->tcf",
        projected,
        metric,
        projected,
        optimize=True,
    )
    return np.asarray(total), np.asarray(families)


def causal_integrate_frozen_window(
    times_seconds: np.ndarray,
    values: np.ndarray,
    window_seconds: tuple[float, float],
) -> np.ndarray:
    """Trapezoid-integrate a history over a fixed, interpolated window."""

    times = np.asarray(times_seconds, dtype=float)
    history = np.asarray(values, dtype=float)
    lower, upper = (float(item) for item in window_seconds)
    if (
        times.ndim != 1
        or history.shape[0] != times.size
        or times.size < 2
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or np.any(~np.isfinite(history))
        or lower < times[0]
        or upper > times[-1]
        or not lower < upper
    ):
        raise ValueError("frozen-window integration inputs are invalid")
    interior = (times > lower) & (times < upper)
    selected_times = np.concatenate(([lower], times[interior], [upper]))
    flattened = history.reshape(times.size, -1)
    selected_values = np.empty(
        (selected_times.size, flattened.shape[1]),
        dtype=float,
    )
    for column in range(flattened.shape[1]):
        selected_values[:, column] = np.interp(
            selected_times,
            times,
            flattened[:, column],
        )
    integrated = np.trapezoid(
        selected_values,
        selected_times,
        axis=0,
    )
    return integrated.reshape(history.shape[1:])


def causal_one_way_energy_history(
    physical_history: np.ndarray,
    *,
    log_edges: np.ndarray,
    energy_metrics: np.ndarray,
    flux_metrics: np.ndarray,
    projectors: np.ndarray,
    lower_evolution_blocks: dict[str, np.ndarray],
    downstream_face: int,
    interface_face: int,
    face_flux_metrics: np.ndarray | None = None,
    face_projectors: np.ndarray | None = None,
) -> CausalOneWayEnergyHistory:
    """Evaluate the complete physical one-way control-volume history."""

    history, energy, flux = _validate_history(
        physical_history,
        energy_metrics,
        flux_metrics,
    )
    edges = np.asarray(log_edges, dtype=float)
    families = np.asarray(projectors, dtype=float)
    lower_face = int(downstream_face)
    upper_face = int(interface_face)
    cells = history.shape[2]
    if (
        edges.shape != (cells + 1,)
        or np.any(np.diff(edges) <= 0.0)
        or families.shape != (cells, _N_FIELDS, _N_FIELDS, _N_FIELDS)
        or not 0 < lower_face < upper_face < cells
    ):
        raise ValueError("one-way control-volume geometry is invalid")
    for name, block in lower_evolution_blocks.items():
        if np.asarray(block).shape != energy.shape:
            raise ValueError(f"lower work block {name!r} is invalid")

    incident_trace = causal_linear_face_trace(history, upper_face)
    transmitted_trace = causal_linear_face_trace(history, lower_face)
    if face_flux_metrics is None:
        face_flux = 0.5 * (flux[:-1] + flux[1:])
        face_flux = np.concatenate(
            (flux[:1], face_flux, flux[-1:]),
            axis=0,
        )
    else:
        face_flux = np.asarray(face_flux_metrics, dtype=float)
    if face_projectors is None:
        face_family = 0.5 * (families[:-1] + families[1:])
        face_family = np.concatenate(
            (families[:1], face_family, families[-1:]),
            axis=0,
        )
    else:
        face_family = np.asarray(face_projectors, dtype=float)
    expected_face_matrix = (cells + 1, _N_FIELDS, _N_FIELDS)
    expected_face_projector = (
        cells + 1,
        _N_FIELDS,
        _N_FIELDS,
        _N_FIELDS,
    )
    if (
        face_flux.shape != expected_face_matrix
        or face_family.shape != expected_face_projector
        or np.any(~np.isfinite(face_flux))
        or np.any(~np.isfinite(face_family))
    ):
        raise ValueError("one-way face energy data are invalid")
    incident_metric = face_flux[upper_face]
    transmitted_metric = face_flux[lower_face]
    incident_projectors = face_family[upper_face]
    transmitted_projectors = face_family[lower_face]
    incident, incident_families = causal_inward_energy_flux(
        incident_trace,
        incident_metric,
        incident_projectors,
    )
    transmitted, transmitted_families = causal_inward_energy_flux(
        transmitted_trace,
        transmitted_metric,
        transmitted_projectors,
    )

    region = slice(lower_face, upper_face)
    widths = np.diff(edges)[region]
    stored = 0.5 * np.einsum(
        "tcni,nij,tcnj,n->tc",
        history[:, :, region],
        energy[region],
        history[:, :, region],
        widths,
        optimize=True,
    )
    lower_work = {
        name: C
        * np.einsum(
            "tcni,nij,njk,tcnk,n->tc",
            history[:, :, region],
            energy[region],
            np.asarray(block, dtype=float)[region],
            history[:, :, region],
            widths,
            optimize=True,
        )
        for name, block in lower_evolution_blocks.items()
    }
    centers = 0.5 * (edges[:-1] + edges[1:])
    flux_derivative = np.gradient(
        flux,
        centers,
        axis=0,
        edge_order=2,
    )
    background = 0.5 * C * np.einsum(
        "tcni,nij,tcnj,n->tc",
        history[:, :, region],
        flux_derivative[region],
        history[:, :, region],
        widths,
        optimize=True,
    )
    return CausalOneWayEnergyHistory(
        incident_total_flux=incident,
        transmitted_total_flux=transmitted,
        incident_family_fluxes=incident_families,
        transmitted_family_fluxes=transmitted_families,
        stored_energy=stored,
        lower_work_by_block=lower_work,
        background_gradient_work=background,
    )


def causal_integrated_one_way_ledger(
    history: CausalOneWayEnergyHistory,
    times_seconds: np.ndarray,
    *,
    incident_window_seconds: tuple[float, float],
    transmitted_window_seconds: tuple[float, float],
) -> CausalOneWayIntegratedLedger:
    """Integrate transmission and close the declared complete ledger.

    Physical work is evaluated from the continuum lower blocks and background
    coefficient derivative.  The remaining semidiscrete transport/descriptor
    work is reported explicitly; including it makes the numerical
    control-volume ledger algebraically complete without disguising it as
    physical dissipation.
    """

    times = np.asarray(times_seconds, dtype=float)
    incident = causal_integrate_frozen_window(
        times,
        history.incident_total_flux,
        incident_window_seconds,
    )
    transmitted = causal_integrate_frozen_window(
        times,
        history.transmitted_total_flux,
        transmitted_window_seconds,
    )
    incident_families = causal_integrate_frozen_window(
        times,
        history.incident_family_fluxes,
        incident_window_seconds,
    )
    transmitted_families = causal_integrate_frozen_window(
        times,
        history.transmitted_family_fluxes,
        transmitted_window_seconds,
    )
    lower = {
        name: np.trapezoid(values, times, axis=0)
        for name, values in history.lower_work_by_block.items()
    }
    background = np.trapezoid(
        history.background_gradient_work,
        times,
        axis=0,
    )
    stored_change = history.stored_energy[-1] - history.stored_energy[0]
    physical_work = (
        np.sum(np.asarray(tuple(lower.values())), axis=0)
        + background
    )
    discrete_remainder = (
        stored_change - incident + transmitted - physical_work
    )
    residual = (
        incident
        - transmitted
        + physical_work
        + discrete_remainder
        - stored_change
    )
    scale = np.maximum.reduce(
        (
            np.abs(incident),
            np.abs(transmitted),
            np.abs(stored_change),
            np.abs(physical_work),
            np.finfo(float).tiny * np.ones_like(incident),
        )
    )
    transmission = transmitted / np.maximum(
        incident,
        np.finfo(float).tiny,
    )
    family_transmission = transmitted_families / np.maximum(
        incident[:, None],
        np.finfo(float).tiny,
    )
    return CausalOneWayIntegratedLedger(
        incident_energy=incident,
        transmitted_energy=transmitted,
        incident_family_energy=incident_families,
        transmitted_family_energy=transmitted_families,
        transmission=transmission,
        family_transmission=family_transmission,
        stored_energy_change=stored_change,
        lower_work_by_block=lower,
        background_gradient_work=background,
        discrete_remainder_work=discrete_remainder,
        ledger_residual=residual,
        maximum_relative_ledger_defect=float(
            np.max(np.abs(residual) / scale)
        ),
    )


def causal_amplitude_scaling_defect(
    reference: np.ndarray,
    comparison: np.ndarray,
    expected_factor: float,
) -> float:
    """Return a fixed-scale relative amplitude-scaling defect."""

    first = np.asarray(reference, dtype=float)
    second = np.asarray(comparison, dtype=float)
    factor = float(expected_factor)
    if (
        first.shape != second.shape
        or np.any(~np.isfinite(first))
        or np.any(~np.isfinite(second))
        or not np.isfinite(factor)
    ):
        raise ValueError("amplitude-scaling inputs are invalid")
    scale = max(
        float(np.max(np.abs(second))),
        float(np.max(np.abs(factor * first))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(second - factor * first)) / scale)
